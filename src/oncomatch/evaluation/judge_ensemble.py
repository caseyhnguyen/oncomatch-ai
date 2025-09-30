"""
Consolidated Judge Ensemble with Dynamic Complexity-Aware Routing
Combines the best features from all judge implementations
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
import time
from datetime import datetime
import numpy as np
from tqdm import tqdm

from oncomatch.models import Patient, ClinicalTrial, MatchResult
from oncomatch.llm_providers import OpenAIProvider, AnthropicProvider, GeminiProvider
from oncomatch.llm_registry import resolve_best_model

logger = logging.getLogger(__name__)


class JudgeRole(str, Enum):
    """Roles for different judge specializations"""
    ACCURACY = "accuracy_judge"
    SAFETY = "safety_judge"  
    COMPLETENESS = "completeness_judge"
    BIAS = "bias_judge"
    ROBUSTNESS = "robustness_judge"
    CLINICAL_TEXT = "clinical_text_judge"
    TRIALGPT = "trialgpt_judge"


class LLMJudgeModel(str, Enum):
    """Available judge models (September 2025)"""
    # OpenAI
    GPT5 = "gpt-5"
    GPT5_MINI = "gpt-5-mini"
    GPT5_NANO = "gpt-5-nano"
    GPT4O = "gpt-4o"
    GPT4O_MINI = "gpt-4o-mini"
    O3 = "o3"
    O3_PRO = "o3-pro"
    O3_MINI = "o3-mini"
    O4_MINI = "o4-mini"
    
    # Anthropic (Claude 3.7 is current stable, Claude 4 requires Bedrock)
    CLAUDE_37_SONNET = "claude-3-7-sonnet-20250219"  # PRIMARY - Feb 2025
    CLAUDE_35_SONNET = "claude-3-5-sonnet-20241022"  # Stable fallback
    CLAUDE_3_OPUS = "claude-3-opus-20240229"  # Previous gen fallback
    
    # Google (Gemini 2.5 is latest as of Sept 2025)
    GEMINI_25_PRO = "gemini-2.5-pro"
    GEMINI_25_FLASH = "gemini-2.5-flash"
    
    # Medical (note: these are research models, not publicly available)
    TRIALGPT = "trialgpt"  # NIH research model


@dataclass
class ComplexityAnalyzer:
    """Analyzes patient-trial complexity for optimal model routing"""
    
    def compute_complexity(
        self, 
        patient: Patient, 
        trial: ClinicalTrial,
        match_result: Optional[MatchResult] = None
    ) -> float:
        """
        Compute complexity score between 0 and 1.
        Higher scores indicate more complex matching scenarios.
        """
        complexity = 0.0
        factors = []
        
        # Patient complexity factors
        if patient.biomarkers_detected:
            biomarker_count = len(patient.biomarkers_detected)
            if biomarker_count > 3:
                complexity += 0.2
                factors.append(f"{biomarker_count} biomarkers")
            elif biomarker_count > 1:
                complexity += 0.1
                factors.append(f"{biomarker_count} biomarkers")
        
        # Age extremes
        if patient.age < 18 or patient.age > 75:
            complexity += 0.15
            factors.append(f"age {patient.age}")
        
        # Advanced stage
        if patient.cancer_stage in ["III", "IV"]:
            complexity += 0.15
            factors.append(f"stage {patient.cancer_stage}")
        
        # ECOG status
        if patient.ecog_status and patient.ecog_status.value >= 2:
            complexity += 0.1
            factors.append(f"ECOG {patient.ecog_status.value}")
        
        # Trial complexity
        if trial.phase and "1" in trial.phase:
            complexity += 0.15
            factors.append("early phase trial")
        
        # Eligibility criteria complexity
        if trial.eligibility:
            if len(trial.eligibility.inclusion_criteria) > 10:
                complexity += 0.1
                factors.append("complex eligibility")
            if len(trial.eligibility.exclusion_criteria) > 10:
                complexity += 0.1
                factors.append("many exclusions")
        
        # Previous treatments
        if patient.previous_treatments and len(patient.previous_treatments) > 2:
            complexity += 0.1
            factors.append("heavily pretreated")
        
        # Normalize to 0-1
        complexity = min(complexity, 1.0)
        
        logger.debug(f"Complexity: {complexity:.2f} - Factors: {', '.join(factors)}")
        return complexity
    
    def compute_urgency(self, patient: Patient) -> float:
        """Compute urgency score for workflow prioritization.
        
        Note: Urgency is used for queue prioritization and clinical alerts,
        NOT for model selection. High-urgency patients need the best
        quality analysis, not faster/cheaper models.
        """
        urgency = 0.0
        
        if patient.cancer_stage == "IV":
            urgency += 0.3
        
        if patient.ecog_status and patient.ecog_status.value >= 3:
            urgency += 0.4
        
        if patient.patient_intent == "Palliative":
            urgency += 0.2
        
        # Recent recurrence
        if patient.is_recurrence:
            urgency += 0.1
        
        return min(urgency, 1.0)


@dataclass  
class JudgeEnsemble:
    """
    Unified Judge Ensemble with dynamic complexity-aware model routing
    """
    
    # Judge configuration with lightweight and heavyweight models
    JUDGE_CONFIG = {
        JudgeRole.ACCURACY: {
            "model": LLMJudgeModel.GPT4O,  # Use GPT-4o (reliable, fast)
            "lightweight": [LLMJudgeModel.GPT4O_MINI, LLMJudgeModel.GEMINI_25_FLASH],
            "preferred": [LLMJudgeModel.GPT4O, LLMJudgeModel.GEMINI_25_PRO],
            "fallbacks": [LLMJudgeModel.GPT4O_MINI, LLMJudgeModel.GEMINI_25_FLASH],
            "weight": 0.25,
            "temperature": 0.1,
            "max_tokens": 1500
        },
        JudgeRole.SAFETY: {
            "model": LLMJudgeModel.CLAUDE_37_SONNET,  # Use Claude 3.7 (actually available)
            "lightweight": [LLMJudgeModel.CLAUDE_35_SONNET],
            "preferred": [LLMJudgeModel.CLAUDE_37_SONNET, LLMJudgeModel.CLAUDE_3_OPUS],
            "fallbacks": [LLMJudgeModel.CLAUDE_35_SONNET, LLMJudgeModel.GPT5],
            "weight": 0.20,
            "temperature": 0.0,
            "max_tokens": 1000
        },
        JudgeRole.COMPLETENESS: {
            "model": LLMJudgeModel.GEMINI_25_PRO,
            "lightweight": [LLMJudgeModel.GEMINI_25_FLASH],
            "preferred": [LLMJudgeModel.GEMINI_25_PRO],
            "fallbacks": [LLMJudgeModel.GEMINI_25_FLASH, LLMJudgeModel.GPT4O_MINI],
            "weight": 0.15,
            "temperature": 0.1,
            "max_tokens": 3000  # Higher for Gemini's reasoning tokens
        },
        JudgeRole.BIAS: {
            "model": LLMJudgeModel.GPT4O,  # Use GPT-4o (reliable)
            "lightweight": [LLMJudgeModel.GPT4O_MINI],
            "preferred": [LLMJudgeModel.GPT4O, LLMJudgeModel.GEMINI_25_PRO],
            "fallbacks": [LLMJudgeModel.GPT4O_MINI, LLMJudgeModel.CLAUDE_35_SONNET],
            "weight": 0.15,
            "temperature": 0.2,
            "max_tokens": 1500
        },
        JudgeRole.ROBUSTNESS: {
            "model": LLMJudgeModel.GEMINI_25_PRO,  # Use Gemini Pro (fast, good)
            "lightweight": [LLMJudgeModel.GEMINI_25_FLASH, LLMJudgeModel.GPT4O_MINI],
            "preferred": [LLMJudgeModel.GEMINI_25_PRO, LLMJudgeModel.GPT4O],
            "fallbacks": [LLMJudgeModel.GEMINI_25_FLASH, LLMJudgeModel.GPT4O_MINI],
            "weight": 0.10,
            "temperature": 0.3,
            "max_tokens": 2500  # Higher for Gemini's reasoning tokens
        },
        JudgeRole.CLINICAL_TEXT: {
            "model": LLMJudgeModel.CLAUDE_35_SONNET,  # Use Claude (strong at text)
            "lightweight": [LLMJudgeModel.GPT4O_MINI],
            "preferred": [LLMJudgeModel.CLAUDE_37_SONNET, LLMJudgeModel.CLAUDE_35_SONNET],
            "fallbacks": [LLMJudgeModel.GPT4O, LLMJudgeModel.GEMINI_25_PRO],
            "weight": 0.10,
            "temperature": 0.1,
            "max_tokens": 1200
        },
        JudgeRole.TRIALGPT: {
            "model": LLMJudgeModel.GPT4O_MINI,  # Use GPT-4o-mini (fast fallback)
            "lightweight": [LLMJudgeModel.GPT4O_MINI],
            "preferred": [LLMJudgeModel.GPT4O, LLMJudgeModel.GEMINI_25_FLASH],
            "fallbacks": [LLMJudgeModel.GPT4O_MINI, LLMJudgeModel.GEMINI_25_FLASH],
            "weight": 0.05,
            "temperature": 0.1,
            "max_tokens": 1000
        }
    }
    
    def __init__(
        self,
        enable_complexity_routing: bool = True,
        force_mode: Optional[str] = None,  # "heavy", "light", or None
        debug: bool = False
    ):
        self.providers = {}
        self.available_judges = {}
        self.complexity_analyzer = ComplexityAnalyzer()
        self.enable_complexity_routing = enable_complexity_routing
        self.force_mode = force_mode
        self.debug = debug
        self.metrics = {
            "evaluations_run": 0,
            "model_selections": {},
            "average_complexity": 0.0,
            "total_cost": 0.0
        }
    
    async def initialize(self):
        """Initialize providers and check model availability"""
        logger.info("Initializing Judge Ensemble...")
        
        # Initialize providers
        self.providers = {
            "openai": OpenAIProvider(),
            "anthropic": AnthropicProvider(),
            "google": GeminiProvider()  # Always include Gemini
        }
        
        # Check judge availability
        await self._check_judge_availability()
        
        logger.info(f"✅ Judge Ensemble initialized with {len(self.available_judges)} judges")
    
    async def _check_judge_availability(self):
        """Check which judge models are available"""
        for role, config in self.JUDGE_CONFIG.items():
            primary_model = config["model"].value
            fallback_models = [f.value for f in config.get("fallbacks", [])]
            all_models = [primary_model] + fallback_models
            
            model_found = False
            for model in all_models:
                for provider_name, provider in self.providers.items():
                    try:
                        models = await provider.available_models()
                        if model in models:
                            self.available_judges[role] = (provider_name, model)
                            if model == primary_model:
                                logger.info(f"✅ Judge {role.value}: {model} via {provider_name}")
                            else:
                                logger.info(f"⚠️ Judge {role.value} using fallback: {model} via {provider_name}")
                            model_found = True
                            break
                    except:
                        continue
                
                if model_found:
                    break
            
            if not model_found:
                logger.warning(f"❌ No models available for {role.value}, using GPT-4o fallback")
                self.available_judges[role] = ("openai", "gpt-4o")
    
    def _select_model_for_judge(
        self, 
        role: JudgeRole,
        complexity: float,
        urgency: float  # Kept for logging but not used in selection
    ) -> Tuple[str, str]:
        """Select optimal model based on complexity.
        
        Note: Urgency parameter is retained for backwards compatibility and
        logging but is NOT used for model selection. All patients get
        quality-based model selection regardless of urgency.
        """
        
        if self.force_mode == "heavy":
            # Force heavyweight models
            model_list = self.JUDGE_CONFIG[role]["preferred"]
        elif self.force_mode == "light":
            # Force lightweight models
            model_list = self.JUDGE_CONFIG[role]["lightweight"]
        else:
            # Dynamic selection based on complexity
            if complexity < 0.3:
                # Simple case - use lightweight
                model_list = self.JUDGE_CONFIG[role]["lightweight"]
                logger.debug(f"Using lightweight model for {role.value} (complexity: {complexity:.2f})")
            elif complexity > 0.6:
                # Complex case - use heavyweight
                model_list = self.JUDGE_CONFIG[role]["preferred"]
                logger.debug(f"Using heavyweight model for {role.value} (complexity: {complexity:.2f})")
            else:
                # Medium complexity - balance speed and accuracy
                model_list = (
                    self.JUDGE_CONFIG[role]["lightweight"] + 
                    self.JUDGE_CONFIG[role]["preferred"]
                )[:2]
        
        # Try to find an available model from the list
        for model in model_list:
            model_str = model.value if hasattr(model, 'value') else model
            for provider_name, provider in self.providers.items():
                try:
                    available = asyncio.run(provider.available_models())
                    if model_str in available:
                        self.metrics["model_selections"][role.value] = model_str
                        return provider_name, model_str
                except:
                    continue
        
        # Log high urgency for workflow prioritization
        if urgency > 0.7:
            logger.warning(f"High urgency patient (urgency={urgency:.2f}) - prioritize review")
        
        # Fallback to configured judge
        return self.available_judges.get(role, ("openai", "gpt-4o"))
    
    async def evaluate_match(
        self,
        patient: Patient,
        trial: ClinicalTrial,
        match_result: MatchResult,
        debug: bool = False
    ) -> Dict[str, Any]:
        """Run full judge ensemble evaluation with dynamic routing"""
        
        start_time = time.time()
        
        # Compute complexity for model selection and urgency for workflow
        complexity = self.complexity_analyzer.compute_complexity(patient, trial, match_result)
        urgency = self.complexity_analyzer.compute_urgency(patient)
        
        # Handle workflow prioritization for high-urgency patients
        if urgency > 0.7:
            logger.warning(f"High-urgency patient detected (urgency={urgency:.2f})")
            logger.warning(f"Patient {patient.patient_id}: Stage {patient.cancer_stage}, ECOG {patient.ecog_status}")
            logger.warning("Recommend: Expedited review, care team notification")
        
        self.metrics["evaluations_run"] += 1
        self.metrics["average_complexity"] = (
            (self.metrics["average_complexity"] * (self.metrics["evaluations_run"] - 1) + complexity) /
            self.metrics["evaluations_run"]
        )
        
        # Run judges with progress bar
        judge_scores = {}
        judges_to_run = [role for role in JudgeRole]
        
        for role in tqdm(judges_to_run, desc="Running judges", disable=not debug):
            provider_name, model = self._select_model_for_judge(role, complexity, urgency)
            provider = self.providers[provider_name]
            
            try:
                score = await self._run_single_judge(
                    role, provider, model, patient, trial, match_result
                )
                judge_scores[role.value] = score
            except Exception as e:
                logger.error(f"Error running {role.value}: {e}")
                judge_scores[role.value] = {
                    "score": 0.5,
                    "confidence": 0.0,
                    "reasoning": f"Error: {str(e)}"
                }
        
        # Aggregate scores
        final_score = self._aggregate_scores(judge_scores)
        
        # Calculate consensus
        scores = [s.get("score", 0.5) for s in judge_scores.values()]
        consensus = 1 - np.std(scores) if scores else 0.5
        
        elapsed_time = time.time() - start_time
        
        result = {
            "overall_score": final_score,
            "consensus": consensus,
            "complexity": complexity,
            "urgency": urgency,  # For workflow prioritization
            "workflow_priority": "high" if urgency > 0.7 else "normal",
            "individual_scores": judge_scores,
            "models_used": self.metrics["model_selections"],
            "evaluation_time": elapsed_time,
            "timestamp": datetime.now().isoformat()
        }
        
        if debug or self.debug:
            result["debug"] = {
                "force_mode": self.force_mode,
                "complexity_routing": self.enable_complexity_routing,
                "metrics": self.metrics
            }
        
        return result
    
    async def _run_single_judge(
        self,
        role: JudgeRole,
        provider: Any,
        model: str,
        patient: Patient,
        trial: ClinicalTrial,
        match_result: MatchResult
    ) -> Dict[str, Any]:
        """Run a single judge evaluation"""
        
        prompt = self._build_judge_prompt(role, patient, trial, match_result)
        config = self.JUDGE_CONFIG[role]
        
        try:
            # LLM providers expect system and user prompts, not messages
            system_prompt = "You are an expert clinical oncologist evaluating clinical trial eligibility. Always respond in valid JSON format."
            
            response = await provider.generate_json(
                model=model,
                system=system_prompt,
                user=prompt,
                temperature=config["temperature"],
                max_tokens=config["max_tokens"]
            )
            
            # Parse and validate response
            if isinstance(response, dict):
                return {
                    "score": float(response.get("score", 0.5)),
                    "confidence": float(response.get("confidence", 0.5)),
                    "reasoning": response.get("reasoning", ""),
                    "concerns": response.get("concerns", [])
                }
            else:
                return {
                    "score": 0.5,
                    "confidence": 0.0,
                    "reasoning": "Invalid response format"
                }
                
        except Exception as e:
            logger.error(f"Error in {role.value} with {model}: {e}")
            raise
    
    def _build_judge_prompt(
        self,
        role: JudgeRole,
        patient: Patient,
        trial: ClinicalTrial,
        match_result: MatchResult
    ) -> str:
        """Build role-specific evaluation prompt"""
        
        base_context = f"""
        Patient: {patient.age}yo {patient.gender}, {patient.cancer_type} Stage {patient.cancer_stage}
        Biomarkers: {', '.join([f"{b.name}:{b.status}" for b in patient.biomarkers_detected])}
        ECOG: {patient.ecog_status.value if patient.ecog_status else 'Unknown'}
        
        Trial: {trial.nct_id} - {trial.title}
        Phase: {trial.phase}
        
        Match Score: {match_result.overall_score:.2f}
        Confidence: {match_result.confidence:.2f}
        """
        
        role_prompts = {
            JudgeRole.ACCURACY: f"""
                {base_context}
                
                Evaluate the medical accuracy of this match. Consider:
                1. Are eligibility criteria correctly interpreted?
                2. Are biomarker requirements properly matched?
                3. Is the staging appropriate for this trial?
                
                Return JSON: {{"score": 0-1, "confidence": 0-1, "reasoning": "...", "concerns": [...]}}
            """,
            JudgeRole.SAFETY: f"""
                {base_context}
                
                Evaluate safety considerations. Consider:
                1. Patient's performance status vs trial demands
                2. Potential drug interactions
                3. Risk factors and comorbidities
                
                Return JSON: {{"score": 0-1, "confidence": 0-1, "reasoning": "...", "concerns": [...]}}
            """,
            JudgeRole.COMPLETENESS: f"""
                {base_context}
                
                Evaluate match completeness. Consider:
                1. Are all eligibility criteria addressed?
                2. Is key information missing?
                3. Are assumptions clearly stated?
                
                Return JSON: {{"score": 0-1, "confidence": 0-1, "reasoning": "...", "concerns": [...]}}
            """
        }
        
        return role_prompts.get(role, base_context + "\nEvaluate this match. Return JSON with score, confidence, and reasoning.")
    
    def _aggregate_scores(self, judge_scores: Dict[str, Dict]) -> float:
        """Aggregate individual judge scores into final score"""
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for role, score_data in judge_scores.items():
            if role in [r.value for r in JudgeRole]:
                weight = self.JUDGE_CONFIG[JudgeRole(role)]["weight"]
                score = score_data.get("score", 0.5)
                confidence = score_data.get("confidence", 1.0)
                
                # Weight by both configured weight and confidence
                effective_weight = weight * confidence
                weighted_sum += score * effective_weight
                total_weight += effective_weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return 0.5  # Default neutral score
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get evaluation metrics"""
        return {
            **self.metrics,
            "judges_available": len(self.available_judges),
            "complexity_routing_enabled": self.enable_complexity_routing,
            "force_mode": self.force_mode
        }
