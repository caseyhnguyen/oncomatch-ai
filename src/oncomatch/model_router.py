"""
Advanced model routing system for clinical trial matching.

This module provides a sophisticated, configurable model routing system that
selects the optimal LLM based on case complexity, cost constraints, and
performance requirements.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, IntEnum
from dataclasses import dataclass, field
import json
from functools import lru_cache

from oncomatch.models import Patient, ClinicalTrial

logger = logging.getLogger(__name__)


class ComplexityLevel(IntEnum):
    """Standardized complexity levels for routing decisions."""
    TRIVIAL = 0     # Very simple cases
    SIMPLE = 1      # Basic eligibility checks
    MODERATE = 2    # Standard clinical cases
    COMPLEX = 3     # Multi-biomarker, multi-line therapy
    VERY_COMPLEX = 4  # Phase 1, rare mutations, heavy pretreat
    EXTREME = 5     # Exceptional complexity


class ModelCapability(str, Enum):
    """Model capabilities for matching requirements."""
    REASONING = "reasoning"
    MEDICAL = "medical"
    SAFETY = "safety"
    SPEED = "speed"
    COST_EFFECTIVE = "cost_effective"
    JSON_RELIABLE = "json_reliable"
    BIOMARKER_EXPERT = "biomarker_expert"


@dataclass
class ModelProfile:
    """Complete profile of an LLM model."""
    name: str
    provider: str
    capabilities: List[ModelCapability]
    cost_per_1k_tokens: float
    avg_latency_ms: int
    quality_score: float  # 0-1 scale
    max_context: int = 128000
    supports_json_mode: bool = True
    supports_function_calling: bool = False
    specialized_domains: List[str] = field(default_factory=list)
    
    @property
    def cost_quality_ratio(self) -> float:
        """Calculate cost-effectiveness (higher is better)."""
        if self.cost_per_1k_tokens == 0:
            return float('inf')
        return self.quality_score / self.cost_per_1k_tokens
    
    @property
    def speed_quality_ratio(self) -> float:
        """Calculate speed-effectiveness (higher is better)."""
        if self.avg_latency_ms == 0:
            return float('inf')
        return (self.quality_score * 1000) / self.avg_latency_ms


class ModelRegistry:
    """Registry of all available models with their profiles."""
    
    def __init__(self):
        self.models: Dict[str, ModelProfile] = self._initialize_models()
        self.available_models: Dict[str, bool] = {}
        self._check_availability()
    
    def _initialize_models(self) -> Dict[str, ModelProfile]:
        """Initialize model profiles with September 2025 data."""
        return {
            # GPT-5 Series (Best reasoning)
            "gpt-5": ModelProfile(
                name="gpt-5",
                provider="openai",
                capabilities=[ModelCapability.REASONING, ModelCapability.MEDICAL, 
                             ModelCapability.BIOMARKER_EXPERT],
                cost_per_1k_tokens=1.25,
                avg_latency_ms=10000,
                quality_score=0.98,
                specialized_domains=["oncology", "genomics"]
            ),
            "gpt-5-mini": ModelProfile(
                name="gpt-5-mini",
                provider="openai",
                capabilities=[ModelCapability.REASONING, ModelCapability.COST_EFFECTIVE],
                cost_per_1k_tokens=0.25,
                avg_latency_ms=7000,
                quality_score=0.92
            ),
            "gpt-5-nano": ModelProfile(
                name="gpt-5-nano",
                provider="openai",
                capabilities=[ModelCapability.SPEED, ModelCapability.COST_EFFECTIVE],
                cost_per_1k_tokens=0.05,
                avg_latency_ms=5000,
                quality_score=0.85
            ),
            
            # Claude 3.7 Sonnet (PRIMARY - February 2025, newest stable)
            "claude-3-7-sonnet-20250219": ModelProfile(
                name="claude-3-7-sonnet-20250219",
                provider="anthropic",
                capabilities=[ModelCapability.REASONING, ModelCapability.SAFETY,
                             ModelCapability.MEDICAL, ModelCapability.COST_EFFECTIVE],
                cost_per_1k_tokens=0.03,
                avg_latency_ms=6000,
                quality_score=0.95,
                specialized_domains=["medical_ethics", "safety", "clinical_reasoning"]
            ),
            # Claude 3.5 Sonnet (Stable, reliable)
            "claude-3-5-sonnet-20241022": ModelProfile(
                name="claude-3-5-sonnet-20241022",
                provider="anthropic",
                capabilities=[ModelCapability.REASONING, ModelCapability.SAFETY,
                             ModelCapability.MEDICAL, ModelCapability.COST_EFFECTIVE],
                cost_per_1k_tokens=0.015,
                avg_latency_ms=6000,
                quality_score=0.92,
                specialized_domains=["medical_ethics"]
            ),
            # Claude 3 Opus (Fallback)
            "claude-3-opus-20240229": ModelProfile(
                name="claude-3-opus-20240229",
                provider="anthropic",
                capabilities=[ModelCapability.REASONING, ModelCapability.SAFETY,
                             ModelCapability.MEDICAL],
                cost_per_1k_tokens=0.075,
                avg_latency_ms=9000,
                quality_score=0.94,
                specialized_domains=["medical_ethics", "safety"]
            ),
            # Claude 4.1 series (Future - may not be generally available yet via direct Anthropic API)
            "claude-opus-4.1": ModelProfile(
                name="claude-opus-4.1",
                provider="anthropic",
                capabilities=[ModelCapability.REASONING, ModelCapability.SAFETY,
                             ModelCapability.MEDICAL],
                cost_per_1k_tokens=0.10,
                avg_latency_ms=10000,
                quality_score=0.98,
                specialized_domains=["medical_ethics", "safety", "complex_reasoning"]
            ),
            "claude-sonnet-4": ModelProfile(
                name="claude-sonnet-4",
                provider="anthropic",
                capabilities=[ModelCapability.REASONING, ModelCapability.SAFETY,
                             ModelCapability.MEDICAL, ModelCapability.COST_EFFECTIVE],
                cost_per_1k_tokens=0.05,
                avg_latency_ms=6000,
                quality_score=0.96,
                specialized_domains=["medical_ethics", "safety"]
            ),
            
            # Gemini 2.5 Series (PRIMARY for speed - 887 tok/s)
            "gemini-2.5-pro": ModelProfile(
                name="gemini-2.5-pro",
                provider="google",
                capabilities=[ModelCapability.REASONING, ModelCapability.MEDICAL,
                             ModelCapability.COST_EFFECTIVE],
                cost_per_1k_tokens=0.35,
                avg_latency_ms=5000,
                quality_score=0.91,
                specialized_domains=["oncology", "clinical_reasoning"]
            ),
            "gemini-2.5-flash": ModelProfile(
                name="gemini-2.5-flash",
                provider="google",
                capabilities=[ModelCapability.SPEED, ModelCapability.COST_EFFECTIVE,
                             ModelCapability.JSON_RELIABLE, ModelCapability.REASONING],
                cost_per_1k_tokens=0.01,  # Very cost-effective
                avg_latency_ms=2000,  # FASTEST (887 tok/s)
                quality_score=0.87,  # Better than initially thought
                specialized_domains=["general_medical"]
            ),
            "gemini-2.5-flash-lite": ModelProfile(
                name="gemini-2.5-flash-lite",
                provider="google",
                capabilities=[ModelCapability.SPEED, ModelCapability.COST_EFFECTIVE],
                cost_per_1k_tokens=0.005,
                avg_latency_ms=1500,
                quality_score=0.75  # Slightly better than baseline
            ),
            
            # GPT-4 Series (Previous gen)
            "gpt-4o": ModelProfile(
                name="gpt-4o",
                provider="openai",
                capabilities=[ModelCapability.REASONING],
                cost_per_1k_tokens=0.30,
                avg_latency_ms=7000,
                quality_score=0.88
            ),
            "gpt-4o-mini": ModelProfile(
                name="gpt-4o-mini",
                provider="openai",
                capabilities=[ModelCapability.COST_EFFECTIVE, ModelCapability.JSON_RELIABLE],
                cost_per_1k_tokens=0.015,
                avg_latency_ms=4000,
                quality_score=0.80
            ),
        }
    
    def _check_availability(self):
        """Check which models are available based on API keys."""
        # OpenAI models
        if os.getenv('OPENAI_API_KEY'):
            for model_name, profile in self.models.items():
                if profile.provider == "openai":
                    self.available_models[model_name] = True
        
        # Anthropic models
        if os.getenv('ANTHROPIC_API_KEY'):
            for model_name, profile in self.models.items():
                if profile.provider == "anthropic":
                    self.available_models[model_name] = True
        
        # Google models
        if os.getenv('GOOGLE_API_KEY'):
            for model_name, profile in self.models.items():
                if profile.provider == "google":
                    self.available_models[model_name] = True
    
    def get_available_models(self, 
                           min_quality: float = 0.0,
                           max_cost: float = float('inf'),
                           max_latency_ms: int = float('inf'),
                           required_capabilities: List[ModelCapability] = None
                          ) -> List[ModelProfile]:
        """Get all available models matching the criteria."""
        available = []
        for model_name, is_available in self.available_models.items():
            if not is_available:
                continue
            
            profile = self.models[model_name]
            
            # Apply filters
            if profile.quality_score < min_quality:
                continue
            if profile.cost_per_1k_tokens > max_cost:
                continue
            if profile.avg_latency_ms > max_latency_ms:
                continue
            
            # Check required capabilities
            if required_capabilities:
                if not all(cap in profile.capabilities for cap in required_capabilities):
                    continue
            
            available.append(profile)
        
        return available


class ComplexityAnalyzer:
    """Sophisticated complexity analysis for patient-trial pairs."""
    
    @staticmethod
    def calculate_complexity(patient: Patient, trial: ClinicalTrial) -> Tuple[ComplexityLevel, Dict[str, Any]]:
        """
        Calculate detailed complexity with explanatory factors.
        
        Returns:
            Tuple of (complexity_level, factors_dict)
        """
        factors = {
            "biomarker_complexity": 0,
            "therapy_complexity": 0,
            "trial_complexity": 0,
            "patient_complexity": 0,
            "safety_complexity": 0,
            "details": []
        }
        
        # Biomarker complexity
        biomarker_count = len(patient.biomarkers_detected)
        if biomarker_count > 5:
            factors["biomarker_complexity"] = 3
            factors["details"].append(f"High biomarker complexity ({biomarker_count} markers)")
        elif biomarker_count > 3:
            factors["biomarker_complexity"] = 2
            factors["details"].append(f"Moderate biomarker complexity ({biomarker_count} markers)")
        elif biomarker_count > 0:
            factors["biomarker_complexity"] = 1
        
        # Check for rare/complex biomarkers
        complex_biomarkers = ["NTRK", "RET", "BRAF", "MSI-H", "TMB-H", "HRD"]
        for biomarker in patient.biomarkers_detected:
            if any(cb in biomarker.name.upper() for cb in complex_biomarkers):
                factors["biomarker_complexity"] += 1
                factors["details"].append(f"Rare/complex biomarker: {biomarker.name}")
        
        # Therapy complexity
        line_of_therapy = patient.get_line_of_therapy()
        if line_of_therapy > 3:
            factors["therapy_complexity"] = 3
            factors["details"].append(f"Heavily pretreated (line {line_of_therapy})")
        elif line_of_therapy > 1:
            factors["therapy_complexity"] = 2
            factors["details"].append(f"Prior therapy (line {line_of_therapy})")
        elif line_of_therapy > 0:
            factors["therapy_complexity"] = 1
        
        # Trial complexity
        if trial.phase and "Phase 1" in trial.phase.value:
            factors["trial_complexity"] += 2
            factors["details"].append("Phase 1 trial (complex eligibility)")
        elif trial.phase and "Phase 2" in trial.phase.value:
            factors["trial_complexity"] += 1
        
        inclusion_count = len(trial.eligibility.inclusion_criteria)
        if inclusion_count > 15:
            factors["trial_complexity"] += 2
            factors["details"].append(f"Many inclusion criteria ({inclusion_count})")
        elif inclusion_count > 10:
            factors["trial_complexity"] += 1
        
        # Patient complexity
        if patient.cancer_stage == "IV":
            factors["patient_complexity"] += 2
            factors["details"].append("Stage IV disease")
        elif patient.cancer_stage == "III":
            factors["patient_complexity"] += 1
            factors["details"].append("Stage III disease")
        
        if patient.ecog_status and patient.ecog_status.value >= 3:
            factors["patient_complexity"] += 2
            factors["details"].append(f"Poor performance status (ECOG {patient.ecog_status.value})")
        elif patient.ecog_status and patient.ecog_status.value >= 2:
            factors["patient_complexity"] += 1
        
        # Safety complexity
        if patient.other_conditions:
            relevant_conditions = ["heart disease", "kidney disease", "liver disease", 
                                  "immunodeficiency", "autoimmune"]
            for condition in patient.other_conditions:
                if any(rc in condition.lower() for rc in relevant_conditions):
                    factors["safety_complexity"] += 1
                    factors["details"].append(f"Comorbidity: {condition}")
        
        # Calculate total complexity
        total_score = sum([
            factors["biomarker_complexity"],
            factors["therapy_complexity"],
            factors["trial_complexity"],
            factors["patient_complexity"],
            factors["safety_complexity"]
        ])
        
        # Map to complexity level
        if total_score >= 10:
            level = ComplexityLevel.EXTREME
        elif total_score >= 7:
            level = ComplexityLevel.VERY_COMPLEX
        elif total_score >= 5:
            level = ComplexityLevel.COMPLEX
        elif total_score >= 3:
            level = ComplexityLevel.MODERATE
        elif total_score >= 1:
            level = ComplexityLevel.SIMPLE
        else:
            level = ComplexityLevel.TRIVIAL
        
        factors["total_score"] = total_score
        factors["level"] = level.name
        
        return level, factors


@dataclass
class RoutingDecision:
    """Detailed routing decision with explanation."""
    selected_model: str
    complexity_level: ComplexityLevel
    complexity_factors: Dict[str, Any]
    selection_reasons: List[str]
    alternative_models: List[str]
    estimated_cost: float
    estimated_latency_ms: int
    quality_score: float


class SmartModelRouter:
    """
    Intelligent model routing with multi-factor optimization.
    
    This router considers complexity, cost, latency, and quality to select
    the optimal model for each patient-trial pair.
    """
    
    def __init__(self, 
                 budget_per_trial_usd: float = 0.05,
                 max_latency_ms: int = 15000,
                 min_quality: float = 0.7,
                 prefer_providers: List[str] = None):
        """
        Initialize the router with constraints.
        
        Args:
            budget_per_trial_usd: Maximum cost per trial analysis
            max_latency_ms: Maximum acceptable latency
            min_quality: Minimum quality score (0-1)
            prefer_providers: Preferred providers in order
        """
        self.registry = ModelRegistry()
        self.analyzer = ComplexityAnalyzer()
        self.budget = budget_per_trial_usd
        self.max_latency = max_latency_ms
        self.min_quality = min_quality
        # OPTIMIZED provider priority: Gemini first (fastest 887 tok/s), OpenAI second, Anthropic last (rate limits)
        self.prefer_providers = prefer_providers or ["google", "openai", "anthropic"]
        
        # Complexity to quality mapping
        self.complexity_quality_requirements = {
            ComplexityLevel.TRIVIAL: 0.70,
            ComplexityLevel.SIMPLE: 0.75,
            ComplexityLevel.MODERATE: 0.85,
            ComplexityLevel.COMPLEX: 0.90,
            ComplexityLevel.VERY_COMPLEX: 0.95,
            ComplexityLevel.EXTREME: 0.98
        }
    
    def route(self, patient: Patient, trial: ClinicalTrial, 
              override_model: Optional[str] = None) -> RoutingDecision:
        """
        Route to the optimal model based on all factors.
        
        Args:
            patient: Patient data
            trial: Clinical trial data
            override_model: Force a specific model (for testing)
            
        Returns:
            RoutingDecision with selected model and reasoning
        """
        # Allow manual override
        if override_model and override_model in self.registry.available_models:
            return RoutingDecision(
                selected_model=override_model,
                complexity_level=ComplexityLevel.SIMPLE,
                complexity_factors={},
                selection_reasons=["Manual override"],
                alternative_models=[],
                estimated_cost=self.registry.models[override_model].cost_per_1k_tokens,
                estimated_latency_ms=self.registry.models[override_model].avg_latency_ms,
                quality_score=self.registry.models[override_model].quality_score
            )
        
        # Analyze complexity
        complexity_level, complexity_factors = self.analyzer.calculate_complexity(patient, trial)
        
        # Determine requirements based on complexity
        min_quality_required = max(
            self.min_quality,
            self.complexity_quality_requirements[complexity_level]
        )
        
        # Get candidate models
        required_capabilities = self._get_required_capabilities(complexity_level, patient, trial)
        
        candidates = self.registry.get_available_models(
            min_quality=min_quality_required,
            max_cost=self.budget * 1000,  # Convert to per-1k tokens
            max_latency_ms=self.max_latency,
            required_capabilities=required_capabilities
        )
        
        if not candidates:
            # Relax constraints if no candidates
            candidates = self.registry.get_available_models(
                min_quality=min_quality_required * 0.9,  # Relax quality by 10%
                max_cost=self.budget * 1500,  # Increase budget by 50%
                max_latency_ms=self.max_latency * 2  # Double latency allowance
            )
        
        if not candidates:
            # Emergency fallback
            candidates = self.registry.get_available_models(min_quality=0.7)
        
        # Score and rank candidates
        scored_models = []
        for model in candidates:
            score = self._score_model(model, complexity_level, required_capabilities, patient)
            scored_models.append((model, score))
        
        # Sort by score (higher is better)
        scored_models.sort(key=lambda x: x[1], reverse=True)
        
        # Select best model
        best_model = scored_models[0][0] if scored_models else None
        
        if not best_model:
            # Ultimate fallback
            logger.error("No suitable model found, using emergency fallback")
            best_model = self.registry.models.get("gpt-4o-mini")
        
        # Build decision
        selection_reasons = self._explain_selection(
            best_model, complexity_level, complexity_factors, scored_models
        )
        
        return RoutingDecision(
            selected_model=best_model.name,
            complexity_level=complexity_level,
            complexity_factors=complexity_factors,
            selection_reasons=selection_reasons,
            alternative_models=[m[0].name for m in scored_models[1:4]],  # Top 3 alternatives
            estimated_cost=best_model.cost_per_1k_tokens / 1000,
            estimated_latency_ms=best_model.avg_latency_ms,
            quality_score=best_model.quality_score
        )
    
    def _get_required_capabilities(self, 
                                  complexity: ComplexityLevel,
                                  patient: Patient,
                                  trial: ClinicalTrial) -> List[ModelCapability]:
        """Determine required capabilities based on the case."""
        capabilities = []
        
        # Complexity-based requirements
        if complexity >= ComplexityLevel.COMPLEX:
            capabilities.append(ModelCapability.REASONING)
        
        if complexity >= ComplexityLevel.VERY_COMPLEX:
            capabilities.append(ModelCapability.MEDICAL)
        
        # Patient-specific requirements
        if patient.ecog_status and patient.ecog_status.value >= 3:
            capabilities.append(ModelCapability.SAFETY)
        
        if len(patient.biomarkers_detected) > 3:
            capabilities.append(ModelCapability.BIOMARKER_EXPERT)
        
        # Trial-specific requirements
        if trial.phase and "Phase 1" in trial.phase.value:
            capabilities.append(ModelCapability.SAFETY)
        
        return capabilities
    
    def _score_model(self, 
                    model: ModelProfile,
                    complexity: ComplexityLevel,
                    required_capabilities: List[ModelCapability],
                    patient: Optional[Patient] = None) -> float:
        """
        Score a model based on multiple factors.
        
        Higher score is better.
        """
        score = 0.0
        
        # Quality score (weighted by complexity)
        quality_weight = 1 + (complexity.value * 0.2)  # More weight for complex cases
        score += model.quality_score * quality_weight * 100
        
        # Cost efficiency
        if complexity <= ComplexityLevel.MODERATE:
            # Prefer cost-effective for simple cases
            score += model.cost_quality_ratio * 10
        
        # Speed (important for all cases, but less critical for complex)
        speed_weight = 2.0 - (complexity.value * 0.2)
        score += model.speed_quality_ratio * speed_weight
        
        # Capability matching
        for cap in required_capabilities:
            if cap in model.capabilities:
                score += 20
        
        # Provider preference
        if model.provider in self.prefer_providers:
            idx = self.prefer_providers.index(model.provider)
            score += (len(self.prefer_providers) - idx) * 5
        
        # Specialized domain bonus
        if "oncology" in model.specialized_domains:
            score += 15
        if patient and "genomics" in model.specialized_domains and len(patient.biomarkers_detected) > 2:
            score += 10
        
        return score
    
    def _explain_selection(self,
                          selected_model: ModelProfile,
                          complexity: ComplexityLevel,
                          factors: Dict[str, Any],
                          alternatives: List[Tuple[ModelProfile, float]]) -> List[str]:
        """Generate human-readable explanation for the selection."""
        reasons = []
        
        # Complexity explanation
        reasons.append(f"Case complexity: {complexity.name} (score: {factors['total_score']})")
        
        # Model strengths
        if ModelCapability.REASONING in selected_model.capabilities:
            reasons.append(f"{selected_model.name} selected for strong reasoning capabilities")
        
        if complexity >= ComplexityLevel.VERY_COMPLEX:
            reasons.append(f"High complexity requires {selected_model.name}'s advanced capabilities")
        elif complexity <= ComplexityLevel.SIMPLE:
            reasons.append(f"Simple case allows efficient processing with {selected_model.name}")
        
        # Quality/cost trade-off
        if selected_model.cost_quality_ratio > 20:
            reasons.append("Excellent cost-quality balance")
        
        # Speed consideration
        if selected_model.avg_latency_ms <= 4000:
            reasons.append("Fast processing for quick results")
        
        # Add specific complexity factors
        if factors["details"]:
            reasons.append(f"Key factors: {', '.join(factors['details'][:2])}")
        
        return reasons


# Convenience function for easy integration
@lru_cache(maxsize=1)
def get_router() -> SmartModelRouter:
    """Get or create the singleton router instance."""
    return SmartModelRouter(
        budget_per_trial_usd=float(os.getenv("LLM_BUDGET_USD", "0.05")),
        max_latency_ms=int(os.getenv("LLM_BUDGET_MS", "15000")),
        min_quality=float(os.getenv("MIN_MODEL_QUALITY", "0.7"))
    )
