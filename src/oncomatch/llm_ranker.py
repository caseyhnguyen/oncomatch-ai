"""
Multi-model LLM ranking system for clinical trial matching.
Uses GPT-4 for primary medical reasoning with fallback to other models.
"""

import asyncio
import json
import logging
import os
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

import openai
from anthropic import Anthropic
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm

# Load environment variables
load_dotenv()

from oncomatch.models import (
    Patient, 
    ClinicalTrial, 
    MatchResult, 
    MatchReason,
    ECOGStatus
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LLMModel(str, Enum):
    """Available LLM models for ranking (September 2025)."""
    # OpenAI GPT-5 series (Best reasoning models as of Sept 2025)
    GPT5 = "gpt-5"  # Superior reasoning + integrated routing ($1.25/1M input, $10/1M output)
    GPT5_MINI = "gpt-5-mini"  # Balanced GPT-5 ($0.25/1M input, $2/1M output)
    GPT5_NANO = "gpt-5-nano"  # Fast GPT-5 ($0.05/1M input, $0.40/1M output)
    
    # OpenAI o-series (Previous gen reasoning, now superseded by GPT-5)
    O4_MINI = "o4-mini"  # Mini reasoning model (fallback)
    O3 = "o3"  # Reasoning model (April 2025, fallback)
    O3_MINI = "o3-mini"  # Fast reasoning (fallback)
    O3_PRO = "o3-pro"  # Professional reasoning (fallback)
    
    # OpenAI Fast Models (Optimized for speed)
    GPT41_NANO = "gpt-4.1-nano"  # Fastest OpenAI model (~5s latency)
    
    # OpenAI GPT-4 series (Previous generation)
    GPT41 = "gpt-4.1-2025-04-14"  # GPT-4.1 update
    GPT4O = "gpt-4o"  # GPT-4o
    GPT4O_MINI = "gpt-4o-mini"  # Fast multimodal, budget-friendly
    
    # Anthropic Claude series (Claude 3.7 is current stable)
    CLAUDE_37_SONNET = "claude-3-7-sonnet-20250219"  # PRIMARY - Feb 2025
    CLAUDE_35_SONNET = "claude-3-5-sonnet-20241022"  # Stable fallback
    CLAUDE_3_OPUS = "claude-3-opus-20240229"  # Previous gen fallback
    
    # Google Gemini series  
    GEMINI_25_PRO = "gemini-2.5-pro"  # Most capable
    GEMINI_25_FLASH = "gemini-2.5-flash"  # Fast with good reasoning
    GEMINI_25_FLASH_LITE = "gemini-2.5-flash-lite"  # Fastest Gemini (887 tokens/sec)
    
    # Specialized medical models (availability uncertain)
    TRIALGPT = "trialgpt"  # NIH's specialized model


@dataclass
class RankingConfig:
    """Configuration for LLM ranking with model defaults."""
    primary_model: LLMModel = LLMModel.GEMINI_25_FLASH  # Default to Gemini Flash for balance
    fallback_model: LLMModel = LLMModel.GPT4O_MINI  # Reliable fallback
    safety_model: LLMModel = LLMModel.CLAUDE_37_SONNET  # Claude 3.7 for safety/ethics
    speed_model: LLMModel = LLMModel.GEMINI_25_FLASH_LITE  # Fastest model (887 tokens/sec)
    fast_openai_model: LLMModel = LLMModel.GPT41_NANO  # Fastest OpenAI (~5s latency)
    complex_model: LLMModel = LLMModel.GPT5  # Best for complex reasoning
    medical_model: LLMModel = LLMModel.TRIALGPT  # Specialized medical model
    temperature: float = 0.1
    max_tokens: int = 3000  # Increased for reasoning models
    use_structured_output: bool = True
    enable_safety_checks: bool = True
    cost_optimization: bool = True
    enable_model_routing: bool = True  # Smart routing based on complexity
    use_reasoning_models: bool = True  # Enable o-series reasoning
    enable_claude: bool = True  # Enable Claude models
    enable_gemini: bool = True  # Enable Gemini models
    enable_medical: bool = False  # Enable specialized medical models (if available)


class LLMRanker:
    """Multi-model LLM system for ranking clinical trials."""
    
    def __init__(self, config: RankingConfig = None):
        self.config = config or RankingConfig()
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize LLM clients (API keys from environment variables)."""
        # Initialize OpenAI client
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            try:
                self.openai_client = openai.AsyncOpenAI(
                    api_key=openai_key,
                    base_url=os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
                )
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
                self.openai_client = None
        else:
            logger.warning("OPENAI_API_KEY not found in environment")
            self.openai_client = None
        
        # Initialize Anthropic client
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        if anthropic_key:
            try:
                self.anthropic_client = Anthropic(
                    api_key=anthropic_key,
                    base_url=os.getenv('ANTHROPIC_BASE_URL', 'https://api.anthropic.com')  # Fix: Use correct base URL
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic client: {e}")
                self.anthropic_client = None
        else:
            logger.warning("ANTHROPIC_API_KEY not found in environment")
            self.anthropic_client = None
        
        # Initialize Google Gemini client
        google_key = os.getenv('GOOGLE_API_KEY')
        if google_key:
            try:
                genai.configure(api_key=google_key)
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini client: {e}")
        else:
            logger.warning("GOOGLE_API_KEY not found in environment")
    
    def _create_eligibility_prompt(self, patient: Patient, trial: ClinicalTrial) -> str:
        """Create detailed prompt for eligibility assessment."""
        # Format patient information
        patient_info = f"""
PATIENT INFORMATION:
- ID: {patient.patient_id}
- Age: {patient.age} years
- Gender: {patient.gender.value}
- Cancer Type: {patient.cancer_type}
- Stage: {patient.cancer_stage}
- Grade: {patient.cancer_grade or 'Not specified'}
- ECOG Status: {patient.ecog_status.value if patient.ecog_status else 'Not specified'}
- Treatment Stage: {patient.treatment_stage.value if patient.treatment_stage else 'Not specified'}

BIOMARKERS:
Detected: {', '.join([f"{b.name} ({b.status})" for b in patient.biomarkers_detected]) or 'None'}
Ruled Out: {', '.join(patient.biomarkers_ruled_out) or 'None'}

TREATMENT HISTORY:
- Surgeries: {', '.join(patient.surgeries) or 'None'}
- Previous Treatments: {', '.join(patient.previous_treatments) or 'None'}
- Current Medications: {', '.join(patient.current_medications) or 'None'}
- Line of Therapy: {patient.get_line_of_therapy()}

COMORBIDITIES:
{', '.join(patient.other_conditions) or 'None'}

ADDITIONAL INFO:
- Smoking: {patient.smoking_status or 'Unknown'}
- Drinking: {patient.drinking_status or 'Unknown'}
- Intent: {patient.patient_intent or 'Not specified'}
"""
        
        # Format trial information
        trial_info = f"""
CLINICAL TRIAL:
- NCT ID: {trial.nct_id}
- Title: {trial.title}
- Phase: {trial.phase.value if trial.phase else 'Not specified'}
- Conditions: {', '.join(trial.conditions)}
- Interventions: {', '.join(trial.interventions)}

ELIGIBILITY CRITERIA:
Inclusion:
{chr(10).join(['- ' + criterion for criterion in trial.eligibility.inclusion_criteria[:10]]) or 'Not specified'}

Exclusion:
{chr(10).join(['- ' + criterion for criterion in trial.eligibility.exclusion_criteria[:10]]) or 'Not specified'}

STRUCTURED REQUIREMENTS:
- Age Range: {trial.eligibility.min_age or 'No minimum'} - {trial.eligibility.max_age or 'No maximum'}
- Required Biomarkers: {', '.join(trial.eligibility.required_biomarkers) or 'None'}
- Excluded Biomarkers: {', '.join(trial.eligibility.excluded_biomarkers) or 'None'}
- Max Prior Therapies: {trial.eligibility.max_prior_therapies or 'No limit'}
"""
        
        prompt = f"""You are an expert oncologist evaluating whether a cancer patient is eligible for a clinical trial.

{patient_info}

{trial_info}

TASK: Evaluate the patient's eligibility for this trial. For EACH major criterion, assess whether the patient meets it.

Consider these key factors:
1. Cancer type and stage compatibility
2. Biomarker requirements (both required and excluded)
3. Prior treatment limitations
4. Age and performance status requirements
5. Comorbidities that may exclude participation
6. Safety concerns

Provide your analysis in the following JSON format:
{{
    "is_eligible": boolean,
    "confidence": 0.0-1.0,
    "eligibility_score": 0.0-1.0,
    "biomarker_score": 0.0-1.0,
    "safety_score": 0.0-1.0,
    "match_reasons": [
        {{
            "criterion": "specific criterion text",
            "matched": boolean,
            "explanation": "detailed explanation",
            "confidence": 0.0-1.0,
            "category": "inclusion|exclusion|biomarker|safety"
        }}
    ],
    "summary": "2-3 sentence summary of eligibility decision",
    "safety_concerns": ["list of specific safety concerns if any"],
    "warnings": ["list of warnings or uncertainties"]
}}

Be conservative - if uncertain about eligibility, lean towards exclusion for patient safety."""
        
        return prompt
    
    def _create_comparison_prompt(self, patient: Patient, trials: List[ClinicalTrial]) -> str:
        """Create prompt for comparing multiple trials."""
        trials_info = []
        for i, trial in enumerate(trials[:5]):  # Limit to 5 for context
            trials_info.append(f"""
TRIAL {i+1} - {trial.nct_id}:
- Title: {trial.title}
- Phase: {trial.phase.value if trial.phase else 'Not specified'}
- Key Interventions: {', '.join(trial.interventions[:3])}
- Required Biomarkers: {', '.join(trial.eligibility.required_biomarkers) or 'None'}
""")
        
        prompt = f"""Compare these clinical trials for the following patient and rank them:

PATIENT SUMMARY:
- Cancer: {patient.cancer_type} Stage {patient.cancer_stage}
- Biomarkers: {', '.join([b.name for b in patient.biomarkers_detected])}
- Prior Lines: {patient.get_line_of_therapy()}
- ECOG: {patient.ecog_status.value if patient.ecog_status else 'Unknown'}

TRIALS TO COMPARE:
{''.join(trials_info)}

Rank these trials from best to worst match, considering:
1. Biomarker targeting specificity
2. Appropriateness for disease stage
3. Innovation/potential benefit
4. Safety profile given patient's condition

Return a JSON array of NCT IDs in order of preference with brief rationale."""
        
        return prompt
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def _call_openai(self, prompt: str, model: LLMModel) -> Dict[str, Any]:
        """Call OpenAI API with retry logic."""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
        
        try:
            # Build request parameters
            params = {
                "model": model.value,
                "messages": [
                    {"role": "system", "content": "You are an expert clinical oncologist with deep knowledge of clinical trials and eligibility criteria."},
                    {"role": "user", "content": prompt}
                ],
            }
            
            # Handle GPT-5 specific requirements
            if model in {LLMModel.GPT5, LLMModel.GPT5_MINI, LLMModel.GPT5_NANO}:
                # GPT-5 models use max_completion_tokens instead of max_tokens
                params["max_completion_tokens"] = self.config.max_tokens
                # GPT-5 models only support temperature=1 (default)
                # Don't add temperature parameter if it's not 1
                if self.config.temperature == 1.0:
                    params["temperature"] = 1.0
            else:
                # Other models use standard parameters
                params["max_tokens"] = self.config.max_tokens
                params["temperature"] = self.config.temperature
            
            # Add response format if structured output is enabled
            if self.config.use_structured_output:
                params["response_format"] = {"type": "json_object"}
            
            response = await self.openai_client.chat.completions.create(**params)
            
            content = response.choices[0].message.content
            return json.loads(content) if content else {}
            
        except Exception as e:
            logger.error(f"OpenAI API error with {model.value}: {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def _call_anthropic(self, prompt: str) -> Dict[str, Any]:
        """Call Anthropic Claude API."""
        if not self.anthropic_client:
            raise ValueError("Anthropic client not initialized")
        
        try:
            # Use Claude 3.7 Sonnet as the primary model (Feb 2025)
            response = self.anthropic_client.messages.create(
                model="claude-3-7-sonnet-20250219",  # PRIMARY - Claude 3.7 Sonnet
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system="You are an expert clinical oncologist evaluating clinical trial eligibility. Always respond in valid JSON format.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text if response.content else "{}"
            return json.loads(content)
            
        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def _call_gemini(self, prompt: str, model_name: LLMModel = None) -> Dict[str, Any]:
        """Call Google Gemini API."""
        try:
            # Map internal model names to actual Gemini model IDs (with models/ prefix)
            gemini_model_map = {
                LLMModel.GEMINI_25_PRO: "models/gemini-2.5-pro",
                LLMModel.GEMINI_25_FLASH: "models/gemini-2.5-flash",
                LLMModel.GEMINI_25_FLASH_LITE: "models/gemini-2.5-flash-lite",
            }
            
            # Use the mapped model or default to gemini-2.5-flash
            if model_name and model_name in gemini_model_map:
                model_id = gemini_model_map[model_name]
            else:
                model_id = "models/gemini-2.5-flash"  # Default to flash
            
            model = genai.GenerativeModel(model_id)
            
            # Add JSON instruction to prompt
            json_prompt = prompt + "\n\nRespond only with valid JSON, no markdown or explanation."
            
            response = await asyncio.to_thread(model.generate_content, json_prompt)
            
            # Clean response text
            text = response.text.strip()
            if text.startswith('```json'):
                text = text[7:]
            if text.endswith('```'):
                text = text[:-3]
            
            # Try to repair common JSON issues from Gemini
            text = self._repair_json_string(text)
            
            try:
                return json.loads(text)
            except json.JSONDecodeError as je:
                logger.error(f"Failed to parse Gemini JSON response: {je}")
                logger.debug(f"Raw response: {text[:500]}...")  # Log first 500 chars
                # Return a safe default response
                return {
                    "is_eligible": False,
                    "confidence": 0.0,
                    "eligibility_score": 0.0,
                    "biomarker_score": 0.0,
                    "safety_score": 0.5,
                    "match_reasons": [],
                    "summary": "Unable to parse Gemini response - manual review required",
                    "safety_concerns": ["JSON parsing error from Gemini"],
                    "warnings": ["Gemini response was malformed"]
                }
            
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            raise
    
    def _repair_json_string(self, text: str) -> str:
        """
        Repair common JSON formatting issues from LLM responses.
        Handles trailing commas, missing quotes, etc.
        """
        import re
        
        # Remove trailing commas before } or ]
        text = re.sub(r',\s*([}\]])', r'\1', text)
        
        # Remove any non-JSON text before the first { or [
        match = re.search(r'[\[{]', text)
        if match:
            text = text[match.start():]
        
        # Remove any non-JSON text after the last } or ]
        for i in range(len(text) - 1, -1, -1):
            if text[i] in '}]':
                text = text[:i+1]
                break
        
        # Try to fix single quotes (risky but sometimes necessary)
        # Only do this if normal parsing fails
        try:
            json.loads(text)
            return text
        except:
            # Replace single quotes with double quotes, but preserve escaped quotes
            text = re.sub(r"(?<![\\])'", '"', text)
            
        return text
    
    async def _call_specialized_medical(self, prompt: str, model: LLMModel) -> Dict[str, Any]:
        """Call specialized medical models (TrialGPT, Meditron, etc.)."""
        # In production, these would connect to specialized medical AI services
        # For now, fallback to primary model
        logger.info(f"Specialized model {model.value} not available, using fallback")
        return await self._call_openai(prompt, self.config.fallback_model)
    
    def _select_model_for_complexity(self, patient: Patient, trial: ClinicalTrial) -> LLMModel:
        """Select appropriate model based on case complexity.
        
        Note: Urgency no longer affects model selection. All patients deserve
        the best quality analysis regardless of their clinical urgency.
        High-urgency patients should be prioritized in workflow, not given
        lower-quality analysis.
        """
        if not self.config.cost_optimization or not self.config.enable_model_routing:
            return self.config.primary_model
        
        complexity_score = 0
        
        # Assess complexity
        if len(patient.biomarkers_detected) > 3:
            complexity_score += 2
        if patient.get_line_of_therapy() > 2:
            complexity_score += 2
        if trial.phase and 'Phase 1' in trial.phase.value:
            complexity_score += 1
        if len(trial.eligibility.inclusion_criteria) > 10:
            complexity_score += 1
        if patient.cancer_stage in ['III', 'IV']:
            complexity_score += 1
        
        # Log patient urgency for workflow prioritization (not model selection)
        urgency_indicators = []
        if patient.ecog_status and patient.ecog_status.value >= 3:
            urgency_indicators.append("ECOG≥3")
        if patient.cancer_stage == 'IV':
            urgency_indicators.append("Stage IV")
        
        if urgency_indicators:
            logger.info(f"High-urgency patient ({', '.join(urgency_indicators)}) - prioritize in workflow queue")
        
        logger.debug(f"Complexity score: {complexity_score} for trial {trial.nct_id}")
        
        # Smart model routing based on complexity ONLY
        
        # Special case: Medical models for specific scenarios
        if self.config.enable_medical and trial.phase and 'Phase 1' in trial.phase.value:
            # Early phase trials benefit from specialized medical models
            if self._is_model_available(LLMModel.TRIALGPT):
                return LLMModel.TRIALGPT
        
        # Try to use specialized models if available (will fallback if not)
        if self.config.use_reasoning_models and complexity_score >= 4:
            # Complex medical reasoning needed - GPT-5 is now best for reasoning (Sept 2025)
            candidates = [
                LLMModel.GPT5,  # Best reasoning model (supersedes O3 series)
                LLMModel.CLAUDE_37_SONNET,  # Claude 3.7 for nuanced medical ethics
                LLMModel.O3_PRO,  # O3 as fallback option
                LLMModel.O3,  # O3 as secondary fallback
            ]
            for model in candidates:
                if self._is_model_available(model):
                    return model
            return LLMModel.GPT5_MINI  # Fallback
        
        if complexity_score >= 5:
            # Very complex case: GPT-5 excels at complex reasoning
            candidates = [
                LLMModel.GPT5,  # Best overall reasoning (Sept 2025)
                LLMModel.CLAUDE_37_SONNET,  # Claude 3.7 for medical ethics
                LLMModel.GEMINI_25_PRO,  # Alternative high-quality model
                LLMModel.O3_PRO,  # Fallback reasoning model
            ]
            for model in candidates:
                if self._is_model_available(model):
                    return model
            return LLMModel.GPT5_MINI  # Quality fallback
        elif complexity_score >= 3:
            # Medium complexity: Balance speed and quality
            candidates = [
                LLMModel.GPT5_MINI,
                LLMModel.CLAUDE_37_SONNET,  # Claude 3.7 - balanced
                LLMModel.GEMINI_25_PRO,  # Balanced Gemini
                LLMModel.O3_MINI,
            ]
            for model in candidates:
                if self._is_model_available(model):
                    return model
            return LLMModel.GPT5_NANO  # Balanced fallback
        else:
            # Simple case: Still use good quality models (not just fastest)
            candidates = [
                LLMModel.GPT5_NANO,  # Good quality, efficient
                LLMModel.GPT4O_MINI,  # Multimodal, balanced
                LLMModel.GEMINI_25_FLASH,  # Good balance of speed/quality
                LLMModel.CLAUDE_35_SONNET,  # Previous gen Claude, still good
                LLMModel.GEMINI_25_FLASH_LITE,  # Only as last resort
                self.config.primary_model,
            ]
            for model in candidates:
                if self._is_model_available(model):
                    return model
            return self.config.fallback_model
    
    def _is_model_available(self, model: LLMModel) -> bool:
        """Check if a model is available based on API keys and configuration."""
        # OpenAI models (always available if API key exists)
        if self.openai_client and model in {
            LLMModel.GPT5, LLMModel.GPT5_MINI, LLMModel.GPT5_NANO,
            LLMModel.GPT4O, LLMModel.GPT4O_MINI, LLMModel.GPT41, LLMModel.GPT41_NANO,
            LLMModel.O3, LLMModel.O3_MINI, LLMModel.O3_PRO, LLMModel.O4_MINI,
        }:
            return True
        
        # Claude models (available if Anthropic is configured)
        if self.anthropic_client and self.config.enable_claude and model in {
            LLMModel.CLAUDE_37_SONNET, LLMModel.CLAUDE_35_SONNET, LLMModel.CLAUDE_3_OPUS
        }:
            return True
        
        # Gemini models (available if Google API key exists OR if fallback to OpenAI is possible)
        google_key = os.getenv('GOOGLE_API_KEY')
        if self.config.enable_gemini and model in {
            LLMModel.GEMINI_25_PRO, LLMModel.GEMINI_25_FLASH, LLMModel.GEMINI_25_FLASH_LITE
        }:
            # Check if Google API key is available
            if google_key:
                return True
            # If no Google key but OpenAI is available, we can still say it's "available"
            # and handle the fallback in the actual call
            if self.openai_client:
                logger.debug(f"Gemini model {model.value} requested but no Google API key - will use OpenAI fallback")
                return False  # Actually return False so we skip to next candidate
            return False
        
        # Medical models (only if explicitly enabled and configured)
        if self.config.enable_medical and model == LLMModel.TRIALGPT:
            trialgpt_key = os.getenv('TRIALGPT_API_KEY')
            return trialgpt_key is not None
        
        return False
    
    async def rank_single_trial(
        self, 
        patient: Patient, 
        trial: ClinicalTrial
    ) -> MatchResult:
        """Rank a single trial for a patient."""
        start_time = time.time()
        
        # Select model based on complexity
        model = self._select_model_for_complexity(patient, trial)
        logger.info(f"Using {model.value} for trial {trial.nct_id}")
        
        # Create prompt
        prompt = self._create_eligibility_prompt(patient, trial)
        
        # Try primary model
        result_data = None
        try:
            # OpenAI models (GPT-5, O-series, GPT-4)
            openai_models = [
                LLMModel.GPT5, LLMModel.GPT5_MINI, LLMModel.GPT5_NANO,
                LLMModel.O3, LLMModel.O3_MINI, LLMModel.O3_PRO, LLMModel.O4_MINI,
                LLMModel.GPT4O, LLMModel.GPT4O_MINI, LLMModel.GPT41
            ]
            
            if model in openai_models:
                result_data = await self._call_openai(prompt, model)
            elif model in [LLMModel.CLAUDE_37_SONNET, LLMModel.CLAUDE_35_SONNET, LLMModel.CLAUDE_3_OPUS]:
                result_data = await self._call_anthropic(prompt)
            elif model in [LLMModel.GEMINI_25_PRO, LLMModel.GEMINI_25_FLASH]:
                result_data = await self._call_gemini(prompt, model)
            elif model == LLMModel.TRIALGPT:
                # Try specialized medical model if available
                result_data = await self._call_specialized_medical(prompt, model)
            else:
                # Fallback to primary model
                result_data = await self._call_openai(prompt, self.config.fallback_model)
        except Exception as e:
            logger.warning(f"Primary model failed, trying fallback: {e}")
            # Try fallback model
            try:
                result_data = await self._call_openai(prompt, self.config.fallback_model)
            except Exception as e2:
                logger.error(f"All models failed: {e2}")
                # Return conservative no-match
                return self._create_conservative_no_match(patient, trial)
        
        # Parse result into MatchResult
        try:
            match_reasons = []
            for reason_data in result_data.get('match_reasons', []):
                # Handle None values in matched field
                matched = reason_data.get('matched')
                if matched is None:
                    matched = False
                    
                match_reasons.append(MatchReason(
                    criterion=reason_data.get('criterion', 'Unknown'),
                    matched=bool(matched),  # Ensure it's a boolean
                    explanation=reason_data.get('explanation', ''),
                    confidence=reason_data.get('confidence', 0.5),
                    category=reason_data.get('category', 'unknown')
                ))
            
            # Calculate overall score with weighted components
            eligibility_score = result_data.get('eligibility_score', 0.0)
            biomarker_score = result_data.get('biomarker_score', 0.0)
            safety_score = result_data.get('safety_score', 1.0)
            
            # Weighted average (biomarkers most important for targeted therapy)
            if patient.biomarkers_detected:
                overall_score = (eligibility_score * 0.3 + 
                               biomarker_score * 0.5 + 
                               safety_score * 0.2)
            else:
                overall_score = (eligibility_score * 0.6 + 
                               safety_score * 0.4)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return MatchResult(
                patient_id=patient.patient_id,
                nct_id=trial.nct_id,
                overall_score=overall_score,
                eligibility_score=eligibility_score,
                biomarker_score=biomarker_score,
                geographic_score=1.0,  # To be calculated separately
                is_eligible=result_data.get('is_eligible', False),
                confidence=result_data.get('confidence', 0.5),
                match_reasons=match_reasons,
                summary=result_data.get('summary', 'No summary available'),
                safety_concerns=result_data.get('safety_concerns', []),
                warnings=result_data.get('warnings', []),
                trial_phase=trial.phase,
                llm_model_used=model.value,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Failed to parse LLM result: {e}")
            return self._create_conservative_no_match(patient, trial)
    
    async def rank_multiple_trials(
        self,
        patient: Patient,
        trials: List[ClinicalTrial],
        parallel: bool = True
    ) -> List[MatchResult]:
        """Rank multiple trials for a patient."""
        if parallel:
            # Process trials in parallel with concurrency limit
            from tqdm.asyncio import tqdm as atqdm
            
            # Limit concurrent LLM calls to avoid overwhelming the API
            MAX_CONCURRENT = 5  # Process 5 trials at a time
            semaphore = asyncio.Semaphore(MAX_CONCURRENT)
            
            async def rank_with_semaphore(trial):
                async with semaphore:
                    return await self.rank_single_trial(patient, trial)
            
            # Create all tasks with semaphore
            tasks = [rank_with_semaphore(trial) for trial in trials]
            
            # Process all tasks in parallel with progress tracking
            valid_results = []
            with tqdm(total=len(tasks), desc="Ranking trials", leave=False, 
                      bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} trials') as pbar:
                # Use as_completed for true parallel processing with progress
                for coro in asyncio.as_completed(tasks):
                    try:
                        result = await coro
                        if isinstance(result, MatchResult):
                            valid_results.append(result)
                    except Exception as e:
                        logger.error(f"Trial ranking failed: {e}")
                    pbar.update(1)
            
            return sorted(valid_results, key=lambda x: x.overall_score, reverse=True)
        else:
            # Process sequentially (for rate limiting)
            results = []
            for trial in tqdm(trials, desc="Ranking trials"):
                try:
                    result = await self.rank_single_trial(patient, trial)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to rank trial {trial.nct_id}: {e}")
                    continue
            
            return sorted(results, key=lambda x: x.overall_score, reverse=True)
    
    def _create_conservative_no_match(self, patient: Patient, trial: ClinicalTrial) -> MatchResult:
        """Create a conservative no-match result when LLM fails."""
        return MatchResult(
            patient_id=patient.patient_id,
            nct_id=trial.nct_id,
            overall_score=0.0,
            eligibility_score=0.0,
            biomarker_score=0.0,
            geographic_score=0.0,
            is_eligible=False,
            confidence=0.0,
            match_reasons=[
                MatchReason(
                    criterion="System Error",
                    matched=False,
                    explanation="Unable to assess eligibility due to system error. Manual review required.",
                    confidence=0.0,
                    category="safety"
                )
            ],
            summary="Eligibility could not be determined due to system error. Patient safety requires manual review.",
            safety_concerns=["Automated assessment failed - manual review required"],
            warnings=["System error during eligibility assessment"],
            trial_phase=trial.phase,
            llm_model_used="none",
            processing_time_ms=0
        )
    
    async def explain_ranking(
        self,
        patient: Patient,
        ranked_results: List[MatchResult]
    ) -> str:
        """Generate natural language explanation of ranking."""
        if not ranked_results:
            return "No trials were found to be suitable for this patient."
        
        explanation = f"Trial Matching Results for Patient {patient.patient_id}\n"
        explanation += "=" * 60 + "\n\n"
        
        # Top recommendations
        top_matches = [r for r in ranked_results if r.is_eligible][:3]
        
        if top_matches:
            explanation += "TOP RECOMMENDED TRIALS:\n\n"
            for i, match in enumerate(top_matches, 1):
                explanation += f"{i}. {match.nct_id} (Score: {match.overall_score:.2f}, Confidence: {match.confidence:.2f})\n"
                explanation += f"   {match.summary}\n"
                
                # Highlight key matching factors
                positive_reasons = [r for r in match.match_reasons if r.matched and r.category in ['inclusion', 'biomarker']]
                if positive_reasons:
                    explanation += "   Key Matches:\n"
                    for reason in positive_reasons[:3]:
                        explanation += f"   • {reason.explanation}\n"
                
                if match.warnings:
                    explanation += f"   ⚠ Warnings: {', '.join(match.warnings)}\n"
                
                explanation += "\n"
        else:
            explanation += "No trials met eligibility criteria.\n\n"
        
        # Excluded trials summary
        excluded = [r for r in ranked_results if not r.is_eligible][:3]
        if excluded:
            explanation += "TOP EXCLUDED TRIALS (not eligible):\n\n"
            for match in excluded:
                explanation += f"• {match.nct_id}: {match.get_primary_exclusion_reason()}\n"
        
        return explanation
