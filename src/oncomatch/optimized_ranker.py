"""
Optimized LLM ranking system with aggressive parallelization and caching.

This module provides a high-performance trial ranking system that can process
40 trials in under 15 seconds through parallelization, batching, and caching.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import pickle
from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from diskcache import Cache

from oncomatch.models import Patient, ClinicalTrial, MatchResult, MatchReason
from oncomatch.model_router import get_router, SmartModelRouter
from oncomatch.llm_providers import OpenAIProvider, AnthropicProvider, GeminiProvider

logger = logging.getLogger(__name__)


@dataclass
class BatchRequest:
    """Container for batch LLM requests."""
    patient: Patient
    trials: List[ClinicalTrial]
    model: str
    request_id: str


@dataclass 
class PerformanceMetrics:
    """Track performance metrics for optimization."""
    total_trials: int = 0
    cache_hits: int = 0
    batch_requests: int = 0
    parallel_tasks: int = 0
    total_time_ms: float = 0
    llm_time_ms: float = 0
    
    @property
    def cache_hit_rate(self) -> float:
        return self.cache_hits / max(self.total_trials, 1)
    
    @property
    def avg_time_per_trial(self) -> float:
        return self.total_time_ms / max(self.total_trials, 1)


class OptimizedLLMRanker:
    """
    High-performance LLM ranking system with:
    - Aggressive parallelization (20+ concurrent requests)
    - Smart batching (5-10 trials per LLM call)
    - Multi-level caching (memory + disk)
    - Connection pooling
    - Optimized prompts
    """
    
    def __init__(
        self,
        max_concurrent_requests: int = 20,  # Much higher concurrency
        batch_size: int = 5,  # Batch multiple trials per request
        cache_ttl_hours: int = 24,
        enable_cache: bool = True,
        enable_batching: bool = True,
        enable_connection_pool: bool = True,
        cache_dir: str = "./cache/llm_results"
    ):
        self.max_concurrent = max_concurrent_requests
        self.batch_size = batch_size
        self.enable_cache = enable_cache
        self.enable_batching = enable_batching
        
        # Initialize router for model selection
        self.router = get_router()
        
        # Initialize providers
        self.providers = {}
        
        # Always initialize OpenAI if key exists
        if os.getenv('OPENAI_API_KEY'):
            from oncomatch.llm_providers import OpenAIProvider
            self.providers['openai'] = OpenAIProvider()
        
        # Initialize Anthropic if key exists
        if os.getenv('ANTHROPIC_API_KEY'):
            from oncomatch.llm_providers import AnthropicProvider
            self.providers['anthropic'] = AnthropicProvider()
        
        # Initialize Google if key exists
        if os.getenv('GOOGLE_API_KEY'):
            from oncomatch.llm_providers import GeminiProvider
            self.providers['google'] = GeminiProvider()
        
        # Ensure we have at least one provider
        if not self.providers:
            raise ValueError("No LLM API keys configured. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY")
        
        # Set up caching
        if self.enable_cache:
            self.cache = Cache(cache_dir, eviction_policy='least-recently-used')
            self.memory_cache = {}  # Fast in-memory cache
            self.cache_ttl_seconds = cache_ttl_hours * 3600
        
        # Connection pooling for faster API calls
        if enable_connection_pool:
            self.http_client = httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=max_concurrent_requests * 2,
                    max_keepalive_connections=max_concurrent_requests
                ),
                timeout=httpx.Timeout(30.0, connect=5.0)
            )
        
        # Performance tracking
        self.metrics = PerformanceMetrics()
        
        # Thread pool for CPU-bound tasks
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Flag to track if providers are warmed up
        self._providers_warmed = False
    
    async def _warm_up_providers(self):
        """Pre-warm provider caches to avoid concurrent probes during batch processing."""
        if self._providers_warmed:
            return
        
        logger.debug("Pre-warming provider model caches...")
        tasks = []
        
        # Probe all providers in parallel (but only once)
        for provider_name, provider in self.providers.items():
            if hasattr(provider, 'available_models'):
                tasks.append(provider.available_models())
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.debug(f"✅ Provider caches warmed for {len(tasks)} providers")
        
        self._providers_warmed = True
    
    def _get_cache_key(self, patient_id: str, trial_id: str, model: str) -> str:
        """Generate cache key for patient-trial-model combination."""
        key_data = f"{patient_id}:{trial_id}:{model}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_batch_cache_key(self, patient_id: str, trial_ids: List[str], model: str) -> str:
        """Generate cache key for batch requests."""
        trial_ids_str = ":".join(sorted(trial_ids))
        key_data = f"batch:{patient_id}:{trial_ids_str}:{model}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def _check_cache(self, cache_key: str) -> Optional[MatchResult]:
        """Check both memory and disk cache."""
        if not self.enable_cache:
            return None
        
        # Check memory cache first (fastest)
        if cache_key in self.memory_cache:
            self.metrics.cache_hits += 1
            return self.memory_cache[cache_key]
        
        # Check disk cache
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            # Validate TTL
            if time.time() - cached_data['timestamp'] < self.cache_ttl_seconds:
                result = cached_data['result']
                # Update memory cache
                self.memory_cache[cache_key] = result
                self.metrics.cache_hits += 1
                return result
        
        return None
    
    async def _save_cache(self, cache_key: str, result: MatchResult):
        """Save to both memory and disk cache."""
        if not self.enable_cache:
            return
        
        # Save to memory cache
        self.memory_cache[cache_key] = result
        
        # Save to disk cache
        self.cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }
    
    def _create_batch_prompt(self, patient: Patient, trials: List[ClinicalTrial]) -> str:
        """Create detailed clinical prompt for accurate matching."""
        # Detailed patient profile
        patient_summary = f"""PATIENT PROFILE:
- Age: {patient.age} years, Gender: {patient.gender}
- Cancer: {patient.cancer_type}, Stage: {patient.cancer_stage}"""
        
        if patient.biomarkers_detected:
            biomarkers = ', '.join([f"{b.name}: {b.status}" for b in patient.biomarkers_detected])
            patient_summary += f"\n- Biomarkers: {biomarkers}"
        
        if patient.ecog_status:
            patient_summary += f"\n- ECOG Status: {patient.ecog_status}"
        
        if patient.previous_treatments:
            therapies = ', '.join(patient.previous_treatments[:3])
            patient_summary += f"\n- Previous Treatments: {therapies}"
        
        patient_summary += f"\n- Location: {patient.city}, {patient.state}"
        
        # Detailed trial information
        trials_text = "\n\nCLINICAL TRIALS TO EVALUATE:"
        for i, trial in enumerate(trials, 1):
            trial_info = f"\n\n{i}. NCT ID: {trial.nct_id}"
            trial_info += f"\n   Title: {trial.title[:150]}"
            if trial.brief_summary:
                trial_info += f"\n   Summary: {trial.brief_summary[:200]}"
            if trial.conditions:
                trial_info += f"\n   Conditions: {', '.join(trial.conditions[:3])}"
            if trial.phase:
                trial_info += f"\n   Phase: {trial.phase.value if hasattr(trial.phase, 'value') else str(trial.phase)}"
            trials_text += trial_info
        
        prompt = f"""{patient_summary}{trials_text}

TASK: As a clinical oncologist, evaluate each trial for this patient.

MATCHING CRITERIA:
1. Cancer type must match (or be related)
2. Stage appropriateness (early stage vs advanced)
3. Biomarker requirements (ER/PR/HER2, mutations, etc.)
4. Prior therapy exclusions
5. ECOG performance status
6. Geographic accessibility

SCORING GUIDELINES:
- 0.9-1.0: Excellent match - all criteria met, biomarkers align, stage appropriate
- 0.7-0.8: Good match - cancer type matches, stage appropriate, some biomarker alignment  
- 0.5-0.6: Moderate match - cancer type matches, stage borderline, biomarkers partial
- 0.3-0.4: Weak match - related cancer or similar stage, but key criteria missing
- 0.0-0.2: No match - wrong cancer type or severe contraindications

CALIBRATION: If cancer type matches and stage is appropriate, minimum score is 0.5.
Expected distribution: 10% excellent, 30% good, 40% moderate, 20% weak/none.

Return ONLY a valid JSON array with this exact format:
[
  {{
    "nct_id": "NCT12345678",
    "is_eligible": true,
    "score": 0.85,
    "confidence": 0.8,
    "reason": "Specific clinical rationale for score (mention key matching criteria)"
  }}
]

Be thorough and clinically accurate. Consider ALL eligibility factors."""
        
        return prompt
    
    def _create_optimized_prompt_v2(self, patient: Patient, trials: List[ClinicalTrial]) -> str:
        """Create an optimized prompt for all LLMs that encourages higher, more realistic scores."""
        # Patient summary
        patient_summary = f"""PATIENT PROFILE:
- Age: {patient.age}, {patient.gender}
- Cancer: {patient.cancer_type}, Stage {patient.cancer_stage}
- ECOG Status: {patient.ecog_status if patient.ecog_status is not None else 'Unknown'}"""
        
        # Add biomarkers if present
        if patient.biomarkers_detected:
            # Convert biomarker objects to strings if needed
            biomarker_strs = []
            for b in patient.biomarkers_detected:
                if hasattr(b, 'name'):
                    biomarker_strs.append(b.name)
                else:
                    biomarker_strs.append(str(b))
            patient_summary += f"\n- Biomarkers: {', '.join(biomarker_strs)}"
        
        # Add previous treatments if present  
        if patient.previous_treatments:
            patient_summary += f"\n- Prior therapies: {', '.join(patient.previous_treatments)}"
        
        patient_summary += f"\n- Location: {patient.city}, {patient.state}"
        
        # Detailed trial information
        trials_text = "\n\nCLINICAL TRIALS TO EVALUATE:"
        for i, trial in enumerate(trials, 1):
            trial_info = f"\n\n{i}. NCT ID: {trial.nct_id}"
            trial_info += f"\n   Title: {trial.title[:150]}"
            if trial.brief_summary:
                trial_info += f"\n   Summary: {trial.brief_summary[:200]}"
            if trial.conditions:
                trial_info += f"\n   Conditions: {', '.join(trial.conditions[:3])}"
            if trial.phase:
                trial_info += f"\n   Phase: {trial.phase.value if hasattr(trial.phase, 'value') else str(trial.phase)}"
            trials_text += trial_info
        
        # Universal optimized prompt that works for all LLMs
        prompt = f"""{patient_summary}{trials_text}

TASK: Evaluate each clinical trial for potential patient benefit. Focus on OPPORTUNITIES rather than barriers.

⚠️ CRITICAL SCORING INSTRUCTION:
You are evaluating trials that have ALREADY been pre-filtered for this cancer type.
Therefore, most trials have inherent relevance. Score them accordingly!

SCORING FRAMEWORK (USE THESE EXACT RANGES):

📍 BASELINE SCORES BY MATCH TYPE:
• Same cancer type → START at 0.70 minimum
• Same cancer + matching stage → START at 0.80 minimum  
• Same cancer + biomarker match → START at 0.85 minimum
• Related/similar cancer → START at 0.50 minimum
• Pan-cancer/basket trial → START at 0.60 minimum

📊 SCORE ADJUSTMENT FACTORS:
• Add +0.10 for each matching biomarker
• Add +0.05 for appropriate treatment line
• Subtract -0.10 ONLY for clear contraindications
• Subtract -0.05 for stage mismatch (but still could work)

🎯 TARGET OUTPUT DISTRIBUTION:
Your scores should approximately follow this distribution:
- 20% of trials: 0.80-1.00 (excellent matches)
- 40% of trials: 0.60-0.79 (good matches)
- 30% of trials: 0.40-0.59 (moderate matches)  
- 10% of trials: <0.40 (poor matches)

EXAMPLES OF PROPER SCORING:
• Breast cancer trial for breast cancer patient → 0.75-0.95
• Lung cancer trial for lung cancer patient → 0.75-0.95
• Solid tumor trial for any cancer patient → 0.50-0.70
• Phase I dose escalation (any cancer) → 0.45-0.65
• Wrong cancer type entirely → 0.10-0.30

Remember: These patients NEED trial options. Be appropriately optimistic while maintaining clinical validity.

Return ONLY valid JSON:
[
  {{
    "nct_id": "NCT12345678",
    "is_eligible": true,
    "score": 0.75,
    "confidence": 0.85,
    "reason": "Breast cancer trial matches patient's diagnosis"
  }}
]

FINAL REMINDER: If in doubt between two scores, ALWAYS choose the higher one."""
        
        return prompt
    
    def _create_gemini_optimized_prompt(self, patient: Patient, trials: List[ClinicalTrial]) -> str:
        """Create an optimized prompt specifically for Gemini that counters its conservative scoring."""
        # Patient summary
        patient_summary = f"""PATIENT PROFILE:
- Age: {patient.age}, {patient.gender}
- Cancer: {patient.cancer_type}, Stage {patient.cancer_stage}
- ECOG Status: {patient.ecog_status if patient.ecog_status is not None else 'Unknown'}"""
        
        # Add biomarkers if present
        if patient.biomarkers_detected:
            # Convert biomarker objects to strings if needed
            biomarker_strs = []
            for b in patient.biomarkers_detected:
                if hasattr(b, 'name'):
                    biomarker_strs.append(b.name)
                else:
                    biomarker_strs.append(str(b))
            patient_summary += f"\n- Biomarkers: {', '.join(biomarker_strs)}"
        
        # Add previous treatments if present  
        if patient.previous_treatments:
            patient_summary += f"\n- Prior therapies: {', '.join(patient.previous_treatments)}"
        
        patient_summary += f"\n- Location: {patient.city}, {patient.state}"
        
        # Detailed trial information
        trials_text = "\n\nCLINICAL TRIALS TO EVALUATE:"
        for i, trial in enumerate(trials, 1):
            trial_info = f"\n\n{i}. NCT ID: {trial.nct_id}"
            trial_info += f"\n   Title: {trial.title[:150]}"
            if trial.brief_summary:
                trial_info += f"\n   Summary: {trial.brief_summary[:200]}"
            if trial.conditions:
                trial_info += f"\n   Conditions: {', '.join(trial.conditions[:3])}"
            if trial.phase:
                trial_info += f"\n   Phase: {trial.phase.value if hasattr(trial.phase, 'value') else str(trial.phase)}"
            trials_text += trial_info
        
        # Gemini-specific optimized prompt with more directive language and examples
        prompt = f"""{patient_summary}{trials_text}

IMPORTANT: You are evaluating clinical trials for potential patient matches. Be appropriately optimistic about matching possibilities while maintaining clinical accuracy.

SCORING CALIBRATION (CRITICAL - READ CAREFULLY):
Most trials for the same cancer type have SOME relevance. Score generously but accurately:

SCORING EXAMPLES WITH RATIONALE:
• Score 0.95: Breast cancer trial for ER+/PR+ patient with matching biomarkers → Perfect alignment
• Score 0.85: Lung cancer trial for NSCLC patient, right stage, most criteria met → Strong match
• Score 0.75: Prostate cancer trial, patient has prostate cancer, stage appropriate → Good match
• Score 0.65: Breast cancer trial, patient has breast cancer but different subtype → Still relevant
• Score 0.55: Cancer trial for similar stage but slightly different biomarkers → Moderate match
• Score 0.45: Related cancer type (e.g., GI cancers) or pan-cancer trial → Possible match
• Score 0.30: Different cancer but similar treatment approach → Weak but notable
• Score 0.10: Completely wrong cancer type with clear contraindications → Poor match

CRITICAL SCORING RULES:
1. If cancer type MATCHES → minimum score = 0.60
2. If cancer type matches AND stage is appropriate → minimum score = 0.70
3. If related/similar cancer → minimum score = 0.40
4. Pan-cancer or basket trials → minimum score = 0.50
5. Only score below 0.30 if CLEARLY wrong cancer type

TARGET DISTRIBUTION (AIM FOR THIS):
- 15% scores ≥ 0.80 (excellent matches)
- 35% scores 0.60-0.79 (good matches)
- 35% scores 0.40-0.59 (moderate matches)
- 15% scores < 0.40 (poor matches)

EVALUATION CRITERIA:
1. Cancer type match (MOST important - heavily weight this)
2. Stage appropriateness (important but flexible)
3. Biomarker alignment (bonus points if matching)
4. Prior therapy considerations
5. Geographic accessibility (minor factor)
6. ECOG status compatibility

Return a JSON array. For EACH trial, think: "How could this potentially help this patient?" 
Be optimistic but clinically sound. Remember: most cancer trials have SOME relevance to cancer patients.

Format (scores should reflect the generous calibration above):
[
  {{
    "nct_id": "NCT12345678",
    "is_eligible": true,
    "score": 0.75,
    "confidence": 0.85,
    "reason": "Clear match: same cancer type, appropriate stage"
  }}
]

REMINDER: Score generously! If unsure between two scores, choose the HIGHER one."""
        
        return prompt
    
    async def _call_llm_batch(
        self,
        patient: Patient,
        trials: List[ClinicalTrial],
        model_name: str,
        provider_name: str
    ) -> List[MatchResult]:
        """Call LLM with batched trials for efficiency, with provider fallback."""
        # Always use optimized prompt for better scoring across all providers
        # Since we're having rate limit issues with Gemini and falling back to OpenAI often
        prompt = self._create_optimized_prompt_v2(patient, trials)
        
        # Define OPTIMIZED fallback order: Gemini → OpenAI → Anthropic
        # Gemini is fastest (887 tok/s), OpenAI is reliable, Anthropic has rate limits
        provider_priority = []
        
        # Start with Gemini (FASTEST, no rate limits)
        if 'google' in self.providers:
            provider_priority.append('google')
        
        # Then OpenAI (reliable, high availability)
        if 'openai' in self.providers:
            provider_priority.append('openai')
        
        # Finally Anthropic (rate limits, use sparingly)
        if 'anthropic' in self.providers:
            provider_priority.append('anthropic')
        
        last_error = None
        response = None
        success = False
        
        for attempt_provider in provider_priority:
            try:
                provider = self.providers[attempt_provider]
                # Select OPTIMAL model for this provider
                if attempt_provider == 'google':
                    # Gemini 2.5 Flash: FASTEST (887 tok/s), no rate limits
                    attempt_model = 'gemini-2.5-flash'
                elif attempt_provider == 'openai':
                    # GPT-4o-mini: Fast, reliable, good availability
                    attempt_model = 'gpt-4o-mini'
                elif attempt_provider == 'anthropic':
                    # Claude 3.7 Sonnet: Use sparingly (rate limits)
                    attempt_model = 'claude-3-7-sonnet-20250219'
                else:
                    attempt_model = model_name
                
                logger.debug(f"Attempting LLM call with {attempt_provider}:{attempt_model}")
                
                # Adjust token limit based on provider and batch size
                # Gemini 2.5 Flash uses internal reasoning tokens (~4000) + output
                if attempt_provider == 'google':
                    max_tokens = 8000  # Gemini needs LOTS: 4k reasoning + 4k output
                else:
                    max_tokens = 2000  # OpenAI/Anthropic are more efficient
                
                response = await provider.generate_json(
                    model=attempt_model,
                    system="You are an expert clinical oncologist specializing in trial matching. Provide thorough, accurate eligibility assessments. Return only valid JSON.",
                    user=prompt,
                    temperature=0.2,
                    max_tokens=max_tokens
                )
                
                # Success! Log if we used a fallback
                if attempt_provider != provider_name:
                    logger.warning(f"✅ Successfully used fallback provider {attempt_provider} (original: {provider_name})")
                
                success = True
                break  # Success, exit retry loop
                
            except Exception as e:
                last_error = e
                error_msg = str(e).lower()
                
                # Check if it's a rate limit error
                if '429' in error_msg or 'rate limit' in error_msg:
                    logger.warning(f"⚠️ Rate limit hit for {attempt_provider}, trying next provider...")
                    continue  # Try next provider
                elif 'unavailable' in error_msg or '503' in error_msg or '502' in error_msg:
                    logger.warning(f"⚠️ Provider {attempt_provider} unavailable, trying next provider...")
                    continue  # Try next provider
                else:
                    # Other error, try next provider
                    logger.error(f"Error with {attempt_provider}: {e}")
                    continue
        
        # If all providers failed, raise the last error
        if not success or response is None:
            logger.error(f"❌ All providers failed for batch LLM call")
            if last_error:
                raise last_error
            else:
                raise Exception("All LLM providers failed")
        
        # Parse batch response (only reached if we got a successful response)
        results = []
        logger.debug(f"Batch response type: {type(response)}")
        
        if isinstance(response, list):
            for item in response:
                try:
                    # Log the item structure for debugging
                    logger.debug(f"Processing batch item keys: {list(item.keys()) if isinstance(item, dict) else 'not a dict'}")
                    
                    nct_id = item.get('nct_id', item.get('trial_id', item.get('id', None)))
                    if not nct_id:
                        logger.warning(f"Response item missing NCT ID: {item}")
                        continue
                    
                    trial = next((t for t in trials if t.nct_id == nct_id), None)
                    if trial:
                        result = self._parse_batch_response_item(patient, trial, item)
                        results.append(result)
                    else:
                        logger.warning(f"No trial found for NCT ID: {nct_id}")
                except KeyError as ke:
                    logger.error(f"KeyError in batch item processing: {ke}, item keys: {list(item.keys()) if isinstance(item, dict) else 'N/A'}")
                    logger.error(f"Full item: {item}")
                except Exception as e:
                    logger.error(f"Error processing batch item: {e}, item: {item}")
        elif isinstance(response, dict):
            # Response might be a dict, try to extract list or handle single item
            logger.debug(f"Response is dict, extracting list from keys")
            # If it's wrapped in a key, try to extract
            if 'results' in response:
                response = response['results']
            elif 'trials' in response:
                response = response['trials']
            elif 'matches' in response:
                response = response['matches']
            else:
                logger.warning(f"Dict response without expected keys: {list(response.keys())}")
            
            # If we got a list now, process it
            if isinstance(response, list):
                for item in response:
                    try:
                        nct_id = item.get('nct_id', item.get('trial_id', item.get('id', None)))
                        if not nct_id:
                            logger.warning(f"Response item missing NCT ID: {item}")
                            continue
                        
                        trial = next((t for t in trials if t.nct_id == nct_id), None)
                        if trial:
                            result = self._parse_batch_response_item(patient, trial, item)
                            results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing batch item from dict: {e}")
        
        # If we got no results, return all trials with default scores instead of nothing
        if not results:
            logger.warning(f"No valid results from batch LLM response, creating default results for {len(trials)} trials")
            for trial in trials:
                results.append(MatchResult(
                    patient_id=patient.patient_id,
                    nct_id=trial.nct_id,
                    is_eligible=False,
                    overall_score=0.2,  # Low but not zero
                    eligibility_score=0.2,
                    biomarker_score=0.2,
                    geographic_score=0.5,
                    confidence=0.3,
                    match_reasons=[MatchReason(
                        criterion="Batch evaluation",
                        matched=False,
                        explanation="Unable to parse detailed evaluation",
                        confidence=0.3,
                        category='inclusion'
                    )],
                    summary="Batch evaluation completed but parsing failed"
                ))
        
        return results
    
    def _parse_batch_response_item(
        self,
        patient: Patient,
        trial: ClinicalTrial,
        response_item: Dict[str, Any]
    ) -> MatchResult:
        """Parse a single trial response from batch with robust error handling."""
        try:
            match_reasons = []
            
            # Extract data with safe defaults
            reason_text = response_item.get('reason', response_item.get('explanation', 'No details provided'))
            is_eligible = response_item.get('is_eligible', response_item.get('eligible', False))
            score = response_item.get('score', response_item.get('match_score', 0.0))
            confidence = response_item.get('confidence', 0.5)
            
            # Create match reason
            match_reasons.append(MatchReason(
                criterion="Eligibility",
                matched=is_eligible,
                explanation=reason_text,
                confidence=confidence,
                category='inclusion'
            ))
            
            return MatchResult(
                patient_id=patient.patient_id,
                nct_id=trial.nct_id,
                is_eligible=is_eligible,
                overall_score=score,
                eligibility_score=score,
                biomarker_score=score * 0.8,  # Slightly lower for biomarkers
                geographic_score=0.5,  # Default geographic score
                confidence=confidence,
                match_reasons=match_reasons,
                summary=reason_text[:200] if len(reason_text) > 200 else reason_text
            )
        except Exception as e:
            logger.error(f"Error parsing response item for {trial.nct_id}: {e}, item: {response_item}")
            # Return a low-confidence default result
            return MatchResult(
                patient_id=patient.patient_id,
                nct_id=trial.nct_id,
                is_eligible=False,
                overall_score=0.1,
                eligibility_score=0.1,
                biomarker_score=0.1,
                geographic_score=0.5,
                confidence=0.2,
                match_reasons=[MatchReason(
                    criterion="Parsing Error",
                    matched=False,
                    explanation=f"Failed to parse LLM response: {str(e)[:100]}",
                    confidence=0.2,
                    category='inclusion'
                )],
                summary="Response parsing failed - low confidence result"
            )
    
    def _create_conservative_no_match(
        self,
        patient: Patient,
        trial: ClinicalTrial
    ) -> MatchResult:
        """Create a conservative no-match result."""
        return MatchResult(
            patient_id=patient.patient_id,
            nct_id=trial.nct_id,
            is_eligible=False,
            overall_score=0.0,
            eligibility_score=0.0,
            biomarker_score=0.0,
            geographic_score=0.0,
            confidence=0.0,
            match_reasons=[],
            summary="Unable to evaluate - conservative no match"
        )
    
    async def rank_trials_optimized(
        self,
        patient: Patient,
        trials: List[ClinicalTrial],
        use_batching: Optional[bool] = None,
        use_cache: Optional[bool] = None
    ) -> List[MatchResult]:
        """
        Rank trials with aggressive optimization.
        
        Target: Process 40 trials in <15 seconds.
        """
        start_time = time.time()
        self.metrics.total_trials = len(trials)
        
        # Pre-warm provider caches ONCE before any parallel processing
        await self._warm_up_providers()
        
        use_batching = self.enable_batching if use_batching is None else use_batching
        use_cache = self.enable_cache if use_cache is None else use_cache
        
        logger.info(f"Processing {len(trials)} trials with optimization "
                   f"(batching={use_batching}, cache={use_cache}, "
                   f"max_concurrent={self.max_concurrent})")
        
        # Step 1: Check cache for all trials
        results = {}
        trials_to_process = []
        
        if use_cache:
            cache_check_start = time.time()
            cache_tasks = []
            
            for trial in trials:
                # Use simple model selection for cache keys
                cache_key = self._get_cache_key(patient.patient_id, trial.nct_id, "cached")
                cache_tasks.append(self._check_cache(cache_key))
            
            cached_results = await asyncio.gather(*cache_tasks)
            
            for trial, cached_result in zip(trials, cached_results):
                if cached_result:
                    results[trial.nct_id] = cached_result
                else:
                    trials_to_process.append(trial)
            
            cache_time = (time.time() - cache_check_start) * 1000
            logger.info(f"Cache check complete in {cache_time:.0f}ms: "
                       f"{len(results)} hits, {len(trials_to_process)} to process")
        else:
            trials_to_process = trials
        
        # Step 2: Process uncached trials
        if trials_to_process:
            if use_batching and len(trials_to_process) > 1:
                # Process in batches
                new_results = await self._process_batched(patient, trials_to_process)
            else:
                # Process in parallel (no batching)
                new_results = await self._process_parallel(patient, trials_to_process)
            
            results.update(new_results)
        
        # Step 3: Score normalization (boost if LLM is too conservative)
        all_results = list(results.values())
        if all_results:
            all_scores = [r.overall_score for r in all_results if r.overall_score > 0]
            if all_scores:
                avg_score = sum(all_scores) / len(all_scores)
                # More aggressive normalization for Gemini since it tends to be very conservative
                # Check if we used Gemini (if most results came from cache or Gemini was primary)
                used_gemini = any('gemini' in str(getattr(r, 'llm_model_used', '')).lower() for r in all_results)
                
                if used_gemini and avg_score < 0.55:  # Higher threshold for Gemini
                    boost_factor = min(1.8, 0.65 / avg_score)  # More aggressive boost for Gemini
                    logger.info(f"Gemini normalization: avg {avg_score:.3f} < 0.55, boosting by {boost_factor:.2f}x")
                    for result in all_results:
                        result.overall_score = min(1.0, result.overall_score * boost_factor)
                        result.eligibility_score = min(1.0, result.eligibility_score * boost_factor)
                        result.biomarker_score = min(1.0, result.biomarker_score * boost_factor)
                elif avg_score < 0.5:  # Standard normalization for other models
                    boost_factor = min(1.5, 0.6 / avg_score)
                    logger.info(f"Standard normalization: avg {avg_score:.3f} < 0.5, boosting by {boost_factor:.2f}x")
                    for result in all_results:
                        result.overall_score = min(1.0, result.overall_score * boost_factor)
                        result.eligibility_score = min(1.0, result.eligibility_score * boost_factor)
                        result.biomarker_score = min(1.0, result.biomarker_score * boost_factor)
        
        # Sort by score
        all_results.sort(key=lambda x: x.overall_score, reverse=True)
        
        # Metrics
        self.metrics.total_time_ms = (time.time() - start_time) * 1000
        
        logger.info(f"✅ Processed {len(trials)} trials in {self.metrics.total_time_ms:.0f}ms "
                   f"(avg {self.metrics.avg_time_per_trial:.0f}ms/trial, "
                   f"cache hit rate: {self.metrics.cache_hit_rate:.1%})")
        
        return all_results
    
    async def _process_batched(
        self,
        patient: Patient,
        trials: List[ClinicalTrial]
    ) -> Dict[str, MatchResult]:
        """Process trials in batches for efficiency."""
        results = {}
        
        # Adjust batch size based on primary provider
        # Gemini 2.5 Flash uses internal reasoning tokens, needs smaller batches
        if 'google' in self.providers:
            effective_batch_size = 3  # Smaller for Gemini (reasoning tokens)
        else:
            effective_batch_size = self.batch_size  # Standard for OpenAI/Anthropic
        
        # Create batches
        batches = []
        for i in range(0, len(trials), effective_batch_size):
            batch = trials[i:i+effective_batch_size]
            batches.append(batch)
        
        logger.info(f"Processing {len(trials)} trials in {len(batches)} batches "
                   f"(size={self.batch_size})")
        
        # Process batches in parallel
        semaphore = asyncio.Semaphore(self.max_concurrent // 2)  # Limit concurrent batches
        
        async def process_batch(batch_trials):
            async with semaphore:
                # Select model for batch - OPTIMIZED for speed and cost
                # Use Gemini 2.5 Flash as primary (FASTEST at 887 tok/s, cheapest)
                if 'google' in self.providers:
                    model_name = 'gemini-2.5-flash'
                    provider_name = 'google'
                elif 'openai' in self.providers:
                    model_name = 'gpt-4o-mini'
                    provider_name = 'openai'
                else:
                    # Fallback to router decision
                    routing_decision = self.router.route(patient, batch_trials[0])
                    model_name = routing_decision.selected_model
                    provider_name = self._get_provider_for_model(model_name)
                
                # Check batch cache
                batch_key = self._get_batch_cache_key(
                    patient.patient_id,
                    [t.nct_id for t in batch_trials],
                    model_name
                )
                
                if self.enable_cache and batch_key in self.memory_cache:
                    self.metrics.cache_hits += len(batch_trials)
                    return self.memory_cache[batch_key]
                
                # Call LLM with batch
                batch_results = await self._call_llm_batch(
                    patient, batch_trials, model_name, provider_name
                )
                
                # Cache results
                if self.enable_cache:
                    self.memory_cache[batch_key] = batch_results
                    for trial, result in zip(batch_trials, batch_results):
                        cache_key = self._get_cache_key(patient.patient_id, trial.nct_id, model_name)
                        await self._save_cache(cache_key, result)
                
                return batch_results
        
        # Execute all batches in parallel
        batch_tasks = [process_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Collect results
        for batch_trials, batch_result in zip(batches, batch_results):
            if isinstance(batch_result, Exception):
                logger.error(f"Batch processing failed: {batch_result}")
                # Fallback to individual processing
                for trial in batch_trials:
                    results[trial.nct_id] = self._create_conservative_no_match(patient, trial)
            else:
                for result in batch_result:
                    results[result.nct_id] = result
        
        self.metrics.batch_requests = len(batches)
        return results
    
    async def _process_provider_batch(
        self,
        patient: Patient,
        batch_trials: List[ClinicalTrial],
        provider_name: str
    ) -> Dict[str, MatchResult]:
        """Process a batch with a specific provider."""
        # Use provider-specific semaphore
        provider_sem = self.provider_semaphores.get(provider_name, self.semaphore)
        
        async with provider_sem:
            # Select optimal model for provider
            if provider_name == 'google':
                model_name = 'gemini-2.5-flash'
            elif provider_name == 'openai':
                model_name = 'gpt-4o-mini'
            elif provider_name == 'anthropic':
                model_name = 'claude-3-7-sonnet-20250219'
            else:
                model_name = 'gpt-4o-mini'
            
            try:
                # Call LLM with batch
                batch_results = await self._call_llm_batch(
                    patient, batch_trials, model_name, provider_name
                )
                
                # Convert to dict and cache
                result_dict = {}
                for result in batch_results:
                    result_dict[result.nct_id] = result
                    if self.enable_cache:
                        cache_key = self._get_cache_key(patient.patient_id, result.nct_id, model_name)
                        # Fire and forget cache save
                        asyncio.create_task(self._save_cache(cache_key, result))
                
                return result_dict
            except Exception as e:
                logger.error(f"Provider batch failed ({provider_name}): {e}")
                return {
                    trial.nct_id: self._create_conservative_no_match(patient, trial)
                    for trial in batch_trials
                }
    
    async def _process_parallel(
        self,
        patient: Patient,
        trials: List[ClinicalTrial]
    ) -> Dict[str, MatchResult]:
        """Process trials in parallel without batching."""
        results = {}
        
        # Use higher concurrency for parallel processing
        semaphore = asyncio.Semaphore(min(self.max_concurrent, len(trials)))
        
        async def process_single_trial(trial):
            async with semaphore:
                # Check cache
                cache_key = self._get_cache_key(patient.patient_id, trial.nct_id, "model")
                cached = await self._check_cache(cache_key)
                if cached:
                    return cached
                
                # Select model
                routing_decision = self.router.route(patient, trial)
                model_name = routing_decision.selected_model
                provider_name = self._get_provider_for_model(model_name)
                
                # Call LLM (simplified for single trial)
                try:
                    provider = self.providers[provider_name]
                    # Use a simplified single-trial prompt
                    prompt = self._create_single_trial_prompt(patient, trial)
                    
                    response = await provider.generate_json(
                        model=model_name,
                        system="Evaluate trial eligibility. Be concise.",
                        user=prompt,
                        temperature=0.1,
                        max_tokens=500
                    )
                    
                    result = self._parse_single_response(patient, trial, response)
                    
                    # Cache result
                    await self._save_cache(cache_key, result)
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Failed to process trial {trial.nct_id}: {e}")
                    return self._create_conservative_no_match(patient, trial)
        
        # Process all trials in parallel
        tasks = [process_single_trial(trial) for trial in trials]
        trial_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for trial, result in zip(trials, trial_results):
            if isinstance(result, Exception):
                logger.error(f"Trial processing failed: {result}")
                results[trial.nct_id] = self._create_conservative_no_match(patient, trial)
            else:
                results[trial.nct_id] = result
        
        self.metrics.parallel_tasks = len(tasks)
        return results
    
    def _create_single_trial_prompt(self, patient: Patient, trial: ClinicalTrial) -> str:
        """Create a minimal prompt for single trial evaluation."""
        return f"""
Patient: {patient.age}yo {patient.cancer_type} Stage {patient.cancer_stage}
Biomarkers: {', '.join([b.name for b in patient.biomarkers_detected[:3]])}
Trial: {trial.nct_id} Phase {trial.phase.value if trial.phase else '?'}
Required biomarkers: {', '.join(trial.eligibility.required_biomarkers[:3]) or 'None'}

Return JSON:
{{"eligible": true/false, "score": 0.0-1.0, "reason": "one sentence"}}
"""
    
    def _parse_single_response(
        self,
        patient: Patient,
        trial: ClinicalTrial,
        response: Dict[str, Any]
    ) -> MatchResult:
        """Parse single trial response."""
        return MatchResult(
            patient_id=patient.patient_id,
            nct_id=trial.nct_id,
            is_eligible=response.get('eligible', False),
            overall_score=response.get('score', 0.0),
            eligibility_score=response.get('score', 0.0),
            biomarker_score=response.get('score', 0.0) * 0.8,
            geographic_score=0.5,
            confidence=0.7,  # Default confidence for speed
            match_reasons=[
                MatchReason(
                    criterion="Evaluation",
                    matched=response.get('eligible', False),
                    explanation=response.get('reason', 'No details'),
                    confidence=0.7,
                    category='inclusion'
                )
            ],
            summary=response.get('reason', 'Evaluated')
        )
    
    def _get_provider_for_model(self, model_name: str) -> str:
        """Determine provider from model name."""
        if 'gpt' in model_name.lower() or 'o1' in model_name.lower() or 'o3' in model_name.lower():
            return 'openai'
        elif 'claude' in model_name.lower():
            return 'anthropic'
        elif 'gemini' in model_name.lower():
            return 'google'
        else:
            return 'openai'  # Default
    
    async def warm_up_connections(self):
        """Pre-warm connection pool for faster first requests."""
        if hasattr(self, 'http_client'):
            # Make dummy requests to establish connections
            tasks = []
            for provider in self.providers.values():
                if hasattr(provider, 'warm_up'):
                    tasks.append(provider.warm_up())
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
                logger.info("Connection pool warmed up")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance metrics."""
        return {
            'total_trials': self.metrics.total_trials,
            'total_time_ms': self.metrics.total_time_ms,
            'avg_time_per_trial_ms': self.metrics.avg_time_per_trial,
            'cache_hit_rate': f"{self.metrics.cache_hit_rate:.1%}",
            'cache_hits': self.metrics.cache_hits,
            'batch_requests': self.metrics.batch_requests,
            'parallel_tasks': self.metrics.parallel_tasks,
            'estimated_cost_savings': f"${self.metrics.cache_hits * 0.001:.2f}"  # Rough estimate
        }
    
    async def close(self):
        """Clean up resources."""
        if hasattr(self, 'http_client'):
            await self.http_client.aclose()
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
        if hasattr(self, 'cache'):
            self.cache.close()


# Global instance for easy access
_ranker_instance = None

def get_optimized_ranker() -> OptimizedLLMRanker:
    """Get or create singleton optimized ranker."""
    global _ranker_instance
    if _ranker_instance is None:
        _ranker_instance = OptimizedLLMRanker(
            max_concurrent_requests=int(os.getenv('MAX_CONCURRENT_LLM', '20')),
            batch_size=int(os.getenv('LLM_BATCH_SIZE', '5')),
            cache_ttl_hours=int(os.getenv('CACHE_TTL_HOURS', '24')),
            enable_cache=os.getenv('ENABLE_LLM_CACHE', 'true').lower() == 'true',
            enable_batching=os.getenv('ENABLE_BATCHING', 'true').lower() == 'true'
        )
    return _ranker_instance
