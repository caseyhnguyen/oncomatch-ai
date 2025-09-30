"""
BioMCP SDK Integration for Clinical Trial Matching.
Fetches real trials from ClinicalTrials.gov with caching and rate limiting.
"""

import asyncio
import json
import logging
import os
import hashlib
import time
import random
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Set, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from biomcp.trials.search import search_trials
    from biomcp.trials import TrialQuery, RecruitingStatus
    BIOMCP_AVAILABLE = True
except ImportError:
    # Fallback if BioMCP not installed
    BIOMCP_AVAILABLE = False
    
    # Define mock fallbacks
    async def search_trials(*args, **kwargs):
        """Mock search function that returns empty list"""
        return []
    
    class TrialQuery:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    
    class RecruitingStatus:
        RECRUITING = "recruiting"
        NOT_RECRUITING = "not_recruiting"
        NOT_YET_RECRUITING = "not_yet_recruiting"
        ENROLLING_BY_INVITATION = "enrolling_by_invitation"

# Import BioMCP SDK wrapper (as per take-home requirements)
try:
    from oncomatch.biomcp_wrapper import BioMCPWrapper
    USE_BIOMCP_SDK = True
except ImportError:
    USE_BIOMCP_SDK = False

# Fallback to direct API if needed (but BioMCP is the requirement)
try:
    from oncomatch.real_trial_fetcher import RealTrialFetcher
    USE_REAL_TRIALS = True
except ImportError:
    USE_REAL_TRIALS = False
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_exponential_jitter,
    retry_if_exception_type,
    before_sleep_log,
    after_log
)
import diskcache
from concurrent.futures import ThreadPoolExecutor

from oncomatch.models import (
    ClinicalTrial, 
    Patient, 
    Location, 
    EligibilityCriteria,
    TrialPhase,
    RecruitmentStatus as ModelRecruitmentStatus
)

# Configure structured logging
from pathlib import Path
log_dir = Path('outputs/logs')
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / 'biomcp_client.log')
    ]
)
logger = logging.getLogger(__name__)

# Log SDK status (moved after logger initialization)
if USE_BIOMCP_SDK:
    logger.info("✅ BioMCP SDK available - will fetch trials via BioMCP (as per requirements)")
elif USE_REAL_TRIALS:
    logger.info("⚠️ Using fallback ClinicalTrials.gov API (BioMCP SDK is preferred)")
else:
    logger.info("⚠️ No real trial fetcher available, will use mock trials")


class TrialDiversityStrategy(Enum):
    """Strategy for ensuring trial diversity."""
    BALANCED_PHASES = "balanced_phases"  # Mix of Phase I/II/III
    GEOGRAPHIC_SPREAD = "geographic_spread"  # Trials from different regions
    INTERVENTION_VARIETY = "intervention_variety"  # Different treatment types
    BIOMARKER_COVERAGE = "biomarker_coverage"  # Cover rare mutations
    EQUITY_FOCUSED = "equity_focused"  # Include underserved populations


@dataclass
class TrialQualityMetrics:
    """Quality metrics for trial assessment."""
    enrollment_rate: float = 0.0  # Historical enrollment speed
    completion_likelihood: float = 0.0  # Probability of completion
    site_quality_score: float = 0.0  # Quality of trial sites
    protocol_complexity: float = 0.0  # Complexity of eligibility
    patient_burden: float = 0.0  # Visit frequency, procedures
    innovation_score: float = 0.0  # Novel mechanisms/approaches
    
    @property
    def overall_quality(self) -> float:
        """Calculate overall quality score."""
        weights = {
            'enrollment': 0.20,
            'completion': 0.15,
            'site_quality': 0.15,
            'protocol': 0.20,
            'burden': 0.15,
            'innovation': 0.15
        }
        
        return (
            weights['enrollment'] * self.enrollment_rate +
            weights['completion'] * self.completion_likelihood +
            weights['site_quality'] * self.site_quality_score +
            weights['protocol'] * (1 - self.protocol_complexity) +
            weights['burden'] * (1 - self.patient_burden) +
            weights['innovation'] * self.innovation_score
        )


@dataclass
class FetchMetrics:
    """Performance metrics for trial fetching."""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    api_errors: int = 0
    rate_limit_waits: int = 0
    avg_response_time_ms: float = 0.0
    total_trials_fetched: int = 0
    unique_trials_fetched: int = 0
    
    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
    
    def update_avg_response_time(self, new_time_ms: float):
        """Update rolling average response time."""
        if self.total_requests == 0:
            self.avg_response_time_ms = new_time_ms
        else:
            alpha = 0.1  # Exponential moving average
            self.avg_response_time_ms = (
                alpha * new_time_ms + 
                (1 - alpha) * self.avg_response_time_ms
            )


class EnhancedRateLimiter:
    """
    Rate limiter with token bucket algorithm and burst support.
    Handles BioMCP's 45 req/min limit with intelligent queueing.
    """
    
    def __init__(
        self, 
        max_requests: int = 45, 
        window_seconds: int = 60,
        burst_size: int = 10,
        enable_priority: bool = True
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.burst_size = burst_size
        self.enable_priority = enable_priority
        
        self.tokens = max_requests
        self.last_refill = time.time()
        self.request_queue = asyncio.Queue()
        self.priority_queue = asyncio.PriorityQueue()
        self.lock = asyncio.Lock()
        self.metrics = FetchMetrics()
    
    async def acquire(self, priority: int = 5):
        """
        Acquire permission to make a request.
        Priority: 1 (highest) to 10 (lowest)
        """
        async with self.lock:
            # Refill tokens
            now = time.time()
            elapsed = now - self.last_refill
            tokens_to_add = (elapsed / self.window_seconds) * self.max_requests
            self.tokens = min(self.max_requests + self.burst_size, 
                            self.tokens + tokens_to_add)
            self.last_refill = now
            
            # Wait if no tokens available
            while self.tokens < 1:
                wait_time = (1 - self.tokens) * (self.window_seconds / self.max_requests)
                logger.info(f"Rate limit: waiting {wait_time:.2f}s (priority={priority})")
                self.metrics.rate_limit_waits += 1
                await asyncio.sleep(wait_time)
                
                # Refill again after wait
                now = time.time()
                elapsed = now - self.last_refill
                tokens_to_add = (elapsed / self.window_seconds) * self.max_requests
                self.tokens = min(self.max_requests + self.burst_size, 
                                self.tokens + tokens_to_add)
                self.last_refill = now
            
            # Consume token
            self.tokens -= 1
            self.metrics.total_requests += 1


class HybridTrialCache:
    """
    Hybrid caching system using both disk and memory for optimal performance.
    Supports evaluation-specific caching strategies.
    """
    
    def __init__(
        self, 
        disk_cache_dir: str = "outputs/cache/biomcp",
        memory_size_mb: int = 512,
        disk_size_gb: int = 10,
        ttl_hours: int = 72
    ):
        # Disk cache for persistence
        self.disk_cache = diskcache.Cache(
            disk_cache_dir,
            size_limit=disk_size_gb * 1024 * 1024 * 1024,
            eviction_policy='least-recently-used'
        )
        
        # Memory cache for speed
        self.memory_cache: Dict[str, Tuple[datetime, Any]] = {}
        self.memory_size_limit = memory_size_mb * 1024 * 1024
        self.ttl = timedelta(hours=ttl_hours)
        
        # Specialized caches for evaluation
        self.trial_quality_cache: Dict[str, TrialQualityMetrics] = {}
        self.diversity_cache: Dict[str, Dict[str, float]] = {}
    
    def _get_key(self, query_params: Dict[str, Any]) -> str:
        """Generate cache key from query parameters."""
        sorted_params = json.dumps(query_params, sort_keys=True)
        return hashlib.sha256(sorted_params.encode()).hexdigest()
    
    def get(self, query_params: Dict[str, Any]) -> Optional[List[Dict]]:
        """Get from cache with two-tier lookup."""
        key = self._get_key(query_params)
        
        # Check memory cache first
        if key in self.memory_cache:
            timestamp, data = self.memory_cache[key]
            if datetime.now() - timestamp < self.ttl:
                logger.debug(f"Memory cache hit: {key[:8]}...")
                return data
            else:
                del self.memory_cache[key]
        
        # Check disk cache
        try:
            disk_entry = self.disk_cache.get(key)
            if disk_entry:
                timestamp, data = disk_entry
                if datetime.now() - timestamp < self.ttl:
                    # Promote to memory cache
                    self.memory_cache[key] = (timestamp, data)
                    logger.debug(f"Disk cache hit: {key[:8]}...")
                    return data
                else:
                    self.disk_cache.delete(key)
        except Exception as e:
            logger.warning(f"Disk cache error: {e}")
        
        return None
    
    def set(self, query_params: Dict[str, Any], data: List[Dict]):
        """Store in both cache tiers."""
        key = self._get_key(query_params)
        timestamp = datetime.now()
        
        # Store in memory
        self.memory_cache[key] = (timestamp, data)
        
        # Store on disk
        try:
            self.disk_cache[key] = (timestamp, data)
        except Exception as e:
            logger.warning(f"Failed to write to disk cache: {e}")
        
        # Clean memory cache if too large
        if len(self.memory_cache) > 1000:  # Simple size check
            self._evict_memory_cache()
    
    def _evict_memory_cache(self):
        """Evict oldest entries from memory cache."""
        sorted_items = sorted(
            self.memory_cache.items(),
            key=lambda x: x[1][0]  # Sort by timestamp
        )
        
        # Keep newest 80%
        keep_count = int(len(sorted_items) * 0.8)
        self.memory_cache = dict(sorted_items[-keep_count:])
    
    def get_trial_quality(self, nct_id: str) -> Optional[TrialQualityMetrics]:
        """Get cached trial quality metrics."""
        return self.trial_quality_cache.get(nct_id)
    
    def set_trial_quality(self, nct_id: str, metrics: TrialQualityMetrics):
        """Cache trial quality metrics."""
        self.trial_quality_cache[nct_id] = metrics
    
    def clear_all(self):
        """Clear all caches."""
        self.memory_cache.clear()
        self.disk_cache.clear()
        self.trial_quality_cache.clear()
        self.diversity_cache.clear()
        logger.info("All caches cleared")


class BioMCPClient:
    """
    BioMCP integration for clinical trial matching.
    Supports patient evaluation with caching and rate limiting.
    """
    
    def __init__(
        self,
        enable_quality_scoring: bool = True,
        enable_diversity_optimization: bool = True,
        enable_trialgpt_mode: bool = False,
        max_workers: int = 10
    ):
        self.rate_limiter = EnhancedRateLimiter()
        self.cache = HybridTrialCache()
        self.enable_quality_scoring = enable_quality_scoring
        self.enable_diversity_optimization = enable_diversity_optimization
        self.enable_trialgpt_mode = enable_trialgpt_mode
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Geographic data
        self._state_coordinates = self._load_state_coordinates()
        self._international_sites = self._load_international_sites()
        
        # Performance monitoring
        self.metrics = FetchMetrics()
        self.health_status = {'healthy': True, 'last_check': datetime.now()}
        
        # Check BioMCP API keys
        self.nci_api_key = os.getenv('NCI_API_KEY')
        self.alphagenome_api_key = os.getenv('ALPHAGENOME_API_KEY')
        self.cbio_token = os.getenv('CBIO_TOKEN')
        
        # Log API key status  
        # NCI API CURRENTLY UNAVAILABLE - Service temporarily down
        if self.nci_api_key:
            logger.warning("⚠️ NCI API key found but service is currently unavailable")
        else:
            logger.info("ℹ️ NCI API currently unavailable - using alternative data sources")
        
        if self.alphagenome_api_key:
            logger.info("✅ AlphaGenome API key configured - variant predictions enabled")
        
        if self.cbio_token:
            logger.info("✅ cBioPortal token configured - enhanced genomics queries enabled")
    
    def _load_state_coordinates(self) -> Dict[str, tuple]:
        """Load comprehensive US state coordinates for geographic filtering."""
        # Extended with major cities for better coverage
        coords = {
            'Alabama': [(32.806671, -86.791130), (33.5186, -86.8104)],  # Montgomery, Birmingham
            'Alaska': [(61.370716, -152.404419), (71.2906, -156.7887)],  # Anchorage, Barrow
            'Arizona': [(33.729759, -111.431221), (32.2226, -110.9747)],  # Phoenix, Tucson
            'California': [(34.0522, -118.2437), (37.7749, -122.4194), (32.7157, -117.1611)],  # LA, SF, SD
            'Colorado': [(39.059811, -105.311104), (38.8339, -104.8214)],  # Denver, Colorado Springs
            'Connecticut': [(41.597782, -72.755371), (41.3083, -72.9279)],  # Hartford, New Haven
            'Florida': [(25.7617, -80.1918), (30.3322, -81.6557), (27.9506, -82.4572)],  # Miami, Jax, Tampa
            'Georgia': [(33.7490, -84.3880), (32.0809, -81.0912)],  # Atlanta, Savannah
            'Illinois': [(41.8781, -87.6298), (39.7817, -89.6501)],  # Chicago, Springfield
            'Massachusetts': [(42.3601, -71.0589), (42.3736, -71.1097)],  # Boston, Cambridge
            'Michigan': [(42.3314, -83.0458), (42.9634, -85.6681)],  # Detroit, Grand Rapids
            'New York': [(40.7128, -74.0060), (42.8864, -78.8784), (43.0481, -76.1474)],  # NYC, Buffalo, Syracuse
            'Texas': [(29.7604, -95.3698), (32.7767, -96.7970), (30.2672, -97.7431)],  # Houston, Dallas, Austin
            'Washington': [(47.6062, -122.3321), (47.6588, -117.4260)],  # Seattle, Spokane
            # Add remaining states...
        }
        
        # Fill in remaining states with single coordinates
        single_coords = {
            'Delaware': (39.318523, -75.507141),
            'Hawaii': (21.094318, -157.498337),
            'Idaho': (44.240459, -114.478828),
            'Indiana': (39.849426, -86.258278),
            'Iowa': (42.011539, -93.210526),
            'Kansas': (38.526600, -96.726486),
            'Kentucky': (37.668140, -84.670067),
            'Louisiana': (31.169546, -91.867805),
            'Maine': (44.693947, -69.381927),
            'Maryland': (39.063946, -76.802101),
            'Minnesota': (45.694454, -93.900192),
            'Mississippi': (32.320513, -90.075913),
            'Missouri': (38.456085, -92.288368),
            'Montana': (46.921925, -110.454353),
            'Nebraska': (41.125370, -98.268082),
            'Nevada': (38.313515, -117.055374),
            'New Hampshire': (43.452492, -71.563896),
            'New Jersey': (40.298904, -74.521011),
            'New Mexico': (34.840515, -106.248482),
            'North Carolina': (35.630066, -79.806419),
            'North Dakota': (47.528912, -99.784012),
            'Ohio': (40.388783, -82.764915),
            'Oklahoma': (35.565342, -96.928917),
            'Oregon': (44.572021, -122.070938),
            'Pennsylvania': (40.590752, -77.209755),
            'Rhode Island': (41.680893, -71.511780),
            'South Carolina': (33.856892, -80.945007),
            'South Dakota': (44.299782, -99.438828),
            'Tennessee': (35.747845, -86.692345),
            'Utah': (40.150032, -111.862434),
            'Vermont': (44.045876, -72.710686),
            'Virginia': (37.769337, -78.169968),
            'West Virginia': (38.491226, -80.954456),
            'Wisconsin': (44.268543, -89.616508),
            'Wyoming': (42.755966, -107.302490)
        }
        
        # Merge and ensure all states have at least one coordinate
        for state, coord in single_coords.items():
            if state not in coords:
                coords[state] = [coord]
        
        return coords
    
    def _load_international_sites(self) -> Dict[str, tuple]:
        """Load international clinical trial sites for equity testing."""
        return {
            'Canada': [(43.6532, -79.3832), (45.5017, -73.5673)],  # Toronto, Montreal
            'Mexico': [(19.4326, -99.1332)],  # Mexico City
            'UK': [(51.5074, -0.1278)],  # London
            'Germany': [(52.5200, 13.4050)],  # Berlin
            'Japan': [(35.6762, 139.6503)],  # Tokyo
            'Australia': [(-33.8688, 151.2093)],  # Sydney
            'Brazil': [(-23.5505, -46.6333)],  # São Paulo
            'India': [(19.0760, 72.8777)],  # Mumbai
        }
    
    def _assess_trial_quality(self, trial: ClinicalTrial) -> TrialQualityMetrics:
        """Assess quality metrics for a clinical trial."""
        metrics = TrialQualityMetrics()
        
        # Enrollment rate based on status and start date
        if trial.status == ModelRecruitmentStatus.RECRUITING:
            metrics.enrollment_rate = 0.7
        elif trial.status == ModelRecruitmentStatus.NOT_YET_RECRUITING:
            metrics.enrollment_rate = 0.5
        else:
            metrics.enrollment_rate = 0.3
        
        # Completion likelihood based on phase
        phase_completion = {
            TrialPhase.PHASE_3: 0.8,
            TrialPhase.PHASE_2: 0.7,
            TrialPhase.PHASE_1: 0.6,
            TrialPhase.EARLY_PHASE_1: 0.4
        }
        metrics.completion_likelihood = phase_completion.get(trial.phase, 0.5)
        
        # Site quality based on number of locations
        if trial.locations:
            metrics.site_quality_score = min(1.0, len(trial.locations) * 0.1)
        else:
            metrics.site_quality_score = 0.3
        
        # Protocol complexity based on eligibility criteria
        if trial.eligibility:
            criteria_count = (
                len(trial.eligibility.inclusion_criteria or []) +
                len(trial.eligibility.exclusion_criteria or [])
            )
            metrics.protocol_complexity = min(1.0, criteria_count * 0.05)
        else:
            metrics.protocol_complexity = 0.5
        
        # Patient burden (simplified)
        metrics.patient_burden = 0.3 if trial.phase == TrialPhase.PHASE_1 else 0.5
        
        # Innovation score based on intervention types
        if trial.interventions:
            novel_keywords = ['car-t', 'gene therapy', 'immunotherapy', 'checkpoint', 
                            'bispecific', 'antibody-drug conjugate', 'cell therapy']
            for intervention in trial.interventions:
                if any(keyword in intervention.lower() for keyword in novel_keywords):
                    metrics.innovation_score = 0.8
                    break
            else:
                metrics.innovation_score = 0.4
        else:
            metrics.innovation_score = 0.3
        
        return metrics
    
    def _calculate_diversity_scores(
        self, 
        trials: List[ClinicalTrial]
    ) -> Dict[str, float]:
        """Calculate diversity metrics for a set of trials."""
        diversity_scores = {}
        
        # Phase diversity
        phase_counts = Counter(trial.phase for trial in trials)
        phase_entropy = -sum(
            (count/len(trials)) * np.log(count/len(trials) + 1e-10)
            for count in phase_counts.values()
        )
        diversity_scores['phase_diversity'] = phase_entropy / np.log(len(TrialPhase))
        
        # Geographic diversity
        unique_states = set()
        for trial in trials:
            if trial.locations:
                for loc in trial.locations:
                    unique_states.add(loc.state)
        diversity_scores['geographic_diversity'] = len(unique_states) / 50  # US states
        
        # Intervention diversity
        intervention_types = set()
        for trial in trials:
            if trial.interventions:
                intervention_types.update(trial.interventions)
        diversity_scores['intervention_diversity'] = min(1.0, len(intervention_types) / 20)
        
        # Sponsor diversity
        sponsors = set(trial.sponsor for trial in trials if trial.sponsor)
        diversity_scores['sponsor_diversity'] = min(1.0, len(sponsors) / 10)
        
        return diversity_scores
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.INFO)
    )
    async def _fetch_trials_with_retry(
        self, 
        query_params: Dict[str, Any],
        priority: int = 5
    ) -> List[Dict]:
        """Fetch trials with retry logic and caching."""
        # Check cache
        cached = self.cache.get(query_params)
        if cached is not None:
            self.metrics.cache_hits += 1
            return cached
        
        self.metrics.cache_misses += 1
        
        # Rate limit
        await self.rate_limiter.acquire(priority=priority)
        
        # Fetch from API
        start_time = time.time()
        
        try:
            # Build TrialQuery object
            trial_query = TrialQuery(
                condition=query_params.get('condition'),
                recruiting_status=query_params.get('recruiting_status'),
                latitude=query_params.get('latitude'),
                longitude=query_params.get('longitude'),
                distance=query_params.get('distance')
            )
            
            # Execute search
            results = await asyncio.to_thread(search_trials, trial_query)
            
            # Update metrics
            elapsed_ms = (time.time() - start_time) * 1000
            self.metrics.update_avg_response_time(elapsed_ms)
            self.metrics.total_trials_fetched += len(results)
            
            # Cache results
            self.cache.set(query_params, results)
            
            return results
            
        except Exception as e:
            self.metrics.api_errors += 1
            logger.error(f"API error: {str(e)}")
            raise
    
    async def fetch_trials_for_patient(
        self,
        patient: Patient,
        max_trials: Optional[int] = None,
        use_cache: bool = True
    ) -> List[ClinicalTrial]:
        """
        Fetch trials for a specific patient based on their characteristics.
        
        Args:
            patient: Patient object with cancer type, biomarkers, etc.
            max_trials: Maximum number of trials to return (None = all available)
            use_cache: Whether to use cached results
            
        Returns:
            List of relevant clinical trials
        """
        # Build search query based on patient
        conditions = [patient.cancer_type] if patient.cancer_type else []
        
        # Add stage-based filtering
        if patient.cancer_stage:
            if patient.cancer_stage in ["I", "II"]:
                conditions.append("early stage")
            elif patient.cancer_stage in ["III", "IV"]:
                conditions.append("advanced")
                conditions.append("metastatic")
        
        # Search for trials
        query = TrialQuery(
            conditions=conditions,
            recruiting_status=RecruitingStatus.RECRUITING,
            max_results=max_trials * 2 if max_trials else 100  # Get more to filter
        )
        
        # Try BioMCP SDK first (as per take-home requirements)
        if USE_BIOMCP_SDK:
            try:
                logger.info(f"Fetching trials from BioMCP SDK for patient {patient.patient_id}")
                wrapper = BioMCPWrapper()
                trials = await wrapper.fetch_trials_for_patient(patient, max_trials=max_trials)
                
                if trials:
                    logger.info(f"✅ Found {len(trials)} trials via BioMCP SDK for patient {patient.patient_id}")
                    return trials
                else:
                    logger.info("No BioMCP trials found, trying fallback methods")
                    
            except Exception as e:
                logger.error(f"Error fetching from BioMCP SDK: {e}")
                logger.info("Trying fallback methods")
        
        # Fallback to direct ClinicalTrials.gov API if BioMCP not available
        elif USE_REAL_TRIALS:
            try:
                logger.info(f"Fetching trials from ClinicalTrials.gov (fallback) for patient {patient.patient_id}")
                async with RealTrialFetcher() as fetcher:
                    trials = await fetcher.search_trials(patient, max_trials=max_trials)
                
                if trials:
                    logger.info(f"✅ Found {len(trials)} trials via ClinicalTrials.gov for patient {patient.patient_id}")
                    return trials
                else:
                    logger.warning("No trials found from ClinicalTrials.gov")
                    return []
                    
            except Exception as e:
                logger.error(f"Error fetching trials: {e}")
                logger.warning("Returning empty list - no mock trials will be used")
                return []
        
        # Execute mock search as fallback
        try:
            trials_data = await search_trials(query)
            
            # Convert to internal model
            trials = []
            for trial_data in trials_data[:max_trials]:
                trial = self._convert_to_clinical_trial(trial_data)
                if trial:
                    trials.append(trial)
            
            logger.info(f"Found {len(trials)} trials for patient {patient.patient_id}")
            
            # If no real trials found, return empty list
            if not trials:
                logger.warning(f"No real trials found for patient {patient.patient_id}")
                return []
            
            return trials
            
        except Exception as e:
            logger.error(f"Error fetching trials: {e}")
            # Return empty list instead of mock trials
            logger.warning("Returning empty list - no mock trials will be used")
            return []
    
    def _generate_mock_trials(self, cancer_type: str, num_trials: int = 10) -> List[ClinicalTrial]:
        """
        DEPRECATED: Mock trials should NOT be used. 
        Always fetch real trials from ClinicalTrials.gov or BioMCP.
        This function is kept for backward compatibility only.
        """
        logger.error("WARNING: Mock trial generation called but should not be used!")
        return []  # Return empty list instead of mock trials
        
        # Common trial titles by cancer type
        trial_templates = {
            "Breast": [
                "Phase III Study of Novel CDK4/6 Inhibitor in ER+ Breast Cancer",
                "Phase II Trial of Immunotherapy Plus Chemotherapy in Triple-Negative Breast Cancer",
                "Phase III Study of HER2-Targeted Therapy in Advanced Breast Cancer",
                "Phase II Investigation of PARP Inhibitor in BRCA-Mutated Breast Cancer",
                "Phase III Combination Therapy for Hormone-Receptor Positive Breast Cancer"
            ],
            "Lung": [
                "Phase III Study of PD-1 Inhibitor in Advanced NSCLC",
                "Phase II Trial of Targeted Therapy for EGFR-Mutated Lung Cancer",
                "Phase III Combination Immunotherapy in Small Cell Lung Cancer",
                "Phase II Study of Novel ALK Inhibitor in ALK-Positive NSCLC",
                "Phase III Trial of Checkpoint Inhibitors in Stage III NSCLC"
            ]
        }
        
        # Get templates for cancer type or use generic
        templates = trial_templates.get(cancer_type, [
            f"Phase III Study in Advanced {cancer_type}",
            f"Phase II Trial of Novel Agent in {cancer_type}",
            f"Phase III Combination Therapy for {cancer_type}"
        ])
        
        for i in range(min(num_trials, len(templates))):
            trial = ClinicalTrial(
                nct_id=f"NCT{random.randint(10000000, 99999999)}",
                title=templates[i % len(templates)],
                phase=random.choice(["Phase 1", "Phase 2", "Phase 3"]),
                status="Recruiting",  # Required field
                conditions=[cancer_type],
                interventions=[f"Drug_{i+1}"],
                sponsor=random.choice(["Pharma Corp", "University Medical Center", "NCI"]),
                eligibility=EligibilityCriteria(
                    inclusion_criteria=["18 years and older", f"Confirmed {cancer_type}"],
                    exclusion_criteria=["Pregnant or nursing", "Severe organ dysfunction"]
                ),
                locations=[
                    Location(
                        name="Medical Center",
                        city="New York",
                        state="NY",
                        country="USA"
                    )
                ],
                recruitment_status=ModelRecruitmentStatus.RECRUITING,
                start_date="2024-01-01",
                completion_date="2026-12-31"
            )
            mock_trials.append(trial)
        
        return mock_trials
    
    async def fetch_trials_for_evaluation_suite(
        self, 
        patients: List[Patient],
        trials_per_patient: int = 20,
        ensure_diversity: bool = True,
        parallel_batch_size: int = 10
    ) -> Dict[str, List[ClinicalTrial]]:
        """
        Optimized fetching for evaluation suite requirements.
        Handles 5000+ patients efficiently with diversity guarantees.
        """
        results = {}
        patient_batches = [
            patients[i:i+parallel_batch_size]
            for i in range(0, len(patients), parallel_batch_size)
        ]
        
        logger.info(f"Processing {len(patients)} patients in {len(patient_batches)} batches")
        
        # Progress bar for batch processing
        for batch_idx, batch in tqdm(enumerate(patient_batches), 
                                     total=len(patient_batches), 
                                     desc="Processing patient batches"):
            batch_start = time.time()
            
            # Process batch in parallel
            tasks = []
            for patient in batch:
                task = self._fetch_diverse_trials_for_patient(
                    patient,
                    max_trials=trials_per_patient,
                    ensure_diversity=ensure_diversity,
                    priority=self._get_patient_priority(patient)
                )
                tasks.append(task)
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Store results
            for patient, trials in zip(batch, batch_results):
                if isinstance(trials, Exception):
                    logger.error(f"Failed to fetch trials for {patient.patient_id}: {trials}")
                    results[patient.patient_id] = []
                else:
                    results[patient.patient_id] = trials
                    self.metrics.unique_trials_fetched += len(set(t.nct_id for t in trials))
            
            # Log batch progress
            batch_elapsed = time.time() - batch_start
            logger.info(
                f"Batch {batch_idx+1}/{len(patient_batches)} completed "
                f"in {batch_elapsed:.2f}s "
                f"(avg {batch_elapsed/len(batch):.2f}s/patient)"
            )
            
            # Check health
            if batch_idx % 10 == 0:
                await self._check_api_health()
        
        # Log final metrics
        self._log_evaluation_metrics(results)
        
        return results
    
    async def _fetch_diverse_trials_for_patient(
        self,
        patient: Patient,
        max_trials: int = 20,
        ensure_diversity: bool = True,
        priority: int = 5
    ) -> List[ClinicalTrial]:
        """Fetch diverse trial portfolio for a single patient."""
        all_trials = []
        seen_nct_ids = set()
        
        # Strategy 1: Cancer type and stage
        cancer_query = self._build_intelligent_cancer_query(patient)
        cancer_trials = await self._fetch_trials_with_retry(cancer_query, priority)
        
        for trial_data in cancer_trials[:max_trials//2]:
            if trial_data['nct_id'] not in seen_nct_ids:
                try:
                    trial = self._parse_trial_data(trial_data)
                    
                    # Quality assessment if enabled
                    if self.enable_quality_scoring:
                        quality = self._assess_trial_quality(trial)
                        if quality.overall_quality < 0.3:
                            continue  # Skip low-quality trials
                        trial.quality_metrics = quality
                    
                    all_trials.append(trial)
                    seen_nct_ids.add(trial.nct_id)
                except Exception as e:
                    logger.debug(f"Parse error: {e}")
        
        # Strategy 2: Biomarker-specific
        if patient.biomarkers_detected:
            biomarker_query = self._build_biomarker_query(patient)
            if biomarker_query:
                biomarker_trials = await self._fetch_trials_with_retry(biomarker_query, priority)
                
                for trial_data in biomarker_trials[:max_trials//3]:
                    if trial_data['nct_id'] not in seen_nct_ids:
                        try:
                            trial = self._parse_trial_data(trial_data)
                            all_trials.append(trial)
                            seen_nct_ids.add(trial.nct_id)
                        except Exception:
                            pass
        
        # Strategy 3: Edge case coverage (if applicable)
        if ensure_diversity and self._is_edge_case_patient(patient):
            edge_query = self._build_edge_case_query(patient)
            edge_trials = await self._fetch_trials_with_retry(edge_query, priority-1)
            
            for trial_data in edge_trials[:max_trials//4]:
                if trial_data['nct_id'] not in seen_nct_ids:
                    try:
                        trial = self._parse_trial_data(trial_data)
                        all_trials.append(trial)
                        seen_nct_ids.add(trial.nct_id)
                    except Exception:
                        pass
        
        # Optimize diversity if enabled
        if ensure_diversity and len(all_trials) > max_trials:
            all_trials = self._optimize_trial_diversity(all_trials, max_trials)
        
        return all_trials[:max_trials]
    
    def _optimize_trial_diversity(
        self, 
        trials: List[ClinicalTrial], 
        target_count: int
    ) -> List[ClinicalTrial]:
        """Select subset of trials optimizing for diversity."""
        if len(trials) <= target_count:
            return trials
        
        # Score each trial
        trial_scores = []
        for trial in trials:
            score = 0.0
            
            # Phase diversity bonus
            phase_weights = {
                TrialPhase.PHASE_1: 1.2,  # Boost early phase
                TrialPhase.PHASE_2: 1.0,
                TrialPhase.PHASE_3: 0.9,
                TrialPhase.PHASE_4: 0.8
            }
            score += phase_weights.get(trial.phase, 1.0)
            
            # Geographic bonus
            if trial.locations:
                unique_states = len(set(loc.state for loc in trial.locations))
                score += min(2.0, unique_states * 0.2)
            
            # Innovation bonus
            if hasattr(trial, 'quality_metrics'):
                score += trial.quality_metrics.innovation_score
            
            trial_scores.append((trial, score))
        
        # Sort by score and select top
        trial_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Ensure phase distribution
        selected = []
        phase_counts = defaultdict(int)
        max_per_phase = target_count // 4  # Roughly balanced
        
        for trial, score in trial_scores:
            if phase_counts[trial.phase] < max_per_phase or len(selected) < target_count - 5:
                selected.append(trial)
                phase_counts[trial.phase] += 1
            
            if len(selected) >= target_count:
                break
        
        return selected
    
    async def fetch_trialgpt_comparison_set(
        self, 
        patient: Patient,
        use_trialgpt_criteria: bool = True
    ) -> List[ClinicalTrial]:
        """
        Fetch trials matching TrialGPT's methodology for fair comparison.
        TrialGPT uses specific search strategies that need to be replicated.
        """
        trials = []
        seen_nct_ids = set()
        
        if use_trialgpt_criteria:
            # TrialGPT-specific query building
            trialgpt_query = {
                'condition': patient.cancer_type,
                'recruiting_status': [
                    RecruitingStatus.RECRUITING,
                    RecruitingStatus.NOT_YET_RECRUITING
                ],
                # TrialGPT focuses on interventional studies
                'study_type': 'Interventional',
                # TrialGPT uses broader geographic search
                'distance': 500  # Wider radius
            }
            
            # Add location if available
            if patient.state in self._state_coordinates:
                coords = self._state_coordinates[patient.state][0]
                trialgpt_query['latitude'] = coords[0]
                trialgpt_query['longitude'] = coords[1]
            
            # Fetch with high priority for evaluation
            raw_trials = await self._fetch_trials_with_retry(trialgpt_query, priority=2)
            
            # TrialGPT typically returns top 10-20 trials
            for trial_data in raw_trials[:20]:
                if trial_data['nct_id'] not in seen_nct_ids:
                    try:
                        trial = self._parse_trial_data(trial_data)
                        
                        # Apply TrialGPT-like filtering
                        if self._passes_trialgpt_filter(trial, patient):
                            trials.append(trial)
                            seen_nct_ids.add(trial.nct_id)
                    except Exception as e:
                        logger.debug(f"TrialGPT parse error: {e}")
        
        return trials[:20]  # TrialGPT typically returns 10-20 trials
    
    def _passes_trialgpt_filter(self, trial: ClinicalTrial, patient: Patient) -> bool:
        """Apply TrialGPT-style filtering rules."""
        # TrialGPT filters
        
        # 1. Must be actively recruiting
        if trial.status not in [
            ModelRecruitmentStatus.RECRUITING,
            ModelRecruitmentStatus.NOT_YET_RECRUITING
        ]:
            return False
        
        # 2. Must match cancer type
        if trial.conditions:
            cancer_match = any(
                patient.cancer_type.lower() in condition.lower()
                for condition in trial.conditions
            )
            if not cancer_match:
                return False
        
        # 3. Appropriate phase for patient
        if patient.previous_treatments:
            # Heavily pretreated -> later phase trials
            if len(patient.previous_treatments) > 3:
                if trial.phase in [TrialPhase.EARLY_PHASE_1, TrialPhase.PHASE_1]:
                    return True  # Phase 1 for heavily pretreated
            else:
                # Less treated -> standard phases
                if trial.phase in [TrialPhase.PHASE_2, TrialPhase.PHASE_3]:
                    return True
        
        return True
    
    def _get_patient_priority(self, patient: Patient) -> int:
        """Determine fetch priority based on patient characteristics."""
        priority = 5  # Default medium priority
        
        # Higher priority for edge cases
        if self._is_edge_case_patient(patient):
            priority = 2
        
        # Higher priority for advanced disease
        elif patient.cancer_stage and 'IV' in patient.cancer_stage:
            priority = 3
        
        # Higher priority for rare biomarkers
        elif patient.biomarkers_detected:
            rare_biomarkers = ['RET', 'NTRK', 'TMB-H', 'HRD', 'POLE']
            if any(bm.name in rare_biomarkers for bm in patient.biomarkers_detected):
                priority = 3
        
        return priority
    
    def _is_edge_case_patient(self, patient: Patient) -> bool:
        """Identify edge case patients needing special handling."""
        # Pediatric
        if patient.age < 18:
            return True
        
        # Elderly
        if patient.age > 80:
            return True
        
        # Poor performance status
        if patient.ecog_status and patient.ecog_status.value >= 3:
            return True
        
        # Heavily pretreated
        if patient.previous_treatments and len(patient.previous_treatments) > 5:
            return True
        
        # Rare cancer types
        rare_cancers = ['mesothelioma', 'glioblastoma', 'cholangiocarcinoma', 
                       'neuroendocrine', 'sarcoma']
        if any(cancer in patient.cancer_type.lower() for cancer in rare_cancers):
            return True
        
        return False
    
    def _build_intelligent_cancer_query(self, patient: Patient) -> Dict[str, Any]:
        """Build intelligent query based on comprehensive patient profile."""
        query = {
            'condition': patient.cancer_type,
            'recruiting_status': [
                RecruitingStatus.RECRUITING,
                RecruitingStatus.NOT_YET_RECRUITING,
                RecruitingStatus.ENROLLING_BY_INVITATION
            ]
        }
        
        # Geographic optimization
        if patient.state in self._state_coordinates:
            coords_list = self._state_coordinates[patient.state]
            # Use closest major city
            lat, lon = coords_list[0]
            query['latitude'] = lat
            query['longitude'] = lon
            
            # Adjust radius based on state population density
            dense_states = ['California', 'New York', 'Texas', 'Florida']
            query['distance'] = 200 if patient.state in dense_states else 400
        
        # Build comprehensive search terms
        search_terms = [patient.cancer_type]
        
        # Stage-specific
        if patient.cancer_stage:
            if 'IV' in patient.cancer_stage:
                search_terms.extend(['metastatic', 'advanced', 'stage 4'])
            elif 'III' in patient.cancer_stage:
                search_terms.extend(['locally advanced', 'stage 3'])
            elif patient.cancer_stage in ['I', 'II']:
                search_terms.extend(['early stage', 'adjuvant'])
        
        # Treatment stage
        if patient.treatment_stage:
            search_terms.append(patient.treatment_stage.lower())
        
        # Line of therapy
        line = patient.get_line_of_therapy()
        if line >= 3:
            search_terms.extend(['refractory', 'resistant', 'salvage'])
        elif line == 2:
            search_terms.append('second line')
        elif line == 1:
            search_terms.append('first line')
        
        query['search_terms'] = ' OR '.join(search_terms)
        return query
    
    def _build_biomarker_query(self, patient: Patient) -> Optional[Dict[str, Any]]:
        """Build biomarker query with targeted therapy matches."""
        if not patient.biomarkers_detected:
            return None
        
        # Extended biomarker to drug mapping (2025)
        biomarker_map = {
            # Breast
            'HER2': ['trastuzumab deruxtecan', 'tucatinib', 'margetuximab', 'HER2 ADC'],
            'ESR1': ['elacestrant', 'selective estrogen receptor degrader', 'SERD'],
            'PIK3CA': ['alpelisib', 'inavolisib', 'PI3K inhibitor'],
            'AKT': ['capivasertib', 'ipatasertib', 'AKT inhibitor'],
            
            # Lung
            'EGFR': ['osimertinib', 'amivantamab', 'lazertinib', 'EGFR-MET bispecific'],
            'KRAS G12C': ['sotorasib', 'adagrasib', 'KRAS G12C inhibitor'],
            'MET': ['capmatinib', 'tepotinib', 'MET inhibitor'],
            'RET': ['selpercatinib', 'pralsetinib', 'RET inhibitor'],
            'NTRK': ['larotrectinib', 'entrectinib', 'TRK inhibitor'],
            
            # Pan-cancer
            'TMB-H': ['pembrolizumab', 'nivolumab', 'immune checkpoint inhibitor'],
            'MSI-H': ['dostarlimab', 'pembrolizumab', 'MMR deficient'],
            'HRD': ['PARP inhibitor', 'niraparib', 'talazoparib'],
            'BRAF V600E': ['encorafenib', 'dabrafenib', 'BRAF MEK combination']
        }
        
        search_terms = []
        for biomarker in patient.biomarkers_detected:
            key = biomarker.name.upper()
            
            # Handle specific mutations
            if 'KRAS' in key and 'G12C' in biomarker.value:
                key = 'KRAS G12C'
            elif 'BRAF' in key and 'V600E' in biomarker.value:
                key = 'BRAF V600E'
            
            if key in biomarker_map:
                search_terms.extend(biomarker_map[key])
            else:
                # Generic biomarker search
                search_terms.append(f"{biomarker.name} positive")
        
        if search_terms:
            return {
                'condition': patient.cancer_type,
                'search_terms': ' OR '.join(search_terms),
                'recruiting_status': [
                    RecruitingStatus.RECRUITING,
                    RecruitingStatus.NOT_YET_RECRUITING
                ]
            }
        
        return None
    
    def _build_edge_case_query(self, patient: Patient) -> Dict[str, Any]:
        """Build query for edge case patients."""
        query = {
            'condition': patient.cancer_type,
            'recruiting_status': [
                RecruitingStatus.RECRUITING,
                RecruitingStatus.NOT_YET_RECRUITING
            ]
        }
        
        search_terms = []
        
        # Pediatric
        if patient.age < 18:
            search_terms.extend(['pediatric', 'childhood', 'adolescent'])
        
        # Elderly
        elif patient.age > 75:
            search_terms.extend(['elderly', 'older adults', 'geriatric'])
        
        # Poor performance
        if patient.ecog_status and patient.ecog_status.value >= 2:
            search_terms.extend(['poor performance', 'ECOG 2-3', 'palliative'])
        
        # Brain metastases
        if any('brain' in str(cond).lower() for cond in (patient.other_conditions or [])):
            search_terms.extend(['brain metastases', 'CNS metastases', 'leptomeningeal'])
        
        if search_terms:
            query['search_terms'] = ' OR '.join(search_terms)
        
        return query
    
    def _parse_trial_data(self, trial_data: Dict) -> ClinicalTrial:
        """Parse raw trial data into ClinicalTrial model."""
        try:
            # Parse eligibility criteria
            eligibility = None
            if 'eligibility' in trial_data:
                elig_data = trial_data['eligibility']
                eligibility = EligibilityCriteria(
                    gender=elig_data.get('gender'),
                    min_age=self._parse_age(elig_data.get('minimum_age')),
                    max_age=self._parse_age(elig_data.get('maximum_age')),
                    inclusion_criteria=self._parse_criteria(
                        elig_data.get('criteria', {}).get('textblock', '')
                    )
                )
            
            # Parse locations
            locations = []
            if 'location' in trial_data:
                for loc_data in trial_data.get('location', []):
                    facility = loc_data.get('facility', {})
                    locations.append(Location(
                        facility_name=facility.get('name'),
                        city=facility.get('address', {}).get('city'),
                        state=facility.get('address', {}).get('state'),
                        country=facility.get('address', {}).get('country'),
                        status=loc_data.get('status')
                    ))
            
            # Map recruitment status
            status_map = {
                'Recruiting': ModelRecruitmentStatus.RECRUITING,
                'Not yet recruiting': ModelRecruitmentStatus.NOT_YET_RECRUITING,
                'Enrolling by invitation': ModelRecruitmentStatus.ENROLLING_BY_INVITATION,
                'Active, not recruiting': ModelRecruitmentStatus.ACTIVE_NOT_RECRUITING,
                'Completed': ModelRecruitmentStatus.COMPLETED
            }
            
            status_str = trial_data.get('overall_status', 'Unknown')
            status = status_map.get(status_str, ModelRecruitmentStatus.RECRUITING)
            
            # Map phase
            phase_map = {
                'Early Phase 1': TrialPhase.EARLY_PHASE_1,
                'Phase 1': TrialPhase.PHASE_1,
                'Phase 1/Phase 2': TrialPhase.PHASE_1_2,
                'Phase 2': TrialPhase.PHASE_2,
                'Phase 2/Phase 3': TrialPhase.PHASE_2_3,
                'Phase 3': TrialPhase.PHASE_3,
                'Phase 4': TrialPhase.PHASE_4
            }
            
            phase_str = trial_data.get('phase', 'N/A')
            phase = phase_map.get(phase_str, TrialPhase.NOT_APPLICABLE)
            
            # Extract conditions and interventions
            conditions = trial_data.get('condition', [])
            if not isinstance(conditions, list):
                conditions = [conditions]
            
            interventions = []
            for interv in trial_data.get('intervention', []):
                interventions.append(interv.get('intervention_name', ''))
            
            return ClinicalTrial(
                nct_id=trial_data['nct_id'],
                title=trial_data.get('brief_title', ''),
                official_title=trial_data.get('official_title'),
                brief_summary=trial_data.get('brief_summary', {}).get('textblock'),
                detailed_description=trial_data.get('detailed_description', {}).get('textblock'),
                phase=phase,
                status=status,
                study_type=trial_data.get('study_type'),
                conditions=conditions,
                interventions=interventions,
                eligibility=eligibility,
                locations=locations,
                sponsor=trial_data.get('sponsors', {}).get('lead_sponsor', {}).get('agency')
            )
            
        except Exception as e:
            logger.error(f"Parse error for trial {trial_data.get('nct_id')}: {e}")
            raise
    
    def _parse_age(self, age_str: str) -> Optional[int]:
        """Parse age string to integer."""
        if not age_str:
            return None
        
        # Handle "18 Years", "65 Years and older", etc.
        import re
        match = re.search(r'(\d+)', age_str)
        if match:
            return int(match.group(1))
        return None
    
    def _parse_criteria(self, criteria_text: str) -> List[str]:
        """Parse eligibility criteria text into list."""
        if not criteria_text:
            return []
        
        # Simple split by common delimiters
        lines = criteria_text.split('\n')
        criteria = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # Remove numbering
                line = re.sub(r'^\d+\.?\s*', '', line)
                if line:
                    criteria.append(line)
        
        return criteria[:20]  # Limit to top 20 criteria
    
    async def _check_api_health(self):
        """Monitor API health and adjust strategy if needed."""
        if self.metrics.api_errors > 10:
            self.health_status['healthy'] = False
            logger.warning(f"API health degraded: {self.metrics.api_errors} errors")
            
            # Implement circuit breaker
            if self.metrics.api_errors > 20:
                logger.error("Circuit breaker activated - pausing requests")
                await asyncio.sleep(30)
                self.metrics.api_errors = 0  # Reset
        
        self.health_status['last_check'] = datetime.now()
    
    def _log_evaluation_metrics(self, results: Dict[str, List[ClinicalTrial]]):
        """Log comprehensive metrics for evaluation suite."""
        total_patients = len(results)
        total_trials = sum(len(trials) for trials in results.values())
        unique_trials = len(set(
            trial.nct_id 
            for trials in results.values() 
            for trial in trials
        ))
        
        # Calculate diversity
        all_trials = [t for trials in results.values() for t in trials]
        if all_trials:
            diversity_scores = self._calculate_diversity_scores(all_trials)
        else:
            diversity_scores = {}
        
        logger.info(f"""
        ========== BioMCP Fetch Summary ==========
        Total Patients: {total_patients}
        Total Trials Fetched: {total_trials}
        Unique Trials: {unique_trials}
        Average Trials/Patient: {total_trials/total_patients:.1f}
        
        Performance Metrics:
        - Cache Hit Rate: {self.metrics.cache_hit_rate:.1%}
        - Avg Response Time: {self.metrics.avg_response_time_ms:.1f}ms
        - API Errors: {self.metrics.api_errors}
        - Rate Limit Waits: {self.metrics.rate_limit_waits}
        
        Diversity Scores:
        - Phase Diversity: {diversity_scores.get('phase_diversity', 0):.2f}
        - Geographic Diversity: {diversity_scores.get('geographic_diversity', 0):.2f}
        - Intervention Diversity: {diversity_scores.get('intervention_diversity', 0):.2f}
        - Sponsor Diversity: {diversity_scores.get('sponsor_diversity', 0):.2f}
        ==========================================
        """)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            'fetch_metrics': {
                'total_requests': self.metrics.total_requests,
                'cache_hit_rate': self.metrics.cache_hit_rate,
                'avg_response_time_ms': self.metrics.avg_response_time_ms,
                'api_errors': self.metrics.api_errors,
                'rate_limit_waits': self.metrics.rate_limit_waits
            },
            'trial_metrics': {
                'total_fetched': self.metrics.total_trials_fetched,
                'unique_fetched': self.metrics.unique_trials_fetched
            },
            'health_status': self.health_status,
            'cache_status': {
                'memory_size': len(self.cache.memory_cache),
                'quality_cache_size': len(self.cache.trial_quality_cache)
            }
        }
    
    def close(self):
        """Clean up resources."""
        self.executor.shutdown(wait=True)
        self.cache.disk_cache.close()
        logger.info("BioMCP client closed")
