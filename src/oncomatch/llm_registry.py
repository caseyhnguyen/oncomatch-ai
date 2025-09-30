"""
LLM Registry with model tags and capability-based, budget-aware routing.
"""

import logging
from typing import Dict, Any, Optional, Set, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ModelCapability(str, Enum):
    """Model capabilities for routing."""
    REASONING = "reasoning"
    SAFETY = "safety"
    MEDICAL = "medical"
    SPEED = "speed"
    COST = "cost"


@dataclass
class ModelProfile:
    """Profile for a model with capabilities and costs."""
    name: str
    provider: str
    capabilities: Set[str]
    speed: str  # "ultra", "fast", "med", "slow"
    cost: str   # "low", "med", "high"
    cost_per_1k_tokens: float
    avg_latency_ms: int
    
    def matches_requirements(
        self,
        want_reasoning: bool = False,
        want_safety: bool = False,
        want_medical: bool = False,
        max_latency_ms: Optional[int] = None,
        max_cost_usd: Optional[float] = None
    ) -> bool:
        """Check if model matches requirements."""
        if want_reasoning and "reasoning" not in self.capabilities:
            return False
        if want_safety and "safety" not in self.capabilities:
            return False
        if want_medical and "medical" not in self.capabilities:
            return False
        if max_latency_ms and self.avg_latency_ms > max_latency_ms:
            return False
        if max_cost_usd and self.cost_per_1k_tokens > max_cost_usd:
            return False
        return True


# Model registry with tags and profiles
MODEL_REGISTRY = {
    # OpenAI o4 series (newest - September 2025)
    "o4-mini": ModelProfile(
        name="o4-mini",
        provider="openai",
        capabilities={"reasoning", "medical"},
        speed="fast",
        cost="low",
        cost_per_1k_tokens=0.006,
        avg_latency_ms=4000
    ),
    
    # OpenAI o3 series (latest reasoning models - September 2025)
    "o3": ModelProfile(
        name="o3",
        provider="openai",
        capabilities={"reasoning", "medical"},
        speed="med",
        cost="high",
        cost_per_1k_tokens=0.025,
        avg_latency_ms=20000
    ),
    "o3-pro": ModelProfile(
        name="o3-pro",
        provider="openai",
        capabilities={"reasoning", "medical"},
        speed="slow",
        cost="high",
        cost_per_1k_tokens=0.035,
        avg_latency_ms=30000
    ),
    "o3-mini": ModelProfile(
        name="o3-mini",
        provider="openai",
        capabilities={"reasoning"},
        speed="fast",
        cost="low",
        cost_per_1k_tokens=0.005,
        avg_latency_ms=5000
    ),
    # Previous generation (o1 series - still available)
    "o1-preview": ModelProfile(
        name="o1-preview",
        provider="openai",
        capabilities={"reasoning"},
        speed="med",
        cost="med",
        cost_per_1k_tokens=0.015,
        avg_latency_ms=25000
    ),
    
    # OpenAI GPT-5 series (September 2025 - Latest)
    "gpt-5": ModelProfile(
        name="gpt-5",
        provider="openai",
        capabilities={"reasoning", "medical"},
        speed="med",
        cost="high",
        cost_per_1k_tokens=0.08,
        avg_latency_ms=15000
    ),
    "gpt-5-mini": ModelProfile(
        name="gpt-5-mini",
        provider="openai",
        capabilities={"reasoning"},
        speed="fast",
        cost="med",
        cost_per_1k_tokens=0.03,
        avg_latency_ms=6000
    ),
    "gpt-5-nano": ModelProfile(
        name="gpt-5-nano",
        provider="openai",
        capabilities={},
        speed="ultra",
        cost="low",
        cost_per_1k_tokens=0.015,
        avg_latency_ms=3000
    ),
    
    # OpenAI GPT-4 series updates
    "gpt-4.1": ModelProfile(
        name="gpt-4.1",
        provider="openai",
        capabilities={"reasoning"},
        speed="fast",
        cost="med",
        cost_per_1k_tokens=0.035,
        avg_latency_ms=6000
    ),
    
    # Anthropic Claude 3.7 series (February 2025 - PRIMARY, newest stable)
    "claude-3-7-sonnet-20250219": ModelProfile(
        name="claude-3-7-sonnet-20250219",
        provider="anthropic",
        capabilities={"safety", "reasoning", "medical"},
        speed="fast",
        cost="med",
        cost_per_1k_tokens=0.03,
        avg_latency_ms=6000
    ),
    # Anthropic Claude 3.5 series (stable, reliable)
    "claude-3-5-sonnet-20241022": ModelProfile(
        name="claude-3-5-sonnet-20241022",
        provider="anthropic",
        capabilities={"safety", "reasoning", "medical"},
        speed="fast",
        cost="med",
        cost_per_1k_tokens=0.015,
        avg_latency_ms=6000
    ),
    # Claude 4.1 series (May 2025 - Latest, PRIMARY)
    "claude-opus-4.1": ModelProfile(
        name="claude-opus-4.1",
        provider="anthropic",
        capabilities={"safety", "reasoning", "medical"},
        speed="med",
        cost="high",
        cost_per_1k_tokens=0.10,
        avg_latency_ms=10000
    ),
    "claude-sonnet-4": ModelProfile(
        name="claude-sonnet-4",
        provider="anthropic",
        capabilities={"safety", "reasoning", "medical"},
        speed="fast",
        cost="med",
        cost_per_1k_tokens=0.05,
        avg_latency_ms=6000
    ),
    # Anthropic Claude 3 Opus (fallback)
    "claude-3-opus-20240229": ModelProfile(
        name="claude-3-opus-20240229",
        provider="anthropic",
        capabilities={"safety", "reasoning", "medical"},
        speed="med",
        cost="high",
        cost_per_1k_tokens=0.075,
        avg_latency_ms=12000
    ),
    # Claude 3.5 (Previous generation)
    "claude-3.5-sonnet": ModelProfile(
        name="claude-3.5-sonnet",
        provider="anthropic",
        capabilities={"safety"},
        speed="fast",
        cost="low",
        cost_per_1k_tokens=0.025,
        avg_latency_ms=4000
    ),
    
    # Google Gemini 2.5 series (September 2025 - Latest)
    "gemini-2.5-pro": ModelProfile(
        name="gemini-2.5-pro",
        provider="gemini",
        capabilities={"reasoning", "medical"},
        speed="fast",
        cost="med",
        cost_per_1k_tokens=0.03,
        avg_latency_ms=5000
    ),
    "gemini-2.5-flash": ModelProfile(
        name="gemini-2.5-flash",
        provider="gemini",
        capabilities={},
        speed="ultra",
        cost="low",
        cost_per_1k_tokens=0.01,
        avg_latency_ms=2000
    ),
    "gemini-2.5-flash-lite": ModelProfile(
        name="gemini-2.5-flash-lite",
        provider="gemini",
        capabilities={},
        speed="ultra",
        cost="low",
        cost_per_1k_tokens=0.005,
        avg_latency_ms=1500
    ),
    
    # Specialized medical models
    "trialgpt": ModelProfile(
        name="trialgpt",
        provider="medical",
        capabilities={"medical", "reasoning"},
        speed="med",
        cost="med",
        cost_per_1k_tokens=0.05,
        avg_latency_ms=10000
    ),
    "trialgpt-pro": ModelProfile(
        name="trialgpt-pro",
        provider="medical",
        capabilities={"medical", "reasoning"},
        speed="med",
        cost="high",
        cost_per_1k_tokens=0.10,
        avg_latency_ms=15000
    ),
    "meditron-70b": ModelProfile(
        name="meditron-70b",
        provider="medical",
        capabilities={"medical"},
        speed="slow",
        cost="med",
        cost_per_1k_tokens=0.04,
        avg_latency_ms=20000
    ),
    
    # Fallback models
    "gpt-4o": ModelProfile(
        name="gpt-4o",
        provider="openai",
        capabilities={"reasoning"},
        speed="fast",
        cost="med",
        cost_per_1k_tokens=0.03,
        avg_latency_ms=5000
    ),
    "gpt-4o-mini": ModelProfile(
        name="gpt-4o-mini",
        provider="openai",
        capabilities={},
        speed="fast",
        cost="low",
        cost_per_1k_tokens=0.015,
        avg_latency_ms=3000
    ),
}


def calculate_complexity(
    patient_data: Dict[str, Any],
    trial_data: Optional[Dict[str, Any]] = None
) -> int:
    """Calculate case complexity (0-10 scale)."""
    complexity = 0
    
    # Biomarker complexity
    biomarkers = patient_data.get("biomarkers_detected", [])
    if len(biomarkers) >= 4:
        complexity += 3
    elif len(biomarkers) >= 2:
        complexity += 2
    elif biomarkers:
        complexity += 1
    
    # Stage complexity
    stage = patient_data.get("cancer_stage", "")
    if stage.startswith("IV"):
        complexity += 2
    elif stage.startswith("III"):
        complexity += 1
    
    # Performance status
    ecog = patient_data.get("ecog_status", {}).get("value", 0) if isinstance(patient_data.get("ecog_status"), dict) else patient_data.get("ecog_status", 0)
    if ecog >= 2:
        complexity += 1
    
    # Trial complexity
    if trial_data:
        phase = trial_data.get("phase", "")
        if "Phase 1" in phase:
            complexity += 2
        elif "Phase 2" in phase:
            complexity += 1
        
        # Criteria length
        criteria_text = ""
        if "eligibility" in trial_data:
            criteria_text = str(trial_data["eligibility"])
        if len(criteria_text) > 2000:
            complexity += 2
        elif len(criteria_text) > 1000:
            complexity += 1
    
    # Treatment history
    prior_treatments = patient_data.get("previous_treatments", [])
    if len(prior_treatments) >= 3:
        complexity += 1
    
    return min(complexity, 10)


def calculate_urgency(patient_data: Dict[str, Any]) -> int:
    """Calculate case urgency (0-5 scale)."""
    urgency = 0
    
    # Performance status
    ecog = patient_data.get("ecog_status", {}).get("value", 0) if isinstance(patient_data.get("ecog_status"), dict) else patient_data.get("ecog_status", 0)
    if ecog >= 3:
        urgency += 2
    elif ecog >= 2:
        urgency += 1
    
    # Stage
    stage = patient_data.get("cancer_stage", "")
    if stage.startswith("IV"):
        urgency += 2
    elif stage.startswith("III"):
        urgency += 1
    
    # Recurrence
    if patient_data.get("is_recurrence"):
        urgency += 1
    
    return min(urgency, 5)


def resolve_best_model(
    *,
    complexity: int,
    urgency: int,
    budget_ms: int = 15000,
    budget_usd: float = 0.05,
    want_safety: bool = False,
    want_medical: bool = False,
    candidates: Optional[Set[str]] = None,
    available_models: Optional[Dict[str, Set[str]]] = None
) -> str:
    """
    Resolve best model based on requirements and constraints.
    
    Note: Urgency is now used for workflow prioritization only, not model selection.
    High-urgency patients deserve the best analysis quality, not faster/cheaper models.
    
    Args:
        complexity: Case complexity (0-10)
        urgency: Case urgency (0-5) - used for logging/metrics only
        budget_ms: Latency budget in milliseconds
        budget_usd: Cost budget in USD per 1000 tokens
        want_safety: Require safety evaluation
        want_medical: Prefer medical-specialized model
        candidates: Set of candidate model names (if None, consider all)
        available_models: Dict of provider -> set of available model names
    
    Returns:
        Best model name
    """
    # Filter to candidates if provided
    if candidates:
        models = {name: profile for name, profile in MODEL_REGISTRY.items() if name in candidates}
    else:
        models = MODEL_REGISTRY.copy()
    
    # Filter by availability if provided
    if available_models:
        available_set = set()
        for provider_models in available_models.values():
            available_set.update(provider_models)
        models = {name: profile for name, profile in models.items() if name in available_set}
    
    if not models:
        logger.warning("No models available, falling back to default")
        return "gpt-4o-mini"  # Ultimate fallback
    
    # Determine requirements based on complexity ONLY (not urgency)
    want_reasoning = complexity >= 6
    max_latency = budget_ms
    max_cost = budget_usd
    
    # Always prefer accuracy over speed - all patients deserve quality analysis
    # Removed urgency-based speed optimization as it was clinically inappropriate
    preferred_speeds = ["med", "fast", "ultra"]  # Balanced approach for all patients
    
    # High complexity: require reasoning (using latest available models)
    if complexity >= 7:
        want_reasoning = True
        # Note: Claude 4.1 models listed but may not be generally available yet (require beta/waitlist)
        preferred_models = ["o3-pro", "o3", "gpt-5", "claude-3-7-sonnet-20250219", "claude-opus-4.1", "claude-sonnet-4"]
    elif complexity >= 4:
        want_reasoning = True
        preferred_models = ["o4-mini", "o3", "gpt-5-mini", "gpt-4.1", "gemini-2.5-pro"]
    else:
        preferred_models = ["o3-mini", "gpt-5-nano", "gemini-2.5-flash", "gpt-4o-mini"]
    
    # Filter by requirements
    suitable_models = []
    for name, profile in models.items():
        if profile.matches_requirements(
            want_reasoning=want_reasoning,
            want_safety=want_safety,
            want_medical=want_medical,
            max_latency_ms=max_latency,
            max_cost_usd=max_cost
        ):
            suitable_models.append((name, profile))
    
    if not suitable_models:
        # Relax requirements
        logger.warning("No models meet all requirements, relaxing constraints")
        for name, profile in models.items():
            if profile.avg_latency_ms <= budget_ms * 2:  # Relax latency
                suitable_models.append((name, profile))
    
    if not suitable_models:
        # Return fastest available
        return min(models.items(), key=lambda x: x[1].avg_latency_ms)[0]
    
    # Score models
    scored_models = []
    for name, profile in suitable_models:
        score = 0
        
        # Preference bonus
        if name in preferred_models:
            score += 10 - preferred_models.index(name)
        
        # Speed bonus (removed urgency multiplier - all patients get same quality)
        if profile.speed in preferred_speeds:
            score += (3 - preferred_speeds.index(profile.speed))  # Small bonus for speed
        
        # Reasoning bonus for complexity
        if want_reasoning and "reasoning" in profile.capabilities:
            score += complexity
        
        # Medical bonus
        if want_medical and "medical" in profile.capabilities:
            score += 5
        
        # Safety bonus
        if want_safety and "safety" in profile.capabilities:
            score += 3
        
        # Cost penalty (lower is better)
        if profile.cost == "low":
            score += 2
        elif profile.cost == "high":
            score -= 2
        
        scored_models.append((name, score))
    
    # Sort by score (highest first)
    scored_models.sort(key=lambda x: x[1], reverse=True)
    
    selected = scored_models[0][0]
    logger.info(f"Selected model: {selected} (complexity={complexity}, urgency={urgency}, safety={want_safety})")
    
    # Log high urgency cases for workflow prioritization
    if urgency >= 4:
        logger.warning(f"High urgency patient (urgency={urgency}) - consider expedited review")
    
    return selected


def handle_urgency_workflow(
    patient_data: Dict[str, Any],
    urgency: int,
    match_results: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Handle urgency-based workflow actions (not model selection).
    
    High-urgency patients get workflow prioritization, not lower-quality analysis.
    
    Args:
        patient_data: Patient information
        urgency: Calculated urgency score (0-5)
        match_results: Optional trial matching results
    
    Returns:
        Workflow actions and recommendations
    """
    actions = {
        "priority_level": "standard",
        "review_timeline": "standard",
        "notifications": [],
        "recommendations": []
    }
    
    if urgency >= 4:
        # Critical urgency - Stage IV with poor ECOG
        actions["priority_level"] = "critical"
        actions["review_timeline"] = "immediate"
        actions["notifications"].append("care_team")
        actions["notifications"].append("trial_coordinator")
        actions["recommendations"].append("Schedule urgent consultation")
        actions["recommendations"].append("Consider compassionate use programs")
        
    elif urgency >= 3:
        # High urgency
        actions["priority_level"] = "high"
        actions["review_timeline"] = "expedited"
        actions["notifications"].append("trial_coordinator")
        actions["recommendations"].append("Expedite trial screening")
        
    elif urgency >= 2:
        # Moderate urgency
        actions["priority_level"] = "moderate"
        actions["review_timeline"] = "priority"
    
    # Add specific recommendations based on patient factors
    ecog = patient_data.get("ecog_status", {}).get("value", 0)
    if ecog >= 3:
        actions["recommendations"].append("Consider trials with flexible performance status criteria")
        actions["recommendations"].append("Evaluate for supportive care trials")
    
    stage = patient_data.get("cancer_stage", "")
    if stage.startswith("IV"):
        actions["recommendations"].append("Prioritize trials with rapid enrollment")
        actions["recommendations"].append("Consider expanded access programs")
    
    return actions


def get_model_config(model_name: str) -> Optional[ModelProfile]:
    """Get model configuration."""
    return MODEL_REGISTRY.get(model_name)


def estimate_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int
) -> float:
    """Estimate cost for model usage."""
    profile = MODEL_REGISTRY.get(model_name)
    if not profile:
        return 0.0
    
    total_tokens = input_tokens + output_tokens
    return (total_tokens / 1000) * profile.cost_per_1k_tokens


def get_models_by_capability(
    capability: str,
    available_models: Optional[Dict[str, Set[str]]] = None
) -> List[str]:
    """Get models with specific capability."""
    models = []
    
    for name, profile in MODEL_REGISTRY.items():
        if capability in profile.capabilities:
            # Check availability if provided
            if available_models:
                available_set = set()
                for provider_models in available_models.values():
                    available_set.update(provider_models)
                if name not in available_set:
                    continue
            
            models.append(name)
    
    # Sort by cost (cheapest first)
    models.sort(key=lambda x: MODEL_REGISTRY[x].cost_per_1k_tokens)
    
    return models

