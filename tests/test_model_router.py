"""
Tests for the improved model routing system.
"""

import os
# import pytest  # Commented for demo
from unittest.mock import patch
from oncomatch.model_router import (
    SmartModelRouter, 
    ComplexityLevel,
    ComplexityAnalyzer,
    ModelCapability
)
from oncomatch.models import Patient, ClinicalTrial, Biomarker, ECOGStatus


def create_simple_patient():
    """Create a simple patient for testing."""
    from oncomatch.models import Gender
    return Patient(
        patient_id="P001",
        name="Test Patient",
        age=45,
        gender=Gender.FEMALE,
        city="Boston",
        state="MA",
        cancer_type="Breast",
        cancer_stage="II",
        biomarkers_detected=[
            Biomarker(name="ER", status="positive"),
            Biomarker(name="PR", status="positive")
        ],
        ecog_status=ECOGStatus(value=1)
    )


def create_complex_patient():
    """Create a complex patient for testing."""
    from oncomatch.models import Gender
    return Patient(
        patient_id="P002",
        name="Complex Patient",
        age=65,
        gender=Gender.MALE,
        city="New York",
        state="NY",
        cancer_type="Lung",
        cancer_stage="IV",
        biomarkers_detected=[
            Biomarker(name="EGFR", status="positive"),
            Biomarker(name="ALK", status="positive"),
            Biomarker(name="ROS1", status="positive"),
            Biomarker(name="BRAF", status="positive"),
            Biomarker(name="PD-L1", status="positive", value="80%")
        ],
        previous_treatments=["Chemotherapy", "Immunotherapy", "Targeted therapy", "Radiation"],
        ecog_status=ECOGStatus(value=3),
        other_conditions=["Heart disease", "Kidney disease"]
    )


def create_simple_trial():
    """Create a simple trial for testing."""
    from oncomatch.models import EligibilityCriteria, TrialPhase, RecruitmentStatus
    return ClinicalTrial(
        nct_id="NCT12345678",  # Proper format
        title="Simple Breast Cancer Trial",
        phase=TrialPhase.PHASE_3,
        status=RecruitmentStatus.RECRUITING,
        conditions=["Breast Cancer"],
        interventions=["Drug: Paclitaxel"],
        eligibility=EligibilityCriteria(
            inclusion_criteria=["Age >= 18", "Breast cancer"],
            exclusion_criteria=[],
            min_age=18
        )
    )


def create_complex_trial():
    """Create a complex trial for testing."""
    from oncomatch.models import EligibilityCriteria, TrialPhase, RecruitmentStatus
    return ClinicalTrial(
        nct_id="NCT87654321",  # Proper format
        title="Complex Phase 1 Immunotherapy Trial",
        phase=TrialPhase.PHASE_1,
        status=RecruitmentStatus.RECRUITING,
        conditions=["Non-Small Cell Lung Cancer"],
        interventions=["Drug: Pembrolizumab", "Drug: Experimental Agent"],
        eligibility=EligibilityCriteria(
            inclusion_criteria=[
                f"Criterion {i}" for i in range(20)  # Many criteria
            ],
            exclusion_criteria=[],
            required_biomarkers=["EGFR", "ALK", "PD-L1>50%"],
            min_age=18
        )
    )


class TestComplexityAnalyzer:
    """Test the complexity analyzer."""
    
    def test_simple_case(self):
        """Test complexity analysis for simple case."""
        analyzer = ComplexityAnalyzer()
        patient = create_simple_patient()
        trial = create_simple_trial()
        
        level, factors = analyzer.calculate_complexity(patient, trial)
        
        assert level == ComplexityLevel.SIMPLE
        assert factors["biomarker_complexity"] == 1  # Has biomarkers
        assert factors["trial_complexity"] == 0  # Phase 3, few criteria
        assert factors["patient_complexity"] == 0  # Stage II
    
    def test_complex_case(self):
        """Test complexity analysis for complex case."""
        analyzer = ComplexityAnalyzer()
        patient = create_complex_patient()
        trial = create_complex_trial()
        
        level, factors = analyzer.calculate_complexity(patient, trial)
        
        assert level >= ComplexityLevel.VERY_COMPLEX
        assert factors["biomarker_complexity"] >= 3  # Many complex biomarkers
        assert factors["trial_complexity"] >= 3  # Phase 1 + many criteria
        assert factors["patient_complexity"] >= 3  # Stage IV + ECOG 3
        assert factors["safety_complexity"] >= 2  # Comorbidities


class TestSmartModelRouter:
    """Test the smart model router."""
    
    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-key',
        'ANTHROPIC_API_KEY': 'test-key',
        'GOOGLE_API_KEY': 'test-key'
    })
    def test_simple_case_routing(self):
        """Test routing for simple case."""
        router = SmartModelRouter()
        patient = create_simple_patient()
        trial = create_simple_trial()
        
        decision = router.route(patient, trial)
        
        # Should select an efficient model for simple case
        assert decision.complexity_level == ComplexityLevel.SIMPLE
        assert decision.quality_score >= 0.75  # Minimum for simple
        assert decision.estimated_cost <= 0.05  # Within budget
        print(f"\nSimple case routed to: {decision.selected_model}")
        print(f"Reasons: {decision.selection_reasons}")
    
    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-key',
        'ANTHROPIC_API_KEY': 'test-key',
        'GOOGLE_API_KEY': 'test-key'
    })
    def test_complex_case_routing(self):
        """Test routing for complex case."""
        router = SmartModelRouter()
        patient = create_complex_patient()
        trial = create_complex_trial()
        
        decision = router.route(patient, trial)
        
        # Should select a high-quality model for complex case
        assert decision.complexity_level >= ComplexityLevel.VERY_COMPLEX
        assert decision.quality_score >= 0.95  # High quality required
        assert ModelCapability.REASONING in router.registry.models[decision.selected_model].capabilities
        print(f"\nComplex case routed to: {decision.selected_model}")
        print(f"Reasons: {decision.selection_reasons}")
        print(f"Alternatives: {decision.alternative_models}")
    
    @patch.dict(os.environ, {'GOOGLE_API_KEY': 'test-key'})
    def test_limited_providers(self):
        """Test routing with only Google API available."""
        router = SmartModelRouter()
        patient = create_simple_patient()
        trial = create_simple_trial()
        
        decision = router.route(patient, trial)
        
        # Should select a Gemini model
        assert "gemini" in decision.selected_model
        print(f"\nWith only Google API, routed to: {decision.selected_model}")
    
    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-key',
        'ANTHROPIC_API_KEY': 'test-key',
        'GOOGLE_API_KEY': 'test-key'
    })
    def test_override_model(self):
        """Test manual model override."""
        router = SmartModelRouter()
        patient = create_simple_patient()
        trial = create_simple_trial()
        
        decision = router.route(patient, trial, override_model="claude-4-opus")
        
        assert decision.selected_model == "claude-4-opus"
        assert "Manual override" in decision.selection_reasons
    
    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-key',
        'ANTHROPIC_API_KEY': 'test-key',
        'GOOGLE_API_KEY': 'test-key'
    })
    def test_budget_constraints(self):
        """Test routing with tight budget constraints."""
        router = SmartModelRouter(
            budget_per_trial_usd=0.01,  # Very tight budget
            max_latency_ms=5000  # Fast required
        )
        patient = create_complex_patient()
        trial = create_complex_trial()
        
        decision = router.route(patient, trial)
        
        # Should find a model within budget (or relax constraints)
        print(f"\nWith $0.01 budget, routed to: {decision.selected_model}")
        print(f"Estimated cost: ${decision.estimated_cost:.4f}")
        assert decision.selected_model is not None


if __name__ == "__main__":
    # Run some demonstrations
    print("=" * 60)
    print("Model Router Demonstration")
    print("=" * 60)
    
    # Set up environment
    os.environ['OPENAI_API_KEY'] = 'demo-key'
    os.environ['ANTHROPIC_API_KEY'] = 'demo-key'
    os.environ['GOOGLE_API_KEY'] = 'demo-key'
    
    router = SmartModelRouter()
    
    # Demo 1: Simple case
    print("\n📊 SIMPLE CASE")
    simple_patient = create_simple_patient()
    simple_trial = create_simple_trial()
    decision = router.route(simple_patient, simple_trial)
    print(f"Selected: {decision.selected_model}")
    print(f"Quality: {decision.quality_score:.2f}")
    print(f"Cost: ${decision.estimated_cost:.4f}")
    print(f"Latency: {decision.estimated_latency_ms}ms")
    print(f"Reasons: {', '.join(decision.selection_reasons[:2])}")
    
    # Demo 2: Complex case
    print("\n🔬 COMPLEX CASE")
    complex_patient = create_complex_patient()
    complex_trial = create_complex_trial()
    decision = router.route(complex_patient, complex_trial)
    print(f"Selected: {decision.selected_model}")
    print(f"Quality: {decision.quality_score:.2f}")
    print(f"Cost: ${decision.estimated_cost:.4f}")
    print(f"Latency: {decision.estimated_latency_ms}ms")
    print(f"Complexity factors: {', '.join(decision.complexity_factors['details'][:3])}")
    print(f"Reasons: {', '.join(decision.selection_reasons[:2])}")
    
    # Demo 3: Show all available models
    print("\n📋 AVAILABLE MODELS")
    for model_name, profile in router.registry.models.items():
        if router.registry.available_models.get(model_name):
            print(f"  ✓ {model_name:20} - Quality: {profile.quality_score:.2f}, "
                  f"Cost: ${profile.cost_per_1k_tokens:.3f}/1k, "
                  f"Speed: {profile.avg_latency_ms}ms")
