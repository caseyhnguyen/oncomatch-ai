"""
Tests for the evaluation suite
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from oncomatch.models import Patient, ClinicalTrial, MatchResult, Biomarker, MatchReason
from oncomatch.evaluation.judge_ensemble import JudgeEnsemble, ComplexityAnalyzer
from oncomatch.evaluation.metrics import EvaluationMetrics, AggregateMetrics
from oncomatch.evaluation.synthetic_patients import AdvancedSyntheticPatientGenerator, PatientCategory


class TestComplexityAnalyzer:
    """Test complexity analysis"""
    
    def test_simple_patient(self):
        """Test complexity scoring for simple patient"""
        analyzer = ComplexityAnalyzer()
        
        patient = Patient(
            patient_id="TEST001",
            name="Test Patient",
            age=50,
            gender="Female",
            city="New York",
            state="NY", 
            cancer_type="Breast",
            cancer_stage="I"
        )
        
        trial = ClinicalTrial(
            nct_id="NCT12345",
            title="Test Trial",
            phase="Phase 3",
            status="Recruiting",
            conditions=["Breast Cancer"]
        )
        
        complexity = analyzer.compute_complexity(patient, trial)
        assert complexity < 0.3, "Simple patient should have low complexity"
    
    def test_complex_patient(self):
        """Test complexity scoring for complex patient"""
        analyzer = ComplexityAnalyzer()
        
        patient = Patient(
            patient_id="TEST002",
            name="Complex Patient",
            age=85,
            gender="Female",
            city="New York",
            state="NY",
            cancer_type="Breast", 
            cancer_stage="IV",
            biomarkers_detected=[
                Biomarker(name="ER", status="positive"),
                Biomarker(name="PR", status="positive"),
                Biomarker(name="HER2", status="negative"),
                Biomarker(name="BRCA1", status="positive")
            ],
            previous_treatments=["Chemotherapy", "Radiation", "Surgery"]
        )
        
        trial = ClinicalTrial(
            nct_id="NCT12345",
            title="Phase 1 Trial",
            phase="Phase 1",
            status="Recruiting",
            conditions=["Breast Cancer"]
        )
        
        complexity = analyzer.compute_complexity(patient, trial)
        assert complexity > 0.5, "Complex patient should have high complexity"


class TestEvaluationMetrics:
    """Test evaluation metrics calculations"""
    
    def test_precision_at_k(self):
        """Test Precision@K calculation"""
        rankings = [
            {"nct_id": "NCT001"},
            {"nct_id": "NCT002"},
            {"nct_id": "NCT003"},
            {"nct_id": "NCT004"},
            {"nct_id": "NCT005"}
        ]
        relevant_ids = ["NCT001", "NCT003", "NCT005"]
        
        metrics = EvaluationMetrics()
        precision_at_3 = metrics.precision_at_k(rankings, relevant_ids, k=3)
        assert precision_at_3 == 2/3, f"Expected 0.667, got {precision_at_3}"
        
        precision_at_5 = metrics.precision_at_k(rankings, relevant_ids, k=5)
        assert precision_at_5 == 3/5, f"Expected 0.6, got {precision_at_5}"
    
    def test_recall_at_k(self):
        """Test Recall@K calculation"""
        rankings = [
            {"nct_id": "NCT001"},
            {"nct_id": "NCT002"},
            {"nct_id": "NCT003"}
        ]
        relevant_ids = ["NCT001", "NCT003", "NCT005", "NCT006"]
        
        metrics = EvaluationMetrics()
        recall_at_3 = metrics.recall_at_k(rankings, relevant_ids, k=3)
        assert recall_at_3 == 2/4, f"Expected 0.5, got {recall_at_3}"
    
    def test_mean_reciprocal_rank(self):
        """Test MRR calculation"""
        rankings = [
            {"nct_id": "NCT001"},
            {"nct_id": "NCT002"},
            {"nct_id": "NCT003"}
        ]
        
        metrics = EvaluationMetrics()
        
        # First relevant at position 1
        mrr1 = metrics.mean_reciprocal_rank(rankings, ["NCT001"])
        assert mrr1 == 1.0
        
        # First relevant at position 3
        mrr3 = metrics.mean_reciprocal_rank(rankings, ["NCT003"])
        assert mrr3 == 1/3
        
        # No relevant
        mrr0 = metrics.mean_reciprocal_rank(rankings, ["NCT999"])
        assert mrr0 == 0.0


class TestSyntheticPatients:
    """Test synthetic patient generation"""
    
    def test_generate_standard_patients(self):
        """Test standard patient generation"""
        generator = AdvancedSyntheticPatientGenerator()
        patients = generator.generate_cohort(
            n_patients=10,
            category_distribution={
                PatientCategory.STANDARD: 1.0
            }
        )
        
        assert len(patients) == 10
        for patient in patients:
            assert patient.age >= 18 and patient.age <= 85
            assert patient.cancer_stage in ["I", "II", "III", "IV"]
    
    def test_generate_edge_cases(self):
        """Test edge case patient generation"""
        generator = AdvancedSyntheticPatientGenerator()
        patients = generator.generate_cohort(
            n_patients=5,
            category_distribution={
                PatientCategory.EDGE_CASE: 1.0
            }
        )
        
        assert len(patients) == 5
        # Edge cases should have extreme characteristics
        has_extreme = any(
            p.age < 18 or p.age > 85 or 
            len(p.biomarkers_detected) > 3 or
            p.cancer_type in ["Unknown", "Rare"]
            for p in patients
        )
        assert has_extreme, "Edge cases should have extreme characteristics"


@pytest.mark.asyncio
class TestJudgeEnsemble:
    """Test judge ensemble functionality"""
    
    async def test_judge_initialization(self):
        """Test judge ensemble initialization"""
        ensemble = JudgeEnsemble(enable_complexity_routing=True)
        await ensemble.initialize()
        
        assert len(ensemble.available_judges) > 0
        assert ensemble.complexity_analyzer is not None
    
    async def test_complexity_routing(self):
        """Test dynamic model routing based on complexity"""
        ensemble = JudgeEnsemble(enable_complexity_routing=True)
        await ensemble.initialize()
        
        # Create simple patient and trial
        patient = Patient(
            patient_id="TEST001",
            name="Test Patient",
            age=50,
            gender="Female",
            city="New York",
            state="NY",
            cancer_type="Breast",
            cancer_stage="I"
        )
        
        trial = ClinicalTrial(
            nct_id="NCT12345",
            title="Test Trial",
            phase="Phase 3",
            status="Recruiting",
            conditions=["Breast Cancer"]
        )
        
        match_result = MatchResult(
            patient_id="TEST001",
            nct_id="NCT12345",
            overall_score=0.8,
            eligibility_score=0.85,
            biomarker_score=0.75,
            geographic_score=0.9,
            is_eligible=True,
            confidence=0.8,
            match_reasons=[
                MatchReason(
                    criterion="Age",
                    matched=True,
                    explanation="Patient age meets criteria",
                    confidence=0.9,
                    category="inclusion"
                )
            ],
            summary="Good match"
        )
        
        # Run evaluation
        result = await ensemble.evaluate_match(patient, trial, match_result)
        
        assert "overall_score" in result
        assert "complexity" in result
        assert result["complexity"] < 0.5  # Should be low complexity
        assert "models_used" in result

