"""
Shared test fixtures for OncoMatch AI
"""

import pytest
import pandas as pd
from typing import Dict, List


@pytest.fixture
def sample_patient() -> Dict:
    """Sample patient data for testing"""
    return {
        "name": "Test Patient",
        "age": 55,
        "gender": "Female",
        "cancer_type": "Breast",
        "cancer_stage": "II",
        "biomarkers_detected": "ER+, PR+, HER2-",
        "ecog_status": 1,
        "previous_treatments": "Chemotherapy",
        "city": "New York",
        "state": "NY"
    }


@pytest.fixture
def sample_trials() -> List[Dict]:
    """Sample clinical trials for testing"""
    return [
        {
            "nct_id": "NCT12345678",
            "title": "Phase 3 Study of Novel CDK4/6 Inhibitor",
            "phase": "Phase 3",
            "status": "Recruiting",
            "conditions": ["Breast Cancer"],
            "brief_summary": "Testing a new CDK4/6 inhibitor",
            "eligibility": {
                "min_age": "18 Years",
                "max_age": "N/A",
                "sex": "All"
            }
        },
        {
            "nct_id": "NCT87654321",
            "title": "Immunotherapy for Triple-Negative Breast Cancer",
            "phase": "Phase 2",
            "status": "Recruiting",
            "conditions": ["Triple Negative Breast Cancer"],
            "brief_summary": "Testing immunotherapy combination",
            "eligibility": {
                "min_age": "18 Years",
                "max_age": "75 Years",
                "sex": "All"
            }
        }
    ]


@pytest.fixture
def mock_llm_response() -> Dict:
    """Mock LLM response for testing"""
    return {
        "rankings": [
            {
                "nct_id": "NCT12345678",
                "score": 0.85,
                "rationale": "Strong biomarker match with ER+ status"
            },
            {
                "nct_id": "NCT87654321",
                "score": 0.45,
                "rationale": "Less suitable due to triple-negative requirement"
            }
        ]
    }

