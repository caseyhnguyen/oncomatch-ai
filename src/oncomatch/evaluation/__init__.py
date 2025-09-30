"""
OncoMatch Evaluation Suite
Metrics and evaluation tools for clinical trial matching
"""

from .synthetic_patients import SyntheticPatientGenerator, PatientCategory
from .metrics import EvaluationMetrics, AggregateMetrics
from .metrics_core import (
    ClinicalMetrics,
    EquityMetrics, 
    EnsembleMetrics,
    AblationMetrics,
    PerformanceMetrics,
    ErrorAnalysis
)
from .judge_ensemble import JudgeEnsemble
from .evaluator import Evaluator

__all__ = [
    'SyntheticPatientGenerator',
    'PatientCategory',
    'EvaluationMetrics',
    'AggregateMetrics',
    'ClinicalMetrics',
    'EquityMetrics',
    'EnsembleMetrics',
    'AblationMetrics',
    'PerformanceMetrics',
    'ErrorAnalysis',
    'JudgeEnsemble',
    'Evaluator'
]