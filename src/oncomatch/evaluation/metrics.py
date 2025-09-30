"""
Consolidated Evaluation Metrics for Clinical Trial Matching
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Core evaluation metrics for clinical trial matching"""
    
    @staticmethod
    def precision_at_k(rankings: List[Dict], relevant_ids: List[str], k: int = 5) -> float:
        """Calculate Precision@K"""
        if not rankings or k <= 0:
            return 0.0
        
        top_k = rankings[:k]
        relevant_in_top_k = sum(1 for r in top_k if r.get('nct_id') in relevant_ids)
        return relevant_in_top_k / k
    
    @staticmethod
    def recall_at_k(rankings: List[Dict], relevant_ids: List[str], k: int = 5) -> float:
        """Calculate Recall@K"""
        if not rankings or not relevant_ids or k <= 0:
            return 0.0
        
        top_k = rankings[:k]
        relevant_in_top_k = sum(1 for r in top_k if r.get('nct_id') in relevant_ids)
        return relevant_in_top_k / len(relevant_ids)
    
    @staticmethod
    def ndcg_at_k(rankings: List[Dict], relevance_scores: Dict[str, float], k: int = 5) -> float:
        """Calculate Normalized Discounted Cumulative Gain@K"""
        if not rankings or k <= 0:
            return 0.0
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(rankings[:k]):
            nct_id = item.get('nct_id')
            relevance = relevance_scores.get(nct_id, 0.0)
            # Use log2(i+2) as denominator (i+2 because i starts at 0)
            dcg += relevance / np.log2(i + 2)
        
        # Calculate ideal DCG
        ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = sum(score / np.log2(i + 2) for i, score in enumerate(ideal_scores))
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def mean_reciprocal_rank(rankings: List[Dict], relevant_ids: List[str]) -> float:
        """Calculate Mean Reciprocal Rank"""
        if not rankings or not relevant_ids:
            return 0.0
        
        for i, item in enumerate(rankings):
            if item.get('nct_id') in relevant_ids:
                return 1.0 / (i + 1)
        
        return 0.0
    
    @staticmethod
    def trial_diversity(rankings: List[Dict]) -> Dict[str, float]:
        """Measure diversity of trial recommendations"""
        if not rankings:
            return {"phase_diversity": 0.0, "mechanism_diversity": 0.0, "geographic_diversity": 0.0}
        
        phases = set()
        mechanisms = set()
        locations = set()
        
        for trial in rankings:
            if 'phase' in trial:
                phases.add(trial['phase'])
            if 'interventions' in trial:
                mechanisms.update(trial['interventions'])
            if 'locations' in trial:
                for loc in trial['locations']:
                    locations.add(loc.get('city', ''))
        
        # Shannon entropy for diversity
        def entropy(items, total):
            if total == 0:
                return 0.0
            proportions = [1/len(items) for _ in items] if items else []
            return -sum(p * np.log2(p) if p > 0 else 0 for p in proportions)
        
        return {
            "phase_diversity": entropy(phases, len(rankings)),
            "mechanism_diversity": entropy(mechanisms, len(rankings)),
            "geographic_diversity": entropy(locations, len(rankings))
        }
    
    @staticmethod
    def equity_metrics(patient_demographics: List[Dict], matched_patients: List[str]) -> Dict[str, float]:
        """Measure equity in matching across demographics"""
        if not patient_demographics:
            return {"coverage": 0.0, "demographic_parity": 0.0}
        
        demographic_groups = defaultdict(list)
        for patient in patient_demographics:
            key = f"{patient.get('race', 'Unknown')}_{patient.get('gender', 'Unknown')}"
            demographic_groups[key].append(patient['patient_id'])
        
        # Calculate coverage per group
        group_coverage = {}
        for group, patients in demographic_groups.items():
            matched_in_group = sum(1 for p in patients if p in matched_patients)
            group_coverage[group] = matched_in_group / len(patients) if patients else 0
        
        # Overall coverage
        total_matched = len(matched_patients)
        total_patients = len(patient_demographics)
        overall_coverage = total_matched / total_patients if total_patients > 0 else 0
        
        # Demographic parity (lower is better - 0 means perfect parity)
        coverages = list(group_coverage.values())
        parity = np.std(coverages) if coverages else 0
        
        return {
            "overall_coverage": overall_coverage,
            "demographic_parity": 1 - parity,  # Convert to higher-is-better
            "group_coverage": group_coverage
        }
    
    @staticmethod
    def safety_metrics(matches: List[Dict]) -> Dict[str, float]:
        """Evaluate safety aspects of matches"""
        if not matches:
            return {"safety_score": 0.0, "confidence_calibration": 0.0}
        
        safety_flags = 0
        total_confidence = 0
        confidence_calibrated = 0
        
        for match in matches:
            # Check for safety concerns
            if 'safety_concerns' in match and match['safety_concerns']:
                safety_flags += len(match['safety_concerns'])
            
            # Check confidence calibration
            if 'confidence' in match and 'actual_outcome' in match:
                confidence = match['confidence']
                outcome = match['actual_outcome']
                total_confidence += confidence
                
                # Check if confidence aligns with outcome
                if (confidence > 0.7 and outcome) or (confidence < 0.3 and not outcome):
                    confidence_calibrated += 1
        
        safety_score = 1 - (safety_flags / (len(matches) * 3))  # Assume max 3 flags per match
        safety_score = max(0, safety_score)
        
        calibration_score = confidence_calibrated / len(matches) if matches else 0
        
        return {
            "safety_score": safety_score,
            "confidence_calibration": calibration_score,
            "total_safety_flags": safety_flags
        }


@dataclass
class PerformanceMetrics:
    """Performance and efficiency metrics"""
    
    @staticmethod
    def calculate_latency_percentiles(latencies: List[float]) -> Dict[str, float]:
        """Calculate latency percentiles"""
        if not latencies:
            return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
        
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)
        
        return {
            "p50": sorted_latencies[int(n * 0.5)],
            "p90": sorted_latencies[int(n * 0.9)],
            "p95": sorted_latencies[int(n * 0.95)],
            "p99": sorted_latencies[int(n * 0.99)] if n > 99 else sorted_latencies[-1]
        }
    
    @staticmethod
    def calculate_throughput(num_requests: int, total_time: float) -> float:
        """Calculate throughput (requests per second)"""
        if total_time <= 0:
            return 0.0
        return num_requests / total_time
    
    @staticmethod
    def calculate_cost_efficiency(total_cost: float, successful_matches: int) -> float:
        """Calculate cost per successful match"""
        if successful_matches <= 0:
            return float('inf')
        return total_cost / successful_matches


class AggregateMetrics:
    """Aggregate and summarize all evaluation metrics"""
    
    def __init__(self):
        self.eval_metrics = EvaluationMetrics()
        self.perf_metrics = PerformanceMetrics()
        
    def calculate_all_metrics(
        self,
        rankings: List[Dict],
        relevant_ids: List[str],
        relevance_scores: Dict[str, float],
        patient_demographics: Optional[List[Dict]] = None,
        matched_patients: Optional[List[str]] = None,
        latencies: Optional[List[float]] = None,
        total_cost: float = 0.0
    ) -> Dict[str, Any]:
        """Calculate all metrics and return comprehensive report"""
        
        metrics = {}
        
        # Ranking metrics
        metrics['ranking'] = {
            'precision_at_5': self.eval_metrics.precision_at_k(rankings, relevant_ids, k=5),
            'precision_at_10': self.eval_metrics.precision_at_k(rankings, relevant_ids, k=10),
            'recall_at_5': self.eval_metrics.recall_at_k(rankings, relevant_ids, k=5),
            'recall_at_10': self.eval_metrics.recall_at_k(rankings, relevant_ids, k=10),
            'ndcg_at_5': self.eval_metrics.ndcg_at_k(rankings, relevance_scores, k=5),
            'ndcg_at_10': self.eval_metrics.ndcg_at_k(rankings, relevance_scores, k=10),
            'mrr': self.eval_metrics.mean_reciprocal_rank(rankings, relevant_ids)
        }
        
        # Diversity metrics
        metrics['diversity'] = self.eval_metrics.trial_diversity(rankings)
        
        # Equity metrics
        if patient_demographics and matched_patients:
            metrics['equity'] = self.eval_metrics.equity_metrics(
                patient_demographics, matched_patients
            )
        
        # Safety metrics
        metrics['safety'] = self.eval_metrics.safety_metrics(rankings)
        
        # Performance metrics
        if latencies:
            metrics['performance'] = {
                'latency_percentiles': self.perf_metrics.calculate_latency_percentiles(latencies),
                'average_latency': np.mean(latencies),
                'throughput': self.perf_metrics.calculate_throughput(
                    len(latencies), sum(latencies)
                )
            }
        
        # Cost metrics
        if total_cost > 0:
            successful_matches = len([r for r in rankings if r.get('is_eligible', False)])
            metrics['cost'] = {
                'total_cost': total_cost,
                'cost_per_match': self.perf_metrics.calculate_cost_efficiency(
                    total_cost, successful_matches
                ),
                'successful_matches': successful_matches
            }
        
        # Overall score (weighted combination)
        overall_score = self._calculate_overall_score(metrics)
        metrics['overall_score'] = overall_score
        metrics['grade'] = self._calculate_grade(overall_score)
        
        return metrics
    
    def _calculate_overall_score(self, metrics: Dict) -> float:
        """Calculate weighted overall score"""
        
        weights = {
            'ranking': 0.35,
            'safety': 0.25,
            'diversity': 0.15,
            'equity': 0.15,
            'performance': 0.10
        }
        
        scores = {
            'ranking': np.mean([
                metrics.get('ranking', {}).get('precision_at_5', 0),
                metrics.get('ranking', {}).get('recall_at_5', 0),
                metrics.get('ranking', {}).get('ndcg_at_5', 0),
                metrics.get('ranking', {}).get('mrr', 0)
            ]),
            'safety': metrics.get('safety', {}).get('safety_score', 0),
            'diversity': np.mean(list(metrics.get('diversity', {}).values())),
            'equity': metrics.get('equity', {}).get('overall_coverage', 0),
            'performance': 1.0  # Placeholder - could be based on latency targets
        }
        
        # Calculate weighted score
        total_score = sum(
            scores.get(component, 0) * weight 
            for component, weight in weights.items()
        )
        
        return min(1.0, max(0.0, total_score))
    
    def _calculate_grade(self, score: float) -> str:
        """Convert score to letter grade"""
        if score >= 0.9:
            return 'A'
        elif score >= 0.8:
            return 'B'
        elif score >= 0.7:
            return 'C'
        elif score >= 0.6:
            return 'D'
        else:
            return 'F'

