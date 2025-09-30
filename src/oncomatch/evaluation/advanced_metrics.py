"""
Evaluation Metrics for Clinical Trial Matching
Implements metrics for medical AI evaluation
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import json
import logging
from scipy import stats
from sklearn.metrics import cohen_kappa_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class ClinicalMetrics:
    """
    Clinical trial matching metrics for evaluation.
    
    These metrics are designed to meet clinical evaluation standards
    and ML performance requirements.
    """
    
    @staticmethod
    def ndcg_at_k(rankings: List[Dict], relevance_scores: Dict[str, float], k: int = 5) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at K.
        
        NDCG is crucial for clinical trials as it considers graded relevance -
        some trials are better matches than others.
        
        Args:
            rankings: List of ranked trials with nct_id
            relevance_scores: Dict mapping nct_id to relevance score (0-1)
            k: Number of top results to consider
            
        Returns:
            NDCG@k score between 0 and 1
        """
        if not rankings or k <= 0:
            return 0.0
        
        # Calculate DCG (Discounted Cumulative Gain)
        dcg = 0.0
        for i, item in enumerate(rankings[:k]):
            nct_id = item.get('nct_id')
            relevance = relevance_scores.get(nct_id, 0.0)
            # Use log2(i+2) as per standard NDCG formula
            dcg += (2**relevance - 1) / np.log2(i + 2)
        
        # Calculate ideal DCG (perfect ranking)
        ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = sum((2**score - 1) / np.log2(i + 2) for i, score in enumerate(ideal_scores))
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def precision_recall_f1_at_k(
        rankings: List[Dict], 
        relevant_ids: Set[str], 
        k: int = 5
    ) -> Tuple[float, float, float]:
        """
        Calculate Precision, Recall, and F1 at K.
        
        Critical for clinical settings where both finding relevant trials (recall)
        and avoiding irrelevant ones (precision) matter.
        
        Returns:
            Tuple of (precision@k, recall@k, f1@k)
        """
        if not rankings or k <= 0:
            return 0.0, 0.0, 0.0
        
        top_k_ids = {r.get('nct_id') for r in rankings[:k]}
        
        true_positives = len(top_k_ids & relevant_ids)
        
        precision = true_positives / k if k > 0 else 0.0
        recall = true_positives / len(relevant_ids) if relevant_ids else 0.0
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return precision, recall, f1
    
    @staticmethod
    def mean_reciprocal_rank(rankings: List[Dict], relevant_ids: Set[str]) -> float:
        """
        Calculate Mean Reciprocal Rank.
        
        MRR measures how quickly the first relevant trial appears,
        critical for time-sensitive cancer treatment decisions.
        """
        if not rankings or not relevant_ids:
            return 0.0
        
        for i, item in enumerate(rankings):
            if item.get('nct_id') in relevant_ids:
                return 1.0 / (i + 1)
        
        return 0.0
    
    @staticmethod
    def safety_violation_rate(
        rankings: List[Dict],
        safety_violations: Dict[str, List[str]],
        k: int = 10
    ) -> Dict[str, float]:
        """
        Calculate safety violation metrics.
        
        Critical for patient safety - identifies dangerous or inappropriate
        trial recommendations.
        
        Args:
            rankings: Ranked trials
            safety_violations: Dict mapping nct_id to list of violation types
            k: Top-k to check for violations
            
        Returns:
            Dict with violation rates by type and overall
        """
        top_k = rankings[:k] if len(rankings) >= k else rankings
        
        violation_counts = defaultdict(int)
        total_violations = 0
        
        for trial in top_k:
            nct_id = trial.get('nct_id')
            if nct_id in safety_violations:
                violations = safety_violations[nct_id]
                total_violations += 1
                for violation_type in violations:
                    violation_counts[violation_type] += 1
        
        results = {
            'overall_violation_rate': total_violations / len(top_k) if top_k else 0.0,
            'violations_by_type': dict(violation_counts),
            'critical_violations': sum(1 for v in violation_counts if 'critical' in v.lower())
        }
        
        return results
    
    @staticmethod
    def critical_miss_rate(
        rankings: List[Dict],
        gold_standard_trials: Set[str],
        n: int = 20
    ) -> float:
        """
        Calculate rate of missing best-available trials.
        
        Measures how often the system fails to identify known good matches,
        which could delay optimal treatment.
        
        Args:
            rankings: System's ranked trials
            gold_standard_trials: Expert-identified best trials
            n: Check if gold standard appears in top-N
            
        Returns:
            Miss rate (0 = perfect, 1 = all missed)
        """
        if not gold_standard_trials:
            return 0.0
        
        top_n_ids = {r.get('nct_id') for r in rankings[:n]}
        missed = gold_standard_trials - top_n_ids
        
        return len(missed) / len(gold_standard_trials)


@dataclass
class EquityMetrics:
    """
    Metrics for evaluating fairness and equity in trial matching.
    
    Ensures the system doesn't perpetuate healthcare disparities.
    """
    
    @staticmethod
    def subgroup_performance(
        results_by_subgroup: Dict[str, List[Dict]],
        relevant_by_subgroup: Dict[str, Set[str]],
        k: int = 10
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate performance metrics broken down by patient subgroups.
        
        Critical for ensuring equitable access to trials across demographics.
        
        Args:
            results_by_subgroup: Rankings grouped by subgroup
            relevant_by_subgroup: Relevant trials for each subgroup
            k: Top-k for metrics
            
        Returns:
            Nested dict of subgroup -> metric -> value
        """
        clinical_metrics = ClinicalMetrics()
        subgroup_metrics = {}
        
        for subgroup, rankings in results_by_subgroup.items():
            relevant_ids = relevant_by_subgroup.get(subgroup, set())
            
            # Calculate metrics for this subgroup
            precision, recall, f1 = clinical_metrics.precision_recall_f1_at_k(
                rankings, relevant_ids, k
            )
            
            # Build relevance scores for NDCG
            relevance_scores = {
                r['nct_id']: r.get('score', 0.0) 
                for r in rankings
            }
            
            ndcg = clinical_metrics.ndcg_at_k(rankings, relevance_scores, k)
            mrr = clinical_metrics.mean_reciprocal_rank(rankings, relevant_ids)
            
            subgroup_metrics[subgroup] = {
                'precision_at_k': precision,
                'recall_at_k': recall,
                'f1_at_k': f1,
                'ndcg_at_k': ndcg,
                'mrr': mrr,
                'num_patients': len(rankings)
            }
        
        # Calculate disparity metrics
        all_values = defaultdict(list)
        for metrics in subgroup_metrics.values():
            for key, value in metrics.items():
                if key != 'num_patients':
                    all_values[key].append(value)
        
        # Add disparity analysis
        disparity_metrics = {}
        for metric_name, values in all_values.items():
            if values:
                disparity_metrics[f'{metric_name}_disparity'] = {
                    'std': np.std(values),
                    'range': max(values) - min(values),
                    'cv': np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
                }
        
        subgroup_metrics['_disparity_analysis'] = disparity_metrics
        
        return subgroup_metrics
    
    @staticmethod
    def biomarker_rarity_impact(
        results: List[Dict],
        patient_biomarkers: Dict[str, List[str]],
        biomarker_prevalence: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Analyze matching performance for rare vs common biomarkers.
        
        Ensures patients with rare mutations get equal access to trials.
        """
        rare_threshold = 0.05  # 5% prevalence
        
        rare_biomarker_patients = []
        common_biomarker_patients = []
        
        for patient_id, biomarkers in patient_biomarkers.items():
            has_rare = any(
                biomarker_prevalence.get(bm, 1.0) < rare_threshold 
                for bm in biomarkers
            )
            
            if has_rare:
                rare_biomarker_patients.append(patient_id)
            else:
                common_biomarker_patients.append(patient_id)
        
        # Compare performance
        rare_results = [r for r in results if r.get('patient_id') in rare_biomarker_patients]
        common_results = [r for r in results if r.get('patient_id') in common_biomarker_patients]
        
        return {
            'rare_biomarker_match_rate': len(rare_results) / len(rare_biomarker_patients) if rare_biomarker_patients else 0,
            'common_biomarker_match_rate': len(common_results) / len(common_biomarker_patients) if common_biomarker_patients else 0,
            'rare_biomarker_avg_score': np.mean([r.get('score', 0) for r in rare_results]) if rare_results else 0,
            'common_biomarker_avg_score': np.mean([r.get('score', 0) for r in common_results]) if common_results else 0,
            'equity_gap': abs(len(rare_results)/max(1, len(rare_biomarker_patients)) - 
                             len(common_results)/max(1, len(common_biomarker_patients)))
        }


@dataclass
class EnsembleMetrics:
    """
    Metrics for evaluating judge ensemble agreement and reliability.
    """
    
    @staticmethod
    def krippendorff_alpha(ratings: np.ndarray) -> float:
        """
        Calculate Krippendorff's alpha for inter-rater reliability.
        
        Gold standard for measuring agreement in content analysis,
        handles missing data and multiple raters.
        
        Args:
            ratings: 2D array where rows are items, columns are raters
            
        Returns:
            Alpha value (-1 to 1, where 1 = perfect agreement)
        """
        try:
            import krippendorff
            return krippendorff.alpha(ratings, level_of_measurement='interval')
        except ImportError:
            logger.warning("krippendorff package not installed, using simplified calculation")
            # Simplified calculation
            return EnsembleMetrics._simplified_alpha(ratings)
    
    @staticmethod
    def _simplified_alpha(ratings: np.ndarray) -> float:
        """Simplified inter-rater agreement calculation."""
        n_items, n_raters = ratings.shape
        
        # Calculate observed disagreement
        observed_disagreement = 0
        comparisons = 0
        
        for i in range(n_items):
            item_ratings = ratings[i, ~np.isnan(ratings[i, :])]
            if len(item_ratings) > 1:
                for j in range(len(item_ratings)):
                    for k in range(j+1, len(item_ratings)):
                        observed_disagreement += (item_ratings[j] - item_ratings[k])**2
                        comparisons += 1
        
        if comparisons > 0:
            observed_disagreement /= comparisons
        
        # Calculate expected disagreement
        all_ratings = ratings[~np.isnan(ratings)].flatten()
        expected_disagreement = np.var(all_ratings) * 2
        
        if expected_disagreement == 0:
            return 1.0 if observed_disagreement == 0 else 0.0
        
        return 1 - (observed_disagreement / expected_disagreement)
    
    @staticmethod
    def fleiss_kappa(ratings: np.ndarray, n_categories: int = 5) -> float:
        """
        Calculate Fleiss' kappa for multiple raters.
        
        Standard metric for categorical agreement among multiple judges.
        """
        n_items, n_raters = ratings.shape
        
        # Convert to category counts
        category_counts = np.zeros((n_items, n_categories))
        for i in range(n_items):
            for j in range(n_raters):
                if not np.isnan(ratings[i, j]):
                    category = int(ratings[i, j])
                    if 0 <= category < n_categories:
                        category_counts[i, category] += 1
        
        # Calculate P_i (extent of agreement for item i)
        n_ratings_per_item = np.sum(category_counts, axis=1)
        P = np.zeros(n_items)
        for i in range(n_items):
            if n_ratings_per_item[i] > 1:
                P[i] = (np.sum(category_counts[i]**2) - n_ratings_per_item[i]) / \
                       (n_ratings_per_item[i] * (n_ratings_per_item[i] - 1))
        
        P_bar = np.mean(P)
        
        # Calculate P_e (chance agreement)
        p_j = np.sum(category_counts, axis=0) / np.sum(category_counts)
        P_e = np.sum(p_j**2)
        
        if P_e == 1:
            return 1.0 if P_bar == 1 else 0.0
        
        return (P_bar - P_e) / (1 - P_e)
    
    @staticmethod
    def judge_vote_analysis(
        judge_scores: Dict[str, Dict[str, float]],
        threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Analyze voting patterns in judge ensemble.
        
        Identifies when judges agree/disagree on safety and eligibility.
        
        Args:
            judge_scores: Dict mapping patient_id to dict of judge_name -> score
            threshold: Threshold for eligible decision
        """
        if not judge_scores:
            return {
                'unanimous_eligible_rate': 0.0,
                'majority_eligible_rate': 0.0,
                'unanimous_reject_rate': 0.0,
                'majority_reject_rate': 0.0,
                'high_disagreement_rate': 0.0,
                'average_agreement': 0.0
            }
        
        n_items = len(judge_scores)
        
        unanimous_eligible = 0
        majority_eligible = 0
        unanimous_reject = 0
        majority_reject = 0
        high_disagreement = 0
        all_stds = []
        
        for patient_id, judge_votes in judge_scores.items():
            scores = list(judge_votes.values())
            n_judges = len(scores)
            
            if n_judges == 0:
                continue
            
            eligible_votes = sum(1 for s in scores if s >= threshold)
            
            if eligible_votes == n_judges:
                unanimous_eligible += 1
            elif eligible_votes > n_judges / 2:
                majority_eligible += 1
            elif eligible_votes == 0:
                unanimous_reject += 1
            elif eligible_votes < n_judges / 2:
                majority_reject += 1
            
            # Check disagreement
            std_dev = np.std(scores) if len(scores) > 1 else 0
            all_stds.append(std_dev)
            if std_dev > 0.3:  # High variance indicates disagreement
                high_disagreement += 1
        
        return {
            'unanimous_eligible_rate': unanimous_eligible / n_items if n_items > 0 else 0,
            'majority_eligible_rate': majority_eligible / n_items if n_items > 0 else 0,
            'unanimous_reject_rate': unanimous_reject / n_items if n_items > 0 else 0,
            'majority_reject_rate': majority_reject / n_items if n_items > 0 else 0,
            'high_disagreement_rate': high_disagreement / n_items if n_items > 0 else 0,
            'average_agreement': 1 - np.mean(all_stds) if all_stds else 0.0
        }


@dataclass
class AblationMetrics:
    """
    Metrics for ablation studies and robustness testing.
    """
    
    @staticmethod
    def compare_configurations(
        baseline_results: Dict[str, Any],
        ablation_results: Dict[str, Dict[str, Any]],
        key_metrics: List[str] = None
    ) -> pd.DataFrame:
        """
        Compare performance across different system configurations.
        
        Essential for understanding which components contribute most to performance.
        """
        if key_metrics is None:
            key_metrics = ['precision_at_10', 'recall_at_10', 'ndcg_at_10', 'mrr', 'safety_score']
        
        comparison_data = []
        
        # Add baseline
        baseline_row = {'configuration': 'baseline'}
        for metric in key_metrics:
            baseline_row[metric] = baseline_results.get(metric, 0.0)
        comparison_data.append(baseline_row)
        
        # Add ablations
        for config_name, results in ablation_results.items():
            row = {'configuration': config_name}
            for metric in key_metrics:
                value = results.get(metric, 0.0)
                row[metric] = value
                row[f'{metric}_delta'] = value - baseline_results.get(metric, 0.0)
                row[f'{metric}_pct_change'] = (
                    ((value - baseline_results.get(metric, 0.0)) / baseline_results.get(metric, 1.0) * 100)
                    if baseline_results.get(metric, 0) != 0 else 0
                )
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        return df
    
    @staticmethod
    def robustness_analysis(
        perturbation_results: Dict[str, List[Dict]],
        baseline_results: List[Dict]
    ) -> Dict[str, Any]:
        """
        Analyze system robustness to input perturbations.
        
        Tests stability against noisy or adversarial inputs.
        """
        clinical_metrics = ClinicalMetrics()
        
        robustness_scores = {}
        
        for perturbation_type, perturbed_results in perturbation_results.items():
            # Compare rankings
            baseline_rankings = {r['nct_id']: i for i, r in enumerate(baseline_results)}
            perturbed_rankings = {r['nct_id']: i for i, r in enumerate(perturbed_results)}
            
            # Calculate rank correlation
            common_trials = set(baseline_rankings.keys()) & set(perturbed_rankings.keys())
            if common_trials:
                baseline_ranks = [baseline_rankings[t] for t in common_trials]
                perturbed_ranks = [perturbed_rankings[t] for t in common_trials]
                correlation, _ = stats.spearmanr(baseline_ranks, perturbed_ranks)
            else:
                correlation = 0.0
            
            # Calculate score stability
            baseline_scores = {r['nct_id']: r.get('score', 0) for r in baseline_results}
            perturbed_scores = {r['nct_id']: r.get('score', 0) for r in perturbed_results}
            
            score_diffs = []
            for trial_id in common_trials:
                diff = abs(baseline_scores[trial_id] - perturbed_scores[trial_id])
                score_diffs.append(diff)
            
            robustness_scores[perturbation_type] = {
                'rank_correlation': correlation,
                'mean_score_deviation': np.mean(score_diffs) if score_diffs else 0,
                'max_score_deviation': max(score_diffs) if score_diffs else 0,
                'top10_stability': len(set(r['nct_id'] for r in baseline_results[:10]) & 
                                       set(r['nct_id'] for r in perturbed_results[:10])) / 10
            }
        
        return robustness_scores


@dataclass
class PerformanceMetrics:
    """
    System performance and cost metrics.
    """
    
    @staticmethod
    def latency_analysis(latencies: List[float]) -> Dict[str, float]:
        """
        Comprehensive latency analysis with clinical context.
        
        In clinical settings, response time affects workflow adoption.
        """
        if not latencies:
            return {
                'p50': 0.0, 'p75': 0.0, 'p90': 0.0, 'p95': 0.0, 'p99': 0.0,
                'mean': 0.0, 'std': 0.0, 'max': 0.0, 'min': 0.0
            }
        
        return {
            'p50': np.percentile(latencies, 50),
            'p75': np.percentile(latencies, 75),
            'p90': np.percentile(latencies, 90),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99),
            'mean': np.mean(latencies),
            'std': np.std(latencies),
            'max': max(latencies),
            'min': min(latencies),
            'clinical_acceptable_rate': sum(1 for l in latencies if l < 15.0) / len(latencies)
        }
    
    @staticmethod
    def cost_analysis(
        token_counts: Dict[str, int],
        model_costs: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Detailed cost analysis for LLM usage.
        
        Critical for budget planning in clinical deployment.
        """
        total_cost = 0.0
        cost_by_model = {}
        
        for model, tokens in token_counts.items():
            cost_per_1k = model_costs.get(model, 0.01)  # Default $0.01 per 1k tokens
            model_cost = (tokens / 1000) * cost_per_1k
            cost_by_model[model] = model_cost
            total_cost += model_cost
        
        return {
            'total_cost': total_cost,
            'cost_by_model': cost_by_model,
            'total_tokens': sum(token_counts.values()),
            'tokens_by_model': token_counts,
            'avg_cost_per_patient': total_cost / max(1, len(token_counts)),
            'projected_monthly_cost': total_cost * 30 * 100  # Assuming 100 patients/day
        }


@dataclass
class ErrorAnalysis:
    """
    Detailed error analysis and failure mode identification.
    """
    
    @staticmethod
    def identify_hard_cases(
        results: List[Dict],
        judge_scores: Optional[Dict[str, Dict]] = None,
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Identify the most challenging patient-trial matches.
        
        These cases help identify system limitations and improvement areas.
        """
        hard_cases = []
        
        for result in results:
            difficulty_indicators = []
            
            # Low confidence
            if result.get('confidence', 1.0) < 0.5:
                difficulty_indicators.append('low_confidence')
            
            # High judge disagreement
            if judge_scores and result['patient_id'] in judge_scores:
                scores = list(judge_scores[result['patient_id']].values())
                if np.std(scores) > 0.3:
                    difficulty_indicators.append('high_disagreement')
            
            # Poor performance
            if result.get('score', 1.0) < threshold:
                difficulty_indicators.append('low_score')
            
            # No matches found
            if not result.get('matches'):
                difficulty_indicators.append('no_matches')
            
            if difficulty_indicators:
                hard_cases.append({
                    'patient_id': result.get('patient_id'),
                    'difficulty_indicators': difficulty_indicators,
                    'score': result.get('score', 0),
                    'confidence': result.get('confidence', 0),
                    'num_matches': len(result.get('matches', [])),
                    'reasoning': result.get('reasoning', ''),
                    'judge_std': np.std(scores) if judge_scores and result['patient_id'] in judge_scores else None
                })
        
        # Sort by difficulty (most difficult first)
        hard_cases.sort(key=lambda x: (len(x['difficulty_indicators']), -x['score']), reverse=True)
        
        return hard_cases[:20]  # Top 20 hardest cases
    
    @staticmethod
    def failure_mode_analysis(
        errors: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Categorize and analyze failure modes.
        
        Understanding failure patterns is crucial for system improvement.
        """
        failure_modes = defaultdict(list)
        
        for error in errors:
            # Categorize by error type
            if 'no_matches' in error.get('difficulty_indicators', []):
                failure_modes['no_matches_found'].append(error['patient_id'])
            
            if 'low_confidence' in error.get('difficulty_indicators', []):
                failure_modes['low_confidence'].append(error['patient_id'])
            
            if 'high_disagreement' in error.get('difficulty_indicators', []):
                failure_modes['judge_disagreement'].append(error['patient_id'])
            
            # Check for specific medical scenarios
            if error.get('reasoning'):
                reasoning = error['reasoning'].lower()
                if 'rare' in reasoning or 'uncommon' in reasoning:
                    failure_modes['rare_condition'].append(error['patient_id'])
                if 'complex' in reasoning or 'multiple' in reasoning:
                    failure_modes['complex_case'].append(error['patient_id'])
                if 'exclusion' in reasoning:
                    failure_modes['exclusion_criteria'].append(error['patient_id'])
        
        # Summarize
        summary = {
            'total_failures': len(errors),
            'failure_categories': {
                category: {
                    'count': len(patients),
                    'percentage': len(patients) / len(errors) * 100 if errors else 0,
                    'examples': patients[:5]  # First 5 examples
                }
                for category, patients in failure_modes.items()
            },
            'most_common_failure': max(failure_modes.items(), key=lambda x: len(x[1]))[0] if failure_modes else None
        }
        
        return summary
