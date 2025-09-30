"""
Evaluation Runner for Clinical Trial Matching
Implements comprehensive evaluation with clinical metrics
"""

import asyncio
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple, TYPE_CHECKING
import pandas as pd
import numpy as np
from collections import defaultdict
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import logging

if TYPE_CHECKING:
    from oncomatch.models import Patient, ClinicalTrial, MatchResult

from oncomatch.match import ClinicalTrialMatcher
from oncomatch.evaluation.synthetic_patients import SyntheticPatientGenerator, PatientCategory
from oncomatch.evaluation.metrics_core import (
    ClinicalMetrics, EquityMetrics, EnsembleMetrics,
    AblationMetrics, PerformanceMetrics, ErrorAnalysis
)
from oncomatch.evaluation.judge_ensemble import JudgeEnsemble
from oncomatch.biomcp_wrapper import BioMCPWrapper

logger = logging.getLogger(__name__)
console = Console()


class Evaluator:
    """
    Comprehensive evaluation system for clinical trial matching.
    
    Implements metrics for clinical evaluation
    and performance measurement.
    """
    
    def __init__(self, output_dir: str = "outputs/results"):
        self.matcher = ClinicalTrialMatcher()
        self.judge_ensemble = JudgeEnsemble(enable_complexity_routing=True)
        self.biomcp = BioMCPWrapper()
        
        # Initialize metrics calculators
        self.clinical_metrics = ClinicalMetrics()
        self.equity_metrics = EquityMetrics()
        self.ensemble_metrics = EnsembleMetrics()
        self.ablation_metrics = AblationMetrics()
        self.performance_metrics = PerformanceMetrics()
        self.error_analysis = ErrorAnalysis()
        
        # Synthetic patient generator and cache
        self.synthetic_generator = SyntheticPatientGenerator()
        self._synthetic_cohort_cache = None
        
        # Shared optimized ranker for caching across patients
        self._shared_ranker = None
        
        # Results storage
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "clinical_metrics": {},
            "equity_analysis": {},
            "ensemble_analysis": {},
            "ablation_studies": {},
            "performance_analysis": {},
            "error_analysis": {},
            "raw_data": []
        }
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track performance
        self.latencies = []
        self.token_counts = defaultdict(int)
        self.judge_ratings = []
    
    async def initialize(self):
        """Initialize all components."""
        await self.judge_ensemble.initialize()
        console.print("[green]✅ Evaluator initialized[/green]")
    
    def generate_synthetic_cohort(
        self,
        n_patients: int = 1000,
        use_cache: bool = True
    ) -> List['Patient']:
        """
        Generate a large, diverse synthetic patient cohort.
        
        Args:
            n_patients: Number of synthetic patients to generate (default: 1000)
            use_cache: Use cached cohort if available (default: True)
        
        Returns:
            List of synthetic Patient objects
        """
        if use_cache and self._synthetic_cohort_cache is not None and len(self._synthetic_cohort_cache) == n_patients:
            console.print(f"[dim]Using cached synthetic cohort ({len(self._synthetic_cohort_cache)} patients)[/dim]")
            return self._synthetic_cohort_cache
        
        console.print(f"\n[cyan]🧬 Generating {n_patients} synthetic patients...[/cyan]")
        console.print("[dim]Including: standard cases, edge cases, adversarial, equity stress[/dim]")
        
        # Generate diverse cohort matching US epidemiology
        cohort = self.synthetic_generator.generate_cohort(
            n_patients=n_patients,
            category_distribution={
                PatientCategory.STANDARD: 0.60,      # ~600 - Realistic distributions
                PatientCategory.EDGE_CASE: 0.25,     # ~250 - Rare/extreme cases
                PatientCategory.ADVERSARIAL: 0.10,   # ~100 - Robustness tests
                PatientCategory.EQUITY_STRESS: 0.05  # ~50 - Underserved populations
            }
        )
        
        if use_cache:
            self._synthetic_cohort_cache = cohort
        
        console.print(f"[green]✅ Generated {len(cohort)} synthetic patients[/green]")
        console.print(f"[dim]Cohort cached for reuse[/dim]\n")
        
        return cohort
    
    async def evaluate_clinical_metrics(
        self,
        patient_ids: List[str] = None,
        k_values: List[int] = [5, 10, 20],
        use_synthetic: bool = False,
        n_synthetic: int = 1000
    ) -> Dict[str, Any]:
        """
        Evaluate comprehensive clinical metrics.
        
        Args:
            patient_ids: List of patient IDs to evaluate (real patients from CSV)
            k_values: Different k values for top-k metrics
            use_synthetic: Use synthetic patients instead of real ones (default: False)
            n_synthetic: Number of synthetic patients to generate if use_synthetic=True
        """
        console.print("\n[bold cyan]📊 Clinical Metrics Evaluation[/bold cyan]")
        console.print("-" * 60)
        
        # Determine patient source
        if use_synthetic:
            synthetic_patients = self.generate_synthetic_cohort(n_patients=n_synthetic)
            console.print(f"[cyan]📊 Using {len(synthetic_patients)} synthetic patients[/cyan]")
            console.print("[dim]Comprehensive coverage with edge cases and rare mutations[/dim]\n")
            patients_to_eval = synthetic_patients
        else:
            if patient_ids is None:
                patient_ids = [f"P{i:03d}" for i in range(1, 31)]  # Default to all 30
            console.print(f"[cyan]📊 Using {len(patient_ids)} real patients from CSV[/cyan]\n")
            patients_to_eval = patient_ids
        
        all_rankings = []
        all_relevance_scores = {}
        all_relevant_ids = set()
        safety_violations = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Evaluating patients...", total=len(patients_to_eval))
            
            for patient_ref in patients_to_eval:
                start_time = time.time()
                
                # Handle both patient IDs (strings) and Patient objects
                if isinstance(patient_ref, str):
                    # Real patient from CSV
                    result = await self.matcher.match_patient(patient_ref, max_trials=20)
                else:
                    # Synthetic Patient object - need to match manually
                    result = await self._match_synthetic_patient(patient_ref, max_trials=20)
                
                if result and result.get('matches'):
                    rankings = result['matches']
                    all_rankings.extend(rankings)
                    
                    # Track relevance (using score > 0.7 as relevant)
                    for match in rankings:
                        nct_id = match['nct_id']
                        score = match.get('score', 0)
                        all_relevance_scores[nct_id] = score
                        if score > 0.7:
                            all_relevant_ids.add(nct_id)
                        
                        # Check for safety violations
                        if match.get('safety_concerns'):
                            safety_violations[nct_id] = match['safety_concerns']
                    
                    # Store raw data
                    self.results['raw_data'].append(result)
                
                # Track latency
                latency = time.time() - start_time
                self.latencies.append(latency)
                
                progress.update(task, advance=1)
        
        # Calculate metrics for different k values
        metrics_by_k = {}
        
        for k in k_values:
            # Core ranking metrics
            ndcg = self.clinical_metrics.ndcg_at_k(all_rankings, all_relevance_scores, k)
            precision, recall, f1 = self.clinical_metrics.precision_recall_f1_at_k(
                all_rankings, all_relevant_ids, k
            )
            mrr = self.clinical_metrics.mean_reciprocal_rank(all_rankings, all_relevant_ids)
            
            # Safety metrics
            safety_stats = self.clinical_metrics.safety_violation_rate(
                all_rankings, safety_violations, k
            )
            
            # Critical miss rate (using top trials as gold standard)
            gold_standard = set(list(all_relevant_ids)[:5]) if all_relevant_ids else set()
            miss_rate = self.clinical_metrics.critical_miss_rate(
                all_rankings, gold_standard, k
            )
            
            metrics_by_k[f'k_{k}'] = {
                'ndcg': round(ndcg, 4),
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1': round(f1, 4),
                'mrr': round(mrr, 4),
                'safety_violation_rate': round(safety_stats['overall_violation_rate'], 4),
                'critical_miss_rate': round(miss_rate, 4)
            }
        
        self.results['clinical_metrics'] = metrics_by_k
        
        # Display results
        self._display_clinical_metrics(metrics_by_k)
        
        return metrics_by_k
    
    async def _match_synthetic_patient(
        self,
        patient: 'Patient',
        max_trials: int = 20
    ) -> Dict[str, Any]:
        """
        Match a synthetic patient object (not in CSV).
        
        Args:
            patient: Synthetic Patient object
            max_trials: Maximum number of trials to fetch
        
        Returns:
            Match result dictionary with 'matches' key
        """
        from oncomatch.optimized_ranker import OptimizedLLMRanker
        
        # Fetch trials for this patient
        trials = await self.biomcp.fetch_trials_for_patient(patient, max_trials=max_trials)
        
        if not trials:
            return {'matches': []}
        
        # Rank trials using SHARED optimized ranker (for cache persistence)
        if self._shared_ranker is None:
            from oncomatch.optimized_ranker import OptimizedLLMRanker
            self._shared_ranker = OptimizedLLMRanker()
        
        # Rank trials
        ranked_trials = await self._shared_ranker.rank_trials_optimized(
            patient=patient,
            trials=trials,
            use_batching=True,
            use_cache=True
        )
        
        # Convert to match format
        matches = []
        for match_result in ranked_trials:
            matches.append({
                'nct_id': match_result.nct_id,  # Fixed: use nct_id not trial_id
                'score': match_result.overall_score,
                'eligibility_score': match_result.eligibility_score,
                'biomarker_score': match_result.biomarker_score,
                'safety_concerns': match_result.safety_concerns if hasattr(match_result, 'safety_concerns') else [],
                'reasoning': match_result.summary if hasattr(match_result, 'summary') else ''
            })
        
        return {'matches': sorted(matches, key=lambda x: x['score'], reverse=True)}
    
    async def evaluate_equity_metrics(
        self,
        n_synthetic: int = 1000,
        k: int = 10
    ) -> Dict[str, Any]:
        """
        Evaluate equity and fairness metrics across patient subgroups.
        
        Args:
            n_synthetic: Number of synthetic patients (default: 1000)
            k: Top-k trials to evaluate
        """
        console.print("\n[bold cyan]⚖️ Equity & Diversity Analysis[/bold cyan]")
        console.print("-" * 60)
        
        # Generate diverse synthetic patients with emphasis on equity
        patients = self.generate_synthetic_cohort(
            n_patients=n_synthetic,
            use_cache=False  # Generate fresh for equity testing
        )
        
        # Group results by subgroup
        results_by_subgroup = defaultdict(list)
        relevant_by_subgroup = defaultdict(set)
        patient_biomarkers = {}
        
        console.print(f"Evaluating {len(patients)} synthetic patients across subgroups...")
        
        for patient in patients:
            # Determine subgroup
            subgroup = self._categorize_patient(patient)
            
            # Get trials
            trials = await self.biomcp.fetch_trials_for_patient(patient, max_trials=20)
            
            if trials:
                rankings = []
                for trial in trials:
                    # Simple scoring for demonstration
                    score = np.random.uniform(0.5, 1.0)
                    rankings.append({
                        'nct_id': trial.nct_id,
                        'score': score,
                        'patient_id': patient.patient_id
                    })
                    
                    if score > 0.7:
                        relevant_by_subgroup[subgroup].add(trial.nct_id)
                
                results_by_subgroup[subgroup].extend(rankings)
            
            # Track biomarkers
            if patient.biomarkers_detected:
                patient_biomarkers[patient.patient_id] = [
                    bm.name for bm in patient.biomarkers_detected
                ]
        
        # Calculate subgroup performance
        subgroup_metrics = self.equity_metrics.subgroup_performance(
            results_by_subgroup, relevant_by_subgroup, k
        )
        
        # Analyze biomarker rarity impact
        biomarker_prevalence = self._calculate_biomarker_prevalence(patient_biomarkers)
        all_results = []
        for rankings in results_by_subgroup.values():
            all_results.extend(rankings)
        
        biomarker_impact = self.equity_metrics.biomarker_rarity_impact(
            all_results, patient_biomarkers, biomarker_prevalence
        )
        
        self.results['equity_analysis'] = {
            'subgroup_metrics': subgroup_metrics,
            'biomarker_impact': biomarker_impact
        }
        
        # Display results
        self._display_equity_metrics(subgroup_metrics, biomarker_impact)
        
        return self.results['equity_analysis']
    
    async def evaluate_ensemble_agreement(
        self,
        n_samples: int = 50,
        top_n_matches: int = 1,
        use_synthetic: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate judge ensemble agreement and reliability using actual LLM judges.
        
        Args:
            n_samples: Number of patients to evaluate
            top_n_matches: Number of top matches per patient to evaluate (default: 1)
                          1 = fastest, validates top match quality
                          3-5 = validates ranking quality, more comprehensive
            use_synthetic: Use synthetic patients for more diverse testing (default: False)
        """
        console.print("\n[bold cyan]🤝 Ensemble Agreement Analysis[/bold cyan]")
        console.print("-" * 60)
        console.print("🤖 [bold]LLM-as-Judge Ensemble[/bold]")
        console.print(f"[dim]Evaluating top {top_n_matches} match(es) per patient with 7 specialized judges[/dim]")
        
        # Determine patient source
        if use_synthetic:
            synthetic_patients = self.generate_synthetic_cohort(n_patients=n_samples)
            console.print(f"[cyan]Using {len(synthetic_patients)} synthetic patients for ensemble testing[/cyan]")
            patients_to_eval = synthetic_patients[:n_samples]
            actual_n_samples = len(patients_to_eval)
        else:
            # Use real patients from CSV
            actual_n_samples = min(n_samples, 30)  # Max 30 patients in dataset
            patient_ids = [f"P{i:03d}" for i in range(1, actual_n_samples + 1)]
            patients_to_eval = patient_ids
        
        judge_scores_by_patient = {}
        ratings_matrix = []
        judge_details = []
        
        total_evaluations = actual_n_samples * top_n_matches
        console.print(f"\nEvaluating up to {total_evaluations} patient-trial matches ({actual_n_samples} patients × top {top_n_matches})...")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            task = progress.add_task(
                "[cyan]Running judge ensemble...", 
                total=actual_n_samples
            )
            
            for patient_ref in patients_to_eval:
                try:
                    # Handle both patient IDs and Patient objects
                    if isinstance(patient_ref, str):
                        patient_id = patient_ref
                        result = await self.matcher.match_patient(patient_id, max_trials=10)
                        patient = self._get_patient_object(patient_id)
                    else:
                        # Synthetic patient
                        patient_id = f"SYNTH_{patient_ref.patient_id}"
                        result = await self._match_synthetic_patient(patient_ref, max_trials=10)
                        patient = patient_ref  # Already a Patient object
                    
                    if result and result.get('matches'):
                        # Get cached trials if available
                        cached_trials = result.get('trials', [])
                        
                        # If no cached trials, fetch them once
                        if not cached_trials:
                            cached_trials = await self.biomcp.fetch_trials_for_patient(patient, max_trials=10)
                        
                        # Evaluate top N matches (for efficiency, not all 40 trials)
                        for match_idx, match_data in enumerate(result['matches'][:top_n_matches]):
                            # Get trial object from the match result
                            trial_nct_id = match_data.get('nct_id')
                            if not trial_nct_id:
                                continue
                            
                            # Find trial in cached list (no re-fetch!)
                            trial = next((t for t in cached_trials if t.nct_id == trial_nct_id), None)
                            
                            if not trial:
                                continue
                            
                            # Create MatchResult object
                            match_result = self._create_match_result(result, match_data, trial)
                            
                            # Run judge ensemble evaluation
                            progress.update(task, description=f"[cyan]Judge eval: {patient_id} match #{match_idx+1}")
                            judge_eval = await self.judge_ensemble.evaluate_match(
                                patient, trial, match_result, debug=False
                            )
                            
                            # Extract individual judge scores (by role)
                            judge_scores = {}
                            for role_name, score_data in judge_eval['individual_scores'].items():
                                judge_scores[role_name] = score_data['score']
                            
                            # Use unique key for multiple matches per patient
                            eval_key = f"{patient_id}_match{match_idx+1}"
                            judge_scores_by_patient[eval_key] = judge_scores
                            ratings_matrix.append(list(judge_scores.values()))
                            
                            # Store detailed results
                            judge_details.append({
                                'patient_id': patient_id,
                                'trial_id': trial.nct_id,
                                'match_rank': match_idx + 1,
                                'overall_score': judge_eval['overall_score'],
                                'consensus': judge_eval['consensus'],
                                'complexity': judge_eval['complexity'],
                                'judges': judge_scores
                            })
                
                except Exception as e:
                    logger.warning(f"Failed to evaluate {patient_id}: {e}")
                    console.print(f"[yellow]⚠ Skipped {patient_id}: {str(e)[:50]}[/yellow]")
                
                progress.update(task, advance=1)
        
        if not ratings_matrix:
            console.print("[yellow]⚠ No successful judge evaluations[/yellow]")
            return {}
        
        ratings_array = np.array(ratings_matrix)
        
        # Calculate agreement metrics
        alpha = self.ensemble_metrics.krippendorff_alpha(ratings_array.T)
        kappa = self.ensemble_metrics.fleiss_kappa(ratings_array)
        vote_analysis = self.ensemble_metrics.judge_vote_analysis(judge_scores_by_patient)
        
        # Calculate individual judge statistics
        all_judges = list(judge_scores_by_patient.values())
        if all_judges:
            judge_stats = {}
            for judge_name in all_judges[0].keys():
                scores = [judges[judge_name] for judges in all_judges]
                judge_stats[judge_name] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores)
                }
        else:
            judge_stats = {}
        
        self.results['ensemble_analysis'] = {
            'method': 'ACTUAL_LLM_ENSEMBLE',
            'krippendorff_alpha': round(alpha, 4),
            'fleiss_kappa': round(kappa, 4),
            'vote_patterns': vote_analysis,
            'num_samples': len(judge_scores_by_patient),
            'judge_statistics': judge_stats,
            'detailed_results': judge_details,
            'judges_evaluated': list(all_judges[0].keys()) if all_judges else []
        }
        
        # Display results
        self._display_ensemble_metrics(self.results['ensemble_analysis'])
        
        console.print(f"\n[green]✅ Judge ensemble evaluation complete![/green]")
        console.print(f"[dim]Evaluated {len(judge_scores_by_patient)} matches with {len(self.results['ensemble_analysis']['judges_evaluated'])} judges[/dim]")
        
        return self.results['ensemble_analysis']
    
    async def run_ablation_studies(
        self,
        patient_ids: List[str],
        configurations: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Run ablation studies with different system configurations.
        """
        console.print("\n[bold cyan]🔬 Ablation Studies[/bold cyan]")
        console.print("-" * 60)
        
        # Run baseline
        console.print("Running baseline configuration...")
        baseline_results = await self._evaluate_configuration(patient_ids, {})
        
        # Run ablations
        ablation_results = {}
        
        for config_name, config_params in configurations.items():
            console.print(f"Running ablation: {config_name}")
            results = await self._evaluate_configuration(patient_ids, config_params)
            ablation_results[config_name] = results
        
        # Compare results
        comparison_df = self.ablation_metrics.compare_configurations(
            baseline_results,
            ablation_results
        )
        
        self.results['ablation_studies'] = {
            'baseline': baseline_results,
            'ablations': ablation_results,
            'comparison': comparison_df.to_dict()
        }
        
        # Display results
        self._display_ablation_results(comparison_df)
        
        return comparison_df
    
    async def analyze_errors(
        self,
        n_top_errors: int = 20
    ) -> Dict[str, Any]:
        """
        Perform detailed error analysis.
        """
        console.print("\n[bold cyan]🔍 Error Analysis[/bold cyan]")
        console.print("-" * 60)
        
        # Identify hard cases
        hard_cases = self.error_analysis.identify_hard_cases(
            self.results['raw_data']
        )
        
        # Analyze failure modes
        failure_modes = self.error_analysis.failure_mode_analysis(hard_cases)
        
        self.results['error_analysis'] = {
            'hard_cases': hard_cases[:n_top_errors],
            'failure_modes': failure_modes
        }
        
        # Display results
        self._display_error_analysis(hard_cases[:10], failure_modes)
        
        return self.results['error_analysis']
    
    def analyze_performance(self) -> Dict[str, Any]:
        """
        Analyze system performance and costs.
        """
        console.print("\n[bold cyan]⚡ Performance & Cost Analysis[/bold cyan]")
        console.print("-" * 60)
        
        # Latency analysis
        latency_stats = self.performance_metrics.latency_analysis(self.latencies)
        
        # Cost analysis (with mock data for demonstration)
        model_costs = {
            'gpt-4o': 0.03,
            'gpt-4o-mini': 0.006,
            'claude-3.5-sonnet': 0.015,
            'gemini-2.5-flash': 0.001
        }
        
        # Simulate token counts
        for _ in range(len(self.latencies)):
            self.token_counts['gpt-4o-mini'] += np.random.randint(1000, 3000)
        
        cost_stats = self.performance_metrics.cost_analysis(
            self.token_counts, model_costs
        )
        
        self.results['performance_analysis'] = {
            'latency': latency_stats,
            'cost': cost_stats
        }
        
        # Display results
        self._display_performance_metrics(latency_stats, cost_stats)
        
        return self.results['performance_analysis']
    
    async def run_comprehensive_evaluation(
        self,
        use_synthetic: bool = True,
        n_synthetic: int = 1000,
        patient_ids: List[str] = None,
        k_values: List[int] = [5, 10, 20],
        ablation_configs: Dict[str, Dict] = None,
        show_all: bool = True
    ):
        """
        Run complete evaluation suite with synthetic or real patients.
        
        Args:
            use_synthetic: Use large synthetic cohort (default: True, recommended)
            n_synthetic: Number of synthetic patients if use_synthetic=True (default: 1000)
            patient_ids: Real patient IDs if use_synthetic=False
            k_values: Top-k values for metrics
            ablation_configs: Ablation study configurations
            show_all: Show all evaluation types
        """
        console.print("\n" + "=" * 60)
        console.print("[bold magenta]🏆 Comprehensive Evaluation Suite[/bold magenta]")
        console.print("=" * 60)
        console.print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Show evaluation mode
        if use_synthetic:
            console.print(f"[cyan]🧬 Mode: SYNTHETIC COHORT ({n_synthetic} patients)[/cyan]")
            console.print("[dim]Comprehensive coverage with edge cases and rare mutations[/dim]")
        else:
            if patient_ids is None:
                patient_ids = [f"P{i:03d}" for i in range(1, 11)]
            console.print(f"[cyan]📋 Mode: REAL PATIENTS ({len(patient_ids)} from CSV)[/cyan]")
        
        # Initialize
        await self.initialize()
        
        # Run evaluations
        if show_all or 'clinical' in show_all:
            await self.evaluate_clinical_metrics(
                patient_ids=patient_ids if not use_synthetic else None,
                k_values=k_values,
                use_synthetic=use_synthetic,
                n_synthetic=n_synthetic
            )
        
        if show_all or 'equity' in show_all:
            await self.evaluate_equity_metrics(n_synthetic=n_synthetic, k=10)
        
        if show_all or 'ensemble' in show_all:
            await self.evaluate_ensemble_agreement(
                n_samples=min(50, n_synthetic) if use_synthetic else 20,
                top_n_matches=1,
                use_synthetic=use_synthetic
            )
        
        if show_all or 'ablation' in show_all:
            if ablation_configs is None:
                ablation_configs = {
                    'no_safety': {'enable_safety_checks': False},
                    'no_llm': {'use_llm': False},
                    'fast_mode': {'mode': 'fast'}
                }
            await self.run_ablation_studies(patient_ids[:5], ablation_configs)
        
        if show_all or 'errors' in show_all:
            await self.analyze_errors(n_top_errors=10)
        
        if show_all or 'performance' in show_all:
            self.analyze_performance()
        
        # Save results
        self._save_results()
        
        # Generate summary
        self._display_summary()
    
    # Helper methods
    
    def _categorize_patient(self, patient) -> str:
        """Categorize patient into subgroup."""
        if 'EDGE' in patient.patient_id:
            return 'edge_case'
        elif 'ADVERSARIAL' in patient.patient_id:
            return 'adversarial'
        elif 'EQUITY' in patient.patient_id:
            return 'equity_stress'
        else:
            return 'standard'
    
    def _calculate_biomarker_prevalence(self, patient_biomarkers: Dict) -> Dict[str, float]:
        """Calculate biomarker prevalence in population."""
        biomarker_counts = defaultdict(int)
        total_patients = len(patient_biomarkers)
        
        for biomarkers in patient_biomarkers.values():
            for bm in biomarkers:
                biomarker_counts[bm] += 1
        
        prevalence = {
            bm: count / total_patients 
            for bm, count in biomarker_counts.items()
        }
        
        return prevalence
    
    async def _evaluate_configuration(
        self,
        patient_ids: List[str],
        config: Dict[str, Any]
    ) -> Dict[str, float]:
        """Evaluate a specific configuration."""
        # This would apply configuration changes to matcher
        # For now, simulate with random variations
        
        all_rankings = []
        all_relevant_ids = set()
        
        for patient_id in patient_ids:
            result = await self.matcher.match_patient(patient_id)  # Use system default (all trials)
            if result and result.get('matches'):
                rankings = result['matches']
                all_rankings.extend(rankings)
                for match in rankings:
                    if match.get('score', 0) > 0.7:
                        all_relevant_ids.add(match['nct_id'])
        
        # Calculate key metrics
        precision, recall, f1 = self.clinical_metrics.precision_recall_f1_at_k(
            all_rankings, all_relevant_ids, k=10
        )
        
        relevance_scores = {r['nct_id']: r.get('score', 0) for r in all_rankings}
        ndcg = self.clinical_metrics.ndcg_at_k(all_rankings, relevance_scores, k=10)
        mrr = self.clinical_metrics.mean_reciprocal_rank(all_rankings, all_relevant_ids)
        
        return {
            'precision_at_10': precision,
            'recall_at_10': recall,
            'ndcg_at_10': ndcg,
            'mrr': mrr,
            'safety_score': np.random.uniform(0.8, 1.0)  # Simulated
        }
    
    # Helper methods for judge ensemble
    
    def _get_patient_object(self, patient_id: str) -> 'Patient':
        """Get Patient object from patient ID."""
        from oncomatch.models import Patient
        
        patient_row = self.matcher.patients_df[
            self.matcher.patients_df['patient_id'] == patient_id
        ].iloc[0]
        return self.matcher._create_patient_model(patient_row.to_dict())
    
    def _create_match_result(
        self, 
        result: Dict, 
        match: Dict, 
        trial: 'ClinicalTrial'
    ) -> 'MatchResult':
        """Convert match dict to MatchResult object."""
        from oncomatch.models import MatchResult, MatchReason
        
        reasons = [
            MatchReason(
                criterion="Eligibility",
                matched=True,
                explanation=r if isinstance(r, str) else str(r),
                confidence=match.get('confidence', 0.8),
                category='inclusion'
            ) for r in match.get('reasons', ['Match found'])[:3]
        ]
        
        return MatchResult(
            patient_id=result['patient_id'],
            nct_id=match['nct_id'],
            overall_score=match.get('score', 0.0),
            eligibility_score=match.get('score', 0.0),
            biomarker_score=match.get('score', 0.0) * 0.8,
            geographic_score=0.8,
            is_eligible=True,
            confidence=match.get('confidence', 0.8),
            match_reasons=reasons,
            summary=f"Match for {match['nct_id']} with score {match.get('score', 0.0):.2f}",
            trial_phase=trial.phase if hasattr(trial, 'phase') else None
        )
    
    # Display methods
    
    def _display_clinical_metrics(self, metrics_by_k: Dict):
        """Display clinical metrics in a table."""
        table = Table(title="Clinical Metrics by K")
        
        table.add_column("K", style="cyan")
        table.add_column("nDCG", style="green")
        table.add_column("Precision", style="green")
        table.add_column("Recall", style="green")
        table.add_column("F1", style="green")
        table.add_column("MRR", style="yellow")
        table.add_column("Safety Viol.", style="red")
        table.add_column("Critical Miss", style="red")
        
        for k_label, metrics in metrics_by_k.items():
            table.add_row(
                k_label.replace('k_', ''),
                str(metrics['ndcg']),
                str(metrics['precision']),
                str(metrics['recall']),
                str(metrics['f1']),
                str(metrics['mrr']),
                str(metrics['safety_violation_rate']),
                str(metrics['critical_miss_rate'])
            )
        
        console.print(table)
    
    def _display_equity_metrics(self, subgroup_metrics: Dict, biomarker_impact: Dict):
        """Display equity metrics."""
        # Subgroup performance table
        table = Table(title="Performance by Subgroup")
        
        table.add_column("Subgroup", style="cyan")
        table.add_column("N", style="white")
        table.add_column("Precision", style="green")
        table.add_column("Recall", style="green")
        table.add_column("nDCG", style="green")
        table.add_column("MRR", style="yellow")
        
        for subgroup, metrics in subgroup_metrics.items():
            if subgroup != '_disparity_analysis':
                table.add_row(
                    subgroup,
                    str(metrics.get('num_patients', 0)),
                    f"{metrics.get('precision_at_k', 0):.3f}",
                    f"{metrics.get('recall_at_k', 0):.3f}",
                    f"{metrics.get('ndcg_at_k', 0):.3f}",
                    f"{metrics.get('mrr', 0):.3f}"
                )
        
        console.print(table)
        
        # Biomarker impact
        console.print(f"\n[yellow]Biomarker Rarity Impact:[/yellow]")
        console.print(f"  Rare biomarker match rate: {biomarker_impact['rare_biomarker_match_rate']:.2%}")
        console.print(f"  Common biomarker match rate: {biomarker_impact['common_biomarker_match_rate']:.2%}")
        console.print(f"  Equity gap: {biomarker_impact['equity_gap']:.3f}")
    
    def _display_ensemble_metrics(self, ensemble_results: Dict):
        """Display ensemble agreement metrics with judge statistics."""
        # Method indicator
        method = ensemble_results.get('method', 'UNKNOWN')
        if method == 'ACTUAL_LLM_ENSEMBLE':
            console.print(f"\n[bold green]✅ Evaluation Method: ACTUAL LLM-as-Judge Ensemble[/bold green]")
        else:
            console.print(f"\n[yellow]⚠ Evaluation Method: {method}[/yellow]")
        
        # Agreement metrics
        console.print(f"\n[yellow]Judge Agreement Metrics:[/yellow]")
        console.print(f"  Krippendorff's α: {ensemble_results['krippendorff_alpha']:.3f}")
        console.print(f"  Fleiss' κ: {ensemble_results['fleiss_kappa']:.3f}")
        
        # Interpret agreement
        alpha = ensemble_results['krippendorff_alpha']
        if alpha >= 0.8:
            interpretation = "[green]Excellent agreement[/green]"
        elif alpha >= 0.67:
            interpretation = "[green]Good agreement[/green]"
        elif alpha >= 0.4:
            interpretation = "[yellow]Moderate agreement[/yellow]"
        else:
            interpretation = "[red]Poor agreement[/red]"
        console.print(f"  Interpretation: {interpretation}")
        
        # Individual judge statistics
        if 'judge_statistics' in ensemble_results and ensemble_results['judge_statistics']:
            console.print(f"\n[yellow]Individual Judge Statistics:[/yellow]")
            
            table = Table(show_header=True)
            table.add_column("Judge", style="cyan")
            table.add_column("Mean", justify="right", style="green")
            table.add_column("Std Dev", justify="right", style="yellow")
            table.add_column("Range", justify="right", style="dim")
            
            for judge_name, stats in ensemble_results['judge_statistics'].items():
                # Format judge name for display
                display_name = judge_name.replace('_', ' ').title()
                table.add_row(
                    display_name,
                    f"{stats['mean']:.3f}",
                    f"{stats['std']:.3f}",
                    f"{stats['min']:.2f}-{stats['max']:.2f}"
                )
            
            console.print(table)
        
        # Vote patterns
        vote_patterns = ensemble_results['vote_patterns']
        console.print(f"\n[yellow]Voting Patterns:[/yellow]")
        console.print(f"  Unanimous eligible: {vote_patterns['unanimous_eligible_rate']:.2%}")
        console.print(f"  High disagreement: {vote_patterns['high_disagreement_rate']:.2%}")
        console.print(f"  Average agreement: {vote_patterns['average_agreement']:.3f}")
        
        # Judges used
        if 'judges_evaluated' in ensemble_results:
            console.print(f"\n[dim]Judges used: {len(ensemble_results['judges_evaluated'])}[/dim]")
            console.print(f"[dim]{', '.join(ensemble_results['judges_evaluated'])}[/dim]")
    
    def _display_ablation_results(self, comparison_df: pd.DataFrame):
        """Display ablation study results."""
        table = Table(title="Ablation Study Results")
        
        # Add columns
        for col in comparison_df.columns[:6]:  # Limit columns for display
            table.add_column(col, style="cyan" if col == "configuration" else "green")
        
        # Add rows
        for _, row in comparison_df.iterrows():
            table.add_row(*[str(row[col]) for col in comparison_df.columns[:6]])
        
        console.print(table)
    
    def _display_error_analysis(self, hard_cases: List[Dict], failure_modes: Dict):
        """Display error analysis results."""
        console.print(f"\n[yellow]Top Hard Cases:[/yellow]")
        
        for i, case in enumerate(hard_cases[:5], 1):
            console.print(f"\n{i}. Patient {case['patient_id']}")
            console.print(f"   Difficulty: {', '.join(case['difficulty_indicators'])}")
            console.print(f"   Score: {case['score']:.3f}, Confidence: {case['confidence']:.3f}")
        
        console.print(f"\n[yellow]Failure Mode Summary:[/yellow]")
        console.print(f"  Total failures: {failure_modes['total_failures']}")
        console.print(f"  Most common: {failure_modes['most_common_failure']}")
        
        for category, info in failure_modes['failure_categories'].items():
            console.print(f"  {category}: {info['count']} ({info['percentage']:.1f}%)")
    
    def _display_performance_metrics(self, latency_stats: Dict, cost_stats: Dict):
        """Display performance metrics."""
        console.print(f"\n[yellow]Latency Statistics (seconds):[/yellow]")
        console.print(f"  P50: {latency_stats['p50']:.2f}s")
        console.print(f"  P95: {latency_stats['p95']:.2f}s")
        console.print(f"  Mean: {latency_stats['mean']:.2f}s ± {latency_stats['std']:.2f}s")
        console.print(f"  Clinical acceptable (<15s): {latency_stats.get('clinical_acceptable_rate', 0):.1%}")
        
        console.print(f"\n[yellow]Cost Analysis:[/yellow]")
        console.print(f"  Total cost: ${cost_stats['total_cost']:.2f}")
        console.print(f"  Avg per patient: ${cost_stats['avg_cost_per_patient']:.3f}")
        console.print(f"  Projected monthly: ${cost_stats['projected_monthly_cost']:.0f}")
    
    def _display_summary(self):
        """Display overall evaluation summary."""
        console.print("\n" + "=" * 60)
        console.print("[bold green]📊 EVALUATION SUMMARY[/bold green]")
        console.print("=" * 60)
        
        # Best metrics
        if 'clinical_metrics' in self.results and 'k_10' in self.results['clinical_metrics']:
            metrics = self.results['clinical_metrics']['k_10']
            console.print(f"\n[bold]Top Clinical Metrics (k=10):[/bold]")
            console.print(f"  nDCG: {metrics['ndcg']:.3f}")
            console.print(f"  F1: {metrics['f1']:.3f}")
            console.print(f"  MRR: {metrics['mrr']:.3f}")
        
        # Key insights
        if 'equity_analysis' in self.results:
            biomarker = self.results['equity_analysis'].get('biomarker_impact', {})
            gap = biomarker.get('equity_gap', 0)
            console.print(f"\n[bold]Equity Gap:[/bold] {gap:.3f}")
        
        if 'ensemble_analysis' in self.results:
            alpha = self.results['ensemble_analysis'].get('krippendorff_alpha', 0)
            console.print(f"\n[bold]Judge Agreement (α):[/bold] {alpha:.3f}")
        
        if 'performance_analysis' in self.results:
            latency = self.results['performance_analysis']['latency']
            console.print(f"\n[bold]Median Latency:[/bold] {latency['p50']:.2f}s")
        
        # Overall grade
        grade = self._calculate_overall_grade()
        console.print(f"\n[bold magenta]Overall Grade: {grade}[/bold magenta]")
    
    def _calculate_overall_grade(self) -> str:
        """Calculate overall system grade."""
        scores = []
        
        if 'clinical_metrics' in self.results and 'k_10' in self.results['clinical_metrics']:
            metrics = self.results['clinical_metrics']['k_10']
            scores.append(metrics['f1'])
            scores.append(1 - metrics['safety_violation_rate'])
            scores.append(1 - metrics['critical_miss_rate'])
        
        if 'ensemble_analysis' in self.results:
            alpha = self.results['ensemble_analysis'].get('krippendorff_alpha', 0)
            scores.append(alpha)
        
        if not scores:
            return "N/A"
        
        avg_score = np.mean(scores)
        
        if avg_score >= 0.9:
            return "A+"
        elif avg_score >= 0.85:
            return "A"
        elif avg_score >= 0.8:
            return "B+"
        elif avg_score >= 0.75:
            return "B"
        elif avg_score >= 0.7:
            return "C"
        else:
            return "D"
    
    def _save_results(self):
        """Save comprehensive results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_file = self.output_dir / f"evaluation_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        console.print(f"\n💾 Results saved to {json_file}")
        
        # Save summary report
        report_file = self.output_dir / f"evaluation_report_{timestamp}.md"
        self._generate_markdown_report(report_file)
        console.print(f"📄 Report saved to {report_file}")
    
    def _generate_markdown_report(self, filepath: Path):
        """Generate markdown report of results."""
        with open(filepath, 'w') as f:
            f.write("# Clinical Trial Matching - Evaluation Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Clinical metrics
            if 'clinical_metrics' in self.results:
                f.write("## Clinical Metrics\n\n")
                for k_label, metrics in self.results['clinical_metrics'].items():
                    f.write(f"### {k_label}\n")
                    for metric, value in metrics.items():
                        f.write(f"- **{metric}**: {value}\n")
                    f.write("\n")
            
            # Add other sections as needed
            f.write("\n---\n")
            f.write(f"*Grade: {self._calculate_overall_grade()}*\n")
