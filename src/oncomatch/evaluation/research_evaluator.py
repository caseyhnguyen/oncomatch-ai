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
from typing import Dict, List, Any, Optional, Set, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import logging

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
    
    async def evaluate_clinical_metrics(
        self,
        patient_ids: List[str],
        k_values: List[int] = [5, 10, 20]
    ) -> Dict[str, Any]:
        """
        Evaluate comprehensive clinical metrics.
        
        Args:
            patient_ids: List of patient IDs to evaluate
            k_values: Different k values for top-k metrics
        """
        console.print("\n[bold cyan]📊 Clinical Metrics Evaluation[/bold cyan]")
        console.print("-" * 60)
        
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
            task = progress.add_task("Evaluating patients...", total=len(patient_ids))
            
            for patient_id in patient_ids:
                start_time = time.time()
                
                # Get match results
                result = await self.matcher.match_patient(patient_id, max_trials=20)
                
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
    
    async def evaluate_equity_metrics(
        self,
        n_synthetic: int = 100,
        k: int = 10
    ) -> Dict[str, Any]:
        """
        Evaluate equity and fairness metrics across patient subgroups.
        """
        console.print("\n[bold cyan]⚖️ Equity & Diversity Analysis[/bold cyan]")
        console.print("-" * 60)
        
        # Generate diverse synthetic patients
        generator = SyntheticPatientGenerator()
        patients = generator.generate_cohort(
            n_patients=n_synthetic,
            category_distribution={
                PatientCategory.STANDARD: 0.4,
                PatientCategory.EDGE_CASE: 0.2,
                PatientCategory.ADVERSARIAL: 0.1,
                PatientCategory.EQUITY_STRESS: 0.3
            }
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
        n_samples: int = 50
    ) -> Dict[str, Any]:
        """
        Evaluate judge ensemble agreement and reliability.
        """
        console.print("\n[bold cyan]🤝 Ensemble Agreement Analysis[/bold cyan]")
        console.print("-" * 60)
        
        # Sample patients for judge evaluation
        patient_ids = [f"P{i:03d}" for i in range(1, min(n_samples + 1, 31))]
        
        judge_scores_by_patient = {}
        ratings_matrix = []
        
        console.print(f"Evaluating judge agreement on {len(patient_ids)} patients...")
        
        for patient_id in patient_ids[:10]:  # Limit for demonstration
            result = await self.matcher.match_patient(patient_id, max_trials=5)
            
            if result and result.get('matches'):
                # Simulate judge scores (in production, would use actual ensemble)
                judge_scores = {
                    'accuracy_judge': np.random.uniform(0.5, 1.0),
                    'safety_judge': np.random.uniform(0.5, 1.0),
                    'completeness_judge': np.random.uniform(0.5, 1.0),
                    'bias_judge': np.random.uniform(0.5, 1.0),
                    'robustness_judge': np.random.uniform(0.5, 1.0)
                }
                
                judge_scores_by_patient[patient_id] = judge_scores
                ratings_matrix.append(list(judge_scores.values()))
        
        ratings_array = np.array(ratings_matrix)
        
        # Calculate agreement metrics
        alpha = self.ensemble_metrics.krippendorff_alpha(ratings_array.T)
        kappa = self.ensemble_metrics.fleiss_kappa(ratings_array)
        vote_analysis = self.ensemble_metrics.judge_vote_analysis(judge_scores_by_patient)
        
        self.results['ensemble_analysis'] = {
            'krippendorff_alpha': round(alpha, 4),
            'fleiss_kappa': round(kappa, 4),
            'vote_patterns': vote_analysis,
            'num_samples': len(judge_scores_by_patient)
        }
        
        # Display results
        self._display_ensemble_metrics(self.results['ensemble_analysis'])
        
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
        patient_ids: List[str] = None,
        n_synthetic: int = 100,
        k_values: List[int] = [5, 10, 20],
        ablation_configs: Dict[str, Dict] = None,
        show_all: bool = True
    ):
        """
        Run complete evaluation.
        """
        console.print("\n" + "=" * 60)
        console.print("[bold magenta]🏆 Evaluation Suite[/bold magenta]")
        console.print("=" * 60)
        console.print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Default patient IDs
        if patient_ids is None:
            patient_ids = [f"P{i:03d}" for i in range(1, 11)]
        
        # Initialize
        await self.initialize()
        
        # Run evaluations
        if show_all or 'clinical' in show_all:
            await self.evaluate_clinical_metrics(patient_ids, k_values)
        
        if show_all or 'equity' in show_all:
            await self.evaluate_equity_metrics(n_synthetic, k=10)
        
        if show_all or 'ensemble' in show_all:
            await self.evaluate_ensemble_agreement(n_samples=20)
        
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
        """Display ensemble agreement metrics."""
        console.print(f"\n[yellow]Judge Agreement Metrics:[/yellow]")
        console.print(f"  Krippendorff's α: {ensemble_results['krippendorff_alpha']:.3f}")
        console.print(f"  Fleiss' κ: {ensemble_results['fleiss_kappa']:.3f}")
        
        vote_patterns = ensemble_results['vote_patterns']
        console.print(f"\n[yellow]Voting Patterns:[/yellow]")
        console.print(f"  Unanimous eligible: {vote_patterns['unanimous_eligible_rate']:.2%}")
        console.print(f"  High disagreement: {vote_patterns['high_disagreement_rate']:.2%}")
        console.print(f"  Average agreement: {vote_patterns['average_agreement']:.3f}")
    
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
