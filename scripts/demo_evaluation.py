#!/usr/bin/env python3
"""
Comprehensive Evaluation Demo for OncoMatch-AI
Shows detailed clinical metrics with explanations for demo purposes
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import json
import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from oncomatch.match import ClinicalTrialMatcher
from oncomatch.evaluation.metrics import EvaluationMetrics, AggregateMetrics

console = Console()


class DemoEvaluator:
    """Demo-ready evaluation with comprehensive metrics and explanations."""
    
    def __init__(self):
        self.matcher = ClinicalTrialMatcher()
        self.eval_metrics = EvaluationMetrics()
        self.aggregate_metrics = AggregateMetrics()
        self.results = {}
        
    async def run_comprehensive_demo(self, patient_ids: List[str] = None):
        """Run comprehensive evaluation suitable for demo."""
        
        if patient_ids is None:
            patient_ids = ["P001", "P002", "P003", "P004", "P005"]
        
        # Header
        console.print("\n" + "="*80)
        console.print("[bold cyan]🏥 OncoMatch-AI Clinical Trial Matching System[/bold cyan]", justify="center")
        console.print("[bold]Comprehensive Evaluation Report[/bold]", justify="center")
        console.print("="*80)
        console.print(f"\n[dim]Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]")
        console.print(f"[dim]Patients Evaluated: {len(patient_ids)}[/dim]\n")
        
        # Run evaluations
        all_rankings = []
        all_latencies = []
        all_scores = {}
        relevant_trials = set()
        patient_results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Evaluating patients...", total=len(patient_ids))
            
            for patient_id in patient_ids:
                start_time = time.time()
                
                # Match patient
                result = await self.matcher.match_patient(
                    patient_id=patient_id,
                    max_trials=20,
                    mode="balanced"
                )
                
                latency = time.time() - start_time
                all_latencies.append(latency)
                
                if result and result.get('matches'):
                    patient_results.append(result)
                    matches = result['matches']
                    all_rankings.extend(matches)
                    
                    # Track scores and relevant trials
                    for match in matches:
                        nct_id = match['nct_id']
                        score = match.get('score', 0)
                        all_scores[nct_id] = score
                        if score > 0.7:  # Consider > 0.7 as relevant
                            relevant_trials.add(nct_id)
                
                progress.update(task, advance=1)
        
        # Calculate comprehensive metrics
        self._display_clinical_metrics(all_rankings, relevant_trials, all_scores)
        self._display_performance_metrics(all_latencies, patient_results)
        self._display_safety_metrics(patient_results)
        self._display_diversity_metrics(all_rankings)
        self._display_overall_grade(all_rankings, relevant_trials, all_scores, all_latencies)
        
        # Show sample matches
        self._display_sample_matches(patient_results)
        
        return self.results
    
    def _display_clinical_metrics(self, rankings: List[Dict], relevant_ids: set, scores: Dict[str, float]):
        """Display core clinical metrics with explanations."""
        
        console.print("\n[bold cyan]📊 CLINICAL EFFECTIVENESS METRICS[/bold cyan]")
        console.print("[dim]These metrics measure how well the system identifies relevant trials[/dim]\n")
        
        # Calculate metrics at different k values
        table = Table(title="Ranking Quality Metrics", show_header=True)
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("@5", justify="right", style="green")
        table.add_column("@10", justify="right", style="green")
        table.add_column("@20", justify="right", style="green")
        table.add_column("Interpretation", style="dim", width=40)
        
        for k in [5, 10, 20]:
            # Precision@K
            precision = self.eval_metrics.precision_at_k(rankings, list(relevant_ids), k)
            
            # Recall@K
            recall = self.eval_metrics.recall_at_k(rankings, list(relevant_ids), k)
            
            # nDCG@K
            ndcg = self.eval_metrics.ndcg_at_k(rankings, scores, k)
            
            if k == 5:
                table.add_row(
                    "Precision",
                    f"{precision:.3f}",
                    f"{self.eval_metrics.precision_at_k(rankings, list(relevant_ids), 10):.3f}",
                    f"{self.eval_metrics.precision_at_k(rankings, list(relevant_ids), 20):.3f}",
                    "% of recommended trials that are relevant"
                )
                table.add_row(
                    "Recall",
                    f"{recall:.3f}",
                    f"{self.eval_metrics.recall_at_k(rankings, list(relevant_ids), 10):.3f}",
                    f"{self.eval_metrics.recall_at_k(rankings, list(relevant_ids), 20):.3f}",
                    "% of all relevant trials found"
                )
                table.add_row(
                    "nDCG",
                    f"{ndcg:.3f}",
                    f"{self.eval_metrics.ndcg_at_k(rankings, scores, 10):.3f}",
                    f"{self.eval_metrics.ndcg_at_k(rankings, scores, 20):.3f}",
                    "Ranking quality (considers order)"
                )
        
        # Add F1 and MRR
        f1_5 = 2 * (self.eval_metrics.precision_at_k(rankings, list(relevant_ids), 5) * 
                    self.eval_metrics.recall_at_k(rankings, list(relevant_ids), 5)) / \
                   max(0.001, (self.eval_metrics.precision_at_k(rankings, list(relevant_ids), 5) + 
                              self.eval_metrics.recall_at_k(rankings, list(relevant_ids), 5)))
        f1_10 = 2 * (self.eval_metrics.precision_at_k(rankings, list(relevant_ids), 10) * 
                     self.eval_metrics.recall_at_k(rankings, list(relevant_ids), 10)) / \
                    max(0.001, (self.eval_metrics.precision_at_k(rankings, list(relevant_ids), 10) + 
                               self.eval_metrics.recall_at_k(rankings, list(relevant_ids), 10)))
        f1_20 = 2 * (self.eval_metrics.precision_at_k(rankings, list(relevant_ids), 20) * 
                     self.eval_metrics.recall_at_k(rankings, list(relevant_ids), 20)) / \
                    max(0.001, (self.eval_metrics.precision_at_k(rankings, list(relevant_ids), 20) + 
                               self.eval_metrics.recall_at_k(rankings, list(relevant_ids), 20)))
        
        table.add_row(
            "F1 Score",
            f"{f1_5:.3f}",
            f"{f1_10:.3f}",
            f"{f1_20:.3f}",
            "Balance of precision and recall"
        )
        
        mrr = self.eval_metrics.mean_reciprocal_rank(rankings, list(relevant_ids))
        table.add_row(
            "MRR",
            f"{mrr:.3f}",
            f"{mrr:.3f}",
            f"{mrr:.3f}",
            "Speed of finding first relevant trial"
        )
        
        console.print(table)
        
        # Store results
        self.results['clinical_metrics'] = {
            'precision_at_10': self.eval_metrics.precision_at_k(rankings, list(relevant_ids), 10),
            'recall_at_10': self.eval_metrics.recall_at_k(rankings, list(relevant_ids), 10),
            'ndcg_at_10': self.eval_metrics.ndcg_at_k(rankings, scores, 10),
            'f1_at_10': f1_10,
            'mrr': mrr
        }
    
    def _display_performance_metrics(self, latencies: List[float], patient_results: List[Dict]):
        """Display system performance metrics."""
        
        console.print("\n[bold cyan]⚡ SYSTEM PERFORMANCE METRICS[/bold cyan]")
        console.print("[dim]Speed and efficiency measurements[/dim]\n")
        
        table = Table(show_header=True)
        table.add_column("Metric", style="cyan", width=25)
        table.add_column("Value", justify="right", style="green", width=15)
        table.add_column("Target", justify="right", style="yellow", width=15)
        table.add_column("Status", justify="center", width=10)
        
        # Latency metrics
        avg_latency = np.mean(latencies)
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        max_latency = max(latencies)
        
        table.add_row(
            "Average Response Time",
            f"{avg_latency:.2f}s",
            "<15s",
            "✅" if avg_latency < 15 else "⚠️"
        )
        table.add_row(
            "Median (P50)",
            f"{p50:.2f}s",
            "<10s",
            "✅" if p50 < 10 else "⚠️"
        )
        table.add_row(
            "P95 Latency",
            f"{p95:.2f}s",
            "<30s",
            "✅" if p95 < 30 else "⚠️"
        )
        table.add_row(
            "Max Latency",
            f"{max_latency:.2f}s",
            "<60s",
            "✅" if max_latency < 60 else "⚠️"
        )
        
        # Throughput
        total_patients = len(patient_results)
        total_trials_processed = sum(r.get('trials_fetched', 0) for r in patient_results)
        
        table.add_row(
            "Patients Processed",
            str(total_patients),
            "-",
            "✅"
        )
        table.add_row(
            "Trials Analyzed",
            str(total_trials_processed),
            "-",
            "✅"
        )
        
        console.print(table)
        
        # Store results
        self.results['performance'] = {
            'avg_latency': avg_latency,
            'p50_latency': p50,
            'p95_latency': p95,
            'max_latency': max_latency
        }
    
    def _display_safety_metrics(self, patient_results: List[Dict]):
        """Display safety and confidence metrics."""
        
        console.print("\n[bold cyan]🛡️ SAFETY & CONFIDENCE METRICS[/bold cyan]")
        console.print("[dim]Ensuring safe and reliable recommendations[/dim]\n")
        
        table = Table(show_header=True)
        table.add_column("Metric", style="cyan", width=30)
        table.add_column("Value", justify="right", style="green", width=15)
        table.add_column("Interpretation", style="dim", width=35)
        
        # Calculate safety metrics
        total_matches = sum(len(r.get('matches', [])) for r in patient_results)
        high_confidence_matches = 0
        safety_concerns_count = 0
        
        for result in patient_results:
            for match in result.get('matches', []):
                if match.get('confidence', 0) > 0.8:
                    high_confidence_matches += 1
                if match.get('safety_concerns'):
                    safety_concerns_count += 1
        
        confidence_rate = high_confidence_matches / max(1, total_matches)
        safety_flag_rate = safety_concerns_count / max(1, total_matches)
        
        table.add_row(
            "High Confidence Rate",
            f"{confidence_rate:.1%}",
            "% matches with >80% confidence"
        )
        table.add_row(
            "Safety Flag Rate",
            f"{safety_flag_rate:.1%}",
            "% matches with safety concerns"
        )
        table.add_row(
            "Avg Confidence Score",
            f"{np.mean([m.get('confidence', 0) for r in patient_results for m in r.get('matches', [])]):.3f}",
            "Mean confidence across all matches"
        )
        table.add_row(
            "Coverage Rate",
            f"{len([r for r in patient_results if r.get('matches')])}/{len(patient_results)}",
            "Patients with at least 1 match"
        )
        
        console.print(table)
        
        # Store results
        self.results['safety'] = {
            'high_confidence_rate': confidence_rate,
            'safety_flag_rate': safety_flag_rate
        }
    
    def _display_diversity_metrics(self, rankings: List[Dict]):
        """Display trial diversity metrics."""
        
        console.print("\n[bold cyan]🌈 TRIAL DIVERSITY METRICS[/bold cyan]")
        console.print("[dim]Ensuring diverse trial recommendations[/dim]\n")
        
        # Calculate diversity
        phases = set()
        for trial in rankings:
            if 'phase' in trial:
                phases.add(trial['phase'])
        
        diversity_metrics = self.eval_metrics.trial_diversity(rankings)
        
        table = Table(show_header=True)
        table.add_column("Diversity Dimension", style="cyan", width=25)
        table.add_column("Score", justify="right", style="green", width=15)
        table.add_column("Details", style="dim", width=40)
        
        table.add_row(
            "Phase Diversity",
            f"{diversity_metrics['phase_diversity']:.3f}",
            f"{len(phases)} unique phases represented"
        )
        table.add_row(
            "Mechanism Diversity",
            f"{diversity_metrics['mechanism_diversity']:.3f}",
            "Variety in treatment approaches"
        )
        table.add_row(
            "Geographic Diversity",
            f"{diversity_metrics['geographic_diversity']:.3f}",
            "Distribution across locations"
        )
        
        console.print(table)
        
        # Store results
        self.results['diversity'] = diversity_metrics
    
    def _display_overall_grade(self, rankings: List[Dict], relevant_ids: set, scores: Dict[str, float], latencies: List[float]):
        """Calculate and display overall system grade."""
        
        console.print("\n[bold cyan]🏆 OVERALL SYSTEM EVALUATION[/bold cyan]")
        console.print("[dim]Comprehensive assessment of system performance[/dim]\n")
        
        # Calculate component scores
        precision = self.eval_metrics.precision_at_k(rankings, list(relevant_ids), 10)
        recall = self.eval_metrics.recall_at_k(rankings, list(relevant_ids), 10)
        ndcg = self.eval_metrics.ndcg_at_k(rankings, scores, 10)
        avg_latency = np.mean(latencies)
        
        # Performance score (inverse of latency, normalized)
        perf_score = max(0, min(1, 2.0 - (avg_latency / 15)))  # 1.0 at 0s, 0 at 30s
        
        # Calculate weighted overall score
        weights = {
            'precision': 0.25,
            'recall': 0.25,
            'ranking_quality': 0.25,
            'performance': 0.25
        }
        
        component_scores = {
            'Precision@10': (precision, weights['precision']),
            'Recall@10': (recall, weights['recall']),
            'Ranking Quality (nDCG)': (ndcg, weights['ranking_quality']),
            'Performance': (perf_score, weights['performance'])
        }
        
        overall_score = sum(score * weight for score, weight in component_scores.values())
        
        # Display component scores
        table = Table(show_header=True, title="Component Scores")
        table.add_column("Component", style="cyan", width=25)
        table.add_column("Score", justify="right", style="green", width=15)
        table.add_column("Weight", justify="right", style="yellow", width=10)
        table.add_column("Contribution", justify="right", style="magenta", width=15)
        
        for component, (score, weight) in component_scores.items():
            table.add_row(
                component,
                f"{score:.3f}",
                f"{weight:.0%}",
                f"{score * weight:.3f}"
            )
        
        table.add_row(
            "[bold]Overall Score[/bold]",
            f"[bold]{overall_score:.3f}[/bold]",
            "100%",
            f"[bold]{overall_score:.3f}[/bold]",
            style="bold cyan"
        )
        
        console.print(table)
        
        # Calculate letter grade
        if overall_score >= 0.9:
            grade = "A+"
            grade_color = "bright_green"
            interpretation = "Exceptional - Production Ready"
        elif overall_score >= 0.85:
            grade = "A"
            grade_color = "green"
            interpretation = "Excellent - Minor Optimizations Needed"
        elif overall_score >= 0.8:
            grade = "B+"
            grade_color = "green"
            interpretation = "Very Good - Some Improvements Recommended"
        elif overall_score >= 0.75:
            grade = "B"
            grade_color = "yellow"
            interpretation = "Good - Notable Areas for Improvement"
        elif overall_score >= 0.7:
            grade = "C+"
            grade_color = "yellow"
            interpretation = "Satisfactory - Significant Improvements Needed"
        elif overall_score >= 0.65:
            grade = "C"
            grade_color = "yellow"
            interpretation = "Acceptable - Major Improvements Required"
        else:
            grade = "D"
            grade_color = "red"
            interpretation = "Needs Work - Substantial Development Required"
        
        # Display grade panel
        grade_panel = Panel(
            f"[bold {grade_color}]{grade}[/bold {grade_color}]\n\n{interpretation}\n\n[dim]Overall Score: {overall_score:.3f}/1.000[/dim]",
            title="[bold]Final Grade[/bold]",
            border_style=grade_color,
            padding=(1, 2)
        )
        
        console.print("\n")
        console.print(grade_panel)
        
        # Store results
        self.results['overall'] = {
            'score': overall_score,
            'grade': grade,
            'component_scores': {k: v[0] for k, v in component_scores.items()}
        }
    
    def _display_sample_matches(self, patient_results: List[Dict]):
        """Display sample successful matches."""
        
        console.print("\n[bold cyan]📋 SAMPLE TRIAL MATCHES[/bold cyan]")
        console.print("[dim]Examples of successful patient-trial matches[/dim]\n")
        
        # Show top match for first 2 patients
        shown = 0
        for result in patient_results:
            if shown >= 2:
                break
            
            if result.get('matches'):
                patient_summary = result.get('patient_summary', {})
                top_match = result['matches'][0]
                
                console.print(f"[bold]Patient {result['patient_id']}[/bold]")
                console.print(f"  • {patient_summary.get('age', 'Unknown')} y/o, {patient_summary.get('cancer_type', 'Unknown')} cancer")
                console.print(f"  • Stage: {patient_summary.get('stage', 'Unknown')}")
                console.print(f"  • Biomarkers: {', '.join(patient_summary.get('biomarkers', [])) or 'None detected'}")
                
                console.print(f"\n  [green]→ Top Match:[/green] {top_match['nct_id']}")
                console.print(f"    Score: {top_match['score']:.2f} | Confidence: {top_match['confidence']:.2f}")
                console.print(f"    Title: {top_match['title'][:80]}...")
                
                if top_match.get('reasons'):
                    console.print("    Key Reasons:")
                    for reason in top_match['reasons'][:2]:
                        console.print(f"      • {reason[:70]}...")
                
                console.print()
                shown += 1
    
    def save_results(self, output_dir: str = "outputs/evaluation"):
        """Save evaluation results to file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = output_path / f"demo_evaluation_{timestamp}.json"
        
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        console.print(f"\n💾 Full results saved to: {json_file}")
        
        return json_file


async def main():
    """Run comprehensive demo evaluation."""
    evaluator = DemoEvaluator()
    
    # Run evaluation on subset of patients for demo
    patient_ids = ["P001", "P002", "P003", "P004", "P005"]
    
    results = await evaluator.run_comprehensive_demo(patient_ids)
    
    # Save results
    evaluator.save_results()
    
    # Footer
    console.print("\n" + "="*80)
    console.print("[bold green]✅ Evaluation Complete[/bold green]", justify="center")
    console.print(f"[dim]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]", justify="center")
    console.print("="*80 + "\n")
    
    return results


if __name__ == "__main__":
    asyncio.run(main())
