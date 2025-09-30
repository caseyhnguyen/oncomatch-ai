#!/usr/bin/env python
"""
Test Suite for Clinical Trial Matching System
Run: python tests/test_matching.py

This implements evaluation metrics for the clinical trial matching system.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import asyncio
import argparse
from datetime import datetime
import json
from typing import List, Set
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Import evaluator
from oncomatch.evaluation.evaluator import Evaluator
from oncomatch.match import ClinicalTrialMatcher


def parse_arguments():
    """Parse command line arguments for evaluation."""
    parser = argparse.ArgumentParser(
        description="Clinical Trial Matching Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all metrics
  python tests/test_matching.py --metrics all
  
  # Show specific analyses
  python tests/test_matching.py --metrics clinical equity --show-agreement
  
  # Run ablation studies
  python tests/test_matching.py --ablation --configs "no_safety,no_llm,fast_mode"
  
  # Analyze errors and failures
  python tests/test_matching.py --top-errors 20 --by-category
  
  # Full research evaluation
  python tests/test_matching.py --metrics all --show-agreement --ablation --top-errors --by-category
        """
    )
    
    # Metric selection
    parser.add_argument(
        '--metrics',
        nargs='+',
        choices=['all', 'clinical', 'equity', 'ensemble', 'performance', 'errors'],
        default=['clinical'],
        help='Which metrics to evaluate (default: clinical)'
    )
    
    # Agreement analysis
    parser.add_argument(
        '--show-agreement',
        action='store_true',
        help='Show judge ensemble agreement analysis (Krippendorff α, Fleiss κ)'
    )
    
    # Category breakdown
    parser.add_argument(
        '--by-category',
        action='store_true',
        help='Show metrics broken down by patient category/subgroup'
    )
    
    # Ablation studies
    parser.add_argument(
        '--ablation',
        action='store_true',
        help='Run ablation studies to test component contributions'
    )
    
    parser.add_argument(
        '--configs',
        type=str,
        help='Comma-separated ablation configs (e.g., "no_safety,no_llm,fast_mode")'
    )
    
    # Error analysis
    parser.add_argument(
        '--top-errors',
        type=int,
        default=0,
        help='Number of top errors/hard cases to analyze (0 = skip)'
    )
    
    # Patient selection
    parser.add_argument(
        '--patient-ids',
        nargs='+',
        help='Specific patient IDs to test (e.g., P001 P002 P003)'
    )
    
    parser.add_argument(
        '--n-patients',
        type=int,
        default=10,
        help='Number of patients to test (default: 10)'
    )
    
    parser.add_argument(
        '--n-synthetic',
        type=int,
        default=100,
        help='Number of synthetic patients for equity analysis (default: 100)'
    )
    
    # K values for top-k metrics
    parser.add_argument(
        '--k-values',
        nargs='+',
        type=int,
        default=[5, 10, 20],
        help='K values for top-k metrics (default: 5 10 20)'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/results',
        help='Directory for output files'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output detailed JSON results'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Minimal output (results only)'
    )
    
    return parser.parse_args()


async def run_simple_tests():
    """Run comprehensive evaluation suitable for demo."""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    console = Console()
    
    # Header
    console.print("\n" + "="*80)
    console.print("[bold cyan]🏥 OncoMatch-AI Clinical Trial Matching System[/bold cyan]", justify="center")
    console.print("[bold]Comprehensive Clinical Evaluation[/bold]", justify="center")
    console.print("="*80)
    console.print(f"\n[dim]Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]")
    
    from oncomatch.match import ClinicalTrialMatcher
    from oncomatch.evaluation.metrics import EvaluationMetrics, AggregateMetrics
    
    # Initialize with progress
    with tqdm(total=1, desc="Initializing system", leave=False) as pbar:
        matcher = ClinicalTrialMatcher()
        eval_metrics = EvaluationMetrics()
        aggregate = AggregateMetrics()
        pbar.update(1)
    
    # Test patients - use first 5 for more comprehensive evaluation
    test_patients = ["P001", "P002", "P003", "P004", "P005"]
    all_rankings = []
    all_latencies = []
    all_scores = {}
    relevant_trials = {}  # Changed to dict to track per-patient
    test_results = []
    patient_rankings = {}  # Track rankings per patient
    
    console.print(f"\n[bold cyan]📊 PATIENT MATCHING EVALUATION[/bold cyan]")
    console.print(f"[dim]Evaluating {len(test_patients)} patients with real clinical trials[/dim]\n")
    
    # Run evaluations
    with tqdm(total=len(test_patients), desc="Evaluating patients", 
              bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}') as pbar:
        
        for patient_id in test_patients:
            import time
            start_time = time.time()
            
            try:
                result = await matcher.match_patient(patient_id, mode="balanced")  # Use default (all trials)
                elapsed = time.time() - start_time
                all_latencies.append(elapsed)
                
                if result and result.get('matches'):
                    matches = result['matches']
                    patient_rankings[patient_id] = matches
                    all_rankings.extend(matches)
                    
                    # Track scores and relevant trials per patient
                    patient_relevant = set()
                    for match in matches:
                        nct_id = match['nct_id']
                        score = match.get('score', 0)
                        all_scores[nct_id] = score
                        if score > 0.5:  # Threshold for relevance (lowered from 0.7 to include moderate matches)
                            patient_relevant.add(nct_id)
                    
                    relevant_trials[patient_id] = patient_relevant
                    
                    test_results.append({
                        "patient": patient_id,
                        "status": "SUCCESS",
                        "trials_found": len(matches),
                        "relevant_found": len(patient_relevant),
                        "top_score": matches[0]['score'] if matches else 0,
                        "avg_score": np.mean([m['score'] for m in matches]),
                        "confidence_avg": np.mean([m.get('confidence', 0.5) for m in matches]),
                        "time": elapsed,
                        "result": result
                    })
                else:
                    test_results.append({
                        "patient": patient_id,
                        "status": "NO_MATCHES",
                        "trials_found": 0,
                        "relevant_found": 0,
                        "time": elapsed
                    })
            except Exception as e:
                test_results.append({
                    "patient": patient_id,
                    "status": "ERROR",
                    "error": str(e),
                    "trials_found": 0,
                    "relevant_found": 0,
                    "time": elapsed
                })
            
            pbar.update(1)
    
    # Display Clinical Metrics
    console.print("\n[bold cyan]📈 CLINICAL EFFECTIVENESS METRICS[/bold cyan]")
    console.print("[dim]Core metrics for evaluating trial matching quality[/dim]\n")
    
    if all_rankings and relevant_trials:
        # Aggregate relevant trials across all patients
        all_relevant = set()
        for patient_trials in relevant_trials.values():
            all_relevant.update(patient_trials)
        
        # Calculate metrics with aggregated relevant trials
        precision_3 = eval_metrics.precision_at_k(all_rankings, list(all_relevant), 3)
        precision_5 = eval_metrics.precision_at_k(all_rankings, list(all_relevant), 5)
        precision_10 = eval_metrics.precision_at_k(all_rankings, list(all_relevant), 10)
        recall_3 = eval_metrics.recall_at_k(all_rankings, list(all_relevant), 3)
        recall_5 = eval_metrics.recall_at_k(all_rankings, list(all_relevant), 5)
        recall_10 = eval_metrics.recall_at_k(all_rankings, list(all_relevant), 10)
        ndcg_3 = eval_metrics.ndcg_at_k(all_rankings, all_scores, 3)
        ndcg_5 = eval_metrics.ndcg_at_k(all_rankings, all_scores, 5)
        ndcg_10 = eval_metrics.ndcg_at_k(all_rankings, all_scores, 10)
        mrr = eval_metrics.mean_reciprocal_rank(all_rankings, list(all_relevant))
        
        # F1 scores
        f1_5 = 2 * (precision_5 * recall_5) / (precision_5 + recall_5) if (precision_5 + recall_5) > 0 else 0
        f1_10 = 2 * (precision_10 * recall_10) / (precision_10 + recall_10) if (precision_10 + recall_10) > 0 else 0
        
        table = Table(show_header=True, title="🎯 Ranking Quality Metrics (Higher is Better)")
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("@3", justify="right", style="yellow")
        table.add_column("@5", justify="right", style="green")
        table.add_column("@10", justify="right", style="green")
        table.add_column("Interpretation", style="dim", width=40)
        
        table.add_row("Precision", f"{precision_3:.3f}", f"{precision_5:.3f}", f"{precision_10:.3f}", 
                     "% of recommended trials that are relevant")
        table.add_row("Recall", f"{recall_3:.3f}", f"{recall_5:.3f}", f"{recall_10:.3f}", 
                     "% of all relevant trials found")
        table.add_row("F1 Score", f"-", f"{f1_5:.3f}", f"{f1_10:.3f}", 
                     "Harmonic mean of precision and recall")
        table.add_row("nDCG", f"{ndcg_3:.3f}", f"{ndcg_5:.3f}", f"{ndcg_10:.3f}", 
                     "Quality of ranking order (position-aware)")
        table.add_row("MRR", f"{mrr:.3f}", f"{mrr:.3f}", f"{mrr:.3f}", 
                     "1/rank of first relevant result")
        
        console.print(table)
        
        # Additional context
        console.print(f"\n[dim]📊 Context:[/dim]")
        console.print(f"  • Total trials evaluated: {len(all_rankings)}")
        console.print(f"  • Unique relevant trials found: {len(all_relevant)}")
        console.print(f"  • Relevance threshold: score > 0.40")
        console.print(f"  • Average trial score: {np.mean(list(all_scores.values())):.3f}")
    
    # Display Performance Metrics
    console.print("\n[bold cyan]⚡ SYSTEM PERFORMANCE[/bold cyan]")
    console.print("[dim]Speed and efficiency measurements[/dim]\n")
    
    perf_table = Table(show_header=True)
    perf_table.add_column("Metric", style="cyan", width=25)
    perf_table.add_column("Value", justify="right", style="green")
    perf_table.add_column("Target", justify="right", style="yellow")
    perf_table.add_column("Status", justify="center")
    
    avg_latency = np.mean(all_latencies) if all_latencies else 0
    max_latency = max(all_latencies) if all_latencies else 0
    p95_latency = np.percentile(all_latencies, 95) if all_latencies else 0
    
    perf_table.add_row(
        "Avg Response Time",
        f"{avg_latency:.2f}s",
        "<15s",
        "✅" if avg_latency < 15 else "⚠️"
    )
    perf_table.add_row(
        "P95 Latency",
        f"{p95_latency:.2f}s",
        "<30s",
        "✅" if p95_latency < 30 else "⚠️"
    )
    perf_table.add_row(
        "Max Latency",
        f"{max_latency:.2f}s",
        "<60s",
        "✅" if max_latency < 60 else "❌"
    )
    
    total_trials = sum(r.get('trials_found', 0) for r in test_results)
    perf_table.add_row(
        "Total Trials Analyzed",
        str(total_trials),
        ">20",
        "✅" if total_trials > 20 else "⚠️"
    )
    
    console.print(perf_table)
    
    # Display Patient Results
    console.print("\n[bold cyan]🔬 INDIVIDUAL PATIENT RESULTS[/bold cyan]")
    console.print("[dim]Per-patient matching outcomes and quality metrics[/dim]\n")
    
    patient_table = Table(show_header=True, title="Patient-Level Performance")
    patient_table.add_column("Patient", style="cyan")
    patient_table.add_column("Status", justify="center")
    patient_table.add_column("Trials", justify="right", style="green")
    patient_table.add_column("Relevant", justify="right", style="yellow")
    patient_table.add_column("Top Score", justify="right", style="yellow")
    patient_table.add_column("Avg Score", justify="right", style="dim")
    patient_table.add_column("Confidence", justify="right", style="dim")
    patient_table.add_column("Time", justify="right")
    
    for result in test_results:
        status_icon = "✅" if result['status'] == "SUCCESS" else "❌"
        patient_table.add_row(
            result['patient'],
            status_icon,
            str(result.get('trials_found', 0)),
            str(result.get('relevant_found', 0)),
            f"{result.get('top_score', 0):.2f}" if result.get('top_score') else "N/A",
            f"{result.get('avg_score', 0):.2f}" if result.get('avg_score') else "N/A",
            f"{result.get('confidence_avg', 0):.2f}" if result.get('confidence_avg') else "N/A",
            f"{result.get('time', 0):.2f}s"
        )
    
    console.print(patient_table)
    
    # Summary statistics
    success_count = len([r for r in test_results if r['status'] == "SUCCESS"])
    if success_count > 0:
        console.print(f"\n[dim]📈 Summary Statistics:[/dim]")
        avg_trials = np.mean([r.get('trials_found', 0) for r in test_results if r['status'] == "SUCCESS"])
        avg_relevant = np.mean([r.get('relevant_found', 0) for r in test_results if r['status'] == "SUCCESS"])
        avg_top_score = np.mean([r.get('top_score', 0) for r in test_results if r['status'] == "SUCCESS"])
        console.print(f"  • Average trials per patient: {avg_trials:.1f}")
        console.print(f"  • Average relevant trials: {avg_relevant:.1f}")
        console.print(f"  • Average top score: {avg_top_score:.3f}")
    
    # Calculate Overall Grade
    console.print("\n[bold cyan]🏆 OVERALL SYSTEM EVALUATION[/bold cyan]\n")
    
    # Component scores
    success_rate = len([r for r in test_results if r['status'] == "SUCCESS"]) / len(test_results)
    avg_trials = np.mean([r.get('trials_found', 0) for r in test_results])
    performance_score = 1.0 if avg_latency < 15 else (0.7 if avg_latency < 30 else 0.3)
    
    # Calculate overall score with detailed components
    if all_rankings and all_relevant:
        # Clinical effectiveness (40%)
        clinical_precision = (precision_3 * 0.2 + precision_5 * 0.3 + precision_10 * 0.5)
        clinical_recall = (recall_3 * 0.2 + recall_5 * 0.3 + recall_10 * 0.5)
        clinical_ranking = (ndcg_3 * 0.2 + ndcg_5 * 0.3 + ndcg_10 * 0.5)
        clinical_score = (clinical_precision * 0.35 + clinical_recall * 0.35 + clinical_ranking * 0.30)
        
        # Quality score (20%) - based on confidence and scores
        avg_confidence = np.mean([r.get('confidence_avg', 0.5) for r in test_results if r['status'] == "SUCCESS"])
        avg_score_quality = np.mean([r.get('avg_score', 0.5) for r in test_results if r['status'] == "SUCCESS"])
        quality_score = (avg_confidence * 0.5 + avg_score_quality * 0.5)
    else:
        clinical_score = 0.5  # Default if no rankings
        quality_score = 0.5
    
    overall_score = (
        clinical_score * 0.35 +     # Clinical effectiveness
        quality_score * 0.20 +       # Match quality
        success_rate * 0.25 +        # Success rate
        performance_score * 0.20     # Performance
    )
    
    # Determine grade
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
        interpretation = "Very Good - Ready for Pilot"
    elif overall_score >= 0.75:
        grade = "B"
        grade_color = "yellow"
        interpretation = "Good - Some Improvements Recommended"
    elif overall_score >= 0.7:
        grade = "C+"
        grade_color = "yellow"
        interpretation = "Satisfactory - Notable Areas for Improvement"
    else:
        grade = "C"
        grade_color = "yellow"
        interpretation = "Acceptable - Significant Improvements Needed"
    
    # Display grade with detailed breakdown
    grade_content = f"[bold {grade_color}]{grade}[/bold {grade_color}]\n\n"
    grade_content += f"{interpretation}\n\n"
    grade_content += f"[bold]Overall Score: {overall_score:.3f}/1.000[/bold]\n\n"
    grade_content += "[dim]Component Breakdown:[/dim]\n"
    grade_content += f"  Clinical Effectiveness (35%): {clinical_score:.3f}\n"
    if all_rankings and all_relevant:
        grade_content += f"    • Precision: {clinical_precision:.3f}\n"
        grade_content += f"    • Recall: {clinical_recall:.3f}\n"
        grade_content += f"    • Ranking Quality: {clinical_ranking:.3f}\n"
    grade_content += f"  Match Quality (20%): {quality_score:.3f}\n"
    if all_rankings and all_relevant:
        grade_content += f"    • Avg Confidence: {avg_confidence:.3f}\n"
        grade_content += f"    • Avg Score: {avg_score_quality:.3f}\n"
    grade_content += f"  Success Rate (25%): {success_rate:.1%}\n"
    grade_content += f"  Performance (20%): {performance_score:.2f}\n"
    grade_content += f"    • Avg Latency: {avg_latency:.1f}s (target <15s)\n"
    
    grade_panel = Panel(
        grade_content,
        title="[bold]System Grade[/bold]",
        border_style=grade_color,
        padding=(1, 2)
    )
    
    console.print(grade_panel)
    
    # Add recommendations based on scores
    console.print("\n[bold cyan]💡 RECOMMENDATIONS[/bold cyan]\n")
    recommendations = []
    
    if performance_score < 0.7:
        recommendations.append("⚠️  [yellow]Performance:[/yellow] Consider optimization - average response time exceeds 15s target")
    if clinical_recall < 0.5:
        recommendations.append("⚠️  [yellow]Recall:[/yellow] System missing relevant trials - review search/filtering logic")
    if clinical_precision < 0.7:
        recommendations.append("⚠️  [yellow]Precision:[/yellow] Too many irrelevant trials recommended - tighten matching criteria")
    if quality_score < 0.7:
        recommendations.append("⚠️  [yellow]Match Quality:[/yellow] Low confidence scores - review LLM prompts and scoring logic")
    
    if recommendations:
        for rec in recommendations:
            console.print(f"  {rec}")
    else:
        console.print("  ✅ [green]All metrics within acceptable ranges![/green]")
    
    # Show sample match
    if test_results and test_results[0].get('result', {}).get('matches'):
        console.print("\n[bold cyan]📋 SAMPLE MATCH DETAIL[/bold cyan]\n")
        result = test_results[0]['result']
        patient_summary = result.get('patient_summary', {})
        top_match = result['matches'][0]
        
        console.print(f"[bold]Patient {result['patient_id']}:[/bold]")
        console.print(f"  • Age: {patient_summary.get('age', 'Unknown')}")
        console.print(f"  • Cancer: {patient_summary.get('cancer_type', 'Unknown')}")
        console.print(f"  • Stage: {patient_summary.get('stage', 'Unknown')}")
        
        console.print(f"\n[green]Top Trial Match:[/green]")
        console.print(f"  • NCT ID: {top_match['nct_id']}")
        console.print(f"  • Score: {top_match['score']:.2f}")
        console.print(f"  • Confidence: {top_match['confidence']:.2f}")
        console.print(f"  • Title: {top_match['title'][:70]}...")
    
    console.print("\n" + "="*80)
    
    return test_results


async def run_evaluation(args):
    """Run evaluation based on arguments."""
    
    # Initialize evaluator with progress
    with tqdm(total=1, desc="Setting up evaluation", leave=False) as pbar:
        evaluator = Evaluator(output_dir=args.output_dir)
        pbar.update(1)
    
    # Determine what to evaluate
    metrics_to_run = set()
    if 'all' in args.metrics:
        metrics_to_run = {'clinical', 'equity', 'ensemble', 'performance', 'errors'}
    else:
        metrics_to_run = set(args.metrics)
    
    # Add additional analyses based on flags
    if args.show_agreement:
        metrics_to_run.add('ensemble')
    if args.top_errors > 0:
        metrics_to_run.add('errors')
    if args.ablation:
        metrics_to_run.add('ablation')
    
    # Get patient IDs
    if args.patient_ids:
        patient_ids = args.patient_ids
    else:
        patient_ids = [f"P{i:03d}" for i in range(1, args.n_patients + 1)]
    
    # Parse ablation configs
    ablation_configs = None
    if args.ablation:
        if args.configs:
            ablation_configs = {}
            for config_name in args.configs.split(','):
                if config_name == 'no_safety':
                    ablation_configs['no_safety'] = {'enable_safety_checks': False}
                elif config_name == 'no_llm':
                    ablation_configs['no_llm'] = {'use_llm': False}
                elif config_name == 'fast_mode':
                    ablation_configs['fast_mode'] = {'mode': 'fast'}
                elif config_name == 'rule_only':
                    ablation_configs['rule_only'] = {'use_rules_only': True}
        else:
            # Default ablation configs
            ablation_configs = {
                'no_safety': {'enable_safety_checks': False},
                'no_llm': {'use_llm': False},
                'fast_mode': {'mode': 'fast'}
            }
    
    # Run comprehensive evaluation
    await evaluator.run_comprehensive_evaluation(
        patient_ids=patient_ids,
        n_synthetic=args.n_synthetic,
        k_values=args.k_values,
        ablation_configs=ablation_configs,
        show_all=metrics_to_run
    )
    
    # Output JSON if requested
    if args.json:
        json_file = Path(args.output_dir) / f"detailed_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_file, 'w') as f:
            json.dump(evaluator.results, f, indent=2, default=str)
        
        if not args.quiet:
            print(f"\n📊 Detailed JSON results saved to: {json_file}")
    
    return evaluator.results


async def main():
    """Main entry point for the test suite."""
    
    # Parse arguments
    args = parse_arguments()
    
    # Check if any advanced options are specified
    has_advanced_options = (
        args.metrics != ['clinical'] or
        args.show_agreement or
        args.by_category or
        args.ablation or
        args.top_errors > 0
    )
    
    if has_advanced_options:
        # Run evaluation
        print("\n" + "=" * 60)
        print("🏆 Clinical Trial Matching Evaluation")
        print("=" * 60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if not args.quiet:
            print(f"\nConfiguration:")
            print(f"  Metrics: {', '.join(args.metrics)}")
            print(f"  Patients: {args.n_patients}")
            print(f"  K values: {args.k_values}")
            if args.show_agreement:
                print(f"  Judge agreement: Yes")
            if args.ablation:
                print(f"  Ablation studies: Yes")
            if args.top_errors:
                print(f"  Error analysis: Top {args.top_errors}")
        
        results = await run_evaluation(args)
        
        # Print final summary
        if 'clinical_metrics' in results and 'k_10' in results['clinical_metrics']:
            metrics = results['clinical_metrics']['k_10']
            print("\n" + "=" * 60)
            print("📊 KEY RESULTS (k=10)")
            print("=" * 60)
            print(f"nDCG:      {metrics['ndcg']:.3f}")
            print(f"Precision: {metrics['precision']:.3f}")
            print(f"Recall:    {metrics['recall']:.3f}")
            print(f"F1:        {metrics['f1']:.3f}")
            print(f"MRR:       {metrics['mrr']:.3f}")
            
            # Safety metrics
            if metrics.get('safety_violation_rate', 0) > 0:
                print(f"\n⚠️ Safety violation rate: {metrics['safety_violation_rate']:.2%}")
            if metrics.get('critical_miss_rate', 0) > 0:
                print(f"⚠️ Critical miss rate: {metrics['critical_miss_rate']:.2%}")
    else:
        # Run simple tests for backward compatibility
        results = await run_simple_tests()
    
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    print("Starting at", datetime.now().isoformat())
    asyncio.run(main())