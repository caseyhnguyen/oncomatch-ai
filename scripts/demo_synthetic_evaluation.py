#!/usr/bin/env python
"""
Demo: Comprehensive Evaluation with Synthetic Data

This script demonstrates the evaluation suite using a large synthetic cohort (1000+ patients)
that includes edge cases, rare mutations, and equity stress tests.

Run: python scripts/demo_synthetic_evaluation.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import asyncio
from datetime import datetime
from rich.console import Console
from rich.table import Table
from oncomatch.evaluation.synthetic_patients import SyntheticPatientGenerator, PatientCategory
from oncomatch.biomcp_wrapper import BioMCPWrapper
from oncomatch.optimized_ranker import OptimizedLLMRanker
import numpy as np

console = Console()

async def demo_synthetic_evaluation():
    """Run a demo evaluation with synthetic patients."""
    
    console.print("\n" + "="*70)
    console.print("[bold magenta]🧬 Synthetic Patient Evaluation Demo[/bold magenta]", justify="center")
    console.print("="*70 + "\n")
    
    # Step 1: Generate synthetic cohort
    console.print("[bold cyan]📊 Step 1: Generating Synthetic Cohort[/bold cyan]")
    console.print("[dim]Creating 1000 diverse oncology patients...[/dim]\n")
    
    start_time = datetime.now()
    generator = SyntheticPatientGenerator()
    
    # Generate comprehensive cohort
    cohort = generator.generate_cohort(
        n_patients=1000,
        category_distribution={
            PatientCategory.STANDARD: 0.60,      # ~600 - Realistic distributions
            PatientCategory.EDGE_CASE: 0.25,     # ~250 - Rare/extreme cases
            PatientCategory.ADVERSARIAL: 0.10,   # ~100 - Robustness tests
            PatientCategory.EQUITY_STRESS: 0.05  # ~50 - Underserved populations
        }
    )
    
    generation_time = (datetime.now() - start_time).total_seconds()
    
    console.print(f"[green]✅ Generated {len(cohort)} synthetic patients in {generation_time:.1f}s[/green]\n")
    
    # Step 2: Analyze cohort composition
    console.print("[bold cyan]📈 Step 2: Cohort Composition Analysis[/bold cyan]\n")
    
    # Count by category
    categories = {}
    cancer_types = {}
    age_groups = {"<18": 0, "18-64": 0, "65-79": 0, "≥80": 0}
    stages = {}
    biomarker_count = 0
    
    for patient in cohort:
        # Category
        cat = getattr(patient, 'category', 'standard')
        categories[cat] = categories.get(cat, 0) + 1
        
        # Cancer type
        cancer_types[patient.cancer_type] = cancer_types.get(patient.cancer_type, 0) + 1
        
        # Age
        if patient.age < 18:
            age_groups["<18"] += 1
        elif patient.age < 65:
            age_groups["18-64"] += 1
        elif patient.age < 80:
            age_groups["65-79"] += 1
        else:
            age_groups["≥80"] += 1
        
        # Stage
        if patient.cancer_stage:
            stages[str(patient.cancer_stage)] = stages.get(str(patient.cancer_stage), 0) + 1
        
        # Biomarkers
        if patient.biomarkers_detected:
            biomarker_count += len(patient.biomarkers_detected)
    
    # Display composition table
    table = Table(title="Synthetic Cohort Breakdown", show_header=True)
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right", style="green")
    table.add_column("Percentage", justify="right", style="yellow")
    
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        pct = (count / len(cohort)) * 100
        table.add_row(cat, str(count), f"{pct:.1f}%")
    
    console.print(table)
    console.print()
    
    # Top cancer types
    console.print("[bold]Top Cancer Types:[/bold]")
    for cancer, count in sorted(cancer_types.items(), key=lambda x: -x[1])[:5]:
        pct = (count / len(cohort)) * 100
        console.print(f"  • {cancer}: {count} ({pct:.1f}%)")
    console.print()
    
    # Age distribution
    console.print("[bold]Age Distribution:[/bold]")
    for age_group, count in age_groups.items():
        pct = (count / len(cohort)) * 100
        console.print(f"  • {age_group}: {count} ({pct:.1f}%)")
    console.print()
    
    # Biomarkers
    avg_biomarkers = biomarker_count / len(cohort)
    console.print(f"[bold]Biomarkers:[/bold] {biomarker_count} total ({avg_biomarkers:.1f} avg per patient)\n")
    
    # Step 3: Sample evaluation on 10 patients
    console.print("[bold cyan]🔬 Step 3: Sample Matching Evaluation (10 patients)[/bold cyan]")
    console.print("[dim]Demonstrating trial matching with synthetic patients...[/dim]\n")
    
    biomcp = BioMCPWrapper()
    ranker = OptimizedLLMRanker()
    
    sample_patients = cohort[:10]
    match_results = []
    
    for idx, patient in enumerate(sample_patients, 1):
        try:
            console.print(f"[cyan]{idx}/10[/cyan] Matching {patient.cancer_type}, Stage {patient.cancer_stage}, Age {patient.age}...")
            
            # Fetch trials
            trials = await biomcp.fetch_trials_for_patient(patient, max_trials=5)
            
            if trials:
                # Rank trials
                ranked_trials = await ranker.rank_trials_optimized(
                    patient=patient,
                    trials=trials,
                    use_batching=True,
                    use_cache=True
                )
                
                # Get top match
                if ranked_trials:
                    top_score = ranked_trials[0].overall_score
                    match_results.append(top_score)
                    console.print(f"  [green]✓ Top match score: {top_score:.3f}[/green]")
                else:
                    console.print(f"  [yellow]⚠ No rankings returned[/yellow]")
            else:
                console.print(f"  [yellow]⚠ No trials found[/yellow]")
        
        except Exception as e:
            console.print(f"  [red]✗ Error: {str(e)[:50]}[/red]")
    
    console.print()
    
    # Step 4: Summary statistics
    if match_results:
        console.print("[bold cyan]📊 Step 4: Match Quality Summary[/bold cyan]\n")
        
        console.print(f"[bold]Match Score Statistics:[/bold]")
        console.print(f"  • Mean: {np.mean(match_results):.3f}")
        console.print(f"  • Median: {np.median(match_results):.3f}")
        console.print(f"  • Std Dev: {np.std(match_results):.3f}")
        console.print(f"  • Min: {np.min(match_results):.3f}")
        console.print(f"  • Max: {np.max(match_results):.3f}")
        console.print()
    
    # Step 5: Key advantages
    console.print("[bold cyan]✨ Key Advantages of Synthetic Data[/bold cyan]\n")
    
    advantages = [
        ("Comprehensive Coverage", f"{len(cohort)} patients vs 30 real patients"),
        ("Edge Cases", f"{categories.get('edge_case', 0)} rare/extreme cases for robustness"),
        ("Adversarial Testing", f"{categories.get('adversarial', 0)} adversarial cases"),
        ("Equity Stress", f"{categories.get('equity_stress', 0)} underserved population cases"),
        ("Reproducibility", "Same cohort every run (when cached)"),
        ("Rare Mutations", "KRAS G12C, BRAF V600E, MSI-H, etc."),
        ("Age Extremes", f"Pediatric: {age_groups['<18']}, Elderly: {age_groups['≥80']}"),
        ("Epidemiologically Realistic", "Matches US cancer incidence rates")
    ]
    
    for advantage, detail in advantages:
        console.print(f"  [green]✓[/green] [bold]{advantage}:[/bold] {detail}")
    
    console.print()
    
    # Final summary
    console.print("="*70)
    console.print("[bold green]🎉 Synthetic Evaluation Demo Complete![/bold green]", justify="center")
    console.print("="*70)
    console.print()
    console.print("[dim]Next steps:[/dim]")
    console.print("[dim]  1. Run full evaluation: python scripts/run_full_evaluation.py[/dim]")
    console.print("[dim]  2. Test judge ensemble: python scripts/test_judge_ensemble.py[/dim]")
    console.print("[dim]  3. Check documentation: docs/SYNTHETIC_DATA_INTEGRATION.md[/dim]")
    console.print()

if __name__ == "__main__":
    console.print(f"[dim]Started at {datetime.now().isoformat()}[/dim]")
    asyncio.run(demo_synthetic_evaluation())
    console.print(f"[dim]Completed at {datetime.now().isoformat()}[/dim]")
