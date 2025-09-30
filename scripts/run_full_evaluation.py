#!/usr/bin/env python
"""
Full Evaluation Suite with Synthetic Data and Judge Ensemble

This script runs the comprehensive evaluation with:
- 1000 synthetic patients (standard, edge, adversarial, equity stress)
- Clinical metrics (nDCG, Precision, Recall, MRR, Safety violations)
- Equity analysis (subgroup performance, biomarker fairness)
- LLM-as-Judge ensemble (7 specialized judges)

Run: python scripts/run_full_evaluation.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import asyncio
from datetime import datetime
from oncomatch.evaluation.evaluator import Evaluator
from rich.console import Console

console = Console()

async def main():
    """Run comprehensive evaluation with synthetic data."""
    
    console.print("\n" + "="*80)
    console.print("[bold magenta]🚀 OncoMatch-AI: Full Evaluation Suite[/bold magenta]", justify="center")
    console.print("="*80 + "\n")
    
    console.print("[cyan]This evaluation will:[/cyan]")
    console.print("  1. Generate 1000 diverse synthetic patients")
    console.print("  2. Evaluate clinical metrics (nDCG, Precision, Recall, MRR)")
    console.print("  3. Analyze equity across subgroups")
    console.print("  4. Run 7-judge LLM ensemble for quality validation")
    console.print(f"\n[dim]Estimated time: 30-60 minutes[/dim]")
    console.print(f"[dim]Results will be saved to: outputs/results/[/dim]\n")
    
    # Create evaluator
    evaluator = Evaluator(output_dir="outputs/results")
    
    # Run comprehensive evaluation with synthetic data
    await evaluator.run_comprehensive_evaluation(
        use_synthetic=True,      # Use synthetic patients (recommended)
        n_synthetic=1000,        # 1000 diverse patients
        k_values=[5, 10, 20],    # Top-k metrics
        show_all=True            # Run all evaluation types
    )
    
    console.print("\n" + "="*80)
    console.print("[bold green]✅ Evaluation Complete![/bold green]", justify="center")
    console.print("="*80 + "\n")
    console.print("[cyan]Results saved to: outputs/results/[/cyan]")
    console.print("[dim]Check the generated JSON and visualizations[/dim]\n")

if __name__ == "__main__":
    start_time = datetime.now()
    console.print(f"[dim]Started at {start_time.isoformat()}[/dim]\n")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Evaluation interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Error: {str(e)}[/red]")
        raise
    
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    console.print(f"[dim]Completed at {end_time.isoformat()} (elapsed: {elapsed/60:.1f} min)[/dim]")
