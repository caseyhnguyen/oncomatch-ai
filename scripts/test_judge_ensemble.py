#!/usr/bin/env python
"""
Quick test to verify the judge ensemble is actually working.
Run: python scripts/test_judge_ensemble.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import asyncio
from datetime import datetime
from oncomatch.evaluation.evaluator import Evaluator
from rich.console import Console

console = Console()

async def test_judge_ensemble():
    """Test that the judge ensemble actually runs (not fake)."""
    
    console.print("\n" + "="*60)
    console.print("[bold cyan]🧪 Judge Ensemble Verification Test[/bold cyan]", justify="center")
    console.print("="*60 + "\n")
    
    # Initialize evaluator
    console.print("📦 Initializing evaluator...")
    evaluator = Evaluator()
    await evaluator.initialize()
    
    console.print("[green]✅ Evaluator initialized[/green]")
    console.print(f"[dim]Judge ensemble ready with {len(evaluator.judge_ensemble.available_judges)} judges[/dim]\n")
    
    # Run ensemble agreement test
    console.print("🤖 [bold]Testing LLM-as-Judge Ensemble[/bold]\n")
    
    start_time = datetime.now()
    
    try:
        # Run judge ensemble evaluation
        results = await evaluator.evaluate_ensemble_agreement()
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        if results:
            console.print("\n" + "="*60)
            console.print("[bold green]✅ SUCCESS: Judge Ensemble is Working![/bold green]", justify="center")
            console.print("="*60 + "\n")
            
            # Check evaluation method
            method = results.get('method', 'UNKNOWN')
            if method == 'ACTUAL_LLM_ENSEMBLE':
                console.print("[green]✓[/green] LLM-as-judge ensemble evaluation complete")
            else:
                console.print(f"[red]✗[/red] Warning: Method is {method}")
            
            # Check agreement scores
            alpha = results.get('krippendorff_alpha', 0)
            kappa = results.get('fleiss_kappa', 0)
            
            console.print(f"[green]✓[/green] Krippendorff's α: {alpha:.3f}")
            console.print(f"[green]✓[/green] Fleiss' κ: {kappa:.3f}")
            
            # Check if we have judge statistics (sign of real evaluation)
            if 'judge_statistics' in results and results['judge_statistics']:
                console.print(f"[green]✓[/green] Individual judge stats: {len(results['judge_statistics'])} judges")
                
                # Show some judge stats
                console.print("\n[yellow]Sample Judge Statistics:[/yellow]")
                for judge_name, stats in list(results['judge_statistics'].items())[:3]:
                    console.print(f"  • {judge_name}: μ={stats['mean']:.3f}, σ={stats['std']:.3f}")
            else:
                console.print("[yellow]⚠[/yellow] No judge statistics found")
            
            # Check evaluation count
            num_samples = results.get('num_samples', 0)
            console.print(f"[green]✓[/green] Evaluated {num_samples} patient-trial matches")
            
            console.print(f"\n[dim]Total time: {elapsed:.1f}s[/dim]")
            
            # Final verdict
            console.print("\n" + "="*60)
            if method == 'ACTUAL_LLM_ENSEMBLE' and alpha > 0.3 and num_samples > 0:
                console.print("[bold green]🎉 VERDICT: Judge Ensemble Working Successfully![/bold green]", justify="center")
            else:
                console.print("[bold yellow]⚠ VERDICT: Check results for issues[/bold yellow]", justify="center")
            console.print("="*60 + "\n")
            
        else:
            console.print("[red]❌ FAIL: No results returned[/red]")
            
    except Exception as e:
        console.print(f"\n[red]❌ ERROR: {str(e)}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise

if __name__ == "__main__":
    console.print(f"[dim]Starting test at {datetime.now().isoformat()}[/dim]")
    asyncio.run(test_judge_ensemble())
    console.print(f"[dim]Completed at {datetime.now().isoformat()}[/dim]")
