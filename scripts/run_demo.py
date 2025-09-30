#!/usr/bin/env python3
"""
OncoMatch-AI Demo Script
Run a simple demonstration of the clinical trial matching system
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.oncomatch.match import ClinicalTrialMatcher


async def run_demo():
    """Run demo matching for a single patient"""
    print("=" * 60)
    print("🧬 OncoMatch-AI Clinical Trial Matching Demo")
    print("=" * 60)
    
    # Initialize matcher
    matcher = ClinicalTrialMatcher()
    
    # Match patient P002 (33yo female with breast cancer)
    patient_id = "P002"
    print(f"\nMatching patient {patient_id}...")
    
    result = await matcher.match_patient(
        patient_id=patient_id,
        max_trials=10,
        mode="balanced"
    )
    
    if result:
        # Display results
        print(f"\n✅ Successfully matched patient {result['patient_id']}")
        print(f"   Name: {result.get('patient_name', 'Unknown')}")
        print(f"   Trials fetched: {result['trials_fetched']}")
        print(f"   Processing time: {result['processing_time_seconds']:.2f}s")
        
        # Show top matches
        print("\n📊 Top Trial Matches:")
        print("-" * 40)
        
        matches = result.get('matches', [])
        for i, match in enumerate(matches[:5], 1):
            print(f"\n{i}. {match['nct_id']}")
            print(f"   Score: {match['score']:.2f}")
            print(f"   Confidence: {match['confidence']:.2f}")
            print(f"   Title: {match['title'][:60]}...")
            
            reasons = match.get('reasons', [])
            if reasons:
                print(f"   Key Matches:")
                for reason in reasons[:2]:
                    print(f"     • {reason[:70]}...")
        
        # Save full results
        from pathlib import Path
        output_dir = Path("outputs/demos")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"demo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Full results saved to {output_file}")
        
    else:
        print("❌ Failed to match patient")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_demo())
