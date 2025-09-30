#!/usr/bin/env python3
"""
OncoMatch-AI Comprehensive Evaluation Script
Run full evaluation suite on the clinical trial matching system
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.oncomatch.match import ClinicalTrialMatcher
from src.oncomatch.evaluation.synthetic_patients import AdvancedSyntheticPatientGenerator, PatientCategory
from src.oncomatch.evaluation.metrics import AggregateMetrics
from src.oncomatch.biomcp_wrapper import BioMCPWrapper


class EvaluationRunner:
    """Run comprehensive evaluation of the matching system"""
    
    def __init__(self):
        self.matcher = ClinicalTrialMatcher()
        self.metrics_calculator = AggregateMetrics()
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "metrics": {},
            "summary": {}
        }
    
    async def evaluate_real_patients(self, num_patients: int = 5) -> Dict[str, Any]:
        """Evaluate matching for real patients from CSV"""
        print("\n📊 Evaluating Real Patients")
        print("-" * 40)
        
        patient_ids = [f"P{i:03d}" for i in range(1, num_patients + 1)]
        rankings = []
        latencies = []
        matched_patients = []
        
        for patient_id in patient_ids:
            print(f"Matching patient {patient_id}...")
            
            start_time = asyncio.get_event_loop().time()
            result = await self.matcher.match_patient(patient_id, max_trials=10)
            latency = asyncio.get_event_loop().time() - start_time
            latencies.append(latency)
            
            if result and result.get('matches'):
                rankings.extend(result['matches'])
                matched_patients.append(patient_id)
                print(f"  ✅ Found {len(result['matches'])} matches in {latency:.2f}s")
            else:
                print(f"  ❌ No matches found")
        
        # Calculate metrics
        relevant_ids = [r['nct_id'] for r in rankings if r.get('score', 0) > 0.7]
        relevance_scores = {r['nct_id']: r.get('score', 0) for r in rankings}
        
        metrics = self.metrics_calculator.calculate_all_metrics(
            rankings=rankings,
            relevant_ids=relevant_ids,
            relevance_scores=relevance_scores,
            latencies=latencies
        )
        
        return {
            "patients_tested": len(patient_ids),
            "successful_matches": len(matched_patients),
            "total_trials": len(rankings),
            "metrics": metrics
        }
    
    async def evaluate_synthetic_patients(self, num_patients: int = 20) -> Dict[str, Any]:
        """Evaluate matching for synthetic patients"""
        print("\n🧬 Evaluating Synthetic Patients")
        print("-" * 40)
        
        # Generate synthetic patients
        generator = AdvancedSyntheticPatientGenerator()
        patients = generator.generate_cohort(
            n_patients=num_patients,
            category_distribution={
                PatientCategory.STANDARD: 0.6,
                PatientCategory.EDGE_CASE: 0.2,
                PatientCategory.ADVERSARIAL: 0.1,
                PatientCategory.EQUITY_STRESS: 0.1
            }
        )
        
        print(f"Generated {len(patients)} synthetic patients")
        
        # Test matching
        category_results = {}
        wrapper = BioMCPWrapper()
        
        for patient in patients:
            # Determine category
            category = "standard"
            if "EDGE" in patient.patient_id:
                category = "edge_case"
            elif "ADVERSARIAL" in patient.patient_id:
                category = "adversarial"
            elif "EQUITY" in patient.patient_id:
                category = "equity"
            
            if category not in category_results:
                category_results[category] = {"total": 0, "matched": 0}
            
            category_results[category]["total"] += 1
            
            # Quick trial fetch
            trials = await wrapper.fetch_trials_for_patient(patient, max_trials=5)
            if trials:
                category_results[category]["matched"] += 1
        
        return {
            "total_patients": len(patients),
            "category_breakdown": category_results,
            "overall_match_rate": sum(c["matched"] for c in category_results.values()) / len(patients)
        }
    
    async def evaluate_biomcp_integration(self) -> Dict[str, Any]:
        """Test BioMCP integration with various cancer types"""
        print("\n🔬 Evaluating BioMCP Integration")
        print("-" * 40)
        
        wrapper = BioMCPWrapper()
        cancer_types = ["Breast", "Lung", "Melanoma", "Prostate", "Colorectal"]
        results = {}
        
        for cancer_type in cancer_types:
            from src.oncomatch.models import Patient
            patient = Patient(
                patient_id=f"TEST_{cancer_type}",
                name="Test Patient",
                age=55,
                gender="Female" if cancer_type == "Breast" else "Male",
                city="New York",
                state="NY",
                cancer_type=cancer_type,
                cancer_stage="II"
            )
            
            trials = await wrapper.fetch_trials_for_patient(patient, max_trials=10)
            results[cancer_type] = len(trials)
            print(f"  {cancer_type}: {len(trials)} trials found")
        
        return {
            "cancer_types_tested": len(cancer_types),
            "total_trials": sum(results.values()),
            "breakdown": results,
            "using_real_data": sum(results.values()) > 0
        }
    
    async def run_full_evaluation(self):
        """Run complete evaluation suite"""
        print("\n" + "=" * 60)
        print("🧪 OncoMatch-AI Comprehensive Evaluation")
        print("=" * 60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run evaluations
        self.results["tests"]["real_patients"] = await self.evaluate_real_patients()
        self.results["tests"]["synthetic_patients"] = await self.evaluate_synthetic_patients()
        self.results["tests"]["biomcp_integration"] = await self.evaluate_biomcp_integration()
        
        # Generate summary
        self.generate_summary()
        
        # Save results
        self.save_results()
        
        # Display summary
        self.display_summary()
    
    def generate_summary(self):
        """Generate evaluation summary"""
        tests = self.results["tests"]
        
        # Calculate overall metrics
        total_patients = (
            tests["real_patients"]["patients_tested"] +
            tests["synthetic_patients"]["total_patients"]
        )
        
        real_match_rate = (
            tests["real_patients"]["successful_matches"] /
            tests["real_patients"]["patients_tested"]
        )
        
        synthetic_match_rate = tests["synthetic_patients"]["overall_match_rate"]
        
        self.results["summary"] = {
            "total_patients_tested": total_patients,
            "real_patient_match_rate": f"{real_match_rate * 100:.1f}%",
            "synthetic_patient_match_rate": f"{synthetic_match_rate * 100:.1f}%",
            "biomcp_working": tests["biomcp_integration"]["using_real_data"],
            "overall_grade": self._calculate_grade(real_match_rate, synthetic_match_rate)
        }
    
    def _calculate_grade(self, real_rate: float, synthetic_rate: float) -> str:
        """Calculate overall system grade"""
        avg_rate = (real_rate + synthetic_rate) / 2
        if avg_rate >= 0.9:
            return "A"
        elif avg_rate >= 0.8:
            return "B"
        elif avg_rate >= 0.7:
            return "C"
        elif avg_rate >= 0.6:
            return "D"
        else:
            return "F"
    
    def save_results(self):
        """Save evaluation results to file"""
        output_dir = Path("outputs/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 Results saved to {filename}")
    
    def display_summary(self):
        """Display evaluation summary"""
        summary = self.results["summary"]
        
        print("\n" + "=" * 60)
        print("📈 EVALUATION SUMMARY")
        print("=" * 60)
        
        print(f"\nTotal Patients Tested: {summary['total_patients_tested']}")
        print(f"Real Patient Match Rate: {summary['real_patient_match_rate']}")
        print(f"Synthetic Patient Match Rate: {summary['synthetic_patient_match_rate']}")
        print(f"BioMCP Integration: {'✅ Working' if summary['biomcp_working'] else '❌ Not Working'}")
        print(f"\nOverall Grade: {summary['overall_grade']}")
        
        print("\n" + "=" * 60)
        if summary['overall_grade'] in ['A', 'B']:
            print("✅ SYSTEM EVALUATION: PASSED")
        else:
            print("❌ SYSTEM EVALUATION: NEEDS IMPROVEMENT")
        print("=" * 60)


async def main():
    """Main entry point"""
    runner = EvaluationRunner()
    await runner.run_full_evaluation()


if __name__ == "__main__":
    asyncio.run(main())
