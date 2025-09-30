#!/usr/bin/env python
"""
OncoMatch AI - Clinical Trial Matching CLI
"""

import asyncio
import json
import argparse
from typing import Dict, Any, List, Optional
from datetime import datetime
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm
import pandas as pd
from pathlib import Path
import logging
import sys

# Import OncoMatch components
from oncomatch.models import Patient
from oncomatch.biomcp_client import BioMCPClient
from oncomatch.data_processor import DataProcessor
from oncomatch.trial_analyzer import TrialAnalyzer
from oncomatch.llm_ranker import LLMRanker
from oncomatch.optimized_ranker import OptimizedLLMRanker, get_optimized_ranker
from oncomatch.config import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClinicalTrialMatcher:
    """
    Main orchestrator for clinical trial matching
    """
    
    def __init__(self, patients_file: str = "patients.csv", use_optimized: bool = True):
        """Initialize the matcher with all components
        
        Args:
            patients_file: Path to patient CSV file
            use_optimized: Use optimized LLM ranker for <15s performance
        """
        
        # Validate configuration
        if not Config.validate():
            logger.warning("Configuration validation warnings - some features may be limited")
        
        # Initialize components
        self.biomcp_client = BioMCPClient()
        self.data_processor = DataProcessor()
        self.trial_analyzer = TrialAnalyzer()
        
        # Choose ranker based on optimization flag
        self.use_optimized = use_optimized
        if use_optimized:
            logger.info("🚀 Using OPTIMIZED ranker for <15s performance")
            self.llm_ranker = get_optimized_ranker()
        else:
            logger.info("Using standard ranker")
            self.llm_ranker = LLMRanker()
        
        # Load patient data
        self.patients_df = self._load_patients(patients_file)
        
        logger.info(f"Initialized matcher with {len(self.patients_df)} patients")
        logger.info(f"Available providers: {Config.get_available_providers()}")
    
    def _load_patients(self, patients_file: str) -> pd.DataFrame:
        """Load patient data from CSV"""
        try:
            df = pd.read_csv(patients_file)
            
            # Add patient_id column if not present (P001, P002, etc.)
            if 'patient_id' not in df.columns:
                df['patient_id'] = [f'P{i:03d}' for i in range(1, len(df) + 1)]
            
            logger.info(f"Loaded {len(df)} patients from {patients_file}")
            return df
            
        except FileNotFoundError:
            logger.error(f"Patient file {patients_file} not found")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading patients: {e}")
            return pd.DataFrame()
    
    async def match_patient(
        self, 
        patient_id: str, 
        max_trials: Optional[int] = None, 
        mode: str = "balanced"
    ) -> Optional[Dict[str, Any]]:
        """
        Match a single patient to clinical trials
        
        Args:
            patient_id: Patient identifier (name or ID)
            max_trials: Maximum number of trials to fetch (None = all available)
            mode: Matching mode (fast, balanced, accurate)
        
        Returns:
            Match results dictionary
        """
        
        # Find patient in dataframe
        patient_row = None
        if patient_id in self.patients_df['patient_id'].values:
            patient_row = self.patients_df[self.patients_df['patient_id'] == patient_id].iloc[0]
        elif 'name' in self.patients_df.columns and patient_id in self.patients_df['name'].values:
            patient_row = self.patients_df[self.patients_df['name'] == patient_id].iloc[0]
        
        if patient_row is None:
            logger.error(f"Patient '{patient_id}' not found")
            return None
        
        logger.info(f"Processing patient: {patient_id}")
        start_time = datetime.now()
        
        try:
            # Convert to Patient model
            patient_dict = patient_row.to_dict()
            patient = self._create_patient_model(patient_dict)
            
            # Process patient data (use the same patient object)
            # processed_data = self.data_processor.process_patient(patient_dict)
            logger.info(f"Processed patient data: {patient.cancer_type}, Stage {patient.cancer_stage}")
            
            # Fetch trials from BioMCP
            logger.info(f"Fetching trials for {patient.cancer_type}...")
            trials = await self.biomcp_client.fetch_trials_for_patient(
                patient=patient,
                max_trials=max_trials if max_trials else 100  # Default to 100 if not specified
            )
            
            if not trials:
                logger.warning("No trials found for patient")
                return {
                    "patient_id": patient_id,
                    "status": "no_trials_found",
                    "trials": []
                }
            
            logger.info(f"Found {len(trials)} potential trials")
            
            # Determine actual number of trials to process
            trials_to_process = trials[:max_trials] if max_trials else trials
            
            # Analyze trials in parallel (but keep the original trials for ranking)
            analysis_tasks = [
                self.trial_analyzer.analyze_trial(trial) 
                for trial in trials_to_process
            ]
            
            # Process in batches to avoid overwhelming the system
            batch_size = 20  # Increased from sequential to 20 parallel
            trial_analyses = []
            
            for i in range(0, len(analysis_tasks), batch_size):
                batch = analysis_tasks[i:i + batch_size]
                batch_results = await asyncio.gather(*batch, return_exceptions=True)
                
                # Handle any exceptions gracefully
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.warning(f"Trial analysis failed: {result}")
                        # Create a basic analysis for failed trials
                        trial_analyses.append({
                            "nct_id": trials_to_process[i + j].nct_id,
                            "error": str(result)
                        })
                    else:
                        trial_analyses.append(result)
            
            # Rank trials using LLM (pass actual trial objects, not analyses)
            logger.info("Ranking trials with LLM...")
            if self.use_optimized:
                # Use optimized ranker for <15s performance
                rankings = await self.llm_ranker.rank_trials_optimized(
                    patient=patient,
                    trials=trials_to_process,
                    use_batching=True,
                    use_cache=True
                )
            else:
                # Use standard ranker
                with tqdm(total=1, desc="LLM Ranking", leave=False) as pbar:
                    rankings = await self.llm_ranker.rank_multiple_trials(
                        patient=patient,
                        trials=trials_to_process,
                        parallel=True
                    )
                    pbar.update(1)
            
            # Calculate metrics
            elapsed_time = (datetime.now() - start_time).total_seconds()
            
            # Format results
            formatted_rankings = []
            # Create a lookup dict for trials
            trials_dict = {trial.nct_id: trial for trial in trials}
            
            for ranking in rankings:
                trial = trials_dict.get(ranking.nct_id)
                if trial:
                    formatted_rankings.append({
                        "nct_id": ranking.nct_id,
                        "title": trial.title,
                        "score": ranking.overall_score,
                        "confidence": ranking.confidence,
                        "phase": trial.phase,
                        "reasons": [r.explanation for r in ranking.match_reasons]  # Show all reasons
                    })
            
            result = {
                "patient_id": patient_id,
                "patient_name": patient_dict.get('name', patient_id),
                "timestamp": datetime.now().isoformat(),
                "mode": mode,
                "trials_fetched": len(trials),
                "trials_ranked": len(rankings),
                "processing_time_seconds": elapsed_time,
                "matches": formatted_rankings,
                "trials": trials_to_process,  # Cache trials to avoid re-fetching for judge eval
                "patient_summary": {
                    "age": patient.age,
                    "cancer_type": patient.cancer_type,
                    "stage": patient.cancer_stage,
                    "biomarkers": [f"{b.name}:{b.status}" for b in patient.biomarkers_detected]
                }
            }
            
            logger.info(f"Matching complete in {elapsed_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error matching patient: {e}")
            return {
                "patient_id": patient_id,
                "status": "error",
                "error": str(e)
            }
    
    def _create_patient_model(self, patient_dict: Dict) -> Patient:
        """Create a Patient model from dictionary"""
        import pandas as pd
        
        # Handle NaN values and convert strings to lists where needed
        def handle_list_field(value):
            if pd.isna(value) or value == "" or value is None:
                return []
            if isinstance(value, str):
                return [item.strip() for item in value.split(',')]
            if isinstance(value, list):
                return value
            return []
        
        def handle_string_field(value, default=""):
            if pd.isna(value) or value is None:
                return default
            return str(value)
        
        # Handle biomarkers as Biomarker objects
        def create_biomarker_objects(value):
            from oncomatch.models import Biomarker
            biomarkers = []
            if pd.isna(value) or value == "" or value is None:
                return biomarkers
            if isinstance(value, str):
                for item in value.split(','):
                    item = item.strip()
                    if item:
                        # Create Biomarker object
                        biomarkers.append(Biomarker(
                            name=item.replace('+', '').replace('-', ''),
                            status="positive" if '+' in item else "negative" if '-' in item else "unknown",
                            value=""
                        ))
            return biomarkers
        
        # Map CSV columns to Patient model fields with proper handling
        mapped = {
            "patient_id": patient_dict.get("patient_id", patient_dict.get("name", "Unknown")),
            "name": handle_string_field(patient_dict.get("name"), "Unknown"),
            "age": patient_dict.get("age", 50) if not pd.isna(patient_dict.get("age")) else 50,
            "gender": handle_string_field(patient_dict.get("gender"), "Unknown"),
            "race": handle_string_field(patient_dict.get("race"), "Unknown"),
            "city": handle_string_field(patient_dict.get("city"), "Unknown"),
            "state": handle_string_field(patient_dict.get("state"), "Unknown"),
            "cancer_type": handle_string_field(patient_dict.get("cancer_type"), "Unknown"),
            "cancer_stage": handle_string_field(patient_dict.get("cancer_stage"), "Unknown"),
            "cancer_substage": handle_string_field(patient_dict.get("cancer_substage"), ""),
            "biomarkers_detected": create_biomarker_objects(patient_dict.get("biomarkers_detected")),
            "biomarkers_ruled_out": handle_list_field(patient_dict.get("biomarkers_ruled_out")),
            "ecog_status": patient_dict.get("ecog_status", 0) if not pd.isna(patient_dict.get("ecog_status")) else 0,
            "previous_treatments": handle_list_field(patient_dict.get("previous_treatments")),
            "current_medications": handle_list_field(patient_dict.get("current_medications")),
            "other_conditions": handle_list_field(patient_dict.get("other_conditions")),  # Changed to list
            "smoking_status": handle_string_field(patient_dict.get("smoking_status"), "Unknown"),
            "family_history": handle_string_field(patient_dict.get("family_history"), "")
        }
        
        return Patient(**mapped)
    
    async def match_batch(
        self,
        patient_ids: Optional[List[str]] = None,
        max_trials: Optional[int] = None,
        mode: str = "balanced"
    ) -> List[Dict[str, Any]]:
        """
        Match multiple patients in batch
        
        Args:
            patient_ids: List of patient IDs (None = all patients)
            max_trials: Maximum trials per patient
            mode: Matching mode
        
        Returns:
            List of match results
        """
        
        if patient_ids is None:
            patient_ids = self.patients_df['patient_id'].tolist()
        
        logger.info(f"Batch matching {len(patient_ids)} patients")
        
        results = []
        # Progress bar for batch processing
        for patient_id in tqdm(patient_ids, desc="Matching patients"):
            result = await self.match_patient(patient_id, max_trials, mode)
            if result:
                results.append(result)
        
        return results


def format_human_output(result: Dict[str, Any]) -> str:
    """Format results for human reading"""
    
    output = []
    output.append("=" * 60)
    output.append(f"Clinical Trial Matches for {result.get('patient_name', result['patient_id'])}")
    output.append("=" * 60)
    
    # Patient summary
    summary = result.get('patient_summary', {})
    output.append(f"\nPatient Summary:")
    output.append(f"  Age: {summary.get('age')}")
    output.append(f"  Cancer: {summary.get('cancer_type')} Stage {summary.get('stage')}")
    output.append(f"  Biomarkers: {', '.join(summary.get('biomarkers', []))}")
    
    # Processing info
    output.append(f"\nProcessing:")
    output.append(f"  Mode: {result.get('mode', 'balanced')}")
    output.append(f"  Trials fetched: {result.get('trials_fetched', 0)}")
    output.append(f"  Time: {result.get('processing_time_seconds', 0):.2f}s")
    
    # Top matches
    output.append(f"\nTop Clinical Trial Matches:")
    output.append("-" * 40)
    
    matches = result.get('matches', [])
    if not matches:
        output.append("No matches found")
    else:
        for i, match in enumerate(matches[:10], 1):  # Show top 10 matches
            output.append(f"\n{i}. {match.get('nct_id', 'Unknown')}")
            output.append(f"   Score: {match.get('score', 0):.2f}")
            
            # Display full title without truncation
            title = match.get('title', 'No title')
            output.append(f"   Title: {title}")
            
            # Display all reasons without truncation
            reasons = match.get('reasons', [])
            if reasons:
                output.append(f"   Key Matches:")
                for reason in reasons:  # Show all reasons, not just first 2
                    output.append(f"     • {reason}")  # Full reason without truncation
    
    output.append("\n" + "=" * 60)
    
    return "\n".join(output)


async def main():
    """Main CLI entry point"""
    
    parser = argparse.ArgumentParser(
        description="OncoMatch AI - Clinical Trial Matching System",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "--patient_id",
        type=str,
        help="Patient ID or name to match"
    )
    
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run batch matching for all patients"
    )
    
    parser.add_argument(
        "--format",
        choices=["human", "json"],
        default="human",
        help="Output format (default: human)"
    )
    
    parser.add_argument(
        "--mode",
        choices=["fast", "balanced", "accurate"],
        default="balanced",
        help="Matching mode:\n"
             "  fast: Quick results with lower accuracy\n"
             "  balanced: Good balance of speed and accuracy (default)\n"
             "  accurate: Highest accuracy but slower"
    )
    
    parser.add_argument(
        "--max_trials",
        type=int,
        default=None,
        help="Maximum number of trials to process (default: all available, up to 100)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (JSON format)"
    )
    
    parser.add_argument(
        "--optimized",
        action="store_true",
        default=True,
        help="Use optimized ranker for <15s performance (default: True)"
    )
    
    parser.add_argument(
        "--no-optimized",
        action="store_false",
        dest="optimized",
        help="Disable optimized ranker (use standard ranker)"
    )
    
    parser.add_argument(
        "--patients_file",
        type=str,
        default="patients.csv",
        help="Path to patients CSV file (default: patients.csv)"
    )
    
    args = parser.parse_args()
    
    # Initialize matcher with progress indicator
    with tqdm(total=3, desc="Initializing", bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}') as init_pbar:
        init_pbar.set_description("Loading components")
        matcher = ClinicalTrialMatcher(args.patients_file, use_optimized=args.optimized)
        init_pbar.update(1)
        
        init_pbar.set_description("Validating data")
        if not len(matcher.patients_df):
            print("Error: No patients loaded")
            sys.exit(1)
        init_pbar.update(1)
        
        init_pbar.set_description("Ready")
        init_pbar.update(1)
    
    # Run matching
    if args.batch:
        results = await matcher.match_batch(
            max_trials=args.max_trials,
            mode=args.mode
        )
    elif args.patient_id:
        # Single patient matching with overall progress
        with tqdm(total=4, desc="Matching Progress", 
                  bar_format='{desc}: {percentage:3.0f}%|{bar}| Step {n_fmt}/{total_fmt}', 
                  leave=True) as main_pbar:
            main_pbar.set_description("Processing patient")
            main_pbar.update(1)
            
            main_pbar.set_description("Fetching trials")
            main_pbar.update(1)
            
            result = await matcher.match_patient(
                args.patient_id,
                max_trials=args.max_trials,
                mode=args.mode
            )
            main_pbar.set_description("Ranking trials")
            main_pbar.update(1)
            
            main_pbar.set_description("Complete")
            main_pbar.update(1)
            
        results = [result] if result else []
    else:
        parser.print_help()
        print("\nExample usage:")
        print('  python -m oncomatch.match --patient_id "Kerry Bird" --format human')
        print('  python -m oncomatch.match --batch --output results.json')
        sys.exit(0)
    
    # Output results
    if not results:
        print("No results generated")
        sys.exit(1)
    
    # Format output
    if args.format == "json" or args.output:
        output = json.dumps(results[0] if len(results) == 1 else results, indent=2)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"Results saved to {args.output}")
        else:
            print(output)
    else:
        # Human-readable output
        for result in results:
            if result:
                print(format_human_output(result))


if __name__ == "__main__":
    asyncio.run(main())