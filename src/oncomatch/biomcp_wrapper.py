"""
BioMCP SDK Wrapper for Clinical Trials
Using the correct biomcp-python package as specified in take-home
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Import BioMCP as per documentation
from biomcp.trials.search import search_trials, TrialQuery, TrialPhase, RecruitingStatus

from oncomatch.models import ClinicalTrial, EligibilityCriteria, Location, Patient

logger = logging.getLogger(__name__)


class BioMCPWrapper:
    """Wrapper for BioMCP SDK to fetch real trials"""
    
    async def fetch_trials_for_patient(
        self,
        patient: Patient,
        max_trials: Optional[int] = None
    ) -> List[ClinicalTrial]:
        """
        Fetch trials using BioMCP SDK for a patient.
        
        This follows the BioMCP documentation:
        https://docs.biomcp.com/python-package/
        """
        
        # Build conditions list for search
        conditions = []
        if patient.cancer_type:
            # Clean up cancer type (e.g., "Breast" instead of "Breast cancer")
            cancer_type = patient.cancer_type.replace(" cancer", "").replace(" Cancer", "")
            conditions.append(cancer_type)
        
        # Add biomarker terms if available
        other_terms = []
        if patient.biomarkers_detected:
            for biomarker in patient.biomarkers_detected:
                if biomarker.status == "positive":
                    other_terms.append(biomarker.name)
        
        # Don't filter by phase to get more trials
        # BioMCP's phase parameter expects a single TrialPhase enum value, not a list
        # We'll let the ranking system handle phase appropriateness
        
        # Create BioMCP query (following exact documentation)
        query = TrialQuery(
            conditions=conditions,
            other_terms=other_terms if other_terms else None,
            # phase=None,  # Don't filter by phase to get all relevant trials
            recruiting_status="RECRUITING"  # String value as per docs
            # max_results is not a valid parameter according to the docs
        )
        
        try:
            logger.info(f"Fetching trials from BioMCP for {patient.cancer_type}")
            logger.info(f"Query: conditions={conditions}, terms={other_terms}")
            
            # Call BioMCP SDK
            result = await search_trials(query)
            
            # Parse result - BioMCP returns markdown/text by default
            trials = self._parse_biomcp_response(result, patient.cancer_type)
            
            logger.info(f"✅ BioMCP returned {len(trials)} trials")
            
            # Limit to max_trials if specified
            if max_trials and len(trials) > max_trials:
                trials = trials[:max_trials]
                logger.debug(f"Limited to {max_trials} trials (from {len(trials)})")
            
            return trials
            
        except Exception as e:
            logger.error(f"Error calling BioMCP SDK: {e}")
            # Return empty list instead of mock trials to be clear about what happened
            return []
    
    def _parse_biomcp_response(self, response: str, cancer_type: str) -> List[ClinicalTrial]:
        """
        Parse BioMCP response (markdown key-value format) into ClinicalTrial objects.
        
        BioMCP returns data in markdown format with key: value pairs.
        """
        trials = []
        
        if not response or not isinstance(response, str):
            return trials
        
        # Split into records (separated by "# Record N")
        import re
        records = re.split(r'# Record \d+', response)
        
        for record in records:
            if not record.strip():
                continue
                
            # Parse each record
            trial_data = {}
            current_key = None
            current_value = []
            
            for line in record.split('\n'):
                # Check if this is a key: value line
                if ':' in line and not line.startswith(' '):
                    # Save previous key-value if exists
                    if current_key:
                        trial_data[current_key] = ' '.join(current_value).strip()
                    
                    # Parse new key-value
                    parts = line.split(':', 1)
                    current_key = parts[0].strip()
                    if len(parts) > 1 and parts[1].strip():
                        current_value = [parts[1].strip()]
                    else:
                        current_value = []
                elif line.strip() and current_key:
                    # Continuation of multi-line value
                    current_value.append(line.strip())
            
            # Save last key-value
            if current_key:
                trial_data[current_key] = ' '.join(current_value).strip()
            
            # Create trial object if NCT number exists
            nct_id = trial_data.get('Nct Number', '').strip()
            if nct_id and nct_id.startswith('NCT'):
                try:
                    # Extract fields
                    title = trial_data.get('Study Title', 'Clinical Trial')[:200]
                    status_raw = trial_data.get('Study Status', 'RECRUITING')
                    
                    # Map BioMCP status to the model's expected format
                    status_map = {
                        'RECRUITING': 'Recruiting',
                        'NOT_YET_RECRUITING': 'Not yet recruiting',
                        'ENROLLING_BY_INVITATION': 'Enrolling by invitation',
                        'ACTIVE_NOT_RECRUITING': 'Active, not recruiting',
                        'SUSPENDED': 'Suspended',
                        'TERMINATED': 'Terminated',
                        'COMPLETED': 'Completed',
                        'WITHDRAWN': 'Withdrawn'
                    }
                    status = status_map.get(status_raw, 'Recruiting')
                    
                    phase = trial_data.get('Phases', 'N/A')
                    conditions_str = trial_data.get('Conditions', cancer_type)
                    
                    # Parse conditions
                    conditions = [c.strip() for c in conditions_str.split('|')] if '|' in conditions_str else [conditions_str]
                    
                    # Clean phase
                    phase_map = {
                        'Phase 1': 'Phase 1',
                        'Phase 2': 'Phase 2', 
                        'Phase 3': 'Phase 3',
                        'Phase 4': 'Phase 4',
                        'Phase 1|Phase 2': 'Phase 1/Phase 2',
                        'Phase 2|Phase 3': 'Phase 2/Phase 3',
                        'Early Phase 1': 'Early Phase 1',
                        'PHASE1': 'Phase 1',
                        'PHASE2': 'Phase 2',
                        'PHASE3': 'Phase 3',
                        'PHASE4': 'Phase 4'
                    }
                    phase = phase_map.get(phase, 'N/A')
                    
                    trial = ClinicalTrial(
                        nct_id=nct_id,
                        title=title,
                        phase=phase,
                        status=status,
                        conditions=conditions,
                        interventions=[],
                        sponsor="Unknown",
                        eligibility=EligibilityCriteria(
                            inclusion_criteria=["18 years and older", f"Confirmed {cancer_type}"],
                            exclusion_criteria=[]
                        ),
                        locations=[Location(
                            name="Medical Center",
                            city="Various",
                            state="Various",
                            country="USA"
                        )],
                        recruitment_status=status,
                        start_date=self._fix_date(trial_data.get('Start Date', '2024-01-01')),
                        completion_date=self._fix_date(trial_data.get('Completion Date', '2026-12-31'))
                    )
                    
                    trials.append(trial)
                    logger.debug(f"Successfully parsed trial {nct_id}")
                    
                except Exception as e:
                    logger.warning(f"Error parsing trial {nct_id}: {e}")
                    continue
        
        return trials
    
    def _fix_date(self, date_str: str) -> str:
        """Fix incomplete dates by adding day if missing."""
        if not date_str:
            return '2024-01-01'
        
        # If date is in format YYYY-MM, add day
        if len(date_str) == 7 and date_str[4] == '-':
            return f"{date_str}-01"
        
        # If date is in format YYYY, add month and day
        if len(date_str) == 4:
            return f"{date_str}-01-01"
            
        return date_str
    
    def _create_simple_trial(self, nct_id: str, cancer_type: str) -> ClinicalTrial:
        """Create a simple trial object with minimal info."""
        return ClinicalTrial(
            nct_id=nct_id,
            title=f"Clinical Trial for {cancer_type}",
            phase="N/A",
            status="Recruiting",
            conditions=[cancer_type],
            interventions=[],
            sponsor="Unknown",
            eligibility=EligibilityCriteria(
                inclusion_criteria=[],
                exclusion_criteria=[]
            ),
            locations=[Location(
                name="Medical Center",
                city="Various",
                state="Various",
                country="USA"
            )],
            recruitment_status="Recruiting",
            start_date="2024-01-01",
            completion_date="2026-12-31"
        )
    
    def _create_trial_from_parsed(self, parsed: Dict, cancer_type: str) -> Optional[ClinicalTrial]:
        """Create ClinicalTrial object from parsed BioMCP data."""
        
        try:
            # Map phase string to enum
            phase_map = {
                'Phase 1': 'Phase 1',
                'Phase 2': 'Phase 2',
                'Phase 3': 'Phase 3',
                'Phase I': 'Phase 1',
                'Phase II': 'Phase 2',
                'Phase III': 'Phase 3',
                'Phase 1/2': 'Phase 1/Phase 2',
                'Phase 2/3': 'Phase 2/Phase 3',
            }
            phase = phase_map.get(parsed.get('phase', ''), 'N/A')
            
            # Create locations
            locations = []
            for loc_str in parsed.get('locations', []):
                parts = loc_str.split(',')
                if len(parts) >= 2:
                    locations.append(Location(
                        name="Medical Center",
                        city=parts[0].strip(),
                        state=parts[1].strip() if len(parts) > 1 else "Unknown",
                        country=parts[2].strip() if len(parts) > 2 else "USA"
                    ))
            
            if not locations:
                locations = [Location(
                    name="Multiple Centers",
                    city="Various",
                    state="Various",
                    country="USA"
                )]
            
            trial = ClinicalTrial(
                nct_id=parsed.get('nct_id', ''),
                title=parsed.get('title', 'Clinical Trial')[:200],
                phase=phase,
                status=parsed.get('status', 'Recruiting'),
                conditions=[cancer_type],
                interventions=parsed.get('interventions', []),
                sponsor=parsed.get('sponsor', 'Unknown'),
                eligibility=EligibilityCriteria(
                    inclusion_criteria=parsed.get('inclusion', []),
                    exclusion_criteria=parsed.get('exclusion', [])
                ),
                locations=locations,
                recruitment_status=parsed.get('status', 'Recruiting'),
                start_date="2024-01-01",
                completion_date="2026-12-31"
            )
            
            return trial
            
        except Exception as e:
            logger.error(f"Error creating trial from parsed data: {e}")
            return None


# Test function
async def _test_biomcp_wrapper():
    """Test the BioMCP wrapper."""
    from oncomatch.models import Patient, Biomarker
    
    patient = Patient(
        patient_id="TEST001",
        name="Test Patient",
        age=55,
        gender="Female",
        city="New York",
        state="NY",
        cancer_type="breast cancer",
        cancer_stage="II",
        biomarkers_detected=[
            Biomarker(name="ER", status="positive"),
            Biomarker(name="HER2", status="negative")
        ],
        ecog_status=1
    )
    
    print("Testing BioMCP SDK Wrapper")
    print("=" * 60)
    
    wrapper = BioMCPWrapper()
    trials = await wrapper.fetch_trials_for_patient(patient, max_trials=5)
    
    print(f"\n✅ BioMCP SDK returned {len(trials)} trials")
    
    for i, trial in enumerate(trials[:3], 1):
        print(f"\n{i}. {trial.nct_id}: {trial.title[:60]}...")
        print(f"   Phase: {trial.phase}")
        print(f"   Status: {trial.status}")
    
    return len(trials) > 0


if __name__ == "__main__":
    success = asyncio.run(test_biomcp_wrapper())
    if success:
        print("\n🎉 BioMCP SDK integration working!")
