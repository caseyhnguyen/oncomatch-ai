"""
Real Clinical Trial Fetcher using ClinicalTrials.gov API directly
Since BioMCP has import issues, we'll use the direct API
"""

import asyncio
import aiohttp
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import os

from oncomatch.models import ClinicalTrial, EligibilityCriteria, Location, Patient

logger = logging.getLogger(__name__)


class StudyType(str, Enum):
    """ClinicalTrials.gov study types"""
    INTERVENTIONAL = "INTERVENTIONAL"
    OBSERVATIONAL = "OBSERVATIONAL"
    EXPANDED_ACCESS = "EXPANDED_ACCESS"


class RecruitmentStatus(str, Enum):
    """ClinicalTrials.gov recruitment status"""
    RECRUITING = "RECRUITING"
    NOT_YET_RECRUITING = "NOT_YET_RECRUITING"
    ENROLLING_BY_INVITATION = "ENROLLING_BY_INVITATION"
    ACTIVE_NOT_RECRUITING = "ACTIVE_NOT_RECRUITING"
    COMPLETED = "COMPLETED"
    SUSPENDED = "SUSPENDED"
    TERMINATED = "TERMINATED"
    WITHDRAWN = "WITHDRAWN"


@dataclass
class TrialSearchQuery:
    """Query parameters for ClinicalTrials.gov API v2"""
    condition: Optional[str] = None
    term: Optional[str] = None
    status: Optional[List[str]] = None
    location: Optional[str] = None
    intervention: Optional[str] = None
    phase: Optional[List[str]] = None
    study_type: Optional[str] = None
    page_size: int = 20
    
    def to_params(self) -> Dict[str, Any]:
        """Convert to API parameters."""
        params = {
            "format": "json",
            "pageSize": str(self.page_size),
            "countTotal": "true",
            "fields": "NCTId,BriefTitle,Condition,InterventionName,Phase,OverallStatus,LocationCity,LocationState,LocationCountry,EligibilityCriteria,BriefSummary,DetailedDescription,StudyType,EnrollmentCount,LeadSponsorName,StartDate,PrimaryCompletionDate"
        }
        
        query_parts = []
        
        if self.condition:
            query_parts.append(f"AREA[Condition]{self.condition}")
        
        if self.intervention:
            query_parts.append(f"AREA[InterventionName]{self.intervention}")
            
        if self.term:
            query_parts.append(self.term)
            
        if self.status:
            status_query = " OR ".join([f"AREA[OverallStatus]{s}" for s in self.status])
            query_parts.append(f"({status_query})")
            
        if self.phase:
            phase_query = " OR ".join([f'AREA[Phase]"{p}"' for p in self.phase])
            query_parts.append(f"({phase_query})")
            
        if self.study_type:
            query_parts.append(f'AREA[StudyType]"{self.study_type}"')
            
        if self.location:
            query_parts.append(f"AREA[LocationCity]{self.location} OR AREA[LocationState]{self.location}")
        
        if query_parts:
            params["query.cond"] = " AND ".join(query_parts)
            
        return params


class RealTrialFetcher:
    """Fetch real clinical trials from ClinicalTrials.gov API v2"""
    
    BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
    
    def __init__(self):
        self.session = None
        logger.info("Initialized RealTrialFetcher using ClinicalTrials.gov API v2")
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()
    
    async def search_trials(
        self,
        patient: Patient,
        max_trials: int = 20
    ) -> List[ClinicalTrial]:
        """Search for trials matching patient characteristics."""
        
        # Build query based on patient
        query = TrialSearchQuery(
            condition=patient.cancer_type,
            status=[RecruitmentStatus.RECRUITING.value, RecruitmentStatus.NOT_YET_RECRUITING.value],
            page_size=max_trials
        )
        
        # Add biomarker terms if available
        if patient.biomarkers_detected:
            biomarker_terms = []
            for biomarker in patient.biomarkers_detected:
                if biomarker.status == "positive":
                    biomarker_terms.append(biomarker.name)
            if biomarker_terms:
                query.term = " OR ".join(biomarker_terms)
        
        # Add stage-specific filtering
        if patient.cancer_stage in ["I", "II"]:
            query.phase = ["PHASE2", "PHASE3"]
        elif patient.cancer_stage in ["III", "IV"]:
            query.phase = ["PHASE1", "PHASE2", "PHASE3"]
        
        return await self._fetch_trials(query)
    
    async def _fetch_trials(self, query: TrialSearchQuery) -> List[ClinicalTrial]:
        """Fetch trials from ClinicalTrials.gov API."""
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        params = query.to_params()
        
        try:
            logger.info(f"Fetching trials from ClinicalTrials.gov with params: {params}")
            
            async with self.session.get(
                self.BASE_URL,
                params=params,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    studies = data.get("studies", [])
                    
                    logger.info(f"✅ Retrieved {len(studies)} real trials from ClinicalTrials.gov")
                    
                    # Convert to our ClinicalTrial model
                    trials = []
                    for study_wrapper in studies[:query.page_size]:
                        study = study_wrapper.get("protocolSection", {})
                        trial = self._convert_to_trial(study)
                        if trial:
                            trials.append(trial)
                    
                    return trials
                    
                else:
                    logger.error(f"API error: {response.status}")
                    text = await response.text()
                    logger.error(f"Response: {text[:500]}")
                    return []
                    
        except asyncio.TimeoutError:
            logger.error("ClinicalTrials.gov API timeout")
            return []
        except Exception as e:
            logger.error(f"Error fetching trials: {e}")
            return []
    
    def _convert_to_trial(self, study: Dict[str, Any]) -> Optional[ClinicalTrial]:
        """Convert ClinicalTrials.gov study to our ClinicalTrial model."""
        
        try:
            # Extract basic info
            id_module = study.get("identificationModule", {})
            status_module = study.get("statusModule", {})
            desc_module = study.get("descriptionModule", {})
            conditions_module = study.get("conditionsModule", {})
            interventions_module = study.get("armsInterventionsModule", {})
            eligibility_module = study.get("eligibilityModule", {})
            locations_module = study.get("contactsLocationsModule", {})
            sponsor_module = study.get("sponsorCollaboratorsModule", {})
            design_module = study.get("designModule", {})
            
            nct_id = id_module.get("nctId", "")
            if not nct_id:
                return None
            
            # Extract phase and map to our expected format
            phases = design_module.get("phases", [])
            phase_raw = phases[0] if phases else "NOT_APPLICABLE"
            
            # Map ClinicalTrials.gov phase format to our model format
            phase_mapping = {
                "EARLY_PHASE1": "Early Phase 1",
                "PHASE1": "Phase 1",
                "PHASE1_PHASE2": "Phase 1/Phase 2",
                "PHASE2": "Phase 2",
                "PHASE2_PHASE3": "Phase 2/Phase 3",
                "PHASE3": "Phase 3",
                "PHASE4": "Phase 4",
                "NOT_APPLICABLE": "N/A",
                "Phase1": "Phase 1",
                "Phase2": "Phase 2",
                "Phase3": "Phase 3",
                "Phase4": "Phase 4"
            }
            phase = phase_mapping.get(phase_raw, "N/A")
            
            # Extract eligibility criteria
            criteria_text = eligibility_module.get("eligibilityCriteria", "")
            inclusion = []
            exclusion = []
            
            if criteria_text:
                # Simple parsing - in production would use more sophisticated NLP
                lines = criteria_text.split("\n")
                current_section = None
                for line in lines:
                    line = line.strip()
                    if "inclusion" in line.lower():
                        current_section = "inclusion"
                    elif "exclusion" in line.lower():
                        current_section = "exclusion"
                    elif line and current_section == "inclusion":
                        inclusion.append(line)
                    elif line and current_section == "exclusion":
                        exclusion.append(line)
            
            # Extract locations
            locations = []
            location_list = locations_module.get("locations", [])
            for loc in location_list[:5]:  # Limit locations
                location = Location(
                    name=loc.get("facility", ""),
                    city=loc.get("city", ""),
                    state=loc.get("state", ""),
                    country=loc.get("country", "USA")
                )
                locations.append(location)
            
            # If no locations, add a default
            if not locations:
                locations = [Location(
                    name="Multiple Centers",
                    city="Various",
                    state="Various",
                    country="USA"
                )]
            
            # Extract interventions
            interventions = []
            intervention_list = interventions_module.get("interventions", [])
            for interv in intervention_list:
                name = interv.get("name", "")
                if name:
                    interventions.append(name)
            
            # Map status to our expected format
            status_raw = status_module.get("overallStatus", "UNKNOWN")
            status_mapping = {
                "RECRUITING": "Recruiting",
                "NOT_YET_RECRUITING": "Not yet recruiting",
                "ENROLLING_BY_INVITATION": "Enrolling by invitation",
                "ACTIVE_NOT_RECRUITING": "Active, not recruiting",
                "SUSPENDED": "Suspended",
                "TERMINATED": "Terminated",
                "COMPLETED": "Completed",
                "WITHDRAWN": "Withdrawn"
            }
            status = status_mapping.get(status_raw, "Recruiting")  # Default to recruiting
            
            # Get dates with defaults
            import datetime
            start_date_raw = status_module.get("startDateStruct", {}).get("date", "")
            completion_date_raw = status_module.get("completionDateStruct", {}).get("date", "")
            
            # Fix date formats (ClinicalTrials.gov sometimes uses YYYY-MM format)
            def fix_date(date_str):
                if not date_str:
                    return None
                if len(date_str) == 7:  # YYYY-MM format
                    return date_str + "-01"  # Add day
                return date_str
            
            start_date = fix_date(start_date_raw)
            completion_date = fix_date(completion_date_raw)
            
            if not completion_date:
                # Set a default future date if no completion date
                completion_date = "2026-12-31"
            
            # Create trial object
            trial = ClinicalTrial(
                nct_id=nct_id,
                title=id_module.get("briefTitle", "")[:200],  # Limit title length
                phase=phase,
                status=status,
                conditions=conditions_module.get("conditions", []),
                interventions=interventions,
                sponsor=sponsor_module.get("leadSponsor", {}).get("name", "Unknown"),
                eligibility=EligibilityCriteria(
                    inclusion_criteria=inclusion[:10],  # Limit criteria
                    exclusion_criteria=exclusion[:10]
                ),
                locations=locations,
                brief_summary=desc_module.get("briefSummary", ""),
                recruitment_status=status,  # Use mapped status
                start_date=start_date if start_date else "2024-01-01",
                completion_date=completion_date
            )
            
            return trial
            
        except Exception as e:
            logger.error(f"Error converting trial {study.get('identificationModule', {}).get('nctId', 'unknown')}: {e}")
            return None


async def _test_real_fetcher():
    """Test the real trial fetcher."""
    from oncomatch.models import Patient, Biomarker
    
    # Create a test patient
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
    
    print("Testing Real Clinical Trial Fetcher")
    print("=" * 60)
    
    async with RealTrialFetcher() as fetcher:
        trials = await fetcher.search_trials(patient, max_trials=5)
        
        print(f"\n✅ Found {len(trials)} real trials!")
        
        for i, trial in enumerate(trials[:3], 1):
            print(f"\n{i}. {trial.nct_id}: {trial.title}")
            print(f"   Phase: {trial.phase}")
            print(f"   Status: {trial.status}")
            print(f"   Conditions: {', '.join(trial.conditions[:3])}")
            print(f"   Sponsor: {trial.sponsor}")
            if trial.locations:
                loc = trial.locations[0]
                print(f"   Location: {loc.city}, {loc.state}")
    
    return len(trials) > 0


if __name__ == "__main__":
    success = asyncio.run(test_real_fetcher())
    if success:
        print("\n🎉 Real trial fetching works!")
