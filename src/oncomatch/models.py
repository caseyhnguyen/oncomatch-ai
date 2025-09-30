"""
Pydantic models for clinical trial matching system.
Includes comprehensive medical field validation for patient data, trials, and match results.
"""

from __future__ import annotations
from datetime import date, datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator, root_validator
import re


class Gender(str, Enum):
    """Gender options following clinical trial standards."""
    MALE = "Male"
    FEMALE = "Female"
    OTHER = "Other"
    UNKNOWN = "Unknown"


class CancerType(str, Enum):
    """Common cancer types with standardized naming."""
    BREAST = "Breast"
    LUNG = "Lung"
    COLORECTAL = "Colorectal"
    PROSTATE = "Prostate"
    PANCREATIC = "Pancreatic"
    BLADDER = "Bladder"
    OVARIAN = "Ovarian"
    MELANOMA = "Melanoma"
    LEUKEMIA = "Leukemia"
    LYMPHOMA = "Lymphoma"
    OTHER = "Other"


class CancerStage(str, Enum):
    """TNM staging system representation."""
    STAGE_0 = "0"
    STAGE_I = "I"
    STAGE_IA = "IA"
    STAGE_IB = "IB"
    STAGE_II = "II"
    STAGE_IIA = "IIA"
    STAGE_IIB = "IIB"
    STAGE_III = "III"
    STAGE_IIIA = "IIIA"
    STAGE_IIIB = "IIIB"
    STAGE_IIIC = "IIIC"
    STAGE_IV = "IV"
    STAGE_IVA = "IVA"
    STAGE_IVB = "IVB"
    UNKNOWN = "Unknown"


class ECOGStatus(int, Enum):
    """ECOG Performance Status scale."""
    FULLY_ACTIVE = 0  # Fully active, able to carry on all activities
    RESTRICTED = 1    # Restricted in strenuous activity but ambulatory
    AMBULATORY = 2    # Ambulatory and capable of self-care but unable to work
    LIMITED = 3       # Capable of only limited self-care
    DISABLED = 4      # Completely disabled
    DEAD = 5         # Dead


class TreatmentStage(str, Enum):
    """Treatment timing relative to primary therapy."""
    NEOADJUVANT = "neoadjuvant"  # Before primary treatment
    ADJUVANT = "adjuvant"        # After primary treatment
    METASTATIC = "metastatic"    # For metastatic disease
    SURVEILLANCE = "surveillance" # Active monitoring
    PALLIATIVE = "palliative"    # Symptom management


class TrialPhase(str, Enum):
    """Clinical trial phases."""
    EARLY_PHASE_1 = "Early Phase 1"
    PHASE_1 = "Phase 1"
    PHASE_1_2 = "Phase 1/Phase 2"
    PHASE_2 = "Phase 2"
    PHASE_2_3 = "Phase 2/Phase 3"
    PHASE_3 = "Phase 3"
    PHASE_4 = "Phase 4"
    NOT_APPLICABLE = "N/A"


class RecruitmentStatus(str, Enum):
    """Trial recruitment status."""
    RECRUITING = "Recruiting"
    NOT_YET_RECRUITING = "Not yet recruiting"
    ENROLLING_BY_INVITATION = "Enrolling by invitation"
    ACTIVE_NOT_RECRUITING = "Active, not recruiting"
    SUSPENDED = "Suspended"
    TERMINATED = "Terminated"
    COMPLETED = "Completed"
    WITHDRAWN = "Withdrawn"


class Biomarker(BaseModel):
    """Represents a single biomarker with its status and value."""
    name: str = Field(..., description="Biomarker name (e.g., EGFR, HER2)")
    status: Optional[str] = Field(None, description="Status: positive, negative, mutated, etc.")
    value: Optional[Union[str, float]] = Field(None, description="Quantitative value if applicable")
    
    @validator('name')
    def standardize_biomarker_name(cls, v):
        """Standardize common biomarker names."""
        standardization = {
            'HER2+': 'HER2',
            'ER+': 'ER',
            'PR+': 'PR',
            'PDL1': 'PD-L1',
            'EGFR_MUTATION': 'EGFR',
        }
        return standardization.get(v.upper(), v.upper())


class Location(BaseModel):
    """Geographic location for trial sites."""
    city: str
    state: str
    country: str = "USA"
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    distance_km: Optional[float] = Field(None, description="Distance from patient location")


class Patient(BaseModel):
    """Comprehensive patient model with medical history and biomarkers."""
    # Demographics
    patient_id: str = Field(..., description="Unique patient identifier")
    name: str
    age: int = Field(..., ge=0, le=120)
    gender: Gender
    race: Optional[str] = None
    
    # Location
    city: str
    state: str
    
    # Physical measurements
    height_cm: Optional[float] = Field(None, gt=0, le=300)
    weight_kg: Optional[float] = Field(None, gt=0, le=500)
    bmi: Optional[float] = Field(None, gt=0, le=100)
    
    # Cancer details
    cancer_type: str
    cancer_stage: str
    cancer_substage: Optional[str] = None
    cancer_grade: Optional[str] = None
    
    # Diagnosis information
    initial_diagnosis_date: Optional[date] = None
    diagnosis_month: Optional[int] = Field(None, ge=1, le=12)
    diagnosis_year: Optional[int] = Field(None, ge=1900, le=2100)
    is_recurrence: bool = False
    recurrence_date: Optional[date] = None
    
    # Treatment information
    treatment_stage: Optional[TreatmentStage] = None
    surgeries: List[str] = Field(default_factory=list)
    previous_treatments: List[str] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)
    
    # Biomarkers
    biomarkers_detected: List[Biomarker] = Field(default_factory=list)
    biomarkers_ruled_out: List[str] = Field(default_factory=list)
    
    # Clinical status
    ecog_status: Optional[ECOGStatus] = None
    smoking_status: Optional[str] = None
    drinking_status: Optional[str] = None
    other_conditions: List[str] = Field(default_factory=list)
    family_history: Optional[str] = None
    patient_intent: Optional[str] = None
    
    @validator('cancer_type')
    def standardize_cancer_type(cls, v):
        """Map cancer type to standardized enum value."""
        type_mapping = {
            'breast': CancerType.BREAST,
            'lung': CancerType.LUNG,
            'colorectal': CancerType.COLORECTAL,
            'colon': CancerType.COLORECTAL,
            'prostate': CancerType.PROSTATE,
            'pancreatic': CancerType.PANCREATIC,
            'bladder': CancerType.BLADDER,
            'ovarian': CancerType.OVARIAN,
        }
        return type_mapping.get(v.lower(), v).value if hasattr(type_mapping.get(v.lower(), v), 'value') else v
    
    @root_validator(skip_on_failure=True)
    def calculate_bmi_if_missing(cls, values):
        """Calculate BMI if height and weight are provided but BMI is not."""
        if not values.get('bmi') and values.get('height_cm') and values.get('weight_kg'):
            height_m = values['height_cm'] / 100
            values['bmi'] = round(values['weight_kg'] / (height_m ** 2), 1)
        return values
    
    def get_line_of_therapy(self) -> int:
        """Calculate the line of therapy based on previous treatments."""
        if not self.previous_treatments:
            return 1
        # Count distinct treatment regimens
        return min(len(self.previous_treatments), 5)  # Cap at 5th line
    
    def has_biomarker(self, biomarker_name: str) -> bool:
        """Check if patient has a specific biomarker detected."""
        return any(b.name.upper() == biomarker_name.upper() 
                  for b in self.biomarkers_detected)
    
    def is_heavily_pretreated(self) -> bool:
        """Determine if patient has received extensive prior therapy."""
        return len(self.previous_treatments) >= 3


class EligibilityCriteria(BaseModel):
    """Structured eligibility criteria for clinical trials."""
    inclusion_criteria: List[str] = Field(default_factory=list)
    exclusion_criteria: List[str] = Field(default_factory=list)
    
    # Structured criteria
    min_age: Optional[int] = None
    max_age: Optional[int] = None
    gender_requirement: Optional[Gender] = None
    cancer_types: List[str] = Field(default_factory=list)
    cancer_stages: List[str] = Field(default_factory=list)
    required_biomarkers: List[str] = Field(default_factory=list)
    excluded_biomarkers: List[str] = Field(default_factory=list)
    max_prior_therapies: Optional[int] = None
    required_ecog_status: Optional[List[ECOGStatus]] = None
    
    def extract_structured_criteria(self):
        """Extract structured information from free-text criteria."""
        # Age extraction
        for criterion in self.inclusion_criteria:
            age_match = re.search(r'age[s]?\s*(?:>=?|≥)\s*(\d+)', criterion.lower())
            if age_match:
                self.min_age = int(age_match.group(1))
            age_match = re.search(r'age[s]?\s*(?:<=?|≤)\s*(\d+)', criterion.lower())
            if age_match:
                self.max_age = int(age_match.group(1))
        
        # ECOG extraction
        for criterion in self.inclusion_criteria:
            ecog_match = re.search(r'ecog\s*(?:<=?|≤)\s*(\d)', criterion.lower())
            if ecog_match:
                max_ecog = int(ecog_match.group(1))
                self.required_ecog_status = [ECOGStatus(i) for i in range(max_ecog + 1)]


class ClinicalTrial(BaseModel):
    """Comprehensive clinical trial model."""
    nct_id: str = Field(..., pattern=r'^NCT\d{8}$')
    title: str
    official_title: Optional[str] = None
    brief_summary: Optional[str] = None
    detailed_description: Optional[str] = None
    
    # Trial characteristics
    phase: Optional[TrialPhase] = None
    status: RecruitmentStatus
    study_type: Optional[str] = None
    
    # Medical focus
    conditions: List[str] = Field(default_factory=list)
    interventions: List[str] = Field(default_factory=list)
    
    # Eligibility
    eligibility: EligibilityCriteria = Field(default_factory=EligibilityCriteria)
    
    # Locations
    locations: List[Location] = Field(default_factory=list)
    
    # Dates
    start_date: Optional[date] = None
    completion_date: Optional[date] = None
    last_update_posted: Optional[date] = None
    
    # Sponsor information
    sponsor: Optional[str] = None
    collaborators: List[str] = Field(default_factory=list)
    
    def is_applicable_for_cancer_type(self, cancer_type: str) -> bool:
        """Check if trial is applicable for a specific cancer type."""
        cancer_lower = cancer_type.lower()
        return any(cancer_lower in condition.lower() for condition in self.conditions)
    
    def get_nearest_location(self, patient_lat: float, patient_lon: float) -> Optional[Location]:
        """Find the nearest trial location to the patient."""
        if not self.locations:
            return None
        
        from math import radians, cos, sin, asin, sqrt
        
        def haversine(lon1, lat1, lon2, lat2):
            """Calculate the great circle distance between two points."""
            lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            r = 6371  # Radius of earth in kilometers
            return c * r
        
        nearest = None
        min_distance = float('inf')
        
        for location in self.locations:
            if location.latitude and location.longitude:
                distance = haversine(patient_lon, patient_lat, 
                                   location.longitude, location.latitude)
                if distance < min_distance:
                    min_distance = distance
                    nearest = location
                    nearest.distance_km = distance
        
        return nearest


class MatchReason(BaseModel):
    """Detailed reasoning for trial match."""
    criterion: str = Field(..., description="Specific eligibility criterion")
    matched: bool = Field(..., description="Whether patient matches this criterion")
    explanation: str = Field(..., description="Detailed explanation of match/mismatch")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in assessment")
    category: str = Field(..., description="Category: inclusion, exclusion, biomarker, etc.")


class MatchResult(BaseModel):
    """Result of matching a patient to a clinical trial."""
    patient_id: str
    nct_id: str
    
    # Scores
    overall_score: float = Field(..., ge=0, le=1, description="Overall match score")
    eligibility_score: float = Field(..., ge=0, le=1, description="Eligibility criteria match")
    biomarker_score: float = Field(..., ge=0, le=1, description="Biomarker compatibility")
    geographic_score: float = Field(..., ge=0, le=1, description="Geographic accessibility")
    
    # Match details
    is_eligible: bool = Field(..., description="Binary eligibility decision")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in match")
    
    # Reasoning
    match_reasons: List[MatchReason] = Field(default_factory=list)
    summary: str = Field(..., description="Human-readable match summary")
    
    # Safety flags
    safety_concerns: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    # Metadata
    distance_km: Optional[float] = None
    trial_phase: Optional[TrialPhase] = None
    matching_timestamp: datetime = Field(default_factory=datetime.now)
    llm_model_used: Optional[str] = None
    processing_time_ms: Optional[int] = None
    
    def get_primary_exclusion_reason(self) -> Optional[str]:
        """Get the most important reason for exclusion if not eligible."""
        if self.is_eligible:
            return None
        
        exclusion_reasons = [r for r in self.match_reasons 
                            if not r.matched and r.category == 'exclusion']
        if exclusion_reasons:
            # Sort by confidence and return highest confidence exclusion
            return max(exclusion_reasons, key=lambda x: x.confidence).explanation
        return "No specific exclusion reason identified"
    
    class Config:
        """Pydantic config."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat()
        }


class MatchingRequest(BaseModel):
    """Request model for trial matching."""
    patient: Patient
    max_trials: int = Field(10, ge=1, le=100)
    max_distance_km: Optional[float] = Field(500, gt=0)
    phases: Optional[List[TrialPhase]] = None
    exclude_completed: bool = True
    prioritize_biomarkers: bool = True
    include_observational: bool = False


class MatchingResponse(BaseModel):
    """Response model for trial matching."""
    patient_id: str
    matches: List[MatchResult]
    total_trials_screened: int
    processing_time_ms: int
    timestamp: datetime = Field(default_factory=datetime.now)
    
    def get_top_matches(self, n: int = 5) -> List[MatchResult]:
        """Get top N matches by score."""
        eligible = [m for m in self.matches if m.is_eligible]
        eligible.sort(key=lambda x: x.overall_score, reverse=True)
        return eligible[:n]
