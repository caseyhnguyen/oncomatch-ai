"""
Patient data processing pipeline.
Handles loading, validation, and normalization of patient data from CSV.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, date
import logging
import re

from oncomatch.models import (
    Patient, 
    Biomarker, 
    Gender, 
    CancerType, 
    CancerStage,
    ECOGStatus,
    TreatmentStage
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Process and validate patient data from various sources."""
    
    def __init__(self):
        self.biomarker_patterns = self._compile_biomarker_patterns()
        self.treatment_ontology = self._load_treatment_ontology()
    
    def _compile_biomarker_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for biomarker extraction."""
        return {
            'HER2': re.compile(r'HER2[+-]?', re.IGNORECASE),
            'ER': re.compile(r'ER[+-]?', re.IGNORECASE),
            'PR': re.compile(r'PR[+-]?', re.IGNORECASE),
            'EGFR': re.compile(r'EGFR(?:\s+mutation)?', re.IGNORECASE),
            'ALK': re.compile(r'ALK(?:\s+(?:fusion|rearrangement))?', re.IGNORECASE),
            'ROS1': re.compile(r'ROS1(?:\s+(?:fusion|rearrangement))?', re.IGNORECASE),
            'BRAF': re.compile(r'BRAF(?:\s+V600[EK]?)?', re.IGNORECASE),
            'KRAS': re.compile(r'KRAS(?:\s+G12[CD]?)?', re.IGNORECASE),
            'PD-L1': re.compile(r'PD-?L1(?:\s+(?:high|positive|\d+%))?', re.IGNORECASE),
            'BRCA1': re.compile(r'BRCA1(?:\s+mutation)?', re.IGNORECASE),
            'BRCA2': re.compile(r'BRCA2(?:\s+mutation)?', re.IGNORECASE),
            'MSI-H': re.compile(r'MSI-?H(?:igh)?', re.IGNORECASE),
            'dMMR': re.compile(r'dMMR|MMR\s+deficient', re.IGNORECASE),
            'PIK3CA': re.compile(r'PIK3CA(?:\s+mutation)?', re.IGNORECASE),
            'HRD': re.compile(r'HRD(?:\s+positive)?', re.IGNORECASE),
            'TMB': re.compile(r'TMB-?(?:high|H)', re.IGNORECASE),
            'TNBC': re.compile(r'(?:triple[-\s]?negative|TNBC)', re.IGNORECASE),
            'PSA': re.compile(r'PSA(?:\s+elevated)?', re.IGNORECASE)
        }
    
    def _load_treatment_ontology(self) -> Dict[str, List[str]]:
        """Load treatment ontology for standardization."""
        return {
            'chemotherapy': [
                'FOLFOX', 'FOLFIRI', 'FOLFIRINOX', 'carboplatin', 'cisplatin',
                'paclitaxel', 'docetaxel', 'gemcitabine', 'pemetrexed', 
                'etoposide', 'irinotecan', 'oxaliplatin', '5-FU', 'capecitabine',
                'doxorubicin', 'cyclophosphamide', 'vincristine'
            ],
            'targeted_therapy': [
                'trastuzumab', 'pertuzumab', 'T-DM1', 'osimertinib', 'erlotinib',
                'afatinib', 'alectinib', 'brigatinib', 'crizotinib', 'entrectinib',
                'dabrafenib', 'vemurafenib', 'sotorasib', 'adagrasib', 'bevacizumab',
                'ramucirumab', 'cetuximab', 'panitumumab'
            ],
            'immunotherapy': [
                'pembrolizumab', 'nivolumab', 'atezolizumab', 'durvalumab',
                'ipilimumab', 'avelumab', 'cemiplimab', 'checkpoint inhibitor'
            ],
            'hormone_therapy': [
                'tamoxifen', 'letrozole', 'anastrozole', 'exemestane', 'fulvestrant',
                'endocrine therapy', 'aromatase inhibitor', 'antiandrogen',
                'abiraterone', 'enzalutamide', 'bicalutamide'
            ],
            'parp_inhibitors': [
                'olaparib', 'rucaparib', 'niraparib', 'talazoparib', 'PARP inhibitor'
            ],
            'radiation': [
                'radiation', 'radiotherapy', 'SBRT', 'SRS', 'proton therapy',
                'brachytherapy', 'chemoradiation'
            ]
        }
    
    def load_patients_from_csv(self, filepath: Path) -> List[Patient]:
        """Load and process patient data from CSV file."""
        try:
            logger.info(f"Loading patient data from {filepath}")
            df = pd.read_csv(filepath)
            
            # Remove completely empty rows
            df = df.dropna(how='all')
            
            # Log data quality
            logger.info(f"Loaded {len(df)} patient records")
            logger.info(f"Columns: {df.columns.tolist()}")
            
            patients = []
            for idx, row in df.iterrows():
                try:
                    patient = self._process_patient_row(row, idx)
                    if patient:
                        patients.append(patient)
                except Exception as e:
                    logger.warning(f"Failed to process row {idx}: {str(e)}")
                    continue
            
            logger.info(f"Successfully processed {len(patients)} patients")
            return patients
            
        except Exception as e:
            logger.error(f"Failed to load patients from {filepath}: {str(e)}")
            raise
    
    def process_patient(self, patient_data: Union[Dict, pd.Series]) -> Patient:
        """Process patient data from dictionary or Series into Patient model."""
        if isinstance(patient_data, dict):
            # Convert dict to Series for consistent processing
            patient_data = pd.Series(patient_data)
        
        # Use existing row processing logic
        return self._process_patient_row(patient_data, 0)
    
    def _process_patient_row(self, row: pd.Series, idx: int) -> Optional[Patient]:
        """Process a single patient row from the dataframe."""
        # Skip if essential fields are missing
        if pd.isna(row.get('name')) or pd.isna(row.get('cancer_type')):
            return None
        
        # Generate patient ID (1-indexed for P001, P002, etc.)
        patient_id = f"P{idx+1:03d}"
        
        # Process demographics
        gender = self._parse_gender(row.get('gender'))
        age = self._safe_int(row.get('age'), default=50)
        
        # Process cancer information
        cancer_type = self._normalize_cancer_type(row.get('cancer_type'))
        cancer_stage = self._normalize_cancer_stage(row.get('cancer_stage'))
        
        # Process biomarkers
        biomarkers_detected = self._extract_biomarkers(row.get('biomarkers_detected'))
        biomarkers_ruled_out = self._extract_biomarker_names(row.get('biomarkers_ruled_out'))
        
        # Process treatments
        surgeries = self._parse_list_field(row.get('surgeries'))
        previous_treatments = self._parse_list_field(row.get('previous_treatments'))
        current_medications = self._parse_list_field(row.get('current_medications'))
        
        # Process dates
        diagnosis_date = self._parse_date(row.get('initial_diagnosis_date'))
        recurrence_date = self._parse_date(row.get('recurrence_date'))
        
        # Process clinical status
        ecog_status = self._parse_ecog_status(row.get('ecog_status'))
        treatment_stage = self._parse_treatment_stage(row.get('treatment_stage'))
        
        # Process comorbidities
        other_conditions = self._parse_list_field(row.get('other_conditions'))
        
        # Create Patient object
        patient = Patient(
            patient_id=patient_id,
            name=str(row['name']),
            age=age,
            gender=gender,
            race=self._safe_str(row.get('race')),
            city=self._safe_str(row.get('city', 'Unknown')),
            state=self._safe_str(row.get('state', 'Unknown')),
            height_cm=self._safe_float(row.get('height_cm')),
            weight_kg=self._safe_float(row.get('weight_kg')),
            bmi=self._safe_float(row.get('bmi')),
            cancer_type=cancer_type,
            cancer_stage=cancer_stage,
            cancer_substage=self._safe_str(row.get('cancer_substage')),
            cancer_grade=self._safe_str(row.get('cancer_grade')),
            initial_diagnosis_date=diagnosis_date,
            diagnosis_month=self._safe_int(row.get('diagnosis_month')),
            diagnosis_year=self._safe_int(row.get('diagnosis_year')),
            is_recurrence=self._safe_bool(row.get('is_recurrence')),
            recurrence_date=recurrence_date,
            treatment_stage=treatment_stage,
            surgeries=surgeries,
            previous_treatments=previous_treatments,
            current_medications=current_medications,
            biomarkers_detected=biomarkers_detected,
            biomarkers_ruled_out=biomarkers_ruled_out,
            ecog_status=ecog_status,
            smoking_status=self._safe_str(row.get('smoking_status')),
            drinking_status=self._safe_str(row.get('drinking_status')),
            other_conditions=other_conditions,
            family_history=self._safe_str(row.get('family_history')),
            patient_intent=self._safe_str(row.get('patient_intent'))
        )
        
        return patient
    
    def _parse_gender(self, value: Any) -> Gender:
        """Parse gender value to enum."""
        if pd.isna(value):
            return Gender.UNKNOWN
        
        gender_str = str(value).lower()
        if 'female' in gender_str or gender_str == 'f':
            return Gender.FEMALE
        elif 'male' in gender_str or gender_str == 'm':
            return Gender.MALE
        elif 'other' in gender_str:
            return Gender.OTHER
        else:
            return Gender.UNKNOWN
    
    def _normalize_cancer_type(self, value: Any) -> str:
        """Normalize cancer type to standard naming."""
        if pd.isna(value):
            return "Unknown"
        
        cancer_str = str(value).lower()
        
        # Map to standard types
        type_map = {
            'breast': 'Breast',
            'lung': 'Lung',
            'nsclc': 'Lung',
            'sclc': 'Lung',
            'colorectal': 'Colorectal',
            'colon': 'Colorectal',
            'rectal': 'Colorectal',
            'prostate': 'Prostate',
            'pancreatic': 'Pancreatic',
            'pancreas': 'Pancreatic',
            'bladder': 'Bladder',
            'urothelial': 'Bladder',
            'ovarian': 'Ovarian',
            'ovary': 'Ovarian',
            'melanoma': 'Melanoma',
            'leukemia': 'Leukemia',
            'lymphoma': 'Lymphoma'
        }
        
        for key, standard in type_map.items():
            if key in cancer_str:
                return standard
        
        # Return original if no match
        return str(value).title()
    
    def _normalize_cancer_stage(self, value: Any) -> str:
        """Normalize cancer stage to standard format."""
        if pd.isna(value):
            return "Unknown"
        
        stage_str = str(value).upper().strip()
        
        # Remove 'STAGE' prefix if present
        stage_str = stage_str.replace('STAGE', '').strip()
        
        # Map to standard stages
        valid_stages = ['0', 'I', 'IA', 'IB', 'IC', 'II', 'IIA', 'IIB', 'IIC',
                       'III', 'IIIA', 'IIIB', 'IIIC', 'IV', 'IVA', 'IVB', 'IVC']
        
        if stage_str in valid_stages:
            return stage_str
        
        # Handle numeric stages
        if stage_str in ['1', '2', '3', '4']:
            return ['I', 'II', 'III', 'IV'][int(stage_str) - 1]
        
        return stage_str if stage_str else "Unknown"
    
    def _extract_biomarkers(self, value: Any) -> List[Biomarker]:
        """Extract structured biomarkers from string field."""
        biomarkers = []
        
        if pd.isna(value) or not value:
            return biomarkers
        
        biomarker_str = str(value)
        
        # Check for each known biomarker pattern
        for name, pattern in self.biomarker_patterns.items():
            match = pattern.search(biomarker_str)
            if match:
                matched_text = match.group(0)
                
                # Determine status
                status = None
                if '+' in matched_text or 'positive' in matched_text.lower():
                    status = 'positive'
                elif '-' in matched_text or 'negative' in matched_text.lower():
                    status = 'negative'
                elif 'mutation' in matched_text.lower() or name in ['EGFR', 'KRAS', 'BRAF', 'PIK3CA']:
                    status = 'mutated'
                elif 'high' in matched_text.lower():
                    status = 'high'
                else:
                    status = 'detected'
                
                # Extract value if present (e.g., PD-L1 50%)
                value_match = re.search(r'(\d+)%', matched_text)
                value_data = None
                if value_match:
                    value_data = f"{value_match.group(1)}%"
                
                biomarkers.append(Biomarker(
                    name=name,
                    status=status,
                    value=value_data
                ))
        
        # Handle triple negative as special case
        if 'triple negative' in biomarker_str.lower() or 'TNBC' in biomarker_str:
            # Add negative markers
            for marker in ['ER', 'PR', 'HER2']:
                if not any(b.name == marker for b in biomarkers):
                    biomarkers.append(Biomarker(name=marker, status='negative'))
        
        return biomarkers
    
    def _extract_biomarker_names(self, value: Any) -> List[str]:
        """Extract biomarker names from ruled out field."""
        if pd.isna(value) or not value:
            return []
        
        biomarker_str = str(value)
        ruled_out = []
        
        for name, pattern in self.biomarker_patterns.items():
            if pattern.search(biomarker_str):
                ruled_out.append(name)
        
        return ruled_out
    
    def _parse_list_field(self, value: Any) -> List[str]:
        """Parse comma-separated list field."""
        if pd.isna(value) or not value:
            return []
        
        # Split by comma and clean
        items = str(value).split(',')
        return [item.strip() for item in items if item.strip()]
    
    def _parse_date(self, value: Any) -> Optional[date]:
        """Parse date from various formats."""
        if pd.isna(value) or not value:
            return None
        
        try:
            # Try parsing as datetime
            if isinstance(value, str):
                # Common date formats
                for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d']:
                    try:
                        return datetime.strptime(value, fmt).date()
                    except:
                        continue
            elif isinstance(value, (datetime, date)):
                return value if isinstance(value, date) else value.date()
        except Exception as e:
            logger.debug(f"Failed to parse date {value}: {e}")
        
        return None
    
    def _parse_ecog_status(self, value: Any) -> Optional[ECOGStatus]:
        """Parse ECOG status value."""
        if pd.isna(value):
            return None
        
        try:
            ecog_int = int(float(value))
            if 0 <= ecog_int <= 4:
                return ECOGStatus(ecog_int)
        except:
            pass
        
        return None
    
    def _parse_treatment_stage(self, value: Any) -> Optional[TreatmentStage]:
        """Parse treatment stage value."""
        if pd.isna(value) or not value:
            return None
        
        stage_str = str(value).lower()
        
        if 'neoadjuvant' in stage_str:
            return TreatmentStage.NEOADJUVANT
        elif 'adjuvant' in stage_str:
            return TreatmentStage.ADJUVANT
        elif 'metastatic' in stage_str:
            return TreatmentStage.METASTATIC
        elif 'surveillance' in stage_str:
            return TreatmentStage.SURVEILLANCE
        elif 'palliative' in stage_str:
            return TreatmentStage.PALLIATIVE
        
        return None
    
    def _safe_str(self, value: Any, default: str = None) -> Optional[str]:
        """Safely convert value to string."""
        if pd.isna(value):
            return default
        return str(value).strip() if value else default
    
    def _safe_int(self, value: Any, default: int = None) -> Optional[int]:
        """Safely convert value to integer."""
        if pd.isna(value):
            return default
        try:
            return int(float(value))
        except:
            return default
    
    def _safe_float(self, value: Any, default: float = None) -> Optional[float]:
        """Safely convert value to float."""
        if pd.isna(value):
            return default
        try:
            return float(value)
        except:
            return default
    
    def _safe_bool(self, value: Any, default: bool = False) -> bool:
        """Safely convert value to boolean."""
        if pd.isna(value):
            return default
        
        if isinstance(value, bool):
            return value
        
        str_val = str(value).lower()
        return str_val in ['true', 'yes', '1', 't', 'y']
    
    def categorize_treatments(self, treatments: List[str]) -> Dict[str, List[str]]:
        """Categorize treatments by type using ontology."""
        categorized = {
            'chemotherapy': [],
            'targeted_therapy': [],
            'immunotherapy': [],
            'hormone_therapy': [],
            'parp_inhibitors': [],
            'radiation': [],
            'other': []
        }
        
        for treatment in treatments:
            treatment_lower = treatment.lower()
            found = False
            
            for category, drugs in self.treatment_ontology.items():
                if any(drug.lower() in treatment_lower for drug in drugs):
                    categorized[category].append(treatment)
                    found = True
                    break
            
            if not found:
                categorized['other'].append(treatment)
        
        return {k: v for k, v in categorized.items() if v}  # Return only non-empty categories
    
    def validate_patient_data(self, patient: Patient) -> Tuple[bool, List[str]]:
        """Validate patient data for completeness and consistency."""
        errors = []
        
        # Essential fields
        if not patient.patient_id:
            errors.append("Missing patient ID")
        
        if not patient.cancer_type:
            errors.append("Missing cancer type")
        
        if patient.age < 0 or patient.age > 120:
            errors.append(f"Invalid age: {patient.age}")
        
        # Clinical consistency checks
        if patient.is_recurrence and not patient.recurrence_date:
            errors.append("Recurrence flagged but no recurrence date")
        
        if patient.ecog_status and patient.ecog_status.value > 4:
            errors.append(f"Invalid ECOG status: {patient.ecog_status.value}")
        
        # BMI consistency
        if patient.height_cm and patient.weight_kg:
            calculated_bmi = patient.weight_kg / ((patient.height_cm / 100) ** 2)
            if patient.bmi and abs(calculated_bmi - patient.bmi) > 2:
                errors.append(f"BMI inconsistency: calculated {calculated_bmi:.1f} vs provided {patient.bmi}")
        
        # Treatment stage consistency
        if patient.cancer_stage in ['IV', 'IVA', 'IVB'] and patient.treatment_stage == TreatmentStage.NEOADJUVANT:
            errors.append("Neoadjuvant treatment unusual for stage IV cancer")
        
        is_valid = len(errors) == 0
        return is_valid, errors

