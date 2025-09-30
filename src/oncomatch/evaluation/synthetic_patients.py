"""
Synthetic Patient Generator for realistic oncology test cases.
Includes standard, edge, and stress test cases.
"""

import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
from tqdm import tqdm

from oncomatch.models import (
    Patient, Gender, ECOGStatus, CancerStage, 
    TrialPhase, Biomarker
)

logger = logging.getLogger(__name__)


class PatientCategory(str, Enum):
    """Categories of synthetic patients."""
    STANDARD = "standard"           # ~60% - Realistic distributions
    EDGE_CASE = "edge_case"         # ~25% - Rare/extreme cases  
    ADVERSARIAL = "adversarial"     # ~10% - Attack/robustness tests
    EQUITY_STRESS = "equity_stress" # ~5% - Underserved populations


@dataclass
class SyntheticPatientProfile:
    """Profile for generating synthetic patients."""
    category: PatientCategory
    cancer_type: str
    age_range: Tuple[int, int]
    biomarker_probability: Dict[str, float]
    comorbidity_rate: float
    treatment_lines_range: Tuple[int, int]
    ecog_distribution: Dict[int, float]
    geographic_distribution: Dict[str, float]
    demographic_weights: Dict[str, float]


class EpidemiologyData:
    """2025 epidemiology priors for realistic patient generation."""
    
    # US cancer incidence rates (per 100,000)
    CANCER_INCIDENCE = {
        "Breast": 128.0,
        "Lung": 54.0,
        "Prostate": 109.0,
        "Colorectal": 37.0,
        "Melanoma": 22.0,
        "Bladder": 20.0,
        "Kidney": 18.0,
        "Pancreatic": 13.0,
        "Ovarian": 11.0,
        "Liver": 9.0
    }
    
    # Age distributions by cancer type
    AGE_DISTRIBUTIONS = {
        "Breast": {"mean": 62, "std": 13},
        "Lung": {"mean": 70, "std": 10},
        "Prostate": {"mean": 66, "std": 9},
        "Colorectal": {"mean": 68, "std": 12},
        "Melanoma": {"mean": 65, "std": 15},
        "Bladder": {"mean": 73, "std": 9},
        "Kidney": {"mean": 64, "std": 12},
        "Pancreatic": {"mean": 70, "std": 10},
        "Ovarian": {"mean": 63, "std": 13},
        "Liver": {"mean": 63, "std": 11}
    }
    
    # Biomarker prevalence by cancer type
    BIOMARKER_PREVALENCE = {
        "Breast": {
            "ER+": 0.70,
            "PR+": 0.65,
            "HER2+": 0.15,
            "BRCA1": 0.05,
            "BRCA2": 0.05,
            "PIK3CA": 0.40,
            "Triple-negative": 0.15
        },
        "Lung": {
            "EGFR": 0.15,
            "ALK": 0.05,
            "ROS1": 0.02,
            "BRAF": 0.02,
            "KRAS": 0.30,
            "PD-L1 ≥50%": 0.30,
            "PD-L1 1-49%": 0.35
        },
        "Colorectal": {
            "MSI-H": 0.15,
            "KRAS": 0.40,
            "BRAF": 0.10,
            "HER2": 0.03
        },
        "Melanoma": {
            "BRAF V600": 0.50,
            "NRAS": 0.25,
            "KIT": 0.02
        }
    }
    
    # Biomarker co-occurrence patterns
    BIOMARKER_COOCCURRENCE = {
        ("BRCA1", "BRCA2"): 0.001,  # Mutually exclusive mostly
        ("ER+", "PR+"): 0.85,       # High co-occurrence
        ("ER+", "HER2+"): 0.10,     # Low co-occurrence
        ("EGFR", "ALK"): 0.001,     # Mutually exclusive
        ("EGFR", "KRAS"): 0.01,     # Rarely together
        ("MSI-H", "BRAF"): 0.30     # Some correlation
    }
    
    # Treatment resistance patterns
    RESISTANCE_PATTERNS = {
        "Breast": {
            "endocrine": ["ESR1 mutation", "CDK4/6 resistance"],
            "HER2": ["HER2 mutation", "PIK3CA activation"],
            "chemotherapy": ["MDR1 overexpression", "TUBB3 mutation"]
        },
        "Lung": {
            "EGFR-TKI": ["T790M", "C797S", "MET amplification"],
            "ALK-TKI": ["Secondary ALK mutations", "Bypass signaling"],
            "immunotherapy": ["JAK1/2 loss", "B2M loss", "PTEN loss"]
        }
    }
    
    # Geographic distribution (US regions)
    GEOGRAPHIC_DISTRIBUTION = {
        "Northeast": 0.17,
        "Southeast": 0.24,
        "Midwest": 0.21,
        "Southwest": 0.11,
        "West": 0.23,
        "Rural": 0.04
    }
    
    # Demographic distribution
    DEMOGRAPHIC_DISTRIBUTION = {
        "race": {
            "White": 0.60,
            "Black": 0.13,
            "Hispanic": 0.18,
            "Asian": 0.06,
            "Other": 0.03
        },
        "gender_by_cancer": {
            "Breast": {"Female": 0.99, "Male": 0.01},
            "Prostate": {"Male": 1.0, "Female": 0.0},
            "Lung": {"Male": 0.54, "Female": 0.46},
            "default": {"Male": 0.50, "Female": 0.50}
        }
    }


class SyntheticPatientGenerator:
    """
    Generate 5,000+ realistic synthetic oncology patients.
    Includes edge cases, adversarial examples, and equity stress tests.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize generator with optional seed for reproducibility."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.epidemiology = EpidemiologyData()
        self.patient_id_counter = 0
    
    def generate_cohort(
        self,
        n_patients: int = 5000,
        category_distribution: Optional[Dict[PatientCategory, float]] = None
    ) -> List[Patient]:
        """
        Generate a cohort of synthetic patients.
        
        Args:
            n_patients: Number of patients to generate
            category_distribution: Distribution of patient categories
            
        Returns:
            List of synthetic patients
        """
        if category_distribution is None:
            category_distribution = {
                PatientCategory.STANDARD: 0.60,
                PatientCategory.EDGE_CASE: 0.25,
                PatientCategory.ADVERSARIAL: 0.10,
                PatientCategory.EQUITY_STRESS: 0.05
            }
        
        patients = []
        
        # Calculate total number to generate for progress bar
        total_to_generate = sum(int(n_patients * proportion) for proportion in category_distribution.values())
        
        with tqdm(total=total_to_generate, desc="Generating synthetic patients") as pbar:
            for category, proportion in category_distribution.items():
                n_category = int(n_patients * proportion)
                pbar.set_postfix({"Category": category.value})
                
                for _ in range(n_category):
                    if category == PatientCategory.STANDARD:
                        patient = self._generate_standard_patient()
                    elif category == PatientCategory.EDGE_CASE:
                        patient = self._generate_edge_case_patient()
                    elif category == PatientCategory.ADVERSARIAL:
                        patient = self._generate_adversarial_patient()
                    elif category == PatientCategory.EQUITY_STRESS:
                        patient = self._generate_equity_stress_patient()
                    else:
                        patient = self._generate_standard_patient()
                    
                    patients.append(patient)
                    pbar.update(1)
        
        # Shuffle to mix categories
        random.shuffle(patients)
        
        logger.info(f"Generated {len(patients)} synthetic patients")
        logger.info(f"Categories: Standard={int(n_patients*0.6)}, Edge={int(n_patients*0.25)}, "
                   f"Adversarial={int(n_patients*0.1)}, Equity={int(n_patients*0.05)}")
        
        return patients
    
    def _generate_standard_patient(self) -> Patient:
        """Generate standard patient with realistic distributions."""
        # Select cancer type based on incidence
        cancer_type = self._select_cancer_type()
        
        # Generate age based on cancer type
        age_dist = self.epidemiology.AGE_DISTRIBUTIONS[cancer_type]
        age = int(np.random.normal(age_dist["mean"], age_dist["std"]))
        age = max(18, min(100, age))  # Clamp to reasonable range
        
        # Select gender based on cancer type
        gender = self._select_gender(cancer_type)
        
        # Generate stage
        stage = self._select_stage_standard()
        
        # Generate biomarkers
        biomarkers = self._generate_biomarkers_standard(cancer_type)
        
        # Generate treatment history
        treatments = self._generate_treatment_history_standard(cancer_type, stage)
        
        # Generate ECOG status
        ecog = self._select_ecog_standard(age, stage)
        
        # Generate demographics
        demographics = self._generate_demographics_standard()
        
        # Generate comorbidities
        comorbidities = self._generate_comorbidities(age, standard=True)
        
        return self._create_patient(
            patient_category="standard",
            cancer_type=cancer_type,
            age=age,
            gender=gender,
            stage=stage,
            biomarkers=biomarkers,
            treatments=treatments,
            ecog=ecog,
            demographics=demographics,
            comorbidities=comorbidities
        )
    
    def _generate_edge_case_patient(self) -> Patient:
        """Generate edge case patient."""
        edge_case_type = random.choice([
            "pediatric",
            "elderly",
            "rare_mutation",
            "heavy_pretreat",
            "pregnancy",
            "multi_primary",
            "borderline_eligible",
            "conflicting_biomarkers",
            "rural_isolated",
            "immunocompromised"
        ])
        
        if edge_case_type == "pediatric":
            return self._generate_pediatric_patient()
        elif edge_case_type == "elderly":
            return self._generate_elderly_patient()
        elif edge_case_type == "rare_mutation":
            return self._generate_rare_mutation_patient()
        elif edge_case_type == "heavy_pretreat":
            return self._generate_heavy_pretreat_patient()
        elif edge_case_type == "pregnancy":
            return self._generate_pregnancy_patient()
        elif edge_case_type == "multi_primary":
            return self._generate_multi_primary_patient()
        elif edge_case_type == "borderline_eligible":
            return self._generate_borderline_patient()
        elif edge_case_type == "conflicting_biomarkers":
            return self._generate_conflicting_biomarkers_patient()
        elif edge_case_type == "rural_isolated":
            return self._generate_rural_patient()
        else:  # immunocompromised
            return self._generate_immunocompromised_patient()
    
    def _generate_pediatric_patient(self) -> Patient:
        """Generate pediatric cancer patient."""
        age = random.randint(2, 17)
        cancer_type = random.choice(["Leukemia", "Brain", "Lymphoma", "Neuroblastoma"])
        
        return self._create_patient(
            patient_category="edge_case_pediatric",
            cancer_type=cancer_type,
            age=age,
            gender=random.choice([Gender.MALE, Gender.FEMALE]),
            stage="III",  # Often advanced at diagnosis
            biomarkers=[],  # Limited biomarker testing in pediatrics
            treatments=[],
            ecog=ECOGStatus.FULLY_ACTIVE if age > 10 else ECOGStatus.RESTRICTED,
            demographics=self._generate_demographics_standard(),
            comorbidities=[]
        )
    
    def _generate_elderly_patient(self) -> Patient:
        """Generate elderly patient with multiple comorbidities."""
        age = random.randint(80, 95)
        cancer_type = random.choice(["Lung", "Colorectal", "Bladder", "Prostate"])
        
        # Many comorbidities
        comorbidities = random.sample([
            "Heart failure",
            "COPD",
            "Diabetes",
            "Chronic kidney disease",
            "Dementia",
            "Osteoporosis",
            "Atrial fibrillation"
        ], k=random.randint(3, 5))
        
        return self._create_patient(
            patient_category="edge_case_elderly",
            cancer_type=cancer_type,
            age=age,
            gender=random.choice([Gender.MALE, Gender.FEMALE]),
            stage=random.choice(["II", "III", "IV"]),
            biomarkers=self._generate_biomarkers_standard(cancer_type),
            treatments=["Surgery"] if random.random() > 0.5 else [],
            ecog=random.choice([ECOGStatus.AMBULATORY, ECOGStatus.LIMITED]),
            demographics=self._generate_demographics_standard(),
            comorbidities=comorbidities
        )
    
    def _generate_rare_mutation_patient(self) -> Patient:
        """Generate patient with rare mutations."""
        cancer_type = random.choice(["Lung", "Colorectal", "Melanoma"])
        
        # Rare mutations
        rare_biomarkers = [
            Biomarker(name="RET fusion", status="positive"),
            Biomarker(name="NTRK fusion", status="positive"),
            Biomarker(name="FGFR2 fusion", status="positive"),
            Biomarker(name="MET exon 14", status="positive")
        ]
        
        selected_rare = random.sample(rare_biomarkers, k=random.randint(1, 2))
        
        return self._create_patient(
            patient_category="edge_case_rare_mutation",
            cancer_type=cancer_type,
            age=random.randint(35, 75),
            gender=random.choice([Gender.MALE, Gender.FEMALE]),
            stage=random.choice(["III", "IV"]),
            biomarkers=selected_rare,
            treatments=["Chemotherapy", "Immunotherapy"],
            ecog=ECOGStatus.RESTRICTED,
            demographics=self._generate_demographics_standard(),
            comorbidities=[]
        )
    
    def _generate_heavy_pretreat_patient(self) -> Patient:
        """Generate heavily pretreated patient."""
        cancer_type = random.choice(["Lung", "Breast", "Colorectal"])
        
        # Many prior treatments
        treatments = [
            "Surgery",
            "Radiation",
            "Chemotherapy - Line 1",
            "Chemotherapy - Line 2", 
            "Chemotherapy - Line 3",
            "Immunotherapy",
            "Targeted therapy",
            "Clinical trial drug"
        ]
        
        return self._create_patient(
            patient_category="edge_case_heavy_pretreat",
            cancer_type=cancer_type,
            age=random.randint(45, 70),
            gender=random.choice([Gender.MALE, Gender.FEMALE]),
            stage="IV",
            biomarkers=self._generate_biomarkers_standard(cancer_type),
            treatments=treatments,
            ecog=random.choice([ECOGStatus.AMBULATORY, ECOGStatus.LIMITED]),
            demographics=self._generate_demographics_standard(),
            comorbidities=["Fatigue", "Neuropathy", "Anemia"]
        )
    
    def _generate_pregnancy_patient(self) -> Patient:
        """Generate pregnant cancer patient."""
        cancer_type = random.choice(["Breast", "Cervical", "Lymphoma"])
        age = random.randint(25, 40)
        
        return self._create_patient(
            patient_category="edge_case_pregnancy",
            cancer_type=cancer_type,
            age=age,
            gender=Gender.FEMALE,
            stage=random.choice(["I", "II"]),
            biomarkers=[],
            treatments=[],
            ecog=ECOGStatus.FULLY_ACTIVE,
            demographics=self._generate_demographics_standard(),
            comorbidities=["Pregnancy - 2nd trimester"]
        )
    
    def _generate_multi_primary_patient(self) -> Patient:
        """Generate patient with multiple primary cancers."""
        primary_cancer = random.choice(["Breast", "Lung", "Colorectal"])
        secondary_cancer = random.choice(["Prostate", "Bladder", "Melanoma"])
        
        return self._create_patient(
            patient_category="edge_case_multi_primary",
            cancer_type=f"{primary_cancer} + {secondary_cancer}",
            age=random.randint(60, 80),
            gender=Gender.MALE if "Prostate" in secondary_cancer else random.choice([Gender.MALE, Gender.FEMALE]),
            stage="II",  # Primary stage
            biomarkers=[],
            treatments=["Surgery", "Radiation"],
            ecog=ECOGStatus.RESTRICTED,
            demographics=self._generate_demographics_standard(),
            comorbidities=[]
        )
    
    def _generate_borderline_patient(self) -> Patient:
        """Generate borderline eligibility patient."""
        cancer_type = random.choice(["Lung", "Breast", "Colorectal"])
        
        # Borderline values
        age = random.choice([17, 18, 75, 80])  # Age boundaries
        ecog = ECOGStatus.AMBULATORY  # Often a cutoff
        
        # Borderline lab values in comorbidities
        comorbidities = [
            "Creatinine 1.4 mg/dL",  # Just below usual cutoff
            "Platelets 95K",  # Just below 100K cutoff
            "ANC 1.4",  # Just below 1.5 cutoff
        ]
        
        return self._create_patient(
            patient_category="edge_case_borderline",
            cancer_type=cancer_type,
            age=age,
            gender=random.choice([Gender.MALE, Gender.FEMALE]),
            stage=random.choice(["II", "III"]),
            biomarkers=self._generate_biomarkers_standard(cancer_type),
            treatments=["Chemotherapy"],
            ecog=ecog,
            demographics=self._generate_demographics_standard(),
            comorbidities=comorbidities
        )
    
    def _generate_conflicting_biomarkers_patient(self) -> Patient:
        """Generate patient with conflicting biomarker results."""
        cancer_type = "Lung"
        
        # Conflicting biomarkers (usually mutually exclusive)
        biomarkers = [
            Biomarker(name="EGFR", status="positive"),
            Biomarker(name="ALK", status="positive"),  # Conflict!
            Biomarker(name="KRAS", status="positive")  # Triple conflict!
        ]
        
        return self._create_patient(
            patient_category="edge_case_conflicting",
            cancer_type=cancer_type,
            age=random.randint(50, 70),
            gender=random.choice([Gender.MALE, Gender.FEMALE]),
            stage="IV",
            biomarkers=biomarkers,
            treatments=["Chemotherapy"],
            ecog=ECOGStatus.RESTRICTED,
            demographics=self._generate_demographics_standard(),
            comorbidities=[]
        )
    
    def _generate_rural_patient(self) -> Patient:
        """Generate rural/isolated patient."""
        cancer_type = random.choice(["Lung", "Colorectal", "Breast"])
        
        demographics = {
            "race": "White",  # Rural areas often less diverse
            "city": random.choice(["Rural Town", "Remote", "Frontier"]),
            "state": random.choice(["Montana", "Wyoming", "Alaska", "North Dakota"]),
            "distance_to_center": random.randint(200, 500)  # Miles
        }
        
        return self._create_patient(
            patient_category="edge_case_rural",
            cancer_type=cancer_type,
            age=random.randint(50, 75),
            gender=random.choice([Gender.MALE, Gender.FEMALE]),
            stage=random.choice(["III", "IV"]),  # Often diagnosed later
            biomarkers=[],  # Limited testing access
            treatments=["Surgery"],
            ecog=ECOGStatus.RESTRICTED,
            demographics=demographics,
            comorbidities=[]
        )
    
    def _generate_immunocompromised_patient(self) -> Patient:
        """Generate immunocompromised patient."""
        cancer_type = random.choice(["Lymphoma", "Leukemia", "Lung"])
        
        comorbidities = random.choice([
            ["HIV/AIDS", "CD4 count 150"],
            ["Organ transplant recipient", "On immunosuppressants"],
            ["Autoimmune disease", "On biologics"],
            ["Primary immunodeficiency"]
        ])
        
        return self._create_patient(
            patient_category="edge_case_immunocompromised",
            cancer_type=cancer_type,
            age=random.randint(35, 65),
            gender=random.choice([Gender.MALE, Gender.FEMALE]),
            stage=random.choice(["II", "III", "IV"]),
            biomarkers=[],
            treatments=["Chemotherapy - modified dose"],
            ecog=ECOGStatus.AMBULATORY,
            demographics=self._generate_demographics_standard(),
            comorbidities=comorbidities
        )
    
    def _generate_adversarial_patient(self) -> Patient:
        """Generate adversarial patient for robustness testing."""
        attack_type = random.choice([
            "biomarker_spoof",
            "demographic_manipulation",
            "history_falsification",
            "temporal_inconsistency",
            "data_poisoning",
            "boundary_attack",
            "missing_data_exploit",
            "unit_confusion"
        ])
        
        if attack_type == "biomarker_spoof":
            # All positive biomarkers (impossible)
            biomarkers = [
                Biomarker(name=marker, status="positive")
                for marker in ["EGFR", "ALK", "ROS1", "BRAF", "KRAS", "HER2", "PD-L1"]
            ]
            cancer_type = "Lung"
            
        elif attack_type == "demographic_manipulation":
            # Impossible demographics
            demographics = {
                "race": "Unknown123",
                "city": "'; DROP TABLE patients; --",
                "state": "XX",
                "age": -5  # Will be overridden but shows attack
            }
            cancer_type = "Breast"
            biomarkers = []
            
        elif attack_type == "history_falsification":
            # Impossible treatment sequence
            treatments = [
                "Line 5 therapy",
                "Line 1 therapy",  # Out of order
                "Surgery after metastasis",
                "Radiation to all sites"
            ]
            cancer_type = "Colorectal"
            biomarkers = []
            
        elif attack_type == "temporal_inconsistency":
            # Future dates, impossible timelines
            demographics = {
                "diagnosis_date": "2030-01-01",  # Future
                "birth_date": "2025-01-01"  # Future
            }
            cancer_type = "Melanoma"
            biomarkers = []
            
        else:
            # Generic adversarial
            cancer_type = "Unknown"
            biomarkers = []
            demographics = self._generate_demographics_standard()
        
        return self._create_patient(
            patient_category=f"adversarial_{attack_type}",
            cancer_type=cancer_type,
            age=random.randint(0, 120),  # Edge case ages within model limits
            gender=random.choice([Gender.MALE, Gender.FEMALE]),
            stage=random.choice(["0", "V", "Unknown"]),  # Invalid stages
            biomarkers=biomarkers if 'biomarkers' in locals() else [],
            treatments=treatments if 'treatments' in locals() else [],
            ecog=ECOGStatus.DISABLED,  # Often excluded
            demographics=demographics if 'demographics' in locals() else {},
            comorbidities=["Test condition", "NaN", ""]
        )
    
    def _generate_equity_stress_patient(self) -> Patient:
        """Generate patient to stress test equity."""
        equity_type = random.choice([
            "minority_elderly",
            "low_ses",
            "non_english",
            "undocumented",
            "disability",
            "indigenous"
        ])
        
        if equity_type == "minority_elderly":
            demographics = {
                "race": random.choice(["Black", "Native American", "Pacific Islander"]),
                "age": random.randint(70, 85),
                "insurance": "Medicare only",
                "income": "Below poverty line"
            }
            
        elif equity_type == "low_ses":
            demographics = {
                "race": random.choice(["Black", "Hispanic"]),
                "insurance": "Uninsured",
                "employment": "Unemployed",
                "housing": "Unstable",
                "transportation": "None"
            }
            
        elif equity_type == "non_english":
            demographics = {
                "race": random.choice(["Asian", "Hispanic"]),
                "primary_language": random.choice(["Spanish", "Mandarin", "Vietnamese"]),
                "english_proficiency": "Limited",
                "health_literacy": "Low"
            }
            
        elif equity_type == "undocumented":
            demographics = {
                "race": "Hispanic",
                "documentation": "Undocumented",
                "insurance": "None",
                "fear_deportation": True
            }
            
        elif equity_type == "disability":
            comorbidities = random.choice([
                ["Blindness", "Requires assistance"],
                ["Deafness", "ASL only"],
                ["Wheelchair bound", "No vehicle access"],
                ["Cognitive impairment", "Requires caregiver"]
            ])
            demographics = self._generate_demographics_standard()
            
        else:  # indigenous
            demographics = {
                "race": "Native American",
                "tribe": random.choice(["Navajo", "Cherokee", "Sioux"]),
                "reservation": True,
                "distance_to_center": random.randint(300, 600)
            }
            comorbidities = ["Diabetes", "Alcoholism history"]
        
        cancer_type = random.choice(["Lung", "Colorectal", "Breast", "Liver"])
        
        return self._create_patient(
            patient_category=f"equity_{equity_type}",
            cancer_type=cancer_type,
            age=demographics.get("age", random.randint(45, 75)),
            gender=random.choice([Gender.MALE, Gender.FEMALE]),
            stage=random.choice(["III", "IV"]),  # Often diagnosed late
            biomarkers=[],  # Limited testing
            treatments=[],  # Limited access
            ecog=random.choice([ECOGStatus.AMBULATORY, ECOGStatus.LIMITED]),
            demographics=demographics,
            comorbidities=comorbidities if 'comorbidities' in locals() else ["Diabetes", "Hypertension"]
        )
    
    # Helper methods
    
    def _select_cancer_type(self) -> str:
        """Select cancer type based on incidence rates."""
        cancer_types = list(self.epidemiology.CANCER_INCIDENCE.keys())
        weights = list(self.epidemiology.CANCER_INCIDENCE.values())
        return random.choices(cancer_types, weights=weights)[0]
    
    def _select_gender(self, cancer_type: str) -> Gender:
        """Select gender based on cancer type."""
        dist = self.epidemiology.DEMOGRAPHIC_DISTRIBUTION["gender_by_cancer"]
        
        if cancer_type in dist:
            gender_dist = dist[cancer_type]
        else:
            gender_dist = dist["default"]
        
        if random.random() < gender_dist.get("Female", 0.5):
            return Gender.FEMALE
        else:
            return Gender.MALE
    
    def _select_stage_standard(self) -> str:
        """Select cancer stage with realistic distribution."""
        # Approximate US stage distribution at diagnosis
        stages = ["I", "II", "III", "IV"]
        weights = [0.30, 0.35, 0.20, 0.15]
        return random.choices(stages, weights=weights)[0]
    
    def _select_ecog_standard(self, age: int, stage: str) -> ECOGStatus:
        """Select ECOG status based on age and stage."""
        if stage == "IV":
            weights = [0.1, 0.3, 0.4, 0.15, 0.05]  # Worse performance
        elif stage == "III":
            weights = [0.2, 0.4, 0.3, 0.08, 0.02]
        else:
            weights = [0.4, 0.4, 0.15, 0.04, 0.01]
        
        # Adjust for age
        if age > 75:
            # Shift toward worse ECOG
            weights = [w * 0.7 for w in weights[:2]] + [w * 1.3 for w in weights[2:]]
        
        # Normalize
        total = sum(weights)
        weights = [w / total for w in weights]
        
        ecog_values = [
            ECOGStatus.FULLY_ACTIVE,
            ECOGStatus.RESTRICTED,
            ECOGStatus.AMBULATORY,
            ECOGStatus.LIMITED,
            ECOGStatus.DISABLED
        ]
        
        return random.choices(ecog_values, weights=weights)[0]
    
    def _generate_biomarkers_standard(self, cancer_type: str) -> List[Biomarker]:
        """Generate biomarkers based on cancer type prevalence."""
        biomarkers = []
        
        if cancer_type not in self.epidemiology.BIOMARKER_PREVALENCE:
            return biomarkers
        
        prevalence = self.epidemiology.BIOMARKER_PREVALENCE[cancer_type]
        
        for marker, prob in prevalence.items():
            if random.random() < prob:
                # Check co-occurrence rules
                if not self._check_cooccurrence_conflict(marker, biomarkers):
                    status = "positive"
                    if "PD-L1" in marker:
                        # Special handling for PD-L1 levels
                        value = marker
                    else:
                        value = None
                    
                    biomarkers.append(
                        Biomarker(name=marker.split()[0], status=status, value=value)
                    )
        
        return biomarkers
    
    def _check_cooccurrence_conflict(
        self,
        new_marker: str,
        existing: List[Biomarker]
    ) -> bool:
        """Check if new biomarker conflicts with existing."""
        existing_names = [b.name for b in existing]
        
        # Mutually exclusive pairs
        exclusive_pairs = [
            ("EGFR", "ALK"),
            ("EGFR", "KRAS"),
            ("BRCA1", "BRCA2"),
            ("ER+", "Triple-negative"),
            ("PR+", "Triple-negative"),
            ("HER2+", "Triple-negative")
        ]
        
        for pair in exclusive_pairs:
            if new_marker in pair[0] and any(pair[1] in e for e in existing_names):
                return True
            if new_marker in pair[1] and any(pair[0] in e for e in existing_names):
                return True
        
        return False
    
    def _generate_treatment_history_standard(
        self,
        cancer_type: str,
        stage: str
    ) -> List[str]:
        """Generate realistic treatment history."""
        treatments = []
        
        # Surgery often first for early stage
        if stage in ["I", "II"]:
            if random.random() < 0.8:
                treatments.append("Surgery")
        
        # Radiation for local control
        if stage in ["II", "III"]:
            if random.random() < 0.6:
                treatments.append("Radiation")
        
        # Systemic therapy
        if stage in ["III", "IV"] or (stage == "II" and random.random() < 0.4):
            # Cancer-specific treatments
            if cancer_type == "Breast":
                if random.random() < 0.7:
                    treatments.append("Endocrine therapy")
                if random.random() < 0.4:
                    treatments.append("Chemotherapy")
                if random.random() < 0.2:
                    treatments.append("HER2-targeted therapy")
                    
            elif cancer_type == "Lung":
                if random.random() < 0.6:
                    treatments.append("Chemotherapy")
                if random.random() < 0.4:
                    treatments.append("Immunotherapy")
                if random.random() < 0.2:
                    treatments.append("Targeted therapy")
                    
            else:
                if random.random() < 0.6:
                    treatments.append("Chemotherapy")
        
        return treatments
    
    def _generate_demographics_standard(self) -> Dict[str, Any]:
        """Generate standard demographics."""
        race_dist = self.epidemiology.DEMOGRAPHIC_DISTRIBUTION["race"]
        race = random.choices(
            list(race_dist.keys()),
            list(race_dist.values())
        )[0]
        
        region_dist = self.epidemiology.GEOGRAPHIC_DISTRIBUTION
        region = random.choices(
            list(region_dist.keys()),
            list(region_dist.values())
        )[0]
        
        # Map region to states
        region_states = {
            "Northeast": ["New York", "Massachusetts", "Pennsylvania", "New Jersey"],
            "Southeast": ["Florida", "Georgia", "North Carolina", "Virginia"],
            "Midwest": ["Illinois", "Ohio", "Michigan", "Wisconsin"],
            "Southwest": ["Texas", "Arizona", "New Mexico", "Oklahoma"],
            "West": ["California", "Washington", "Oregon", "Colorado"],
            "Rural": ["Montana", "Wyoming", "North Dakota", "South Dakota"]
        }
        
        state = random.choice(region_states[region])
        
        return {
            "race": race,
            "state": state,
            "city": f"{random.choice(['North', 'South', 'East', 'West'])} {random.choice(['ville', 'town', 'city', 'burg'])}",
            "insurance": random.choice(["Private", "Medicare", "Medicaid", "Private+Medicare"])
        }
    
    def _generate_comorbidities(self, age: int, standard: bool = True) -> List[str]:
        """Generate age-appropriate comorbidities."""
        comorbidities = []
        
        # Age-related comorbidity rates
        if age > 65:
            if random.random() < 0.4:
                comorbidities.append("Hypertension")
            if random.random() < 0.25:
                comorbidities.append("Diabetes")
            if random.random() < 0.15:
                comorbidities.append("Heart disease")
            if random.random() < 0.1:
                comorbidities.append("COPD")
        elif age > 50:
            if random.random() < 0.25:
                comorbidities.append("Hypertension")
            if random.random() < 0.15:
                comorbidities.append("Diabetes")
        
        return comorbidities
    
    def _create_patient(
        self,
        patient_category: str,
        cancer_type: str,
        age: int,
        gender: Gender,
        stage: str,
        biomarkers: List[Biomarker],
        treatments: List[str],
        ecog: ECOGStatus,
        demographics: Dict[str, Any],
        comorbidities: List[str]
    ) -> Patient:
        """Create Patient object from generated data."""
        self.patient_id_counter += 1
        patient_id = f"SYNTH_{patient_category}_{self.patient_id_counter:05d}"
        
        # Generate dates (use date() to get date without time)
        diagnosis_date = (datetime.now() - timedelta(days=random.randint(30, 730))).date()
        
        # Calculate BMI (realistic range)
        height_cm = random.gauss(170, 10)
        weight_kg = random.gauss(75, 15)
        bmi = weight_kg / ((height_cm / 100) ** 2)
        
        return Patient(
            patient_id=patient_id,
            name=f"Synthetic Patient {self.patient_id_counter}",
            age=age,
            gender=gender,
            race=demographics.get("race", "Unknown"),
            city=demographics.get("city", "Unknown"),
            state=demographics.get("state", "Unknown"),
            height_cm=height_cm,
            weight_kg=weight_kg,
            bmi=bmi,
            cancer_type=cancer_type,
            cancer_stage=stage,
            cancer_substage=None,
            cancer_grade=random.choice(["Grade 1", "Grade 2", "Grade 3"]),
            initial_diagnosis_date=diagnosis_date,
            diagnosis_month=diagnosis_date.month,
            diagnosis_year=diagnosis_date.year,
            is_recurrence=random.random() < 0.2 if stage in ["III", "IV"] else False,
            recurrence_date=None,
            treatment_stage=random.choice(["neoadjuvant", "adjuvant", "metastatic"]),
            surgeries=["Surgery"] if "Surgery" in treatments else [],
            previous_treatments=treatments,
            current_medications=random.sample(
                ["Aspirin", "Metformin", "Lisinopril", "Atorvastatin"],
                k=random.randint(0, 2)
            ),
            biomarkers_detected=biomarkers,
            biomarkers_ruled_out=[],
            ecog_status=ecog,
            smoking_status=random.choice(["Never", "Former", "Current"]),
            drinking_status=random.choice(["None", "Social", "Heavy"]),
            other_conditions=comorbidities,
            family_history=random.choice([
                "No family history",
                "Mother had breast cancer",
                "Father had prostate cancer",
                "Multiple family members with cancer"
            ]),
            patient_intent=random.choice(["Curative", "Palliative"])
        )

