"""
Deep trial analysis module for extracting and scoring trial characteristics.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from oncomatch.models import ClinicalTrial, TrialPhase, ECOGStatus


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrialAnalyzer:
    """Analyze clinical trials for detailed eligibility and characteristics."""
    
    def __init__(self):
        self.criteria_patterns = self._compile_criteria_patterns()
        self.biomarker_patterns = self._compile_biomarker_patterns()
        self.safety_keywords = self._load_safety_keywords()
    
    def _compile_criteria_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for common eligibility criteria."""
        return {
            'age_min': re.compile(r'age[s]?\s*(?:>=?|≥|at least|minimum)\s*(\d+)', re.IGNORECASE),
            'age_max': re.compile(r'age[s]?\s*(?:<=?|≤|maximum|up to)\s*(\d+)', re.IGNORECASE),
            'age_range': re.compile(r'age[s]?\s*(?:between|from)\s*(\d+)\s*(?:to|and|-)\s*(\d+)', re.IGNORECASE),
            'ecog': re.compile(r'ECOG\s*(?:performance status|PS)?\s*(?:of|<=?|≤)?\s*(\d+)(?:\s*(?:to|-)\s*(\d+))?', re.IGNORECASE),
            'karnofsky': re.compile(r'Karnofsky\s*(?:>=?|≥)\s*(\d+)', re.IGNORECASE),
            'life_expectancy': re.compile(r'life expectancy\s*(?:of|>=?|≥|at least)\s*(\d+)\s*(weeks?|months?)', re.IGNORECASE),
            'prior_therapy': re.compile(r'(?:no more than|maximum|up to|≤)\s*(\d+)\s*(?:prior|previous)\s*(?:line|regimen|therapy)', re.IGNORECASE),
            'washout': re.compile(r'(?:at least|minimum|≥)\s*(\d+)\s*(?:days?|weeks?|months?)\s*(?:since|from|after)', re.IGNORECASE),
            'measurable_disease': re.compile(r'measurable disease|RECIST', re.IGNORECASE),
            'brain_mets': re.compile(r'brain metastas[ei]s|CNS metastas[ei]s|intracranial', re.IGNORECASE),
            'pregnancy': re.compile(r'pregnan(?:t|cy)|nursing|breastfeeding', re.IGNORECASE),
            'organ_function': re.compile(r'adequate (?:organ|liver|renal|kidney|bone marrow) function', re.IGNORECASE)
        }
    
    def _compile_biomarker_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for biomarker mentions."""
        return {
            'HER2': re.compile(r'\bHER2(?:[+-]|\s+(?:positive|negative|amplified))?\b', re.IGNORECASE),
            'ER': re.compile(r'\bER(?:[+-]|\s+(?:positive|negative))?\b|\bestrogen receptor', re.IGNORECASE),
            'PR': re.compile(r'\bPR(?:[+-]|\s+(?:positive|negative))?\b|\bprogesterone receptor', re.IGNORECASE),
            'EGFR': re.compile(r'\bEGFR\b|\bepidermal growth factor receptor', re.IGNORECASE),
            'ALK': re.compile(r'\bALK\b|\banaplastic lymphoma kinase', re.IGNORECASE),
            'ROS1': re.compile(r'\bROS1\b', re.IGNORECASE),
            'BRAF': re.compile(r'\bBRAF(?:\s+V600[EK]?)?\b', re.IGNORECASE),
            'KRAS': re.compile(r'\bKRAS(?:\s+G12[CD]?)?\b', re.IGNORECASE),
            'NRAS': re.compile(r'\bNRAS\b', re.IGNORECASE),
            'PD-L1': re.compile(r'\bPD-?L1\b|\bprogrammed death ligand', re.IGNORECASE),
            'PD-1': re.compile(r'\bPD-?1\b|\bprogrammed death', re.IGNORECASE),
            'BRCA': re.compile(r'\bBRCA[12]?\b', re.IGNORECASE),
            'MSI': re.compile(r'\bMSI(?:-H)?\b|\bmicrosatellite instability', re.IGNORECASE),
            'MMR': re.compile(r'\b(?:d)?MMR\b|\bmismatch repair', re.IGNORECASE),
            'TMB': re.compile(r'\bTMB\b|\btumor mutational burden', re.IGNORECASE),
            'HRD': re.compile(r'\bHRD\b|\bhomologous recombination deficiency', re.IGNORECASE),
            'PIK3CA': re.compile(r'\bPIK3CA\b', re.IGNORECASE),
            'FGFR': re.compile(r'\bFGFR[1-4]?\b', re.IGNORECASE),
            'MET': re.compile(r'\b(?:c-)?MET\b', re.IGNORECASE),
            'RET': re.compile(r'\bRET\b', re.IGNORECASE)
        }
    
    def _load_safety_keywords(self) -> List[str]:
        """Load keywords indicating safety concerns."""
        return [
            'cardiac', 'heart', 'QT', 'ejection fraction', 'LVEF',
            'hepatic', 'liver', 'cirrhosis', 'hepatitis',
            'renal', 'kidney', 'creatinine', 'dialysis',
            'pulmonary', 'lung function', 'interstitial',
            'bleeding', 'hemorrhage', 'anticoagulation',
            'infection', 'HIV', 'immunocompromised',
            'autoimmune', 'transplant',
            'psychiatric', 'suicide', 'psychosis',
            'seizure', 'epilepsy',
            'uncontrolled', 'active'
        ]
    
    async def analyze_trial(self, trial: ClinicalTrial) -> Dict[str, Any]:
        """
        Perform deep analysis of a clinical trial.
        
        Returns:
            Dictionary with analysis results including:
            - extracted_criteria: Structured eligibility criteria
            - biomarkers_mentioned: List of biomarkers found
            - complexity_score: Trial complexity (0-10)
            - phase_appropriateness: Score for phase appropriateness
            - safety_flags: List of safety concerns
        """
        analysis = {
            'nct_id': trial.nct_id,
            'extracted_criteria': {},
            'biomarkers_mentioned': [],
            'complexity_score': 0,
            'phase_appropriateness': 0,
            'safety_flags': [],
            'geographic_accessibility': 0,
            'innovation_score': 0
        }
        
        # Extract structured criteria
        analysis['extracted_criteria'] = self._extract_structured_criteria(trial)
        
        # Identify biomarkers
        analysis['biomarkers_mentioned'] = self._identify_biomarkers(trial)
        
        # Calculate complexity score
        analysis['complexity_score'] = self._calculate_complexity_score(trial)
        
        # Assess phase appropriateness
        analysis['phase_appropriateness'] = self._assess_phase_appropriateness(trial)
        
        # Identify safety flags
        analysis['safety_flags'] = self._identify_safety_flags(trial)
        
        # Calculate geographic accessibility
        analysis['geographic_accessibility'] = self._calculate_geographic_accessibility(trial)
        
        # Calculate innovation score
        analysis['innovation_score'] = self._calculate_innovation_score(trial)
        
        # Don't try to store analysis in trial object (it's a Pydantic model)
        # Return analysis dictionary
        return analysis
    
    def _extract_structured_criteria(self, trial: ClinicalTrial) -> Dict[str, Any]:
        """Extract structured criteria from free text."""
        criteria = {}
        
        # Combine all criteria text
        all_text = ' '.join(trial.eligibility.inclusion_criteria + 
                          trial.eligibility.exclusion_criteria)
        
        # Age requirements
        age_min_match = self.criteria_patterns['age_min'].search(all_text)
        if age_min_match:
            criteria['min_age'] = int(age_min_match.group(1))
        
        age_max_match = self.criteria_patterns['age_max'].search(all_text)
        if age_max_match:
            criteria['max_age'] = int(age_max_match.group(1))
        
        age_range_match = self.criteria_patterns['age_range'].search(all_text)
        if age_range_match:
            criteria['min_age'] = int(age_range_match.group(1))
            criteria['max_age'] = int(age_range_match.group(2))
        
        # ECOG status
        ecog_match = self.criteria_patterns['ecog'].search(all_text)
        if ecog_match:
            max_ecog = int(ecog_match.group(1))
            if ecog_match.group(2):  # Range specified
                min_ecog = max_ecog
                max_ecog = int(ecog_match.group(2))
                criteria['ecog_range'] = (min_ecog, max_ecog)
            else:
                criteria['max_ecog'] = max_ecog
        
        # Prior therapy limit
        prior_therapy_match = self.criteria_patterns['prior_therapy'].search(all_text)
        if prior_therapy_match:
            criteria['max_prior_therapies'] = int(prior_therapy_match.group(1))
        
        # Life expectancy
        life_exp_match = self.criteria_patterns['life_expectancy'].search(all_text)
        if life_exp_match:
            value = int(life_exp_match.group(1))
            unit = life_exp_match.group(2).lower()
            if 'month' in unit:
                value *= 4  # Convert to weeks
            criteria['min_life_expectancy_weeks'] = value
        
        # Measurable disease requirement
        if self.criteria_patterns['measurable_disease'].search(all_text):
            criteria['requires_measurable_disease'] = True
        
        # Brain metastases
        brain_mets_text = self.criteria_patterns['brain_mets'].findall(all_text)
        if brain_mets_text:
            # Check if allowed or excluded
            for text in brain_mets_text:
                context = all_text[max(0, all_text.find(text) - 50):all_text.find(text) + 50]
                if any(word in context.lower() for word in ['no ', 'without', 'exclude', 'must not']):
                    criteria['excludes_brain_mets'] = True
                elif any(word in context.lower() for word in ['allowed', 'permitted', 'stable', 'treated']):
                    criteria['allows_brain_mets'] = True
        
        # Pregnancy exclusion
        if self.criteria_patterns['pregnancy'].search(all_text):
            criteria['excludes_pregnancy'] = True
        
        # Organ function requirements
        if self.criteria_patterns['organ_function'].search(all_text):
            criteria['requires_adequate_organ_function'] = True
        
        return criteria
    
    def _identify_biomarkers(self, trial: ClinicalTrial) -> List[str]:
        """Identify biomarkers mentioned in trial."""
        biomarkers = set()
        
        # Check in eligibility criteria
        all_text = ' '.join(trial.eligibility.inclusion_criteria + 
                          trial.eligibility.exclusion_criteria)
        
        # Also check title and summary
        if trial.title:
            all_text += ' ' + trial.title
        if trial.brief_summary:
            all_text += ' ' + trial.brief_summary
        
        # Search for each biomarker
        for biomarker_name, pattern in self.biomarker_patterns.items():
            if pattern.search(all_text):
                biomarkers.add(biomarker_name)
        
        # Also add from structured eligibility
        biomarkers.update(trial.eligibility.required_biomarkers)
        biomarkers.update(trial.eligibility.excluded_biomarkers)
        
        return sorted(list(biomarkers))
    
    def _calculate_complexity_score(self, trial: ClinicalTrial) -> float:
        """Calculate trial complexity score (0-10)."""
        score = 0.0
        
        # Phase complexity
        if trial.phase:
            if 'Phase 1' in trial.phase.value:
                score += 3  # Early phase = more complex
            elif 'Phase 2' in trial.phase.value:
                score += 2
            elif 'Phase 3' in trial.phase.value:
                score += 1
        
        # Number of eligibility criteria
        num_criteria = len(trial.eligibility.inclusion_criteria) + len(trial.eligibility.exclusion_criteria)
        if num_criteria > 20:
            score += 3
        elif num_criteria > 10:
            score += 2
        elif num_criteria > 5:
            score += 1
        
        # Biomarker requirements
        num_biomarkers = len(trial.eligibility.required_biomarkers)
        if num_biomarkers > 2:
            score += 2
        elif num_biomarkers > 0:
            score += 1
        
        # Multi-arm or combination therapy
        if len(trial.interventions) > 2:
            score += 1
        
        # Normalize to 0-10
        return min(score, 10.0)
    
    def _assess_phase_appropriateness(self, trial: ClinicalTrial) -> float:
        """Assess phase appropriateness (0-1 score)."""
        if not trial.phase:
            return 0.5  # Unknown phase
        
        # Phase 2 and 3 are generally most appropriate
        if 'Phase 2' in trial.phase.value or 'Phase 3' in trial.phase.value:
            return 1.0
        elif 'Phase 1/Phase 2' in trial.phase.value:
            return 0.8
        elif 'Phase 1' in trial.phase.value:
            return 0.6  # More experimental
        elif 'Phase 4' in trial.phase.value:
            return 0.7  # Post-market
        else:
            return 0.5
    
    def _identify_safety_flags(self, trial: ClinicalTrial) -> List[str]:
        """Identify potential safety concerns in eligibility criteria."""
        flags = []
        
        # Check exclusion criteria for safety issues
        exclusion_text = ' '.join(trial.eligibility.exclusion_criteria).lower()
        
        for keyword in self.safety_keywords:
            if keyword in exclusion_text:
                # Find the specific criterion mentioning this
                for criterion in trial.eligibility.exclusion_criteria:
                    if keyword in criterion.lower():
                        flags.append(f"{keyword.title()}: {criterion[:100]}")
                        break
        
        # Check for strict ECOG requirements
        if hasattr(trial.eligibility, 'max_ecog') and trial.eligibility.max_ecog <= 1:
            flags.append("Strict performance status requirement (ECOG ≤1)")
        
        # Check for brain metastases exclusion
        if hasattr(trial.eligibility, 'excludes_brain_mets') and trial.eligibility.excludes_brain_mets:
            flags.append("Excludes brain metastases")
        
        return flags[:5]  # Limit to top 5 flags
    
    def _calculate_geographic_accessibility(self, trial: ClinicalTrial) -> float:
        """Calculate geographic accessibility score."""
        if not trial.locations:
            return 0.0
        
        # Score based on number of locations
        num_locations = len(trial.locations)
        
        if num_locations >= 10:
            return 1.0
        elif num_locations >= 5:
            return 0.8
        elif num_locations >= 3:
            return 0.6
        elif num_locations >= 2:
            return 0.4
        else:
            return 0.2
    
    def _calculate_innovation_score(self, trial: ClinicalTrial) -> float:
        """Calculate innovation score based on trial characteristics."""
        score = 0.0
        
        # Check for innovative keywords in interventions
        innovative_keywords = [
            'CAR-T', 'cell therapy', 'gene therapy', 'immunotherapy',
            'checkpoint', 'antibody-drug conjugate', 'ADC', 'bispecific',
            'PROTAC', 'RNA', 'vaccine', 'oncolytic virus', 'nanoparticle'
        ]
        
        intervention_text = ' '.join(trial.interventions).lower()
        title_text = (trial.title or '').lower()
        
        for keyword in innovative_keywords:
            if keyword.lower() in intervention_text or keyword.lower() in title_text:
                score += 0.2
        
        # Biomarker-driven trials are more innovative
        if trial.eligibility.required_biomarkers:
            score += 0.2
        
        # First-in-human or early phase
        if trial.phase and 'Early Phase 1' in trial.phase.value:
            score += 0.3
        elif trial.phase and 'Phase 1' in trial.phase.value:
            score += 0.2
        
        # Combination therapies
        if len(trial.interventions) > 1:
            score += 0.1
        
        return min(score, 1.0)
    
    def score_trial_quality(self, trial: ClinicalTrial, analysis: Optional[Dict] = None) -> float:
        """Calculate overall trial quality score."""
        if analysis is None:
            # Default scoring without analysis
            return 0.5
        
        
        # Weight different factors
        quality_score = (
            analysis['phase_appropriateness'] * 0.3 +
            analysis['geographic_accessibility'] * 0.2 +
            analysis['innovation_score'] * 0.2 +
            (1.0 - analysis['complexity_score'] / 10.0) * 0.2 +
            (1.0 - len(analysis['safety_flags']) / 10.0) * 0.1
        )
        
        return quality_score

