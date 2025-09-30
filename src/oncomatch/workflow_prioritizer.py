"""
Workflow prioritization system for clinical trial matching.

This module handles urgency-based workflow prioritization, ensuring that
high-urgency patients get expedited review and appropriate clinical alerts.
Urgency affects workflow, NOT model quality - all patients deserve the best analysis.
"""

import logging
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
from collections import deque

from oncomatch.models import Patient, MatchResult

logger = logging.getLogger(__name__)


class PriorityLevel(str, Enum):
    """Clinical priority levels for patient workflow."""
    CRITICAL = "critical"  # Immediate attention needed
    HIGH = "high"          # Expedited review (24-48h)
    MODERATE = "moderate"  # Priority review (2-3 days)
    STANDARD = "standard"  # Normal workflow (5-7 days)


class NotificationType(str, Enum):
    """Types of clinical notifications."""
    CARE_TEAM = "care_team"
    TRIAL_COORDINATOR = "trial_coordinator"
    PATIENT_ADVOCATE = "patient_advocate"
    PHARMACY = "pharmacy"
    SOCIAL_WORKER = "social_worker"


@dataclass
class WorkflowAction:
    """Represents a workflow action for a patient."""
    patient_id: str
    priority_level: PriorityLevel
    review_timeline: str
    notifications: List[NotificationType] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    urgency_score: float = 0.0
    clinical_indicators: List[str] = field(default_factory=list)


class WorkflowPrioritizer:
    """
    Manages patient workflow prioritization based on clinical urgency.
    
    This system ensures that patients with high clinical needs get
    expedited processing while maintaining quality for all patients.
    """
    
    def __init__(self):
        self.queue_critical = deque()
        self.queue_high = deque()
        self.queue_moderate = deque()
        self.queue_standard = deque()
        self.processed_patients = {}
        self.notifications_sent = {}
    
    def calculate_urgency(self, patient: Patient) -> float:
        """
        Calculate clinical urgency score (0.0 - 1.0).
        
        Args:
            patient: Patient data
            
        Returns:
            Urgency score between 0 and 1
        """
        urgency = 0.0
        indicators = []
        
        # Performance status is most critical
        if patient.ecog_status:
            if patient.ecog_status.value >= 4:
                urgency += 0.5
                indicators.append("ECOG 4 (bedridden)")
            elif patient.ecog_status.value == 3:
                urgency += 0.3
                indicators.append("ECOG 3 (limited self-care)")
            elif patient.ecog_status.value == 2:
                urgency += 0.1
                indicators.append("ECOG 2")
        
        # Disease stage
        if patient.cancer_stage == "IV":
            urgency += 0.3
            indicators.append("Stage IV (metastatic)")
        elif patient.cancer_stage.startswith("III"):
            urgency += 0.1
            indicators.append(f"Stage {patient.cancer_stage}")
        
        # Treatment intent
        if patient.patient_intent == "Palliative":
            urgency += 0.2
            indicators.append("Palliative care")
        
        # Recurrence
        if patient.is_recurrence:
            urgency += 0.1
            indicators.append("Recurrent disease")
        
        # Aggressive histology (if available)
        if patient.cancer_type:
            aggressive_types = [
                "small cell", "triple negative", "glioblastoma",
                "pancreatic", "acute leukemia"
            ]
            if any(agg in patient.cancer_type.lower() for agg in aggressive_types):
                urgency += 0.1
                indicators.append("Aggressive histology")
        
        logger.info(f"Patient {patient.patient_id} urgency: {min(urgency, 1.0):.2f} - {', '.join(indicators)}")
        
        return min(urgency, 1.0)
    
    def prioritize_patient(
        self,
        patient: Patient,
        match_results: Optional[List[MatchResult]] = None
    ) -> WorkflowAction:
        """
        Determine workflow prioritization for a patient.
        
        Args:
            patient: Patient data
            match_results: Optional trial matching results
            
        Returns:
            WorkflowAction with prioritization details
        """
        urgency = self.calculate_urgency(patient)
        
        action = WorkflowAction(
            patient_id=patient.patient_id,
            urgency_score=urgency,
            priority_level=PriorityLevel.STANDARD,
            review_timeline="5-7 business days",
            clinical_indicators=[]
        )
        
        # Determine priority level and actions
        if urgency >= 0.8:
            # CRITICAL: Immediate attention needed
            action.priority_level = PriorityLevel.CRITICAL
            action.review_timeline = "Same day"
            action.notifications = [
                NotificationType.CARE_TEAM,
                NotificationType.TRIAL_COORDINATOR,
                NotificationType.PATIENT_ADVOCATE
            ]
            action.recommendations = [
                "Schedule immediate consultation",
                "Consider compassionate use programs",
                "Evaluate for expedited trial enrollment",
                "Assess supportive care needs",
                "Consider hospice referral if appropriate"
            ]
            
        elif urgency >= 0.6:
            # HIGH: Expedited review needed
            action.priority_level = PriorityLevel.HIGH
            action.review_timeline = "24-48 hours"
            action.notifications = [
                NotificationType.TRIAL_COORDINATOR,
                NotificationType.CARE_TEAM
            ]
            action.recommendations = [
                "Expedite trial screening",
                "Schedule urgent follow-up",
                "Consider expanded access programs",
                "Evaluate symptom management needs"
            ]
            
        elif urgency >= 0.4:
            # MODERATE: Priority review
            action.priority_level = PriorityLevel.MODERATE
            action.review_timeline = "2-3 business days"
            action.notifications = [NotificationType.TRIAL_COORDINATOR]
            action.recommendations = [
                "Prioritize trial matching review",
                "Schedule follow-up within week"
            ]
        
        # Add specific recommendations based on patient factors
        if patient.ecog_status and patient.ecog_status.value >= 3:
            action.recommendations.append("Consider trials with flexible performance status criteria")
            action.recommendations.append("Evaluate for home health services")
            if NotificationType.SOCIAL_WORKER not in action.notifications:
                action.notifications.append(NotificationType.SOCIAL_WORKER)
        
        if patient.cancer_stage == "IV":
            action.recommendations.append("Focus on trials allowing prior systemic therapy")
            action.recommendations.append("Consider quality of life as primary endpoint")
        
        if patient.patient_intent == "Palliative":
            action.recommendations.append("Prioritize symptom control trials")
            action.recommendations.append("Consider best supportive care trials")
        
        # Handle trial-specific recommendations
        if match_results:
            eligible_trials = [r for r in match_results if r.is_eligible]
            if eligible_trials:
                if action.priority_level in [PriorityLevel.CRITICAL, PriorityLevel.HIGH]:
                    action.recommendations.append(
                        f"Fast-track screening for {len(eligible_trials)} eligible trials"
                    )
            else:
                action.recommendations.append("Consider expanded eligibility criteria")
                action.recommendations.append("Evaluate for basket trials")
        
        # Add to appropriate queue
        self._add_to_queue(action)
        
        return action
    
    def _add_to_queue(self, action: WorkflowAction):
        """Add patient to appropriate priority queue."""
        if action.priority_level == PriorityLevel.CRITICAL:
            self.queue_critical.append(action)
            logger.critical(f"CRITICAL priority patient {action.patient_id} added to queue")
        elif action.priority_level == PriorityLevel.HIGH:
            self.queue_high.append(action)
            logger.warning(f"HIGH priority patient {action.patient_id} added to queue")
        elif action.priority_level == PriorityLevel.MODERATE:
            self.queue_moderate.append(action)
            logger.info(f"MODERATE priority patient {action.patient_id} added to queue")
        else:
            self.queue_standard.append(action)
            logger.info(f"Standard priority patient {action.patient_id} added to queue")
    
    async def send_notifications(self, action: WorkflowAction):
        """
        Send clinical notifications for high-priority patients.
        
        This is a placeholder for integration with hospital notification systems.
        """
        if not action.notifications:
            return
        
        for notification_type in action.notifications:
            logger.info(f"Sending {notification_type.value} notification for patient {action.patient_id}")
            # In production, integrate with:
            # - Hospital paging systems
            # - Secure messaging (Epic InBasket, Cerner Message Center)
            # - Email alerts
            # - SMS for critical cases
            
            # Track notifications
            if action.patient_id not in self.notifications_sent:
                self.notifications_sent[action.patient_id] = []
            self.notifications_sent[action.patient_id].append({
                "type": notification_type.value,
                "timestamp": datetime.now(),
                "priority": action.priority_level.value
            })
    
    def get_next_patient(self) -> Optional[WorkflowAction]:
        """
        Get next patient from queues based on priority.
        
        Returns:
            Next patient action or None if queues are empty
        """
        if self.queue_critical:
            return self.queue_critical.popleft()
        elif self.queue_high:
            return self.queue_high.popleft()
        elif self.queue_moderate:
            return self.queue_moderate.popleft()
        elif self.queue_standard:
            return self.queue_standard.popleft()
        return None
    
    def get_queue_stats(self) -> Dict[str, int]:
        """Get current queue statistics."""
        return {
            "critical": len(self.queue_critical),
            "high": len(self.queue_high),
            "moderate": len(self.queue_moderate),
            "standard": len(self.queue_standard),
            "total": (
                len(self.queue_critical) + 
                len(self.queue_high) + 
                len(self.queue_moderate) + 
                len(self.queue_standard)
            )
        }
    
    def generate_workflow_report(self) -> Dict[str, Any]:
        """Generate workflow prioritization report."""
        return {
            "queue_stats": self.get_queue_stats(),
            "notifications_sent": len(self.notifications_sent),
            "patients_processed": len(self.processed_patients),
            "critical_patients": [
                action.patient_id for action in self.queue_critical
            ],
            "high_priority_patients": [
                action.patient_id for action in self.queue_high
            ],
            "recommendations": {
                "staffing": "Increase coordinators" if len(self.queue_critical) > 5 else "Normal",
                "alerts": "Active" if len(self.queue_critical) > 0 else "Standard",
                "review_capacity": "Exceeded" if self.get_queue_stats()["total"] > 50 else "Normal"
            }
        }


# Global instance for use across the application
workflow_prioritizer = WorkflowPrioritizer()
