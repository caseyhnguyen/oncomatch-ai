"""
Tests for LLM JSON contract validation and repair.
"""

import json
import pytest
from pydantic import ValidationError

from src.modernized_llm_ranker import LLMEligibilityResponse, LLMComparisonResponse


class TestLLMJSONContract:
    """Test JSON parsing and validation."""
    
    def test_valid_eligibility_response(self):
        """Test valid eligibility response."""
        valid_json = {
            "is_eligible": True,
            "confidence": 0.85,
            "eligibility_score": 0.8,
            "biomarker_score": 0.9,
            "safety_score": 0.7,
            "match_reasons": [
                {
                    "criterion": "Age ≥ 18",
                    "matched": True,
                    "explanation": "Patient is 45 years old",
                    "confidence": 1.0,
                    "category": "inclusion"
                }
            ],
            "summary": "Patient meets key eligibility criteria",
            "safety_concerns": [],
            "warnings": []
        }
        
        response = LLMEligibilityResponse(**valid_json)
        assert response.is_eligible is True
        assert response.confidence == 0.85
        assert len(response.match_reasons) == 1
    
    def test_malformed_json_repair(self):
        """Test repair of malformed JSON."""
        # JSON with trailing comma
        malformed = '{"is_eligible": true, "confidence": 0.8,}'
        
        # Remove trailing comma
        import re
        repaired = re.sub(r',\s*}', '}', malformed)
        parsed = json.loads(repaired)
        
        assert parsed["is_eligible"] is True
        assert parsed["confidence"] == 0.8
    
    def test_missing_required_fields(self):
        """Test handling of missing required fields."""
        incomplete_json = {
            "is_eligible": True,
            "confidence": 0.8
            # Missing other required fields
        }
        
        # Should raise validation error
        with pytest.raises(ValidationError):
            LLMEligibilityResponse(**incomplete_json)
    
    def test_invalid_value_ranges(self):
        """Test validation of value ranges."""
        invalid_json = {
            "is_eligible": True,
            "confidence": 1.5,  # Invalid: > 1.0
            "eligibility_score": -0.1,  # Invalid: < 0.0
            "biomarker_score": 0.5,
            "safety_score": 0.5,
            "match_reasons": [],
            "summary": "Test",
            "safety_concerns": [],
            "warnings": []
        }
        
        with pytest.raises(ValidationError) as exc_info:
            LLMEligibilityResponse(**invalid_json)
        
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("confidence",) for e in errors)
        assert any(e["loc"] == ("eligibility_score",) for e in errors)
    
    def test_markdown_fence_removal(self):
        """Test removal of markdown fences."""
        fenced_json = """```json
{
    "is_eligible": true,
    "confidence": 0.9,
    "eligibility_score": 0.85,
    "biomarker_score": 0.8,
    "safety_score": 0.9,
    "match_reasons": [],
    "summary": "Good match",
    "safety_concerns": [],
    "warnings": []
}
```"""
        
        # Remove fences
        if fenced_json.startswith("```json"):
            clean = fenced_json[7:]
        elif fenced_json.startswith("```"):
            clean = fenced_json[3:]
        else:
            clean = fenced_json
            
        if clean.endswith("```"):
            clean = clean[:-3]
        
        clean = clean.strip()
        parsed = json.loads(clean)
        
        response = LLMEligibilityResponse(**parsed)
        assert response.is_eligible is True
        assert response.confidence == 0.9
    
    def test_comparison_response_validation(self):
        """Test comparison response validation."""
        valid_json = {
            "rankings": [
                {
                    "rank": 1,
                    "nct_id": "NCT123456",
                    "overall_score": 0.9,
                    "rationale": "Best match"
                },
                {
                    "rank": 2,
                    "nct_id": "NCT234567",
                    "overall_score": 0.7,
                    "rationale": "Good match"
                }
            ],
            "top_recommendation": {
                "nct_id": "NCT123456",
                "key_advantages": ["Biomarker match", "Phase 2"],
                "considerations": ["Distance to site"]
            },
            "comparison_summary": "NCT123456 is the best match due to biomarker alignment"
        }
        
        response = LLMComparisonResponse(**valid_json)
        assert len(response.rankings) == 2
        assert response.rankings[0]["nct_id"] == "NCT123456"
        assert response.top_recommendation["nct_id"] == "NCT123456"
    
    def test_conservative_fallback_on_parse_error(self):
        """Test conservative fallback when parsing fails."""
        completely_invalid = "This is not JSON at all"
        
        # When parsing fails completely, create conservative response
        try:
            json.loads(completely_invalid)
        except json.JSONDecodeError:
            # Create conservative fallback
            fallback = LLMEligibilityResponse(
                is_eligible=False,
                confidence=0.3,
                eligibility_score=0.3,
                biomarker_score=0.0,
                safety_score=0.5,
                match_reasons=[],
                summary="Unable to evaluate due to technical error",
                safety_concerns=["Automated evaluation failed"],
                warnings=["Manual review required"]
            )
            
            assert fallback.is_eligible is False
            assert fallback.confidence == 0.3
            assert "Manual review required" in fallback.warnings
    
    def test_nested_structure_validation(self):
        """Test validation of nested structures."""
        json_with_nested = {
            "is_eligible": True,
            "confidence": 0.8,
            "eligibility_score": 0.75,
            "biomarker_score": 0.85,
            "safety_score": 0.9,
            "match_reasons": [
                {
                    "criterion": "EGFR mutation",
                    "matched": True,
                    "explanation": "Patient has EGFR L858R mutation",
                    "confidence": 0.95,
                    "category": "biomarker"
                },
                {
                    "criterion": "ECOG ≤ 2",
                    "matched": True,
                    "explanation": "Patient ECOG is 1",
                    "confidence": 1.0,
                    "category": "inclusion"
                }
            ],
            "summary": "Strong biomarker match with good performance status",
            "safety_concerns": ["Monitor for skin toxicity"],
            "warnings": []
        }
        
        response = LLMEligibilityResponse(**json_with_nested)
        assert len(response.match_reasons) == 2
        assert response.match_reasons[0]["category"] == "biomarker"
        assert response.match_reasons[1]["criterion"] == "ECOG ≤ 2"
        assert len(response.safety_concerns) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

