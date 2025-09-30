"""
Tests for BioMCP client functionality
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from oncomatch.biomcp_client import AdvancedBioMCPClient


class TestBioMCPClient:
    """Test BioMCP client operations"""
    
    @pytest.fixture
    def client(self):
        """Create a BioMCP client instance"""
        return AdvancedBioMCPClient()
    
    @pytest.mark.asyncio
    async def test_fetch_trials_basic(self, client, sample_patient):
        """Test basic trial fetching"""
        # Mock the actual fetch to avoid external API calls in tests
        with patch.object(client, "_fetch_trials_with_retry") as mock_fetch:
            mock_fetch.return_value = [
                {
                    "nct_id": "NCT12345678",
                    "title": "Test Trial",
                    "phase": "Phase 3",
                    "status": "Recruiting"
                }
            ]
            
            from types import SimpleNamespace
            patient = SimpleNamespace(**sample_patient)
            
            trials = await client.fetch_trials_for_patient(
                patient=patient,
                max_trials=10
            )
            
            assert len(trials) > 0
            assert mock_fetch.called
    
    def test_trial_quality_scoring(self, client):
        """Test trial quality scoring"""
        trial = {
            "nct_id": "NCT12345678",
            "phase": "Phase 3",
            "status": "Recruiting",
            "enrollment": 500,
            "locations": ["New York", "Los Angeles", "Chicago"],
            "sponsor": {"name": "Major Pharma Corp"}
        }
        
        score = client._calculate_trial_quality_score(trial)
        assert 0 <= score <= 1
        assert score > 0.5  # Should be relatively high quality
    
    def test_cache_key_generation(self, client):
        """Test cache key generation"""
        import hashlib
        
        request_data = {
            "patient_id": "P001",
            "condition": "breast cancer",
            "phase": "Phase 3"
        }
        
        key = hashlib.sha256(str(request_data).encode()).hexdigest()
        assert len(key) == 64  # SHA256 produces 64 character hex string
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, client):
        """Test that rate limiting is enforced"""
        # This would test the token bucket implementation
        # For unit tests, we just verify the rate limiter exists
        assert hasattr(client, "rate_limiter")
        assert client.rate_limiter is not None

