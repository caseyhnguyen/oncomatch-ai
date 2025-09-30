"""
Configuration management for OncoMatch AI
Reads all configuration from environment variables
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Central configuration from environment variables"""
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
    
    # Medical Model APIs
    TRIALGPT_BASE_URL: Optional[str] = os.getenv("TRIALGPT_BASE_URL")
    TRIALGPT_API_KEY: Optional[str] = os.getenv("TRIALGPT_API_KEY")
    BIOMCP_API_KEY: Optional[str] = os.getenv("BIOMCP_API_KEY")
    MEDITRON_API_KEY: Optional[str] = os.getenv("MEDITRON_API_KEY")
    MEDPALM_SERVICE_ACCOUNT: Optional[str] = os.getenv("MEDPALM_SERVICE_ACCOUNT")
    
    # BioMCP Configuration
    NCI_API_KEY: Optional[str] = os.getenv("NCI_API_KEY")
    CBIO_TOKEN: Optional[str] = os.getenv("CBIO_TOKEN")
    ALPHAGENOME_API_KEY: Optional[str] = os.getenv("ALPHAGENOME_API_KEY")
    
    # System Configuration
    LLM_BUDGET_USD: float = float(os.getenv("LLM_BUDGET_USD", "0.05"))
    LLM_BUDGET_MS: int = int(os.getenv("LLM_BUDGET_MS", "15000"))
    LLM_MAX_CONCURRENCY: int = int(os.getenv("LLM_MAX_CONCURRENCY", "5"))
    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    
    # Modes
    DEFAULT_MODE: str = os.getenv("DEFAULT_MODE", "balanced")
    ENABLE_SAFETY_CHECK: bool = os.getenv("ENABLE_SAFETY_CHECK", "true").lower() == "true"
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: Optional[str] = os.getenv("LOG_FILE")
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is present"""
        warnings = []
        
        if not cls.OPENAI_API_KEY and not cls.ANTHROPIC_API_KEY and not cls.GOOGLE_API_KEY:
            warnings.append("No LLM API keys configured (OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY)")
        
        for warning in warnings:
            print(f"Warning: {warning}")
        
        return len(warnings) == 0
    
    @classmethod
    def get_available_providers(cls) -> list:
        """Get list of available LLM providers based on configured API keys"""
        providers = []
        if cls.OPENAI_API_KEY:
            providers.append("openai")
        if cls.ANTHROPIC_API_KEY:
            providers.append("anthropic")
        if cls.GOOGLE_API_KEY:
            providers.append("google")
        if cls.TRIALGPT_API_KEY:
            providers.append("trialgpt")
        return providers

