"""
LLM Provider Adapters with runtime capability detection.
Supports OpenAI (o1, GPT-5), Anthropic (Claude 4), Google (Gemini 2.5), and specialized medical models.
"""

import os
import json
import asyncio
import logging
from typing import Protocol, Optional, Set, Dict, Any
from abc import ABC, abstractmethod
import hashlib
from datetime import datetime

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


class ProviderUnavailable(Exception):
    """Raised when provider is unavailable or model not found."""
    pass


class BaseProvider(Protocol):
    """Protocol for LLM providers."""
    
    async def available_models(self) -> Set[str]:
        """Probe and return available models (cached)."""
        ...
    
    async def generate_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        timeout_s: float = 30.0
    ) -> Dict[str, Any]:
        """Generate JSON response from model."""
        ...


class OpenAIProvider:
    """OpenAI provider supporting GPT-5 and o3/o4 reasoning models."""
    
    MODELS = {
        "o4-mini",  # Newest reasoning model (September 2025)
        "o3",  # Reasoning model
        "o3-pro",  # Professional tier
        "o3-mini",  # Fast reasoning
        "gpt-5",  # Most advanced GPT (Sept 2025)
        "gpt-5-mini",  # Smaller GPT-5
        "gpt-5-nano",  # Smallest GPT-5
        "gpt-4.1-2025-04-14",  # GPT-4.1 update
        "gpt-4o",  # GPT-4o
        "gpt-4o-mini"  # Fast fallback
    }
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self._available_models_cache: Optional[Set[str]] = None
        self._last_probe: Optional[datetime] = None
        self.base_url = "https://api.openai.com/v1"
    
    async def available_models(self) -> Set[str]:
        """Probe available models (cached for 1 hour)."""
        if self._available_models_cache and self._last_probe:
            if (datetime.now() - self._last_probe).seconds < 3600:
                return self._available_models_cache
        
        if not self.api_key:
            logger.warning("OpenAI API key not configured")
            return set()
        
        available = set()
        
        async with httpx.AsyncClient() as client:
            try:
                # Try to list models
                response = await client.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=10.0
                )
                if response.status_code == 200:
                    data = response.json()
                    model_ids = {m["id"] for m in data.get("data", [])}
                    available = self.MODELS.intersection(model_ids)
                    
                    # If list doesn't show all, probe individually
                    if len(available) < 3:
                        for model in self.MODELS:
                            if model not in available:
                                if await self._probe_model(client, model):
                                    available.add(model)
            except Exception as e:
                logger.error(f"Failed to probe OpenAI models: {e}")
                # Assume standard models available
                available = {"gpt-4o", "gpt-4o-mini"}
        
        self._available_models_cache = available
        self._last_probe = datetime.now()
        logger.info(f"OpenAI available models: {available}")
        return available
    
    async def _probe_model(self, client: httpx.AsyncClient, model: str) -> bool:
        """Probe if specific model is available."""
        try:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 1,
                    "temperature": 0
                },
                timeout=5.0
            )
            return response.status_code != 404
        except:
            return False
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,))
    )
    async def generate_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        timeout_s: float = 30.0
    ) -> Dict[str, Any]:
        """Generate JSON response from OpenAI model."""
        if not self.api_key:
            raise ProviderUnavailable("OpenAI API key not configured")
        
        available = await self.available_models()
        if model not in available:
            # Try fallback
            if "gpt-4o" in available:
                logger.warning(f"Model {model} not available, falling back to gpt-4o")
                model = "gpt-4o"
            else:
                raise ProviderUnavailable(f"Model {model} not available")
        
        async with httpx.AsyncClient() as client:
            # Special handling for o1-series (no system prompt)
            if model.startswith("o1"):
                messages = [
                    {"role": "user", "content": f"{system}\n\n{user}"}
                ]
            else:
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ]
            
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature if not model.startswith("o1") else 1,  # o1 requires temp=1
                    "max_tokens": max_tokens,
                    "response_format": {"type": "json_object"} if not model.startswith("o1") else None
                },
                timeout=timeout_s
            )
            
            if response.status_code != 200:
                raise ProviderUnavailable(f"OpenAI API error: {response.status_code}")
            
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            
            # Parse JSON
            return self._parse_json_response(content)
    
    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse and clean JSON response."""
        # Strip markdown fences
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        
        # Fix common issues
        content = content.strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            # Try to fix trailing commas
            import re
            content = re.sub(r',\s*}', '}', content)
            content = re.sub(r',\s*]', ']', content)
            
            try:
                return json.loads(content)
            except:
                logger.error(f"Failed to parse JSON: {content[:200]}...")
                raise ProviderUnavailable(f"Invalid JSON response: {e}")


class AnthropicProvider:
    """Anthropic provider supporting Claude 4 and 3.5."""
    
    MODELS = {
        # Claude 3.7 series (February 2025 - PRIMARY, currently best available)
        "claude-3-7-sonnet-20250219",  # Claude 3.7 Sonnet - best available
        # Claude 3.5 series (stable, reliable)
        "claude-3-5-sonnet-20241022",  # Claude 3.5 Sonnet
        "claude-3.5-sonnet",           # Alias
        # Claude 3 series (fallback)
        "claude-3-opus-20240229",      # Claude 3 Opus
        "claude-3-sonnet-20240229",    # Claude 3 Sonnet  
        "claude-3-haiku-20240307",     # Claude 3 Haiku
        # Claude 4.1 series (Future - may require waitlist/beta access)
        "claude-opus-4.1",             # Claude 4.1 Opus - for future use
        "claude-sonnet-4",             # Claude 4 Sonnet - for future use
    }
    
    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self._available_models_cache: Optional[Set[str]] = None
        self._last_probe: Optional[datetime] = None
        self._probe_lock = asyncio.Lock()  # Prevent concurrent probes
        self.base_url = "https://api.anthropic.com/v1"
    
    async def available_models(self) -> Set[str]:
        """Probe available models with concurrency protection."""
        if not self.api_key:
            logger.warning("Anthropic API key not configured")
            return set()
        
        # Check cache first (outside lock for speed)
        if (self._available_models_cache is not None and 
            self._last_probe and 
            (datetime.now() - self._last_probe).seconds < 300):
            return self._available_models_cache
        
        # Use lock to ensure only one probe runs at a time
        async with self._probe_lock:
            # Double-check cache after acquiring lock (another thread may have probed)
            if (self._available_models_cache is not None and 
                self._last_probe and 
                (datetime.now() - self._last_probe).seconds < 300):
                return self._available_models_cache
            
            logger.info("Probing Anthropic models...")
            available = set()
            
            async with httpx.AsyncClient() as client:
                # Anthropic doesn't have a list endpoint, probe each model
                for model in self.MODELS:
                    if await self._probe_model(client, model):
                        available.add(model)
            
            # Cache results
            self._available_models_cache = available
            self._last_probe = datetime.now()
            logger.info(f"Anthropic available models: {available}")
            return available
    
    async def _probe_model(self, client: httpx.AsyncClient, model: str) -> bool:
        """Probe if specific model is available."""
        try:
            response = await client.post(
                f"{self.base_url}/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",  # Stable API version
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 1
                },
                timeout=5.0
            )
            # Only consider 200 as truly available and working.
            # 404 means model doesn't exist, other errors mean issues.
            is_available = response.status_code == 200
            if not is_available:
                logger.debug(f"Model {model} probe returned {response.status_code}, marking as unavailable")
            return is_available
        except:
            return False
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,))
    )
    async def generate_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        timeout_s: float = 30.0
    ) -> Dict[str, Any]:
        """Generate JSON response from Anthropic model."""
        if not self.api_key:
            raise ProviderUnavailable("Anthropic API key not configured")
        
        available = await self.available_models()
        if model not in available:
            # Map to available fallback
            logger.warning(f"Model {model} not available in: {available}")
            if "claude-4" in model or "claude-opus-4" in model or "claude-sonnet-4" in model:
                # Try Claude 3.7, then 3.5, then 3 Opus as fallback
                if "claude-3-7-sonnet-20250219" in available:
                    logger.warning(f"Model {model} not available, falling back to Claude 3.7 Sonnet")
                    model = "claude-3-7-sonnet-20250219"
                elif "claude-3-5-sonnet-20241022" in available:
                    logger.warning(f"Model {model} not available, falling back to Claude 3.5 Sonnet")
                    model = "claude-3-5-sonnet-20241022"
                elif "claude-3-opus-20240229" in available:
                    logger.warning(f"Model {model} not available, falling back to Claude 3 Opus")
                    model = "claude-3-opus-20240229"
                else:
                    raise ProviderUnavailable(f"Model {model} not available and no fallback found")
            else:
                raise ProviderUnavailable(f"Model {model} not available")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",  # Stable API version
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                timeout=timeout_s
            )
            
            if response.status_code != 200:
                try:
                    error_data = response.json()
                    error_msg = error_data.get('error', {}).get('message', response.text[:200])
                except:
                    error_msg = response.text[:200] if hasattr(response, 'text') else str(response.content)[:200]
                logger.error(f"Anthropic API error {response.status_code} for model {model}: {error_msg}")
                raise ProviderUnavailable(f"Anthropic API error: {response.status_code} - {error_msg}")
            
            data = response.json()
            content = data["content"][0]["text"]
            
            return self._parse_json_response(content)
    
    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse and clean JSON response."""
        # Strip markdown fences
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        
        content = content.strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            # Try to fix trailing commas
            import re
            content = re.sub(r',\s*}', '}', content)
            content = re.sub(r',\s*]', ']', content)
            
            try:
                return json.loads(content)
            except:
                logger.error(f"Failed to parse JSON: {content[:200]}...")
                raise ProviderUnavailable(f"Invalid JSON response: {e}")


class GeminiProvider:
    """Google Gemini provider supporting 2.5 series (Latest - September 2025)."""
    
    MODELS = {
        "gemini-2.5-pro",          # Latest Gemini Pro (September 2025)
        "gemini-2.5-flash",        # Ultra-fast Gemini (September 2025)
        "gemini-2.5-flash-lite",   # Lightweight version (September 2025)
        "gemini-1.5-pro",          # Previous generation (legacy fallback only)
        "gemini-1.5-flash"         # Previous generation (legacy fallback only)
    }
    
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self._available_models_cache: Optional[Set[str]] = None
        self._last_probe: Optional[datetime] = None
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
    
    async def available_models(self) -> Set[str]:
        """Probe available models."""
        if self._available_models_cache and self._last_probe:
            if (datetime.now() - self._last_probe).seconds < 3600:
                return self._available_models_cache
        
        if not self.api_key:
            logger.warning("Google API key not configured")
            return set()
        
        available = set()
        
        async with httpx.AsyncClient() as client:
            try:
                # List available models
                response = await client.get(
                    f"{self.base_url}/models",
                    params={"key": self.api_key},
                    timeout=10.0
                )
                if response.status_code == 200:
                    data = response.json()
                    for model_data in data.get("models", []):
                        name = model_data.get("name", "").split("/")[-1]
                        if any(name.startswith(m) for m in ["gemini-2.5", "gemini-1.5"]):
                            # Map to internal model names
                            if "2.5-pro" in name:
                                available.add("gemini-2.5-pro")
                            elif "2.5-flash" in name:
                                available.add("gemini-2.5-flash")
                            elif "1.5-pro" in name:
                                available.add("gemini-1.5-pro")
                            elif "1.5-flash" in name:
                                available.add("gemini-1.5-flash")
            except Exception as e:
                logger.error(f"Failed to probe Gemini models: {e}")
                # Assume 1.5 models available
                available = {"gemini-1.5-pro", "gemini-1.5-flash"}
        
        self._available_models_cache = available
        self._last_probe = datetime.now()
        logger.info(f"Gemini available models: {available}")
        return available
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError))
    )
    async def generate_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        timeout_s: float = 30.0
    ) -> Dict[str, Any]:
        """Generate JSON response from Gemini model."""
        if not self.api_key:
            raise ProviderUnavailable("Google API key not configured")
        
        available = await self.available_models()
        if model not in available:
            # Try fallback
            if model.startswith("gemini-2.5") and "gemini-1.5-flash" in available:
                logger.warning(f"Model {model} not available, falling back to gemini-1.5-flash")
                model = "gemini-1.5-flash"
            else:
                raise ProviderUnavailable(f"Model {model} not available")
        
        async with httpx.AsyncClient() as client:
            # Combine system and user prompts
            full_prompt = f"{system}\n\n{user}\n\nRespond with valid JSON only."
            
            response = await client.post(
                f"{self.base_url}/models/{model}:generateContent",
                params={"key": self.api_key},
                json={
                    "contents": [{
                        "parts": [{
                            "text": full_prompt
                        }]
                    }],
                    "generationConfig": {
                        "temperature": temperature,
                        "maxOutputTokens": max_tokens,
                        "responseMimeType": "application/json"
                    }
                },
                timeout=timeout_s
            )
            
            if response.status_code != 200:
                raise ProviderUnavailable(f"Gemini API error: {response.status_code}")
            
            data = response.json()
            
            # Robust extraction with error handling
            try:
                candidates = data.get("candidates", [])
                if not candidates:
                    logger.error(f"Gemini response missing candidates: {data}")
                    raise ProviderUnavailable("Gemini response missing candidates")
                
                content_obj = candidates[0].get("content", {})
                parts = content_obj.get("parts", [])
                
                if not parts:
                    logger.error(f"Gemini response missing parts: {data}")
                    raise ProviderUnavailable("Gemini response missing parts")
                
                content = parts[0].get("text", "")
                
                if not content:
                    logger.error(f"Gemini response missing text content: {data}")
                    raise ProviderUnavailable("Gemini response missing text")
                
            except (KeyError, IndexError, TypeError) as e:
                logger.error(f"Error extracting Gemini response: {e}, data: {data}")
                raise ProviderUnavailable(f"Invalid Gemini response structure: {e}")
            
            return self._parse_json_response(content)
    
    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse and clean JSON response."""
        # Strip markdown fences
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        
        content = content.strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            # Try to fix trailing commas
            import re
            content = re.sub(r',\s*}', '}', content)
            content = re.sub(r',\s*]', ']', content)
            
            try:
                return json.loads(content)
            except:
                logger.error(f"Failed to parse JSON: {content[:200]}...")
                raise ProviderUnavailable(f"Invalid JSON response: {e}")


class MedicalProvider:
    """Specialized medical model provider (TrialGPT, Meditron)."""
    
    MODELS = {
        "trialgpt",
        "trialgpt-pro",
        "meditron-70b"
    }
    
    def __init__(self):
        self.trialgpt_url = os.getenv("TRIALGPT_BASE_URL")
        self.trialgpt_token = os.getenv("TRIALGPT_API_TOKEN")
        self.meditron_url = os.getenv("MEDITRON_ENDPOINT")
        self.meditron_token = os.getenv("MEDITRON_API_TOKEN")
        self._available_models_cache: Optional[Set[str]] = None
        self._last_probe: Optional[datetime] = None
    
    async def available_models(self) -> Set[str]:
        """Probe available medical models."""
        if self._available_models_cache and self._last_probe:
            if (datetime.now() - self._last_probe).seconds < 3600:
                return self._available_models_cache
        
        available = set()
        
        # Check TrialGPT
        if self.trialgpt_url:
            async with httpx.AsyncClient() as client:
                try:
                    headers = {}
                    if self.trialgpt_token:
                        headers["Authorization"] = f"Bearer {self.trialgpt_token}"
                    
                    response = await client.get(
                        f"{self.trialgpt_url}/health",
                        headers=headers,
                        timeout=5.0
                    )
                    if response.status_code == 200:
                        available.update({"trialgpt", "trialgpt-pro"})
                except:
                    pass
        
        # Check Meditron
        if self.meditron_url:
            async with httpx.AsyncClient() as client:
                try:
                    headers = {}
                    if self.meditron_token:
                        headers["Authorization"] = f"Bearer {self.meditron_token}"
                    
                    response = await client.get(
                        f"{self.meditron_url}/health",
                        headers=headers,
                        timeout=5.0
                    )
                    if response.status_code == 200:
                        available.add("meditron-70b")
                except:
                    pass
        
        self._available_models_cache = available
        self._last_probe = datetime.now()
        logger.info(f"Medical models available: {available}")
        return available
    
    async def generate_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        timeout_s: float = 30.0
    ) -> Dict[str, Any]:
        """Generate JSON response from medical model."""
        available = await self.available_models()
        if model not in available:
            raise ProviderUnavailable(f"Medical model {model} not available")
        
        if model.startswith("trialgpt"):
            return await self._call_trialgpt(model, system, user, temperature, max_tokens, timeout_s)
        elif model == "meditron-70b":
            return await self._call_meditron(system, user, temperature, max_tokens, timeout_s)
        else:
            raise ProviderUnavailable(f"Unknown medical model: {model}")
    
    async def _call_trialgpt(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        timeout_s: float
    ) -> Dict[str, Any]:
        """Call TrialGPT API."""
        if not self.trialgpt_url:
            raise ProviderUnavailable("TrialGPT not configured")
        
        async with httpx.AsyncClient() as client:
            headers = {"Content-Type": "application/json"}
            if self.trialgpt_token:
                headers["Authorization"] = f"Bearer {self.trialgpt_token}"
            
            response = await client.post(
                f"{self.trialgpt_url}/generate",
                headers=headers,
                json={
                    "model": model,
                    "system": system,
                    "prompt": user,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "format": "json"
                },
                timeout=timeout_s
            )
            
            if response.status_code != 200:
                raise ProviderUnavailable(f"TrialGPT API error: {response.status_code}")
            
            return response.json()
    
    async def _call_meditron(
        self,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        timeout_s: float
    ) -> Dict[str, Any]:
        """Call Meditron API."""
        if not self.meditron_url:
            raise ProviderUnavailable("Meditron not configured")
        
        async with httpx.AsyncClient() as client:
            headers = {"Content-Type": "application/json"}
            if self.meditron_token:
                headers["Authorization"] = f"Bearer {self.meditron_token}"
            
            response = await client.post(
                f"{self.meditron_url}/generate",
                headers=headers,
                json={
                    "system": system,
                    "prompt": user,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "format": "json"
                },
                timeout=timeout_s
            )
            
            if response.status_code != 200:
                raise ProviderUnavailable(f"Meditron API error: {response.status_code}")
            
            return response.json()


# Provider registry
PROVIDERS = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "medical": MedicalProvider
}


async def get_all_available_models() -> Dict[str, Set[str]]:
    """Get all available models from all providers."""
    results = {}
    
    for name, provider_class in PROVIDERS.items():
        try:
            provider = provider_class()
            models = await provider.available_models()
            if models:
                results[name] = models
        except Exception as e:
            logger.warning(f"Failed to probe {name} provider: {e}")
    
    return results
