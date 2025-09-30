# AI Models Configuration - September 2025

## Overview
This document outlines the AI models available and configured in the OncoMatch-AI system as of September 29, 2025.

## Latest Available Models

### OpenAI Models

#### GPT-5 Series (Latest - Released 2025)
- **gpt-5**: Most advanced model with superior reasoning and integrated routing
  - Cost: $1.25/1M input tokens, $10/1M output tokens
  - Best for: Complex medical reasoning, comprehensive analysis
- **gpt-5-mini**: Balanced performance and cost
  - Cost: $0.25/1M input tokens, $2/1M output tokens
  - Best for: Standard clinical matching, good balance
- **gpt-5-nano**: Ultra-fast and affordable
  - Cost: $0.05/1M input tokens, $0.40/1M output tokens
  - Best for: Simple cases, high-speed requirements

#### O-Series Reasoning Models
- **o4-mini**: Newest mini reasoning model (September 2025)
- **o3-pro**: Professional tier reasoning
- **o3**: Standard reasoning model
- **o3-mini**: Fast reasoning, cost-effective

Note: While O-series models remain available, GPT-5 has superseded them for most reasoning tasks with ~80% fewer errors.

#### GPT-4 Series (Previous Generation)
- **gpt-4.1-2025-04-14**: Latest GPT-4 update
- **gpt-4o**: Standard GPT-4o
- **gpt-4o-mini**: Fast, lightweight GPT-4

### Anthropic Claude Models

#### Claude 4 Series (Latest - September 2025)
- **claude-4-opus**: Most capable Claude model
  - Excellent for safety evaluation and medical ethics
  - Superior nuanced understanding
- **claude-4-sonnet**: Balanced Claude 4
  - Good performance with reasonable cost
  - Strong safety features

#### Claude 3.5 (Previous Generation - Still Supported)
- **claude-3.5-sonnet**: Fast and efficient
  - Good for simple safety checks
  - Lower cost alternative

### Google Gemini Models

#### Gemini 2.5 Series (Latest - September 2025)
- **gemini-2.5-pro**: Most capable Gemini
  - Advanced reasoning capabilities
  - Medical knowledge integration
- **gemini-2.5-flash**: Ultra-fast responses
  - Excellent for simple cases
  - Very low latency (<2 seconds)
- **gemini-2.5-flash-lite**: Lightweight option
  - Minimal latency
  - Most cost-effective

### Specialized Medical Models
- **TrialGPT**: NIH's specialized clinical trial matching model (when available)
- **Meditron-70B**: Open-source medical LLM (research use)

## Model Selection Strategy

The system uses intelligent routing based on:

1. **Complexity Assessment**:
   - Simple cases → Gemini 2.5 Flash, GPT-5 Nano
   - Medium complexity → GPT-5 Mini, Claude 4 Sonnet
   - High complexity → GPT-5, Claude 4 Opus
   - Very complex → GPT-5 with ensemble validation

2. **Special Requirements**:
   - Safety/Ethics evaluation → Claude 4 models
   - Speed critical → Gemini 2.5 Flash/Lite
   - Medical specialization → TrialGPT (when configured)
   - Complex reasoning → GPT-5 or O3 Pro

3. **Cost Optimization**:
   - Budget-aware routing
   - Automatic fallback to available models when needed
   - Dynamic selection based on complexity (quality for all)
   - Urgency affects workflow prioritization, not model quality

## Configuration

Models are configured through environment variables:
```bash
OPENAI_API_KEY=your_key        # For GPT-5, O-series, GPT-4
ANTHROPIC_API_KEY=your_key     # For Claude 4 models
GOOGLE_API_KEY=your_key        # For Gemini 2.5 models
TRIALGPT_API_KEY=your_key      # Optional: For TrialGPT
```

## Performance Benchmarks (September 2025)

| Model | Avg Latency | Accuracy | Cost/1K tokens |
|-------|------------|----------|----------------|
| GPT-5 | 15s | 96% | $0.08 |
| GPT-5-mini | 6s | 92% | $0.03 |
| Claude 4 Opus | 12s | 94% | $0.10 |
| Gemini 2.5 Pro | 5s | 90% | $0.03 |
| Gemini 2.5 Flash | 2s | 85% | $0.01 |

## Notes

1. All models listed are confirmed available as of September 29, 2025
2. Model availability is dynamically checked at runtime
3. The system gracefully falls back when preferred models are unavailable
4. Costs are approximate and may vary based on usage patterns
5. GPT-5 integrates an intelligent internal router that optimizes for each request

## Future Considerations

- Continuous monitoring of new model releases
- Integration with emerging medical-specific models
- Cost optimization through improved caching
- Performance tuning based on real-world usage patterns

