# Performance Optimization Guide

*Last Updated: September 2025*

## 📊 Current Performance Benchmarks

*Based on actual production runs (September 2025)*

| Configuration | Trials | Actual Time | Model Used | Est. Cost |
|--------------|--------|-------------|------------|-----------|
| **Quick Test** | 5 trials | ~15 seconds | Gemini 2.5 Flash-Lite | ~$0.02 |
| **Limited** | 10 trials | **26.3 seconds** | Gemini 2.5 Flash-Lite | ~$0.05 |
| **Default (Single)** | 40 trials | **99 seconds** | Gemini 2.5 Flash-Lite | ~$0.20 |
| **Evaluation (Batch)** | 40 trials x 5 patients | **10.6 minutes** | Gemini 2.5 Flash-Lite | ~$1.00 |

## ✅ Implemented Optimizations

### 1. **Intelligent Model Routing** (September 2025 - Updated)
- Complexity-based model selection (NOT urgency-based)
- Quality models for all patients regardless of clinical urgency
- Premium models (GPT-5, Claude 4 Opus) for complex cases
- Efficient models (GPT-5-nano, Claude 3.5 Haiku) for simple cases - still good quality
- Automatic fallback chains for reliability
- Workflow prioritization handles urgency separately from model selection

### 2. **Parallel Processing**
- Concurrent LLM calls (5 max) with semaphore control
- Async/await pattern throughout the pipeline
- Batch processing for multiple trials
- Non-blocking I/O for all external calls

### 3. **Caching Strategy**
- TTL-based response caching (24-hour default)
- In-memory cache for session persistence
- Disk-based cache for long-term storage
- Cache key: `hash(patient_id + trial_id + model + prompt_version)`

### 4. **Smart Defaults**
- 10 trials default (optimal speed/coverage balance)
- Dynamic timeout adjustment based on complexity
- Progressive complexity escalation
- Budget-aware processing ($0.05 per trial default)

## 🔍 Performance Analysis

### Primary Bottlenecks

#### 1. **LLM API Latency** (60-70% of total time)
| Model Tier | Latency | Use Case |
|------------|---------|----------|
| Premium (GPT-5, Claude 4 Opus) | 7-10s | Complex medical reasoning |
| Standard (GPT-4o, Gemini 2.5 Pro) | 5-7s | General matching |
| Fast (Gemini 2.5 Flash-Lite, GPT-4.1 nano) | 2-3s | Simple screening |
| Network overhead | 0.5-1s | All requests |

#### 2. **Data Fetching** (15-20% of total time)
- BioMCP/ClinicalTrials.gov API: 2-3 seconds
- Rate limiting: 45 req/min for BioMCP
- Parsing and validation: <1 second
- Retry logic adds 1-2s on failures

#### 3. **Post-Processing** (10-15% of total time)
- Safety validation passes: 1-2 seconds
- Score normalization: <0.5 seconds
- Result formatting: <0.5 seconds
- Logging and metrics: <0.2 seconds

## 🚀 Optimization Strategies

### Currently Active Optimizations

```python
# Complexity-based routing (automatic)
Simple cases → Gemini 2.5 Flash-Lite (2-3s, $0.005/1k tokens)
Medium cases → GPT-4o-mini (4-5s, $0.015/1k tokens)  
Complex cases → GPT-5/Claude 4 Opus (7-10s, $0.10/1k tokens)
Safety checks → Claude 4 Sonnet (5-6s, $0.05/1k tokens)
```

### Performance Tuning Parameters

```bash
# Environment variables for optimization
export LLM_BUDGET_MS=15000           # Max latency per trial
export LLM_BUDGET_USD=0.05           # Max cost per trial
export CACHE_TTL_HOURS=24            # Cache duration
export CACHE_DIR=./cache             # Cache location
export MAX_CONCURRENT_TRIALS=10      # Parallel processing
export MAX_RETRIES=3                 # Retry attempts
```

### Usage Modes

```bash
# Fast screening (optimized for speed)
python src/match.py --patient_id P002 --mode fast --max_trials 5

# Balanced approach (default)
python src/match.py --patient_id P002 --mode balanced

# High accuracy (comprehensive analysis)
python src/match.py --patient_id P002 --mode accurate --max_trials 20

# With caching enabled (for repeat queries)
export ENABLE_CACHE=true
python src/match.py --patient_id P002
```

## 📈 Performance Roadmap

### Q4 2025: Rule-Based Pre-filtering
- Implement basic eligibility filters before LLM calls
- Stage, age, and biomarker quick checks
- Expected impact: 30-40% reduction in LLM calls
- Target: <10s for 10 trials

### Q1 2026: Edge Deployment
- Deploy lightweight models locally
- Fine-tuned DistilBERT for initial screening
- Embedding-based similarity search
- Target: <1s for pre-screening

### Q2 2026: Specialized Medical Models
- Cancer-specific fine-tuned models
- Cached trial embeddings database
- Real-time updates via WebSocket
- Target: <5s end-to-end

## 🎯 Performance Targets & Current Status

### September 2025 Actual Results

| Metric | Target | Achieved (10 trials) | Achieved (40 trials) | Status |
|--------|--------|---------------------|---------------------|--------|
| Latency (avg) | <30s (10), <120s (40) | 26.3s | 127s | ⚠️ Mixed |
| Latency (p95) | <30s (10), <180s (40) | 31.2s | 207s | ⚠️ Close |
| nDCG@10 | >0.75 | 0.83 | 0.95 | ✅ Exceeded |
| Precision@10 | >0.70 | 0.70 | 1.00 | ✅ Met |
| MRR | >0.80 | 1.00 | 1.00 | ✅ Exceeded |
| Recall | >0.30 | 0.31 | 0.068 | ❌ Low with 40 |
| System Grade | A | B (0.77) | C+ (0.70) | ⚠️ Trade-off |
| Cost per patient | <$0.50 | ~$0.05 | ~$0.20 | ✅ Good |

### Key Insights
1. **Trade-off exists** between coverage and speed:
   - 10 trials: Fast (26s), good grade (B), but lower recall
   - 40 trials: Comprehensive, but 4-5x slower (99-127s), grade drops to C+
2. **Recall paradox**: Processing more trials (40 vs 10) actually decreased recall (0.068 vs 0.31)
   - This suggests ranking/scoring issues with larger sets
3. **Sweet spot**: Consider 15-20 trials for balance
4. **Gemini 2.5 Flash-Lite** successfully handles all complexity levels at low cost

## ⚠️ Performance Trade-offs

### Speed vs Accuracy
- Fast mode: 15% lower nDCG but 3x faster
- Balanced mode: Optimal trade-off for most cases
- Accurate mode: Best nDCG but 2x slower

### Cost vs Performance
| Strategy | Cost/Patient | Speed | Quality |
|----------|-------------|-------|---------|
| All premium models | $0.50-1.00 | Slow | Excellent |
| Smart routing | $0.10-0.20 | Fast | Very Good |
| All fast models | $0.02-0.05 | Very Fast | Good |

### Coverage vs Speed
- 5 trials: Fast (15s) but may miss opportunities
- 10 trials: Balanced (30s) with good coverage
- 20 trials: Comprehensive (60s) but slower

## 💡 Best Practices

### For Developers
1. **Profile first** - Use `--debug` flag to see timing breakdown
2. **Cache aggressively** - Enable caching for all non-real-time queries
3. **Batch when possible** - Process multiple patients together
4. **Monitor costs** - Set LLM_BUDGET_USD appropriately
5. **Use fast mode for testing** - Iterate quickly during development

### For Production
1. **Warm cache** - Pre-cache common patient profiles
2. **Rate limit clients** - Prevent API exhaustion
3. **Set timeouts** - Cap maximum processing time
4. **Monitor metrics** - Track p50/p95/p99 latencies
5. **A/B test modes** - Find optimal settings for your use case

## 📊 Monitoring & Profiling

### Key Metrics to Track
```python
# Performance metrics
- api_latency_ms: Time per LLM call
- cache_hit_rate: Percentage of cached responses
- parallel_efficiency: Speedup from parallelization
- total_processing_time: End-to-end latency

# Cost metrics
- tokens_per_request: Average token usage
- cost_per_patient: Total API costs
- model_distribution: Which models are used most

# Quality metrics
- ndcg_score: Ranking quality
- safety_violations: Critical misses
- user_satisfaction: Clinical relevance
```

### Debug Output
```bash
# Enable detailed timing information
export LOG_LEVEL=DEBUG
python src/match.py --patient_id P002 --debug

# Output includes:
# - Model selection reasoning
# - API call timing
# - Cache hit/miss details
# - Cost breakdown
```

## 🔧 Troubleshooting

### Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Slow response times | Premium models for all cases | Enable smart routing |
| High costs | No caching | Enable cache with 24h TTL |
| Timeouts | Serial processing | Increase MAX_CONCURRENT_TRIALS |
| Poor accuracy in fast mode | Over-optimization | Use balanced mode |
| Rate limiting | Too many parallel calls | Reduce concurrency |

---

*For implementation details, see [ARCHITECTURE.md](ARCHITECTURE.md)*
*For metrics definitions, see [METRICS_GUIDE.md](METRICS_GUIDE.md)*