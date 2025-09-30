# OncoMatch AI - System Architecture

Technical overview of the clinical trial matching system.

---

## System Overview

OncoMatch AI matches cancer patients to relevant clinical trials using:
- Multi-provider LLM ranking (OpenAI, Anthropic, Google)
- Real-time trial data from ClinicalTrials.gov via BioMCP
- Intelligent caching and batch processing
- Comprehensive evaluation with synthetic patients and LLM judges

### Performance
- **Latency**: 10.5s average per patient
- **Throughput**: ~50 concurrent LLM calls
- **Cache Hit Rate**: 50-80% typical
- **Grade**: B+ (0.82/1.00)

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────┐
│                    Patient Data Input                     │
│                     (patients.csv)                        │
└─────────────────────┬────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────┐
│               Data Processor & Validator                  │
│     • Parse demographics, biomarkers, history            │
│     • Validate ECOG, stage, comorbidities                │
└─────────────────────┬────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────┐
│             Trial Fetcher (BioMCP Wrapper)               │
│     • Query ClinicalTrials.gov                           │
│     • Filter by cancer type, biomarkers, status          │
│     • Cache trials (TTL: 24h)                            │
└─────────────────────┬────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────┐
│             Optimized LLM Ranker                         │
│                                                           │
│  ┌─────────────────────────────────────────────────┐    │
│  │  1. Batch Trials (3-5 per call)                 │    │
│  │  2. Check Cache (50-80% hit rate)               │    │
│  │  3. Parallel LLM Calls (20 max concurrent)      │    │
│  │  4. Score Normalization (if avg < 0.4)          │    │
│  └─────────────────────────────────────────────────┘    │
│                                                           │
│  Provider Routing: Gemini → OpenAI → Anthropic          │
└─────────────────────┬────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────┐
│                 Ranked Match Results                      │
│     • NCT IDs with scores (0-1)                          │
│     • Eligibility explanations                           │
│     • Biomarker alignments                               │
│     • Safety concerns                                    │
└──────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Data Models (`models.py`)

**Patient**
- Demographics: age, gender, race, location
- Clinical: cancer_type, cancer_stage, ecog_status
- Treatment: previous_treatments, lines_of_therapy
- Biomarkers: biomarkers_detected (e.g., "ER+", "HER2-")

**ClinicalTrial**
- Identifiers: nct_id, title, phase, status
- Criteria: inclusion_criteria, exclusion_criteria
- Details: brief_summary, interventions, locations

**MatchResult**
- Scores: overall_score, eligibility_score, biomarker_score
- Metadata: confidence, match_reasons, safety_concerns
- Explanation: summary, reasoning

### 2. Trial Fetching

**BioMCP Wrapper** (`biomcp_wrapper.py`)
- Interfaces with BioMCP SDK for ClinicalTrials.gov
- Rate limiting: 45 requests/min
- Caching: 24-hour TTL (SQLite)
- Fallback to direct API if BioMCP unavailable

**Trial Analyzer** (`trial_analyzer.py`)
- Extracts structured eligibility criteria
- Identifies required/exclusionary biomarkers
- Assesses trial complexity
- Calculates geographic accessibility

### 3. LLM Ranking System

**Optimized Ranker** (`optimized_ranker.py`)

**Key Features**:
- **Batching**: Groups 3-5 trials per LLM call
- **Parallelism**: Up to 20 concurrent API calls
- **Caching**: Multi-level (memory + disk)
- **Normalization**: Auto-adjusts conservative scoring

**Workflow**:
```python
1. Batch trials into groups (size: 3-5)
2. Check cache for existing results
3. Generate prompts with patient + trial details
4. Call LLM API in parallel (with semaphore)
5. Parse JSON responses with retry/fallback
6. Normalize scores if average < 0.4
7. Sort by overall_score and return top matches
```

**Score Normalization**:
```python
if avg_score < 0.4:
    boost_factor = min(1.5, 0.5 / avg_score)
    # Apply boost to all scores (capped at 1.0)
```

### 4. LLM Providers

**Provider Adapters** (`llm_providers.py`)

Three provider implementations with unified interface:

**OpenAIProvider**
- Models: gpt-4o, gpt-5, gpt-5-nano
- Features: JSON mode, function calling
- Error handling: Retries with exponential backoff

**AnthropicProvider**
- Models: claude-3-7-sonnet, claude-3-5-sonnet, claude-3-haiku
- Features: Structured JSON, safety-focused
- Probe caching: Prevents redundant availability checks

**GeminiProvider** (Primary)
- Models: gemini-2.5-flash, gemini-2.5-pro
- Features: Fast, cost-effective (~$0.005/1k tokens)
- Token handling: Increased limits for reasoning tokens

**Fallback Chain**: Gemini → OpenAI → Anthropic

### 5. Model Registry & Routing

**Model Registry** (`llm_registry.py`)
- Tracks model capabilities, costs, and latencies
- Dynamically probes provider availability
- Maintains quality/cost/speed profiles

**Smart Router** (`model_router.py`)
- Complexity-based model selection
- Budget-aware routing ($0.05/trial default)
- Automatic fallback on rate limits

**Routing Logic**:
```python
1. Calculate case complexity (biomarkers, stage, prior treatments)
2. Score available models: quality × 0.5 + cost × 0.3 + speed × 0.2
3. Select highest-scoring model within budget
4. Fallback to next provider if unavailable
```

### 6. Evaluation Suite

**Evaluator** (`evaluation/evaluator.py`)
- Orchestrates comprehensive evaluation
- Manages synthetic patient generation
- Coordinates judge ensemble
- Calculates clinical, safety, and equity metrics

**Synthetic Patient Generator** (`evaluation/synthetic_patients.py`)

Generates 1000 diverse patients:
- **Standard (600)**: Representative demographics
- **Edge Cases (250)**: Rare mutations, unusual stages
- **Adversarial (100)**: Missing data, conflicts
- **Equity Stress (50)**: Underserved populations

**Judge Ensemble** (`evaluation/judge_ensemble.py`)

7 specialized LLM judges:
1. **Accuracy** (gpt-4o): Medical correctness
2. **Safety** (claude-3-7-sonnet): Risk assessment
3. **Completeness** (gemini-2.5-pro): Coverage
4. **Bias** (gpt-4o): Fairness
5. **Robustness** (gemini-2.5-pro): Edge cases
6. **Clinical Text** (claude-3.5-sonnet): Writing quality
7. **TrialGPT** (gpt-4o-mini): Specialized matching

**Metrics** (`evaluation/metrics_core.py`)
- Clinical: nDCG, Precision, Recall, MRR, F1
- Safety: Violation rate, critical miss detection
- Equity: Subgroup performance, biomarker diversity
- Performance: Latency, cache hit rate, cost

---

## Data Flow

### Matching Flow

```
1. Load Patient → Validate & Extract Biomarkers
2. Fetch Trials → Query BioMCP, Apply Filters
3. Analyze Trials → Extract Criteria, Calculate Complexity
4. Batch Trials → Group into batches of 3-5
5. LLM Ranking → Parallel calls with caching
6. Score Normalization → Adjust if avg < 0.4
7. Return Results → Top 10 trials with explanations
```

### Evaluation Flow

```
1. Generate Synthetic Cohort → 1000 patients
2. Match All Patients → Using cached ranker
3. Judge Ensemble → 7 judges evaluate top matches
4. Calculate Metrics → Clinical, safety, equity, performance
5. Grade System → Weighted scoring: B+ (0.80)
```

---

## Performance Optimizations

### 1. Shared Ranker Instance
- Single ranker reused across all patients
- Maintains cache between evaluations
- 50-80% cache hit rate

### 2. Intelligent Batching
- Gemini: 3 trials/batch (reasoning tokens)
- OpenAI/Anthropic: 5 trials/batch
- Dynamic adjustment based on model

### 3. Parallel Processing
- Semaphore limits concurrent calls (20 max)
- asyncio.gather for parallel execution
- Graceful handling of rate limits

### 4. Multi-Level Caching
- Memory cache: Instant lookup (<1ms)
- Disk cache: SQLite with 24h TTL
- Cache key: (patient_id, nct_id, model, prompt_version)

### 5. Score Normalization
- Monitors average scores per batch
- Auto-adjusts if LLM too conservative
- Transparent logging for auditability

---

## Medical Safety Features

### Eligibility Checking
- ECOG performance status (0-4 scale)
- Organ function requirements (liver, kidney, cardiac)
- Brain metastases and CNS involvement
- Prior therapy lines and washout periods

### Biomarker Matching
- Oncogene mutations: EGFR, ALK, ROS1, KRAS
- Hormone receptors: ER, PR, HER2
- Immune markers: PD-L1, MSI-H, TMB
- DNA repair: BRCA1/2, HRD score

### Safety Validation
- Drug interaction screening
- Contraindication detection
- Exclusion criteria verification
- Conservative scoring for uncertain cases

---

## Technology Stack

### Core
- **Python**: 3.10+
- **Async**: asyncio, aiohttp
- **Data**: Pydantic, pandas

### LLM APIs
- **OpenAI**: openai>=1.0.0
- **Anthropic**: anthropic>=0.18.0
- **Google**: google-generativeai>=0.3.0

### Data Sources
- **BioMCP**: biomcp SDK
- **ClinicalTrials.gov**: Direct API fallback

### Evaluation
- **Metrics**: numpy, scipy
- **Progress**: tqdm, rich
- **Storage**: SQLite (cache), JSON (results)

---

## Configuration

### Environment Variables

```bash
# Required (at least one)
GOOGLE_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...

# Performance
MAX_CONCURRENT_LLM=20
LLM_BATCH_SIZE=5
CACHE_TTL_HOURS=24

# Budget
LLM_BUDGET_MS=15000
LLM_BUDGET_USD=0.05
```

### Model Configuration

Located in `llm_registry.py`:
- Quality weights: Model scoring priorities
- Cost limits: Budget constraints
- Latency targets: Performance goals

### Prompt Templates

Located in `optimized_ranker.py`:
- Scoring guidelines (0.0-1.0 scale)
- Expected score distribution
- Medical terminology standards

---

## Extension Points

### Adding New LLM Providers

1. Implement provider class in `llm_providers.py`
2. Add to registry in `llm_registry.py`
3. Update routing logic in `model_router.py`

### Custom Evaluation Metrics

1. Add metric function to `metrics_core.py`
2. Update evaluator in `evaluator.py`
3. Modify grading weights in `test_matching.py`

### Additional Data Sources

1. Create fetcher class (e.g., `european_trials_fetcher.py`)
2. Update `biomcp_wrapper.py` to include new source
3. Add fallback logic for multi-source queries

---

## Monitoring & Observability

### Structured Logging

```python
logger.info(
    "LLM call completed",
    extra={
        "patient_id": patient.patient_id,
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "latency_ms": 1234,
        "cache_hit": False,
        "trials_processed": 5,
        "avg_score": 0.67
    }
)
```

### Key Metrics to Track

**Performance**:
- Average latency per patient
- P95 latency
- Cache hit rate
- API call count

**Quality**:
- Average match scores
- Score distribution
- Normalization frequency
- Judge ensemble agreement

**Cost**:
- API calls per provider
- Estimated cost per patient
- Budget utilization

**Reliability**:
- Success rate
- Failure types
- Fallback frequency
- Retry attempts

---

## Security & Compliance

### HIPAA Considerations
- No PHI in logs (patient_id only, no names/SSN)
- Encrypted API communications (HTTPS)
- Local caching with appropriate permissions
- Data minimization (only fields needed)

### API Key Management
- Environment variables (not hardcoded)
- .env file excluded from git (.gitignore)
- Separate keys for dev/staging/prod

### Audit Trail
- All LLM calls logged with timestamps
- Prompt and response hashing for verification
- Score normalization events recorded
- Judge decisions tracked

---

## Deployment Considerations

### Production Readiness Checklist

- [x] Error handling and retries
- [x] Rate limit management
- [x] Multi-provider fallback
- [x] Comprehensive test suite
- [x] Structured logging
- [x] Performance monitoring hooks
- [x] Cost tracking
- [x] Medical safety checks

### Scaling Recommendations

**Horizontal Scaling**:
- Stateless design allows multiple instances
- Shared cache (e.g., Redis) for distributed systems
- Load balancer for request distribution

**Vertical Scaling**:
- Increase MAX_CONCURRENT_LLM for higher throughput
- Larger batch sizes for efficiency
- More aggressive caching

**Cost Optimization**:
- Use Gemini as primary (most cost-effective)
- Increase cache TTL for stable trials
- Batch multiple patients per session

---

## Known Limitations

### Current Constraints

**Latency**: 10.5s average (target: <15s)
**Recall**: Low at k=10 (9.6%) - returns only top 10 of ~26 relevant trials
**Rate Limits**: 
  - BioMCP: 45 req/min for trial fetching
  - OpenAI: 500 RPM can bottleneck during peaks
  - Anthropic: 50 RPM causes frequent fallbacks
  - Gemini: 60 RPM with token limit issues
**Trial Source**: ClinicalTrials.gov only (no proprietary databases)
**Geographic**: No distance calculations or travel burden assessment
**Languages**: English only

### Future Enhancements

- Geographic distance calculations
- Insurance pre-screening
- Real-time trial updates
- International trial databases
- Multi-language support

---

**Version**: 1.0.0  
**Last Updated**: September 2025