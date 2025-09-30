# OncoMatch AI - Clinical Trial Matching System

An AI-powered system for matching oncology patients to clinical trials using multiple LLM providers and real clinical trial data.

---

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add at least one API key to .env
```

### API Keys

Add to `.env` (at least one required):

```bash
GOOGLE_API_KEY=your_google_key        # Recommended (Gemini)
OPENAI_API_KEY=your_openai_key        # Optional
ANTHROPIC_API_KEY=your_anthropic_key  # Optional
```

### Run It

```bash
# Match a single patient
python src/match.py --patient_id P002

# Run evaluation
python tests/test_matching.py
```

---

## 📋 Usage

### Basic Matching

```bash
# Single patient
python src/match.py --patient_id P002

# All patients
python src/match.py --all

# Limit trials for faster results
python src/match.py --patient_id P002 --max_trials 10
```

### Output Example

```
============================================================
Clinical Trial Matches for Patient P002
============================================================
Patient: 33yo Female, Breast Stage II, ER+/PR+
Trials Analyzed: 40 in 18.3s

Top Matches:
1. NCT04301375 (Score: 0.85, Confidence: 0.90)
   Title: Hormone Therapy in ER+ Breast Cancer
   Phase: III | Status: RECRUITING
   
2. NCT04889469 (Score: 0.72, Confidence: 0.85)
   Title: Novel CDK4/6 Inhibitor Study
   Phase: II | Status: ACTIVE

[... more matches ...]
```

### Evaluation

```bash
# Run test suite
python tests/test_matching.py

# Expected: Grade B+ (0.82), ~2 minutes
```

---

## 🔬 Approach & Key Decisions

### Architecture

**Multi-Provider LLM System**
- Supports OpenAI (GPT-4o, GPT-5), Anthropic (Claude 3.7), Google (Gemini 2.5)
- Intelligent routing with automatic fallback
- Gemini-first strategy for cost-effectiveness

**Optimized Ranker with Parallel Processing**
- **Parallel execution**: Up to 20 concurrent LLM calls for speed
- **Batch processing**: Groups 3-5 trials per LLM call to reduce API overhead  
- **Multi-provider distribution**: Spreads load across Gemini, OpenAI, and Anthropic
- **Multi-level caching**: Memory + disk cache with 50-80% hit rate
- **Score normalization**: Auto-adjusts conservative LLM scoring

**Medical Safety**
- Oncology-specific eligibility checking
- Biomarker matching (ER/PR/HER2, EGFR, KRAS, etc.)
- Stage appropriateness validation
- Safety concern flagging

### Key Design Choices

**1. Complexity-Based Model Selection**
- Model choice based on case complexity, not urgency
- Ensures high-quality analysis for all patients
- Priority: Medical accuracy > Speed

**2. Score Normalization**
- Auto-adjusts if LLM scores too conservatively (avg < 0.4)
- Applies boost factor (max 1.5x) to maintain consistency
- Transparent logging of normalization events

**3. Intelligent Batching**
- Groups trials into batches for efficiency
- Smaller batches (3 trials) for Gemini to accommodate reasoning tokens
- Larger batches (5 trials) for OpenAI/Anthropic

**4. Cache Persistence**
- Shared ranker instance across patients
- 50-80% cache hit rate in typical usage
- Significantly reduces API costs and latency

### Evaluation Suite

**Synthetic Patient Generator**
- 1000 diverse patients (standard, edge cases, adversarial, equity stress)
- Realistic demographics and biomarker distributions
- Comprehensive coverage of oncology scenarios

**7-Judge LLM Ensemble**
- Accuracy, Safety, Completeness, Bias, Robustness, Clinical Text, TrialGPT
- Multi-model evaluation (GPT-4o, Claude 3.7, Gemini 2.5 Pro)
- Agreement metrics (Krippendorff's α, Fleiss' Kappa)

**Comprehensive Metrics**
- Clinical: nDCG, Precision, Recall, MRR, F1
- Safety: Violation rate, critical miss detection
- Performance: Latency, cache hit rate, cost per patient

---

## 📊 Performance

| Metric | Value | Status |
|--------|-------|--------|
| **Overall Grade** | **B+ (0.82)** | Good |
| Avg Latency | **10.5s** | Within target |
| P95 Latency | 14.4s | Within target |
| Precision@10 | 1.00 | High |
| Recall@10 | 0.10 | Low |
| nDCG@10 | 0.97 | High |
| Success Rate | 100% | Complete |
| Match Quality | 0.69 | Acceptable |
| Avg Score | 0.63 | Calibrated |

**How Parallel Processing Achieved 10.5s Latency:**
- **Without parallelization**: ~60s (40 trials × 1.5s per LLM call)
- **With 20 concurrent calls**: 10.5s (83% reduction)
- **Key technique**: Distributes trials across multiple provider APIs simultaneously
- **Result**: Meets <15s target despite analyzing 40 trials

---

## 📁 Project Structure

```
oncomatch-ai/
├── src/
│   ├── match.py                      # Main entry point
│   └── oncomatch/
│       ├── optimized_ranker.py       # High-performance ranker
│       ├── llm_providers.py          # LLM provider adapters
│       ├── llm_registry.py           # Model routing
│       ├── biomcp_wrapper.py         # Trial fetching
│       ├── models.py                 # Data models
│       └── evaluation/
│           ├── evaluator.py          # Evaluation orchestrator
│           ├── judge_ensemble.py     # 7-judge ensemble
│           ├── synthetic_patients.py # Patient generator
│           └── metrics_core.py       # Metrics calculations
├── tests/
│   └── test_matching.py              # Main test suite
├── scripts/
│   ├── run_full_evaluation.py        # Comprehensive eval
│   ├── test_judge_ensemble.py        # Judge verification
│   └── demo_synthetic_evaluation.py  # Synthetic demo
├── patients.csv                       # 30 sample patients
├── requirements.txt                   # Dependencies
└── README.md                          # This file
```

---

## ⚙️ Configuration

### Environment Variables

```bash
# LLM Providers (at least one required)
GOOGLE_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...

# Performance (optional)
MAX_CONCURRENT_LLM=20      # Parallel calls
LLM_BATCH_SIZE=5           # Trials per batch
CACHE_TTL_HOURS=24         # Cache duration
```

### Model Selection

The system automatically selects models based on:
- Case complexity (biomarkers, stage, prior therapies)
- Provider availability and rate limits
- Cost and latency constraints

Default routing: **Gemini** → OpenAI → Anthropic

---

## 🧪 Testing

```bash
# Basic test
python src/match.py --patient_id P001

# Full evaluation
python tests/test_matching.py

# Judge ensemble
python scripts/test_judge_ensemble.py

# Synthetic cohort demo
python scripts/demo_synthetic_evaluation.py
```

---

## 📟 Complete Command Reference

### Basic Matching

```bash
# Match a single patient (default: optimized mode, 40 trials)
python src/match.py --patient_id P002

# Match a specific patient with trial limit
python src/match.py --patient_id P002 --max_trials 10

# Match all patients
python src/match.py --all

# Match all patients with trial limit (faster)
python src/match.py --all --max_trials 20

# Use standard ranker (slower, more thorough)
python src/match.py --patient_id P002 --no-optimized
```

### Evaluation & Testing

```bash
# Run basic test suite (5 patients)
python tests/test_matching.py

# Run evaluation with specific number of patients
python tests/test_matching.py --n-patients 5

# Run comprehensive evaluation (1000 synthetic patients)
python scripts/run_full_evaluation.py

# Test judge ensemble
python scripts/test_judge_ensemble.py

# Demo with synthetic patients
python scripts/demo_synthetic_evaluation.py

# Run unit tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_biomcp_client.py
```

### Advanced Options

```bash
# Match with specific mode (affects model selection)
python src/match.py --patient_id P002 --mode fast
python src/match.py --patient_id P002 --mode balanced
python src/match.py --patient_id P002 --mode accurate

# Enable debug logging
python src/match.py --patient_id P002 --debug

# Specify output format
python src/match.py --patient_id P002 --output json
python src/match.py --patient_id P002 --output csv

# Save results to file
python src/match.py --patient_id P002 --output-file results.json
```

### Cache Management

```bash
# Clear LLM response cache
rm -rf cache/llm_results/*

# Clear trial data cache
rm -rf outputs/cache/biomcp/*

# View cache contents
ls -lah cache/llm_results/

# Check cache size
du -sh cache/
```

### Utility Commands

```bash
# List all patient IDs
python -c "import pandas as pd; print(pd.read_csv('patients.csv')['patient_id'].tolist())"

# Check installed dependencies
pip list | grep -E "openai|anthropic|google-generativeai|biomcp"

# Verify API keys are set
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('✅ Keys loaded' if os.getenv('GOOGLE_API_KEY') or os.getenv('OPENAI_API_KEY') else '❌ No keys found')"

# View system logs
tail -f outputs/logs/*.log

# Check system status
python -c "from src.oncomatch.llm_providers import *; print('System OK')"
```

### Development Commands

```bash
# Run type checking
mypy src/

# Run code formatter
black src/ tests/

# Run linter
ruff check src/ tests/

# Install development dependencies
pip install -r requirements.txt
pip install pytest mypy black ruff

# Run all quality checks
black src/ tests/ && ruff check src/ tests/ && mypy src/
```

### Environment Setup

```bash
# Create .env from example
cp .env.example .env

# Edit environment variables
nano .env  # or vim, code, etc.

# Load environment variables
export $(cat .env | xargs)

# Test environment configuration
python -c "from src.oncomatch.config import *; print('Config OK')"
```

### Performance Testing

```bash
# Quick performance test (5 trials)
time python src/match.py --patient_id P001 --max_trials 5

# Standard performance test (10 trials)
time python src/match.py --patient_id P001 --max_trials 10

# Full performance test (40 trials)
time python src/match.py --patient_id P001 --max_trials 40

# Batch performance test (5 patients)
time python src/match.py --all --max_trials 10 | head -20
```

### Help & Documentation

```bash
# View main CLI help
python src/match.py --help

# View test suite help
python tests/test_matching.py --help

# List all available scripts
ls -1 scripts/*.py

# View documentation
open docs/INDEX.md  # macOS
xdg-open docs/INDEX.md  # Linux
start docs/INDEX.md  # Windows
```

---

## 🎯 Known Limitations & TODOs

### Current Limitations

**1. Low Recall at k=10**: ~10% of relevant trials found
- Trade-off: Perfect precision (1.00) but misses many relevant trials
- System finds ~26 relevant trials per patient but only returns top 10
- **TODO**: Increase default results to top 20-30 for better coverage

**2. Rate Limiting**: LLM API constraints
- OpenAI: 500 RPM limit can cause delays during batch processing
- Anthropic: 50 RPM limit often hit, causing fallbacks
- Gemini: 60 RPM but token limits can cause MAX_TOKENS errors
- **TODO**: Implement request queuing and better rate limit handling

**3. Trial Coverage**: Limited to ClinicalTrials.gov
- No proprietary trials (e.g., pharma-specific databases)
- No international registries (EU, Asia-Pacific)
- Rate limited to 45 requests/min via BioMCP SDK
- **TODO**: Add support for additional trial databases

**4. Geographic Filtering**: Not implemented
- No distance calculations from patient to trial sites
- No travel burden assessment
- **TODO**: Integrate geographic APIs for distance calculations

**5. Biomarker Complexity**: Basic matching only
- Simple string matching for biomarkers
- No variant-level precision (e.g., EGFR L858R vs del19)
- **TODO**: Implement detailed molecular matching logic

### Planned Improvements (TODOs)

**Immediate**
- [ ] Increase default results from 10 to 20 trials for better recall
- [ ] Add request queuing to handle rate limits gracefully
- [ ] Implement connection pooling for HTTP clients
- [ ] Add retry logic with exponential backoff for rate limits

**Short-term**
- [ ] Add support for variant-level biomarker matching
- [ ] Implement geographic distance calculations
- [ ] Add distributed caching (Redis) for production deployment
- [ ] Create rate limit monitoring dashboard

**Medium-term**
- [ ] Integrate additional trial databases (EU Clinical Trials Register, etc.)
- [ ] Add insurance eligibility pre-screening
- [ ] Implement travel burden scoring
- [ ] Add multi-language support for international trials

**Long-term**
- [ ] Expert clinician validation study with oncologists
- [ ] Fine-tune models on historical enrollment data
- [ ] Build real-world trial enrollment tracking
- [ ] Develop custom medical LLM for trial matching

---

## 📚 Additional Documentation

- [Setup Guide](docs/SETUP.md) - Detailed installation
- [Architecture](docs/ARCHITECTURE.md) - System design
- [Metrics Guide](docs/METRICS_GUIDE.md) - Evaluation methodology

## 📝 How to Run This Solution

### Quick Start (2 minutes)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add API key to .env
echo "GOOGLE_API_KEY=your_key_here" > .env

# 3. Run matching for a patient
python src/match.py --patient_id P002

# 4. Run evaluation suite
python tests/test_matching.py
```

### Approach

**Multi-Provider LLM Strategy**: Uses Gemini 2.5 Flash as primary provider, with OpenAI GPT-4o and Anthropic Claude as fallbacks for resilience against rate limits.

**Parallel Processing & Optimization**: The system achieves 10.5s latency through aggressive parallelization - up to 20 concurrent LLM calls distributed across multiple providers (Gemini, OpenAI, Anthropic). Combined with batch processing (3-5 trials per call) and caching, this reduces processing time by ~80% compared to sequential processing.

**Medical Safety**: Conservative scoring when uncertain, safety checks, and 7-judge ensemble validation.

**Caching**: Shared ranker instance with persistent cache maintains 50-80% hit rate.

### Key Architectural Decisions

1. **Parallel Processing Architecture**: Core optimization enabling 10.5s latency
   - 20 concurrent LLM calls maximum
   - Distributed across 3 providers to avoid rate limits
   - Async/await throughout for non-blocking operations
2. **Complexity-based routing**: All patients get high-quality analysis
3. **Score normalization**: Auto-adjusts conservative LLM scoring  
4. **Provider-specific optimization**: Different batch sizes per provider
5. **Ensemble validation**: Multiple specialized judges for evaluation

---

**Version**: 1.0.0  
**Status**: Grade B+ (0.82)  
**Performance**: 10.5s average latency