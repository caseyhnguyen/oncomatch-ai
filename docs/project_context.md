# OncoMatch AI - Project Context & Overview

*Last Updated: September 2025*

## 🎯 Project Mission

OncoMatch AI is an advanced clinical trial matching system that leverageslanguage models to connect oncology patients with relevant clinical trials. The system addresses the critical challenge that only ~5% of cancer patients participate in clinical trials, often due to difficulty finding suitable matches.

## 📋 Core Requirements

### Primary Deliverables
1. **Python package** for patient-trial matching
2. **CLI interface** for easy interaction
3. **Comprehensive evaluation suite** with clinical metrics
4. **Production-ready architecture** with <15s latency target

### Key Commands
```bash
# Match a specific patient
python src/match.py --patient_id P002

# Run evaluation suite
python tests/test_matching.py --eval

# Quick test with metrics
python tests/test_matching.py --eval --n_patients 5
```

## 🏥 Clinical Context

### The Challenge
- **40,000+** active cancer trials on ClinicalTrials.gov
- **Complex eligibility criteria** with medical terminology
- **Time-sensitive decisions** for cancer patients
- **Geographic constraints** affecting accessibility
- **Biomarker requirements** becoming increasingly specific

### The Solution
- **AI-powered matching** using GPT-5, Claude 4, and Gemini 2.5
- **Real-time trial data** from BioMCP SDK and ClinicalTrials.gov
- **Comprehensive safety checks** to avoid dangerous matches
- **Explainable recommendations** with clinical rationale
- **Performance optimization** for rapid results

## 📊 Data Architecture

### Patient Data Schema (patients.csv)
30 oncology patients with comprehensive profiles:

#### Demographics & Basics
- **Identity**: name, age, gender, race
- **Location**: city, state (for geographic matching)
- **Physical**: height_cm, weight_kg, bmi

#### Cancer Information
- **Type**: breast, lung, colorectal, prostate, etc.
- **Stage**: I-IV with substages (e.g., IIIA, IIIB)
- **Grade**: 1-3 or low/intermediate/high
- **Timeline**: diagnosis_date, recurrence_date
- **Status**: is_recurrence, treatment_stage

#### Treatment History
- **Stage**: neoadjuvant, adjuvant, metastatic
- **Interventions**: surgeries (comma-separated)
- **Prior therapy**: previous_treatments (drugs)
- **Current**: current_medications

#### Biomarkers & Molecular Profile
- **Detected**: BRCA1/2, HER2, ER, PR, PD-L1, EGFR, etc.
- **Ruled out**: Negative test results
- **Format**: "BRCA1,HER2+,ER+" (comma-separated)

#### Clinical Status
- **Performance**: ECOG status (0-4 scale)
- **Lifestyle**: smoking_status, drinking_status
- **Comorbidities**: other_conditions
- **History**: family_history
- **Intent**: patient treatment preferences

### Trial Data Sources

#### BioMCP SDK Integration
```python
from biomcp import search_trials, TrialQuery, RecruitingStatus

# Key capabilities:
# - Rate limit: 45 requests/minute
# - Returns: NCT IDs, titles, eligibility, phases
# - Geographic filtering: lat/long/distance
# - Status filtering: recruiting, active, etc.
# - Condition search: cancer type, biomarkers
```

#### ClinicalTrials.gov API
- Direct access for comprehensive trial data
- 40,000+ active cancer trials
- Real-time updates
- Detailed eligibility criteria

## 🤖 AI Model Strategy (September 2025)

### Model Tiers & Selection
| Complexity | Models Used | Use Case | Latency |
|------------|------------|----------|---------|
| **Simple** | Gemini 2.5 Flash-Lite, GPT-4.1 nano | Basic eligibility | 2-3s |
| **Medium** | GPT-4o-mini, Gemini 2.5 Flash | Standard matching | 4-5s |
| **Complex** | GPT-5, Claude 4 Opus, O3 | Complex medical reasoning | 7-10s |
| **Safety** | Claude 4 Sonnet/Opus | Ethics & safety validation | 5-6s |

### Intelligent Routing Logic
```python
def select_model(patient, trial):
    complexity = calculate_complexity(patient, trial)
    
    # Model selection based on complexity ONLY
    # All patients deserve best quality analysis
    if complexity >= 7:  # Complex medical
        return premium_models  # GPT-5, Claude 4
    elif complexity >= 4:
        return balanced_models  # GPT-4o, Gemini Pro
    else:
        return efficient_models  # Still good quality

def prioritize_workflow(patient):
    urgency = calculate_urgency(patient)
    
    # Urgency affects WORKFLOW, not model quality
    if urgency >= 0.8:  # Critical cases
        notify_care_team()
        expedite_review()  # Same-day processing
    elif urgency >= 0.6:
        schedule_priority_review()  # 24-48 hours
```

## 🔬 Technical Architecture

### Core Components
1. **Trial Fetcher** (`biomcp_client.py`)
   - BioMCP SDK integration
   - Caching layer
   - Rate limiting

2. **LLM Ranker** (`llm_ranker.py`)
   - Multi-provider support
   - Parallel processing
   - JSON schema validation

3. **Model Registry** (`llm_registry.py`)
   - Dynamic routing
   - Fallback chains
   - Cost optimization

4. **Evaluation Suite** (`evaluation/`)
   - Synthetic patient generation
   - Judge ensemble
   - Clinical metrics

### Performance Features
- **Parallel Processing**: 5 concurrent LLM calls
- **Caching**: TTL-based with 24h default
- **Retries**: Exponential backoff
- **Timeouts**: Request-level controls
- **Observability**: Structured logging

### Performance Reality (Sept 30, 2025)
- **40 trials analyzed**: 10.5s average (target: <15s)
- **Grade**: B+ (0.82)
- **P95 Latency**: 14.4s
- **Trade-off**: High precision (1.00) but low recall at k=10 (0.10)
- **Note**: System finds ~26 relevant trials per patient but only returns top 10

## 📈 Success Metrics

### Technical Metrics (Achieved Sept 30, 2025)
- **Latency (40 trials)**: 10.5s achieved (target <15s)
- **P95 Latency**: 14.4s
- **Max Latency**: 15.6s
- **Throughput**: ~340 patients/hour with parallel processing
- **Cost**: ~$0.05-$0.20 per patient depending on complexity

### Clinical Metrics (Achieved Sept 30, 2025)
- **nDCG@10**: 0.97 achieved (target >0.80)
- **MRR**: 1.00 achieved (target >0.80)
- **Precision@10**: 1.00 achieved (target >0.70)
- **Recall@10**: 0.10 achieved (target >0.45) - below target

### Business Impact
- **Trial enrollment**: Target 20% increase
- **Time to match**: 90% reduction
- **Geographic coverage**: National scope
- **Equity**: Improved minority representation

## 🚀 Development Roadmap

### Completed (September 2025)
✓ Multi-LLM integration with current models
✓ Routing based on complexity  
✓ Evaluation framework
✓ Performance: 10.5s (target <15s)
✓ Safety validation system
✓ Grade: B+ (0.82)

### Next Steps
- [ ] Rule-based pre-filtering
- [ ] Enhanced biomarker logic
- [ ] Patient preference integration
- [ ] Multi-language support

- [ ] Edge deployment for speed
- [ ] Fine-tuned medical models
- [ ] Real-time trial updates
- [ ] Clinical decision support UI

### Q2 2026
- [ ] Predictive enrollment modeling
- [ ] Outcome tracking
- [ ] Physician collaboration tools
- [ ] FDA submission preparation

## 🔒 Compliance & Safety

### Regulatory Compliance
- **HIPAA**: No PHI in logs, data minimization
- **FDA**: Following clinical decision support guidelines
- **IRB**: Research protocol approved
- **Data Security**: Encryption at rest and in transit

### Safety Measures
1. **Conservative matching** when uncertain
2. **Explicit contraindication checks**
3. **Human-in-the-loop** for critical decisions
4. **Audit trail** for all recommendations
5. **Regular validation** against clinical outcomes

## 💡 Key Innovations

### Technical Innovations
1. **Complexity-aware routing** - Right model for each case
2. **Judge ensemble** - Multi-model validation
3. **Synthetic evaluation** - Realistic test cohorts
4. **Budget optimization** - Cost-effective processing

### Clinical Innovations
1. **Biomarker prioritization** - Molecular profile matching
2. **Geographic optimization** - Travel burden consideration
3. **Stage-appropriate** - Treatment line awareness
4. **Safety-first** - Conservative recommendations

## 📝 Usage Examples

### Basic Matching
```bash
# Single patient, fast mode
python src/match.py --patient_id P002 --mode fast

# All patients with safety checks
python src/match.py --all --enable_safety

# Limited trials for testing
python src/match.py --patient_id P002 --max_trials 5
```

### Advanced Evaluation
```bash
# Full evaluation with all metrics
python tests/test_matching.py --eval --metrics all

# Judge agreement analysis
python tests/test_matching.py --eval --show_agreement

# Ablation studies
python tests/test_matching.py --eval --ablation
```

## 🤝 Stakeholders

### Primary Users
- **Oncologists**: Clinical decision support
- **Research coordinators**: Trial enrollment
- **Patients**: Self-service matching
- **Pharmaceutical companies**: Recruitment optimization

### Technical Team
- **ML Engineers**: Model optimization
- **Clinical Informaticists**: Medical accuracy
- **DevOps**: Production deployment
- **QA**: Validation and testing

## 📚 Resources

### Documentation
- [Architecture Overview](ARCHITECTURE.md)
- [Metrics Guide](METRICS_GUIDE.md)
- [Performance Optimization](PERFORMANCE_NOTES.md)
- [Model Specifications](MODELS_SEPTEMBER_2025.md)

### External Resources
- [ClinicalTrials.gov](https://clinicaltrials.gov)
- [BioMCP SDK Documentation](https://biomcp.readthedocs.io)
- [NIH Trial Matching](https://www.cancer.gov/about-cancer/treatment/clinical-trials)