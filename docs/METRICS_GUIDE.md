# Research-Grade Evaluation Metrics Guide

This document explains the advanced metrics used in OncoMatch-AI's evaluation suite, their clinical significance, and their importance in ML research.

## 📊 Core Clinical Metrics

### nDCG@k (Normalized Discounted Cumulative Gain)
**What it measures**: Quality of ranking when relevance is graded (not just binary)

**Clinical significance**: In clinical trials, some matches are better than others. A Phase 3 trial for the exact cancer type/stage is better than a Phase 1 trial for a related cancer. nDCG captures these nuances.

**Interpretation**:
- 1.0 = Perfect ranking (best trials first)
- 0.8+ = Excellent clinical utility
- 0.6-0.8 = Good, but room for improvement
- <0.6 = Suboptimal ranking

**Formula**: `DCG@k / IDCG@k` where DCG = Σ(2^relevance - 1) / log2(rank + 1)

### Precision@k, Recall@k, F1@k
**What they measure**:
- **Precision**: Of the top-k trials shown, what fraction are actually relevant?
- **Recall**: Of all relevant trials, what fraction appear in the top-k?
- **F1**: Harmonic mean balancing precision and recall

**Clinical significance**: 
- High precision = Less time wasted on irrelevant trials
- High recall = Important trials aren't missed
- F1 = Overall matching quality

**Target values**:
- Precision@10 > 0.7 (70% of shown trials are relevant)
- Recall@10 > 0.5 (50% of all relevant trials are found)
- F1@10 > 0.6 (balanced performance)

### MRR (Mean Reciprocal Rank)
**What it measures**: How quickly the first relevant trial appears

**Clinical significance**: Time is critical for cancer patients. Finding the first good match quickly can accelerate treatment decisions.

**Interpretation**:
- 1.0 = First result is always relevant
- 0.5 = First relevant result at position 2 on average
- 0.33 = First relevant result at position 3 on average

## 🛡️ Safety & Error Metrics

### Safety Violation Rate
**What it measures**: Frequency of recommending trials that would be unsafe or inappropriate

**Clinical significance**: Patient safety is paramount. This metric catches dangerous recommendations like:
- Trials requiring ECOG 0-1 for ECOG 3+ patients
- Trials with conflicting medications
- Trials with unmet exclusion criteria

**Target**: < 0.01 (less than 1% violations in top-k)

### Critical Miss Rate
**What it measures**: How often the system fails to identify known excellent matches

**Clinical significance**: Missing the best available trial could delay optimal treatment. This is especially critical for:
- Rare mutations with targeted therapies
- Time-sensitive aggressive cancers
- Limited trial availability

**Target**: < 0.1 (miss less than 10% of gold-standard trials)

## ⚖️ Equity & Fairness Metrics

### Subgroup Performance Disparity
**What it measures**: Variation in matching quality across demographic groups

**Clinical significance**: Ensures the system doesn't perpetuate healthcare disparities. Checks performance for:
- Different racial/ethnic groups
- Age extremes (pediatric, elderly)
- Rare vs common biomarkers
- Rural vs urban patients

**Metrics**:
- Standard deviation of F1 scores across groups
- Range (max - min performance)
- Coefficient of variation

**Target**: CV < 0.15 (less than 15% relative variation)

### Biomarker Rarity Impact
**What it measures**: Matching performance for rare vs common mutations

**Clinical significance**: Patients with rare mutations shouldn't be disadvantaged. The system should:
- Find niche trials for rare mutations
- Not over-prioritize common mutations
- Balance specificity and coverage

**Target**: Equity gap < 0.1 (less than 10% difference in match rates)

## 🤝 Ensemble Agreement Metrics

### Krippendorff's Alpha (α)
**What it measures**: Inter-rater reliability accounting for chance agreement

**Clinical significance**: Multiple clinical experts often disagree on trial eligibility. High α means the AI judges are more consistent than human experts.

**Interpretation**:
- α > 0.8 = High reliability
- α = 0.67-0.8 = Acceptable agreement
- α < 0.67 = Low agreement (needs investigation)

### Fleiss' Kappa (κ)
**What it measures**: Agreement among multiple raters for categorical decisions

**Clinical significance**: Ensures consistent eligibility decisions across different AI models/judges.

**Interpretation**:
- κ > 0.81 = Almost perfect agreement
- κ = 0.61-0.80 = Substantial agreement
- κ = 0.41-0.60 = Moderate agreement
- κ < 0.40 = Poor agreement

### Judge Vote Patterns
**What it measures**: Distribution of unanimous vs split decisions

**Clinical significance**: 
- Unanimous eligible = High confidence matches
- High disagreement = Cases needing human review
- Split decisions = Borderline eligibility

## 🔬 Ablation Study Metrics

### Component Contribution Analysis
**What it measures**: Performance impact of removing system components

**Clinical significance**: Identifies which features are most critical for:
- Safety (safety checks, exclusion criteria)
- Accuracy (LLM ranking, biomarker matching)
- Efficiency (caching, rule-based filtering)

**Key ablations**:
- `no_safety`: Removes safety checks (should decrease safety score)
- `no_llm`: Uses only rules (should decrease nDCG/F1)
- `rule_only`: No ML ranking (baseline comparison)
- `no_biomarker`: Ignores genetic markers (should hurt precision)

### Robustness Metrics
**What it measures**: Stability under input perturbations

**Clinical significance**: Real patient data is noisy. The system should be robust to:
- Missing data fields
- Typos in biomarker names
- Inconsistent staging notation
- Ambiguous diagnoses

**Metrics**:
- Rank correlation after perturbation
- Score deviation
- Top-10 stability

## ⚡ Performance & Cost Metrics

### Latency Percentiles
**What it measures**: Response time distribution

**Clinical significance**: Clinical workflows have time constraints:
- P50 < 5s: Good for interactive use
- P95 < 15s: Acceptable for clinical decision support
- P99 < 30s: Maximum tolerable wait

### Cost Analysis
**What it measures**: LLM API costs per patient

**Clinical significance**: Healthcare systems have budget constraints. Tracks:
- Cost per patient match
- Monthly projected costs
- Cost by model/component

**Targets**:
- < $0.50 per patient for routine use
- < $2.00 per patient for complex cases

## 🔍 Error Analysis Metrics

### Hard Case Identification
**What it identifies**: Most challenging patient-trial matches

**Clinical significance**: These cases reveal system limitations:
- Rare cancers with few trials
- Complex comorbidities
- Contradictory eligibility criteria
- Edge cases in staging/grading

**Metrics**:
- Confidence scores
- Judge disagreement
- Number of matches found

### Failure Mode Analysis
**What it identifies**: Common patterns in matching failures

**Clinical significance**: Helps prioritize improvements:
- "No matches found" → Need broader search
- "Low confidence" → Need better evidence
- "High disagreement" → Need clearer criteria
- "Exclusion heavy" → Need safety focus

## 📈 Interpretation Guidelines

### Overall System Grade

| Grade | Criteria | Clinical Readiness |
|-------|----------|-------------------|
| **A+** | nDCG>0.9, F1>0.85, Safety>0.99, α>0.8 | Production ready |
| **A** | nDCG>0.85, F1>0.8, Safety>0.98, α>0.75 | Pilot ready |
| **B+** | nDCG>0.8, F1>0.75, Safety>0.97, α>0.7 | Supervised use |
| **B** | nDCG>0.75, F1>0.7, Safety>0.95, α>0.65 | Research use |
| **C** | nDCG>0.7, F1>0.65, Safety>0.93, α>0.6 | Development only |
| **D** | Below C thresholds | Needs improvement |

### Red Flags 🚩

Watch for these warning signs:
- Safety violation rate > 2%
- Critical miss rate > 20%
- Equity gap > 20%
- Judge agreement α < 0.5
- P95 latency > 30s
- No matches for >10% of patients

## 🎯 Using Metrics for Improvement

### Prioritization Matrix

| Metric Issue | Impact | Suggested Action |
|--------------|--------|------------------|
| Low nDCG | Poor ranking | Improve relevance scoring |
| Low precision | Irrelevant results | Tighten eligibility criteria |
| Low recall | Missing trials | Broaden search parameters |
| High safety violations | Patient risk | Strengthen exclusion checks |
| Poor equity | Bias | Add fairness constraints |
| Low agreement | Inconsistency | Align judge training |
| High latency | Poor UX | Optimize queries/caching |

### Benchmark Targets

For publication-quality results, aim for:
- **Clinical metrics**: nDCG@10 > 0.85, F1@10 > 0.8
- **Safety**: Violation rate < 1%
- **Equity**: Disparity CV < 10%
- **Agreement**: Krippendorff's α > 0.75
- **Performance**: P95 < 15 seconds
- **Cost**: < $0.50 per patient

## 📚 References

- Järvelin, K., & Kekäläinen, J. (2002). Cumulated gain-based evaluation of IR techniques. ACM TOIS.
- Krippendorff, K. (2011). Computing Krippendorff's alpha-reliability.
- Fleiss, J. L. (1971). Measuring nominal scale agreement among many raters. Psychological Bulletin.
- Powers, D. M. (2011). Evaluation: from precision, recall and F-measure to ROC, informedness, markedness and correlation.
- Barocas, S., Hardt, M., & Narayanan, A. (2019). Fairness in machine learning. NIPS Tutorial.

