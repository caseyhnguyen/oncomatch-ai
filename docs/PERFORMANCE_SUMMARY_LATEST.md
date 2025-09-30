# Performance Summary - September 30, 2025

## Overall Grade: B+ (0.819/1.000)
**Status**: Good performance

## 📊 Performance Metrics

### System Performance
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Average Response Time** | **10.46s** | <15s | Within target |
| **P95 Latency** | 14.38s | <30s | Within target |
| **Max Latency** | 15.64s | <60s | Within target |
| **Success Rate** | 100% | 100% | Met |

### Clinical Effectiveness
| Metric | @3 | @5 | @10 | Status |
|--------|-----|-----|------|--------|
| **Precision** | 1.000 | 1.000 | 1.000 | High |
| **Recall** | 0.029 | 0.048 | 0.096 | Low |
| **F1 Score** | - | 0.092 | 0.175 | Low |
| **nDCG** | 0.937 | 0.947 | 0.970 | High |
| **MRR** | 1.000 | 1.000 | 1.000 | High |

### Component Breakdown
| Component | Weight | Score | Details |
|-----------|--------|-------|---------|
| **Clinical Effectiveness** | 35% | 0.661 | Excellent ranking, low recall |
| **Match Quality** | 20% | 0.687 | Well-calibrated scores |
| **Success Rate** | 25% | 1.000 | All patients matched |
| **Performance** | 20% | 1.000 | Exceeds all latency targets |

## 📈 Per-Patient Results

| Patient | Trials | Relevant | Top Score | Avg Score | Time |
|---------|--------|----------|-----------|-----------|------|
| P001 | 40 | 27 | 0.95 | 0.65 | 15.64s |
| P002 | 40 | 28 | 0.95 | 0.67 | 9.35s |
| P003 | 40 | 31 | 0.90 | 0.70 | 9.27s |
| P004 | 40 | 26 | 0.85 | 0.64 | 9.30s |
| P005 | 40 | 18 | 0.85 | 0.51 | 8.73s |

**Average**: 26 relevant trials per patient, 0.90 average top score

## Key Achievements

1. **Latency**: 10.5s average (target <15s)
2. **Precision**: 100% of recommended trials are relevant
3. **Ranking**: nDCG of 0.97
4. **Grade**: B+ (0.82)
5. **Consistency**: P95 latency at 14.4s

## ⚠️ Areas for Optimization

### Low Recall (Primary Issue)
- Only 9.6% of relevant trials in top 10
- System finds 26 relevant trials but returns only 10
- **Solution**: Return top 20-30 trials by default

### Rate Limiting Challenges
- Hitting API limits during peak usage
- Causes fallback delays and potential timeouts
- **Solution**: Implement request queuing and better rate limit handling

## 🚀 Recommendations

### Immediate Actions
1. Change default from top 10 to top 20 trials (would double recall)
2. Implement request queuing for rate limit management
3. Add connection pooling to reduce connection overhead

### Performance Impact
With these changes, expect:
- Recall@20: ~20% (2x improvement)
- Latency: Maintain <15s with optimizations
- Grade: Potential A- (0.85+) achievable

## Conclusion

The system achieves Grade B+ (0.82) with latency of 10.5s (target <15s). The primary limitation is low recall due to returning only top 10 trials. Adjustments to return more results could improve recall while maintaining speed.

**Status**: Ready for pilot deployment with monitoring
