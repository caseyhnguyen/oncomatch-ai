# OncoMatch AI Documentation

This directory contains comprehensive documentation for the OncoMatch AI clinical trial matching system.

---

## 📚 Core Documentation

### [ARCHITECTURE.md](ARCHITECTURE.md)
Complete system architecture and technical design
- Component descriptions and data flow
- LLM ranking system details
- Performance optimizations
- Medical safety features
- Deployment considerations

### [SETUP.md](SETUP.md)
Installation and configuration guide
- Prerequisites and API keys
- Step-by-step installation
- Configuration options
- Troubleshooting guide
- Quick reference

### [METRICS_GUIDE.md](METRICS_GUIDE.md)
Evaluation metrics and methodology
- Clinical metrics (nDCG, Precision, Recall, MRR, F1)
- Safety and error metrics
- Equity and fairness metrics
- Judge ensemble agreement
- Interpretation guidelines

---

## 🔍 Reference Documentation

### [MODELS_SEPTEMBER_2025.md](MODELS_SEPTEMBER_2025.md)
LLM models and specifications (September 2025)
- OpenAI models (GPT-5, GPT-4o, O-series)
- Anthropic models (Claude 3.7, Claude 3.5)
- Google models (Gemini 2.5 Pro, Flash)
- Model selection strategy
- Performance benchmarks

### [PERFORMANCE_NOTES.md](PERFORMANCE_NOTES.md)
Performance optimization and benchmarks
- Current performance metrics
- Optimization strategies
- Performance tuning parameters
- Troubleshooting guide
- Best practices

### [project_context.md](project_context.md)
Project background and clinical context
- Project mission and goals
- Clinical challenges and solutions
- Data architecture
- AI model strategy
- Success metrics

---

## 🚀 Quick Start Guide

### For New Users
1. Start with [SETUP.md](SETUP.md) for installation
2. Read [project_context.md](project_context.md) for background
3. Run `python src/match.py --patient_id P002` to test

### For Developers
1. Review [ARCHITECTURE.md](ARCHITECTURE.md) for system design
2. Check [MODELS_SEPTEMBER_2025.md](MODELS_SEPTEMBER_2025.md) for LLM details
3. See [PERFORMANCE_NOTES.md](PERFORMANCE_NOTES.md) for optimization tips

### For Evaluators
1. Understand metrics in [METRICS_GUIDE.md](METRICS_GUIDE.md)
2. Run `python tests/test_matching.py` for evaluation
3. Review results and interpret grades

---

## 📊 System Overview

### Current Performance (September 30, 2025)

| Metric | Value | Status |
|--------|-------|--------|
| **Overall Grade** | **B+ (0.82)** | Good |
| Avg Latency | **10.5s** | Within target |
| P95 Latency | 14.4s | Within target |
| Precision@10 | 1.00 | High |
| Recall@10 | 0.10 | Low |
| nDCG@10 | 0.97 | High |
| Success Rate | 100% | Complete |

### Key Features

- **Multi-LLM Support**: OpenAI, Anthropic, Google
- **Intelligent Routing**: Complexity-based model selection
- **Performance**: 10.5s average latency (target: <15s)
- **Medical Safety**: Conservative scoring, safety checks
- **Comprehensive Evaluation**: 1000 synthetic patients, 7-judge ensemble

---

## 📝 Documentation Standards

All documentation follows these principles:
- **Clear and Concise**: Easy to understand
- **Production-Focused**: Ready-to-use information
- **Well-Organized**: Logical structure
- **Up-to-Date**: Reflects current system state (September 2025)

---

## 🔄 Getting Help

- Check the appropriate doc for your needs
- Run `python src/match.py --help` for CLI options
- Open a GitHub issue for questions
- Review code comments for implementation details

---

**Last Updated**: September 2025  
**Version**: 1.0.0  
**Status**: Production Ready