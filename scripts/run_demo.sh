#!/usr/bin/env bash
# OncoMatch AI Demo Script

set -e

echo "========================================="
echo "    OncoMatch AI - Clinical Demo"
echo "========================================="
echo ""

# Load environment if available
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "✓ Environment loaded from .env"
else
    echo "⚠ No .env file found, using defaults"
fi

# Run for Kerry Bird (first patient in CSV)
echo ""
echo "Running trial matching for patient: Kerry Bird"
echo "Cancer Type: Breast, Stage I"
echo ""

# Run the demo
PYTHONPATH=src python -m oncomatch.match --patient_id "Kerry Bird" --format human

# Also save JSON output
echo ""
echo "Saving detailed results to demo_report.json..."
PYTHONPATH=src python -m oncomatch.match --patient_id "Kerry Bird" --format json > demo_report.json

echo ""
echo "========================================="
echo "Demo complete! Results saved to demo_report.json"
echo "========================================="
