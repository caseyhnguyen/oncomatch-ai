#!/usr/bin/env python
"""
Wrapper script for the clinical trial matching system.
Run: python src/match.py --patient_id P002
"""

import sys
import os

# Add src to path so we can import oncomatch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the main module
from oncomatch.match import main

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())