#!/usr/bin/env python3
"""Launch auto_audio_assist."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aaa.app import main
main()
