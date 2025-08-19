#!/usr/bin/env python3
"""
Quick test script to verify all animations are working
"""

import subprocess
import time

animations = [
    "oscilloscope",
    "signal",
    "starfield",
    "cylon",
    "breath",
    "geometric",
    "glitch",
    "matrix",
    "particles",
    "waves",
    "spiral",
    "spectrum",
    "neural"
]

print("Testing all animations (3 seconds each)...")
print("-" * 40)

for anim in animations:
    print(f"Testing: {anim}")
    try:
        # Run each animation for 3 seconds
        proc = subprocess.Popen(['python3', 'oled_permanent_pro.py', '-a', anim])
        time.sleep(3)
        proc.terminate()
        proc.wait(timeout=1)
        print(f"  ✓ {anim} works!")
    except Exception as e:
        print(f"  ✗ {anim} failed: {e}")
    
print("-" * 40)
print("Test complete!")