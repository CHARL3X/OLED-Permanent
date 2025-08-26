#!/usr/bin/env python3
"""
Simple OLED Test Script with Multiplexer Support
"""

import time
import math
import sys
from i2c_helper import auto_detect_oled

print("OLED Display Test with Multiplexer Support")
print("=" * 40)

# Auto-detect OLED (will try direct connection first, then multiplexer)
device, connection_type = auto_detect_oled(port=1, oled_address=0x3C)

if not device:
    print("\nFailed to detect OLED display!")
    print("Please check:")
    print("  1. OLED is properly connected")
    print("  2. I2C is enabled (raspi-config)")
    print("  3. Correct wiring to multiplexer channel 0 (if using multiplexer)")
    sys.exit(1)

print(f"\nConnection type: {connection_type}")
print("Starting animation tests...\n")

from PIL import Image, ImageDraw

# Test 1: Simple moving dot
print("Test 1: Moving dot")
for x in range(128):
    img = Image.new('1', (128, 64), 0)
    draw = ImageDraw.Draw(img)
    draw.ellipse((x-2, 30, x+2, 34), fill=1)
    device.display(img)
    time.sleep(0.01)

# Test 2: Sine wave
print("Test 2: Sine wave")
for frame in range(100):
    img = Image.new('1', (128, 64), 0)
    draw = ImageDraw.Draw(img)
    points = []
    for x in range(128):
        y = 32 + int(20 * math.sin((x + frame * 2) * 0.1))
        points.append((x, y))
    if len(points) > 1:
        draw.line(points, fill=1, width=1)
    device.display(img)
    time.sleep(0.02)

# Test 3: Display connection info
print("Test 3: Connection info display")
img = Image.new('1', (128, 64), 0)
draw = ImageDraw.Draw(img)

# Draw connection type
text_lines = [
    "OLED Test Complete!",
    "",
    f"Connection: {connection_type}",
    f"{'Direct I2C' if connection_type == 'direct' else 'Via MUX Ch0'}",
    "",
    "All tests passed!"
]

y_offset = 5
for line in text_lines:
    draw.text((5, y_offset), line, fill=1)
    y_offset += 10

device.display(img)
time.sleep(3)

print("\nAll tests complete!")
device.clear()
print("Display cleared.")