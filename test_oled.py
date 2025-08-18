#!/usr/bin/env python3
"""
Simple OLED Test Script
"""

import time
import math
from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306
from PIL import Image, ImageDraw

# Initialize display
serial = i2c(port=1, address=0x3C)
device = ssd1306(serial)

print("Testing OLED display...")

# Test 1: Simple moving dot
for x in range(128):
    img = Image.new('1', (128, 64), 0)
    draw = ImageDraw.Draw(img)
    draw.ellipse((x-2, 30, x+2, 34), fill=1)
    device.display(img)
    time.sleep(0.01)

# Test 2: Sine wave
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

print("Test complete!")
device.clear()