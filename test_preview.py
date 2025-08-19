#!/usr/bin/env python3
"""
Test script for animation preview functionality
"""

import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from oled_permanent_pro import OLEDController, AnimationConfig, AnimationType

def test_preview():
    """Test the preview functionality"""
    print("Testing Animation Preview System")
    print("=" * 40)
    
    # Create configuration
    config = AnimationConfig(fps=15)
    
    # Create controller
    controller = OLEDController(config)
    
    # Initialize display
    if not controller.initialize():
        print("Failed to initialize display")
        return False
    
    print("Display initialized successfully")
    
    # Get available animations
    animations = [a for a in AnimationType if a in controller.animations]
    
    if not animations:
        print("No animations available")
        return False
    
    print(f"Found {len(animations)} animations")
    print("\nTesting preview for each animation (3 seconds each):")
    print("-" * 40)
    
    for i, anim in enumerate(animations[:5]):  # Test first 5 animations
        print(f"{i+1}. Previewing: {anim.value}...", end="", flush=True)
        
        # Start preview
        controller.start_preview(anim)
        
        # Let it run for 3 seconds
        time.sleep(3)
        
        print(" ✓")
    
    print("-" * 40)
    print("Stopping preview...")
    controller.stop_preview()
    
    print("Preview test completed successfully!")
    return True

if __name__ == "__main__":
    try:
        success = test_preview()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during test: {e}")
        sys.exit(1)