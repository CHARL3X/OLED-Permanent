#!/usr/bin/env python3
"""
OLED Permanent Animation Controller - Professional Edition
Enhanced with BlackBox-quality waveform animations and visual standards
"""

import time
import math
import random
import argparse
import signal
import sys
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum
from collections import deque

try:
    from luma.core.interface.serial import i2c
    from luma.oled.device import ssd1306
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Error: Required libraries not installed")
    print("Run: pip install luma.oled pillow")
    sys.exit(1)


class AnimationType(Enum):
    """Available animation types"""
    OSCILLOSCOPE = "oscilloscope"
    SIGNAL_WAVE = "signal"
    STARFIELD = "starfield"
    CYLON = "cylon"
    BREATH = "breath"
    GEOMETRIC = "geometric"
    GLITCH = "glitch"
    MATRIX = "matrix"
    PARTICLES = "particles"
    WAVES = "waves"
    SPIRAL = "spiral"
    SPECTRUM = "spectrum"
    NEURAL = "neural"


@dataclass
class AnimationConfig:
    """Configuration for animations"""
    width: int = 128
    height: int = 64
    i2c_address: int = 0x3C
    i2c_port: int = 1
    fps: int = 20
    duration: Optional[float] = None
    rotation: int = 0


class BaseAnimation:
    """Base class for all animations"""
    
    def __init__(self, config: AnimationConfig):
        self.config = config
        self.frame_count = 0
        self.start_time = time.time()
        self.phase = 0.0
        self.last_update = time.time()
        
    def update(self, dt: float) -> bool:
        """Update animation state"""
        self.frame_count += 1
        self.phase += dt
        self.last_update = time.time()
        
        if self.config.duration and (time.time() - self.start_time) >= self.config.duration:
            return False
        return True
        
    def render(self, draw: ImageDraw) -> None:
        """Render animation frame"""
        pass


class OscilloscopeAnimation(BaseAnimation):
    """Professional oscilloscope visualization with persistence"""
    
    def __init__(self, config: AnimationConfig):
        super().__init__(config)
        self.wave_type = 'sine'
        self.wave_stack = deque(maxlen=8)  # Persistence stack
        self.grid_enabled = True
        self.trigger_level = config.height // 2
        self.time_scale = 1.0
        self.amplitude = 0.8
        self.glitch_timer = 0
        self.glitch_active = False
        
        # Wave components for complex signals
        self.wave_components = [
            {'type': 'sine', 'freq': 1.0, 'amp': 0.8, 'phase': 0},
            {'type': 'sine', 'freq': 2.3, 'amp': 0.2, 'phase': math.pi/4}
        ]
        
        # Grid buffer for efficiency
        self.grid_buffer = self._create_grid()
        
    def _create_grid(self) -> Image:
        """Create oscilloscope grid pattern"""
        img = Image.new('1', (self.config.width, self.config.height), 0)
        draw = ImageDraw.Draw(img)
        
        # Center lines (dotted)
        center_x = self.config.width // 2
        center_y = self.config.height // 2
        
        for x in range(0, self.config.width, 2):
            draw.point((x, center_y), fill=1)
        for y in range(0, self.config.height, 2):
            draw.point((center_x, y), fill=1)
        
        # Grid divisions
        for x in range(0, self.config.width, 16):
            if x % 32 == 0:  # Major divisions
                for y in range(0, self.config.height, 4):
                    draw.point((x, y), fill=1)
        
        for y in range(0, self.config.height, 16):
            if y % 32 == 0:  # Major divisions
                for x in range(0, self.config.width, 4):
                    draw.point((x, y), fill=1)
        
        # Edge markers
        for i in range(0, self.config.width, 32):
            draw.line([(i, self.config.height-3), (i, self.config.height-1)], fill=1)
        
        return img
    
    def _generate_waveform(self) -> List[int]:
        """Generate complex waveform points"""
        points = []
        
        for x in range(self.config.width):
            y_val = 0
            
            # Handle glitch zones
            if self.glitch_active and random.random() < 0.1:
                y_val = random.uniform(-1, 1)
            else:
                # Sum wave components
                for comp in self.wave_components:
                    t = (x / self.config.width) * 2 * math.pi * comp['freq'] * self.time_scale
                    t += comp['phase']
                    
                    if comp['type'] == 'sine':
                        val = math.sin(t)
                    elif comp['type'] == 'square':
                        val = 1 if math.sin(t) > 0 else -1
                    elif comp['type'] == 'triangle':
                        val = 2 * abs(((t / (2 * math.pi)) % 1) - 0.5) - 1
                    elif comp['type'] == 'sawtooth':
                        val = 2 * ((t / (2 * math.pi)) % 1) - 1
                    else:
                        val = 0
                    
                    y_val += val * comp['amp']
            
            # Convert to screen coordinates
            y = int(self.config.height // 2 - y_val * self.config.height * 0.35 * self.amplitude)
            y = max(0, min(self.config.height - 1, y))
            points.append(y)
        
        return points
    
    def update(self, dt: float) -> bool:
        if not super().update(dt):
            return False
        
        # Update wave component phases
        for comp in self.wave_components:
            comp['phase'] += dt * comp['freq'] * 0.5
        
        # Random glitch effects
        if random.random() < 0.01:
            self.glitch_active = True
            self.glitch_timer = random.randint(3, 10)
        
        if self.glitch_timer > 0:
            self.glitch_timer -= 1
        else:
            self.glitch_active = False
        
        # Change wave type occasionally
        if random.random() < 0.005:
            self.wave_components[0]['type'] = random.choice(['sine', 'square', 'triangle', 'sawtooth'])
        
        # Generate and store waveform
        waveform = self._generate_waveform()
        self.wave_stack.append(waveform)
        
        return True
    
    def render(self, draw: ImageDraw) -> None:
        # Draw grid if enabled
        if self.grid_enabled:
            # Paste pre-rendered grid
            for y in range(self.config.height):
                for x in range(self.config.width):
                    if self.grid_buffer.getpixel((x, y)):
                        draw.point((x, y), fill=1)
        
        # Draw waveforms with persistence
        for i, waveform in enumerate(self.wave_stack):
            is_current = (i == len(self.wave_stack) - 1)
            
            if is_current:
                # Draw current waveform as connected line
                for x in range(1, len(waveform)):
                    draw.line([(x-1, waveform[x-1]), (x, waveform[x])], fill=1)
                    
                    # Add vertical lines for square waves
                    if abs(waveform[x] - waveform[x-1]) > 10:
                        draw.line([(x, min(waveform[x], waveform[x-1])),
                                  (x, max(waveform[x], waveform[x-1]))], fill=1)
            else:
                # Draw older waveforms as dots (persistence effect)
                step = len(self.wave_stack) - i
                for x in range(0, len(waveform), step * 2):
                    draw.point((x, waveform[x]), fill=1)
        
        # Draw trigger level indicator
        draw.line([(self.config.width - 5, self.trigger_level),
                  (self.config.width - 1, self.trigger_level)], fill=1)
        
        # Draw time scale indicator
        self._draw_text(draw, f"{self.time_scale:.1f}x", 
                       self.config.width - 20, self.config.height - 8)
        
        # Add glitch artifacts
        if self.glitch_active:
            for _ in range(self.glitch_timer):
                x = random.randint(0, self.config.width - 1)
                y = random.randint(0, self.config.height - 1)
                draw.rectangle([x, y, x+2, y+1], fill=1)
    
    def _draw_text(self, draw, text, x, y):
        """Simple text rendering using points"""
        for i, char in enumerate(text):
            char_x = x + i * 4
            if char == '.':
                draw.point((char_x, y + 3), fill=1)
            elif char == 'x':
                draw.line([(char_x, y), (char_x + 2, y + 4)], fill=1)
                draw.line([(char_x, y + 4), (char_x + 2, y)], fill=1)
            elif char.isdigit():
                # Simple digit representation
                d = int(char)
                if d in [0, 2, 3, 5, 6, 7, 8, 9]:
                    draw.point((char_x, y), fill=1)  # Top
                if d in [0, 4, 5, 6, 8, 9]:
                    draw.point((char_x, y + 2), fill=1)  # Middle
                if d in [0, 2, 3, 5, 6, 8]:
                    draw.point((char_x, y + 4), fill=1)  # Bottom


class SignalWaveAnimation(BaseAnimation):
    """Professional signal visualization with scanning beam"""
    
    def __init__(self, config: AnimationConfig):
        super().__init__(config)
        self.scan_position = 0
        self.scan_direction = 1
        self.scan_speed = 20
        self.wave_center = config.height // 2
        self.wave_amplitude = config.height // 4
        self.wave_frequency = 0.05
        
        # Data points for visualization
        self.data_points = []
        self.signal_history = deque(maxlen=config.width)
        
        # Initialize data points
        for _ in range(8):
            self.data_points.append({
                'x': random.randint(10, config.width - 10),
                'y': random.randint(10, config.height - 10),
                'size': random.choice([1, 2]),
                'blink_rate': random.uniform(0.5, 2.0),
                'phase': random.uniform(0, 2 * math.pi)
            })
        
        # Grid state
        self.show_grid = True
        self.grid_spacing = 16
        
    def update(self, dt: float) -> bool:
        if not super().update(dt):
            return False
        
        # Update scan position
        self.scan_position += self.scan_speed * dt * self.scan_direction
        if self.scan_position >= self.config.width or self.scan_position <= 0:
            self.scan_direction *= -1
            self.show_grid = random.choice([True, False])
        
        # Update signal history
        wave = math.sin(self.phase * 2)
        y = self.wave_center + int(self.wave_amplitude * wave)
        self.signal_history.append(y)
        
        return True
    
    def render(self, draw: ImageDraw) -> None:
        # Draw minimal grid
        if self.show_grid:
            for y in range(self.grid_spacing, self.config.height, self.grid_spacing):
                for x in range(0, self.config.width, self.grid_spacing):
                    if random.random() < 0.6:
                        draw.point((x, y), fill=1)
        
        # Draw signal wave
        points = []
        for x in range(self.config.width):
            wave = math.sin(x * self.wave_frequency + self.phase)
            y = self.wave_center + int(self.wave_amplitude * wave)
            points.append((x, y))
        
        # Draw wave as dashed line
        for i in range(1, len(points)):
            if i % 3 != 0:  # Skip every 3rd segment for dashed effect
                draw.line([points[i-1], points[i]], fill=1)
        
        # Draw scanning beam
        scan_x = int(self.scan_position)
        if 0 <= scan_x < self.config.width:
            draw.line([(scan_x, 5), (scan_x, self.config.height - 5)], fill=1)
            
            # Draw intensity at scan position
            if scan_x < len(self.signal_history):
                y = self.signal_history[scan_x]
                draw.ellipse([scan_x - 2, y - 2, scan_x + 2, y + 2], outline=1)
        
        # Draw data points
        for point in self.data_points:
            phase = self.phase * point['blink_rate'] + point['phase']
            if math.sin(phase) > 0.3:
                x, y = int(point['x']), int(point['y'])
                if point['size'] == 1:
                    draw.point((x, y), fill=1)
                else:
                    draw.ellipse([x-1, y-1, x+1, y+1], outline=1)
        
        # Draw corner indicators
        for i in range(3):
            draw.point((2 + i * 3, 2), fill=1)
            draw.point((self.config.width - 3 - i * 3, 2), fill=1)


class SpectrumAnalyzerAnimation(BaseAnimation):
    """Audio spectrum analyzer visualization"""
    
    def __init__(self, config: AnimationConfig):
        super().__init__(config)
        self.num_bars = 16
        self.bar_width = config.width // self.num_bars - 1
        self.bar_heights = [0] * self.num_bars
        self.target_heights = [0] * self.num_bars
        self.peak_heights = [0] * self.num_bars
        self.peak_decay = [0] * self.num_bars
        
    def update(self, dt: float) -> bool:
        if not super().update(dt):
            return False
        
        # Generate new target heights (simulate audio input)
        for i in range(self.num_bars):
            # Simulate frequency response
            freq_factor = 1.0 - (i / self.num_bars) * 0.5
            base = math.sin(self.phase * (i + 1) * 0.3) * 0.5 + 0.5
            self.target_heights[i] = base * freq_factor * (self.config.height - 10)
        
        # Smooth transition to target heights
        for i in range(self.num_bars):
            diff = self.target_heights[i] - self.bar_heights[i]
            self.bar_heights[i] += diff * 0.3
            
            # Update peaks
            if self.bar_heights[i] > self.peak_heights[i]:
                self.peak_heights[i] = self.bar_heights[i]
                self.peak_decay[i] = 0
            else:
                self.peak_decay[i] += dt * 20
                self.peak_heights[i] = max(0, self.peak_heights[i] - self.peak_decay[i])
        
        return True
    
    def render(self, draw: ImageDraw) -> None:
        # Draw spectrum bars
        for i in range(self.num_bars):
            x = i * (self.bar_width + 1)
            height = int(self.bar_heights[i])
            
            if height > 0:
                # Draw main bar
                draw.rectangle([x, self.config.height - height,
                              x + self.bar_width - 1, self.config.height - 1], fill=1)
                
                # Draw peak indicator
                peak_y = self.config.height - int(self.peak_heights[i])
                if peak_y < self.config.height - 2:
                    draw.line([(x, peak_y), (x + self.bar_width - 1, peak_y)], fill=1)
        
        # Draw frequency labels (simplified)
        draw.line([(0, self.config.height - 1), (self.config.width - 1, self.config.height - 1)], fill=1)


class NeuralNetworkAnimation(BaseAnimation):
    """Neural network visualization"""
    
    def __init__(self, config: AnimationConfig):
        super().__init__(config)
        
        # Neural network structure
        self.layers = [4, 6, 6, 3]  # Neurons per layer
        self.layer_positions = []
        self.connections = []
        self.neuron_states = []
        
        # Calculate positions
        layer_spacing = config.width // (len(self.layers) + 1)
        for i, num_neurons in enumerate(self.layers):
            layer_x = (i + 1) * layer_spacing
            layer_neurons = []
            neuron_spacing = config.height // (num_neurons + 1)
            
            for j in range(num_neurons):
                neuron_y = (j + 1) * neuron_spacing
                layer_neurons.append((layer_x, neuron_y))
            
            self.layer_positions.append(layer_neurons)
            self.neuron_states.append([0] * num_neurons)
        
        # Initialize connections
        for i in range(len(self.layers) - 1):
            layer_connections = []
            for j in range(self.layers[i]):
                neuron_connections = []
                for k in range(self.layers[i + 1]):
                    neuron_connections.append(random.uniform(0.2, 1.0))
                layer_connections.append(neuron_connections)
            self.connections.append(layer_connections)
        
        self.signal_layer = 0
        self.signal_neuron = 0
        
    def update(self, dt: float) -> bool:
        if not super().update(dt):
            return False
        
        # Propagate signal through network
        if random.random() < 0.1:
            # Start new signal
            self.signal_layer = 0
            self.signal_neuron = random.randint(0, self.layers[0] - 1)
            self.neuron_states[0][self.signal_neuron] = 1.0
        
        # Update neuron states
        for i, layer_states in enumerate(self.neuron_states):
            for j, state in enumerate(layer_states):
                # Decay
                self.neuron_states[i][j] = max(0, state - dt * 2)
        
        # Propagate signals
        if random.random() < 0.3 and self.signal_layer < len(self.layers) - 1:
            # Propagate to next layer
            current_layer = self.signal_layer
            current_neuron = self.signal_neuron
            
            if self.neuron_states[current_layer][current_neuron] > 0.5:
                next_layer = current_layer + 1
                # Activate connected neurons
                for k in range(self.layers[next_layer]):
                    weight = self.connections[current_layer][current_neuron][k]
                    if random.random() < weight:
                        self.neuron_states[next_layer][k] = 1.0
                
                self.signal_layer = next_layer
                if self.layers[next_layer] > 0:
                    self.signal_neuron = random.randint(0, self.layers[next_layer] - 1)
        
        return True
    
    def render(self, draw: ImageDraw) -> None:
        # Draw connections
        for i in range(len(self.connections)):
            for j, neuron_connections in enumerate(self.connections[i]):
                x1, y1 = self.layer_positions[i][j]
                
                for k, weight in enumerate(neuron_connections):
                    x2, y2 = self.layer_positions[i + 1][k]
                    
                    # Draw connection if active
                    source_active = self.neuron_states[i][j] > 0.1
                    target_active = self.neuron_states[i + 1][k] > 0.1
                    
                    if source_active or target_active:
                        # Draw with intensity based on activation
                        if source_active and target_active:
                            draw.line([(x1, y1), (x2, y2)], fill=1)
                        elif random.random() < 0.3:  # Sparse for inactive
                            draw.line([(x1, y1), (x2, y2)], fill=1)
        
        # Draw neurons
        for i, layer_neurons in enumerate(self.layer_positions):
            for j, (x, y) in enumerate(layer_neurons):
                activation = self.neuron_states[i][j]
                
                if activation > 0.1:
                    # Active neuron
                    size = 2 if activation > 0.5 else 1
                    draw.ellipse([x - size, y - size, x + size, y + size], fill=1)
                else:
                    # Inactive neuron
                    draw.point((x, y), fill=1)


class EnhancedStarfieldAnimation(BaseAnimation):
    """Enhanced starfield with depth and nebula"""
    
    def __init__(self, config: AnimationConfig):
        super().__init__(config)
        self.stars = []
        self.nebula_phase = 0
        self.warp_speed = False
        self.warp_timer = 0
        
        # Initialize star field
        for _ in range(40):
            self.stars.append({
                'x': random.uniform(0, config.width),
                'y': random.uniform(0, config.height),
                'z': random.uniform(0.1, 1.0),  # Depth
                'speed': random.uniform(0.5, 3),
                'size': random.choice([1, 1, 2]),  # Most are small
                'twinkle': random.random() < 0.3,
                'twinkle_phase': random.uniform(0, 2 * math.pi)
            })
        
        # Nebula clouds
        self.nebula_clouds = []
        for _ in range(3):
            self.nebula_clouds.append({
                'x': random.uniform(0, config.width),
                'y': random.uniform(20, config.height - 20),
                'radius': random.uniform(15, 25),
                'density': random.uniform(0.1, 0.3),
                'drift': random.uniform(-0.5, 0.5)
            })
    
    def update(self, dt: float) -> bool:
        if not super().update(dt):
            return False
        
        self.nebula_phase += dt * 0.3
        
        # Random warp speed events
        if not self.warp_speed and random.random() < 0.002:
            self.warp_speed = True
            self.warp_timer = random.uniform(1, 3)
        
        if self.warp_timer > 0:
            self.warp_timer -= dt
            if self.warp_timer <= 0:
                self.warp_speed = False
        
        # Update stars
        speed_mult = 5 if self.warp_speed else 1
        
        for star in self.stars:
            star['x'] -= star['speed'] * star['z'] * speed_mult
            
            # Add vertical drift for more dynamic movement
            star['y'] += math.sin(self.phase + star['x'] * 0.01) * 0.2
            
            # Update twinkle
            if star['twinkle']:
                star['twinkle_phase'] += dt * 3
            
            # Reset star if it goes off screen
            if star['x'] < 0:
                star['x'] = self.config.width
                star['y'] = random.uniform(0, self.config.height)
                star['z'] = random.uniform(0.1, 1.0)
                star['speed'] = random.uniform(0.5, 3)
            
            # Wrap vertical position
            if star['y'] < 0:
                star['y'] = self.config.height
            elif star['y'] > self.config.height:
                star['y'] = 0
        
        # Update nebula
        for cloud in self.nebula_clouds:
            cloud['x'] += cloud['drift']
            if cloud['x'] < -cloud['radius']:
                cloud['x'] = self.config.width + cloud['radius']
            elif cloud['x'] > self.config.width + cloud['radius']:
                cloud['x'] = -cloud['radius']
        
        return True
    
    def render(self, draw: ImageDraw) -> None:
        # Draw nebula clouds first (background)
        for cloud in self.nebula_clouds:
            # Create nebula effect with random points
            num_points = int(cloud['radius'] * cloud['radius'] * cloud['density'])
            for _ in range(num_points):
                angle = random.uniform(0, 2 * math.pi)
                dist = random.gauss(0, cloud['radius'] / 2)
                if abs(dist) < cloud['radius']:
                    x = cloud['x'] + dist * math.cos(angle)
                    y = cloud['y'] + dist * math.sin(angle)
                    
                    if 0 <= x < self.config.width and 0 <= y < self.config.height:
                        opacity = 1 - (abs(dist) / cloud['radius'])
                        if random.random() < opacity * 0.3:
                            draw.point((int(x), int(y)), fill=1)
        
        # Draw stars
        for star in self.stars:
            # Calculate brightness based on depth and twinkle
            visible = True
            
            if star['twinkle']:
                twinkle_value = math.sin(star['twinkle_phase'])
                visible = twinkle_value > -0.3
            
            if visible:
                x, y = int(star['x']), int(star['y'])
                
                if self.warp_speed:
                    # Draw motion lines during warp
                    line_length = int(10 * star['z'])
                    draw.line([(x, y), (x + line_length, y)], fill=1)
                else:
                    # Normal star rendering
                    if star['size'] == 1:
                        draw.point((x, y), fill=1)
                    else:
                        # Larger stars for foreground
                        brightness = star['z']
                        if brightness > 0.7:
                            draw.ellipse([x-1, y-1, x+1, y+1], outline=1)
                        else:
                            draw.rectangle([x, y, x+1, y+1], fill=1)


class ProfessionalMatrixAnimation(BaseAnimation):
    """Enhanced Matrix rain with better characters"""
    
    def __init__(self, config: AnimationConfig):
        super().__init__(config)
        
        # Column configuration
        self.num_columns = config.width // 6
        self.columns = []
        
        # Character patterns for matrix effect
        self.char_patterns = [
            # Various geometric patterns to simulate characters
            [(0,0), (2,0), (1,1), (0,2), (2,2)],  # H-like
            [(1,0), (0,1), (1,1), (2,1), (1,2)],  # Cross
            [(0,0), (1,0), (2,0), (0,1), (0,2)],  # L
            [(0,0), (1,0), (2,0), (2,1), (2,2)],  # Reverse L
            [(1,0), (1,1), (1,2), (0,1), (2,1)],  # T
            [(0,0), (2,0), (0,2), (2,2)],         # Square corners
            [(1,0), (0,1), (2,1), (1,2)],         # Diamond
            [(0,0), (1,0), (0,1), (1,1)],         # Small square
        ]
        
        # Initialize columns
        for i in range(self.num_columns):
            self.columns.append({
                'x': i * 6 + 1,
                'chars': [],
                'speed': random.uniform(1, 4),
                'spawn_timer': random.uniform(0, 2),
                'intensity_decay': random.uniform(0.7, 0.95)
            })
    
    def update(self, dt: float) -> bool:
        if not super().update(dt):
            return False
        
        for col in self.columns:
            # Update spawn timer
            col['spawn_timer'] -= dt
            
            # Spawn new character
            if col['spawn_timer'] <= 0:
                col['chars'].append({
                    'y': -3,
                    'pattern': random.choice(self.char_patterns),
                    'intensity': 1.0,
                    'changing': random.random() < 0.3  # Some chars change as they fall
                })
                col['spawn_timer'] = random.uniform(0.1, 0.5)
            
            # Update existing characters
            for char in col['chars'][:]:
                char['y'] += col['speed']
                char['intensity'] *= col['intensity_decay']
                
                # Random character changes
                if char['changing'] and random.random() < 0.1:
                    char['pattern'] = random.choice(self.char_patterns)
                
                # Remove off-screen characters
                if char['y'] > self.config.height or char['intensity'] < 0.1:
                    col['chars'].remove(char)
        
        return True
    
    def render(self, draw: ImageDraw) -> None:
        for col in self.columns:
            for i, char in enumerate(col['chars']):
                # Head of trail is brightest
                is_head = (i == len(col['chars']) - 1)
                
                # Draw character pattern
                for px, py in char['pattern']:
                    x = col['x'] + px
                    y = int(char['y']) + py
                    
                    if 0 <= x < self.config.width and 0 <= y < self.config.height:
                        if is_head or char['intensity'] > 0.5:
                            draw.point((x, y), fill=1)
                        elif random.random() < char['intensity']:
                            draw.point((x, y), fill=1)


class OLEDController:
    """Main controller for OLED animations"""
    
    def __init__(self, config: AnimationConfig):
        self.config = config
        self.device = None
        self.current_animation = None
        self.animations = {
            AnimationType.OSCILLOSCOPE: OscilloscopeAnimation,
            AnimationType.SIGNAL_WAVE: SignalWaveAnimation,
            AnimationType.SPECTRUM: SpectrumAnalyzerAnimation,
            AnimationType.NEURAL: NeuralNetworkAnimation,
            AnimationType.STARFIELD: EnhancedStarfieldAnimation,
            AnimationType.MATRIX: ProfessionalMatrixAnimation,
            # Can add more animations here
        }
        self.running = False
        
    def initialize(self):
        """Initialize the OLED display"""
        try:
            serial = i2c(port=self.config.i2c_port, address=self.config.i2c_address)
            self.device = ssd1306(serial, width=self.config.width, height=self.config.height,
                                 rotate=self.config.rotation)
            print(f"Display initialized at 0x{self.config.i2c_address:02X}")
            return True
        except Exception as e:
            print(f"Failed to initialize display: {e}")
            return False
    
    def set_animation(self, animation_type: AnimationType):
        """Set the current animation"""
        if animation_type in self.animations:
            animation_class = self.animations[animation_type]
            self.current_animation = animation_class(self.config)
            print(f"Switched to {animation_type.value} animation")
    
    def run_single(self, animation_type: AnimationType):
        """Run a single animation"""
        self.set_animation(animation_type)
        self.running = True
        
        frame_time = 1.0 / self.config.fps
        last_frame = time.time()
        
        print(f"Running {animation_type.value} animation (Press Ctrl+C to stop)")
        
        while self.running:
            try:
                current_time = time.time()
                dt = current_time - last_frame
                
                if dt >= frame_time:
                    # Update animation
                    if not self.current_animation.update(dt):
                        break
                    
                    # Render frame
                    image = Image.new('1', (self.config.width, self.config.height), 0)
                    draw = ImageDraw.Draw(image)
                    self.current_animation.render(draw)
                    
                    # Display frame
                    self.device.display(image)
                    last_frame = current_time
                else:
                    time.sleep(0.001)
                    
            except KeyboardInterrupt:
                break
    
    def run_cycle(self, animations: List[AnimationType], cycle_time: float = 10):
        """Cycle through multiple animations"""
        self.running = True
        animation_index = 0
        
        print(f"Cycling through {len(animations)} animations every {cycle_time}s")
        print("Press Ctrl+C to stop")
        
        while self.running:
            try:
                # Set current animation
                current_type = animations[animation_index]
                self.set_animation(current_type)
                
                # Run animation for cycle_time seconds
                start_time = time.time()
                frame_time = 1.0 / self.config.fps
                last_frame = time.time()
                
                while time.time() - start_time < cycle_time and self.running:
                    current_time = time.time()
                    dt = current_time - last_frame
                    
                    if dt >= frame_time:
                        # Update animation
                        self.current_animation.update(dt)
                        
                        # Render frame
                        image = Image.new('1', (self.config.width, self.config.height), 0)
                        draw = ImageDraw.Draw(image)
                        self.current_animation.render(draw)
                        
                        # Display frame
                        self.device.display(image)
                        last_frame = current_time
                    else:
                        time.sleep(0.001)
                
                # Move to next animation
                animation_index = (animation_index + 1) % len(animations)
                
            except KeyboardInterrupt:
                break
    
    def stop(self):
        """Stop animations and clear display"""
        self.running = False
        if self.device:
            self.device.clear()
            print("\nDisplay cleared")


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print("\nShutdown signal received")
    sys.exit(0)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='OLED Permanent Pro - BlackBox Quality Animations')
    parser.add_argument('--animation', '-a', type=str,
                       choices=[a.value for a in AnimationType],
                       help='Animation to run')
    parser.add_argument('--cycle', '-c', action='store_true',
                       help='Cycle through all animations')
    parser.add_argument('--cycle-time', '-t', type=float, default=45,
                       help='Time per animation in cycle mode (seconds)')
    parser.add_argument('--fps', '-f', type=int, default=20,
                       help='Frames per second')
    parser.add_argument('--address', type=lambda x: int(x, 0), default=0x3C,
                       help='I2C address (e.g., 0x3C)')
    parser.add_argument('--port', '-p', type=int, default=1,
                       help='I2C port number')
    parser.add_argument('--rotate', '-r', type=int, choices=[0, 1, 2, 3], default=0,
                       help='Display rotation (0, 1, 2, or 3)')
    
    args = parser.parse_args()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create configuration
    config = AnimationConfig(
        i2c_address=args.address,
        i2c_port=args.port,
        fps=args.fps,
        rotation=args.rotate * 90
    )
    
    # Create controller
    controller = OLEDController(config)
    
    # Initialize display
    if not controller.initialize():
        sys.exit(1)
    
    try:
        if args.cycle:
            # Cycle through all animations
            animations = list(AnimationType)
            controller.run_cycle(animations, args.cycle_time)
        elif args.animation:
            # Run specific animation
            animation_type = AnimationType(args.animation)
            controller.run_single(animation_type)
        else:
            # Default: cycle through professional animations
            pro_animations = [
                AnimationType.OSCILLOSCOPE,
                AnimationType.SIGNAL_WAVE,
                AnimationType.SPECTRUM,
                AnimationType.NEURAL,
                AnimationType.STARFIELD,
                AnimationType.MATRIX
            ]
            controller.run_cycle(pro_animations, args.cycle_time)  # Now defaults to 45 seconds
    finally:
        controller.stop()


if __name__ == "__main__":
    main()