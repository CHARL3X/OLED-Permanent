# OLED Permanent Animation System

Advanced animation controller for SSD1306 OLED displays with hardware color zones (orange top ~16 rows, blue bottom ~48 rows).

## Features

- **14+ Unique Animations** optimized for tiny 128x64 display
- **Live Preview**: Interactive menu with real-time animation preview while selecting
- **Color Zone Aware**: Animations creatively utilize hardware's orange/blue zones
- **Modular Architecture**: Clean BaseAnimation class for easy extension
- **Smooth Physics**: Momentum-based transitions (diff * 0.3) for fluid movement

### Zone-Aware Animations
- **Horizon (Lava Lamp)**: Hot/cold fluid dynamics at color boundary
- **Thermal Columns**: Vertical spectrum with collision dynamics  
- **Geometric**: Complex wave interference with opposing scroll directions

### Classic Animations
- **Spectrum Analyzer**: Audio-reactive vertical bars with peak indicators
- **Neural Network**: Calming ASMR neural activity visualization
- **Oscilloscope**: Professional waveform display
- **Matrix**: Digital rain effect
- **Starfield**: Depth-based stars with warp speed events
- **Signal Wave**: Clean sine wave visualization
- **Spiral**: Rotating dot patterns
- **Glitch**: Digital corruption aesthetic
- **Particles**: Physics-based particle system
- **Waves**: Ocean wave simulation
- **Cylon**: Classic scanning pattern
- **Breath**: Pulsing brightness effect

## Installation

```bash
# Install dependencies
pip install luma.oled pillow

# Make script executable
chmod +x oled_permanent.py
```

## Usage

### Run Default Animation Cycle
```bash
./oled_permanent.py
```

### Run Specific Animation
```bash
./oled_permanent.py --animation starfield
./oled_permanent.py -a cylon
./oled_permanent.py -a breath
```

### Cycle Through All Animations
```bash
./oled_permanent.py --cycle --cycle-time 15
```

### Custom Configuration
```bash
# Different I2C address
./oled_permanent.py --address 0x3D

# Rotate display 180 degrees
./oled_permanent.py --rotate 2

# Higher FPS
./oled_permanent.py --fps 30
```

## Command Line Options

- `--animation, -a` : Specific animation to run
- `--cycle, -c` : Cycle through all animations
- `--cycle-time, -t` : Seconds per animation in cycle mode (default: 10)
- `--fps, -f` : Frames per second (default: 20)
- `--address` : I2C address (default: 0x3C)
- `--port, -p` : I2C port number (default: 1)
- `--rotate, -r` : Display rotation 0-3 (default: 0)

## Run as Service

Create `/etc/systemd/system/oled-permanent.service`:

```ini
[Unit]
Description=OLED Permanent Animation Controller
After=multi-user.target

[Service]
Type=simple
User=morph
WorkingDirectory=/home/morph/01_Code/OLED-Permanent
ExecStart=/usr/bin/python3 /home/morph/01_Code/OLED-Permanent/oled_permanent.py --cycle
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable oled-permanent
sudo systemctl start oled-permanent
```

## Hardware Setup

- **Display**: SSD1306 128x64 OLED
- **Connection**: I2C (SDA/SCL)
- **Default Address**: 0x3C
- **Power**: 3.3V

## Troubleshooting

Check I2C connection:
```bash
i2cdetect -y 1
```

Test with specific address:
```bash
./oled_permanent.py --address 0x3D -a cylon
```

## Animation Descriptions

1. **Starfield**: Immersive space travel with parallax stars and nebula clouds
2. **Cylon**: Classic scanning effect from Battlestar Galactica/Knight Rider
3. **Breath**: Meditative breathing rectangles with floating particles
4. **Geometric**: Mathematical beauty with Lissajous curves and spirals
5. **Oscilloscope**: Professional scope display with various waveforms
6. **Matrix**: Digital rain effect with falling characters
7. **Particles**: Realistic particle fountain with gravity
8. **Waves**: Multiple sine waves creating interference patterns
9. **Spiral**: Rotating spiral patterns with multiple arms

## Performance

- Optimized for Raspberry Pi
- Low CPU usage (~5-10%)
- Smooth 20 FPS default
- Minimal memory footprint