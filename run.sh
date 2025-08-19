#!/bin/bash
# OLED Permanent Animation Runner

cd "$(dirname "$0")"

# Check if running as service
if [ "$1" = "service" ]; then
    echo "Starting OLED Permanent as service..."
    exec python3 oled_permanent_pro.py --cycle --cycle-time 45
fi

# Interactive mode
echo "OLED Permanent Animation Controller"
echo "==================================="
echo ""
echo "Quick Start Options:"
echo "1) Cycle best animations (default)"
echo "2) Spectrum Analyzer (loop forever)"
echo "3) Oscilloscope"
echo "4) Neural Network"
echo "5) Signal Wave"
echo "6) Starfield"
echo "7) Matrix rain"
echo "8) Lava Lamp (uses orange/blue zones)"
echo "9) Interactive selection (with live preview)"
echo "10) List all animations"
echo "11) Custom options"
echo ""
read -p "Select option [1-11]: " choice

case $choice in
    1)
        python3 oled_permanent_pro.py
        ;;
    2)
        echo "Running Spectrum Analyzer on loop..."
        python3 oled_permanent_pro.py -a spectrum --loop
        ;;
    3)
        python3 oled_permanent_pro.py -a oscilloscope
        ;;
    4)
        python3 oled_permanent_pro.py -a neural
        ;;
    5)
        python3 oled_permanent_pro.py -a signal
        ;;
    6)
        python3 oled_permanent_pro.py -a starfield
        ;;
    7)
        python3 oled_permanent_pro.py -a matrix
        ;;
    8)
        python3 oled_permanent_pro.py -a horizon
        ;;
    9)
        python3 oled_permanent_pro.py --interactive
        ;;
    10)
        python3 oled_permanent_pro.py --list
        ;;
    11)
        echo ""
        echo "Enter custom command (e.g., --fps 30 --rotate 2):"
        read -p "> python3 oled_permanent_pro.py " custom_args
        python3 oled_permanent_pro.py $custom_args
        ;;
    *)
        echo "Invalid option, running default..."
        python3 oled_permanent_pro.py
        ;;
esac