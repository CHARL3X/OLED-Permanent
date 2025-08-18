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
echo "2) Starfield animation"
echo "3) Cylon scanner"
echo "4) Breathing animation"
echo "5) Geometric patterns"
echo "6) Oscilloscope"
echo "7) Matrix rain"
echo "8) Cycle all animations"
echo "9) Custom options"
echo ""
read -p "Select option [1-9]: " choice

case $choice in
    1)
        python3 oled_permanent_pro.py
        ;;
    2)
        python3 oled_permanent.py -a starfield
        ;;
    3)
        python3 oled_permanent.py -a cylon
        ;;
    4)
        python3 oled_permanent.py -a breath
        ;;
    5)
        python3 oled_permanent.py -a geometric
        ;;
    6)
        python3 oled_permanent.py -a oscilloscope
        ;;
    7)
        python3 oled_permanent.py -a matrix
        ;;
    8)
        python3 oled_permanent_pro.py --cycle --cycle-time 45
        ;;
    9)
        echo ""
        echo "Enter custom command (e.g., --fps 30 --rotate 2):"
        read -p "> python3 oled_permanent.py " custom_args
        python3 oled_permanent.py $custom_args
        ;;
    *)
        echo "Invalid option, running default..."
        python3 oled_permanent_pro.py
        ;;
esac