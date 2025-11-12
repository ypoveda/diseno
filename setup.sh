#!/bin/bash

echo "============================================================"
echo "🌽 AGRICULTURE DASHBOARD - SETUP SCRIPT"
echo "============================================================"
echo ""

# Check Python version
echo "1️⃣ Checking Python version..."
python3 --version
echo ""

# Install dependencies
echo "2️⃣ Installing dependencies..."
echo "   This may take a few minutes..."
pip3 install -r requirements.txt --quiet
echo "   ✅ Dependencies installed"
echo ""

# Verify installation
echo "3️⃣ Verifying installation..."
python3 << 'EOF'
try:
    import streamlit
    import pandas
    import plotly
    import numpy
    import scipy
    print("   ✅ streamlit")
    print("   ✅ pandas")
    print("   ✅ plotly")
    print("   ✅ numpy")
    print("   ✅ scipy")
except ImportError as e:
    print(f"   ❌ Error: {e}")
    exit(1)
EOF
echo ""

# Test data loading
echo "4️⃣ Testing data file..."
python3 << 'EOF'
import pandas as pd
data = pd.read_csv('agriculture_data.csv')
print(f"   ✅ Data loaded: {len(data)} rows")
print(f"   ✅ Columns: {len(data.columns)}")
EOF
echo ""

# Final message
echo "============================================================"
echo "✅ SETUP COMPLETE!"
echo "============================================================"
echo ""
echo "To run the dashboard:"
echo "   streamlit run app.py"
echo ""
echo "The dashboard will open in your browser at:"
echo "   http://localhost:8501"
echo ""
echo "To stop the dashboard:"
echo "   Press Ctrl+C"
echo ""
echo "============================================================"
