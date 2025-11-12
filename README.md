# 🌽 Agriculture Optimization Dashboard

Response Surface Methodology for Maize Production Optimization

**Course:** Diseño de Experimentos  
**Authors:** Yeison Poveda, Victor Díaz  
**Date:** November 2025

---

## 📋 Project Overview

This Streamlit dashboard analyzes a Central Composite Design (CCD) experiment with 48 runs to optimize maize production by balancing:
- 💧 Irrigation (1100-3000 m³/ha)
- 🌿 Nitrogen application (0-150 kg/ha)
- 🌱 Plant density (3.3-10 plants/m²)

**Paper:** Yaguas, O. J. (2017). Metodología de superficie de respuesta para la optimización de una producción agrícola. *Revista Ingeniería Industrial*, 16(1), 205-222.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`

### 3. Stop the Dashboard

Press `Ctrl + C` in the terminal

---

## 📁 Project Structure

```
agriculture-dashboard/
│
├── app.py                      # Main Streamlit application
├── agriculture_data.csv        # Experimental data (48 runs)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── CHECKLIST.md               # Feature implementation checklist
├── REFERENCE.md               # Formulas and constants
└── STATUS.md                  # Project status
```

---

## 🎨 Features

### ✅ Implemented

- **Data Overview Tab**
  - Summary statistics
  - Factor and response ranges
  - Complete dataset view

- **Visualizations Tab**
  - 3D surface plots
  - 3D scatter plots (data points)
  - Multiple response variables
  - Interactive rotation/zoom

- **Optimal Solution Tab**
  - Optimal factor levels from paper
  - Expected performance metrics
  - Desirability function result

- **Economic Analysis Tab**
  - Cost breakdown
  - Revenue and profit calculations
  - ROI analysis
  - Cost distribution chart

- **Interactive Controls**
  - Factor sliders in sidebar
  - Response variable selector
  - Plot type selector

- **Styling**
  - USTA brand colors
  - Professional layout
  - Responsive design

---

## 📊 Dashboard Tabs

### 1️⃣ Data Overview
- Experimental design summary
- Factor levels and ranges
- Response variable statistics
- Raw data table

### 2️⃣ Visualizations
- 3D surface plots showing response surfaces
- 3D scatter plots showing actual data points
- Interactive plot controls

### 3️⃣ Optimal Solution
- Optimal factor levels: Irrigation=1100, N=57.2, Density=10
- Expected responses: Production=6324.2 kg/ha, EUN=54.6, EUA=5.7, RBC=2.3
- Desirability = 0.74

### 4️⃣ Economic Analysis
- Cost breakdown per hectare
- Revenue calculation ($0.30/kg maize)
- Benefit-Cost Ratio analysis

---

## 🧮 Key Formulas

### Nitrogen Use Efficiency (EUN)
```
EUN = Production / (90 + 0.46 × Nitrogen)
```

### Water Use Efficiency (EUA)
```
EUA = Production / Irrigation
```

### Benefit-Cost Ratio (RBC)
```
RBC = (Production × $0.30) / Total_Cost
Where Total_Cost = $913.44 + (N × $0.0035) + (I × $0.0029)
```

---

## 🎯 Optimal Solution (From Paper)

| Factor/Response | Value | Unit |
|----------------|-------|------|
| **Irrigation** | 1100 | m³/ha |
| **Nitrogen** | 57.2 | kg/ha |
| **Density** | 10.0 | plants/m² |
| **Production** | 6324.2 | kg/ha |
| **EUN** | 54.6 | kg/kg |
| **EUA** | 5.7 | kg/m³ |
| **RBC** | 2.3 | - |
| **Desirability** | 0.74 | - |

---

## 🛠️ Troubleshooting

### Dashboard won't run
```bash
# Check Python version (need 3.8+)
python --version

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Plots don't show
- Make sure `agriculture_data.csv` is in the same directory as `app.py`
- Clear Streamlit cache: `streamlit cache clear`

### Import errors
```bash
# Make sure you're in the correct directory
cd /path/to/agriculture-dashboard

# Verify files exist
ls -la
```

---

## 📈 Future Enhancements

Potential features to add (see CHECKLIST.md):

- [ ] Contour overlay plots (like Figure 6 in paper)
- [ ] Polynomial model predictions
- [ ] Comparison: user selection vs optimal
- [ ] Sensitivity analysis
- [ ] Export results to PDF/Excel
- [ ] Interactive desirability function
- [ ] More economic scenarios

---

## 📚 Documentation

- **CHECKLIST.md** - Complete feature list and implementation guide
- **REFERENCE.md** - All formulas, constants, and optimal values
- **STATUS.md** - Current project status

---

## 🎓 For Presentation

### Key Points to Explain:

1. **Methodology**
   - Central Composite Design (CCD) with α=1
   - 48 runs = 16 treatments × 3 blocks
   - Response Surface Methodology

2. **Optimization**
   - Multi-objective: maximize all 4 responses
   - Desirability function combines objectives
   - Trade-offs between production and efficiency

3. **Results**
   - Optimal solution reduces water by 63% (vs max)
   - Reduces nitrogen by 62% (vs max)
   - Still achieves 85% of maximum production
   - High economic efficiency (RBC = 2.3)

4. **Practical Impact**
   - Lower input costs
   - Environmental sustainability
   - Maintained profitability

---

## 📞 Support

If you encounter issues:

1. Check Python version: `python --version` (need 3.8+)
2. Verify all files are present: `ls -la`
3. Check Streamlit docs: https://docs.streamlit.io
4. Check Plotly docs: https://plotly.com/python/

---

## 📖 Citation

```
Yaguas, O. J. (2017). Metodología de superficie de respuesta para la 
optimización de una producción agrícola. Revista Ingeniería Industrial, 
16(1), 205-222. https://doi.org/10.22320/S07179103/2017.13
```

---

## 📄 License

Academic project for Diseño de Experimentos course.

---

**Last Updated:** November 12, 2025
