"""
🌽 Agriculture Optimization Dashboard
Response Surface Methodology for Maize Production

Course: Diseño de Experimentos
Authors: Yeison Poveda, Victor Díaz
Date: November 2025
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Agriculture Optimization",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS (USTA Colors)
# ============================================================================

st.markdown("""
    <style>
        /* USTA Brand Colors */
        :root {
            --primary: #0a2f6b;
            --accent: #f9a602;
            --teal: #1c9c9c;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: var(--primary) !important;
        }
        
        /* Metrics */
        [data-testid="stMetricValue"] {
            font-size: 28px;
            color: var(--primary);
        }
        
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f5f7fa 0%, #e8ecf1 100%);
        }
        
        /* Buttons */
        .stButton > button {
            background-color: var(--accent);
            color: var(--primary);
            font-weight: 600;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
        }
        
        .stButton > button:hover {
            background-color: #ffd166;
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data
def load_data():
    """Load the experimental data from CSV"""
    df = pd.read_csv('agriculture_data.csv')
    return df

# Load data
data = load_data()

# ============================================================================
# OPTIMAL SOLUTION (From Paper - Table 6)
# ============================================================================

OPTIMAL = {
    "Irrigation": 1100.0,
    "Nitrogen": 57.2,
    "Density": 10.0,
    "Production": 6324.2,
    "EUN": 54.6,
    "EUA": 5.7,
    "RBC": 2.3,
    "Desirability": 0.74,
    "Cost_N": 0.2,
    "Cost_Water": 3.2,
    "Cost_Total": 916.9
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_3d_surface(data, x_col, y_col, z_col, title):
    """Create a 3D surface plot"""
    
    # Get unique values for x and y
    x_unique = sorted(data[x_col].unique())
    y_unique = sorted(data[y_col].unique())
    
    # Create a pivot table for the surface
    pivot = data.pivot_table(values=z_col, index=y_col, columns=x_col, aggfunc='mean')
    
    # Create the surface plot
    fig = go.Figure(data=[
        go.Surface(
            x=x_unique,
            y=y_unique,
            z=pivot.values,
            colorscale='Viridis',
            name=z_col
        )
    ])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=f"{x_col} (m³/ha)" if x_col == "Irrigation" else f"{x_col} (kg/ha)" if x_col == "Nitrogen" else f"{x_col} (plants/m²)",
            yaxis_title=f"{y_col} (m³/ha)" if y_col == "Irrigation" else f"{y_col} (kg/ha)" if y_col == "Nitrogen" else f"{y_col} (plants/m²)",
            zaxis_title=z_col,
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        height=600,
        font=dict(size=12)
    )
    
    return fig

def create_scatter_3d(data, x_col, y_col, z_col, title):
    """Create a 3D scatter plot of actual data points"""
    
    fig = go.Figure(data=[
        go.Scatter3d(
            x=data[x_col],
            y=data[y_col],
            z=data[z_col],
            mode='markers',
            marker=dict(
                size=6,
                color=data[z_col],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=z_col)
            ),
            text=[f"Run {i}<br>{x_col}: {x}<br>{y_col}: {y}<br>{z_col}: {z:.1f}" 
                  for i, x, y, z in zip(data['Run'], data[x_col], data[y_col], data[z_col])],
            hoverinfo='text'
        )
    ])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=f"{x_col}",
            yaxis_title=f"{y_col}",
            zaxis_title=z_col,
        ),
        height=600
    )
    
    return fig

def calculate_metrics(irrigation, nitrogen, density):
    """Calculate EUN, EUA, RBC for given factors (simplified - using nearest data point)"""
    
    # Find closest data point
    distances = np.sqrt(
        ((data['Irrigation'] - irrigation) ** 2) +
        ((data['Nitrogen'] - nitrogen) ** 2) +
        ((data['Density'] - density) ** 2)
    )
    
    closest_idx = distances.idxmin()
    row = data.iloc[closest_idx]
    
    return {
        "Production": row['Production'],
        "EUN": row['EUN'],
        "EUA": row['EUA'],
        "RBC": row['RBC']
    }

# ============================================================================
# MAIN APP
# ============================================================================

# Title and Description
st.title("🌽 Agriculture Optimization Dashboard")
st.markdown("""
**Response Surface Methodology for Maize Production Optimization**

This dashboard analyzes a Central Composite Design (CCD) experiment with 48 runs 
to optimize maize production by balancing irrigation, nitrogen application, and plant density.

---
""")

# ============================================================================
# SIDEBAR - FACTOR CONTROLS
# ============================================================================

st.sidebar.header("🎛️ Factor Controls")
st.sidebar.markdown("Adjust the factors to explore different scenarios:")

# Factor sliders
irrigation = st.sidebar.slider(
    "💧 Irrigation (m³/ha)",
    min_value=1100,
    max_value=3000,
    value=2050,
    step=50,
    help="Water applied for irrigation"
)

nitrogen = st.sidebar.slider(
    "🌿 Nitrogen (kg/ha)",
    min_value=0,
    max_value=150,
    value=75,
    step=5,
    help="Nitrogen fertilizer applied"
)

density = st.sidebar.slider(
    "🌱 Plant Density (plants/m²)",
    min_value=3.3,
    max_value=10.0,
    value=6.65,
    step=0.1,
    help="Number of maize plants per square meter"
)

st.sidebar.markdown("---")

# Reset button
if st.sidebar.button("🎯 Set to Optimal Values"):
    st.sidebar.success("Sliders set to optimal values!")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("""
**Paper Reference:**  
Yaguas, O. J. (2017). Metodología de superficie de respuesta para la optimización de una producción agrícola.
""")

# ============================================================================
# MAIN CONTENT - TABS
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Overview", 
    "📈 Visualizations", 
    "🎯 Optimal Solution",
    "💰 Economic Analysis"
])

# ----------------------------------------------------------------------------
# TAB 1: DATA OVERVIEW
# ----------------------------------------------------------------------------

with tab1:
    st.header("Experimental Data Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Runs", len(data))
        st.metric("Design Type", "CCD (α=1)")
    
    with col2:
        st.metric("Factors", 3)
        st.metric("Blocks", 3)
    
    with col3:
        st.metric("Responses", 4)
        st.metric("Treatments", 16)
    
    st.markdown("---")
    
    # Display data summary
    st.subheader("Factor Ranges")
    factor_summary = pd.DataFrame({
        'Factor': ['Irrigation', 'Nitrogen', 'Density'],
        'Min': [data['Irrigation'].min(), data['Nitrogen'].min(), data['Density'].min()],
        'Max': [data['Irrigation'].max(), data['Nitrogen'].max(), data['Density'].max()],
        'Unit': ['m³/ha', 'kg/ha', 'plants/m²'],
        'Levels': ['1100, 2050, 3000', '0, 75, 150', '3.3, 6.65, 10.0']
    })
    st.dataframe(factor_summary, use_container_width=True, hide_index=True)
    
    st.subheader("Response Ranges")
    response_summary = pd.DataFrame({
        'Response': ['Production', 'EUN', 'EUA', 'RBC'],
        'Min': [data['Production'].min(), data['EUN'].min(), data['EUA'].min(), data['RBC'].min()],
        'Max': [data['Production'].max(), data['EUN'].max(), data['EUA'].max(), data['RBC'].max()],
        'Mean': [data['Production'].mean(), data['EUN'].mean(), data['EUA'].mean(), data['RBC'].mean()],
        'Unit': ['kg/ha', 'kg/kg', 'kg/m³', '-'],
        'Goal': ['Maximize', 'Maximize', 'Maximize', 'Maximize']
    })
    st.dataframe(response_summary, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Show raw data
    with st.expander("📋 View Complete Dataset (48 runs)"):
        st.dataframe(data, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 2: VISUALIZATIONS
# ----------------------------------------------------------------------------

with tab2:
    st.header("Response Surface Analysis")
    
    # Response selector
    response = st.selectbox(
        "Select Response Variable",
        ["Production", "EUN", "EUA", "RBC"],
        help="Choose which response to visualize"
    )
    
    # Plot type selector
    plot_type = st.radio(
        "Visualization Type",
        ["3D Surface", "3D Scatter (Data Points)"],
        horizontal=True
    )
    
    st.markdown("---")
    
    # Create two plots side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"{response} vs Irrigation × Nitrogen")
        if plot_type == "3D Surface":
            fig1 = create_3d_surface(data, "Irrigation", "Nitrogen", response, 
                                    f"{response} Surface")
        else:
            fig1 = create_scatter_3d(data, "Irrigation", "Nitrogen", response,
                                    f"{response} Data Points")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader(f"{response} vs Nitrogen × Density")
        if plot_type == "3D Surface":
            fig2 = create_3d_surface(data, "Nitrogen", "Density", response,
                                    f"{response} Surface")
        else:
            fig2 = create_scatter_3d(data, "Nitrogen", "Density", response,
                                    f"{response} Data Points")
        st.plotly_chart(fig2, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 3: OPTIMAL SOLUTION
# ----------------------------------------------------------------------------

with tab3:
    st.header("🎯 Optimal Solution from Paper")
    st.markdown("**Source:** Table 6 - Yaguas (2017)")
    
    st.markdown("---")
    
    # Optimal Factors
    st.subheader("Optimal Factor Levels")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("💧 Irrigation", f"{OPTIMAL['Irrigation']:.0f} m³/ha")
    with col2:
        st.metric("🌿 Nitrogen", f"{OPTIMAL['Nitrogen']:.1f} kg/ha")
    with col3:
        st.metric("🌱 Density", f"{OPTIMAL['Density']:.1f} plants/m²")
    
    st.markdown("---")
    
    # Optimal Responses
    st.subheader("Expected Performance")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Production", f"{OPTIMAL['Production']:.1f} kg/ha")
    with col2:
        st.metric("EUN", f"{OPTIMAL['EUN']:.1f} kg/kg")
    with col3:
        st.metric("EUA", f"{OPTIMAL['EUA']:.1f} kg/m³")
    with col4:
        st.metric("RBC", f"{OPTIMAL['RBC']:.1f}")
    
    st.markdown("---")
    
    # Desirability
    st.subheader("Multi-Objective Optimization")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric(
            "Combined Desirability", 
            f"{OPTIMAL['Desirability']:.2f}",
            help="Ranges from 0 to 1, where 1 is ideal"
        )
    
    with col2:
        st.info("""
        **Interpretation:**  
        The combined desirability of 0.74 means this solution achieves 74% of the ideal 
        performance across all four objectives (Production, EUN, EUA, RBC) simultaneously.
        """)

# ----------------------------------------------------------------------------
# TAB 4: ECONOMIC ANALYSIS
# ----------------------------------------------------------------------------

with tab4:
    st.header("💰 Economic Analysis")
    
    st.subheader("Optimal Solution Costs (per hectare)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Nitrogen Cost", f"${OPTIMAL['Cost_N']:.2f}")
    with col2:
        st.metric("Irrigation Cost", f"${OPTIMAL['Cost_Water']:.2f}")
    with col3:
        st.metric("Total Cost", f"${OPTIMAL['Cost_Total']:.2f}")
    
    st.markdown("---")
    
    st.subheader("Cost Breakdown")
    
    # Revenue calculation
    production = OPTIMAL['Production']
    price_per_kg = 0.30
    revenue = production * price_per_kg
    profit = revenue - OPTIMAL['Cost_Total']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Revenue", f"${revenue:.2f}")
        st.metric("Profit", f"${profit:.2f}", delta=f"{(profit/OPTIMAL['Cost_Total']*100):.1f}% ROI")
    
    with col2:
        # Create cost pie chart
        costs_data = {
            'Category': ['Fixed Costs', 'Nitrogen', 'Irrigation'],
            'Cost': [913.44, OPTIMAL['Cost_N'], OPTIMAL['Cost_Water']]
        }
        costs_df = pd.DataFrame(costs_data)
        
        fig = go.Figure(data=[go.Pie(
            labels=costs_df['Category'],
            values=costs_df['Cost'],
            hole=0.3
        )])
        fig.update_layout(title="Cost Distribution", height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.info("""
    **Key Insights:**
    - Benefit-Cost Ratio (RBC) of 2.3 means for every $1 invested, you get $2.30 back
    - Water and nitrogen costs are minimal compared to fixed production costs
    - Optimal solution prioritizes efficiency over maximum production
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #0a2f6b;'>
    <p><strong>Diseño de Experimentos - Universidad Santo Tomás</strong></p>
    <p>Yeison Poveda • Victor Díaz • November 2025</p>
</div>
""", unsafe_allow_html=True)
