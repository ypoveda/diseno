"""
🌽 Panel de Optimización Agrícola
Metodología de Superficie de Respuesta para Producción de Maíz

Curso: Diseño de Experimentos
Autores: Yeison Poveda, Victor Díaz
Fecha: Noviembre 2025
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.stats import f as f_dist
import io
import base64

# ============================================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================================

st.set_page_config(
    page_title="Optimización Agrícola",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force light theme
st.markdown("""
    <style>
        /* Force light theme - override dark mode */
        :root, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            background-color: #ffffff !important;
        }
        
        /* Main content area */
        .main .block-container {
            background-color: #ffffff !important;
            color: #1e1e1e !important;
        }
        
        /* All text should be dark on light background */
        body, p, span, div, label, .stMarkdown {
            color: #1e1e1e !important;
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# CSS PERSONALIZADO (Colores USTA)
# ============================================================================

st.markdown("""
    <style>
        /* ============================================ */
        /* LIGHT THEME - EXCELLENT CONTRAST           */
        /* ============================================ */
        
        /* Force white backgrounds everywhere */
        :root {
            --primary: #0a2f6b;
            --accent: #f59e0b;
            --success: #059669;
            --info: #0284c7;
        }
        
        /* Main app background */
        .stApp {
            background-color: #ffffff !important;
        }
        
        /* Main content area */
        [data-testid="stAppViewContainer"] {
            background-color: #ffffff !important;
        }
        
        /* Ensure all text is dark */
        body, p, span, div, label, li, td, th {
            color: #1e1e1e !important;
        }
        
        /* Headers - USTA Navy */
        h1, h2, h3, h4, h5, h6 {
            color: var(--primary) !important;
            font-weight: 700 !important;
        }
        
        /* Subheaders */
        .stMarkdown h2, .stMarkdown h3 {
            color: var(--primary) !important;
            margin-top: 1.5rem !important;
        }
        
        /* ============================================ */
        /* SIDEBAR - Light gray background             */
        /* ============================================ */
        
        section[data-testid="stSidebar"] {
            background-color: #f8fafc !important;
            border-right: 1px solid #e2e8f0;
        }
        
        section[data-testid="stSidebar"] * {
            color: #1e1e1e !important;
        }
        
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: var(--primary) !important;
        }
        
        /* ============================================ */
        /* METRICS - Bold and clear                    */
        /* ============================================ */
        
        [data-testid="stMetric"] {
            background-color: #f8fafc;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
        }
        
        [data-testid="stMetricLabel"] {
            color: #475569 !important;
            font-weight: 600 !important;
            font-size: 0.875rem !important;
        }
        
        [data-testid="stMetricValue"] {
            color: var(--primary) !important;
            font-weight: 700 !important;
            font-size: 1.75rem !important;
        }
        
        [data-testid="stMetricDelta"] {
            font-weight: 600 !important;
        }
        
        /* ============================================ */
        /* BUTTONS - Navy with white text              */
        /* ============================================ */
        
        .stButton > button {
            background-color: var(--primary) !important;
            color: #ffffff !important;
            font-weight: 600 !important;
            border: 2px solid var(--primary) !important;
            border-radius: 8px !important;
            padding: 0.6rem 1.5rem !important;
            transition: all 0.3s ease !important;
        }
        
        .stButton > button:hover {
            background-color: #ffffff !important;
            color: var(--primary) !important;
            border: 2px solid var(--primary) !important;
        }
        
        /* ============================================ */
        /* HIGHLIGHT BOX - Amber/Yellow                */
        /* ============================================ */
        
        .selected-values {
            background-color: #fef3c7 !important;
            color: #78350f !important;
            padding: 1.25rem !important;
            border-radius: 10px !important;
            border-left: 5px solid #f59e0b !important;
            font-weight: 500 !important;
            line-height: 1.7 !important;
            margin: 1rem 0 !important;
        }
        
        .selected-values b, .selected-values strong {
            color: #78350f !important;
            font-weight: 700 !important;
        }
        
        /* ============================================ */
        /* TABS - Clear hierarchy                      */
        /* ============================================ */
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px !important;
            background-color: #f8fafc !important;
            padding: 0.5rem !important;
            border-radius: 8px !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            font-weight: 600 !important;
            color: #64748b !important;
            background-color: transparent !important;
            border-radius: 6px !important;
            padding: 0.5rem 1rem !important;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #e2e8f0 !important;
            color: var(--primary) !important;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: var(--primary) !important;
            color: #ffffff !important;
        }
        
        /* ============================================ */
        /* DATAFRAMES - Clean tables                   */
        /* ============================================ */
        
        [data-testid="stDataFrame"] {
            border: 1px solid #e2e8f0 !important;
            border-radius: 8px !important;
            overflow: hidden !important;
        }
        
        /* Table headers */
        [data-testid="stDataFrame"] thead tr th {
            background-color: #f1f5f9 !important;
            color: #1e293b !important;
            font-weight: 600 !important;
            padding: 0.75rem !important;
        }
        
        /* Table cells */
        [data-testid="stDataFrame"] tbody tr td {
            color: #1e1e1e !important;
            padding: 0.75rem !important;
        }
        
        /* Alternating rows */
        [data-testid="stDataFrame"] tbody tr:nth-child(even) {
            background-color: #f8fafc !important;
        }
        
        /* ============================================ */
        /* EXPANDERS - Collapsible sections            */
        /* ============================================ */
        
        .streamlit-expanderHeader {
            background-color: #f8fafc !important;
            color: var(--primary) !important;
            font-weight: 600 !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 8px !important;
        }
        
        .streamlit-expanderHeader:hover {
            background-color: #f1f5f9 !important;
        }
        
        /* ============================================ */
        /* SLIDERS - Better visibility                 */
        /* ============================================ */
        
        .stSlider {
            padding: 1rem 0 !important;
        }
        
        .stSlider > label {
            color: #1e293b !important;
            font-weight: 600 !important;
            margin-bottom: 0.5rem !important;
        }
        
        /* Slider track */
        .stSlider [data-baseweb="slider"] {
            background-color: #e2e8f0 !important;
        }
        
        /* Slider thumb */
        .stSlider [role="slider"] {
            background-color: var(--primary) !important;
        }
        
        /* Slider labels */
        .stSlider [data-testid="stTickBar"] {
            color: #64748b !important;
        }
        
        /* ============================================ */
        /* SELECT BOXES & RADIO - Clean inputs         */
        /* ============================================ */
        
        .stSelectbox > label,
        .stRadio > label {
            color: #1e293b !important;
            font-weight: 600 !important;
        }
        
        /* Select box container - force white background */
        .stSelectbox [data-baseweb="select"] {
            background-color: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
        }
        
        /* Select box dropdown list */
        .stSelectbox [data-baseweb="select"] > div {
            background-color: #ffffff !important;
            color: #1e1e1e !important;
        }
        
        /* Select box selected value */
        .stSelectbox [data-baseweb="select"] input {
            color: #1e1e1e !important;
        }
        
        /* Select box placeholder */
        .stSelectbox [data-baseweb="select"] [data-baseweb="input"] {
            color: #1e1e1e !important;
            background-color: #ffffff !important;
        }
        
        /* Dropdown menu itself */
        [data-baseweb="popover"] {
            background-color: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        }
        
        /* Dropdown menu items */
        [role="listbox"] {
            background-color: #ffffff !important;
        }
        
        [role="option"] {
            background-color: #ffffff !important;
            color: #1e1e1e !important;
            padding: 0.5rem 1rem !important;
        }
        
        [role="option"]:hover {
            background-color: #f1f5f9 !important;
            color: var(--primary) !important;
        }
        
        [aria-selected="true"][role="option"] {
            background-color: #e0e7ff !important;
            color: var(--primary) !important;
        }
        
        /* Radio buttons */
        .stRadio [role="radiogroup"] {
            background-color: transparent !important;
        }
        
        .stRadio [role="radio"] {
            background-color: #ffffff !important;
            border: 2px solid #cbd5e1 !important;
        }
        
        .stRadio [role="radio"][aria-checked="true"] {
            background-color: var(--primary) !important;
            border-color: var(--primary) !important;
        }
        
        .stRadio label {
            color: #1e1e1e !important;
        }
        
        /* Radio button text */
        .stRadio > div > label > div:last-child {
            color: #1e1e1e !important;
        }
        
        /* ============================================ */
        /* DIVIDERS - Subtle separators                */
        /* ============================================ */
        
        hr {
            border-top: 2px solid #e2e8f0 !important;
            margin: 2rem 0 !important;
        }
        
        /* ============================================ */
        /* PLOTLY CHARTS - White background            */
        /* ============================================ */
        
        .js-plotly-plot {
            background-color: #ffffff !important;
        }
        
        .plotly .gtitle {
            color: #1e1e1e !important;
        }
        
        /* ============================================ */
        /* ADDITIONAL OVERRIDES - Nuclear option       */
        /* ============================================ */
        
        /* Force all backgrounds to be light */
        div, section, article, aside {
            background-color: transparent !important;
        }
        
        /* Except these which should be white */
        .main, .block-container, [data-testid="stVerticalBlock"] {
            background-color: #ffffff !important;
        }
        
        /* All text inputs */
        input[type="text"],
        input[type="number"],
        textarea {
            background-color: #ffffff !important;
            color: #1e1e1e !important;
            border: 1px solid #cbd5e1 !important;
        }
        
        /* Number inputs */
        .stNumberInput input {
            background-color: #ffffff !important;
            color: #1e1e1e !important;
        }
        
        /* Text inputs */
        .stTextInput input {
            background-color: #ffffff !important;
            color: #1e1e1e !important;
        }
        
        /* Text areas */
        .stTextArea textarea {
            background-color: #ffffff !important;
            color: #1e1e1e !important;
        }
        
        /* Dropdown overlays */
        [data-baseweb="menu"],
        [data-baseweb="popover"],
        [role="listbox"],
        [role="menu"] {
            background-color: #ffffff !important;
            color: #1e1e1e !important;
        }
        
        /* All list items in dropdowns */
        li[role="option"],
        li[role="menuitem"] {
            background-color: #ffffff !important;
            color: #1e1e1e !important;
        }
        
        li[role="option"]:hover,
        li[role="menuitem"]:hover {
            background-color: #f1f5f9 !important;
        }
        
        /* Code blocks if any */
        code, pre {
            background-color: #f8fafc !important;
            color: #1e1e1e !important;
        }
        
        /* Tooltips */
        [data-baseweb="tooltip"] {
            background-color: #1e293b !important;
            color: #ffffff !important;
        }
        
        /* Ensure spinner/loading is visible */
        .stSpinner > div {
            border-color: var(--primary) !important;
        }
        
        /* Model equations styling */
        .model-equation {
            background-color: #f8fafc !important;
            border: 1px solid #e2e8f0 !important;
            padding: 1rem !important;
            border-radius: 8px !important;
            font-family: monospace !important;
            font-size: 0.9rem !important;
            overflow-x: auto !important;
            margin: 1rem 0 !important;
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# MODELOS RSM - COEFICIENTES DEL PAPER
# ============================================================================

# Coeficientes de los modelos cuadráticos (Tabla 5 del paper)
RSM_COEFFICIENTS = {
    'Production': {
        'b0': 5898.08,
        'b1': 0.514, 'b2': 8.95, 'b3': 91.02,
        'b11': -0.000128, 'b22': -0.0461, 'b33': -2.38,
        'b12': 0.000463, 'b13': -0.00531, 'b23': 0.00896,
        'R2': 0.9814, 'R2_adj': 0.9742
    },
    'EUN': {
        'b0': 52.6,
        'b1': -0.00166, 'b2': -0.223, 'b3': 2.25,
        'b11': 0.394e-6, 'b22': -0.00220, 'b33': -0.0796,
        'b12': 0.154e-5, 'b13': -0.129e-3, 'b23': 0.00463,
        'R2': 0.9832, 'R2_adj': 0.9766
    },
    'EUA': {
        'b0': 4.29,
        'b1': -0.00113, 'b2': 0.00768, 'b3': 0.179,
        'b11': 0.266e-6, 'b22': -0.396e-4, 'b33': -0.00622,
        'b12': 0.423e-6, 'b13': -0.103e-4, 'b23': 0.827e-5,
        'R2': 0.9819, 'R2_adj': 0.9749
    },
    'RBC': {
        'b0': 0.938,
        'b1': 0.296e-3, 'b2': 0.00269, 'b3': 0.0274,
        'b11': -0.740e-7, 'b22': -0.138e-4, 'b33': -0.718e-3,
        'b12': 0.139e-6, 'b13': -0.160e-5, 'b23': 0.271e-5,
        'R2': 0.9813, 'R2_adj': 0.9741
    }
}

# ============================================================================
# CARGA DE DATOS
# ============================================================================

@st.cache_data
def load_data():
    """Cargar los datos experimentales desde CSV"""
    df = pd.read_csv('agriculture_data.csv')
    return df

# Cargar datos
data = load_data()

# ============================================================================
# SOLUCIÓN ÓPTIMA (Del Paper - Tabla 6)
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
# FUNCIONES AUXILIARES MEJORADAS
# ============================================================================

@st.cache_data
def calculate_rsm_response(X1, X2, X3, response_type):
    """
    Calcular la respuesta usando el modelo RSM cuadrático completo
    X1: Irrigación, X2: Nitrógeno, X3: Densidad
    """
    coef = RSM_COEFFICIENTS[response_type]
    
    # Modelo cuadrático completo
    Y = (coef['b0'] + 
         coef['b1']*X1 + coef['b2']*X2 + coef['b3']*X3 +
         coef['b11']*X1**2 + coef['b22']*X2**2 + coef['b33']*X3**2 +
         coef['b12']*X1*X2 + coef['b13']*X1*X3 + coef['b23']*X2*X3)
    
    return Y

def calculate_metrics(irrigation, nitrogen, density):
    """Calcular métricas usando los modelos RSM"""
    # Usar modelos RSM en lugar de buscar punto más cercano
    production = calculate_rsm_response(irrigation, nitrogen, density, 'Production')
    eun = calculate_rsm_response(irrigation, nitrogen, density, 'EUN')
    eua = calculate_rsm_response(irrigation, nitrogen, density, 'EUA')
    rbc = calculate_rsm_response(irrigation, nitrogen, density, 'RBC')
    
    # Calcular costos
    cost_n = nitrogen * 0.0035
    cost_water = irrigation * 0.0029
    cost_total = 913.44 + cost_n + cost_water
    
    # Calcular ingresos
    revenue = production * 0.30
    
    return {
        "Production": production,
        "EUN": eun,
        "EUA": eua,
        "RBC": rbc,
        "Cost_N": cost_n,
        "Cost_Water": cost_water,
        "Cost_Total": cost_total,
        "Revenue": revenue,
        "Profit": revenue - cost_total
    }

@st.cache_data
def create_surface_data(x_range, y_range, _z_func, n_points=50):
    """Crear datos de superficie optimizados"""
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(y_range[0], y_range[1], n_points)
    X, Y = np.meshgrid(x, y)
    Z = _z_func(X, Y)
    return x, y, Z

def create_3d_surface(data, x_col, y_col, z_col, title, selected_point=None):
    """Crear un gráfico de superficie 3D con modelo RSM"""
    
    # Definir rangos
    x_range = [data[x_col].min(), data[x_col].max()]
    y_range = [data[y_col].min(), data[y_col].max()]
    
    # Función para calcular Z basada en los modelos RSM
    if z_col in ['Production', 'EUN', 'EUA', 'RBC']:
        # Determinar qué variables son X1, X2, X3
        vars_map = {'Irrigation': 0, 'Nitrogen': 1, 'Density': 2}
        x_idx = vars_map[x_col]
        y_idx = vars_map[y_col]
        
        # Valor fijo para la tercera variable
        fixed_vals = [1100, 57, 10]  # Valores óptimos por defecto
        if selected_point:
            fixed_vals = [selected_point['Irrigation'], 
                          selected_point['Nitrogen'], 
                          selected_point['Density']]
        
        def z_func(X, Y):
            args = [None, None, None]
            args[x_idx] = X
            args[y_idx] = Y
            # Usar valor fijo para la tercera dimensión
            for i in range(3):
                if args[i] is None:
                    args[i] = fixed_vals[i]
            return calculate_rsm_response(args[0], args[1], args[2], z_col)
        
        x, y, z = create_surface_data(x_range, y_range, z_func)
    else:
        # Para otras variables, usar pivote como antes
        x_unique = sorted(data[x_col].unique())
        y_unique = sorted(data[y_col].unique())
        pivot = data.pivot_table(values=z_col, index=y_col, columns=x_col, aggfunc='mean')
        x, y, z = x_unique, y_unique, pivot.values
    
    # Crear el gráfico de superficie
    fig = go.Figure()
    
    # Agregar superficie
    fig.add_trace(go.Surface(
        x=x,
        y=y,
        z=z,
        colorscale='Viridis',
        name=z_col,
        opacity=0.9,
        showscale=True,
        colorbar=dict(title=z_col)
    ))
    
    # Agregar punto seleccionado si existe
    if selected_point:
        # Calcular Z para el punto seleccionado
        if z_col in ['Production', 'EUN', 'EUA', 'RBC']:
            z_point = selected_point[z_col]
        else:
            z_point = selected_point[z_col]
            
        fig.add_trace(go.Scatter3d(
            x=[selected_point[x_col]],
            y=[selected_point[y_col]],
            z=[z_point],
            mode='markers',
            marker=dict(
                size=12,
                color='red',
                symbol='diamond',
                line=dict(color='white', width=2)
            ),
            name='Selección Actual',
            hovertemplate=f'<b>Tu Selección</b><br>{x_col}: %{{x}}<br>{y_col}: %{{y}}<br>{z_col}: %{{z:.1f}}<extra></extra>'
        ))
    
    # Etiquetas de ejes
    x_label = f"{x_col} (m³/ha)" if x_col == "Irrigation" else f"{x_col} (kg/ha)" if x_col == "Nitrogen" else f"{x_col} (plantas/m²)"
    y_label = f"{y_col} (m³/ha)" if y_col == "Irrigation" else f"{y_col} (kg/ha)" if y_col == "Nitrogen" else f"{y_col} (plantas/m²)"
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title=z_col,
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        height=600,
        font=dict(size=12),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def create_contour_plot(data, x_col, y_col, z_col, title, selected_point=None):
    """Crear gráfico de contorno 2D"""
    
    # Definir rangos
    x_range = [data[x_col].min(), data[x_col].max()]
    y_range = [data[y_col].min(), data[y_col].max()]
    
    # Función para calcular Z basada en los modelos RSM
    if z_col in ['Production', 'EUN', 'EUA', 'RBC']:
        vars_map = {'Irrigation': 0, 'Nitrogen': 1, 'Density': 2}
        x_idx = vars_map[x_col]
        y_idx = vars_map[y_col]
        
        fixed_vals = [1100, 57, 10]
        if selected_point:
            fixed_vals = [selected_point['Irrigation'], 
                          selected_point['Nitrogen'], 
                          selected_point['Density']]
        
        def z_func(X, Y):
            args = [None, None, None]
            args[x_idx] = X
            args[y_idx] = Y
            for i in range(3):
                if args[i] is None:
                    args[i] = fixed_vals[i]
            return calculate_rsm_response(args[0], args[1], args[2], z_col)
        
        x, y, z = create_surface_data(x_range, y_range, z_func)
    else:
        x_unique = sorted(data[x_col].unique())
        y_unique = sorted(data[y_col].unique())
        pivot = data.pivot_table(values=z_col, index=y_col, columns=x_col, aggfunc='mean')
        x, y, z = x_unique, y_unique, pivot.values
    
    fig = go.Figure()
    
    # Agregar contorno
    fig.add_trace(go.Contour(
        x=x,
        y=y,
        z=z,
        colorscale='Viridis',
        contours=dict(
            showlabels=True,
            labelfont=dict(size=12, color='white')
        ),
        colorbar=dict(title=z_col)
    ))
    
    # Agregar punto seleccionado
    if selected_point:
        fig.add_trace(go.Scatter(
            x=[selected_point[x_col]],
            y=[selected_point[y_col]],
            mode='markers',
            marker=dict(
                size=15,
                color='red',
                symbol='diamond',
                line=dict(color='white', width=2)
            ),
            name='Selección Actual',
            hovertemplate=f'<b>Tu Selección</b><br>{x_col}: %{{x}}<br>{y_col}: %{{y}}<br>{z_col}: {selected_point[z_col]:.1f}<extra></extra>'
        ))
    
    # Etiquetas
    x_label = f"{x_col} (m³/ha)" if x_col == "Irrigation" else f"{x_col} (kg/ha)" if x_col == "Nitrogen" else f"{x_col} (plantas/m²)"
    y_label = f"{y_col} (m³/ha)" if y_col == "Irrigation" else f"{y_col} (kg/ha)" if y_col == "Nitrogen" else f"{y_col} (plantas/m²)"
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=600,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    fig.update_xaxes(showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridcolor='lightgray')
    
        
    return fig

def create_sensitivity_analysis(base_values, response='Production'):
    """Análisis de sensibilidad de un factor a la vez"""
    fig = go.Figure()
    
    factors = ['Irrigation', 'Nitrogen', 'Density']
    ranges = {
        'Irrigation': np.linspace(1100, 3000, 50),
        'Nitrogen': np.linspace(0, 150, 50),
        'Density': np.linspace(3.3, 10, 50)
    }
    
    colors = ['blue', 'green', 'orange']
    
    for i, factor in enumerate(factors):
        y_values = []
        for value in ranges[factor]:
            # Copiar valores base
            test_values = base_values.copy()
            test_values[factor] = value
            
            # Calcular respuesta
            result = calculate_rsm_response(
                test_values['Irrigation'],
                test_values['Nitrogen'],
                test_values['Density'],
                response
            )
            y_values.append(result)
        
        fig.add_trace(go.Scatter(
            x=ranges[factor],
            y=y_values,
            mode='lines',
            name=factor,
            line=dict(color=colors[i], width=3)
        ))
    
    # Agregar punto base
    base_result = calculate_rsm_response(
        base_values['Irrigation'],
        base_values['Nitrogen'],
        base_values['Density'],
        response
    )
    
    fig.add_trace(go.Scatter(
        x=[base_values['Irrigation'], base_values['Nitrogen'], base_values['Density']],
        y=[base_result, base_result, base_result],
        mode='markers',
        marker=dict(size=12, color='red', symbol='star'),
        name='Punto Base',
        showlegend=True
    ))
    
    fig.update_layout(
        title=f'Análisis de Sensibilidad - {response}',
        xaxis_title='Valor del Factor',
        yaxis_title=response,
        height=500,
        hovermode='x',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    fig.update_xaxes(showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridcolor='lightgray')
    
    return fig

def generate_report(current_values, current_metrics, optimal_values):
    """Generar reporte en formato texto para descargar"""
    report = f"""REPORTE DE OPTIMIZACIÓN AGRÍCOLA
Metodología de Superficie de Respuesta
Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

=====================================
1. CONFIGURACIÓN SELECCIONADA
=====================================
Irrigación: {current_values['Irrigation']:.1f} m³/ha
Nitrógeno: {current_values['Nitrogen']:.1f} kg/ha
Densidad: {current_values['Density']:.1f} plantas/m²

=====================================
2. PREDICCIONES DEL MODELO
=====================================
Producción: {current_metrics['Production']:.1f} kg/ha
EUN (Eficiencia Uso Nitrógeno): {current_metrics['EUN']:.2f} kg/kg
EUA (Eficiencia Uso Agua): {current_metrics['EUA']:.2f} kg/m³
RBC (Relación Beneficio-Costo): {current_metrics['RBC']:.2f}

=====================================
3. ANÁLISIS ECONÓMICO
=====================================
Costo Nitrógeno: ${current_metrics['Cost_N']:.2f}
Costo Irrigación: ${current_metrics['Cost_Water']:.2f}
Costo Total: ${current_metrics['Cost_Total']:.2f}
Ingresos Esperados: ${current_metrics['Revenue']:.2f}
Ganancia Esperada: ${current_metrics['Profit']:.2f}
ROI: {(current_metrics['Profit']/current_metrics['Cost_Total']*100):.1f}%

=====================================
4. COMPARACIÓN CON ÓPTIMO
=====================================
Diferencia Producción: {((current_metrics['Production'] - optimal_values['Production'])/optimal_values['Production']*100):+.1f}%
Diferencia EUN: {((current_metrics['EUN'] - optimal_values['EUN'])/optimal_values['EUN']*100):+.1f}%
Diferencia EUA: {((current_metrics['EUA'] - optimal_values['EUA'])/optimal_values['EUA']*100):+.1f}%
Diferencia RBC: {((current_metrics['RBC'] - optimal_values['RBC'])/optimal_values['RBC']*100):+.1f}%

=====================================
5. MODELOS RSM UTILIZADOS
=====================================
Los modelos cuadráticos completos tienen R² > 0.98
indicando excelente ajuste a los datos experimentales.

Modelo Producción: R² = {RSM_COEFFICIENTS['Production']['R2']:.4f}
Modelo EUN: R² = {RSM_COEFFICIENTS['EUN']['R2']:.4f}
Modelo EUA: R² = {RSM_COEFFICIENTS['EUA']['R2']:.4f}
Modelo RBC: R² = {RSM_COEFFICIENTS['RBC']['R2']:.4f}

=====================================
6. RECOMENDACIONES
=====================================
"""
    
    # Agregar recomendaciones basadas en la comparación
    if abs(current_metrics['Production'] - optimal_values['Production']) / optimal_values['Production'] > 0.1:
        report += "• Considere ajustar los factores hacia los valores óptimos para mejorar la producción.\n"
    
    if current_metrics['RBC'] < optimal_values['RBC'] * 0.9:
        report += "• La configuración actual tiene menor rentabilidad que la óptima.\n"
    
    if current_values['Nitrogen'] > optimal_values['Nitrogen'] * 1.2:
        report += "• El nivel de nitrógeno es alto, considere reducirlo para mejorar eficiencia.\n"
    
    if current_values['Irrigation'] > optimal_values['Irrigation'] * 1.2:
        report += "• El nivel de irrigación es alto, considere optimizar el uso de agua.\n"
    
    report += """
=====================================
Referencia: Yaguas, O. J. (2017). Metodología de 
superficie de respuesta para la optimización de 
una producción agrícola.
=====================================
"""
    
    return report

# ============================================================================
# APLICACIÓN PRINCIPAL
# ============================================================================

# Título y Descripción
st.title("🌽 Panel de Optimización Agrícola")
st.markdown("""
**Metodología de Superficie de Respuesta para Optimización de Producción de Maíz**

Este panel analiza un experimento de Diseño Compuesto Central (DCC) con 48 corridas 
para optimizar la producción de maíz equilibrando riego, aplicación de nitrógeno y densidad de plantas.

---
""")

# ============================================================================
# BARRA LATERAL - CONTROLES DE FACTORES
# ============================================================================

st.sidebar.header("🎛️ Controles de Factores")
st.sidebar.markdown("**Ajusta los factores para explorar diferentes escenarios:**")

# Controles deslizantes de factores
irrigation = st.sidebar.slider(
    "💧 Riego (m³/ha)",
    min_value=1100,
    max_value=3000,
    value=1100,  # Iniciar con valor óptimo
    step=50,
    help="Agua aplicada para riego"
)

nitrogen = st.sidebar.slider(
    "🌿 Nitrógeno (kg/ha)",
    min_value=0,
    max_value=150,
    value=57,  # Iniciar cerca del valor óptimo
    step=5,
    help="Fertilizante nitrogenado aplicado"
)

density = st.sidebar.slider(
    "🌱 Densidad de Plantas (plantas/m²)",
    min_value=3.3,
    max_value=10.0,
    value=10.0,  # Iniciar con valor óptimo
    step=0.1,
    help="Número de plantas de maíz por metro cuadrado"
)

st.sidebar.markdown("---")

# Calcular métricas para la selección actual
current_metrics = calculate_metrics(irrigation, nitrogen, density)

# Mostrar predicciones para valores seleccionados
st.sidebar.subheader("📊 Predicción para Tu Selección")
st.sidebar.metric("Producción", f"{current_metrics['Production']:.0f} kg/ha")
st.sidebar.metric("EUN", f"{current_metrics['EUN']:.1f} kg/kg")
st.sidebar.metric("EUA", f"{current_metrics['EUA']:.1f} kg/m³")
st.sidebar.metric("RBC", f"{current_metrics['RBC']:.2f}")

st.sidebar.markdown("---")

# Comparación con óptimo
st.sidebar.subheader("🎯 vs Solución Óptima")
prod_diff = ((current_metrics['Production'] - OPTIMAL['Production']) / OPTIMAL['Production'] * 100)
st.sidebar.metric(
    "Diferencia en Producción", 
    f"{prod_diff:+.1f}%",
    delta=f"{current_metrics['Production'] - OPTIMAL['Production']:.0f} kg/ha"
)

st.sidebar.markdown("---")

# Botón de reinicio
if st.sidebar.button("🎯 Establecer a Valores Óptimos"):
    st.rerun()

# Botón de descarga de reporte
st.sidebar.markdown("---")
st.sidebar.subheader("📄 Exportar Resultados")

current_values = {
    'Irrigation': irrigation,
    'Nitrogen': nitrogen,
    'Density': density
}

report_text = generate_report(current_values, current_metrics, OPTIMAL)

st.sidebar.download_button(
    label="📥 Descargar Reporte",
    data=report_text,
    file_name=f"reporte_optimizacion_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt",
    mime="text/plain",
    help="Descargar reporte completo con resultados y análisis"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
    <div style='background-color: #e7f3ff; color: #004085; padding: 1rem; border-radius: 8px; border-left: 4px solid #0c5ba0;'>
        <strong>📚 Referencia del Paper:</strong><br>
        Yaguas, O. J. (2017). Metodología de superficie de respuesta para la optimización de una producción agrícola.
    </div>
""", unsafe_allow_html=True)

# ============================================================================
# CONTENIDO PRINCIPAL - PESTAÑAS
# ============================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Resumen de Datos", 
    "📈 Visualizaciones", 
    "🎯 Solución Óptima",
    "💰 Análisis Económico",
    "🔬 Modelos RSM",
    "📐 Análisis de Sensibilidad"
])

# Crear punto seleccionado para las visualizaciones
selected_point = {
    'Irrigation': irrigation,
    'Nitrogen': nitrogen,
    'Density': density,
    'Production': current_metrics['Production'],
    'EUN': current_metrics['EUN'],
    'EUA': current_metrics['EUA'],
    'RBC': current_metrics['RBC']
}

# ----------------------------------------------------------------------------
# PESTAÑA 1: RESUMEN DE DATOS
# ----------------------------------------------------------------------------

with tab1:
    st.header("Resumen de Datos Experimentales")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total de Corridas", len(data))
        st.metric("Tipo de Diseño", "DCC (α=1)")
    
    with col2:
        st.metric("Factores", 3)
        st.metric("Bloques", 3)
    
    with col3:
        st.metric("Respuestas", 4)
        st.metric("Tratamientos", 16)
    
    st.markdown("---")
    
    # Mostrar resumen de datos
    st.subheader("Rangos de Factores")
    factor_summary = pd.DataFrame({
        'Factor': ['Riego', 'Nitrógeno', 'Densidad'],
        'Mínimo': [data['Irrigation'].min(), data['Nitrogen'].min(), data['Density'].min()],
        'Máximo': [data['Irrigation'].max(), data['Nitrogen'].max(), data['Density'].max()],
        'Unidad': ['m³/ha', 'kg/ha', 'plantas/m²'],
        'Niveles': ['1100, 2050, 3000', '0, 75, 150', '3.3, 6.65, 10.0']
    })
    st.dataframe(factor_summary, width='stretch', hide_index=True)
    
    st.subheader("Rangos de Respuestas")
    response_summary = pd.DataFrame({
        'Respuesta': ['Producción', 'EUN', 'EUA', 'RBC'],
        'Mínimo': [data['Production'].min(), data['EUN'].min(), data['EUA'].min(), data['RBC'].min()],
        'Máximo': [data['Production'].max(), data['EUN'].max(), data['EUA'].max(), data['RBC'].max()],
        'Media': [data['Production'].mean(), data['EUN'].mean(), data['EUA'].mean(), data['RBC'].mean()],
        'Unidad': ['kg/ha', 'kg/kg', 'kg/m³', '-'],
        'Objetivo': ['Maximizar', 'Maximizar', 'Maximizar', 'Maximizar']
    })
    st.dataframe(response_summary, width='stretch', hide_index=True)
    
    st.markdown("---")
    
    # Mostrar datos sin procesar
    with st.expander("📋 Ver Conjunto de Datos Completo (48 corridas)"):
        st.dataframe(data, width='stretch', height=400)

# ----------------------------------------------------------------------------
# PESTAÑA 2: VISUALIZACIONES MEJORADAS
# ----------------------------------------------------------------------------

with tab2:
    st.header("Análisis de Superficie de Respuesta")
    
    # Mostrar valores seleccionados actuales
    st.markdown(f"""
    <div class="selected-values">
        <b>🎯 Tus Valores Seleccionados:</b> 
        Riego = {irrigation} m³/ha | 
        Nitrógeno = {nitrogen} kg/ha | 
        Densidad = {density} plantas/m²
        <br><b>Predicción del Modelo RSM:</b> Producción = {current_metrics['Production']:.0f} kg/ha
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Selector de respuesta
    response = st.selectbox(
        "Seleccionar Variable de Respuesta",
        ["Production", "EUN", "EUA", "RBC"],
        format_func=lambda x: {
            "Production": "Producción (kg/ha)",
            "EUN": "EUN - Eficiencia Uso de Nitrógeno (kg/kg)",
            "EUA": "EUA - Eficiencia Uso de Agua (kg/m³)",
            "RBC": "RBC - Relación Beneficio-Costo"
        }[x],
        help="Elige qué respuesta visualizar"
    )
    
    # Selector de tipo de gráfico
    plot_type = st.radio(
        "Tipo de Visualización",
        ["Superficie 3D", "Contorno 2D", "Dispersión 3D (Puntos de Datos)"],
        horizontal=True,
        help="Los gráficos de contorno 2D son más fáciles de interpretar"
    )
    
    st.markdown("---")
    
    # Crear dos gráficos lado a lado
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"{response} vs Riego × Nitrógeno")
        if plot_type == "Superficie 3D":
            fig1 = create_3d_surface(data, "Irrigation", "Nitrogen", response, 
                                    f"Superficie de {response}", selected_point)
        elif plot_type == "Contorno 2D":
            fig1 = create_contour_plot(data, "Irrigation", "Nitrogen", response,
                                      f"Contorno de {response}", selected_point)
        else:
            from . import create_scatter_3d  # Función original
            fig1 = create_scatter_3d(data, "Irrigation", "Nitrogen", response,
                                    f"Puntos de Datos de {response}", selected_point)
        st.plotly_chart(fig1, width='stretch')
    
    with col2:
        st.subheader(f"{response} vs Nitrógeno × Densidad")
        if plot_type == "Superficie 3D":
            fig2 = create_3d_surface(data, "Nitrogen", "Density", response,
                                    f"Superficie de {response}", selected_point)
        elif plot_type == "Contorno 2D":
            fig2 = create_contour_plot(data, "Nitrogen", "Density", response,
                                      f"Contorno de {response}", selected_point)
        else:
            from . import create_scatter_3d
            fig2 = create_scatter_3d(data, "Nitrogen", "Density", response,
                                    f"Puntos de Datos de {response}", selected_point)
        st.plotly_chart(fig2, width='stretch')
    
    st.markdown("""
        <div style='background-color: #d4edda; color: #155724; padding: 1rem; border-radius: 8px; border-left: 4px solid #28a745; margin-top: 1rem;'>
            <strong>💡 Sugerencia:</strong> Los gráficos de contorno 2D son más fáciles de interpretar que las superficies 3D.
            El punto rojo 💎 muestra tu selección actual. Las líneas en los contornos muestran niveles de respuesta constante.
        </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# PESTAÑA 3: SOLUCIÓN ÓPTIMA
# ----------------------------------------------------------------------------

with tab3:
    st.header("🎯 Solución Óptima del Paper")
    st.markdown("**Fuente:** Tabla 6 - Yaguas (2017)")
    
    st.markdown("---")
    
    # Factores óptimos
    st.subheader("Niveles Óptimos de Factores")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("💧 Riego", f"{OPTIMAL['Irrigation']:.0f} m³/ha")
    with col2:
        st.metric("🌿 Nitrógeno", f"{OPTIMAL['Nitrogen']:.1f} kg/ha")
    with col3:
        st.metric("🌱 Densidad", f"{OPTIMAL['Density']:.1f} plantas/m²")
    
    st.markdown("---")
    
    # Respuestas óptimas
    st.subheader("Rendimiento Esperado")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Producción", f"{OPTIMAL['Production']:.1f} kg/ha")
    with col2:
        st.metric("EUN", f"{OPTIMAL['EUN']:.1f} kg/kg")
    with col3:
        st.metric("EUA", f"{OPTIMAL['EUA']:.1f} kg/m³")
    with col4:
        st.metric("RBC", f"{OPTIMAL['RBC']:.1f}")
    
    st.markdown("---")
    
    # Deseabilidad
    st.subheader("Optimización Multi-Objetivo")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric(
            "Deseabilidad Combinada", 
            f"{OPTIMAL['Desirability']:.2f}",
            help="Varía de 0 a 1, donde 1 es ideal"
        )
    
    with col2:
        st.info("""
        **Interpretación:**  
        La deseabilidad combinada de 0.74 significa que esta solución logra el 74% del 
        rendimiento ideal en los cuatro objetivos (Producción, EUN, EUA, RBC) simultáneamente.
        """)
    
    st.markdown("""
        <div style='background-color: #d1ecf1; color: #0c5460; padding: 1rem; border-radius: 8px; border-left: 4px solid #17a2b8; margin-top: 1rem;'>
            <strong>💡 ¿Qué significa esto?</strong><br>
            Un valor de 0.74 es excelente. Significa que encontramos un balance óptimo donde todas las métricas 
            (producción, eficiencia de nitrógeno, eficiencia de agua, y beneficio-costo) están cerca de sus 
            valores ideales al mismo tiempo.
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Comparación
    st.subheader("📊 Tu Selección vs Óptimo")
    
    comparison_data = pd.DataFrame({
        'Métrica': ['Riego (m³/ha)', 'Nitrógeno (kg/ha)', 'Densidad (plantas/m²)', 
                   'Producción (kg/ha)', 'EUN (kg/kg)', 'EUA (kg/m³)', 'RBC'],
        'Tu Selección': [irrigation, nitrogen, density, 
                        current_metrics['Production'], current_metrics['EUN'], 
                        current_metrics['EUA'], current_metrics['RBC']],
        'Óptimo': [OPTIMAL['Irrigation'], OPTIMAL['Nitrogen'], OPTIMAL['Density'],
                  OPTIMAL['Production'], OPTIMAL['EUN'], OPTIMAL['EUA'], OPTIMAL['RBC']],
        'Diferencia %': [
            (irrigation - OPTIMAL['Irrigation']) / OPTIMAL['Irrigation'] * 100,
            (nitrogen - OPTIMAL['Nitrogen']) / OPTIMAL['Nitrogen'] * 100,
            (density - OPTIMAL['Density']) / OPTIMAL['Density'] * 100,
            (current_metrics['Production'] - OPTIMAL['Production']) / OPTIMAL['Production'] * 100,
            (current_metrics['EUN'] - OPTIMAL['EUN']) / OPTIMAL['EUN'] * 100,
            (current_metrics['EUA'] - OPTIMAL['EUA']) / OPTIMAL['EUA'] * 100,
            (current_metrics['RBC'] - OPTIMAL['RBC']) / OPTIMAL['RBC'] * 100,
        ]
    })
    
    # Formatear la tabla
    st.dataframe(
        comparison_data.style.format({
            'Tu Selección': '{:.1f}',
            'Óptimo': '{:.1f}',
            'Diferencia %': '{:+.1f}%'
        }),
        width='stretch',
        hide_index=True
    )

# ----------------------------------------------------------------------------
# PESTAÑA 4: ANÁLISIS ECONÓMICO
# ----------------------------------------------------------------------------

with tab4:
    st.header("💰 Análisis Económico")
    
    # Mostrar para selección actual
    st.subheader(f"Costos para Tu Selección Actual")
    st.markdown(f"**Configuración:** Riego = {irrigation} m³/ha, Nitrógeno = {nitrogen} kg/ha, Densidad = {density} plantas/m²")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Costo de Nitrógeno", f"${current_metrics['Cost_N']:.2f}")
    with col2:
        st.metric("Costo de Riego", f"${current_metrics['Cost_Water']:.2f}")
    with col3:
        st.metric("Costo Total", f"${current_metrics['Cost_Total']:.2f}")
    with col4:
        st.metric("Ganancia", f"${current_metrics['Profit']:.2f}")
    
    st.markdown("---")
    
    # Comparación de rentabilidad
    st.subheader("📊 Comparación de Rentabilidad")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Tu Selección**")
        st.metric("Ingresos", f"${current_metrics['Revenue']:.2f}")
        st.metric("ROI", f"{(current_metrics['Profit']/current_metrics['Cost_Total']*100):.1f}%")
        st.metric("RBC", f"{current_metrics['RBC']:.2f}")
        
    with col2:
        st.markdown("**Solución Óptima**")
        optimal_revenue = OPTIMAL['Production'] * 0.30
        optimal_profit = optimal_revenue - OPTIMAL['Cost_Total']
        st.metric("Ingresos", f"${optimal_revenue:.2f}")
        st.metric("ROI", f"{(optimal_profit/OPTIMAL['Cost_Total']*100):.1f}%")
        st.metric("RBC", f"{OPTIMAL['RBC']:.2f}")
    
    st.markdown("---")
    
    # Gráfico de costos
    st.subheader("Distribución de Costos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Tu Selección**")
        costs_data_current = {
            'Categoría': ['Costos Fijos', 'Nitrógeno', 'Riego'],
            'Costo': [913.44, current_metrics['Cost_N'], current_metrics['Cost_Water']]
        }
        costs_df_current = pd.DataFrame(costs_data_current)
        
        fig1 = go.Figure(data=[go.Pie(
            labels=costs_df_current['Categoría'],
            values=costs_df_current['Costo'],
            hole=0.3
        )])
        fig1.update_layout(title="Tu Distribución de Costos", height=300)
        st.plotly_chart(fig1, width='stretch')
    
    with col2:
        st.markdown("**Solución Óptima**")
        costs_data_optimal = {
            'Categoría': ['Costos Fijos', 'Nitrógeno', 'Riego'],
            'Costo': [913.44, OPTIMAL['Cost_N'], OPTIMAL['Cost_Water']]
        }
        costs_df_optimal = pd.DataFrame(costs_data_optimal)
        
        fig2 = go.Figure(data=[go.Pie(
            labels=costs_df_optimal['Categoría'],
            values=costs_df_optimal['Costo'],
            hole=0.3
        )])
        fig2.update_layout(title="Distribución de Costos Óptima", height=300)
        st.plotly_chart(fig2, width='stretch')
    
    st.markdown("---")
    
    st.markdown("""
        <div style='background-color: #d4edda; color: #155724; padding: 1.25rem; border-radius: 8px; border-left: 4px solid #28a745; line-height: 1.7;'>
            <strong>💡 Ideas Clave:</strong><br>
            • La Relación Beneficio-Costo (RBC) de 2.3 significa que por cada $1 invertido, se obtienen $2.30 de retorno<br>
            • Los costos de agua y nitrógeno son mínimos comparados con los costos de producción fijos<br>
            • La solución óptima prioriza la eficiencia sobre la producción máxima<br>
            • Precio del maíz asumido: $0.30/kg
        </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# PESTAÑA 5: MODELOS RSM
# ----------------------------------------------------------------------------

with tab5:
    st.header("🔬 Modelos de Superficie de Respuesta")
    
    st.markdown("""
    Los modelos RSM son ecuaciones cuadráticas completas que predicen las respuestas 
    basándose en los factores de entrada. Estos modelos fueron ajustados usando los 48 puntos experimentales.
    """)
    
    st.markdown("---")
    
    # Selector de modelo
    model_response = st.selectbox(
        "Seleccionar Modelo",
        ["Production", "EUN", "EUA", "RBC"],
        format_func=lambda x: {
            "Production": "Modelo de Producción",
            "EUN": "Modelo de Eficiencia de Nitrógeno (EUN)",
            "EUA": "Modelo de Eficiencia de Agua (EUA)",
            "RBC": "Modelo de Relación Beneficio-Costo (RBC)"
        }[x]
    )
    
    # Mostrar ecuación del modelo
    st.subheader(f"Ecuación del Modelo: {model_response}")
    
    coef = RSM_COEFFICIENTS[model_response]
    
    equation = f"""
    Y = {coef['b0']:.2f} + 
        {coef['b1']:.6f}·X₁ + {coef['b2']:.4f}·X₂ + {coef['b3']:.2f}·X₃ +
        {coef['b11']:.6e}·X₁² + {coef['b22']:.6f}·X₂² + {coef['b33']:.4f}·X₃² +
        {coef['b12']:.6e}·X₁X₂ + {coef['b13']:.6e}·X₁X₃ + {coef['b23']:.6f}·X₂X₃
    """
    
    st.markdown(f"""
    <div class="model-equation">
        {equation}
        <br><br>
        Donde: X₁ = Irrigación (m³/ha), X₂ = Nitrógeno (kg/ha), X₃ = Densidad (plantas/m²)
    </div>
    """, unsafe_allow_html=True)
    
    # Mostrar estadísticas del modelo
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("R²", f"{coef['R2']:.4f}", 
                 help="Coeficiente de determinación - qué tan bien el modelo explica la variabilidad")
    with col2:
        st.metric("R² Ajustado", f"{coef['R2_adj']:.4f}",
                 help="R² ajustado por el número de predictores")
    
    st.markdown("---")
    
    # Tabla ANOVA simplificada
    st.subheader("Análisis de Varianza (ANOVA)")
    
    # Valores simulados basados en R² alto
    df_model = 9  # Grados de libertad del modelo
    df_error = 38  # Grados de libertad del error
    df_total = 47  # Grados de libertad total
    
    # Calcular valores F basados en R²
    f_value = (coef['R2'] / df_model) / ((1 - coef['R2']) / df_error)
    p_value = 1 - f_dist.cdf(f_value, df_model, df_error)
    
    anova_data = pd.DataFrame({
        'Fuente': ['Modelo', 'Error', 'Total'],
        'GL': [df_model, df_error, df_total],
        'F': [f_value, '-', '-'],
        'Valor p': [f'{p_value:.2e}' if p_value > 0 else '< 0.0001', '-', '-'],
        'Significancia': ['***', '-', '-']
    })
    
    st.dataframe(anova_data, width='stretch', hide_index=True)
    
    st.info("""
    **Interpretación:** 
    - R² > 0.98 indica que el modelo explica más del 98% de la variabilidad en los datos
    - El valor p < 0.0001 confirma que el modelo es estadísticamente significativo
    - Los modelos cuadráticos capturan bien la curvatura de las superficies de respuesta
    """)

# ----------------------------------------------------------------------------
# PESTAÑA 6: ANÁLISIS DE SENSIBILIDAD
# ----------------------------------------------------------------------------

with tab6:
    st.header("📐 Análisis de Sensibilidad")
    
    st.markdown("""
    El análisis de sensibilidad muestra cómo cambia cada respuesta cuando variamos 
    un factor a la vez, manteniendo los otros factores constantes.
    """)
    
    st.markdown("---")
    
    # Selector de respuesta para sensibilidad
    sensitivity_response = st.selectbox(
        "Seleccionar Respuesta para Análisis",
        ["Production", "EUN", "EUA", "RBC"],
        format_func=lambda x: {
            "Production": "Producción (kg/ha)",
            "EUN": "Eficiencia de Nitrógeno (kg/kg)",
            "EUA": "Eficiencia de Agua (kg/m³)",
            "RBC": "Relación Beneficio-Costo"
        }[x],
        key='sensitivity_response'
    )
    
    # Valores base para el análisis
    base_values = {
        'Irrigation': irrigation,
        'Nitrogen': nitrogen,
        'Density': density
    }
    
    # Crear gráfico de sensibilidad
    fig_sensitivity = create_sensitivity_analysis(base_values, sensitivity_response)
    st.plotly_chart(fig_sensitivity, use_container_width=True)
    
    st.markdown("---")
    
    # Análisis de pendientes
    st.subheader("📊 Análisis de Pendientes en el Punto Actual")
    
    # Calcular derivadas parciales en el punto actual
    h = 0.01  # Paso pequeño para derivada numérica
    
    derivatives = {}
    for factor in ['Irrigation', 'Nitrogen', 'Density']:
        # Calcular derivada numérica
        test_plus = base_values.copy()
        test_minus = base_values.copy()
        test_plus[factor] += h
        test_minus[factor] -= h
        
        y_plus = calculate_rsm_response(
            test_plus['Irrigation'], test_plus['Nitrogen'], 
            test_plus['Density'], sensitivity_response
        )
        y_minus = calculate_rsm_response(
            test_minus['Irrigation'], test_minus['Nitrogen'], 
            test_minus['Density'], sensitivity_response
        )
        
        derivative = (y_plus - y_minus) / (2 * h)
        derivatives[factor] = derivative
    
    # Mostrar tabla de sensibilidades
    sensitivity_data = pd.DataFrame({
        'Factor': list(derivatives.keys()),
        'Derivada Parcial': list(derivatives.values()),
        'Interpretación': [
            f"Un aumento de 1 unidad causa {'aumento' if d > 0 else 'disminución'} de {abs(d):.4f} en {sensitivity_response}"
            for d in derivatives.values()
        ]
    })
    
    st.dataframe(sensitivity_data, width='stretch', hide_index=True)
    
    st.markdown("""
        <div style='background-color: #d1ecf1; color: #0c5460; padding: 1rem; border-radius: 8px; border-left: 4px solid #17a2b8; margin-top: 1rem;'>
            <strong>💡 Interpretación:</strong><br>
            • Las derivadas parciales indican la tasa de cambio instantánea<br>
            • Valores positivos indican que aumentar el factor aumenta la respuesta<br>
            • Valores negativos indican que aumentar el factor disminuye la respuesta<br>
            • Valores cercanos a cero indican que estamos cerca de un óptimo local
        </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PIE DE PÁGINA
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #0a2f6b;'>
    <p><strong>Diseño de Experimentos - Universidad Santo Tomás</strong></p>
    <p>Yeison Poveda • Victor Díaz • Noviembre 2025</p>
</div>
""", unsafe_allow_html=True)
