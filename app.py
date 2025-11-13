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

# ============================================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================================

st.set_page_config(
    page_title="Optimización Agrícola",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CSS PERSONALIZADO (Colores USTA)
# ============================================================================

st.markdown("""
    <style>
        /* Colores de Marca USTA */
        :root {
            --primary: #0a2f6b;
            --accent: #f9a602;
            --teal: #1c9c9c;
        }
        
        /* Encabezados */
        h1, h2, h3 {
            color: var(--primary) !important;
        }
        
        /* Métricas */
        [data-testid="stMetricValue"] {
            font-size: 28px;
            color: var(--primary);
        }
        
        /* Barra lateral */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f5f7fa 0%, #e8ecf1 100%);
        }
        
        /* Botones */
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
        
        /* Destacar valores seleccionados */
        .selected-values {
            background-color: #fff3cd;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid var(--accent);
        }
    </style>
""", unsafe_allow_html=True)

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
# FUNCIONES AUXILIARES
# ============================================================================

def calculate_metrics(irrigation, nitrogen, density):
    """Calcular métricas para valores dados de factores"""
    # Encontrar el punto de datos más cercano
    distances = np.sqrt(
        ((data['Irrigation'] - irrigation) ** 2) / 1000000 +  # Normalizar
        ((data['Nitrogen'] - nitrogen) ** 2) / 10000 +
        ((data['Density'] - density) ** 2)
    )
    
    closest_idx = distances.idxmin()
    row = data.iloc[closest_idx]
    
    # Calcular costos
    cost_n = nitrogen * 0.0035
    cost_water = irrigation * 0.0029
    cost_total = 913.44 + cost_n + cost_water
    
    # Calcular RBC
    revenue = row['Production'] * 0.30
    rbc = revenue / cost_total if cost_total > 0 else 0
    
    return {
        "Production": row['Production'],
        "EUN": row['EUN'],
        "EUA": row['EUA'],
        "RBC": rbc,
        "Cost_N": cost_n,
        "Cost_Water": cost_water,
        "Cost_Total": cost_total,
        "Revenue": revenue,
        "Profit": revenue - cost_total
    }

def create_3d_surface(data, x_col, y_col, z_col, title, selected_point=None):
    """Crear un gráfico de superficie 3D"""
    
    # Obtener valores únicos para x e y
    x_unique = sorted(data[x_col].unique())
    y_unique = sorted(data[y_col].unique())
    
    # Crear una tabla pivote para la superficie
    pivot = data.pivot_table(values=z_col, index=y_col, columns=x_col, aggfunc='mean')
    
    # Crear el gráfico de superficie
    fig = go.Figure()
    
    # Agregar superficie
    fig.add_trace(go.Surface(
        x=x_unique,
        y=y_unique,
        z=pivot.values,
        colorscale='Viridis',
        name=z_col,
        opacity=0.9
    ))
    
    # Agregar punto seleccionado si existe
    if selected_point:
        fig.add_trace(go.Scatter3d(
            x=[selected_point[x_col]],
            y=[selected_point[y_col]],
            z=[selected_point[z_col]],
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
        font=dict(size=12)
    )
    
    return fig

def create_scatter_3d(data, x_col, y_col, z_col, title, selected_point=None):
    """Crear un gráfico de dispersión 3D de puntos de datos reales"""
    
    fig = go.Figure()
    
    # Agregar puntos de datos
    fig.add_trace(go.Scatter3d(
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
        text=[f"Corrida {i}<br>{x_col}: {x}<br>{y_col}: {y}<br>{z_col}: {z:.1f}" 
              for i, x, y, z in zip(data['Run'], data[x_col], data[y_col], data[z_col])],
        hoverinfo='text',
        name='Datos Experimentales'
    ))
    
    # Agregar punto seleccionado si existe
    if selected_point:
        fig.add_trace(go.Scatter3d(
            x=[selected_point[x_col]],
            y=[selected_point[y_col]],
            z=[selected_point[z_col]],
            mode='markers',
            marker=dict(
                size=12,
                color='red',
                symbol='diamond',
                line=dict(color='white', width=2)
            ),
            name='Tu Selección',
            hovertemplate=f'<b>Tu Selección</b><br>{x_col}: %{{x}}<br>{y_col}: %{{y}}<br>{z_col}: %{{z:.1f}}<extra></extra>'
        ))
    
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

st.sidebar.markdown("---")
st.sidebar.info("""
**Referencia del Paper:**  
Yaguas, O. J. (2017). Metodología de superficie de respuesta para la optimización de una producción agrícola.
""")

# ============================================================================
# CONTENIDO PRINCIPAL - PESTAÑAS
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Resumen de Datos", 
    "📈 Visualizaciones", 
    "🎯 Solución Óptima",
    "💰 Análisis Económico"
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
    st.dataframe(factor_summary, use_container_width=True, hide_index=True)
    
    st.subheader("Rangos de Respuestas")
    response_summary = pd.DataFrame({
        'Respuesta': ['Producción', 'EUN', 'EUA', 'RBC'],
        'Mínimo': [data['Production'].min(), data['EUN'].min(), data['EUA'].min(), data['RBC'].min()],
        'Máximo': [data['Production'].max(), data['EUN'].max(), data['EUA'].max(), data['RBC'].max()],
        'Media': [data['Production'].mean(), data['EUN'].mean(), data['EUA'].mean(), data['RBC'].mean()],
        'Unidad': ['kg/ha', 'kg/kg', 'kg/m³', '-'],
        'Objetivo': ['Maximizar', 'Maximizar', 'Maximizar', 'Maximizar']
    })
    st.dataframe(response_summary, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Mostrar datos sin procesar
    with st.expander("📋 Ver Conjunto de Datos Completo (48 corridas)"):
        st.dataframe(data, use_container_width=True)

# ----------------------------------------------------------------------------
# PESTAÑA 2: VISUALIZACIONES
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
        <br><b>Predicción:</b> Producción = {current_metrics['Production']:.0f} kg/ha
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
        ["Superficie 3D", "Dispersión 3D (Puntos de Datos)"],
        horizontal=True
    )
    
    st.markdown("---")
    
    # Crear dos gráficos lado a lado
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"{response} vs Riego × Nitrógeno")
        if plot_type == "Superficie 3D":
            fig1 = create_3d_surface(data, "Irrigation", "Nitrogen", response, 
                                    f"Superficie de {response}", selected_point)
        else:
            fig1 = create_scatter_3d(data, "Irrigation", "Nitrogen", response,
                                    f"Puntos de Datos de {response}", selected_point)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader(f"{response} vs Nitrógeno × Densidad")
        if plot_type == "Superficie 3D":
            fig2 = create_3d_surface(data, "Nitrogen", "Density", response,
                                    f"Superficie de {response}", selected_point)
        else:
            fig2 = create_scatter_3d(data, "Nitrogen", "Density", response,
                                    f"Puntos de Datos de {response}", selected_point)
        st.plotly_chart(fig2, use_container_width=True)
    
    st.info("💡 **Sugerencia:** El punto rojo 💎 en los gráficos muestra tu selección actual de los controles deslizantes.")

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
        use_container_width=True,
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
        st.plotly_chart(fig1, use_container_width=True)
    
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
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    
    st.info("""
    **Ideas Clave:**
    - La Relación Beneficio-Costo (RBC) de 2.3 significa que por cada $1 invertido, se obtienen $2.30 de retorno
    - Los costos de agua y nitrógeno son mínimos comparados con los costos de producción fijos
    - La solución óptima prioriza la eficiencia sobre la producción máxima
    - Precio del maíz asumido: $0.30/kg
    """)

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
