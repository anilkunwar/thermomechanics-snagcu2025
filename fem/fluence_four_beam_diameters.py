import streamlit as st
import numpy as np
import plotly.graph_objects as go
import math
import pandas as pd

st.title('Fluence Comparison for Adjustable Beam Diameters')

# Define the colormap list globally
cmaps = ['balance', 'bluered', 'hsv', 'jet', 'picnic', 'portland', 
         'rainbow', 'rdylbu_r', 'spectral_r', 'turbo']

st.sidebar.header('Parameters')
k = st.sidebar.slider('k (Super-Gaussian order)', min_value=0.1, 
                      max_value=10.0, value=1.0, step=0.1)
C = st.sidebar.slider('C (Shape factor)', min_value=0.1, 
                     max_value=4.0, value=2.0, step=0.1)
colormap_index = st.sidebar.slider('Colormap', min_value=0, 
                                  max_value=9, value=6, step=1)

# Beam diameter sliders with default values
st.sidebar.subheader('Beam Diameters (mm)')
d1 = st.sidebar.slider('Diameter 1', 0.1, 5.0, 1.0, 0.1)
d2 = st.sidebar.slider('Diameter 2', 0.1, 5.0, 1.5, 0.1)
d3 = st.sidebar.slider('Diameter 3', 0.1, 5.0, 1.7, 0.1)
d4 = st.sidebar.slider('Diameter 4', 0.1, 5.0, 2.0, 0.1)
diameters = [d1, d2, d3, d4]

# Constants
ENERGY = 0.0075  # 7.5 mJ in Joules

def calculate_fluence(E, k_val, C_val, diameter):
    """Calculate fluence distribution for given parameters"""
    # Convert diameter to radius in micrometers
    radius_um = (diameter / 2) * 1000  # mm -> μm
    radius_m = radius_um * 1e-6  # Convert to meters
    
    # Create coordinate grid
    max_radius = radius_um * 2  # Display area = 2× beam radius
    x = np.linspace(-max_radius, max_radius, 100)
    y = np.linspace(-max_radius, max_radius, 100)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2 + Y**2)  # in microns
    
    # Calculate peak fluence
    numerator = E * k_val * (C_val ** (1/k_val))
    denominator = (math.gamma(1/k_val) * np.pi * (radius_m ** 2))
    F0 = numerator / denominator
    
    # Fluence distribution
    fluence = F0 * np.exp(-C_val * (r**2 / radius_um**2)**k_val)
    
    return fluence, x, y

# Calculate fluence distributions for all diameters
fluence_data = []
for diam in diameters:
    fluence, x, y = calculate_fluence(ENERGY, k, C, diam)
    fluence_data.append((fluence, x, y, diam))

# Find global maximum for consistent color scaling
global_max = max(f[0].max() for f in fluence_data)

# Create plots in 2x2 grid
st.subheader("Fluence Distribution Comparison")
cols = st.columns(2)
for idx, (fluence, x, y, diam) in enumerate(fluence_data):
    with cols[idx % 2]:  # Alternate between columns
        fig = go.Figure(data=[go.Surface(z=fluence, x=x, y=y, 
                        colorscale=cmaps[colormap_index], 
                        cmin=0, cmax=global_max)])
        fig.update_layout(
            scene=dict(
                xaxis_title='X (μm)',
                yaxis_title='Y (μm)',
                zaxis_title='Fluence (J/m²)',
                zaxis=dict(range=[0, global_max]),
                aspectratio=dict(x=1, y=1, z=0.7)
            ),
            title=dict(text=f"{diam} mm Beam", font=dict(size=18)),
            margin=dict(l=0, r=0, b=0, t=40),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# Display metrics in an expandable section
with st.expander("Show Fluence Metrics"):
    metrics = {
        "Diameter (mm)": [],
        "Peak Fluence (J/m²)": [],
        "Average Fluence (J/m²)": []
    }
    
    for diam in diameters:
        radius_m = (diam/2) * 1e-3  # Convert to meters
        area = np.pi * (radius_m ** 2)
        
        metrics["Diameter (mm)"].append(diam)
        metrics["Peak Fluence (J/m²)"].append(
            next(f[0].max() for f in fluence_data if f[3] == diam)
        )
        metrics["Average Fluence (J/m²)"].append(ENERGY / area)
    
    df = pd.DataFrame(metrics)
    st.table(df.style.format({
        "Peak Fluence (J/m²)": "{:.2e}",
        "Average Fluence (J/m²)": "{:.2e}"
    }))
