import streamlit as st
import numpy as np
import plotly.graph_objects as go
import math

st.title('Fluence Comparison for Different Beam Diameters')

# Define the colormap list globally
cmaps = ['balance', 'bluered', 'hsv', 'jet', 'picnic', 'portland', 
         'rainbow', 'rdylbu_r', 'spectral_r', 'turbo']

st.sidebar.header('Parameters')
k = st.sidebar.slider('k (Super-Gaussian order)', min_value=0.1, 
                      max_value=10.0, value=1.0, step=0.1)
C = st.sidebar.slider('C (Shape factor)', min_value=0.1, 
                     max_value=4.0, value=2.0, step=0.1)
d1 = st.sidebar.slider('Beam Diameter 1 (mm)', min_value=0.1, 
                      max_value=10.0, value=1.0, step=0.1)
d2 = st.sidebar.slider('Beam Diameter 2 (mm)', min_value=0.1, 
                      max_value=10.0, value=2.0, step=0.1)
colormap_index = st.sidebar.slider('Colormap', min_value=0, 
                                  max_value=9, value=6, step=1)

# Constants
ENERGY = 0.0075  # 7.5 mJ in Joules

def calculate_fluence(E, k_val, C_val, diameter):
    """Calculate fluence distribution for given parameters"""
    # Convert diameter to radius in meters
    radius_um = (diameter / 2) * 1000  # mm -> μm
    radius_m = radius_um * 1e-6
    
    # Create coordinate grid
    max_radius = radius_um * 2
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

# Calculate fluence distributions
fluence1, x1, y1 = calculate_fluence(ENERGY, k, C, d1)
fluence2, x2, y2 = calculate_fluence(ENERGY, k, C, d2)

# Find global maximum for consistent color scaling
global_max = max(fluence1.max(), fluence2.max())

# Create plots
def create_plot(fluence, x, y, title):
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
        title=dict(text=title, font=dict(size=24)),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

# Display plots
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(create_plot(fluence1, x1, y1, 
                   f"{d1} mm Beam"), use_container_width=True)
with col2:
    st.plotly_chart(create_plot(fluence2, x2, y2, 
                   f"{d2} mm Beam"), use_container_width=True)

# Calculate and display key metrics
st.subheader("Key Parameters")
st.write(f"Constant Energy: {ENERGY*1000:.1f} mJ")
st.write(f"Peak Fluence (1.0 mm): {fluence1.max():.2e} J/m²")
st.write(f"Peak Fluence (2.0 mm): {fluence2.max():.2e} J/m²")

# Calculate average fluence
area1 = np.pi * ((d1/2 * 1e-3)**2)  # in m²
area2 = np.pi * ((d2/2 * 1e-3)**2)
st.write(f"\nAverage Fluence (1.0 mm): {ENERGY/area1:.2e} J/m²")
st.write(f"Average Fluence (2.0 mm): {ENERGY/area2:.2e} J/m²")
