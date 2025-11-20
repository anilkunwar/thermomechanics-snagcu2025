import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Function to compute the Gaussian heat source with a modulated odd harmonics pulse
def gaussian_heat_source(x, y, time, coeff, alpha, freq, speed, dist0, yzero, pulse_width):
    # Calculate the radial distance
    r = np.sqrt((x - dist0 - time * speed)**2 + (y - yzero)**2)
    
    # Calculate the fundamental frequency
    omega_0 = 2 * np.pi * freq  # Angular frequency (rad/s)
    
    # Approximate pulse using odd harmonics
    harmonic_sum = 0
    for k in range(1, 20, 2):  # Include odd harmonics up to the 19th
        harmonic_sum += (1 / k) * np.sin(k * omega_0 * time)
    
    # Scale the harmonic sum to match the desired pulse width
    phi = harmonic_sum * (pulse_width / (1 / freq))
    
    # Calculate the Gaussian function modulated by the pulse
    f = phi * coeff * np.exp(-2 * r**2 / alpha**2)
    
    return f

# Streamlit Interface
st.title("Gaussian Heat Source with Pulse Modulation")

# Parameters for the simulation
coeff = st.sidebar.slider('Coefficient (Coeff)', min_value=0.1, max_value=10.0, value=1.0)
alpha = st.sidebar.slider('Heat Source Width (Alpha)', min_value=0.1, max_value=10.0, value=1.0)
freq = st.sidebar.slider('Repetition Rate (Frequency, Hz)', min_value=1.0, max_value=100.0, value=30.0)
pulse_width = st.sidebar.slider('Pulse Width (ns)', min_value=1.0, max_value=100.0, value=6.0) * 1e-9
speed = st.sidebar.slider('Heat Source Speed', min_value=0.1, max_value=10.0, value=1.0)
dist0 = st.sidebar.slider('Initial Position (Dist0)', min_value=0.0, max_value=10.0, value=0.0)
yzero = st.sidebar.slider('Initial Y Position (yzero)', min_value=-5.0, max_value=5.0, value=0.0)

# Time slider
time = st.sidebar.slider('Time (ms)', min_value=0, max_value=1000, value=0) * 1e-3  # Convert ms to seconds

# Generate the mesh grid for x, y
x_vals = np.linspace(-10, 10, 200)
y_vals = np.linspace(-10, 10, 200)
x, y = np.meshgrid(x_vals, y_vals)

# Calculate the heat source intensity
f = gaussian_heat_source(x, y, time, coeff, alpha, freq, speed, dist0, yzero, pulse_width)

# Create a Plotly figure to visualize the result
fig = go.Figure(data=go.Surface(z=f, x=x, y=y, colorscale='Viridis'))

# Add labels and title
fig.update_layout(
    title="Heat Source Intensity (Pulse Modulated by Odd Harmonics)",
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Intensity'
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

# Show the figure in Streamlit
st.plotly_chart(fig)

