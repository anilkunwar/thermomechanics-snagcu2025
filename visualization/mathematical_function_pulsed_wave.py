import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Function to compute the Gaussian heat source with pulse modulation
def gaussian_heat_source(x, y, time, coeff, alpha, freq, speed, dist0, yzero, pulse_width):
    # Calculate the radial distance
    r = np.sqrt((x - dist0 - time * speed)**2 + (y - yzero)**2)
    
    # Define the odd harmonics sine wave modulation function (phi)
    harmonic_sum = 0.0
    # Check if the pulse is "on" based on the pulse width
    if (time % (1 / freq)) < pulse_width:
        for k in range(1, 6, 2):  # Sum first 3 odd harmonics (1, 3, 5)
            harmonic_sum += np.sin(k * 2 * np.pi * freq * time) / k
        
        # Set phi to harmonic_sum when pulse is on
        phi = harmonic_sum
    else:
        # Set phi to 0 when pulse is off
        phi = 0.0
    
    # Calculate the Gaussian function modulated by the pulse
    f = phi * coeff * np.exp(-2 * r**2 / alpha**2)
    
    return f

# Streamlit Interface
st.title("Gaussian Heat Source with Pulse Modulation")

# Parameters for the simulation (these can be adjusted through sliders)
coeff = st.sidebar.slider('Coefficient (Coeff)', min_value=0.1, max_value=10.0, value=1.0)
alpha = st.sidebar.slider('Heat Source Width (Alpha)', min_value=0.1, max_value=10.0, value=1.0)
freq = st.sidebar.slider('Repetition Rate (Frequency, Hz)', min_value=1.0, max_value=100.0, value=30.0)
pulse_width = st.sidebar.slider('Apparent Pulse Width (ms)', min_value=1.0, max_value=100.0, value=16.670) * 1e-3
speed = st.sidebar.slider('Heat Source Speed', min_value=0.1, max_value=10.0, value=0.0)
dist0 = st.sidebar.slider('Initial Position (Dist0)', min_value=0.0, max_value=10.0, value=0.0)
yzero = st.sidebar.slider('Initial Y Position (yzero)', min_value=-5.0, max_value=5.0, value=0.0)

# Time slider (time in seconds)
time = st.sidebar.slider('Time (ms)', min_value=0, max_value=67, value=0) * 1e-3  # Convert ms to seconds

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

