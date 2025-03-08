import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Data
T = np.array([
    300, 310, 320, 330, 340, 350, 360, 370, 380, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500,
    510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700,
    710, 720, 730, 740, 750, 760, 770, 780, 790
])
h = np.array([
    4350.02, 6709.24, 9092.20, 11501.14, 13939.11, 16410.67, 18923.40, 21492.02, 25233.03, 30532.85,
    33295.83, 36180.40, 39267.65, 42752.65, 47131.18, 53119.56, 61135.78, 72940.09, 96104.26, 111400.68,
    113907.43, 116404.36, 118890.52, 121367.03, 123834.94, 126295.19, 128748.64, 131196.07, 133638.18, 136075.63,
    138509.01, 140938.84, 143365.61, 145789.76, 148211.69, 150631.75, 153050.25, 155467.49, 157883.72, 160299.17,
    162714.03, 165128.48, 167542.66, 169956.71, 172370.74, 174784.82, 177199.04, 179613.45, 182028.09
]) / 1000  # Convert J/kg to kJ/kg

# Streamlit sidebar controls
st.sidebar.header("Customize Plot Appearance")
title = st.sidebar.text_input("Plot Title", "Enthalpy vs Temperature")
xlabel = st.sidebar.text_input("X-axis Label", "Temperature (K)")
ylabel = st.sidebar.text_input("Y-axis Label", "Enthalpy (kJ/kg)")
#fontsize = st.sidebar.slider("Font Size", 10, 30, 16)
fontsize = st.sidebar.slider("Font Size", 10, 40, 16)
marker_size = st.sidebar.slider("Marker Size", 2, 10, 6)
line_width = st.sidebar.slider("Line Thickness", 1, 5, 2)
border_thickness = st.sidebar.slider("Box Thickness", 1, 5, 2)
#tick_label_size = st.sidebar.slider("Tick Label Size", 8, 20, 12)
tick_label_size = st.sidebar.slider("Tick Label Size", 8, 30, 12)
tick_thickness = st.sidebar.slider("Tick Thickness", 0.5, 3.0, 1.0)
tick_length = st.sidebar.slider("Tick Length", 2, 10, 5)
#tick_number = st.sidebar.slider("Number of Tick Labels", 5, 15, 10)
tick_number = st.sidebar.slider("Number of Tick Labels", 4, 15, 8)
curve_color = st.sidebar.color_picker("Curve Color", "#1f77b4")
marker_color = st.sidebar.color_picker("Marker Color", "#ff7f0e")

# Create figure
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(T, h, marker='o', linestyle='-', markersize=marker_size, linewidth=line_width,
        color=curve_color, markerfacecolor=marker_color, label="h-T Curve")
ax.set_title(title, fontsize=fontsize+2)
ax.set_xlabel(xlabel, fontsize=fontsize)
ax.set_ylabel(ylabel, fontsize=fontsize)
#ax.legend(fontsize=fontsize-2)
ax.grid(True, linestyle='--', alpha=0.6)
ax.spines['top'].set_linewidth(border_thickness)
ax.spines['right'].set_linewidth(border_thickness)
ax.spines['bottom'].set_linewidth(border_thickness)
ax.spines['left'].set_linewidth(border_thickness)
ax.tick_params(axis='both', which='major', labelsize=tick_label_size, width=tick_thickness, length=tick_length)
ax.tick_params(axis='both', which='minor', width=tick_thickness * 0.7, length=tick_length * 0.7)
ax.set_xticks(np.linspace(T.min(), T.max(), tick_number))
ax.set_yticks(np.linspace(h.min(), h.max(), tick_number))

# Show figure in Streamlit
st.pyplot(fig)

