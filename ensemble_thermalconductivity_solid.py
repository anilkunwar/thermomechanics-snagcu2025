import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

st.title("Thermal Conductivity Fit with Ensemble Optimizer")

st.write("Upload CSV files where the first column is Temperature (T) and the second column is Thermal Conductivity (kth). Headers will be ignored if present.")

uploaded_files = st.file_uploader("Upload CSV files", accept_multiple_files=True, type=["csv"])

datasets = []
weights = []

def linear_model(params, T):
    A, B = params
    return A * T + B

def objective(params, T, kth):
    return np.sum((linear_model(params, T) - kth) ** 2)

if uploaded_files:
    for file in uploaded_files:
        df = pd.read_csv(file, header=None)
        df = df.apply(pd.to_numeric, errors='coerce')  # Convert data to numeric, handle errors
        df = df.dropna()  # Remove any NaN values caused by conversion errors
        
        T = df.iloc[:, 0].values.astype(float)
        kth = df.iloc[:, 1].values.astype(float)
        datasets.append((T, kth))
    
    # Default equal weights
    n_datasets = len(datasets)
    weights = np.ones(n_datasets) / n_datasets  
    
    st.write("### Set Weights for Each Dataset (Sum = 1):")
    weight_sliders = []
    total_weight = sum(weights)
    
    for i in range(n_datasets - 1):  # Last weight is determined automatically
        weight = st.slider(f"Weight for Dataset {i+1}", min_value=0.0, max_value=1.0, value=float(weights[i]), step=0.01)
        weight_sliders.append(weight)
    
    last_weight = 1.0 - sum(weight_sliders)
    last_weight = max(0.0, min(1.0, last_weight))  # Ensure the last weight is valid
    weight_sliders.append(last_weight)
    weights = np.array(weight_sliders)
    
    st.write(f"Final Weight for Dataset {n_datasets}: {last_weight:.2f}")
    
    if st.button("Optimize"):
        results = []
        T_fit = np.linspace(min(min(T) for T, _ in datasets), max(max(T) for T, _ in datasets), 100)
        kth_fits = []
        
        for T, kth in datasets:
            bounds = [(-1, 1), (50, 150)]  # Reasonable bounds for A and B
            result = differential_evolution(objective, bounds, args=(T, kth))
            results.append(result.x)
            kth_fits.append(linear_model(result.x, T_fit))
        
        # Compute weighted average parameters
        A_avg = np.sum([res[0] * w for res, w in zip(results, weights)])
        B_avg = np.sum([res[1] * w for res, w in zip(results, weights)])
        kth_fit_avg = linear_model([A_avg, B_avg], T_fit)
        
        # Display fitted parameters
        st.write("### Fitted Parameters:")
        for i, (A, B) in enumerate(results):
            st.write(f"Dataset {i+1}: A = {A:.4f}, B = {B:.4f}, Weight = {weights[i]:.2f}")
        st.write(f"Averaged Fit: A = {A_avg:.4f}, B = {B_avg:.4f}")
        
        # Plot results
        fig, ax = plt.subplots()
        for i, (T, kth) in enumerate(datasets):
            ax.scatter(T, kth, label=f'Data {i+1}')
            ax.plot(T_fit, kth_fits[i], '--', label=f'Fit {i+1}')
        ax.plot(T_fit, kth_fit_avg, 'g-', label='Averaged Fit')
        ax.set_xlabel("Temperature (T)")
        ax.set_ylabel("Thermal Conductivity (kth)")
        ax.legend()
        ax.grid()
        
        st.pyplot(fig)

