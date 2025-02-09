import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

# Quadratic model
def quadratic_model(params, T):
    A, B, C = params
    return A * T**2 + B * T + C

# Objective function
def objective(params, T, nu):
    return np.sum((quadratic_model(params, T) - nu) ** 2)

# R-squared function
def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

st.title("Poisson's Ratio Fit with Quadratic Model")
st.write("Upload CSV files where the first column is Temperature (T) and the second column is Poisson's Ratio (nu). Headers will be ignored if present.")

uploaded_files = st.file_uploader("Upload CSV files", accept_multiple_files=True, type=["csv"])

datasets = []
if uploaded_files:
    for file in uploaded_files:
        df = pd.read_csv(file, header=None)
        df = df.apply(pd.to_numeric, errors='coerce')  # Convert data to numeric, handle errors
        df = df.dropna()  # Remove any NaN values
        
        T = df.iloc[:, 0].values.astype(float)
        nu = df.iloc[:, 1].values.astype(float)
        datasets.append((T, nu))

if datasets:
    results = []
    #T_fit = np.linspace(min(min(T) for T, _ in datasets), max(max(T) for T, _ in datasets), 100) # Finds the maximum values within the original datasets
    T_fit = np.linspace(min(min(T) for T, _ in datasets), 470, 100)  # Extend range up to 470 K # user defined maximum temperature limit
    nu_fits = []
    bounds = [(-1, 1), (-1, 1), (0, 1)]  # Reasonable bounds for A, B, and C
    
    for T, nu in datasets:
        result = differential_evolution(objective, bounds, args=(T, nu))
        results.append(result.x)
        nu_fits.append(quadratic_model(result.x, T_fit))
    
    # Compute averaged parameters
    A_avg = np.mean([res[0] for res in results])
    B_avg = np.mean([res[1] for res in results])
    C_avg = np.mean([res[2] for res in results])
    nu_fit_avg = quadratic_model([A_avg, B_avg, C_avg], T_fit)
    
    # Display fitted parameters
    st.write("### Fitted Parameters:")
    for i, (A, B, C) in enumerate(results):
        r2 = r_squared(datasets[i][1], quadratic_model([A, B, C], datasets[i][0]))
        st.write(f"Dataset {i+1}: A = {A:.3e}, B = {B:.3e}, C = {C:.3e}, R² = {r2:.6f}")
    st.write(f"Averaged Fit: A = {A_avg:.3e}, B = {B_avg:.3e}, C = {C_avg:.3e}")
    
    # Plot results
    fig, ax = plt.subplots()
    for i, (T, nu) in enumerate(datasets):
        ax.scatter(T, nu, label=f'Data {i+1}')
        ax.plot(T_fit, nu_fits[i], '--', label=f'Fit {i+1}')
    ax.plot(T_fit, nu_fit_avg, 'g-', label='Averaged Fit')
    ax.set_xlabel("Temperature (T)")
    ax.set_ylabel("Poisson's ratio (nu)")
    ax.legend()
    ax.grid()
    
    st.pyplot(fig)
    
    # Prepare data for download
    fit_data = pd.DataFrame({'T(K)': T_fit, 'nu': nu_fit_avg})
    csv_data = fit_data.to_csv(index=False)
    dat_data = fit_data.to_csv(index=False, sep=' ')
    
    st.download_button(label="Download CSV", data=csv_data, file_name="fitted_curve.csv", mime="text/csv")
    st.download_button(label="Download DAT", data=dat_data, file_name="fitted_curve.dat", mime="text/plain")


    
