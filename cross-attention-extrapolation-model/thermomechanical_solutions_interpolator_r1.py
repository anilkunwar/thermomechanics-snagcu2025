import streamlit as st
import numpy as np
from numba import jit, prange
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
import pandas as pd
import zipfile
from io import BytesIO
import time
import hashlib
import json
from datetime import datetime
from scipy import stats, interpolate
from scipy.ndimage import gaussian_filter, rotate
import warnings
import pickle
import torch
import sqlite3
from io import StringIO
import traceback
import h5py
import msgpack
import dill
import joblib
from pathlib import Path
import tempfile
import base64
import os
import glob
import re
from typing import List, Dict, Any, Optional

warnings.filterwarnings('ignore')

# =============================================
# PATH CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEA_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "fea_solutions")

# Create directory if it doesn't exist
if not os.path.exists(FEA_SOLUTIONS_DIR):
    os.makedirs(FEA_SOLUTIONS_DIR, exist_ok=True)
    st.info(f"📁 Created fea_solutions directory at: {FEA_SOLUTIONS_DIR}")

# =============================================
# OPTIONAL: PYVISTA FOR FEA .VTU
# =============================================
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    st.warning("⚠️ PyVista not installed. Install with: `pip install pyvista vtk`")

# =============================================
# FEA LASER EXTRAPOLATOR
# =============================================
class FEALaserExtrapolator:
    """
    Physics-informed kernel attention extrapolator for laser FEA simulations.
    Works on .vtu files in fea_solutions/q{x}mJ-delta{y}ns/
    """
    def __init__(self, sigma_param=0.3):
        self.sigma_param = sigma_param
        self.source_db = []  # List of {E, tau, timesteps, features...}
        self.fea_dir = FEA_SOLUTIONS_DIR

    def parse_folder_name(self, folder: str):
        """Parse 'q0p5mJ-delta4p2ns' → E=0.5, tau=4.2"""
        match = re.match(r"q([\d\.p]+)mJ-delta([\d\.p]+)ns", folder)
        if not match:
            return None, None
        e_str, t_str = match.groups()
        try:
            E = float(e_str.replace('p', '.'))
            tau = float(t_str.replace('p', '.'))
            return E, tau
        except:
            return None, None

    def load_all_fea_data(self):
        """Load all .vtu simulations and extract spatiotemporal summaries"""
        if not PYVISTA_AVAILABLE:
            st.error("PyVista not available. FEA .vtu support disabled.")
            return

        self.source_db = []
        pattern = os.path.join(self.fea_dir, "q*mJ-delta*ns")
        folders = glob.glob(pattern)

        for folder in folders:
            E, tau = self.parse_folder_name(os.path.basename(folder))
            if E is None:
                continue

            vtu_files = sorted(glob.glob(os.path.join(folder, "a_t????.vtu")))
            if not vtu_files:
                continue

            timesteps = []
            max_temp = []
            mean_disp = []
            max_vm = []
            strain_energy = []

            for vtu in vtu_files:
                try:
                    mesh = pv.read(vtu)
                    t_step = int(re.search(r't(\d+)', vtu).group(1))  # 1–8
                    t_ns = float(t_step)

                    # Extract fields
                    temp = mesh.point_data.get('temperature', None)
                    disp = mesh.point_data.get('displacement', None)
                    stress = mesh.point_data.get('principal stress', None)

                    # Summarize
                    T_max = float(np.nanmax(temp)) if temp is not None else 0.0

                    disp_mag = 0.0
                    if disp is not None:
                        if disp.ndim == 2 and disp.shape[1] == 3:
                            disp_mag = np.linalg.norm(disp, axis=1)
                        else:
                            disp_mag = np.abs(disp)
                        disp_mag = float(np.nanmean(disp_mag))

                    vm_max = float(np.nanmax(stress)) if stress is not None else 0.0
                    energy = vm_max * disp_mag if vm_max > 0 and disp_mag > 0 else 0.0

                    timesteps.append(t_ns)
                    max_temp.append(T_max)
                    mean_disp.append(disp_mag)
                    max_vm.append(vm_max)
                    strain_energy.append(energy)

                except Exception as e:
                    st.warning(f"Skipping {vtu}: {e}")
                    continue

            if timesteps:
                self.source_db.append({
                    'E': E,
                    'tau': tau,
                    'timesteps': np.array(timesteps),
                    'max_temperature': np.array(max_temp),
                    'mean_displacement': np.array(mean_disp),
                    'max_von_mises': np.array(max_vm),
                    'strain_energy': np.array(strain_energy),
                    'source_folder': os.path.basename(folder)
                })

        st.success(f"✅ Loaded {len(self.source_db)} FEA simulations from `{self.fea_dir}`")

    def _physics_embedding(self, E, tau, t):
        """Physics-aware normalized embedding for (E, tau, t)"""
        # Log-energy (more physical for laser ablation)
        logE = np.log1p(E)
        logE_norm = (logE - np.log1p(0.1)) / (np.log1p(20.0) - np.log1p(0.1))

        # Normalize pulse width (assume range 0.5–10 ns)
        tau_norm = (tau - 0.5) / (10.0 - 0.5)

        # Dimensionless time: relative to pulse duration
        t_rel = t / max(tau, 1e-3)

        # Thermal diffusion proxy (approx)
        Lambda = t * 0.1

        return np.array([logE_norm, tau_norm, t_rel, Lambda], dtype=np.float32)

    def predict_at_time(self, E_query, tau_query, t_query):
        """Predict field summaries at (E, tau, t) — supports t > 8 ns!"""
        if not self.source_db:
            return {k: np.nan for k in ['max_temperature', 'mean_displacement', 'max_von_mises', 'strain_energy', 'confidence']}

        q = self._physics_embedding(E_query, tau_query, t_query)
        weights = []
        values = {k: [] for k in ['max_temperature', 'mean_displacement', 'max_von_mises', 'strain_energy']}

        for sim in self.source_db:
            for i, t_src in enumerate(sim['timesteps']):
                k = self._physics_embedding(sim['E'], sim['tau'], t_src)
                dist = np.sum((q - k) ** 2)
                w = np.exp(-dist / (2 * self.sigma_param ** 2))
                weights.append(w)
                for key in values:
                    values[key].append(sim[key][i])

        if not weights or sum(weights) == 0:
            return {k: np.nan for k in values}

        weights = np.array(weights)
        weights /= np.sum(weights)

        pred = {}
        for key in values:
            pred[key] = float(np.sum(weights * np.array(values[key])))

        # Confidence: distance to nearest training point
        min_dist = min(
            np.sum((q - self._physics_embedding(sim['E'], sim['tau'], t_src)) ** 2)
            for sim in self.source_db for t_src in sim['timesteps']
        )
        pred['confidence'] = float(np.exp(-min_dist / (2 * self.sigma_param ** 2)))

        return pred

    def predict_time_series(self, E_query, tau_query, t_list):
        """Predict over a list of times (e.g., [1, 5, 10, 15])"""
        results = {k: [] for k in ['max_temperature', 'mean_displacement', 'max_von_mises', 'strain_energy', 'confidence']}
        for t in t_list:
            pred = self.predict_at_time(E_query, tau_query, t)
            for k in results:
                results[k].append(pred.get(k, np.nan))
        return results

# =============================================
# FEA EXTRAPOLATION INTERFACE
# =============================================
def render_fea_extrapolation():
    st.header("🔬 FEA Laser Simulation Extrapolation")

    # Initialize FEA extrapolator in session state
    if 'fea_extrapolator' not in st.session_state:
        st.session_state.fea_extrapolator = FEALaserExtrapolator(sigma_param=0.4)
        st.session_state.fea_extrapolator.load_all_fea_data()

    fea = st.session_state.fea_extrapolator

    if not fea.source_db:
        st.warning(f"Add FEA simulations to `{FEA_SOLUTIONS_DIR}` in folders like `q0p5mJ-delta4p2ns/` containing `a_t0001.vtu` ... `a_t0008.vtu`")
        st.write("Expected structure:")
        st.code("""
fea_solutions/
└── q2p0mJ-delta2p0ns/
    ├── a_t0001.vtu
    ├── a_t0002.vtu
    └── ...
""")
        return

    # Sidebar configuration
    st.sidebar.header("🔮 Extrapolation Settings")
    with st.sidebar.expander("⚙️ Model Parameters", expanded=False):
        sigma_param = st.slider("Parameter Sigma (σ_param)", 0.05, 1.0, 0.4, 0.05)
        if st.button("🔄 Update Model Parameters"):
            st.session_state.fea_extrapolator = FEALaserExtrapolator(sigma_param=sigma_param)
            st.session_state.fea_extrapolator.load_all_fea_data()
            st.success("Model parameters updated!")

    # Query parameters
    st.subheader("🎯 Query Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        E_q = st.number_input("Energy (mJ)", 0.1, 50.0, 15.0, 0.5, help="Laser energy per pulse")
    with col2:
        tau_q = st.number_input("Pulse On Duration (ns)", 0.5, 20.0, 5.0, 0.5, help="Duration when laser is ON")
    with col3:
        t_max = st.number_input("Max Prediction Time (ns)", 1, 50, 20, 1, help="Total simulation time to predict")

    t_list = np.arange(1, t_max + 1)

    if st.button("🚀 Predict Extrapolated Response", type="primary"):
        with st.spinner("Computing extrapolated response using physics-informed attention..."):
            try:
                results = fea.predict_time_series(E_q, tau_q, t_list)

                # Plot temperature and stress
                fig, ax1 = plt.subplots(figsize=(10, 5))
                ax1.plot(t_list, results['max_temperature'], 'r-', label='Max Temperature (K)')
                ax1.set_xlabel('Time (ns)')
                ax1.set_ylabel('Temperature (K)', color='r')
                ax1.tick_params(axis='y', labelcolor='r')
                ax1.grid(True, alpha=0.3)

                ax2 = ax1.twinx()
                ax2.plot(t_list, results['max_von_mises'], 'b--', label='Max Von Mises Stress (GPa)')
                ax2.set_ylabel('Stress (GPa)', color='b')
                ax2.tick_params(axis='y', labelcolor='b')

                fig.tight_layout()
                st.pyplot(fig)

                # Confidence metric
                recent_conf = results['confidence'][-min(5, len(results['confidence'])):]
                avg_conf = np.nanmean(recent_conf) if len(recent_conf) > 0 else np.nan
                st.metric("Prediction Confidence (last 5 ns)", f"{avg_conf:.2f}")
                if avg_conf < 0.3:
                    st.warning("⚠️ **High extrapolation risk**: Query is far from training data range")

                # Results table
                df = pd.DataFrame({
                    'Time (ns)': t_list,
                    'Max Temp (K)': results['max_temperature'],
                    'Max VM Stress (GPa)': results['max_von_mises'],
                    'Mean Displacement (nm)': results['mean_displacement'],
                    'Strain Energy': results['strain_energy'],
                    'Confidence': results['confidence']
                })
                st.dataframe(df.style.format({
                    'Max Temp (K)': '{:.1f}',
                    'Max VM Stress (GPa)': '{:.3f}',
                    'Mean Displacement (nm)': '{:.4f}',
                    'Strain Energy': '{:.6f}',
                    'Confidence': '{:.2f}'
                }), use_container_width=True)

                # Source simulations used
                st.subheader("📊 Source Simulations Summary")
                source_folders = list(set(sim['source_folder'] for sim in fea.source_db))
                st.write("**Loaded from:**")
                for folder in sorted(source_folders):
                    st.write(f"- `{folder}`")

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.code(traceback.format_exc())

# =============================================
# MAIN APPLICATION
# =============================================
def main():
    st.sidebar.header("📁 Directory Information")
    st.sidebar.write(f"**App Directory:** `{SCRIPT_DIR}`")
    st.sidebar.write(f"**FEA Solutions Directory:** `{FEA_SOLUTIONS_DIR}`")

    st.sidebar.header("🔧 Operation Mode")
    operation_mode = st.sidebar.radio(
        "Select Mode",
        ["FEA Laser Simulation Extrapolation"],
        index=0
    )

    if operation_mode == "FEA Laser Simulation Extrapolation":
        render_fea_extrapolation()

# =============================================
# THEORETICAL ANALYSIS
# =============================================
with st.expander("🔬 Theoretical Analysis: Physics-Informed Attention Extrapolation", expanded=False):
    st.markdown(f"""
## 🎯 **Physics-Informed Attention for Laser FEA Extrapolation**
### **📁 FEA Solutions Directory Integration**
The system loads simulation files from:
