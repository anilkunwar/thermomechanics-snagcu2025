import streamlit as st
import numpy as np
import os
import glob
import re
import warnings
from datetime import datetime
from typing import List, Dict, Any, Optional
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
import traceback

warnings.filterwarnings('ignore')

# =============================================
# PATH CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEA_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "fea_solutions")

# Create FEA solutions directory if it doesn't exist
if not os.path.exists(FEA_SOLUTIONS_DIR):
    os.makedirs(FEA_SOLUTIONS_DIR, exist_ok=True)
    st.info(f"📁 Created fea_solutions directory at: {FEA_SOLUTIONS_DIR}")

# =============================================
# OPTIONAL: PYVISTA FOR .VTU LOADING
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
    Reads .vtu files from fea_solutions/q*x*mJ-delta*y*ns/ directories.
    """
    def __init__(self, sigma_param=0.3):
        self.sigma_param = sigma_param
        self.source_db = []  # List of simulation records
        self.fea_dir = FEA_SOLUTIONS_DIR

    def parse_folder_name(self, folder: str):
        """Parse 'q0p5mJ-delta4p2ns' → E=0.5 (mJ), tau=4.2 (ns)"""
        match = re.match(r"q([\d\.p]+)mJ-delta([\d\.p]+)ns", folder)
        if not match:
            return None, None
        e_str, t_str = match.groups()
        try:
            E = float(e_str.replace('p', '.'))
            tau = float(t_str.replace('p', '.'))
            return E, tau
        except ValueError:
            return None, None

    def load_all_fea_data(self):
        """Load all .vtu simulations and extract spatiotemporal field summaries"""
        if not PYVISTA_AVAILABLE:
            st.error("PyVista is required to load .vtu files.")
            return

        self.source_db = []
        pattern = os.path.join(self.fea_dir, "q*mJ-delta*ns")
        folders = glob.glob(pattern)

        for folder in folders:
            folder_name = os.path.basename(folder)
            E, tau = self.parse_folder_name(folder_name)
            if E is None:
                continue

            # Get all a_t000k.vtu files (k = 1 to 8)
            vtu_files = sorted(glob.glob(os.path.join(folder, "a_t????.vtu")))
            if not vtu_files:
                continue

            # Initialize lists to hold per-timestep summaries
            timesteps = []
            max_temperature = []
            mean_displacement = []
            max_von_mises = []
            strain_energy = []

            # Process each timestep
            for vtu in vtu_files:
                try:
                    mesh = pv.read(vtu)
                    # Extract timestep number from filename: a_t0001.vtu → 1
                    t_step = int(re.search(r't(\d+)', vtu).group(1))
                    t_ns = float(t_step)  # 1 ns per step

                    # Extract fields (assume point data)
                    temperature = mesh.point_data.get('temperature', None)
                    displacement = mesh.point_data.get('displacement', None)
                    stress = mesh.point_data.get('principal stress', None)

                    # Summarize temperature
                    T_max = float(np.nanmax(temperature)) if temperature is not None else 0.0

                    # Summarize displacement (handle vector fields)
                    disp_mag = 0.0
                    if displacement is not None:
                        if displacement.ndim == 2 and displacement.shape[1] == 3:
                            disp_mag = np.linalg.norm(displacement, axis=1)
                        else:
                            disp_mag = np.abs(displacement)
                        disp_mag = float(np.nanmean(disp_mag))

                    # Summarize stress
                    vm_max = float(np.nanmax(stress)) if stress is not None else 0.0

                    # Strain energy proxy
                    energy = vm_max * disp_mag if vm_max > 0 and disp_mag > 0 else 0.0

                    # Store
                    timesteps.append(t_ns)
                    max_temperature.append(T_max)
                    mean_displacement.append(disp_mag)
                    max_von_mises.append(vm_max)
                    strain_energy.append(energy)

                except Exception as e:
                    st.warning(f"Skipping {vtu}: {e}")
                    continue

            # Save this simulation if valid
            if timesteps:
                self.source_db.append({
                    'E': E,
                    'tau': tau,
                    'timesteps': np.array(timesteps),
                    'max_temperature': np.array(max_temperature),
                    'mean_displacement': np.array(mean_displacement),
                    'max_von_mises': np.array(max_von_mises),
                    'strain_energy': np.array(strain_energy)
                })

        st.success(f"✅ Loaded {len(self.source_db)} FEA simulations from `{self.fea_dir}`")

    def _physics_embedding(self, E, tau, t):
        """
        Compute physics-aware normalized embedding for query (E, tau, t)
        """
        # Log-energy scaling (more physical for laser ablation)
        logE = np.log1p(E)
        logE_norm = (logE - np.log1p(0.1)) / (np.log1p(20.0) - np.log1p(0.1))

        # Normalize pulse width
        tau_norm = (tau - 0.5) / (10.0 - 0.5)

        # Dimensionless time: relative to pulse duration
        t_rel = t / max(tau, 1e-3)

        # Thermal diffusion proxy (t * alpha / L^2) ~ t * 0.1
        Lambda = t * 0.1

        return np.array([logE_norm, tau_norm, t_rel, Lambda], dtype=np.float32)

    def predict_at_time(self, E_query, tau_query, t_query):
        """
        Predict field summaries at a specific (E, tau, t) — supports extrapolation.
        """
        if not self.source_db:
            return {
                'max_temperature': np.nan,
                'mean_displacement': np.nan,
                'max_von_mises': np.nan,
                'strain_energy': np.nan,
                'confidence': 0.0
            }

        query_emb = self._physics_embedding(E_query, tau_query, t_query)
        weights = []
        values = {
            'max_temperature': [],
            'mean_displacement': [],
            'max_von_mises': [],
            'strain_energy': []
        }

        # Compute attention weights over all source timesteps
        for sim in self.source_db:
            for i, t_src in enumerate(sim['timesteps']):
                src_emb = self._physics_embedding(sim['E'], sim['tau'], t_src)
                dist_sq = np.sum((query_emb - src_emb) ** 2)
                weight = np.exp(-dist_sq / (2 * self.sigma_param ** 2))
                weights.append(weight)
                for key in values:
                    values[key].append(sim[key][i])

        # Handle empty or zero-weight case
        if not weights or sum(weights) == 0:
            return {
                'max_temperature': np.nan,
                'mean_displacement': np.nan,
                'max_von_mises': np.nan,
                'strain_energy': np.nan,
                'confidence': 0.0
            }

        # Normalize weights
        weights = np.array(weights, dtype=np.float32)
        weights /= np.sum(weights)

        # Compute weighted predictions
        prediction = {}
        for key in values:
            prediction[key] = float(np.sum(weights * np.array(values[key], dtype=np.float32)))

        # Compute confidence as inverse distance to nearest training point
        min_dist = min(
            np.sum((query_emb - self._physics_embedding(sim['E'], sim['tau'], t_src)) ** 2)
            for sim in self.source_db
            for t_src in sim['timesteps']
        )
        prediction['confidence'] = float(np.exp(-min_dist / (2 * self.sigma_param ** 2)))

        return prediction

    def predict_time_series(self, E_query, tau_query, t_list):
        """
        Predict over a list of times (e.g., [1, 5, 10, 15] ns).
        """
        results = {
            'max_temperature': [],
            'mean_displacement': [],
            'max_von_mises': [],
            'strain_energy': [],
            'confidence': []
        }
        for t in t_list:
            pred = self.predict_at_time(E_query, tau_query, t)
            for key in results:
                results[key].append(pred[key])
        return results

# =============================================
# STREAMLIT UI
# =============================================
def render_fea_extrapolation():
    st.header("🔬 FEA Laser Simulation Extrapolation")

    # Initialize extrapolator in session state
    if 'fea_extrapolator' not in st.session_state:
        st.session_state.fea_extrapolator = FEALaserExtrapolator(sigma_param=0.4)
        st.session_state.fea_extrapolator.load_all_fea_data()

    extrapolator = st.session_state.fea_extrapolator

    # Check if any simulations were loaded
    if not extrapolator.source_db:
        st.warning(f"No FEA simulations found in `{FEA_SOLUTIONS_DIR}`.")
        st.write("### Expected Directory Structure:")
        st.code("""
your_app/
└── fea_solutions/
    ├── q0p5mJ-delta4p2ns/
    │   ├── a_t0001.vtu
    │   ├── a_t0002.vtu
    │   └── ...
    └── q2p0mJ-delta2p0ns/
        ├── a_t0001.vtu
        └── ...
""")
        st.write("Folder names must follow the pattern: `q{energy}mJ-delta{duration}ns`")
        st.write("VTU files must be named: `a_t0001.vtu`, `a_t0002.vtu`, ..., `a_t0008.vtu`")
        return

    # User inputs
    st.subheader("🎯 Query Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        E_query = st.number_input("Energy (mJ)", min_value=0.1, max_value=50.0, value=15.0, step=0.5)
    with col2:
        tau_query = st.number_input("Pulse On Duration (ns)", min_value=0.5, max_value=20.0, value=5.0, step=0.5)
    with col3:
        max_time = st.number_input("Max Prediction Time (ns)", min_value=1, max_value=50, value=20, step=1)

    # Generate time list
    t_list = np.arange(1, max_time + 1)

    # Prediction button
    if st.button("🚀 Predict Extrapolated Response"):
        with st.spinner("Computing extrapolated spatiotemporal response..."):
            try:
                results = extrapolator.predict_time_series(E_query, tau_query, t_list)

                # Plot temperature and stress
                fig, ax1 = plt.subplots(figsize=(10, 5))
                ax1.plot(t_list, results['max_temperature'], 'r-', linewidth=2, label='Max Temperature (K)')
                ax1.set_xlabel('Time (ns)')
                ax1.set_ylabel('Temperature (K)', color='r')
                ax1.tick_params(axis='y', labelcolor='r')
                ax1.grid(True, alpha=0.3)

                ax2 = ax1.twinx()
                ax2.plot(t_list, results['max_von_mises'], 'b--', linewidth=2, label='Max Von Mises Stress (GPa)')
                ax2.set_ylabel('Stress (GPa)', color='b')
                ax2.tick_params(axis='y', labelcolor='b')

                fig.tight_layout()
                st.pyplot(fig)

                # Confidence metric
                last_few = min(5, len(results['confidence']))
                avg_conf = np.nanmean(results['confidence'][-last_few:])
                st.metric("Prediction Confidence (last {} ns)".format(last_few), f"{avg_conf:.2f}")
                if avg_conf < 0.3:
                    st.warning("⚠️ Low confidence: query is far from training data (extrapolation risk)")

                # Results table
                df = pd.DataFrame({
                    'Time (ns)': t_list,
                    'Max Temp (K)': results['max_temperature'],
                    'Max VM Stress (GPa)': results['max_von_mises'],
                    'Mean Disp (nm)': results['mean_displacement'],
                    'Confidence': results['confidence']
                })
                st.dataframe(df.style.format({
                    'Max Temp (K)': '{:.1f}',
                    'Max VM Stress (GPa)': '{:.3f}',
                    'Mean Disp (nm)': '{:.3f}',
                    'Confidence': '{:.2f}'
                }), use_container_width=True)

                # Export button
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"fea_prediction_E{E_query}_tau{tau_query}.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.code(traceback.format_exc())

# =============================================
# MAIN APPLICATION
# =============================================
def main():
    st.set_page_config(page_title="FEA Laser Simulation Extrapolation", layout="wide")
    st.title("🔬 FEA Laser Simulation Extrapolation")
    st.caption(f"Using data from: `{FEA_SOLUTIONS_DIR}`")

    # Render the FEA extrapolation UI
    render_fea_extrapolation()

    # Theoretical background
    with st.expander("📘 Theoretical Background", expanded=False):
        st.markdown("""
## Physics-Informed Attention Extrapolation

This tool enables **extrapolation** of laser-material interaction simulations beyond the range of stable FEA computations.

### Input Parameters
- **Energy (mJ)**: Laser pulse energy
- **Pulse On Duration (ns)**: Time during which laser is active
- **Total Time (ns)**: Query time (can exceed simulation duration)

### Field Summaries Extracted from `.vtu`
- **Max Temperature (K)**
- **Mean Displacement Magnitude (nm)**
- **Max Von Mises Stress (GPa)**
- **Strain Energy Proxy**

### Physics-Aware Embedding
Each simulation timestep is embedded using:
- Log-scaled energy: $\log(1 + E)$
- Normalized pulse width
- Dimensionless time: $t / \tau$
- Thermal diffusion proxy: $\Lambda \propto t$

### Kernel Attention
Prediction uses Gaussian kernel attention:
$$
\hat{y}(q) = \sum_i \alpha_i(q) \cdot y_i, \quad \alpha_i(q) \propto \exp\left(-\frac{\| \phi(q) - \phi(x_i) \|^2}{2\sigma^2}\right)
$$
where $\phi(\cdot)$ is the physics embedding.

This provides **smooth interpolation** within the training domain and **graceful extrapolation** with quantified uncertainty.
""")

if __name__ == "__main__":
    main()
    st.caption(f"🔬 FEA Laser Simulation Extrapolation • {FEA_SOLUTIONS_DIR} • 2025")
