# unified_fea_laser_app.py
import streamlit as st
import os
import glob
import re
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from typing import Dict, Any, Optional
import traceback

warnings.filterwarnings('ignore')

# =============================================
# PATH CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEA_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "fea_solutions")
os.makedirs(FEA_SOLUTIONS_DIR, exist_ok=True)

# =============================================
# FOLDER NAME PARSER
# =============================================
def parse_folder_name(folder: str):
    """
    Parse folder name like 'q0p5mJ-delta4p2ns' → (0.5, 4.2)
    """
    match = re.match(r"q([\dp\.]+)mJ-delta([\dp\.]+)ns", folder)
    if not match:
        return None, None
    e_str, d_str = match.groups()
    try:
        energy = float(e_str.replace("p", "."))
        duration = float(d_str.replace("p", "."))
        return energy, duration
    except ValueError:
        return None, None

# =============================================
# VTU LOADING (meshio as primary, pyvista optional)
# =============================================
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

import meshio

def load_vtu_with_meshio(vtu_path: str) -> Dict[str, Any]:
    """Load .vtu file using meshio and return structured dict."""
    mesh = meshio.read(vtu_path)
    point_data = {}
    for key, value in mesh.point_data.items():
        if isinstance(value, np.ndarray):
            point_data[key] = value.astype(np.float32)
        else:
            point_data[key] = np.array(value, dtype=np.float32)
    triangles = None
    for cell_block in mesh.cells:
        if cell_block.type == "triangle":
            triangles = cell_block.data.astype(np.int32)
            break
    return {
        "points": mesh.points.astype(np.float32),
        "point_data": point_data,
        "triangles": triangles
    }

def load_vtu_with_pyvista(vtu_path: str) -> Dict[str, Any]:
    """Load .vtu file using pyvista and return structured dict."""
    mesh = pv.read(vtu_path)
    point_data = {}
    for key in mesh.point_data.keys():
        arr = mesh.point_data[key]
        if isinstance(arr, np.ndarray):
            point_data[key] = arr.astype(np.float32)
        else:
            point_data[key] = np.array(arr, dtype=np.float32)
    # Extract triangles if present
    triangles = None
    if 'triangles' in mesh.cells_dict:
        triangles = mesh.cells_dict['triangle'].astype(np.int32)
    return {
        "points": mesh.points.astype(np.float32),
        "point_data": point_data,
        "triangles": triangles
    }

def load_vtu(vtu_path: str) -> Dict[str, Any]:
    """Load a single .vtu file using available backend."""
    if PYVISTA_AVAILABLE:
        try:
            return load_vtu_with_pyvista(vtu_path)
        except Exception as e:
            st.warning(f"PyVista failed to read {vtu_path}, falling back to meshio: {e}")
    return load_vtu_with_meshio(vtu_path)

# =============================================
# LOAD ALL SIMULATIONS (FULL FIELDS + SUMMARIES)
# =============================================
@st.cache_data
def load_all_simulations_full():
    """
    Load all simulations from fea_solutions/.
    Returns dict: {
        'sim_name': {
            'energy_mJ', 'duration_ns',
            'points', 'fields', 'field_info',
            'n_timesteps', 'triangles',
            'timesteps', 'max_temperature', 'mean_displacement',
            'max_von_mises', 'strain_energy'
        }
    }
    """
    simulations = {}
    pattern = os.path.join(FEA_SOLUTIONS_DIR, "q*mJ-delta*ns")
    folders = glob.glob(pattern)
    
    for folder in folders:
        folder_name = os.path.basename(folder)
        energy, duration = parse_folder_name(folder_name)
        if energy is None or duration is None:
            continue

        vtu_files = sorted(glob.glob(os.path.join(folder, "a_t????.vtu")))
        if not vtu_files:
            continue

        # Load first file to get structure
        try:
            first_mesh = load_vtu(vtu_files[0])
        except Exception as e:
            st.warning(f"Skipping {folder}: failed to read first VTU: {e}")
            continue

        points = first_mesh["points"]
        triangles = first_mesh["triangles"]
        n_pts = len(points)
        n_steps = len(vtu_files)

        # Initialize field containers
        fields = {}
        field_info = {}
        for key, arr in first_mesh["point_data"].items():
            if arr.ndim == 1:
                field_info[key] = ("scalar", 1)
                fields[key] = np.full((n_steps, n_pts), np.nan, dtype=np.float32)
            elif arr.ndim == 2:
                field_info[key] = ("vector", arr.shape[1])
                fields[key] = np.full((n_steps, n_pts, arr.shape[1]), np.nan, dtype=np.float32)
            else:
                continue  # Skip unsupported
            fields[key][0] = arr

        # Load remaining timesteps
        for t_idx in range(1, n_steps):
            try:
                mesh_data = load_vtu(vtu_files[t_idx])
                for key in field_info:
                    if key in mesh_data["point_data"]:
                        fields[key][t_idx] = mesh_data["point_data"][key]
            except Exception as e:
                st.warning(f"Skipping timestep {t_idx} in {folder}: {e}")
                # Leave as NaN

        # === Extract scalar summaries per timestep ===
        timesteps = np.arange(1, n_steps + 1, dtype=np.float32)  # 1ns per step
        max_temperature = []
        mean_displacement = []
        max_von_mises = []
        strain_energy = []

        for t in range(n_steps):
            # Temperature
            if "temperature" in fields:
                T_vals = fields["temperature"][t]
                T_vals = T_vals[~np.isnan(T_vals)]
                T_max = float(np.max(T_vals)) if T_vals.size > 0 else 0.0
            else:
                T_max = 0.0
            max_temperature.append(T_max)

            # Displacement
            if "displacement" in fields:
                disp = fields["displacement"][t]
                if disp.ndim == 2:
                    disp_mag = np.linalg.norm(disp, axis=1)
                else:
                    disp_mag = np.abs(disp)
                disp_mag = disp_mag[~np.isnan(disp_mag)]
                disp_mean = float(np.mean(disp_mag)) if disp_mag.size > 0 else 0.0
            else:
                disp_mean = 0.0
            mean_displacement.append(disp_mean)

            # Stress (assume "principal stress" or "von_mises")
            stress_key = None
            for key in ["principal stress", "von_mises", "stress"]:
                if key in fields:
                    stress_key = key
                    break
            if stress_key:
                stress_vals = fields[stress_key][t]
                stress_vals = stress_vals[~np.isnan(stress_vals)]
                stress_max = float(np.max(stress_vals)) if stress_vals.size > 0 else 0.0
            else:
                stress_max = 0.0
            max_von_mises.append(stress_max)

            # Strain energy proxy
            energy_proxy = stress_max * disp_mean
            strain_energy.append(energy_proxy)

        # Convert to numpy arrays
        max_temperature = np.array(max_temperature, dtype=np.float32)
        mean_displacement = np.array(mean_displacement, dtype=np.float32)
        max_von_mises = np.array(max_von_mises, dtype=np.float32)
        strain_energy = np.array(strain_energy, dtype=np.float32)

        simulations[folder_name] = {
            "energy_mJ": energy,
            "duration_ns": duration,
            "points": points,
            "fields": fields,
            "field_info": field_info,
            "n_timesteps": n_steps,
            "triangles": triangles,
            "timesteps": timesteps,
            "max_temperature": max_temperature,
            "mean_displacement": mean_displacement,
            "max_von_mises": max_von_mises,
            "strain_energy": strain_energy,
        }

    return simulations

# =============================================
# PHYSICS-INFORMED ATTENTION EXTRAPOLATOR
# =============================================
class FEALaserExtrapolator:
    """
    Transformer-inspired attention extrapolator using physics-aware embeddings.
    Operates on scalar field summaries (not full fields).
    """
    def __init__(self, sigma_param: float = 0.4):
        self.sigma_param = sigma_param
        self.source_db = []  # List of simulation summary records

    def set_source_db(self, simulations: Dict[str, Any]):
        """Initialize from loaded simulations."""
        self.source_db = []
        for sim_data in simulations.values():
            self.source_db.append({
                "E": sim_data["energy_mJ"],
                "tau": sim_data["duration_ns"],
                "timesteps": sim_data["timesteps"],
                "max_temperature": sim_data["max_temperature"],
                "mean_displacement": sim_data["mean_displacement"],
                "max_von_mises": sim_data["max_von_mises"],
                "strain_energy": sim_data["strain_energy"]
            })

    def _physics_embedding(self, E: float, tau: float, t: float) -> np.ndarray:
        """
        Create normalized physics-aware embedding vector.
        Dimensions:
          0: log-energy normalized
          1: pulse duration normalized
          2: dimensionless time (t / tau)
          3: thermal diffusion proxy (t * alpha / L^2 ~ t * 0.1)
        """
        # Log-energy normalization (laser ablation is log-linear)
        logE = np.log1p(E)
        logE_min = np.log1p(0.1)
        logE_max = np.log1p(20.0)
        logE_norm = (logE - logE_min) / (logE_max - logE_min + 1e-8)

        # Pulse duration normalization
        tau_min, tau_max = 0.5, 10.0
        tau_norm = (tau - tau_min) / (tau_max - tau_min + 1e-8)

        # Dimensionless time
        t_rel = t / (tau + 1e-8)

        # Thermal diffusion proxy (assumes alpha ~ 0.1 nm²/ns, L ~ 1 um)
        Lambda = t * 0.1

        return np.array([logE_norm, tau_norm, t_rel, Lambda], dtype=np.float32)

    def predict_at_time(self, E_query: float, tau_query: float, t_query: float) -> Dict[str, float]:
        """Predict field summaries at a single (E, tau, t) point."""
        if not self.source_db:
            return {
                "max_temperature": np.nan,
                "mean_displacement": np.nan,
                "max_von_mises": np.nan,
                "strain_energy": np.nan,
                "confidence": 0.0
            }

        query_emb = self._physics_embedding(E_query, tau_query, t_query)
        weights = []
        values = {
            "max_temperature": [],
            "mean_displacement": [],
            "max_von_mises": [],
            "strain_energy": []
        }

        # Compute attention weights over all source timesteps
        for sim in self.source_db:
            for i, t_src in enumerate(sim["timesteps"]):
                src_emb = self._physics_embedding(sim["E"], sim["tau"], t_src)
                dist_sq = np.sum((query_emb - src_emb) ** 2)
                weight = np.exp(-dist_sq / (2 * self.sigma_param ** 2))
                weights.append(weight)
                for key in values:
                    values[key].append(sim[key][i])

        if not weights or sum(weights) == 0:
            return {
                "max_temperature": np.nan,
                "mean_displacement": np.nan,
                "max_von_mises": np.nan,
                "strain_energy": np.nan,
                "confidence": 0.0
            }

        # Normalize attention weights
        weights = np.array(weights, dtype=np.float32)
        weights /= np.sum(weights) + 1e-12

        # Weighted prediction
        prediction = {}
        for key in values:
            pred_val = np.sum(weights * np.array(values[key], dtype=np.float32))
            prediction[key] = float(pred_val)

        # Confidence = exp(-min distance^2 / (2 sigma^2))
        min_dist_sq = float('inf')
        for sim in self.source_db:
            for t_src in sim["timesteps"]:
                src_emb = self._physics_embedding(sim["E"], sim["tau"], t_src)
                dist_sq = np.sum((query_emb - src_emb) ** 2)
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
        confidence = np.exp(-min_dist_sq / (2 * self.sigma_param ** 2))
        prediction["confidence"] = float(confidence)

        return prediction

    def predict_time_series(self, E_query: float, tau_query: float, t_list: np.ndarray) -> Dict[str, list]:
        """Predict over a list of time points."""
        results = {
            "max_temperature": [],
            "mean_displacement": [],
            "max_von_mises": [],
            "strain_energy": [],
            "confidence": []
        }
        for t in t_list:
            pred = self.predict_at_time(E_query, tau_query, t)
            for key in results:
                results[key].append(pred[key])
        return results

# =============================================
# STREAMLIT APPLICATION
# =============================================
def main():
    st.set_page_config(page_title="Unified FEA Laser Simulation Platform", layout="wide")
    st.title("🔬 Unified FEA Laser Simulation Platform")
    st.caption("3D Visualization + Physics-Informed Attention Extrapolation")

    # Load all simulation data (cached)
    try:
        simulations = load_all_simulations_full()
    except Exception as e:
        st.error(f"Failed to load simulations: {e}")
        st.code(traceback.format_exc())
        return

    if not simulations:
        st.error(f"No valid simulations found in `{FEA_SOLUTIONS_DIR}`.")
        st.write("### Expected directory structure:")
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
        st.write("Folder names must match pattern: `q{energy}mJ-delta{duration}ns`")
        st.write("VTU files must be named: `a_t0001.vtu`, `a_t0002.vtu`, ..., `a_t0008.vtu`")
        return

    # Initialize extrapolator in session state
    if "extrapolator" not in st.session_state:
        extrapolator = FEALaserExtrapolator(sigma_param=0.4)
        extrapolator.set_source_db(simulations)
        st.session_state.extrapolator = extrapolator
    extrapolator = st.session_state.extrapolator

    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "🔍 Simulation Viewer (3D/2D)",
        "🚀 Extrapolation Engine",
        "📊 Comparison Dashboard"
    ])

    # =============================================
    # TAB 1: Simulation Viewer
    # =============================================
    with tab1:
        st.header("🔍 Browse and Visualize Source Simulations")
        sim_names = sorted(simulations.keys())
        selected_sim = st.selectbox("Select a simulation", sim_names, key="viewer_sim_select")
        sim = simulations[selected_sim]

        st.sidebar.header("Simulation Info")
        st.sidebar.write(f"**Energy:** {sim['energy_mJ']} mJ")
        st.sidebar.write(f"**Pulse Duration:** {sim['duration_ns']} ns")
        st.sidebar.write(f"**Timesteps:** {sim['n_timesteps']}")

        # Field selection
        available_fields = list(sim["fields"].keys())
        if not available_fields:
            st.error("No fields found in simulation.")
            return
        field = st.selectbox("Select field to visualize", available_fields, key="viewer_field_select")
        timestep = st.slider(
            "Timestep (ns)",
            0,
            sim["n_timesteps"] - 1,
            0,
            key="viewer_timestep_slider"
        )

        # Extract data for plotting
        pts = sim["points"]
        kind, _ = sim["field_info"][field]
        raw_data = sim["fields"][field][timestep]

        if kind == "scalar":
            values = raw_data
            label = field
        else:
            values = np.linalg.norm(raw_data, axis=1)
            label = f"{field} (magnitude)"

        # Clean NaNs for stats
        clean_values = values[~np.isnan(values)]

        # Plotly 3D Mesh
        st.subheader(f"{label} at timestep {timestep + 1} ns – {selected_sim}")
        if sim["triangles"] is not None and len(sim["triangles"]) > 0:
            tri = sim["triangles"]
            mesh3d = go.Mesh3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                i=tri[:, 0],
                j=tri[:, 1],
                k=tri[:, 2],
                intensity=values,
                colorscale="Viridis",
                intensitymode="vertex",
                colorbar=dict(title=label),
                opacity=0.85,
                lighting=dict(ambient=0.8, diffuse=0.6, specular=0.5, roughness=0.5, fresnel=0.2),
                lightposition=dict(x=100, y=200, z=150),
                hovertemplate=(
                    "Value: %{intensity:.2f}<br>"
                    "X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>"
                ),
            )
        else:
            # Fallback to point cloud if no triangles
            mesh3d = go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                marker=dict(
                    size=3,
                    color=values,
                    colorscale="Viridis",
                    colorbar=dict(title=label),
                    showscale=True
                ),
                hovertemplate=(
                    "Value: %{marker.color:.2f}<br>"
                    "X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>"
                ),
            )

        fig3d = go.Figure(data=mesh3d)
        fig3d.update_layout(
            height=700,
            scene=dict(
                aspectmode="data",
                camera_eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig3d, use_container_width=True)

        # 2D Projection
        with st.expander("🖼️ 2D Projection (X-Y Plane)"):
            fig2d, ax = plt.subplots(figsize=(6, 5))
            scatter = ax.scatter(
                pts[:, 0],
                pts[:, 1],
                c=values,
                s=5,
                cmap="viridis",
                alpha=0.8
            )
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title(f"{label} (2D Projection)")
            plt.colorbar(scatter, ax=ax, label=label)
            st.pyplot(fig2d)

        # Statistics
        with st.expander("📊 Field Statistics"):
            if len(clean_values) > 0:
                stats = {
                    "Min": float(clean_values.min()),
                    "Max": float(clean_values.max()),
                    "Mean": float(clean_values.mean()),
                    "Std": float(clean_values.std()),
                    "Count (non-NaN)": int(len(clean_values))
                }
                st.json(stats)
            else:
                st.write("⚠️ All values are NaN.")

        # Time-series for this simulation
        st.subheader("📈 Scalar Field Time Series (This Simulation)")
        time_series_data = pd.DataFrame({
            "Time (ns)": sim["timesteps"],
            "Max Temperature (K)": sim["max_temperature"],
            "Max Von Mises Stress (GPa)": sim["max_von_mises"],
            "Mean Displacement (nm)": sim["mean_displacement"]
        })
        st.line_chart(time_series_data.set_index("Time (ns)"))

    # =============================================
    # TAB 2: Extrapolation Engine
    # =============================================
    with tab2:
        st.header("🚀 Physics-Informed Extrapolation")
        st.markdown("""
        Enter a query (Energy, Pulse Duration, Time) to **interpolate or extrapolate** field summaries.
        The model uses **Gaussian attention over physics-aware embeddings**.
        """)

        col1, col2, col3 = st.columns(3)
        with col1:
            E_query = st.number_input(
                "Laser Energy (mJ)",
                min_value=0.1,
                max_value=50.0,
                value=15.0,
                step=0.5,
                key="extrap_energy"
            )
        with col2:
            tau_query = st.number_input(
                "Pulse Duration (ns)",
                min_value=0.5,
                max_value=20.0,
                value=5.0,
                step=0.5,
                key="extrap_duration"
            )
        with col3:
            max_time = st.number_input(
                "Max Prediction Time (ns)",
                min_value=1,
                max_value=50,
                value=20,
                step=1,
                key="extrap_max_time"
            )

        t_list = np.arange(1, max_time + 1)

        if st.button("🚀 Predict Response", key="extrap_predict_btn"):
            with st.spinner("Computing attention-weighted prediction..."):
                try:
                    results = extrapolator.predict_time_series(E_query, tau_query, t_list)

                    # Plot temperature and stress
                    fig, ax1 = plt.subplots(figsize=(10, 5))
                    ax1.plot(t_list, results["max_temperature"], 'r-', linewidth=2, label='Max Temperature (K)')
                    ax1.set_xlabel("Time (ns)")
                    ax1.set_ylabel("Temperature (K)", color='r')
                    ax1.tick_params(axis='y', labelcolor='r')
                    ax1.grid(True, alpha=0.3)

                    ax2 = ax1.twinx()
                    ax2.plot(t_list, results["max_von_mises"], 'b--', linewidth=2, label='Max Von Mises Stress (GPa)')
                    ax2.set_ylabel("Stress (GPa)", color='b')
                    ax2.tick_params(axis='y', labelcolor='b')

                    fig.tight_layout()
                    st.pyplot(fig)

                    # Confidence
                    last_n = min(5, len(results["confidence"]))
                    avg_conf = np.nanmean(results["confidence"][-last_n:])
                    st.metric(
                        f"Prediction Confidence (last {last_n} ns)",
                        f"{avg_conf:.2f}",
                        delta=None,
                        delta_color="normal"
                    )

                    if avg_conf < 0.3:
                        st.warning("⚠️ **Low confidence**: query is far from training data (high extrapolation risk).")

                    # Results table
                    df_results = pd.DataFrame({
                        "Time (ns)": t_list,
                        "Max Temp (K)": results["max_temperature"],
                        "Max VM Stress (GPa)": results["max_von_mises"],
                        "Mean Disp (nm)": results["mean_displacement"],
                        "Confidence": results["confidence"]
                    })

                    st.dataframe(
                        df_results.style.format({
                            "Max Temp (K)": "{:.1f}",
                            "Max VM Stress (GPa)": "{:.3f}",
                            "Mean Disp (nm)": "{:.3f}",
                            "Confidence": "{:.2f}"
                        }),
                        use_container_width=True
                    )

                    # Export CSV
                    csv = df_results.to_csv(index=False).encode('utf-8')
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
    # TAB 3: Comparison Dashboard
    # =============================================
    with tab3:
        st.header("📊 Compare Source Simulation vs. Prediction")
        st.write("Select a source simulation and optionally modify its parameters to see interpolation/extrapolation.")

        sim_names = sorted(simulations.keys())
        base_sim = st.selectbox("Base simulation", sim_names, key="comp_base_sim")

        base_data = simulations[base_sim]
        E_default = base_data["energy_mJ"]
        tau_default = base_data["duration_ns"]

        col1, col2 = st.columns(2)
        with col1:
            E_comp = st.number_input(
                "Query Energy (mJ)",
                min_value=0.1,
                max_value=50.0,
                value=E_default,
                step=0.5,
                key="comp_energy"
            )
        with col2:
            tau_comp = st.number_input(
                "Query Pulse Duration (ns)",
                min_value=0.5,
                max_value=20.0,
                value=tau_default,
                step=0.5,
                key="comp_duration"
            )

        # Use same time range as base simulation
        t_max_base = int(base_data["timesteps"][-1])
        t_list_comp = np.arange(1, t_max_base + 1)

        if st.button("Compare", key="comp_compare_btn"):
            # Get prediction
            pred = extrapolator.predict_time_series(E_comp, tau_comp, t_list_comp)

            # Plot comparison
            fig, axs = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f"Comparison: Source='{base_sim}' vs Query=(E={E_comp}, τ={tau_comp})")

            # Max Temperature
            axs[0,0].plot(t_list_comp, base_data["max_temperature"][:len(t_list_comp)], 'ro-', label='Source', markersize=3)
            axs[0,0].plot(t_list_comp, pred["max_temperature"], 'r--', label='Predicted')
            axs[0,0].set_title("Max Temperature (K)")
            axs[0,0].set_xlabel("Time (ns)")
            axs[0,0].legend()
            axs[0,0].grid(True, alpha=0.3)

            # Max Von Mises
            axs[0,1].plot(t_list_comp, base_data["max_von_mises"][:len(t_list_comp)], 'bo-', label='Source', markersize=3)
            axs[0,1].plot(t_list_comp, pred["max_von_mises"], 'b--', label='Predicted')
            axs[0,1].set_title("Max Von Mises Stress (GPa)")
            axs[0,1].set_xlabel("Time (ns)")
            axs[0,1].legend()
            axs[0,1].grid(True, alpha=0.3)

            # Mean Displacement
            axs[1,0].plot(t_list_comp, base_data["mean_displacement"][:len(t_list_comp)], 'go-', label='Source', markersize=3)
            axs[1,0].plot(t_list_comp, pred["mean_displacement"], 'g--', label='Predicted')
            axs[1,0].set_title("Mean Displacement (nm)")
            axs[1,0].set_xlabel("Time (ns)")
            axs[1,0].legend()
            axs[1,0].grid(True, alpha=0.3)

            # Confidence
            axs[1,1].plot(t_list_comp, pred["confidence"], 'k-', label='Confidence')
            axs[1,1].set_title("Prediction Confidence")
            axs[1,1].set_xlabel("Time (ns)")
            axs[1,1].set_ylim(0, 1)
            axs[1,1].legend()
            axs[1,1].grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)

            # Summary metrics
            avg_conf_overall = np.nanmean(pred["confidence"])
            st.metric("Average Confidence Over Time", f"{avg_conf_overall:.2f}")

# =============================================
# ENTRY POINT
# =============================================
if __name__ == "__main__":
    main()
