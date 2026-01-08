import streamlit as st
import os
import glob
import re
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from datetime import datetime
from typing import List, Dict, Any, Optional
from io import BytesIO
import traceback

# Optional dependencies
try:
    import meshio
    MESHIO_AVAILABLE = True
except ImportError:
    MESHIO_AVAILABLE = False
    st.warning("⚠️ Meshio not installed. Install with: `pip install meshio`")

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    st.warning("⚠️ PyVista not installed. Install with: `pip install pyvista vtk`")

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
    q0p5mJ-delta4p2ns → (0.5, 4.2)
    """
    match = re.match(r"q([\dp\.]+)mJ-delta([\dp\.]+)ns", folder)
    if not match:
        return None, None
    e, d = match.groups()
    return float(e.replace("p", ".")), float(d.replace("p", "."))

# =============================================
# INTEGRATED FEA LASER EXTRAPOLATOR
# =============================================
class IntegratedFEALaserExtrapolator:
    """
    Combined loader and extrapolator for FEA simulations.
    """
    def __init__(self, sigma_param=0.4):
        self.sigma_param = sigma_param
        self.fea_dir = FEA_SOLUTIONS_DIR
        self.simulations = {}  # Raw data from meshio
        self.source_db = []    # Summaries from pyvista

    def load_all_data(self):
        """Load raw data with meshio and summaries with pyvista if available."""
        if MESHIO_AVAILABLE:
            self._load_raw_simulations()
        if PYVISTA_AVAILABLE:
            self._load_summary_data()
        st.success(f"✅ Loaded {len(self.simulations)} raw simulations and {len(self.source_db)} summarized simulations.")

    @st.cache_data
    def _load_raw_simulations(self):
        """Load raw VTU data using meshio (from first code)."""
        self.simulations = {}
        folders = glob.glob(os.path.join(self.fea_dir, "q*mJ-delta*ns"))
        for folder in folders:
            name = os.path.basename(folder)
            energy, duration = parse_folder_name(name)
            if energy is None:
                continue
            vtu_files = sorted(glob.glob(os.path.join(folder, "a_t????.vtu")))
            if not vtu_files:
                continue
            mesh0 = meshio.read(vtu_files[0])
            if not mesh0.point_data:
                continue
            points = mesh0.points.astype(np.float32)
            n_pts = len(points)
            n_steps = len(vtu_files)
            fields = {}
            field_info = {}
            triangles = None
            for cell_block in mesh0.cells:
                if cell_block.type == "triangle":
                    triangles = cell_block.data.astype(np.int32)
                    break
            for key in mesh0.point_data.keys():
                arr = mesh0.point_data[key].astype(np.float32)
                if arr.ndim == 1:
                    field_info[key] = ("scalar", 1)
                    fields[key] = np.full((n_steps, n_pts), np.nan, dtype=np.float32)
                else:
                    field_info[key] = ("vector", arr.shape[1])
                    fields[key] = np.full((n_steps, n_pts, arr.shape[1]), np.nan, dtype=np.float32)
                fields[key][0] = arr
            for t in range(1, n_steps):
                mesh = meshio.read(vtu_files[t])
                for key in field_info.keys():
                    fields[key][t] = mesh.point_data[key].astype(np.float32)
            self.simulations[name] = dict(
                energy_mJ=energy,
                duration_ns=duration,
                points=points,
                fields=fields,
                field_info=field_info,
                n_timesteps=n_steps,
                triangles=triangles,
            )

    def _load_summary_data(self):
        """Load summaries using pyvista (from second code)."""
        self.source_db = []
        folders = glob.glob(os.path.join(self.fea_dir, "q*mJ-delta*ns"))
        for folder in folders:
            folder_name = os.path.basename(folder)
            E, tau = parse_folder_name(folder_name)
            if E is None:
                continue
            vtu_files = sorted(glob.glob(os.path.join(folder, "a_t????.vtu")))
            if not vtu_files:
                continue
            timesteps = []
            max_temperature = []
            mean_displacement = []
            max_von_mises = []
            strain_energy = []
            for vtu in vtu_files:
                try:
                    mesh = pv.read(vtu)
                    t_step = int(re.search(r't(\d+)', vtu).group(1))
                    t_ns = float(t_step)
                    temperature = mesh.point_data.get('temperature', None)
                    displacement = mesh.point_data.get('displacement', None)
                    stress = mesh.point_data.get('principal stress', None)
                    T_max = float(np.nanmax(temperature)) if temperature is not None else 0.0
                    disp_mag = 0.0
                    if displacement is not None:
                        if displacement.ndim == 2 and displacement.shape[1] == 3:
                            disp_mag = np.linalg.norm(displacement, axis=1)
                        else:
                            disp_mag = np.abs(displacement)
                        disp_mag = float(np.nanmean(disp_mag))
                    vm_max = float(np.nanmax(stress)) if stress is not None else 0.0
                    energy = vm_max * disp_mag if vm_max > 0 and disp_mag > 0 else 0.0
                    timesteps.append(t_ns)
                    max_temperature.append(T_max)
                    mean_displacement.append(disp_mag)
                    max_von_mises.append(vm_max)
                    strain_energy.append(energy)
                except Exception as e:
                    st.warning(f"Skipping {vtu}: {e}")
                    continue
            if timesteps:
                self.source_db.append({
                    'name': folder_name,
                    'E': E,
                    'tau': tau,
                    'timesteps': np.array(timesteps),
                    'max_temperature': np.array(max_temperature),
                    'mean_displacement': np.array(mean_displacement),
                    'max_von_mises': np.array(max_von_mises),
                    'strain_energy': np.array(strain_energy)
                })

    def _physics_embedding(self, E, tau, t):
        logE = np.log1p(E)
        logE_norm = (logE - np.log1p(0.1)) / (np.log1p(20.0) - np.log1p(0.1))
        tau_norm = (tau - 0.5) / (10.0 - 0.5)
        t_rel = t / max(tau, 1e-3)
        Lambda = t * 0.1
        return np.array([logE_norm, tau_norm, t_rel, Lambda], dtype=np.float32)

    def predict_at_time(self, E_query, tau_query, t_query):
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
        for sim in self.source_db:
            for i, t_src in enumerate(sim['timesteps']):
                src_emb = self._physics_embedding(sim['E'], sim['tau'], t_src)
                dist_sq = np.sum((query_emb - src_emb) ** 2)
                weight = np.exp(-dist_sq / (2 * self.sigma_param ** 2))
                weights.append(weight)
                for key in values:
                    values[key].append(sim[key][i])
        if not weights or sum(weights) == 0:
            return {
                'max_temperature': np.nan,
                'mean_displacement': np.nan,
                'max_von_mises': np.nan,
                'strain_energy': np.nan,
                'confidence': 0.0
            }
        weights = np.array(weights, dtype=np.float32)
        weights /= np.sum(weights)
        prediction = {}
        for key in values:
            prediction[key] = float(np.sum(weights * np.array(values[key], dtype=np.float32)))
        min_dist = min(
            np.sum((query_emb - self._physics_embedding(sim['E'], sim['tau'], t_src)) ** 2)
            for sim in self.source_db for t_src in sim['timesteps']
        )
        prediction['confidence'] = float(np.exp(-min_dist / (2 * self.sigma_param ** 2)))
        return prediction

    def predict_time_series(self, E_query, tau_query, t_list):
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
def main():
    st.set_page_config(page_title="Integrated FEA Laser Viewer & Extrapolator", layout="wide")
    st.title("🔍 Integrated FEA Laser Simulation Viewer & Extrapolator")
    st.caption(f"Using data from: `{FEA_SOLUTIONS_DIR}`")

    # Initialize extrapolator in session state
    if 'extrapolator' not in st.session_state:
        st.session_state.extrapolator = IntegratedFEALaserExtrapolator()
        st.session_state.extrapolator.load_all_data()
    extrapolator = st.session_state.extrapolator

    if not extrapolator.simulations and not extrapolator.source_db:
        st.error("No valid simulations found.")
        st.write("### Expected Directory Structure:")
        st.code("""
your_app/
└── fea_solutions/
    ├── q0p5mJ-delta4p2ns/
    │ ├── a_t0001.vtu
    │ ├── a_t0002.vtu
    │ └── ...
    └── q2p0mJ-delta2p0ns/
        ├── a_t0001.vtu
        └── ...
""")
        return

    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Source Visualization", "Extrapolation", "Advanced Charts"])

    with tab1:
        st.subheader("Source Simulation Visualization")
        if extrapolator.simulations:
            sim_name = st.selectbox("Select simulation", sorted(extrapolator.simulations))
            sim = extrapolator.simulations[sim_name]
            st.write(f"**Energy:** {sim['energy_mJ']} mJ")
            st.write(f"**Pulse Duration:** {sim['duration_ns']} ns")
            field = st.selectbox("Select field", list(sim["fields"].keys()))
            timestep = st.slider("Timestep (ns)", 0, sim["n_timesteps"] - 1, 0)
            pts = sim["points"]
            kind, _ = sim["field_info"][field]
            raw = sim["fields"][field][timestep]
            if kind == "scalar":
                values = raw
                label = field
            else:
                values = np.linalg.norm(raw, axis=1)
                label = f"{field} (magnitude)"
            st.subheader(f"{label} at {timestep + 1} ns – {sim_name}")
            if sim["triangles"] is not None:
                tri = sim["triangles"]
                data = go.Mesh3d(
                    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                    i=tri[:, 0], j=tri[:, 1], k=tri[:, 2],
                    intensity=values, colorscale="Viridis", intensitymode='vertex',
                    colorbar=dict(title=label), opacity=0.85,
                    lighting=dict(ambient=0.8, diffuse=0.6, specular=0.5, roughness=0.5, fresnel=0.2),
                    lightposition=dict(x=100, y=200, z=150),
                    hovertemplate='Value: %{intensity:.2f}<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}',
                )
            else:
                data = go.Mesh3d(
                    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                    alphahull=3, intensity=values, colorscale="Viridis", intensitymode='vertex',
                    colorbar=dict(title=label), opacity=0.85,
                    lighting=dict(ambient=0.8, diffuse=0.6, specular=0.5, roughness=0.5, fresnel=0.2),
                    lightposition=dict(x=100, y=200, z=150),
                    hovertemplate='Value: %{intensity:.2f}<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}',
                )
            fig = go.Figure(data=data)
            fig.update_layout(height=700, scene=dict(aspectmode="data", camera_eye=dict(x=1.5, y=1.5, z=1.5)),
                              margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("🖼️ 2D Projection (Matplotlib)"):
                fig2, ax = plt.subplots(figsize=(6, 5))
                sc = ax.scatter(pts[:, 0], pts[:, 1], c=values, s=5, cmap="viridis")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_title(label)
                plt.colorbar(sc, ax=ax)
                st.pyplot(fig2)

            with st.expander("📊 Field Statistics"):
                clean = values[~np.isnan(values)]
                st.write({
                    "Min": float(clean.min()),
                    "Max": float(clean.max()),
                    "Mean": float(clean.mean()),
                    "Std": float(clean.std()),
                })
        else:
            st.warning("No raw simulations loaded (meshio required).")

    with tab2:
        st.subheader("🎯 Query Parameters for Extrapolation")
        col1, col2, col3 = st.columns(3)
        with col1:
            E_query = st.number_input("Energy (mJ)", min_value=0.1, max_value=50.0, value=15.0, step=0.5)
        with col2:
            tau_query = st.number_input("Pulse On Duration (ns)", min_value=0.5, max_value=20.0, value=5.0, step=0.5)
        with col3:
            max_time = st.number_input("Max Prediction Time (ns)", min_value=1, max_value=50, value=20, step=1)
        t_list = np.arange(1, max_time + 1)
        if st.button("🚀 Predict Extrapolated Response"):
            with st.spinner("Computing..."):
                try:
                    results = extrapolator.predict_time_series(E_query, tau_query, t_list)
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
                    last_few = min(5, len(results['confidence']))
                    avg_conf = np.nanmean(results['confidence'][-last_few:])
                    st.metric("Prediction Confidence (last {} ns)".format(last_few), f"{avg_conf:.2f}")
                    if avg_conf < 0.3:
                        st.warning("⚠️ Low confidence: query is far from training data")
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

    with tab3:
        st.subheader("Advanced Visualizations")
        if 'results' not in locals():
            st.info("Run a prediction in the Extrapolation tab to view advanced charts.")
        else:
            # Data Curves: Multi-line plot for attributes
            st.subheader("Data Curves")
            fig_curves = go.Figure()
            for key, color in zip(['max_temperature', 'mean_displacement', 'max_von_mises', 'strain_energy'],
                                  ['red', 'green', 'blue', 'purple']):
                fig_curves.add_trace(go.Scatter(x=t_list, y=results[key], mode='lines', name=key.replace('_', ' ').title(), line=dict(color=color)))
            # Add source data for comparison (e.g., all sources)
            for sim in extrapolator.source_db:
                fig_curves.add_trace(go.Scatter(x=sim['timesteps'], y=sim['max_temperature'], mode='lines', name=f"Source {sim['name']} Temp", opacity=0.5))
            fig_curves.update_layout(title="Predicted and Source Attributes Over Time", xaxis_title="Time (ns)", yaxis_title="Value")
            st.plotly_chart(fig_curves, use_container_width=True)

            # Sunburst Chart: Hierarchical view of simulations
            st.subheader("Sunburst Chart (Simulation Hierarchy)")
            labels = ["Simulations"]
            parents = [""]
            values = [0]
            for sim in extrapolator.source_db:
                e_str = f"E={sim['E']}"
                tau_str = f"tau={sim['tau']}"
                labels.extend([e_str, tau_str])
                parents.extend(["Simulations", e_str])
                values.extend([0, sim['max_temperature'].max()])  # Example value: max temp
            fig_sunburst = go.Figure(go.Sunburst(labels=labels, parents=parents, values=values, branchvalues="total"))
            fig_sunburst.update_layout(margin=dict(t=0, l=0, r=0, b=0))
            st.plotly_chart(fig_sunburst, use_container_width=True)

            # Radar Chart: Multi-attribute at selected timestep
            st.subheader("Radar Chart (Attribute Comparison)")
            t_select = st.slider("Select Timestep for Radar", 1, int(max_time), 1)
            pred_at_t = extrapolator.predict_at_time(E_query, tau_query, t_select)
            categories = ['Max Temp', 'Mean Disp', 'Max VM Stress', 'Strain Energy']
            # Normalize values (simple min-max across all for demo)
            all_vals = np.array([pred_at_t[k] for k in ['max_temperature', 'mean_displacement', 'max_von_mises', 'strain_energy']])
            norm_vals = (all_vals - all_vals.min()) / (all_vals.max() - all_vals.min() + 1e-6)
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=norm_vals, theta=categories, fill='toself', name='Predicted'))
            # Add a source for comparison (e.g., first source at closest t)
            if extrapolator.source_db:
                sim0 = extrapolator.source_db[0]
                closest_i = np.argmin(np.abs(sim0['timesteps'] - t_select))
                src_vals = np.array([sim0[k][closest_i] for k in ['max_temperature', 'mean_displacement', 'max_von_mises', 'strain_energy']])
                norm_src = (src_vals - src_vals.min()) / (src_vals.max() - src_vals.min() + 1e-6)
                fig_radar.add_trace(go.Scatterpolar(r=norm_src, theta=categories, fill='toself', name=f"Source {sim0['name']}"))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
            st.plotly_chart(fig_radar, use_container_width=True)

    with st.expander("📘 Theoretical Background", expanded=False):
        st.markdown("""
## Physics-Informed Attention Extrapolation
This tool combines visualization and extrapolation of laser-material simulations.
- Raw data loading and viz via Meshio/Plotly.
- Summarization and attention-based interp/extrap via PyVista and Gaussian kernel.
- Advanced charts for comparison.
""")

if __name__ == "__main__":
    main()

