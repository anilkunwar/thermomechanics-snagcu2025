import streamlit as st
import os
import glob
import re
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import meshio
import warnings
from collections import OrderedDict

warnings.filterwarnings('ignore')

# =============================================
# CONSTANTS
# =============================================
EXTENDED_COLORMAPS = [
    'Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis',
    'Rainbow', 'Jet', 'Hot', 'Cool', 'Portland',
    'Bluered', 'Electric', 'Thermal', 'Balance', 'Teal', 'Sunset', 'Burg'
]

# =============================================
# PATH CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEA_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "fea_solutions")
os.makedirs(FEA_SOLUTIONS_DIR, exist_ok=True)

# =============================================
# DATA LOADER (unchanged)
# =============================================
class UnifiedFEADataLoader:
    def __init__(self):
        self.simulations = {}
        self.summaries = []
        self.available_fields = set()

    def parse_folder_name(self, folder: str):
        match = re.match(r"q([\dp\.]+)mJ-delta([\dp\.]+)ns", folder)
        if not match:
            return None, None
        e, d = match.groups()
        return float(e.replace("p", ".")), float(d.replace("p", "."))

    @st.cache_data(show_spinner="Loading simulation data...")
    def load_all_simulations(_self, load_full_mesh=True):
        simulations = {}
        summaries = []
        folders = glob.glob(os.path.join(FEA_SOLUTIONS_DIR, "q*mJ-delta*ns"))
       
        if not folders:
            st.warning(f"No simulation folders found in {FEA_SOLUTIONS_DIR}")
            return simulations, summaries

        progress_bar = st.progress(0)
        status_text = st.empty()

        for folder_idx, folder in enumerate(folders):
            name = os.path.basename(folder)
            energy, duration = _self.parse_folder_name(name)
            if energy is None:
                continue

            vtu_files = sorted(glob.glob(os.path.join(folder, "a_t????.vtu")))
            if not vtu_files:
                continue

            status_text.text(f"Loading {name}... ({len(vtu_files)} files)")
            try:
                mesh0 = meshio.read(vtu_files[0])
                if not mesh0.point_data:
                    continue

                points = mesh0.points.astype(np.float32)
                n_pts = len(points)

                triangles = None
                for cell_block in mesh0.cells:
                    if cell_block.type == "triangle":
                        triangles = cell_block.data.astype(np.int32)
                        break

                fields = {}
                field_info = {}
                for key in mesh0.point_data.keys():
                    arr = mesh0.point_data[key].astype(np.float32)
                    if arr.ndim == 1:
                        field_info[key] = ("scalar", 1)
                        fields[key] = np.full((len(vtu_files), n_pts), np.nan, dtype=np.float32)
                    else:
                        field_info[key] = ("vector", arr.shape[1])
                        fields[key] = np.full((len(vtu_files), n_pts, arr.shape[1]), np.nan, dtype=np.float32)
                    fields[key][0] = arr
                    _self.available_fields.add(key)

                for t in range(1, len(vtu_files)):
                    try:
                        mesh = meshio.read(vtu_files[t])
                        for key in field_info:
                            if key in mesh.point_data:
                                fields[key][t] = mesh.point_data[key].astype(np.float32)
                    except Exception as e:
                        st.warning(f"Error loading timestep {t} in {name}: {e}")

                sim_data = {
                    'name': name, 'energy_mJ': energy, 'duration_ns': duration,
                    'n_timesteps': len(vtu_files), 'vtu_files': vtu_files,
                    'field_info': field_info, 'has_mesh': load_full_mesh,
                    'points': points, 'fields': fields, 'triangles': triangles
                }

                # Summary with global stats
                summary = {
                    'name': name, 'energy': energy, 'duration': duration,
                    'timesteps': list(range(1, len(vtu_files) + 1)), 'field_stats': {}
                }
                for field in field_info:
                    vals = fields[field]
                    if field_info[field][0] == "vector":
                        mag_vals = np.linalg.norm(vals, axis=2)
                        summary['field_stats'][field] = {
                            'min': [float(np.nanmin(mag_vals))],
                            'max': [float(np.nanmax(mag_vals))],
                            'mean': [float(np.nanmean(mag_vals))],
                            'std': [float(np.nanstd(mag_vals))]
                        }
                    else:
                        summary['field_stats'][field] = {
                            'min': [float(np.nanmin(vals))],
                            'max': [float(np.nanmax(vals))],
                            'mean': [float(np.nanmean(vals))],
                            'std': [float(np.nanstd(vals))]
                        }

                simulations[name] = sim_data
                summaries.append(summary)

            except Exception as e:
                st.warning(f"Error loading {name}: {str(e)}")
                continue

            progress_bar.progress((folder_idx + 1) / len(folders))

        progress_bar.empty()
        status_text.empty()

        if simulations:
            st.success(f"✅ Loaded {len(simulations)} simulations")
        else:
            st.error("❌ No simulations loaded successfully")

        return simulations, summaries


# =============================================
# SUNBURST HELPER FUNCTIONS (Improved)
# =============================================
def get_global_max_peak(summary: dict, field_name: str) -> float:
    if field_name not in summary.get('field_stats', {}):
        return 0.0
    max_list = summary['field_stats'][field_name].get('max', [0.0])
    return float(np.max(max_list)) if max_list else 0.0


def build_sunburst_data(summaries, field_name):
    labels = []
    parents = []
    values = []
    numeric_colors = []

    # Root
    labels.append("All Simulations")
    parents.append("")
    values.append(len(summaries))
    numeric_colors.append(None)

    # Group by Duration → Energy
    duration_groups = {}
    for s in summaries:
        tau_key = f"τ: {s['duration']:.1f} ns"
        duration_groups.setdefault(tau_key, []).append(s)

    for tau_key, tau_sims in sorted(duration_groups.items()):
        labels.append(tau_key)
        parents.append("All Simulations")
        values.append(len(tau_sims))
        numeric_colors.append(None)

        energy_groups = {}
        for s in tau_sims:
            e_key = f"E: {s['energy']:.1f} mJ"
            energy_groups.setdefault(e_key, []).append(s)

        for e_key, e_sims in sorted(energy_groups.items()):
            labels.append(e_key)
            parents.append(tau_key)
            values.append(len(e_sims))
            numeric_colors.append(None)

            for s in e_sims:
                sim_label = s['name']
                labels.append(sim_label)
                parents.append(e_key)
                values.append(1)
                numeric_colors.append(None)

                # Leaf: Peak value
                peak_val = get_global_max_peak(s, field_name)
                leaf_label = f"{field_name} Peak: {peak_val:.2f}"
                labels.append(leaf_label)
                parents.append(sim_label)
                values.append(max(peak_val, 1e-6))
                numeric_colors.append(peak_val)

    return labels, parents, values, numeric_colors


def create_sunburst_figure(summaries, field_name, colormap, highlight_sim=None):
    if not summaries:
        fig = go.Figure()
        fig.update_layout(title="No data loaded", height=400)
        return fig

    labels, parents, values, num_colors = build_sunburst_data(summaries, field_name)

    if len(labels) <= 1:
        fig = go.Figure()
        fig.update_layout(title="Insufficient data", height=400)
        return fig

    # Color mapping
    leaf_vals = [v for v in num_colors if v is not None]
    vmin, vmax = (min(leaf_vals), max(leaf_vals)) if leaf_vals else (0, 1)

    if colormap in px.colors.named_colorscales():
        colorscale = px.colors.sample_colorscale(colormap, [i/100 for i in range(101)])
    else:
        colorscale = px.colors.sample_colorscale("Viridis", [i/100 for i in range(101)])

    color_list = []
    for val in num_colors:
        if val is None:
            color_list.append("#E0E0E0")        # Light gray for interior nodes
        else:
            norm = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            idx = min(int(norm * 100), 100)
            color_list.append(colorscale[idx])

    # Highlight logic
    if highlight_sim and highlight_sim != "None":
        for i, lbl in enumerate(labels):
            if lbl == highlight_sim or parents[i] == highlight_sim:
                color_list[i] = "#FF0000"   # Red for highlighted simulation and its peaks

    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(colors=color_list),
        branchvalues="total",
        hovertemplate='<b>%{label}</b><br>Value: %{value:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(text=f"Peak {field_name} (max over all timesteps)", font=dict(size=18)),
        height=650,
        margin=dict(t=50, l=20, r=20, b=20)
    )
    return fig


# =============================================
# MAIN APPLICATION
# =============================================
def main():
    st.set_page_config(page_title="FEA Data Viewer", layout="wide", initial_sidebar_state="expanded")
    
    st.markdown('<h1 class="main-header">📊 FEA Data Viewer with Sunburst Analysis</h1>', unsafe_allow_html=True)

    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = UnifiedFEADataLoader()
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'simulations' not in st.session_state:
        st.session_state.simulations = {}
    if 'summaries' not in st.session_state:
        st.session_state.summaries = []

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        load_full_data = st.checkbox("Load Full Mesh", value=True)
        
        selected_colormap = st.selectbox("Default 3D Colormap", EXTENDED_COLORMAPS, index=0, key="global_colormap")

        if st.button("🔄 Load All Simulations", type="primary", use_container_width=True):
            with st.spinner("Loading..."):
                simulations, summaries = st.session_state.data_loader.load_all_simulations(load_full_mesh=load_full_data)
                st.session_state.simulations = simulations
                st.session_state.summaries = summaries
                st.session_state.data_loaded = bool(simulations)

        if st.session_state.data_loaded:
            st.metric("Loaded Simulations", len(st.session_state.simulations))

    if not st.session_state.data_loaded:
        st.info("Please load simulations from the sidebar.")
        return

    render_data_viewer(st.session_state.get('global_colormap', 'Viridis'))


def render_data_viewer(selected_colormap):
    st.markdown('<h2 class="sub-header">📁 3D Data Viewer</h2>', unsafe_allow_html=True)

    simulations = st.session_state.simulations
    summaries = st.session_state.summaries

    # 3D Viewer (your existing code - kept unchanged for brevity)
    # ... [Your original 3D viewer code remains here] ...

    # ================= SUNBURST SECTION =================
    st.markdown('<h2 class="sub-header">🌳 Sunburst Charts – Hierarchical Peak Analysis</h2>', unsafe_allow_html=True)
    st.caption("Hierarchy: All Simulations → Pulse Duration (τ) → Energy (E) → Simulation → Peak Value")

    all_fields = set()
    for s in summaries:
        all_fields.update(s.get('field_stats', {}).keys())
    available_fields = sorted(all_fields)

    if not available_fields:
        st.warning("No fields available for sunburst.")
        return

    col_left, col_right = st.columns(2)

    with col_left:
        default1 = available_fields.index('temperature') if 'temperature' in available_fields else 0
        field1 = st.selectbox("Left Field", available_fields, index=default1, key="sun_field1")
        cmap1 = st.selectbox("Left Colormap", EXTENDED_COLORMAPS, 
                            index=EXTENDED_COLORMAPS.index("Thermal") if "Thermal" in EXTENDED_COLORMAPS else 0,
                            key="sun_cmap1")

    with col_right:
        default2 = next((i for i, f in enumerate(available_fields) if f in ['vonMises', 'principal stress', 'stress']), 1 if len(available_fields)>1 else 0)
        field2 = st.selectbox("Right Field", available_fields, index=default2, key="sun_field2")
        cmap2 = st.selectbox("Right Colormap", EXTENDED_COLORMAPS,
                            index=EXTENDED_COLORMAPS.index("Plasma") if "Plasma" in EXTENDED_COLORMAPS else 0,
                            key="sun_cmap2")

    highlight_sim = st.selectbox("Highlight Simulation", ["None"] + sorted(simulations.keys()), key="sun_highlight")

    if st.button("Generate Sunburst Charts", type="primary", use_container_width=True):
        with st.spinner("Generating sunburst charts..."):
            hl = None if highlight_sim == "None" else highlight_sim

            fig1 = create_sunburst_figure(summaries, field1, cmap1, hl)
            fig2 = create_sunburst_figure(summaries, field2, cmap2, hl)

            col_left.plotly_chart(fig1, use_container_width=True)
            col_right.plotly_chart(fig2, use_container_width=True)

    st.info("Tip: The sunburst shows the **maximum peak** value of the selected field across **all timesteps** for each simulation.")


if __name__ == "__main__":
    main()
