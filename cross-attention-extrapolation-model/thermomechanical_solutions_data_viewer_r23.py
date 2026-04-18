import streamlit as st
import os
import glob
import re
import numpy as np
import plotly.graph_objects as go
import meshio
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================
# PATH CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEA_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "fea_solutions")
os.makedirs(FEA_SOLUTIONS_DIR, exist_ok=True)

# =============================================
# UNIFIED DATA LOADER
# =============================================
class UnifiedFEADataLoader:
    """Enhanced data loader for FEA simulations"""
    def __init__(self):
        self.simulations = {}
        self.summaries = []
        self.available_fields = set()

    def parse_folder_name(self, folder: str):
        """q0p5mJ-delta4p2ns → (0.5, 4.2)"""
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
                    st.warning(f"No point data in {name}")
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
                
                # Build summary with global max across all timesteps for each field
                summary = {
                    'name': name, 'energy': energy, 'duration': duration,
                    'timesteps': list(range(1, len(vtu_files) + 1)), 'field_stats': {}
                }
                for field in field_info:
                    vals = fields[field]
                    if field_info[field][0] == "vector":
                        mag_vals = np.linalg.norm(vals, axis=2)
                        global_max = float(np.nanmax(mag_vals))
                    else:
                        global_max = float(np.nanmax(vals))
                    summary['field_stats'][field] = {
                        'global_max': global_max
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
# SUNBURST CHART HELPER FUNCTIONS
# =============================================
def build_sunburst_data(summaries, field_name):
    """
    Build hierarchical data for sunburst chart.
    Hierarchy: All Simulations → Pulse Duration (τ) → Energy (E) → Simulation → Field Peak
    Returns: labels, parents, values, numeric_values_for_coloring
    """
    labels = []
    parents = []
    values = []
    numeric_colors = []  # for leaf nodes (field peak values)

    # Root
    labels.append("All Simulations")
    parents.append("")
    values.append(len(summaries))
    numeric_colors.append(None)

    # Group by pulse duration (τ)
    duration_groups = {}
    for s in summaries:
        tau_key = f"τ: {s['duration']:.1f} ns"
        duration_groups.setdefault(tau_key, []).append(s)

    for tau_key, tau_sims in sorted(duration_groups.items()):
        labels.append(tau_key)
        parents.append("All Simulations")
        values.append(len(tau_sims))
        numeric_colors.append(None)

        # Group by energy under this duration
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
                numeric_colors.append(None)  # simulation node has no numeric value

                # Leaf: field peak value
                peak_val = s['field_stats'].get(field_name, {}).get('global_max', 0.0)
                leaf_label = f"{field_name}: {peak_val:.2f}"
                labels.append(leaf_label)
                parents.append(sim_label)
                # Use the peak value for the leaf wedge size (must be positive)
                values.append(max(peak_val, 1e-6))
                numeric_colors.append(peak_val)

    return labels, parents, values, numeric_colors

def create_sunburst_figure(summaries, field_name, colormap_name, highlight_sim=None):
    """Create a Plotly Sunburst figure for a given field."""
    labels, parents, values, num_colors = build_sunburst_data(summaries, field_name)
    
    import plotly.express as px
    
    # Get the colorscale as a list of 101 colors
    if colormap_name in px.colors.named_colorscales():
        colorscale = px.colors.sample_colorscale(colormap_name, [i/100 for i in range(101)])
    else:
        colorscale = px.colors.sample_colorscale("Viridis", [i/100 for i in range(101)])
    
    # Determine the range of leaf values (excluding None)
    leaf_vals = [v for v in num_colors if v is not None]
    if leaf_vals:
        vmin, vmax = min(leaf_vals), max(leaf_vals)
    else:
        vmin, vmax = 0, 1
    
    # Build color list for all wedges
    color_list = []
    for val in num_colors:
        if val is None:
            color_list.append("#CCCCCC")  # light gray for interior nodes
        else:
            # Normalize value to [0,1] within the leaf range
            if vmax > vmin:
                norm = (val - vmin) / (vmax - vmin)
            else:
                norm = 0.5
            idx = int(norm * 100)
            idx = min(idx, 100)
            color_list.append(colorscale[idx])
    
    # Highlight selected simulation if requested
    if highlight_sim and highlight_sim != "None":
        for i, lbl in enumerate(labels):
            if lbl == highlight_sim:
                color_list[i] = "red"  # simulation node
            elif parents[i] == highlight_sim:
                color_list[i] = "red"  # leaf nodes (field peaks)
    
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(colors=color_list),
        branchvalues="total",
        hovertemplate='<b>%{label}</b><br>Value: %{value:.3f}<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text=f"Peak {field_name} (max over all timesteps)", font=dict(size=16)),
        height=600,
        margin=dict(t=40, l=10, r=10, b=10)
    )
    return fig

# =============================================
# MAIN APPLICATION
# =============================================
def main():
    st.set_page_config(
        page_title="FEA Data Viewer with Sunburst",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="📊"
    )

    st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; background: linear-gradient(90deg, #1E88E5, #4A00E0); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; margin-bottom: 1.5rem; font-weight: 800; }
    .sub-header { font-size: 1.5rem; color: #2c3e50; margin-top: 1rem; margin-bottom: 0.5rem; border-bottom: 2px solid #3498db; padding-bottom: 0.3rem; font-weight: 600; }
    .control-box { background: #f8f9fa; border-left: 4px solid #1E88E5; padding: 0.8rem; margin: 0.5rem 0; border-radius: 4px; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">📊 FEA Data Viewer with Sunburst</h1>', unsafe_allow_html=True)

    # Session State Initialization
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
        st.markdown("### ⚙️ Data Settings")
        load_full_data = st.checkbox("Load Full Mesh", value=True, help="Load complete mesh data for 3D visualization")
        
        extended_colormaps = [
            'Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis',
            'Rainbow', 'Jet', 'Hot', 'Cool', 'Portland',
            'Bluered', 'Electric', 'Thermal', 'Balance', 'Teal', 
            'Sunset', 'Burg', 'Sunsetdark'
        ]
        # Store colormap list in session state for global access
        st.session_state.extended_colormaps = extended_colormaps
        
        selected_colormap = st.selectbox("Global Colormap (for 3D plots)", extended_colormaps, index=0, key="global_colormap")

        if st.button("🔄 Load All Simulations", type="primary", use_container_width=True):
            with st.spinner("Loading simulation data..."):
                simulations, summaries = st.session_state.data_loader.load_all_simulations(load_full_mesh=load_full_data)
                st.session_state.simulations = simulations
                st.session_state.summaries = summaries
                st.session_state.data_loaded = bool(simulations)

        if st.session_state.data_loaded:
            st.markdown("---")
            st.markdown("### 📈 Loaded Data")
            st.metric("Simulations", len(st.session_state.simulations))
            if st.session_state.summaries:
                energies = [s['energy'] for s in st.session_state.summaries]
                st.metric("Energy Range", f"{min(energies):.1f} - {max(energies):.1f} mJ")

    # Main Content
    if not st.session_state.data_loaded:
        st.markdown("""
        <div class="control-box" style="border-left-color: #ff9800; background: #fff3e0;">
            <h3>⚠️ No Data Loaded</h3>
            <p>Please load simulations using the "Load All Simulations" button in the sidebar.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    render_data_viewer(st.session_state.get('global_colormap', 'Viridis'))

def render_data_viewer(selected_colormap):
    st.markdown('<h2 class="sub-header">📁 Data Viewer</h2>', unsafe_allow_html=True)
    
    simulations = st.session_state.simulations
    summaries = st.session_state.summaries
    if not simulations:
        return

    # Simulation Selection for 3D view (unchanged)
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        sim_name = st.selectbox("Select Simulation for 3D View", sorted(simulations.keys()), key="viewer_sim_select")
    sim = simulations[sim_name]
    with col2: st.metric("Energy", f"{sim['energy_mJ']:.2f} mJ")
    with col3: st.metric("Duration", f"{sim['duration_ns']:.2f} ns")

    if not sim.get('has_mesh', False):
        st.error("This simulation was loaded without mesh data. Please reload with 'Load Full Mesh' enabled.")
        # Still show sunburst even if mesh missing
    else:
        if 'field_info' not in sim or not sim['field_info']:
            st.error("No field data available for this simulation.")
        else:
            # ================= 3D VISUALIZATION CONTROLS (simplified) =================
            st.markdown('<h4 class="sub-header">🎛️ 3D Visualization Controls</h4>', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                field = st.selectbox("Select Field", sorted(sim['field_info'].keys()), key="viewer_field_select")
            with col2:
                timestep = st.slider("Timestep", 0, sim['n_timesteps'] - 1, 0, key="viewer_timestep_slider")
            with col3:
                aspect_mode = st.selectbox("Aspect Ratio", ["data", "cube", "auto"], index=0, key="aspect_mode")
            with col4:
                bg_mode = st.selectbox("Plot Theme", ["Light", "Dark"], index=0, key="bg_mode")

            # Data processing for 3D plot
            pts = sim['points']
            kind, _ = sim['field_info'][field]
            raw = sim['fields'][field][timestep]
            if kind == "scalar":
                values = np.where(np.isnan(raw), 0, raw)
                label = field
            else:
                magnitude = np.linalg.norm(raw, axis=1)
                values = np.where(np.isnan(magnitude), 0, magnitude)
                label = f"{field} (magnitude)"

            # Theme setup
            if bg_mode == "Dark":
                plot_bgcolor, paper_bgcolor, grid_color, font_color = "rgb(17,17,17)", "rgb(17,17,17)", "rgb(40,40,40)", "white"
            else:
                plot_bgcolor, paper_bgcolor, grid_color, font_color = "white", "white", "lightgray", "black"

            # Build 3D trace
            tri = sim.get('triangles')
            if tri is not None and len(tri) > 0:
                valid_triangles = tri[np.all(tri < len(pts), axis=1)]
                if len(valid_triangles) > 0:
                    trace_data = go.Mesh3d(
                        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                        i=valid_triangles[:, 0], j=valid_triangles[:, 1], k=valid_triangles[:, 2],
                        intensity=values, colorscale=selected_colormap, intensitymode='vertex',
                        opacity=0.9,
                        hovertemplate=f'<b>{label}:</b> %{{intensity:.3f}}<extra></extra>'
                    )
                else:
                    trace_data = go.Scatter3d(
                        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                        mode='markers', marker=dict(size=4, color=values, colorscale=selected_colormap, opacity=0.9)
                    )
            else:
                trace_data = go.Scatter3d(
                    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                    mode='markers', marker=dict(size=4, color=values, colorscale=selected_colormap, opacity=0.9)
                )

            fig = go.Figure(data=trace_data)
            fig.update_layout(
                title=dict(text=f"{label} at Timestep {timestep + 1}", font=dict(size=18, color=font_color)),
                scene=dict(aspectmode=aspect_mode,
                           xaxis=dict(backgroundcolor=plot_bgcolor, gridcolor=grid_color, color=font_color, title="X"),
                           yaxis=dict(backgroundcolor=plot_bgcolor, gridcolor=grid_color, color=font_color, title="Y"),
                           zaxis=dict(backgroundcolor=plot_bgcolor, gridcolor=grid_color, color=font_color, title="Z")),
                plot_bgcolor=plot_bgcolor, paper_bgcolor=paper_bgcolor, height=600, margin=dict(l=0, r=0, t=50, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)

    # ================= SUNBURST CHARTS SECTION =================
    st.markdown('<h2 class="sub-header">🌳 Hierarchical Sunburst – Peaks over all timesteps</h2>', unsafe_allow_html=True)
    st.markdown("""
    This chart aggregates the **maximum peak** of each field across **all time steps** for every simulation.
    The hierarchy is: **All Simulations → Pulse Duration (τ) → Energy (E) → Simulation → Field Peak**.
    """)

    # Get list of available fields from summaries
    all_fields = set()
    for s in summaries:
        all_fields.update(s.get('field_stats', {}).keys())
    available_fields = sorted(all_fields)

    if len(available_fields) < 2:
        st.warning("Need at least two fields to display two sunburst charts.")
    else:
        extended_colormaps = st.session_state.get('extended_colormaps', [
            'Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis',
            'Rainbow', 'Jet', 'Hot', 'Cool', 'Portland',
            'Bluered', 'Electric', 'Thermal', 'Balance', 'Teal', 
            'Sunset', 'Burg', 'Sunsetdark'
        ])
        
        col_left, col_right = st.columns(2)
        with col_left:
            field1 = st.selectbox("Left Sunburst Field", available_fields, index=0, key="sunburst_field1")
            colormap1 = st.selectbox(
                f"Colormap for {field1}", 
                extended_colormaps, 
                index=extended_colormaps.index("Thermal") if "Thermal" in extended_colormaps else 0,
                key="sunburst_cmap1"
            )
        with col_right:
            default_idx = 1 if len(available_fields) > 1 else 0
            field2 = st.selectbox("Right Sunburst Field", available_fields, index=default_idx, key="sunburst_field2")
            colormap2 = st.selectbox(
                f"Colormap for {field2}", 
                extended_colormaps, 
                index=extended_colormaps.index("Plasma") if "Plasma" in extended_colormaps else 0,
                key="sunburst_cmap2"
            )

        highlight_sim = st.selectbox("Highlight a specific simulation (optional)", ["None"] + sorted(simulations.keys()), key="sunburst_highlight")

        if st.button("Generate Sunburst Charts", type="primary", use_container_width=True):
            with st.spinner("Building sunburst charts..."):
                fig1 = create_sunburst_figure(summaries, field1, colormap1, highlight_sim)
                fig2 = create_sunburst_figure(summaries, field2, colormap2, highlight_sim)
                col_left.plotly_chart(fig1, use_container_width=True)
                col_right.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    main()
