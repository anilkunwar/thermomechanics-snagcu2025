import streamlit as st
import os
import glob
import re
import numpy as np
import plotly.graph_objects as go
import meshio
from datetime import datetime
import warnings
from collections import OrderedDict
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
                
                # Find triangles
                triangles = None
                for cell_block in mesh0.cells:
                    if cell_block.type == "triangle":
                        triangles = cell_block.data.astype(np.int32)
                        break

                # Initialize fields
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

                # Load remaining timesteps
                for t in range(1, len(vtu_files)):
                    try:
                        mesh = meshio.read(vtu_files[t])
                        for key in field_info:
                            if key in mesh.point_data:
                                fields[key][t] = mesh.point_data[key].astype(np.float32)
                    except Exception as e:
                        st.warning(f"Error loading timestep {t} in {name}: {e}")

                sim_data = {
                    'name': name,
                    'energy_mJ': energy,
                    'duration_ns': duration,
                    'n_timesteps': len(vtu_files),
                    'vtu_files': vtu_files,
                    'field_info': field_info,
                    'has_mesh': load_full_mesh,
                    'points': points,
                    'fields': fields,
                    'triangles': triangles
                }
                
                # Create summary
                summary = {
                    'name': name,
                    'energy': energy,
                    'duration': duration,
                    'timesteps': list(range(1, len(vtu_files) + 1)),
                    'field_stats': {} # Simplified summary for viewer
                }
                # Populate basic stats
                for field in field_info:
                    summary['field_stats'][field] = {
                        'min': [float(np.nanmin(fields[field][:, :]))],
                        'max': [float(np.nanmax(fields[field][:, :]))],
                        'mean': [float(np.nanmean(fields[field][:, :]))],
                        'std': [float(np.nanstd(fields[field][:, :]))]
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
# MAIN APPLICATION
# =============================================
def main():
    st.set_page_config(
        page_title="FEA Data Viewer",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="📊"
    )

    # CSS Styling
    st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; background: linear-gradient(90deg, #1E88E5, #4A00E0); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; margin-bottom: 1.5rem; font-weight: 800; }
    .sub-header { font-size: 1.5rem; color: #2c3e50; margin-top: 1rem; margin-bottom: 0.5rem; border-bottom: 2px solid #3498db; padding-bottom: 0.3rem; font-weight: 600; }
    .success-box { background: #e8f5e9; border-left: 4px solid #4CAF50; padding: 1rem; margin: 1rem 0; border-radius: 4px; }
    .warning-box { background: #fff3e0; border-left: 4px solid #ff9800; padding: 1rem; margin: 1rem 0; border-radius: 4px; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">📊 FEA Data Viewer</h1>', unsafe_allow_html=True)

    # Session State Initialization
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = UnifiedFEADataLoader()
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'simulations' not in st.session_state:
        st.session_state.simulations = {}
    if 'summaries' not in st.session_state:
        st.session_state.summaries = []
    if 'color_auto_scale' not in st.session_state:
        st.session_state.color_auto_scale = True
    if 'cmin_override' not in st.session_state:
        st.session_state.cmin_override = 0.0
    if 'cmax_override' not in st.session_state:
        st.session_state.cmax_override = 1.0

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Data Settings")
        load_full_data = st.checkbox("Load Full Mesh", value=True, help="Load complete mesh data for 3D visualization")
        
        # Extended colormap options
        extended_colormaps = [
            'Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis',
            'Rainbow', 'Jet', 'Hot', 'Cool', 'Portland',
            'Bluered', 'Electric', 'Thermal', 'Balance',
            'Teal', 'Sunset', 'Burg'
        ]
        selected_colormap = st.selectbox("Colormap", extended_colormaps, index=0)

        if st.button("🔄 Load All Simulations", type="primary", use_container_width=True):
            with st.spinner("Loading simulation data..."):
                simulations, summaries = st.session_state.data_loader.load_all_simulations(load_full_mesh=load_full_data)
                st.session_state.simulations = simulations
                st.session_state.summaries = summaries
                st.session_state.data_loaded = bool(simulations)
                # Reset color scale on load
                st.session_state.color_auto_scale = True

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
        <div class="warning-box">
            <h3>⚠️ No Data Loaded</h3>
            <p>Please load simulations using the "Load All Simulations" button in the sidebar.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    render_data_viewer(selected_colormap)

def render_data_viewer(selected_colormap):
    st.markdown('<h2 class="sub-header">📁 Data Viewer</h2>', unsafe_allow_html=True)
    
    simulations = st.session_state.simulations
    if not simulations:
        return

    # Simulation Selection
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        sim_name = st.selectbox("Select Simulation", sorted(simulations.keys()), key="viewer_sim_select")
    sim = simulations[sim_name]
    
    with col2: st.metric("Energy", f"{sim['energy_mJ']:.2f} mJ")
    with col3: st.metric("Duration", f"{sim['duration_ns']:.2f} ns")

    if not sim.get('has_mesh', False):
        st.error("This simulation was loaded without mesh data. Please reload with 'Load Full Mesh' enabled.")
        return
    if 'field_info' not in sim or not sim['field_info']:
        st.error("No field data available.")
        return

    # Controls: Field, Timestep
    col1, col2 = st.columns(2)
    with col1:
        field = st.selectbox("Select Field", sorted(sim['field_info'].keys()), key="viewer_field_select")
    with col2:
        timestep = st.slider("Timestep", 0, sim['n_timesteps'] - 1, 0, key="viewer_timestep_slider")

    # Get Data
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

    # ==========================================
    # CUSTOM COLOR SCALE LIMITS
    # ==========================================
    st.markdown('<h4 class="sub-header">🎨 Color Scale Customization</h4>', unsafe_allow_html=True)
    
    # Calculate data range for default values
    data_min = float(np.min(values))
    data_max = float(np.max(values))

    # Reset button
    if st.button("🔄 Reset to Auto Scale", use_container_width=True):
        st.session_state.color_auto_scale = True
        st.rerun()

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        auto_scale = st.checkbox("Auto Scale", value=st.session_state.color_auto_scale, key="cb_auto_scale")
    with col_b:
        # If auto scale is ON, show disabled inputs with current auto limits, else show editable
        is_editable = not auto_scale
        cmin_val = st.session_state.cmin_override if not auto_scale else data_min
        cmin = st.number_input("Min Limit", value=cmin_val, format="%.3f", disabled=auto_scale, key="num_cmin")
    with col_c:
        cmax_val = st.session_state.cmax_override if not auto_scale else data_max
        cmax = st.number_input("Max Limit", value=cmax_val, format="%.3f", disabled=auto_scale, key="num_cmax")

    # Update session state
    if not auto_scale:
        st.session_state.color_auto_scale = False
        st.session_state.cmin_override = cmin
        st.session_state.cmax_override = cmax
    else:
        st.session_state.color_auto_scale = True
        cmin = None
        cmax = None

    # Create 3D Visualization
    mesh_data = None
    tri = sim.get('triangles')
    
    if tri is not None and len(tri) > 0:
        valid_triangles = tri[np.all(tri < len(pts), axis=1)]
        if len(valid_triangles) > 0:
            mesh_data = go.Mesh3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                i=valid_triangles[:, 0], j=valid_triangles[:, 1], k=valid_triangles[:, 2],
                intensity=values,
                colorscale=selected_colormap,
                intensitymode='vertex',
                cmin=cmin,  # Applied custom min
                cmax=cmax,  # Applied custom max
                colorbar=dict(title=dict(text=label, font=dict(size=12)), thickness=20, len=0.75),
                opacity=0.9,
                lighting=dict(ambient=0.8, diffuse=0.8, specular=0.5, roughness=0.5),
                hovertemplate='<b>Value:</b> %{intensity:.3f}<extra></extra>'
            )
    
    if mesh_data is None:
        mesh_data = go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode='markers',
            marker=dict(size=4, color=values, colorscale=selected_colormap, cmin=cmin, cmax=cmax, showscale=True),
            hovertemplate='<b>Value:</b> %{marker.color:.3f}<extra></extra>'
        )

    fig = go.Figure(data=mesh_data)
    fig.update_layout(
        title=f"{label} at Timestep {timestep + 1}",
        scene=dict(aspectmode="data", camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))),
        height=700,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Statistics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("Min", f"{np.min(values):.3f}")
    with col2: st.metric("Max", f"{np.max(values):.3f}")
    with col3: st.metric("Mean", f"{np.mean(values):.3f}")
    with col4: st.metric("Std Dev", f"{np.std(values):.3f}")
    with col5: st.metric("Range", f"{np.max(values) - np.min(values):.3f}")

if __name__ == "__main__":
    main()
