import streamlit as st
import os
import glob
import re
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import meshio
from datetime import datetime
import warnings
from collections import OrderedDict
from typing import List, Dict, Any, Optional, Tuple

warnings.filterwarnings('ignore')

# =============================================
# CONSTANTS
# =============================================
EXTENDED_COLORMAPS = [
    'Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis',
    'Rainbow', 'Jet', 'Hot', 'Cool', 'Portland',
    'Bluered', 'Electric', 'Thermal', 'Balance', 'Teal', 'Sunset', 'Burg',
    'Greys', 'YlOrRd', 'Blues', 'Reds', 'Greens', 'Purples', 'Oranges'
]

# =============================================
# PATH CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEA_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "fea_solutions")
os.makedirs(FEA_SOLUTIONS_DIR, exist_ok=True)

# =============================================
# ERROR LOGGING
# =============================================
if 'error_log' not in st.session_state:
    st.session_state.error_log = []

def log_warning(msg: str):
    st.session_state.error_log.append(f"{datetime.now()}: {msg}")

# =============================================
# UNIFIED DATA LOADER (WITH CACHE FIX)
# =============================================
@st.cache_data(show_spinner="Loading simulation data...")
def load_all_simulations(fea_solutions_dir: str, load_full_mesh: bool = True, max_points: int = 50000):
    """Standalone cached function to avoid session state issues."""
    simulations = {}
    summaries = []
    folders = glob.glob(os.path.join(fea_solutions_dir, "q*mJ-delta*ns"))
    
    if not folders:
        st.warning(f"No simulation folders found in {fea_solutions_dir}")
        return simulations, summaries

    def parse_folder_name(folder: str):
        match = re.match(r"q([\dp\.]+)mJ-delta([\dp\.]+)ns", folder)
        if not match:
            return None, None
        e, d = match.groups()
        return float(e.replace("p", ".")), float(d.replace("p", "."))

    progress_bar = st.progress(0)
    status_text = st.empty()

    for folder_idx, folder in enumerate(folders):
        name = os.path.basename(folder)
        energy, duration = parse_folder_name(name)
        if energy is None:
            continue

        vtu_files = sorted(glob.glob(os.path.join(folder, "a_t????.vtu")))
        if not vtu_files:
            continue

        status_text.text(f"Loading {name}... ({len(vtu_files)} files)")
        try:
            mesh0 = meshio.read(vtu_files[0])
            if not mesh0.point_data:
                log_warning(f"No point data in {name}")
                continue

            points = mesh0.points.astype(np.float32)
            n_pts = len(points)

            # Decimate points if needed
            if load_full_mesh and max_points and n_pts > max_points:
                idx = np.random.choice(n_pts, max_points, replace=False)
                points = points[idx]
                # Remap triangles? Simpler: fallback to scatter if decimated
                triangles = None
                log_warning(f"Decimated {name} from {n_pts} to {max_points} points")
            else:
                triangles = None
                for cell_block in mesh0.cells:
                    if cell_block.type == "triangle":
                        triangles = cell_block.data.astype(np.int32)
                        if load_full_mesh and max_points and n_pts > max_points:
                            # Keep only triangles with all indices in decimated set
                            mask = np.isin(triangles, idx).all(axis=1)
                            triangles = triangles[mask]
                            # Remap indices
                            remap = {old: new for new, old in enumerate(idx)}
                            triangles = np.vectorize(remap.get)(triangles)
                        break

            fields = {}
            field_info = {}
            for key in mesh0.point_data.keys():
                arr = mesh0.point_data[key].astype(np.float32)
                if load_full_mesh and max_points and n_pts > max_points:
                    arr = arr[idx]
                if arr.ndim == 1:
                    field_info[key] = ("scalar", 1)
                    fields[key] = np.full((len(vtu_files), len(arr)), np.nan, dtype=np.float32)
                else:
                    field_info[key] = ("vector", arr.shape[1])
                    fields[key] = np.full((len(vtu_files), len(arr), arr.shape[1]), np.nan, dtype=np.float32)
                fields[key][0] = arr

            for t in range(1, len(vtu_files)):
                try:
                    mesh = meshio.read(vtu_files[t])
                    for key in field_info:
                        if key in mesh.point_data:
                            val = mesh.point_data[key].astype(np.float32)
                            if load_full_mesh and max_points and n_pts > max_points:
                                val = val[idx]
                            fields[key][t] = val
                except Exception as e:
                    log_warning(f"Error loading timestep {t} in {name}: {e}")

            sim_data = {
                'name': name, 'energy_mJ': energy, 'duration_ns': duration,
                'n_timesteps': len(vtu_files), 'vtu_files': vtu_files,
                'field_info': field_info, 'has_mesh': load_full_mesh,
                'points': points, 'fields': fields, 'triangles': triangles
            }
            
            summary = {
                'name': name, 'energy': energy, 'duration': duration,
                'timesteps': list(range(1, len(vtu_files) + 1)), 'field_stats': {}
            }
            for field in field_info:
                vals = fields[field]
                if field_info[field][0] == "vector":
                    mag_vals = np.linalg.norm(vals, axis=2)
                    summary['field_stats'][field] = {
                        'min': float(np.nanmin(mag_vals)),
                        'max': float(np.nanmax(mag_vals)),
                        'mean': float(np.nanmean(mag_vals)),
                        'std': float(np.nanstd(mag_vals))
                    }
                else:
                    summary['field_stats'][field] = {
                        'min': float(np.nanmin(vals)),
                        'max': float(np.nanmax(vals)),
                        'mean': float(np.nanmean(vals)),
                        'std': float(np.nanstd(vals))
                    }

            simulations[name] = sim_data
            summaries.append(summary)
        except Exception as e:
            log_warning(f"Error loading {name}: {str(e)}")
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
# ENHANCED SUNBURST VISUALIZER
# =============================================
def create_sunburst_chart(
    summaries: List[Dict],
    selected_field: str,
    hierarchy: List[str] = ['energy', 'duration', 'name'],  # configurable
    cmaps: Dict[str, str] = None,  # per level colormap
    norm_mode: str = 'global',  # 'global', 'per_parent', 'percentile'
    percentile_clip: Tuple[float, float] = (2, 98),
    label_formats: Dict[str, str] = None,  # e.g., {'energy': 'E = {value:.1f} mJ', ...}
    show_labels: Dict[str, bool] = None,
    font_sizes: Dict[str, int] = None,
    highlight_sim: Optional[str] = None,
    leaf_value_as_label: bool = True,
    colorbar_side: str = 'split'  # 'split' -> left/right, else 'right'
):
    """
    Flexible sunburst with configurable hierarchy, per-level colormaps,
    label formatting, and visibility toggles.
    """
    # Defaults
    if cmaps is None:
        cmaps = {'root': 'Greys', 'energy': 'YlOrRd', 'duration': 'Blues', 'name': 'Viridis'}
    if label_formats is None:
        label_formats = {
            'root': 'All Simulations',
            'energy': 'E = {value:.1f} mJ',
            'duration': 'τ = {value:.1f} ns',
            'name': '{value:.3f}'  # will be peak value
        }
    if show_labels is None:
        show_labels = {'root': True, 'energy': True, 'duration': True, 'name': True}
    if font_sizes is None:
        font_sizes = {'root': 16, 'energy': 14, 'duration': 12, 'name': 10}

    # Build tree and collect values
    all_energies = []
    all_durations = []
    all_peaks = []
    tree = {}
    sim_index = 0
    for s in summaries:
        e = s['energy']
        d = s['duration']
        sim = s['name']
        peak = float(s.get('field_stats', {}).get(selected_field, {}).get('max', 0))
        if peak <= 0:
            peak = 1e-6
        all_energies.append(e)
        all_durations.append(d)
        all_peaks.append(peak)
        tree.setdefault(e, {}).setdefault(d, {})[sim] = (peak, sim_index)
        sim_index += 1

    if not all_energies:
        return go.Figure()

    # Normalization helpers
    def norm_val(v, vmin, vmax):
        return max(0.0, min(1.0, (v - vmin) / (vmax - vmin))) if vmax > vmin else 0.5

    if norm_mode == 'percentile':
        e_min, e_max = np.percentile(all_energies, percentile_clip[0]), np.percentile(all_energies, percentile_clip[1])
        d_min, d_max = np.percentile(all_durations, percentile_clip[0]), np.percentile(all_durations, percentile_clip[1])
        p_min, p_max = np.percentile(all_peaks, percentile_clip[0]), np.percentile(all_peaks, percentile_clip[1])
    else:  # global
        e_min, e_max = min(all_energies), max(all_energies)
        d_min, d_max = min(all_durations), max(all_durations)
        p_min, p_max = min(all_peaks), max(all_peaks)

    # Sample colormaps
    cmap_root = px.colors.sample_colorscale(cmaps.get('root', 'Greys'), 101)
    cmap_energy = px.colors.sample_colorscale(cmaps.get('energy', 'YlOrRd'), 101)
    cmap_duration = px.colors.sample_colorscale(cmaps.get('duration', 'Blues'), 101)
    cmap_leaf = px.colors.sample_colorscale(cmaps.get('name', 'Viridis'), 101)

    labels, parents, ids, values, colors, levels = [], [], [], [], [], []
    e_indices = {}
    d_indices = {}

    # Root
    root_id = "root"
    ids.append(root_id)
    root_label = label_formats.get('root', 'All Simulations')
    labels.append(root_label if show_labels.get('root', True) else "")
    parents.append("")
    values.append(0)
    colors.append(cmap_root[50])
    levels.append(0)

    # Energy level
    sorted_energies = sorted(set(all_energies))
    for e in sorted_energies:
        e_id = f"E_{e}"
        e_label = label_formats.get('energy', 'E = {value:.1f} mJ').format(value=e)
        ids.append(e_id)
        labels.append(e_label if show_labels.get('energy', True) else "")
        parents.append(root_id)
        values.append(0)
        colors.append(cmap_energy[int(norm_val(e, e_min, e_max) * 100)])
        levels.append(1)
        e_indices[e] = len(labels) - 1

        if e not in tree:
            continue
        sorted_durations = sorted(tree[e].keys())
        for d in sorted_durations:
            d_id = f"{e_id}_D_{d}"
            d_label = label_formats.get('duration', 'τ = {value:.1f} ns').format(value=d)
            ids.append(d_id)
            labels.append(d_label if show_labels.get('duration', True) else "")
            parents.append(e_id)
            values.append(0)
            colors.append(cmap_duration[int(norm_val(d, d_min, d_max) * 100)])
            levels.append(2)
            d_indices[(e, d)] = len(labels) - 1

            for sim, (peak, idx) in tree[e][d].items():
                leaf_id = f"{d_id}_S_{idx}"
                if leaf_value_as_label:
                    leaf_label = label_formats.get('name', '{value:.3f}').format(value=peak)
                else:
                    leaf_label = sim
                ids.append(leaf_id)
                labels.append(leaf_label if show_labels.get('name', True) else "")
                parents.append(d_id)
                values.append(peak)
                colors.append(cmap_leaf[int(norm_val(peak, p_min, p_max) * 100)])
                levels.append(3)

    # Upward aggregation
    # Leaves already have values
    for (e, d), d_idx in d_indices.items():
        if e in tree and d in tree[e]:
            d_sum = sum(tree[e][d].values(), 0)[0]  # sum peaks
            values[d_idx] = d_sum
    for e, e_idx in e_indices.items():
        e_sum = 0
        if e in tree:
            for d in tree[e]:
                d_idx = d_indices.get((e, d))
                if d_idx is not None:
                    e_sum += values[d_idx]
        values[e_idx] = e_sum
    values[0] = sum(values[e_idx] for e_idx in e_indices.values())

    # Build figure
    fig = go.Figure(go.Sunburst(
        ids=ids, labels=labels, parents=parents, values=values,
        branchvalues="total",
        marker=dict(colors=colors, line=dict(color='white', width=1.5)),
        textfont=dict(size=[font_sizes.get(lvl, 12) for lvl in ['root','energy','duration','name'] for _ in range(levels.count(0) if lvl=='root' else levels.count(1) if lvl=='energy' else levels.count(2) if lvl=='duration' else levels.count(3))],
        hovertemplate='<b>%{label}</b><br>Value: %{value:.3f}<extra></extra>'
    ))

    # Colorbars
    colorbar_data = []
    if colorbar_side == 'split':
        # Left: root and energy
        colorbar_data.append(("FEM Root", cmaps.get('root', 'Greys'), 0, 1, 0.96, 'left'))
        colorbar_data.append(("Energy (mJ)", cmaps.get('energy', 'YlOrRd'), e_min, e_max, 0.74, 'left'))
        # Right: duration and field
        colorbar_data.append(("Pulse Width (ns)", cmaps.get('duration', 'Blues'), d_min, d_max, 0.51, 'right'))
        colorbar_data.append((f"Peak {selected_field}", cmaps.get('name', 'Viridis'), p_min, p_max, 0.28, 'right'))
    else:
        # all on right
        y_positions = [0.96, 0.74, 0.51, 0.28]
        for (title, cmap, cmin, cmax, ypos), side in zip(colorbar_data, ['right']*4):
            colorbar_data.append((title, cmap, cmin, cmax, ypos, side))

    for title, cmap, cmin, cmax, y_pos, side in colorbar_data:
        fig.add_trace(go.Scatter(
            x=[0], y=[0], mode='markers',
            marker=dict(
                showscale=True,
                colorbar=dict(
                    title=title,
                    title_font=dict(size=10),
                    tickfont=dict(size=9),
                    len=0.18,
                    thickness=14,
                    x=1.01 if side == 'right' else -0.05,
                    y=y_pos,
                    xanchor='left' if side == 'right' else 'right',
                    yanchor='top',
                    outlinewidth=0,
                    bordercolor="rgba(0,0,0,0.1)",
                    borderwidth=1
                ),
                colorscale=cmap,
                cmin=cmin,
                cmax=cmax
            ),
            showlegend=False,
            marker=dict(size=0, opacity=0)  # invisible
        ))

    return fig

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

    st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; background: linear-gradient(90deg, #1E88E5, #4A00E0); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; margin-bottom: 1.5rem; font-weight: 800; }
    .sub-header { font-size: 1.5rem; color: #2c3e50; margin-top: 1rem; margin-bottom: 0.5rem; border-bottom: 2px solid #3498db; padding-bottom: 0.3rem; font-weight: 600; }
    .control-box { background: #f8f9fa; border-left: 4px solid #1E88E5; padding: 0.8rem; margin: 0.5rem 0; border-radius: 4px; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">📊 FEA Data Viewer</h1>', unsafe_allow_html=True)

    # Session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'simulations' not in st.session_state:
        st.session_state.simulations = {}
    if 'summaries' not in st.session_state:
        st.session_state.summaries = []

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Data Settings")
        load_full_mesh = st.checkbox("Load Full Mesh", value=True, help="Load complete mesh data for 3D visualization")
        max_points = st.number_input("Max points per mesh (decimation)", min_value=1000, max_value=200000, value=50000, step=5000, disabled=not load_full_mesh)
        default_colormap = st.selectbox("Default Colormap (3D View)", EXTENDED_COLORMAPS, index=0, key="global_colormap")

        if st.button("🔄 Load All Simulations", type="primary", use_container_width=True):
            with st.spinner("Loading simulation data..."):
                sims, sums = load_all_simulations(FEA_SOLUTIONS_DIR, load_full_mesh=load_full_mesh, max_points=max_points)
                st.session_state.simulations = sims
                st.session_state.summaries = sums
                st.session_state.data_loaded = bool(sims)

        if st.session_state.data_loaded:
            st.markdown("---")
            st.markdown("### 📈 Loaded Data")
            st.metric("Simulations", len(st.session_state.simulations))
            if st.session_state.summaries:
                energies = [s['energy'] for s in st.session_state.summaries]
                st.metric("Energy Range", f"{min(energies):.1f} - {max(energies):.1f} mJ")

        # Error log expander
        if st.session_state.error_log:
            with st.expander("⚠️ Error/Warning Log"):
                for msg in st.session_state.error_log[-10:]:
                    st.text(msg)

    if not st.session_state.data_loaded:
        st.info("Click 'Load All Simulations' in the sidebar to begin.")
        return

    render_data_viewer(st.session_state.get('global_colormap', 'Viridis'))

def render_data_viewer(selected_colormap):
    st.markdown('<h2 class="sub-header">📁 Data Viewer</h2>', unsafe_allow_html=True)
    
    simulations = st.session_state.simulations
    summaries = st.session_state.summaries
    if not simulations:
        return

    # Simulation selection
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

    # 3D Visualization Controls
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

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        opacity = st.slider("🔹 Opacity", 0.0, 1.0, 0.9, 0.05, key="opacity")
    with col2:
        point_size = st.slider("🔹 Point Size", 1, 15, 4, key="point_size")
    with col3:
        camera_preset = st.selectbox("📷 Camera View", ["Isometric", "Front", "Side", "Top", "Bottom"], index=0, key="camera_preset")
    with col4:
        lighting_preset = st.selectbox("💡 Lighting", ["Default", "Shiny", "Matte", "High Contrast"], index=0, key="lighting_preset")

    # Theme & lighting
    lighting_map = {
        "Default": dict(ambient=0.8, diffuse=0.8, specular=0.5, roughness=0.5),
        "Shiny": dict(ambient=0.6, diffuse=0.9, specular=0.8, roughness=0.2),
        "Matte": dict(ambient=0.9, diffuse=0.6, specular=0.1, roughness=0.9),
        "High Contrast": dict(ambient=0.4, diffuse=0.9, specular=0.7, roughness=0.4)
    }
    lighting = lighting_map[lighting_preset]

    camera_map = {
        "Isometric": dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        "Front": dict(eye=dict(x=0, y=2, z=0.1)),
        "Side": dict(eye=dict(x=2, y=0, z=0.1)),
        "Top": dict(eye=dict(x=0, y=0, z=2)),
        "Bottom": dict(eye=dict(x=0, y=0, z=-2))
    }
    camera = camera_map[camera_preset]

    if bg_mode == "Dark":
        plot_bgcolor, paper_bgcolor, grid_color, font_color = "rgb(17,17,17)", "rgb(17,17,17)", "rgb(40,40,40)", "white"
    else:
        plot_bgcolor, paper_bgcolor, grid_color, font_color = "white", "white", "lightgray", "black"

    # Data processing
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

    # Color scale limits
    st.markdown('<h5 class="sub-header">🌈 Color Scale Limits</h5>', unsafe_allow_html=True)
    data_min, data_max = float(np.min(values)), float(np.max(values))
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        auto_scale = st.checkbox("Auto Scale", value=True, key=f"auto_scale_{field}")
    with col_b:
        cmin = st.number_input("Min Limit", value=data_min, format="%.3f", disabled=auto_scale, key=f"cmin_{field}")
    with col_c:
        cmax = st.number_input("Max Limit", value=data_max, format="%.3f", disabled=auto_scale, key=f"cmax_{field}")
    
    if auto_scale:
        cmin, cmax = None, None

    # Build 3D trace
    tri = sim.get('triangles')
    trace_data = None
    
    if tri is not None and len(tri) > 0:
        valid_triangles = tri[np.all(tri < len(pts), axis=1)]
        if len(valid_triangles) > 0:
            trace_data = go.Mesh3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                i=valid_triangles[:, 0], j=valid_triangles[:, 1], k=valid_triangles[:, 2],
                intensity=values, colorscale=selected_colormap, intensitymode='vertex',
                cmin=cmin, cmax=cmax, opacity=opacity, lighting=lighting,
                hovertemplate=f'<b>{label}:</b> %{{intensity:.3f}}<br><b>X:</b> %{{x:.3f}}<br><b>Y:</b> %{{y:.3f}}<br><b>Z:</b> %{{z:.3f}}<extra></extra>'
            )
    
    if trace_data is None:
        trace_data = go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode='markers',
            marker=dict(size=point_size, color=values, colorscale=selected_colormap, cmin=cmin, cmax=cmax, opacity=opacity),
            hovertemplate=f'<b>{label}:</b> %{{marker.color:.3f}}<br><b>X:</b> %{{x:.3f}}<br><b>Y:</b> %{{y:.3f}}<br><b>Z:</b> %{{z:.3f}}<extra></extra>'
        )

    fig = go.Figure(data=trace_data)
    fig.update_layout(
        title=dict(text=f"{label} at Timestep {timestep + 1}", font=dict(size=18, color=font_color)),
        scene=dict(
            aspectmode=aspect_mode, camera=camera,
            xaxis=dict(showbackground=True, backgroundcolor=plot_bgcolor, gridcolor=grid_color, color=font_color, title="X"),
            yaxis=dict(showbackground=True, backgroundcolor=plot_bgcolor, gridcolor=grid_color, color=font_color, title="Y"),
            zaxis=dict(showbackground=True, backgroundcolor=plot_bgcolor, gridcolor=grid_color, color=font_color, title="Z")
        ),
        plot_bgcolor=plot_bgcolor, paper_bgcolor=paper_bgcolor, height=700, margin=dict(l=0, r=0, t=50, b=0)
    )
    
    for trace in fig.data:
        if hasattr(trace, 'colorbar') and trace.colorbar:
            trace.colorbar.title.font.color = font_color
            trace.colorbar.tickfont.color = font_color

    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("Min", f"{np.min(values):.3f}")
    with col2: st.metric("Max", f"{np.max(values):.3f}")
    with col3: st.metric("Mean", f"{np.mean(values):.3f}")
    with col4: st.metric("Std Dev", f"{np.std(values):.3f}")
    with col5: st.metric("Range", f"{np.max(values) - np.min(values):.3f}")

    # ================= SUNBURST SECTION =================
    st.markdown('<h2 class="sub-header">🌳 Comparative Sunburst – Peak Values</h2>', unsafe_allow_html=True)
    st.markdown("Hierarchy: **FEM Root → Energy → Pulse Duration → Simulation** (configurable)")

    if not summaries:
        st.warning("No summary data available.")
        return

    all_fields = set()
    for s in summaries:
        all_fields.update(s.get('field_stats', {}).keys())
    available_fields = sorted(all_fields)

    if not available_fields:
        st.warning("No fields found for sunburst.")
        return

    # Sunburst controls
    col1, col2 = st.columns([3, 2])
    with col1:
        sun_field = st.selectbox("Select Field", available_fields, 
                            index=available_fields.index('temperature') if 'temperature' in available_fields else 0,
                            key="sunburst_field")
        highlight_sim = st.selectbox("Highlight Simulation (optional)", 
                                     ["None"] + sorted(simulations.keys()), 
                                     key="sunburst_highlight")
    with col2:
        norm_mode = st.selectbox("Color Normalization", ['global', 'percentile'], index=0, key="norm_mode")
        percentile_clip = st.slider("Percentile Clip (low, high)", 0, 100, (2, 98), step=1, disabled=(norm_mode!='percentile'))

    # Hierarchy colormaps
    st.markdown("### 🎨 Hierarchy Colormaps")
    c_cm1, c_cm2, c_cm3, c_cm4 = st.columns(4)
    with c_cm1:
        cmap_root = st.selectbox("Root Colormap", EXTENDED_COLORMAPS, index=EXTENDED_COLORMAPS.index('Greys') if 'Greys' in EXTENDED_COLORMAPS else 0, key="cmap_root")
    with c_cm2:
        cmap_energy = st.selectbox("Energy Colormap", EXTENDED_COLORMAPS, index=EXTENDED_COLORMAPS.index('YlOrRd') if 'YlOrRd' in EXTENDED_COLORMAPS else 0, key="cmap_energy")
    with c_cm3:
        cmap_tau = st.selectbox("Pulse Width Colormap", EXTENDED_COLORMAPS, index=EXTENDED_COLORMAPS.index('Blues') if 'Blues' in EXTENDED_COLORMAPS else 0, key="cmap_tau")
    with c_cm4:
        cmap_field_sim = st.selectbox("Simulation Colormap", EXTENDED_COLORMAPS, 
                                    index=EXTENDED_COLORMAPS.index(selected_colormap) if selected_colormap in EXTENDED_COLORMAPS else 0,
                                    key="cmap_field_sim")

    # Label customization
    st.markdown("### 🏷️ Label Customization (per hierarchy)")
    col_l1, col_l2, col_l3, col_l4 = st.columns(4)
    with col_l1:
        show_root = st.checkbox("Show Root Label", True, key="show_root")
        root_label = st.text_input("Root Label Format", "All Simulations", key="root_label")
    with col_l2:
        show_energy = st.checkbox("Show Energy Labels", True, key="show_energy")
        energy_label = st.text_input("Energy Label Format", "E = {value:.1f} mJ", key="energy_label")
    with col_l3:
        show_tau = st.checkbox("Show Pulse Width Labels", True, key="show_tau")
        tau_label = st.text_input("Pulse Width Format", "τ = {value:.1f} ns", key="tau_label")
    with col_l4:
        show_sim = st.checkbox("Show Simulation Labels", True, key="show_sim")
        sim_label = st.text_input("Simulation Label Format (value only)", "{value:.3f}", key="sim_label")

    # Font sizes
    st.markdown("### 🖋️ Label Font Sizes")
    c_fs1, c_fs2, c_fs3, c_fs4 = st.columns(4)
    with c_fs1:
        fs_root = st.slider("FEM Root", 10, 24, 16, key="fs_root")
    with c_fs2:
        fs_energy = st.slider("Energy (E=)", 10, 22, 14, key="fs_energy")
    with c_fs3:
        fs_tau = st.slider("Pulse (τ=)", 10, 20, 12, key="fs_tau")
    with c_fs4:
        fs_sim = st.slider("Simulation", 8, 18, 10, key="fs_sim")

    # Build cmaps dict
    cmaps_dict = {
        'root': cmap_root,
        'energy': cmap_energy,
        'duration': cmap_tau,
        'name': cmap_field_sim
    }
    label_formats = {
        'root': root_label,
        'energy': energy_label,
        'duration': tau_label,
        'name': sim_label
    }
    show_labels_dict = {
        'root': show_root,
        'energy': show_energy,
        'duration': show_tau,
        'name': show_sim
    }
    font_sizes_dict = {
        'root': fs_root,
        'energy': fs_energy,
        'duration': fs_tau,
        'name': fs_sim
    }

    fig_sun = create_sunburst_chart(
        summaries,
        sun_field,
        hierarchy=['energy', 'duration', 'name'],
        cmaps=cmaps_dict,
        norm_mode=norm_mode,
        percentile_clip=percentile_clip,
        label_formats=label_formats,
        show_labels=show_labels_dict,
        font_sizes=font_sizes_dict,
        highlight_sim=highlight_sim if highlight_sim != "None" else None,
        leaf_value_as_label=True,
        colorbar_side='split'
    )
    st.plotly_chart(fig_sun, use_container_width=True)

    # Side-by-side comparison (refactored)
    st.markdown("### Compare Two Fields Side-by-Side")
    if len(available_fields) >= 2 and st.checkbox("Show two sunbursts", key="show_compare"):
        col_l, col_r = st.columns(2)
        with col_l:
            f1 = st.selectbox("Left Field", available_fields, key="sun_f1")
            c1 = st.selectbox("Left Colormap", EXTENDED_COLORMAPS, index=3, key="sun_c1")
        with col_r:
            f2 = st.selectbox("Right Field", available_fields, index=1, key="sun_f2")
            c2 = st.selectbox("Right Colormap", EXTENDED_COLORMAPS, index=1, key="sun_c2")
        
        cmaps_left = cmaps_dict.copy()
        cmaps_left['name'] = c1
        cmaps_right = cmaps_dict.copy()
        cmaps_right['name'] = c2

        fig1 = create_sunburst_chart(summaries, f1, cmaps=cmaps_left, norm_mode=norm_mode, percentile_clip=percentile_clip,
                                     label_formats=label_formats, show_labels=show_labels_dict, font_sizes=font_sizes_dict,
                                     leaf_value_as_label=True, colorbar_side='split')
        fig2 = create_sunburst_chart(summaries, f2, cmaps=cmaps_right, norm_mode=norm_mode, percentile_clip=percentile_clip,
                                     label_formats=label_formats, show_labels=show_labels_dict, font_sizes=font_sizes_dict,
                                     leaf_value_as_label=True, colorbar_side='split')
        col_l.plotly_chart(fig1, use_container_width=True)
        col_r.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    main()
