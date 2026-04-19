import streamlit as st
import os
import glob
import re
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import meshio
import io
import warnings
from datetime import datetime
from collections import OrderedDict
from PIL import Image

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
# HELPER FOR SAFE STRING FORMATTING
# =============================================
def safe_format(fmt, **kwargs):
    try:
        return fmt.format(**kwargs)
    except Exception:
        return fmt

# =============================================
# ENHANCED SUNBURST VISUALIZER (SPLIT COLORBARS & CUSTOM LABELS)
# =============================================
def create_sunburst_chart(summaries, selected_field, 
                          cmap_root='Greys', cmap_energy='YlOrRd', cmap_tau='Blues', cmap_field='Viridis',
                          highlight_sim=None,
                          font_size_root=16, font_size_energy=14, font_size_tau=12, font_size_sim=10,
                          show_labels_root=True, show_labels_energy=True, show_labels_tau=True, show_labels_sim=True,
                          fmt_root="All Simulations as FEM",
                          fmt_energy="E = {energy:.1f} mJ",
                          fmt_tau="τ = {tau:.1f} ns",
                          fmt_sim="Peak: {peak:.3f}"):
    """
    Fully customizable Sunburst with:
    - Independent colormaps per hierarchy
    - Split colorbars (2 left, 2 right)
    - Custom label formatting per level
    - Label visibility toggles per level
    - Leaf nodes display actual peak values instead of filenames
    - Strict ID mapping & branchvalues="total" compliance
    """
    labels, parents, ids, values, colors, levels = [], [], [], [], [], []
    e_indices = {}
    d_indices = {}
    
    # 1. Root Node
    ids.append("root")
    root_lbl = safe_format(fmt_root) if show_labels_root else ""
    labels.append(root_lbl)
    parents.append("") 
    values.append(0)   
    colors.append("#E0E0E0") 
    levels.append(0)
    root_idx = 0
    
    # 2. Data Collection for Normalization
    all_energies = [s['energy'] for s in summaries]
    all_taus = [s['duration'] for s in summaries]
    all_peaks = []
    
    tree = {}
    for s in summaries:
        e = s['energy']
        d = s['duration']
        sim = s['name']
        peak = float(s.get('field_stats', {}).get(selected_field, {}).get('max', [0])[0])
        if peak <= 0: peak = 1e-3
        all_peaks.append(peak)
        tree.setdefault(e, {}).setdefault(d, {})[sim] = peak

    if not all_energies: all_energies = [0]
    if not all_taus: all_taus = [0]
    if not all_peaks: all_peaks = [0]

    e_min, e_max = min(all_energies), max(all_energies)
    d_min, d_max = min(all_taus), max(all_taus)
    p_min, p_max = min(all_peaks), max(all_peaks)

    # 3. Independent Color Map Sampling
    cmap_root_colors = px.colors.sample_colorscale(cmap_root, 101)
    cmap_e = px.colors.sample_colorscale(cmap_energy, 101)
    cmap_d = px.colors.sample_colorscale(cmap_tau, 101)
    cmap_p = px.colors.sample_colorscale(cmap_field, 101)

    def norm_val(v, vmin, vmax):
        return max(0.0, min(1.0, (v - vmin) / (vmax - vmin))) if vmax > vmin else 0.5

    # 4. Hierarchy Construction
    sorted_energies = sorted(set(all_energies))
    colors[root_idx] = cmap_root_colors[50]
    
    for e in sorted_energies:
        e_lbl = safe_format(fmt_energy, energy=e) if show_labels_energy else f"E = {e:.1f} mJ"
        e_id = f"E_{e}"
        e_idx = len(labels)
        
        # Add Energy Node
        ids.append(e_id)
        labels.append(e_lbl)
        parents.append("root")
        values.append(0)
        colors.append(cmap_e[int(norm_val(e, e_min, e_max) * 100)])
        levels.append(1)
        e_indices[e] = e_idx
        
        # Iterate only taus present for this energy
        if e in tree:
            sorted_taus_for_e = sorted(tree[e].keys())
            
            for d in sorted_taus_for_e:
                d_lbl = safe_format(fmt_tau, tau=d) if show_labels_tau else f"τ = {d:.1f} ns"
                d_id = f"{e_id}_T_{d}"
                d_idx = len(labels)
                
                # Add Tau Node
                ids.append(d_id)
                labels.append(d_lbl)
                parents.append(e_id)
                values.append(0)
                colors.append(cmap_d[int(norm_val(d, d_min, d_max) * 100)])
                levels.append(2)
                d_indices[(e, d)] = d_idx
                
                sims_for_d = tree[e][d]
                for sim, peak in sims_for_d.items():
                    # Add Simulation Node (Leaf)
                    s_lbl = safe_format(fmt_sim, peak=peak) if show_labels_sim else f"{peak:.3f}"
                    s_id = f"{d_id}_S_{sim}"
                    s_idx = len(labels)
                    
                    ids.append(s_id)
                    labels.append(s_lbl)
                    parents.append(d_id)
                    values.append(peak)
                    colors.append(cmap_p[int(norm_val(peak, p_min, p_max) * 100)])
                    levels.append(3)

    # 5. CRITICAL FIX: Upward Aggregation (Strict branchvalues="total" compliance)
    for (e, d), d_idx in d_indices.items():
        if e in tree and d in tree[e]:
            values[d_idx] = sum(tree[e][d].values())
            
    for e, e_idx in e_indices.items():
        e_sum = sum(values[d_indices.get((e, d), 0)] for d in tree[e] if (e, d) in d_indices)
        values[e_idx] = e_sum
        
    values[root_idx] = sum(values[e_idx] for e_idx in e_indices.values())

    # 6. Font Size Assignment
    font_sizes = []
    for l in levels:
        if l == 0: font_sizes.append(font_size_root)
        elif l == 1: font_sizes.append(font_size_energy)
        elif l == 2: font_sizes.append(font_size_tau)
        else: font_sizes.append(font_size_sim)

    # 7. Create Figure
    fig = go.Figure(go.Sunburst(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        marker=dict(colors=colors, line=dict(color='white', width=1.5)),
        textfont=dict(size=font_sizes, family="sans-serif"),
        hovertemplate='<b>%{label}</b><br>Value: %{value:.3f}<extra></extra>'
    ))
    
    # 8. Layout & Split Colorbars
    fig.update_layout(
        margin=dict(l=150, r=150, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)"
    )

    # 4 Independent Colorbars: 2 Left, 2 Right
    colorbar_positions = [
        # Left Side
        ("Root", cmap_root, 0, 1, -0.12, 0.88, 'right'),
        ("Energy (mJ)", cmap_energy, e_min, e_max, -0.12, 0.62, 'right'),
        # Right Side
        ("Pulse Width (ns)", cmap_tau, d_min, d_max, 1.12, 0.88, 'left'),
        (f"Peak {selected_field}", cmap_field, p_min, p_max, 1.12, 0.62, 'left')
    ]

    for title, cmap, cmin, cmax, x_pos, y_pos, anchor in colorbar_positions:
        fig.add_trace(go.Scatter(
            x=[0], y=[0], mode='markers',
            marker=dict(
                showscale=True,
                colorbar=dict(
                    title=title,
                    title_font=dict(size=11),
                    tickfont=dict(size=10),
                    len=0.2,
                    thickness=15,
                    x=x_pos,
                    y=y_pos,
                    xanchor=anchor,
                    yanchor='top',
                    outlinewidth=0,
                    bordercolor="rgba(0,0,0,0.1)",
                    borderwidth=1
                ),
                colorscale=cmap,
                cmin=cmin,
                cmax=cmax
            ),
            showlegend=False
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
        
        selected_colormap = st.selectbox("Default Colormap (3D View)", EXTENDED_COLORMAPS, index=0, key="global_colormap")

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

    # ================= VISUALIZATION CONTROLS =================
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

    # ================= THEME & LIGHTING SETUP =================
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

    # ================= DATA PROCESSING =================
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

    # ================= COLOR SCALE LIMITS =================
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

    # ================= PLOTLY TRACE CONSTRUCTION =================
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

    # ================= LAYOUT & RENDERING =================
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
    
    # Field Statistics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("Min", f"{np.min(values):.3f}")
    with col2: st.metric("Max", f"{np.max(values):.3f}")
    with col3: st.metric("Mean", f"{np.mean(values):.3f}")
    with col4: st.metric("Std Dev", f"{np.std(values):.3f}")
    with col5: st.metric("Range", f"{np.max(values) - np.min(values):.3f}")

    # ================= SUNBURST SECTION (REACTIVE) =================
    st.markdown('<h2 class="sub-header">🌳 Comparative Sunburst – Peak Values</h2>', unsafe_allow_html=True)
    st.markdown("Hierarchy: **FEM Root → Energy → Pulse Duration → Simulation**")

    if not summaries:
        st.warning("No summary data available.")
        return

    # Get all available fields
    all_fields = set()
    for s in summaries:
        all_fields.update(s.get('field_stats', {}).keys())
    available_fields = sorted(all_fields)

    if not available_fields:
        st.warning("No fields found for sunburst.")
        return

    # Controls
    col1, col2 = st.columns([3, 2])
    with col1:
        sun_field = st.selectbox("Select Field", available_fields, 
                            index=available_fields.index('temperature') if 'temperature' in available_fields else 0,
                            key="sunburst_field")
        highlight_sim = st.selectbox("Highlight Simulation (optional)", 
                                     ["None"] + sorted(simulations.keys()), 
                                     key="sunburst_highlight")
    with col2:
        sun_cmap = st.selectbox("Field Colormap", EXTENDED_COLORMAPS, 
                            index=EXTENDED_COLORMAPS.index(selected_colormap) if selected_colormap in EXTENDED_COLORMAPS else 0,
                            key="sunburst_cmap")

    # 🎨 Hierarchy Colormap Selectors
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
                                    index=EXTENDED_COLORMAPS.index(sun_cmap) if sun_cmap in EXTENDED_COLORMAPS else 0,
                                    key="cmap_field_sim")

    # 🏷️ Label Customization & Visibility
    st.markdown("### 🏷️ Label Customization & Visibility")
    st.markdown("*(Use `{energy}`, `{tau}`, `{peak}` as placeholders for dynamic values)*")
    
    c_vis, c_fmt = st.columns([1, 3])
    with c_vis:
        vis_root = st.checkbox("Show Root Labels", value=True, key="vis_root")
        vis_energy = st.checkbox("Show Energy Labels", value=True, key="vis_energy")
        vis_tau = st.checkbox("Show Tau Labels", value=True, key="vis_tau")
        vis_sim = st.checkbox("Show Sim Labels", value=True, key="vis_sim")
    with c_fmt:
        fmt_root = st.text_input("Root Format", value="All Simulations as FEM", key="fmt_root")
        fmt_energy = st.text_input("Energy Format", value="E = {energy:.1f} mJ", key="fmt_energy")
        fmt_tau = st.text_input("Tau Format", value="τ = {tau:.1f} ns", key="fmt_tau")
        fmt_sim = st.text_input("Sim Format", value="Peak: {peak:.3f}", key="fmt_sim")

    # Font Size Controls
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

    # ✅ Chart renders reactively with all customizations
    fig_sun = create_sunburst_chart(
        summaries, 
        sun_field, 
        cmap_root=cmap_root,
        cmap_energy=cmap_energy,
        cmap_tau=cmap_tau,
        cmap_field=cmap_field_sim,
        highlight_sim=highlight_sim if highlight_sim != "None" else None,
        font_size_root=fs_root,
        font_size_energy=fs_energy,
        font_size_tau=fs_tau,
        font_size_sim=fs_sim,
        show_labels_root=vis_root,
        show_labels_energy=vis_energy,
        show_labels_tau=vis_tau,
        show_labels_sim=vis_sim,
        fmt_root=fmt_root,
        fmt_energy=fmt_energy,
        fmt_tau=fmt_tau,
        fmt_sim=fmt_sim
    )
    st.plotly_chart(fig_sun, use_container_width=True)

    # ==========================================
    # 📥 HD EXPORT SECTION (PNG to JPG)
    # ==========================================
    with st.expander("📥 Export High-Resolution Image", expanded=False):
        st.markdown("*(Note: Requires `kaleido` installed: `pip install kaleido`)*")
        c_e1, c_e2, c_e3 = st.columns(3)
        with c_e1:
            img_width = st.number_input("Width (px)", value=1920, step=100, key="exp_w")
        with c_e2:
            img_height = st.number_input("Height (px)", value=1080, step=100, key="exp_h")
        with c_e3:
            scale_factor = st.slider("Scale Factor (HD Multiplier)", 1.0, 3.0, 2.0, key="exp_s")
            
        jpeg_quality = st.slider("JPG Quality", 70, 100, 95, key="exp_q")
        
        if st.button("📷 Render & Convert to JPG", type="primary", use_container_width=True):
            with st.spinner("⚙️ Rendering HD image & converting..."):
                try:
                    # 1. Render to PNG at high resolution
                    fig_sun.update_layout(margin=dict(l=150, r=150, t=40, b=10))
                    png_bytes = fig_sun.to_image(format="png", scale=scale_factor, width=img_width, height=img_height)
                    
                    # 2. Convert PNG to JPG using Pillow
                    img = Image.open(io.BytesIO(png_bytes))
                    jpg_buffer = io.BytesIO()
                    # Convert to RGB if RGBA (remove alpha for JPG compatibility)
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')
                    img.save(jpg_buffer, format="JPEG", quality=jpeg_quality, optimize=True)
                    jpg_bytes = jpg_buffer.getvalue()
                    
                    st.success(f"✅ Rendered at {int(img_width*scale_factor)}x{int(img_height*scale_factor)} (Scale x{scale_factor})")
                    st.download_button(
                        label="📥 Download HD JPG",
                        data=jpg_bytes,
                        file_name="sunburst_chart_HD.jpg",
                        mime="image/jpeg",
                        use_container_width=True
                    )
                except ImportError:
                    st.error("❌ Missing dependency. Please install `kaleido`: `pip install kaleido`")
                except Exception as e:
                    st.error(f"❌ Export failed. Make sure 'kaleido' is correctly installed.")
                    st.code(f"Error: {str(e)}")

    # Optional: Side-by-Side Comparison
    st.markdown("### Compare Two Fields Side-by-Side (optional)")
    if len(available_fields) >= 2 and st.checkbox("Show two sunbursts"):
        col_l, col_r = st.columns(2)
        with col_l:
            f1 = st.selectbox("Left Field", available_fields, key="sun_f1")
            c1 = st.selectbox("Left Colormap", EXTENDED_COLORMAPS, index=3, key="sun_c1")
        with col_r:
            f2 = st.selectbox("Right Field", available_fields, index=1, key="sun_f2")
            c2 = st.selectbox("Right Colormap", EXTENDED_COLORMAPS, index=1, key="sun_c2")
        
        fig1 = create_sunburst_chart(summaries, f1, cmap_root=cmap_root, cmap_energy=cmap_energy, cmap_tau=cmap_tau, cmap_field=c1,
                                     highlight_sim=highlight_sim if highlight_sim != "None" else None,
                                     font_size_root=fs_root, font_size_energy=fs_energy, font_size_tau=fs_tau, font_size_sim=fs_sim,
                                     show_labels_root=vis_root, show_labels_energy=vis_energy, show_labels_tau=vis_tau, show_labels_sim=vis_sim,
                                     fmt_root=fmt_root, fmt_energy=fmt_energy, fmt_tau=fmt_tau, fmt_sim=fmt_sim)
        fig2 = create_sunburst_chart(summaries, f2, cmap_root=cmap_root, cmap_energy=cmap_energy, cmap_tau=cmap_tau, cmap_field=c2,
                                     highlight_sim=highlight_sim if highlight_sim != "None" else None,
                                     font_size_root=fs_root, font_size_energy=fs_energy, font_size_tau=fs_tau, font_size_sim=fs_sim,
                                     show_labels_root=vis_root, show_labels_energy=vis_energy, show_labels_tau=vis_tau, show_labels_sim=vis_sim,
                                     fmt_root=fmt_root, fmt_energy=fmt_energy, fmt_tau=fmt_tau, fmt_sim=fmt_sim)
        col_l.plotly_chart(fig1, use_container_width=True)
        col_r.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    main()
