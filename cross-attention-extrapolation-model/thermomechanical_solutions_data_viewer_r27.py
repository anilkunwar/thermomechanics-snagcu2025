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

warnings.filterwarnings('ignore')

# =============================================
# CONSTANTS (moved to global scope for reuse)
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
# ENHANCED SUNBURST VISUALIZER
# =============================================
def create_sunburst_chart(summaries, selected_field, colormap='Viridis', highlight_sim=None,
                          font_size_root=16, font_size_energy=14, font_size_tau=12, font_size_sim=10):
    """
    Enhanced sunburst with hierarchical coloring, custom labels, per-level font sizes,
    and dedicated colorbars for each hierarchy level.
    """
    labels, parents, ids, values, colors, levels = [], [], [], [], [], []
    e_indices = {}
    d_indices = {}
    
    # Root Node
    labels.append("All Simulations as FEM")
    parents.append("")
    ids.append("root")
    values.append(0)
    colors.append("#E0E0E0")
    levels.append(0)
    root_idx = 0

    # Collect stats for normalization
    all_energies = [s['energy'] for s in summaries]
    all_taus = [s['duration'] for s in summaries]
    all_peaks = []
    for s in summaries:
        p = s.get('field_stats', {}).get(selected_field, {}).get('max', [0])[0]
        all_peaks.append(float(p) if p > 0 else 1e-3)

    e_min, e_max = min(all_energies), max(all_energies)
    d_min, d_max = min(all_taus), max(all_taus)
    p_min, p_max = min(all_peaks), max(all_peaks)

    # Sample colormaps for each hierarchy
    cmap_e = px.colors.sample_colorscale('YlOrRd', 101)
    cmap_d = px.colors.sample_colorscale('Blues', 101)
    cmap_p = px.colors.sample_colorscale(colormap, 101)

    def norm_val(v, vmin, vmax):
        return max(0.0, min(1.0, (v - vmin) / (vmax - vmin))) if vmax > vmin else 0.5

    # Group data hierarchically
    tree = {}
    for s in summaries:
        e = s['energy']
        d = s['duration']
        sim = s['name']
        peak = float(s.get('field_stats', {}).get(selected_field, {}).get('max', [0])[0])
        if peak <= 0: peak = 1e-3
        tree.setdefault(e, {}).setdefault(d, {})[sim] = peak

    sorted_energies = sorted(set(all_energies))
    sorted_taus = sorted(set(all_taus))

    # Build Hierarchy
    for e in sorted_energies:
        e_lbl = f"E = {e:.1f} mJ"
        e_id = f"E_{e}"
        e_idx = len(labels)
        labels.append(e_lbl)
        parents.append("root")
        ids.append(e_id)
        values.append(0)
        colors.append(cmap_e[int(norm_val(e, e_min, e_max) * 100)])
        levels.append(1)
        e_indices[e] = e_idx

        for d in sorted_taus:
            if d not in tree[e]: continue
            d_lbl = f"τ = {d:.1f} ns"
            d_id = f"E_{e}_T_{d}"
            d_idx = len(labels)
            labels.append(d_lbl)
            parents.append(e_id) # Plotly resolves via IDs if provided
            ids.append(d_id)
            values.append(0)
            colors.append(cmap_d[int(norm_val(d, d_min, d_max) * 100)])
            levels.append(2)
            d_indices[(e, d)] = d_idx

            for sim, peak in tree[e][d].items():
                sim_lbl = f"{sim} (Peak: {peak:.2f})"
                s_id = f"E_{e}_T_{d}_S_{sim}"
                s_idx = len(labels)
                labels.append(sim_lbl)
                parents.append(d_id)
                ids.append(s_id)
                values.append(peak)
                colors.append(cmap_p[int(norm_val(peak, p_min, p_max) * 100)])
                levels.append(3)

    # Aggregate values upward
    for (e, d), d_idx in d_indices.items():
        d_sum = sum(tree[e][d].values())
        values[d_idx] = d_sum
        if e in e_indices:
            values[e_indices[e]] += d_sum

    # Assign font sizes based on hierarchy level
    font_sizes = [
        font_size_root if l == 0 else 
        font_size_energy if l == 1 else 
        font_size_tau if l == 2 else 
        font_size_sim 
        for l in levels
    ]

    # Create Sunburst Trace
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

    # Layout adjustments to accommodate multiple colorbars
    fig.update_layout(
        margin=dict(t=40, l=10, r=220, b=10),
        paper_bgcolor="rgba(0,0,0,0)"
    )

    # Add dedicated colorbars for each hierarchy level using dummy traces
    for title, cmap, cmin, cmax, y_pos in [
        ("Energy (mJ)", 'YlOrRd', e_min, e_max, 0.88),
        ("Pulse Width (ns)", 'Blues', d_min, d_max, 0.58),
        (f"Peak {selected_field}", colormap, p_min, p_max, 0.28)
    ]:
        fig.add_trace(go.Scatter(
            x=[0], y=[0], mode='markers',
            marker=dict(
                showscale=True,
                colorbar=dict(
                    title=title,
                    title_font=dict(size=11),
                    tickfont=dict(size=10),
                    len=0.3,
                    thickness=16,
                    x=1.02,
                    y=y_pos,
                    xanchor='left',
                    yanchor='top',
                    outlinewidth=0
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
        
        selected_colormap = st.selectbox("Colormap", EXTENDED_COLORMAPS, index=0, key="global_colormap")

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
    st.markdown('<h4 class="sub-header">🎛️ Visualization Controls</h4>', unsafe_allow_html=True)
    
    # Row 1: Field, Timestep, Aspect Ratio, Background
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        field = st.selectbox("Select Field", sorted(sim['field_info'].keys()), key="viewer_field_select")
    with col2:
        timestep = st.slider("Timestep", 0, sim['n_timesteps'] - 1, 0, key="viewer_timestep_slider")
    with col3:
        aspect_mode = st.selectbox("Aspect Ratio", ["data", "cube", "auto"], index=0, key="aspect_mode")
    with col4:
        bg_mode = st.selectbox("Plot Theme", ["Light", "Dark"], index=0, key="bg_mode")

    # Row 2: Opacity, Point Size, Camera, Lighting
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
    st.markdown("Hierarchy: **All Simulations as FEM → Energy → Pulse Duration → Simulation** (values = aggregated peaks)")

    summaries = st.session_state.summaries
    simulations = st.session_state.simulations

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
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        sun_field = st.selectbox("Select Field", available_fields, 
                            index=available_fields.index('temperature') if 'temperature' in available_fields else 0,
                            key="sunburst_field")
    with col2:
        sun_cmap = st.selectbox("Colormap", EXTENDED_COLORMAPS, 
                            index=EXTENDED_COLORMAPS.index(selected_colormap) if selected_colormap in EXTENDED_COLORMAPS else 0,
                            key="sunburst_cmap")
    with col3:
        highlight_sim = st.selectbox("Highlight Simulation (optional)", 
                                     ["None"] + sorted(simulations.keys()), 
                                     key="sunburst_highlight")

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

    # ✅ Chart renders reactively without button wrapper
    fig_sun = create_sunburst_chart(
        summaries, 
        sun_field, 
        colormap=sun_cmap, 
        highlight_sim=highlight_sim if highlight_sim != "None" else None,
        font_size_root=fs_root,
        font_size_energy=fs_energy,
        font_size_tau=fs_tau,
        font_size_sim=fs_sim
    )
    st.plotly_chart(fig_sun, use_container_width=True)

    # Optional: two side-by-side sunbursts
    st.markdown("### Compare Two Fields Side-by-Side (optional)")
    if len(available_fields) >= 2 and st.checkbox("Show two sunbursts"):
        col_l, col_r = st.columns(2)
        with col_l:
            f1 = st.selectbox("Left Field", available_fields, key="sun_f1")
            c1 = st.selectbox("Left Colormap", EXTENDED_COLORMAPS, index=3, key="sun_c1")
        with col_r:
            f2 = st.selectbox("Right Field", available_fields, index=1, key="sun_f2")
            c2 = st.selectbox("Right Colormap", EXTENDED_COLORMAPS, index=1, key="sun_c2")
        
        fig1 = create_sunburst_chart(summaries, f1, c1, highlight_sim if highlight_sim != "None" else None,
                                     font_size_root=fs_root, font_size_energy=fs_energy, font_size_tau=fs_tau, font_size_sim=fs_sim)
        fig2 = create_sunburst_chart(summaries, f2, c2, highlight_sim if highlight_sim != "None" else None,
                                     font_size_root=fs_root, font_size_energy=fs_energy, font_size_tau=fs_tau, font_size_sim=fs_sim)
        col_l.plotly_chart(fig1, use_container_width=True)
        col_r.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    main()
