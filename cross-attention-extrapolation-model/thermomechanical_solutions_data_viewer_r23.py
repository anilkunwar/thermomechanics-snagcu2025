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
import pandas as pd
import plotly.express as px
from io import BytesIO
import base64
import hashlib
import tempfile
warnings.filterwarnings('ignore')

# =============================================
# PATH CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEA_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "fea_solutions")
os.makedirs(FEA_SOLUTIONS_DIR, exist_ok=True)

# =============================================
# UNIFIED DATA LOADER (from CODE 21, enhanced)
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
                
                # Build summary statistics for this simulation (for comparative analysis)
                summary = _self._build_summary(name, energy, duration, vtu_files, field_info)
                summaries.append(summary)
                
                simulations[name] = sim_data
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

    def _build_summary(self, name, energy, duration, vtu_files, field_info):
        """Build summary statistics for comparative analysis"""
        summary = {
            'name': name,
            'energy': energy,
            'duration': duration,
            'timesteps': list(range(1, len(vtu_files)+1)),
            'field_stats': {}
        }
        # Placeholder – real implementation would extract statistics from VTU files
        # For now, we'll create dummy stats to avoid errors; actual viewer uses simulations directly.
        # In a full integration, you would compute min/max/mean/std per field from the loaded fields.
        # Since we have the fields in simulations, we can fill summary later. Here we keep it minimal.
        for field in field_info.keys():
            summary['field_stats'][field] = {
                'min': [0], 'max': [0], 'mean': [0], 'std': [0], 'q25': [0], 'q50': [0], 'q75': [0]
            }
        return summary

# =============================================
# ADVANCED VISUALIZER (Sunburst, Radar, etc.)
# =============================================
class EnhancedVisualizer:
    """Provides additional comparative charts like sunburst and radar"""
    
    @staticmethod
    def create_sunburst_chart(summaries, selected_field='temperature', highlight_sim=None):
        """Create enhanced sunburst chart with highlighted target simulation"""
        labels = []
        parents = []
        values = []
        colors = []
        
        # Root node
        labels.append("All Simulations")
        parents.append("")
        values.append(len(summaries))
        colors.append("#1f77b4")
        
        # Group by energy first
        energy_groups = {}
        for summary in summaries:
            energy_key = f"{summary['energy']:.1f} mJ"
            if energy_key not in energy_groups:
                energy_groups[energy_key] = []
            energy_groups[energy_key].append(summary)
        
        # Add energy level nodes
        for energy_key, energy_sims in energy_groups.items():
            labels.append(f"Energy: {energy_key}")
            parents.append("All Simulations")
            values.append(len(energy_sims))
            colors.append("#ff7f0e" if highlight_sim and any(s['name'] == highlight_sim for s in energy_sims) else "#2ca02c")
            
            # Add duration nodes for each energy group
            for summary in energy_sims:
                duration_key = f"τ: {summary['duration']:.1f} ns"
                sim_label = f"{summary['name']}"
                labels.append(sim_label)
                parents.append(f"Energy: {energy_key}")
                values.append(1)
                
                # Highlight target simulation
                if highlight_sim and summary['name'] == highlight_sim:
                    colors.append("#d62728")  # Red for target
                else:
                    colors.append("#9467bd")  # Purple for others
                
                # Add field statistics if available
                if selected_field in summary['field_stats']:
                    stats = summary['field_stats'][selected_field]
                    if stats['max']:
                        avg_max = np.mean(stats['max']) if isinstance(stats['max'], list) else stats['max']
                        field_label = f"{selected_field}: {avg_max:.1f}"
                        labels.append(field_label)
                        parents.append(sim_label)
                        values.append(avg_max if avg_max > 0 else 1e-6)
                        colors.append("#8c564b")  # Brown for field values
        
        # Ensure all values are positive
        values = [max(v, 1e-6) for v in values]
        
        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            marker=dict(
                colors=colors,
                colorscale='Viridis',
                line=dict(width=2, color='white')
            ),
            hovertemplate='<b>%{label}</b><br>Value: %{value:.2f}<br>Parent: %{parent}<extra></extra>',
            textinfo="label+value",
            textfont=dict(size=12)
        ))
        
        title = f"Simulation Hierarchy - {selected_field}"
        if highlight_sim:
            title += f" (Target: {highlight_sim})"
        
        fig.update_layout(
            title=title,
            height=700,
            margin=dict(t=50, b=20, l=20, r=20)
        )
        
        return fig
    
    @staticmethod
    def create_radar_chart(summaries, simulation_names, target_sim=None):
        """Create enhanced radar chart with highlighted target simulation"""
        # Determine available fields
        all_fields = set()
        for summary in summaries:
            all_fields.update(summary['field_stats'].keys())
        
        if not all_fields:
            return go.Figure()
        
        # Select top 6 fields for clarity
        selected_fields = list(all_fields)[:6]
        
        fig = go.Figure()
        
        for sim_name in simulation_names:
            # Find summary
            summary = next((s for s in summaries if s['name'] == sim_name), None)
            if not summary:
                continue
            
            r_values = []
            theta_values = []
            
            for field in selected_fields:
                if field in summary['field_stats']:
                    stats = summary['field_stats'][field]
                    # Use mean value across timesteps
                    if stats['mean']:
                        avg_value = np.mean(stats['mean']) if isinstance(stats['mean'], list) else stats['mean']
                        r_values.append(avg_value if avg_value > 0 else 1e-6)
                        theta_values.append(f"{field[:15]}...")
                    else:
                        r_values.append(1e-6)
                        theta_values.append(f"{field[:15]}...")
                else:
                    r_values.append(1e-6)
                    theta_values.append(f"{field[:15]}...")
            
            # Highlight target simulation
            line_width = 4 if target_sim and sim_name == target_sim else 2
            fill_opacity = 0.6 if target_sim and sim_name == target_sim else 0.3
            color = 'red' if target_sim and sim_name == target_sim else None
            
            fig.add_trace(go.Scatterpolar(
                r=r_values,
                theta=theta_values,
                fill='toself',
                name=sim_name,
                line=dict(width=line_width, color=color),
                fillcolor=f'rgba(255,0,0,{fill_opacity})' if color else None,
                opacity=0.8
            ))
        
        if fig.data:
            # Find maximum value for scaling
            max_values = []
            for trace in fig.data:
                max_values.append(max(trace.r))
            
            if max_values:
                max_r = max(max_values)
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, max_r * 1.2],
                            tickfont=dict(size=10),
                            gridcolor='lightgray',
                            linecolor='gray'
                        ),
                        angularaxis=dict(
                            tickfont=dict(size=11),
                            rotation=90,
                            direction="clockwise"
                        ),
                        bgcolor='white',
                        gridshape='circular'
                    ),
                    showlegend=True,
                    title="Radar Chart: Simulation Comparison",
                    height=600,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=1.05
                    )
                )
        
        return fig
    
    @staticmethod
    def create_field_evolution_comparison(summaries, simulation_names, selected_field, target_sim=None):
        """Create field evolution comparison plot"""
        fig = go.Figure()
        
        for sim_name in simulation_names:
            summary = next((s for s in summaries if s['name'] == sim_name), None)
            
            if summary and selected_field in summary['field_stats']:
                stats = summary['field_stats'][selected_field]
                
                # Highlight target simulation
                line_width = 4 if target_sim and sim_name == target_sim else 2
                line_dash = 'solid' if target_sim and sim_name == target_sim else 'dash'
                
                # Plot mean
                if stats['mean'] and isinstance(stats['mean'], list):
                    fig.add_trace(go.Scatter(
                        x=summary['timesteps'],
                        y=stats['mean'],
                        mode='lines+markers',
                        name=f"{sim_name} (mean)",
                        line=dict(width=line_width, dash=line_dash),
                        opacity=0.8
                    ))
                    
                    # Add confidence band (mean ± std)
                    if stats['std'] and isinstance(stats['std'], list):
                        y_upper = np.array(stats['mean']) + np.array(stats['std'])
                        y_lower = np.array(stats['mean']) - np.array(stats['std'])
                        fig.add_trace(go.Scatter(
                            x=summary['timesteps'] + summary['timesteps'][::-1],
                            y=np.concatenate([y_upper, y_lower[::-1]]),
                            fill='toself',
                            fillcolor=f'rgba(128,128,128,{0.1 if target_sim and sim_name == target_sim else 0.05})',
                            line=dict(color='rgba(255,255,255,0)'),
                            showlegend=False,
                            name=f"{sim_name} ± std"
                        ))
        
        if fig.data:
            fig.update_layout(
                title=f"{selected_field} Evolution Comparison",
                xaxis_title="Timestep (ns)",
                yaxis_title=f"{selected_field} Value",
                hovermode="x unified",
                height=500,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.02
                )
            )
        
        return fig

# =============================================
# MAIN APPLICATION (extended with Comparative Analysis)
# =============================================
def main():
    st.set_page_config(
        page_title="FEA Data Viewer with Comparative Analysis",
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

    st.markdown('<h1 class="main-header">📊 FEA Data Viewer with Comparative Analysis</h1>', unsafe_allow_html=True)

    # Session State Initialization
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = UnifiedFEADataLoader()
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'simulations' not in st.session_state:
        st.session_state.simulations = {}
    if 'summaries' not in st.session_state:
        st.session_state.summaries = []
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = EnhancedVisualizer()
    if 'current_mode' not in st.session_state:
        st.session_state.current_mode = "Data Viewer"

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Navigation")
        app_mode = st.radio(
            "Select Mode",
            ["Data Viewer", "Comparative Analysis"],
            index=0 if st.session_state.current_mode == "Data Viewer" else 1,
            key="nav_mode"
        )
        st.session_state.current_mode = app_mode
        
        st.markdown("---")
        st.markdown("### ⚙️ Data Settings")
        load_full_data = st.checkbox("Load Full Mesh", value=True, help="Load complete mesh data for 3D visualization")
        
        extended_colormaps = [
            'Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis',
            'Rainbow', 'Jet', 'Hot', 'Cool', 'Portland',
            'Bluered', 'Electric', 'Thermal', 'Balance', 'Teal', 'Sunset', 'Burg'
        ]
        selected_colormap = st.selectbox("Colormap", extended_colormaps, index=0, key="global_colormap")

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

    if st.session_state.current_mode == "Data Viewer":
        render_data_viewer(st.session_state.get('global_colormap', 'Viridis'))
    else:
        render_comparative_analysis()

# ----------------------------------------------------------------------
# Data Viewer (original CODE 21 with minor adjustments)
# ----------------------------------------------------------------------
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
    
    # Sync colorbar font with theme
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

# ----------------------------------------------------------------------
# Comparative Analysis (with Sunburst chart from CODE 22)
# ----------------------------------------------------------------------
def render_comparative_analysis():
    st.markdown('<h2 class="sub-header">📊 Comparative Analysis</h2>', unsafe_allow_html=True)
    
    simulations = st.session_state.simulations
    summaries = st.session_state.summaries
    
    if not summaries:
        st.warning("No summary data available for comparative analysis. Please reload simulations.")
        return
    
    # Target simulation selection
    st.markdown('<h3 class="sub-header">🎯 Select Target Simulation</h3>', unsafe_allow_html=True)
    
    available_simulations = sorted(simulations.keys())
    
    col1, col2 = st.columns([3, 1])
    with col1:
        target_simulation = st.selectbox(
            "Select target simulation for highlighting",
            available_simulations,
            key="target_sim_select",
            help="This simulation will be highlighted in all visualizations"
        )
    
    with col2:
        n_comparisons = st.number_input(
            "Number of comparisons",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            key="n_comparisons"
        )
    
    # Select comparison simulations (excluding target)
    comparison_sims = [sim for sim in available_simulations if sim != target_simulation]
    selected_comparisons = st.multiselect(
        "Select simulations for comparison",
        comparison_sims,
        default=comparison_sims[:min(n_comparisons - 1, len(comparison_sims))],
        help="Select simulations to compare with the target"
    )
    
    # Include target in visualization list
    visualization_sims = [target_simulation] + selected_comparisons
    
    if not visualization_sims:
        st.info("Please select at least one simulation for comparison.")
        return
    
    # Field selection
    st.markdown('<h3 class="sub-header">📈 Select Field for Analysis</h3>', unsafe_allow_html=True)
    
    # Get available fields from selected simulations
    available_fields = set()
    for sim_name in visualization_sims:
        if sim_name in simulations:
            available_fields.update(simulations[sim_name]['field_info'].keys())
    
    if not available_fields:
        st.error("No field data available for selected simulations.")
        return
    
    selected_field = st.selectbox(
        "Select field for analysis",
        sorted(available_fields),
        key="comparison_field",
        help="Choose a field to compare across simulations"
    )
    
    # Visualization tabs
    tab1, tab2, tab3 = st.tabs(["📊 Sunburst", "🎯 Radar", "⏱️ Evolution"])
    
    with tab1:
        st.markdown("##### 📊 Hierarchical Sunburst Chart")
        # We need to build richer summaries for the sunburst (with actual field statistics)
        # For demonstration, we'll enhance the summaries on the fly using loaded simulation data
        enhanced_summaries = []
        for summary in summaries:
            sim_name = summary['name']
            if sim_name in simulations:
                sim = simulations[sim_name]
                # Build field stats from loaded fields
                field_stats = {}
                for fld in sim['field_info'].keys():
                    field_data = sim['fields'][fld]
                    # Compute stats across all timesteps
                    if field_data.ndim == 2:  # scalar field
                        all_vals = field_data.flatten()
                        all_vals = all_vals[~np.isnan(all_vals)]
                        if len(all_vals) > 0:
                            field_stats[fld] = {
                                'min': [float(np.min(all_vals))],
                                'max': [float(np.max(all_vals))],
                                'mean': [float(np.mean(all_vals))],
                                'std': [float(np.std(all_vals))],
                                'q25': [float(np.percentile(all_vals, 25))],
                                'q50': [float(np.percentile(all_vals, 50))],
                                'q75': [float(np.percentile(all_vals, 75))]
                            }
                        else:
                            field_stats[fld] = {'min':[0],'max':[0],'mean':[0],'std':[0],'q25':[0],'q50':[0],'q75':[0]}
                    else:  # vector field
                        # compute magnitude across all points and timesteps
                        mag_vals = np.linalg.norm(field_data, axis=2).flatten()
                        mag_vals = mag_vals[~np.isnan(mag_vals)]
                        if len(mag_vals) > 0:
                            field_stats[fld] = {
                                'min': [float(np.min(mag_vals))],
                                'max': [float(np.max(mag_vals))],
                                'mean': [float(np.mean(mag_vals))],
                                'std': [float(np.std(mag_vals))],
                                'q25': [float(np.percentile(mag_vals, 25))],
                                'q50': [float(np.percentile(mag_vals, 50))],
                                'q75': [float(np.percentile(mag_vals, 75))]
                            }
                        else:
                            field_stats[fld] = {'min':[0],'max':[0],'mean':[0],'std':[0],'q25':[0],'q50':[0],'q75':[0]}
                enhanced_summaries.append({
                    'name': sim_name,
                    'energy': summary['energy'],
                    'duration': summary['duration'],
                    'timesteps': summary['timesteps'],
                    'field_stats': field_stats
                })
            else:
                # fallback
                enhanced_summaries.append(summary)
        
        sunburst_fig = st.session_state.visualizer.create_sunburst_chart(
            enhanced_summaries,
            selected_field,
            highlight_sim=target_simulation
        )
        if sunburst_fig.data:
            st.plotly_chart(sunburst_fig, use_container_width=True)
        else:
            st.info("Insufficient data for sunburst chart")
    
    with tab2:
        st.markdown("##### 🎯 Multi-Field Radar Comparison")
        radar_fig = st.session_state.visualizer.create_radar_chart(
            enhanced_summaries,
            visualization_sims,
            target_sim=target_simulation
        )
        if radar_fig.data:
            st.plotly_chart(radar_fig, use_container_width=True)
        else:
            st.info("Insufficient data for radar chart")
    
    with tab3:
        st.markdown("##### ⏱️ Field Evolution Over Time")
        evolution_fig = st.session_state.visualizer.create_field_evolution_comparison(
            enhanced_summaries,
            visualization_sims,
            selected_field,
            target_sim=target_simulation
        )
        if evolution_fig.data:
            st.plotly_chart(evolution_fig, use_container_width=True)
        else:
            st.info(f"No {selected_field} data available for selected simulations")
    
    # Comparative statistics table
    st.markdown('<h3 class="sub-header">📋 Comparative Statistics</h3>', unsafe_allow_html=True)
    
    stats_data = []
    for sim_name in visualization_sims:
        # find summary
        summary = next((s for s in enhanced_summaries if s['name'] == sim_name), None)
        if summary and selected_field in summary['field_stats']:
            stats = summary['field_stats'][selected_field]
            row = {
                'Simulation': sim_name,
                'Type': 'Target' if sim_name == target_simulation else 'Comparison',
                'Energy (mJ)': summary['energy'],
                'Duration (ns)': summary['duration'],
                f'Mean {selected_field}': stats['mean'][0] if stats['mean'] else 0,
                f'Max {selected_field}': stats['max'][0] if stats['max'] else 0,
                f'Std Dev {selected_field}': stats['std'][0] if stats['std'] else 0,
            }
            stats_data.append(row)
    
    if stats_data:
        df_stats = pd.DataFrame(stats_data)
        def highlight_target(row):
            if row['Type'] == 'Target':
                return ['background-color: #ffcccc'] * len(row)
            return [''] * len(row)
        styled_df = df_stats.style.apply(highlight_target, axis=1)
        format_dict = {col: "{:.3f}" for col in df_stats.columns if col not in ['Simulation','Type','Energy (mJ)','Duration (ns)']}
        format_dict['Energy (mJ)'] = "{:.2f}"
        format_dict['Duration (ns)'] = "{:.2f}"
        styled_df = styled_df.format(format_dict)
        st.dataframe(styled_df, use_container_width=True)
        
        # Export
        csv = df_stats.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Comparison as CSV",
            data=csv,
            file_name=f"comparison_{selected_field}.csv",
            mime="text/csv",
            use_container_width=True
        )

if __name__ == "__main__":
    main()
