import streamlit as st
import os
import glob
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from typing import Dict, List, Tuple, Optional
import time
import warnings
warnings.filterwarnings('ignore')

# =============================================
# PATH CONFIGURATION & SETTINGS
# =============================================
@st.cache_resource
def setup_paths():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    FEA_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "fea_solutions")
    os.makedirs(FEA_SOLUTIONS_DIR, exist_ok=True)
    return FEA_SOLUTIONS_DIR

FEA_SOLUTIONS_DIR = setup_paths()

# =============================================
# CUSTOM COLOR SCALES
# =============================================
def create_custom_colormaps():
    """Create custom colormaps for different field types"""
    thermal_cmap = LinearSegmentedColormap.from_list(
        'thermal', ['#00008B', '#4169E1', '#87CEEB', '#FFD700', '#FF4500', '#8B0000']
    )
    
    stress_cmap = LinearSegmentedColormap.from_list(
        'stress', ['#2E86AB', '#73D2DE', '#F6F5AE', '#F5C396', '#ED6A5A']
    )
    
    displacement_cmap = LinearSegmentedColormap.from_list(
        'displacement', ['#003f5c', '#444e86', '#955196', '#dd5182', '#ff6e54', '#ffa600']
    )
    
    return {
        'temperature': thermal_cmap,
        'heat': thermal_cmap,
        'stress': stress_cmap,
        'strain': stress_cmap,
        'displacement': displacement_cmap,
        'velocity': displacement_cmap,
        'default': 'Viridis'
    }

CUSTOM_CMAPS = create_custom_colormaps()

# =============================================
# ADVANCED FOLDER PARSER
# =============================================
class SimulationParser:
    """Robust parser for simulation folder names"""
    
    PATTERNS = [
        r"q([\dp\.]+)mJ-delta([\dp\.]+)ns",
        r"energy_([\d\.]+)mJ-dur_([\d\.]+)ns",
        r"sim_([\d\.]+)_([\d\.]+)",
        r"E([\d\.]+)mJ_D([\d\.]+)ns"
    ]
    
    @staticmethod
    def parse_folder_name(folder: str) -> Tuple[Optional[float], Optional[float]]:
        """Parse folder name with multiple pattern support"""
        base_name = os.path.basename(folder)
        
        for pattern in SimulationParser.PATTERNS:
            match = re.match(pattern, base_name)
            if match:
                try:
                    e_str, d_str = match.groups()
                    energy = float(e_str.replace("p", ".") if "p" in e_str else e_str)
                    duration = float(d_str.replace("p", ".") if "p" in d_str else d_str)
                    return energy, duration
                except ValueError:
                    continue
        
        # Try to extract numbers as fallback
        numbers = re.findall(r"[\d\.]+", base_name)
        if len(numbers) >= 2:
            try:
                return float(numbers[0]), float(numbers[1])
            except ValueError:
                pass
        
        return None, None

# =============================================
# ADVANCED DATA LOADER WITH ERROR HANDLING
# =============================================
@st.cache_data(show_spinner="Loading simulation data...", ttl=3600)
def load_all_simulations_robust():
    """Robust simulation data loader with progress tracking"""
    try:
        import pyvista as pv
        pv.set_jupyter_backend(None)  # Disable rendering backend
    except ImportError:
        st.error("PyVista is required for loading VTU files. Install with: pip install pyvista")
        return {}
    
    simulations = {}
    folders = glob.glob(os.path.join(FEA_SOLUTIONS_DIR, "q*mJ-delta*ns"))
    
    if not folders:
        folders = glob.glob(os.path.join(FEA_SOLUTIONS_DIR, "*"))
        folders = [f for f in folders if os.path.isdir(f)]
    
    if not folders:
        return {}
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, folder in enumerate(folders):
        folder_name = os.path.basename(folder)
        status_text.text(f"Loading {folder_name}... ({idx+1}/{len(folders)})")
        
        try:
            energy, duration = SimulationParser.parse_folder_name(folder_name)
            if energy is None:
                st.warning(f"Skipping folder with unknown format: {folder_name}")
                continue
            
            vtu_files = sorted(glob.glob(os.path.join(folder, "*.vtu")))
            if not vtu_files:
                st.warning(f"No VTU files found in {folder_name}")
                continue
            
            # Sample first file to get structure
            try:
                mesh0 = pv.read(vtu_files[0])
            except Exception as e:
                st.error(f"Error reading {vtu_files[0]}: {e}")
                continue
            
            if not mesh0.point_data:
                st.warning(f"No point data in {folder_name}")
                continue
            
            points = mesh0.points
            n_pts = mesh0.n_points
            n_steps = len(vtu_files)
            
            fields = {}
            field_info = {}
            units = {}
            
            # Detect field types and units
            for key in mesh0.point_data.keys():
                arr = np.asarray(mesh0.point_data[key])
                if arr.ndim == 1:
                    field_type = "scalar"
                    dim = 1
                    unit = guess_unit(key)
                else:
                    field_type = "vector"
                    dim = arr.shape[1]
                    unit = guess_unit(key)
                
                field_info[key] = (field_type, dim)
                units[key] = unit
                
                if field_type == "scalar":
                    fields[key] = np.full((n_steps, n_pts), np.nan, dtype=np.float32)
                else:
                    fields[key] = np.full((n_steps, n_pts, dim), np.nan, dtype=np.float32)
            
            # Load all timesteps with progress
            for t, vtu in enumerate(vtu_files):
                try:
                    mesh = pv.read(vtu)
                    for key, (kind, _) in field_info.items():
                        if key in mesh.point_data:
                            data = np.asarray(mesh.point_data[key], dtype=np.float32)
                            fields[key][t] = data
                except Exception as e:
                    st.warning(f"Error reading timestep {t} in {folder_name}: {e}")
                    continue
            
            # Calculate derived fields
            derived_fields = calculate_derived_fields(fields, field_info)
            fields.update(derived_fields)
            
            # Calculate time array
            time_array = np.linspace(0, duration, n_steps)
            
            simulations[folder_name] = {
                'energy_mJ': energy,
                'duration_ns': duration,
                'points': points,
                'fields': fields,
                'field_info': field_info,
                'units': units,
                'n_timesteps': n_steps,
                'timesteps': time_array,
                'folder': folder_name,
                'loaded_time': time.time()
            }
            
        except Exception as e:
            st.error(f"Error processing {folder_name}: {e}")
            continue
        
        progress_bar.progress((idx + 1) / len(folders))
    
    status_text.empty()
    progress_bar.empty()
    
    if simulations:
        st.success(f"Successfully loaded {len(simulations)} simulations")
    
    return simulations

def guess_unit(field_name: str) -> str:
    """Guess unit based on field name"""
    field_lower = field_name.lower()
    if 'temp' in field_lower:
        return 'K'
    elif 'stress' in field_lower or 'pressure' in field_lower:
        return 'Pa'
    elif 'strain' in field_lower:
        return ''
    elif 'displacement' in field_lower:
        return 'm'
    elif 'velocity' in field_lower:
        return 'm/s'
    elif 'heat' in field_lower or 'flux' in field_lower:
        return 'W/m²'
    else:
        return 'a.u.'

def calculate_derived_fields(fields: Dict, field_info: Dict) -> Dict:
    """Calculate derived fields like gradients, magnitudes, etc."""
    derived = {}
    
    for key, (kind, dim) in field_info.items():
        if kind == "vector":
            # Calculate magnitude
            derived[f"{key}_magnitude"] = np.linalg.norm(fields[key], axis=-1)
            
            # Calculate components if 3D
            if dim == 3:
                derived[f"{key}_x"] = fields[key][..., 0]
                derived[f"{key}_y"] = fields[key][..., 1]
                derived[f"{key}_z"] = fields[key][..., 2]
    
    return derived

# =============================================
# ADVANCED VISUALIZATION FUNCTIONS
# =============================================
class AdvancedVisualizer:
    """Advanced visualization tools for FEA data"""
    
    @staticmethod
    def create_3d_scatter_plot(points, values, field_name, unit, 
                               cmap='Viridis', size=3, opacity=0.8,
                               show_slice=False, slice_axis='z', slice_pos=0.5):
        """Create enhanced 3D scatter plot with slicing options"""
        
        if show_slice:
            if slice_axis == 'x':
                mask = points[:, 0] > slice_pos * points[:, 0].ptp()
            elif slice_axis == 'y':
                mask = points[:, 1] > slice_pos * points[:, 1].ptp()
            else:  # 'z'
                mask = points[:, 2] > slice_pos * points[:, 2].ptp()
            points = points[mask]
            values = values[mask]
        
        fig = go.Figure(data=go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=size,
                color=values,
                colorscale=cmap,
                colorbar=dict(
                    title=f"{field_name} [{unit}]",
                    titlefont=dict(size=14),
                    tickfont=dict(size=12),
                    thickness=20,
                    len=0.75
                ),
                opacity=opacity,
                showscale=True,
                cmin=np.nanpercentile(values, 5),
                cmax=np.nanpercentile(values, 95)
            ),
            hovertemplate=(
                f"<b>{field_name}</b>: %{{color:.3e}} {unit}<br>"
                "X: %{x:.3e}<br>"
                "Y: %{y:.3e}<br>"
                "Z: %{z:.3e}<br>"
                "<extra></extra>"
            )
        ))
        
        fig.update_layout(
            height=700,
            scene=dict(
                aspectmode='data',
                xaxis_title="X [m]",
                yaxis_title="Y [m]",
                zaxis_title="Z [m]",
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            title=dict(
                text=f"3D Visualization: {field_name}",
                font=dict(size=20),
                x=0.5
            )
        )
        
        return fig
    
    @staticmethod
    def create_slice_plot(points, values, field_name, unit,
                         slice_axis='z', slice_pos=0.5, cmap='viridis'):
        """Create 2D slice through 3D data"""
        
        # Create slice
        if slice_axis == 'x':
            slice_coord = points[:, 0]
            x_axis = points[:, 1]
            y_axis = points[:, 2]
            x_label = "Y [m]"
            y_label = "Z [m]"
        elif slice_axis == 'y':
            slice_coord = points[:, 1]
            x_axis = points[:, 0]
            y_axis = points[:, 2]
            x_label = "X [m]"
            y_label = "Z [m]"
        else:  # 'z'
            slice_coord = points[:, 2]
            x_axis = points[:, 0]
            y_axis = points[:, 1]
            x_label = "X [m]"
            y_label = "Y [m]"
        
        # Select points near slice
        tolerance = 0.01 * slice_coord.ptp()
        mask = np.abs(slice_coord - slice_pos * slice_coord.max()) < tolerance
        
        if not np.any(mask):
            return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sc = ax.scatter(
            x_axis[mask],
            y_axis[mask],
            c=values[mask],
            s=10,
            cmap=cmap,
            alpha=0.8,
            edgecolors='none'
        )
        
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(f"{field_name} at {slice_axis} = {slice_pos:.2f} × max", fontsize=14)
        
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(f"{field_name} [{unit}]", fontsize=12)
        
        ax.grid(True, alpha=0.3)
        
        return fig
    
    @staticmethod
    def create_time_evolution_plot(time_data, field_data, field_name, unit,
                                  point_indices=None, statistic='mean'):
        """Create time evolution plot for selected points or statistics"""
        
        fig = go.Figure()
        
        if point_indices is None:
            # Plot statistic over all points
            if statistic == 'mean':
                y_data = np.nanmean(field_data, axis=1)
            elif statistic == 'max':
                y_data = np.nanmax(field_data, axis=1)
            elif statistic == 'min':
                y_data = np.nanmin(field_data, axis=1)
            elif statistic == 'std':
                y_data = np.nanstd(field_data, axis=1)
            
            fig.add_trace(go.Scatter(
                x=time_data,
                y=y_data,
                mode='lines+markers',
                name=f'{statistic.capitalize()} {field_name}',
                line=dict(width=3),
                marker=dict(size=8)
            ))
        else:
            # Plot selected points
            for idx in point_indices[:10]:  # Limit to 10 points
                fig.add_trace(go.Scatter(
                    x=time_data,
                    y=field_data[:, idx],
                    mode='lines',
                    name=f'Point {idx}',
                    opacity=0.7
                ))
        
        fig.update_layout(
            title=f"Time Evolution: {field_name}",
            xaxis_title="Time [ns]",
            yaxis_title=f"{field_name} [{unit}]",
            hovermode='x unified',
            height=500,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    @staticmethod
    def create_distribution_plot(values, field_name, unit):
        """Create distribution analysis plot"""
        
        clean_values = values[~np.isnan(values)].flatten()
        
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=clean_values,
            nbinsx=50,
            name='Distribution',
            opacity=0.7,
            marker_color='royalblue'
        ))
        
        # KDE
        kde = stats.gaussian_kde(clean_values)
        x_kde = np.linspace(clean_values.min(), clean_values.max(), 200)
        y_kde = kde(x_kde)
        
        fig.add_trace(go.Scatter(
            x=x_kde,
            y=y_kde * len(clean_values) * (x_kde[1] - x_kde[0]),
            mode='lines',
            name='KDE',
            line=dict(color='red', width=2)
        ))
        
        # Statistics lines
        mean_val = np.mean(clean_values)
        median_val = np.median(clean_values)
        std_val = np.std(clean_values)
        
        fig.add_vline(x=mean_val, line_dash="dash", line_color="green",
                     annotation_text=f"Mean: {mean_val:.3e}")
        fig.add_vline(x=median_val, line_dash="dot", line_color="orange",
                     annotation_text=f"Median: {median_val:.3e}")
        
        fig.update_layout(
            title=f"Distribution of {field_name}",
            xaxis_title=f"{field_name} [{unit}]",
            yaxis_title="Count",
            barmode='overlay',
            height=500,
            showlegend=True
        )
        
        return fig

# =============================================
# STREAMLIT APP - ENHANCED
# =============================================
def main():
    st.set_page_config(
        page_title="Advanced FEA Laser Simulation Viewer",
        layout="wide",
        page_icon="🔬"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3498db;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🔬 Advanced FEA Laser Simulation Viewer</h1>', 
                unsafe_allow_html=True)
    st.caption("Advanced visualization and analysis tool for FEA simulation data")
    
    # Load simulations
    with st.spinner("Loading simulation data..."):
        simulations = load_all_simulations_robust()
    
    if not simulations:
        st.error("""
        No valid simulations found. Please ensure:
        1. VTU files are in the 'fea_solutions' directory
        2. Folder names follow patterns like 'q0p5mJ-delta4p2ns'
        3. Files have proper point data
        """)
        return
    
    # Sidebar - Enhanced controls
    st.sidebar.header("⚙️ Simulation Configuration")
    
    # Simulation selection with metadata
    sim_names = sorted(simulations.keys())
    sim_name = st.sidebar.selectbox(
        "Select simulation",
        sim_names,
        format_func=lambda x: f"{x} (E={simulations[x]['energy_mJ']}mJ, D={simulations[x]['duration_ns']}ns)"
    )
    sim = simulations[sim_name]
    
    # Display simulation info
    st.sidebar.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.sidebar.metric("Energy", f"{sim['energy_mJ']} mJ")
    st.sidebar.metric("Pulse Duration", f"{sim['duration_ns']} ns")
    st.sidebar.metric("Points", f"{sim['points'].shape[0]:,}")
    st.sidebar.metric("Timesteps", sim['n_timesteps'])
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Field selection
    field_options = list(sim['fields'].keys())
    field = st.sidebar.selectbox("Select field", field_options)
    
    # Component selection for vector fields
    component = None
    if sim['field_info'].get(field, ('scalar', 1))[0] == 'vector':
        component_options = ['magnitude', 'x', 'y', 'z']
        component = st.sidebar.selectbox("Component", component_options)
        display_field = f"{field}_{component}" if component != 'magnitude' else f"{field}_magnitude"
    else:
        display_field = field
    
    # Timestep selection with time display
    timestep = st.sidebar.slider(
        "Timestep",
        0,
        sim['n_timesteps'] - 1,
        0,
        help=f"Time: {sim['timesteps'][0]:.2f} - {sim['timesteps'][-1]:.2f} ns"
    )
    st.sidebar.caption(f"Current time: {sim['timesteps'][timestep]:.2f} ns")
    
    # Visualization settings
    st.sidebar.header("🎨 Visualization Settings")
    
    colormap_options = ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis',
                       'Rainbow', 'Jet', 'Thermal', 'Coolwarm']
    cmap = st.sidebar.selectbox("Colormap", colormap_options, index=0)
    
    marker_size = st.sidebar.slider("Marker size", 1, 10, 3)
    opacity = st.sidebar.slider("Opacity", 0.1, 1.0, 0.8)
    
    # Slicing options
    st.sidebar.header("✂️ Slicing Options")
    enable_slice = st.sidebar.checkbox("Enable slicing", False)
    if enable_slice:
        slice_axis = st.sidebar.selectbox("Slice axis", ['x', 'y', 'z'], index=2)
        slice_pos = st.sidebar.slider("Slice position", 0.0, 1.0, 0.5, 0.01)
    
    # Main content - Tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 3D Visualization", 
        "📈 Time Evolution", 
        "📉 Distribution",
        "🔍 2D Slices",
        "📋 Statistics"
    ])
    
    # Get data
    pts = sim['points']
    raw_data = sim['fields'][field][timestep]
    unit = sim['units'].get(field, 'a.u.')
    
    # Process data based on component selection
    if component == 'x':
        values = raw_data[:, 0] if raw_data.ndim > 1 else raw_data
    elif component == 'y':
        values = raw_data[:, 1] if raw_data.ndim > 1 else raw_data
    elif component == 'z':
        values = raw_data[:, 2] if raw_data.ndim > 1 else raw_data
    elif component == 'magnitude' or (raw_data.ndim > 1 and raw_data.shape[1] > 1):
        values = np.linalg.norm(raw_data, axis=1) if raw_data.ndim > 1 else raw_data
    else:
        values = raw_data
    
    # Tab 1: 3D Visualization
    with tab1:
        st.markdown('<h2 class="section-header">3D Visualization</h2>', 
                   unsafe_allow_html=True)
        
        viz = AdvancedVisualizer()
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig_3d = viz.create_3d_scatter_plot(
                pts, values, field, unit, cmap,
                marker_size, opacity,
                enable_slice if 'enable_slice' in locals() else False,
                slice_axis if 'slice_axis' in locals() else 'z',
                slice_pos if 'slice_pos' in locals() else 0.5
            )
            st.plotly_chart(fig_3d, use_container_width=True)
        
        with col2:
            st.markdown("### Data Statistics")
            clean_values = values[~np.isnan(values)]
            
            metrics = {
                "Min": f"{clean_values.min():.3e}",
                "Max": f"{clean_values.max():.3e}",
                "Mean": f"{clean_values.mean():.3e}",
                "Median": f"{np.median(clean_values):.3e}",
                "Std Dev": f"{clean_values.std():.3e}",
                "95th %": f"{np.percentile(clean_values, 95):.3e}"
            }
            
            for name, value in metrics.items():
                st.metric(name, value)
    
    # Tab 2: Time Evolution
    with tab2:
        st.markdown('<h2 class="section-header">Time Evolution Analysis</h2>', 
                   unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            statistic = st.selectbox(
                "Statistic over points",
                ['mean', 'max', 'min', 'std']
            )
        
        with col2:
            point_selection = st.radio(
                "Analysis mode",
                ['Statistics over all points', 'Specific points']
            )
        
        if point_selection == 'Specific points':
            point_indices = st.multiselect(
                "Select point indices",
                range(min(100, pts.shape[0])),
                default=[0]
            )
        else:
            point_indices = None
        
        fig_time = AdvancedVisualizer.create_time_evolution_plot(
            sim['timesteps'],
            sim['fields'][field],
            field,
            unit,
            point_indices,
            statistic
        )
        
        st.plotly_chart(fig_time, use_container_width=True)
    
    # Tab 3: Distribution
    with tab3:
        st.markdown('<h2 class="section-header">Field Distribution Analysis</h2>', 
                   unsafe_allow_html=True)
        
        fig_dist = AdvancedVisualizer.create_distribution_plot(values, field, unit)
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Q-Q plot for normality check
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Normality Test")
            from scipy.stats import shapiro, normaltest
            
            clean_values = values[~np.isnan(values)]
            if len(clean_values) > 3:
                stat, p = shapiro(clean_values[:5000])  # Limit for Shapiro
                st.write(f"Shapiro-Wilk test: p = {p:.3e}")
                if p > 0.05:
                    st.success("Data appears normally distributed")
                else:
                    st.warning("Data may not be normally distributed")
        
        with col2:
            st.subheader("Percentiles")
            percentiles = np.percentile(clean_values, [5, 25, 50, 75, 95])
            percentile_df = pd.DataFrame({
                'Percentile': ['5th', '25th', '50th', '75th', '95th'],
                'Value': percentiles
            })
            st.dataframe(percentile_df.style.format({'Value': '{:.3e}'}))
    
    # Tab 4: 2D Slices
    with tab4:
        st.markdown('<h2 class="section-header">2D Cross-Sections</h2>', 
                   unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            slice_axis = st.selectbox("Slice axis", ['x', 'y', 'z'], key='slice_2d')
        
        with col2:
            slice_pos = st.slider("Slice position", 0.0, 1.0, 0.5, 0.01, key='pos_2d')
        
        with col3:
            cmap_2d = st.selectbox("Colormap", ['viridis', 'plasma', 'inferno', 'coolwarm'], 
                                  key='cmap_2d')
        
        fig_slice = AdvancedVisualizer.create_slice_plot(
            pts, values, field, unit,
            slice_axis, slice_pos, cmap_2d
        )
        
        if fig_slice:
            st.pyplot(fig_slice)
        else:
            st.warning("No data points found in the selected slice")
    
    # Tab 5: Statistics
    with tab5:
        st.markdown('<h2 class="section-header">Comprehensive Statistics</h2>', 
                   unsafe_allow_html=True)
        
        # Create summary dataframe
        summary_data = []
        for f in field_options[:10]:  # Limit to first 10 fields
            try:
                field_data = sim['fields'][f][timestep].flatten()
                clean_data = field_data[~np.isnan(field_data)]
                
                summary_data.append({
                    'Field': f,
                    'Type': sim['field_info'].get(f, ('scalar', 1))[0],
                    'Min': np.min(clean_data),
                    'Max': np.max(clean_data),
                    'Mean': np.mean(clean_data),
                    'Std': np.std(clean_data),
                    'Unit': sim['units'].get(f, 'a.u.')
                })
            except:
                continue
        
        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(
                df_summary.style.format({
                    'Min': '{:.3e}',
                    'Max': '{:.3e}',
                    'Mean': '{:.3e}',
                    'Std': '{:.3e}'
                }),
                use_container_width=True
            )
        
        # Export options
        st.download_button(
            label="📥 Download Current Timestep Data",
            data=pd.DataFrame({
                'X': pts[:, 0],
                'Y': pts[:, 1],
                'Z': pts[:, 2],
                field: values
            }).to_csv(index=False),
            file_name=f"{sim_name}_{field}_t{timestep}.csv",
            mime="text/csv"
        )
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        **Visualization Features:**
        - 3D scatter plots with slicing
        - Time evolution analysis
        - Statistical distribution
        - 2D cross-sections
        - Comprehensive statistics
        """
    )

# =============================================
# ENTRY POINT
# =============================================
if __name__ == "__main__":
    main()
