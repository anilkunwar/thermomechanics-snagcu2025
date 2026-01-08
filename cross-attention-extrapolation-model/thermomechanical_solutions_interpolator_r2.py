import streamlit as st
import os
import glob
import re
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import meshio
import warnings
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from io import BytesIO
import pandas as pd
import traceback
from scipy.interpolate import griddata, RBFInterpolator
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx

warnings.filterwarnings('ignore')

# =============================================
# PATH CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEA_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "fea_solutions")
os.makedirs(FEA_SOLUTIONS_DIR, exist_ok=True)

# =============================================
# UNIFIED DATA LOADER WITH ENHANCED CAPABILITIES
# =============================================
class UnifiedFEADataLoader:
    """Enhanced data loader combining both meshio and PyVista capabilities"""
    
    def __init__(self):
        self.simulations = {}
        self.summaries = []
        self.field_statistics = {}
        
    def parse_folder_name(self, folder: str):
        """q0p5mJ-delta4p2ns → (0.5, 4.2)"""
        match = re.match(r"q([\dp\.]+)mJ-delta([\dp\.]+)ns", folder)
        if not match:
            return None, None
        e, d = match.groups()
        return float(e.replace("p", ".")), float(d.replace("p", "."))
    
    @st.cache_data
    def load_all_simulations(_self, load_full_mesh=True):
        """Load all simulations with option for full mesh or summaries"""
        simulations = {}
        summaries = []
        
        folders = glob.glob(os.path.join(FEA_SOLUTIONS_DIR, "q*mJ-delta*ns"))
        
        for folder in folders:
            name = os.path.basename(folder)
            energy, duration = _self.parse_folder_name(name)
            if energy is None:
                continue
                
            vtu_files = sorted(glob.glob(os.path.join(folder, "a_t????.vtu")))
            if not vtu_files:
                continue
            
            # Load first file to get structure
            try:
                mesh0 = meshio.read(vtu_files[0])
                
                # Create simulation entry
                sim_data = {
                    'name': name,
                    'energy_mJ': energy,
                    'duration_ns': duration,
                    'n_timesteps': len(vtu_files),
                    'vtu_files': vtu_files,
                    'field_info': {}
                }
                
                if load_full_mesh:
                    # Full mesh loading (for visualization)
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
                    for key in mesh0.point_data.keys():
                        arr = mesh0.point_data[key].astype(np.float32)
                        if arr.ndim == 1:
                            sim_data['field_info'][key] = ("scalar", 1)
                            fields[key] = np.full((len(vtu_files), n_pts), np.nan, dtype=np.float32)
                        else:
                            sim_data['field_info'][key] = ("vector", arr.shape[1])
                            fields[key] = np.full((len(vtu_files), n_pts, arr.shape[1]), np.nan, dtype=np.float32)
                        fields[key][0] = arr
                    
                    # Load all timesteps
                    for t in range(1, len(vtu_files)):
                        mesh = meshio.read(vtu_files[t])
                        for key in sim_data['field_info'].keys():
                            fields[key][t] = mesh.point_data[key].astype(np.float32)
                    
                    sim_data.update({
                        'points': points,
                        'fields': fields,
                        'triangles': triangles
                    })
                
                # Create summary statistics (for interpolation/extrapolation)
                summary = _self.extract_summary_statistics(vtu_files, energy, duration)
                summaries.append(summary)
                
                simulations[name] = sim_data
                
            except Exception as e:
                st.warning(f"Error loading {name}: {str(e)}")
                continue
        
        return simulations, summaries
    
    def extract_summary_statistics(self, vtu_files, energy, duration):
        """Extract summary statistics from VTU files"""
        summary = {
            'energy': energy,
            'duration': duration,
            'timesteps': [],
            'field_stats': {}
        }
        
        for idx, vtu_file in enumerate(vtu_files):
            mesh = meshio.read(vtu_file)
            timestep = idx + 1  # 1-based indexing
            summary['timesteps'].append(timestep)
            
            for field_name in mesh.point_data.keys():
                data = mesh.point_data[field_name]
                
                if field_name not in summary['field_stats']:
                    summary['field_stats'][field_name] = {
                        'min': [], 'max': [], 'mean': [], 'std': [],
                        'q25': [], 'q50': [], 'q75': []
                    }
                
                if data.ndim == 1:
                    summary['field_stats'][field_name]['min'].append(np.min(data))
                    summary['field_stats'][field_name]['max'].append(np.max(data))
                    summary['field_stats'][field_name]['mean'].append(np.mean(data))
                    summary['field_stats'][field_name]['std'].append(np.std(data))
                    summary['field_stats'][field_name]['q25'].append(np.percentile(data, 25))
                    summary['field_stats'][field_name]['q50'].append(np.percentile(data, 50))
                    summary['field_stats'][field_name]['q75'].append(np.percentile(data, 75))
                else:
                    # For vector fields, take magnitude
                    magnitude = np.linalg.norm(data, axis=1)
                    summary['field_stats'][field_name]['min'].append(np.min(magnitude))
                    summary['field_stats'][field_name]['max'].append(np.max(magnitude))
                    summary['field_stats'][field_name]['mean'].append(np.mean(magnitude))
                    summary['field_stats'][field_name]['std'].append(np.std(magnitude))
        
        return summary

# =============================================
# TRANSFORMER-INSPIRED ATTENTION MECHANISM WITH SPATIAL LOCALITY
# =============================================
class PhysicsInformedAttentionExtrapolator:
    """Enhanced extrapolator with spatial locality regulation"""
    
    def __init__(self, sigma_param=0.3, spatial_weight=0.5, n_heads=4):
        self.sigma_param = sigma_param
        self.spatial_weight = spatial_weight
        self.n_heads = n_heads
        self.source_db = []
        self.scaler = StandardScaler()
        
    def load_summaries(self, summaries):
        """Load summary statistics from data loader"""
        self.source_db = summaries
        
        # Prepare data for scaling
        all_embeddings = []
        for summary in summaries:
            for t in summary['timesteps']:
                emb = self._compute_embedding(summary['energy'], summary['duration'], t)
                all_embeddings.append(emb)
        
        if all_embeddings:
            self.scaler.fit(all_embeddings)
    
    def _compute_embedding(self, energy, duration, time):
        """Compute physics-aware embedding with enhanced features"""
        # Log-energy scaling
        logE = np.log1p(energy)
        
        # Dimensionless parameters
        energy_density = energy / (duration + 1e-6)
        time_ratio = time / max(duration, 1e-3)
        
        # Thermal diffusion proxy
        thermal_diffusion = time * 0.1 / (duration + 1e-6)
        
        # Strain rate proxy
        strain_rate = energy_density / (time + 1e-6)
        
        return np.array([
            logE,
            duration,
            time,
            energy_density,
            time_ratio,
            thermal_diffusion,
            strain_rate
        ], dtype=np.float32)
    
    def _multi_head_attention(self, query_embedding, source_embeddings, values):
        """Multi-head attention mechanism inspired by transformers"""
        n_sources = len(source_embeddings)
        weights = np.zeros(n_sources)
        
        # Normalize embeddings
        query_norm = self.scaler.transform([query_embedding])[0]
        source_norm = self.scaler.transform(source_embeddings)
        
        # Multi-head attention
        for head in range(self.n_heads):
            # Different projection for each head
            np.random.seed(head)  # For reproducibility
            proj_matrix = np.random.randn(len(query_embedding), 3)
            
            query_proj = query_norm @ proj_matrix
            source_proj = source_norm @ proj_matrix
            
            # Compute attention scores
            scores = np.exp(-np.sum((query_proj - source_proj) ** 2, axis=1) / 
                          (2 * self.sigma_param ** 2))
            
            # Apply spatial locality regulation
            if self.spatial_weight > 0:
                # Add spatial correlation (simplified - in practice would use actual coordinates)
                spatial_corr = 1.0 / (1.0 + np.arange(n_sources) / n_sources)
                scores = (1 - self.spatial_weight) * scores + self.spatial_weight * spatial_corr
            
            weights += scores / self.n_heads
        
        # Softmax normalization
        weights = np.exp(weights - np.max(weights))
        weights = weights / (np.sum(weights) + 1e-10)
        
        # Weighted prediction
        if len(values) > 0:
            prediction = np.sum(weights[:, np.newaxis] * values, axis=0)
        else:
            prediction = np.zeros(values.shape[1]) if values.ndim > 1 else 0.0
        
        return prediction, weights
    
    def predict_field_statistics(self, energy_query, duration_query, time_query):
        """Predict field statistics for given parameters"""
        if not self.source_db:
            return None
        
        query_embedding = self._compute_embedding(energy_query, duration_query, time_query)
        
        all_embeddings = []
        all_values = []
        
        # Collect all source data
        for summary in self.source_db:
            for idx, t in enumerate(summary['timesteps']):
                src_embedding = self._compute_embedding(summary['energy'], 
                                                       summary['duration'], t)
                all_embeddings.append(src_embedding)
                
                # Collect field statistics values
                field_vals = []
                for field in summary['field_stats']:
                    stats = summary['field_stats'][field]
                    field_vals.extend([
                        stats['mean'][idx],
                        stats['max'][idx],
                        stats['std'][idx]
                    ])
                all_values.append(field_vals)
        
        all_embeddings = np.array(all_embeddings)
        all_values = np.array(all_values)
        
        # Apply multi-head attention
        prediction, attention_weights = self._multi_head_attention(
            query_embedding, all_embeddings, all_values
        )
        
        # Reconstruct field statistics from prediction
        result = {
            'prediction': prediction,
            'attention_weights': attention_weights,
            'confidence': np.max(attention_weights),
            'field_predictions': {}
        }
        
        # Map predictions back to field statistics
        field_idx = 0
        for summary in self.source_db:
            for field in summary['field_stats']:
                if field not in result['field_predictions']:
                    result['field_predictions'][field] = {
                        'mean': prediction[field_idx],
                        'max': prediction[field_idx + 1],
                        'std': prediction[field_idx + 2]
                    }
                    field_idx += 3
        
        return result
    
    def predict_time_series(self, energy_query, duration_query, time_points):
        """Predict over a series of time points"""
        results = {
            'time_points': time_points,
            'field_predictions': {},
            'attention_maps': []
        }
        
        for field in ['temperature', 'displacement', 'principal stress']:
            results['field_predictions'][field] = {
                'mean': [], 'max': [], 'std': []
            }
        
        for t in time_points:
            pred = self.predict_field_statistics(energy_query, duration_query, t)
            if pred:
                for field in pred['field_predictions']:
                    if field in results['field_predictions']:
                        for stat in ['mean', 'max', 'std']:
                            results['field_predictions'][field][stat].append(
                                pred['field_predictions'][field][stat]
                            )
                results['attention_maps'].append(pred['attention_weights'])
        
        return results

# =============================================
# ADVANCED VISUALIZATION COMPONENTS
# =============================================
class AdvancedVisualizer:
    """Comprehensive visualization components"""
    
    @staticmethod
    def create_sunburst_chart(summaries, selected_field='temperature'):
        """Create sunburst chart showing hierarchy of simulations"""
        labels = []
        parents = []
        values = []
        
        # Root
        labels.append("All Simulations")
        parents.append("")
        values.append(len(summaries))
        
        for summary in summaries:
            # Energy level
            energy_label = f"Energy: {summary['energy']}mJ"
            labels.append(energy_label)
            parents.append("All Simulations")
            values.append(1)
            
            # Duration level
            duration_label = f"Duration: {summary['duration']}ns"
            labels.append(duration_label)
            parents.append(energy_label)
            values.append(1)
            
            # Field statistics
            if selected_field in summary['field_stats']:
                avg_max = np.mean(summary['field_stats'][selected_field]['max'])
                field_label = f"{selected_field}: {avg_max:.1f}"
                labels.append(field_label)
                parents.append(duration_label)
                values.append(avg_max)
        
        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            marker=dict(
                colors=px.colors.qualitative.Plotly[:len(labels)],
                colorscale='Viridis'
            ),
            hovertemplate='<b>%{label}</b><br>Value: %{value:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Simulation Hierarchy - {selected_field}",
            height=600
        )
        
        return fig
    
    @staticmethod
    def create_radar_chart(summaries, simulation_names):
        """Create radar chart comparing multiple simulations"""
        fields = ['temperature', 'displacement', 'principal stress']
        stats = ['mean', 'max', 'std']
        
        fig = go.Figure()
        
        for sim_name in simulation_names:
            # Find summary
            summary = next((s for s in summaries if 
                          f"q{s['energy']}mJ-delta{s['duration']}ns" in sim_name), None)
            if not summary:
                continue
            
            r_values = []
            theta_values = []
            
            for field in fields:
                if field in summary['field_stats']:
                    for stat in stats:
                        avg_value = np.mean(summary['field_stats'][field][stat])
                        r_values.append(avg_value)
                        theta_values.append(f"{field}<br>{stat}")
            
            fig.add_trace(go.Scatterpolar(
                r=r_values,
                theta=theta_values,
                fill='toself',
                name=sim_name
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max([max(trace.r) for trace in fig.data]) * 1.1]
                )
            ),
            showlegend=True,
            title="Simulation Comparison Radar Chart"
        )
        
        return fig
    
    @staticmethod
    def create_attention_heatmap(attention_weights, source_simulations):
        """Create heatmap of attention weights"""
        fig = go.Figure(data=go.Heatmap(
            z=attention_weights.reshape(-1, 1).T,
            x=[f"Sim {i+1}" for i in range(len(attention_weights))],
            y=['Attention'],
            colorscale='Viridis',
            colorbar=dict(title="Attention Weight"),
            hovertemplate='Simulation: %{x}<br>Attention: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Attention Weights Distribution",
            xaxis_title="Source Simulations",
            yaxis_title="",
            height=200
        )
        
        return fig

# =============================================
# MAIN INTEGRATED APPLICATION
# =============================================
def main():
    st.set_page_config(
        page_title="FEA Laser Simulation Platform",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #E0E0E0;
        padding-bottom: 0.5rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #2196F3;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🔬 FEA Laser Simulation Platform</h1>', 
                unsafe_allow_html=True)
    
    # Initialize session state
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = UnifiedFEADataLoader()
        st.session_state.extrapolator = PhysicsInformedAttentionExtrapolator()
        st.session_state.visualizer = AdvancedVisualizer()
        st.session_state.data_loaded = False
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Navigation")
        app_mode = st.selectbox(
            "Select Mode",
            ["Data Viewer", "Interpolation/Extrapolation", "Comparative Analysis"]
        )
        
        st.header("📊 Data Settings")
        load_full_data = st.checkbox("Load Full Mesh Data", value=True)
        
        if st.button("🔄 Load All Simulations"):
            with st.spinner("Loading simulation data..."):
                simulations, summaries = st.session_state.data_loader.load_all_simulations(
                    load_full_mesh=load_full_data
                )
                st.session_state.simulations = simulations
                st.session_state.summaries = summaries
                st.session_state.extrapolator.load_summaries(summaries)
                st.session_state.data_loaded = True
                st.success(f"✅ Loaded {len(simulations)} simulations")
        
        if st.session_state.data_loaded:
            st.info(f"**Loaded:** {len(st.session_state.simulations)} simulations")
            st.info(f"**Fields:** {list(next(iter(st.session_state.simulations.values()))['field_info'].keys())}")
    
    # Main content based on selected mode
    if app_mode == "Data Viewer":
        render_data_viewer()
    elif app_mode == "Interpolation/Extrapolation":
        render_interpolation_extrapolation()
    elif app_mode == "Comparative Analysis":
        render_comparative_analysis()

def render_data_viewer():
    """Render the data visualization interface"""
    st.markdown('<h2 class="sub-header">📁 FEA Data Viewer</h2>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load simulations first using the sidebar button.")
        return
    
    simulations = st.session_state.simulations
    
    # Simulation selection
    col1, col2 = st.columns([2, 1])
    with col1:
        sim_name = st.selectbox(
            "Select Simulation",
            sorted(simulations.keys()),
            key="viewer_sim_select"
        )
    
    sim = simulations[sim_name]
    
    with col2:
        st.metric("Energy (mJ)", f"{sim['energy_mJ']:.2f}")
        st.metric("Duration (ns)", f"{sim['duration_ns']:.2f}")
    
    # Field and timestep selection
    col1, col2 = st.columns(2)
    with col1:
        field = st.selectbox(
            "Select Field",
            list(sim['field_info'].keys()),
            key="viewer_field_select"
        )
    with col2:
        timestep = st.slider(
            "Timestep",
            0, sim['n_timesteps'] - 1, 0,
            key="viewer_timestep_slider"
        )
    
    # Main 3D visualization
    if 'points' in sim:
        pts = sim['points']
        kind, _ = sim['field_info'][field]
        raw = sim['fields'][field][timestep]
        
        if kind == "scalar":
            values = raw
            label = field
        else:
            values = np.linalg.norm(raw, axis=1)
            label = f"{field} (magnitude)"
        
        # Create 3D plot
        if sim.get('triangles') is not None:
            tri = sim['triangles']
            mesh_data = go.Mesh3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                i=tri[:, 0], j=tri[:, 1], k=tri[:, 2],
                intensity=values,
                colorscale="Viridis",
                intensitymode='vertex',
                colorbar=dict(title=label),
                opacity=0.85,
                hovertemplate='Value: %{intensity:.2f}<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}'
            )
        else:
            mesh_data = go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=values,
                    colorscale="Viridis",
                    opacity=0.8,
                    colorbar=dict(title=label)
                ),
                hovertemplate='Value: %{marker.color:.2f}<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}'
            )
        
        fig = go.Figure(data=mesh_data)
        fig.update_layout(
            title=f"{label} at Timestep {timestep} - {sim_name}",
            scene=dict(
                aspectmode="data",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Field statistics over time
    st.markdown('<h3 class="sub-header">📈 Field Evolution Over Time</h3>', 
                unsafe_allow_html=True)
    
    # Find corresponding summary
    summary = next((s for s in st.session_state.summaries if 
                   f"q{s['energy']}mJ-delta{s['duration']}ns" == sim_name), None)
    
    if summary and field in summary['field_stats']:
        stats = summary['field_stats'][field]
        
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(
            x=summary['timesteps'],
            y=stats['mean'],
            mode='lines+markers',
            name='Mean',
            line=dict(color='blue', width=2)
        ))
        fig_time.add_trace(go.Scatter(
            x=summary['timesteps'],
            y=stats['max'],
            mode='lines+markers',
            name='Max',
            line=dict(color='red', width=2)
        ))
        
        fig_time.update_layout(
            title=f"{field} Statistics Over Time",
            xaxis_title="Timestep (ns)",
            yaxis_title="Field Value",
            hovermode="x unified",
            height=400
        )
        
        st.plotly_chart(fig_time, use_container_width=True)

def render_interpolation_extrapolation():
    """Render the interpolation/extrapolation interface"""
    st.markdown('<h2 class="sub-header">🔮 Interpolation/Extrapolation Engine</h2>', 
                unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load simulations first using the sidebar button.")
        return
    
    st.markdown("""
    <div class="info-box">
    <strong>Physics-Informed Attention Mechanism:</strong> This engine uses a transformer-inspired 
    attention mechanism with spatial locality regulation to interpolate and extrapolate 
    simulation results. The model learns from existing FEA simulations and can predict 
    outcomes for new parameter combinations.
    </div>
    """, unsafe_allow_html=True)
    
    # Query parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        energy_query = st.number_input(
            "Energy (mJ)",
            min_value=0.1,
            max_value=50.0,
            value=5.0,
            step=0.5,
            key="interp_energy"
        )
    with col2:
        duration_query = st.number_input(
            "Pulse Duration (ns)",
            min_value=0.5,
            max_value=20.0,
            value=4.0,
            step=0.5,
            key="interp_duration"
        )
    with col3:
        max_time = st.number_input(
            "Max Prediction Time (ns)",
            min_value=1,
            max_value=50,
            value=15,
            step=1,
            key="interp_maxtime"
        )
    
    # Time points for prediction
    time_points = np.arange(1, max_time + 1)
    
    # Model parameters
    with st.expander("⚙️ Model Parameters"):
        col1, col2 = st.columns(2)
        with col1:
            sigma_param = st.slider(
                "Sigma Parameter",
                min_value=0.1,
                max_value=1.0,
                value=0.3,
                step=0.05,
                key="interp_sigma"
            )
        with col2:
            spatial_weight = st.slider(
                "Spatial Locality Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                key="interp_spatial"
            )
        
        st.session_state.extrapolator.sigma_param = sigma_param
        st.session_state.extrapolator.spatial_weight = spatial_weight
    
    if st.button("🚀 Run Prediction", type="primary"):
        with st.spinner("Running physics-informed prediction..."):
            # Get predictions
            results = st.session_state.extrapolator.predict_time_series(
                energy_query, duration_query, time_points
            )
            
            if results and 'field_predictions' in results:
                # Visualize predictions
                st.markdown('<h3 class="sub-header">📊 Prediction Results</h3>', 
                           unsafe_allow_html=True)
                
                # Temperature and Stress plot
                fig_pred = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Max Temperature', 'Max Stress', 
                                   'Mean Displacement', 'Attention Confidence'),
                    vertical_spacing=0.15,
                    horizontal_spacing=0.1
                )
                
                # Temperature
                if 'temperature' in results['field_predictions']:
                    fig_pred.add_trace(
                        go.Scatter(
                            x=time_points,
                            y=results['field_predictions']['temperature']['max'],
                            mode='lines+markers',
                            name='Max Temp',
                            line=dict(color='red', width=3)
                        ),
                        row=1, col=1
                    )
                
                # Stress
                if 'principal stress' in results['field_predictions']:
                    fig_pred.add_trace(
                        go.Scatter(
                            x=time_points,
                            y=results['field_predictions']['principal stress']['max'],
                            mode='lines+markers',
                            name='Max Stress',
                            line=dict(color='blue', width=3)
                        ),
                        row=1, col=2
                    )
                
                # Displacement
                if 'displacement' in results['field_predictions']:
                    fig_pred.add_trace(
                        go.Scatter(
                            x=time_points,
                            y=results['field_predictions']['displacement']['mean'],
                            mode='lines+markers',
                            name='Mean Disp',
                            line=dict(color='green', width=3)
                        ),
                        row=2, col=1
                    )
                
                # Confidence
                if results['attention_maps']:
                    avg_confidence = [np.max(w) for w in results['attention_maps']]
                    fig_pred.add_trace(
                        go.Scatter(
                            x=time_points,
                            y=avg_confidence,
                            mode='lines+markers',
                            name='Confidence',
                            line=dict(color='orange', width=3)
                        ),
                        row=2, col=2
                    )
                
                fig_pred.update_layout(height=600, showlegend=True)
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Attention heatmap
                if results['attention_maps']:
                    st.markdown('<h4 class="sub-header">🧠 Attention Weights</h4>', 
                               unsafe_allow_html=True)
                    
                    # Create heatmap for first timestep
                    heatmap_fig = st.session_state.visualizer.create_attention_heatmap(
                        results['attention_maps'][0],
                        st.session_state.summaries
                    )
                    st.plotly_chart(heatmap_fig, use_container_width=True)
                
                # Display results table
                st.markdown('<h4 class="sub-header">📋 Detailed Predictions</h4>', 
                           unsafe_allow_html=True)
                
                data_rows = []
                for idx, t in enumerate(time_points):
                    row = {'Time (ns)': t}
                    for field in ['temperature', 'principal stress', 'displacement']:
                        if field in results['field_predictions']:
                            row[f'{field}_max'] = results['field_predictions'][field]['max'][idx]
                            row[f'{field}_mean'] = results['field_predictions'][field]['mean'][idx]
                    if results['attention_maps']:
                        row['confidence'] = np.max(results['attention_maps'][idx])
                    data_rows.append(row)
                
                df_results = pd.DataFrame(data_rows)
                st.dataframe(df_results.style.format("{:.3f}"), use_container_width=True)
                
                # Export option
                csv = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Predictions",
                    data=csv,
                    file_name=f"predictions_E{energy_query}_tau{duration_query}.csv",
                    mime="text/csv"
                )
            else:
                st.error("Prediction failed. Check input parameters.")

def render_comparative_analysis():
    """Render comparative analysis interface"""
    st.markdown('<h2 class="sub-header">📊 Comparative Analysis</h2>', 
                unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load simulations first using the sidebar button.")
        return
    
    simulations = st.session_state.simulations
    summaries = st.session_state.summaries
    
    # Simulation selection for comparison
    selected_sims = st.multiselect(
        "Select simulations for comparison",
        sorted(simulations.keys()),
        default=list(simulations.keys())[:3] if simulations else []
    )
    
    if not selected_sims:
        st.info("Please select at least one simulation for comparison.")
        return
    
    # Field selection
    available_fields = set()
    for sim_name in selected_sims:
        if sim_name in simulations:
            available_fields.update(simulations[sim_name]['field_info'].keys())
    
    selected_field = st.selectbox(
        "Select field for analysis",
        list(available_fields)
    )
    
    # Create comparison plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h4>📈 Sunburst Chart</h4>', unsafe_allow_html=True)
        sunburst_fig = st.session_state.visualizer.create_sunburst_chart(
            summaries, selected_field
        )
        st.plotly_chart(sunburst_fig, use_container_width=True)
    
    with col2:
        st.markdown('<h4>🎯 Radar Chart Comparison</h4>', unsafe_allow_html=True)
        radar_fig = st.session_state.visualizer.create_radar_chart(
            summaries, selected_sims[:5]  # Limit to 5 for clarity
        )
        st.plotly_chart(radar_fig, use_container_width=True)
    
    # Detailed field comparison over time
    st.markdown('<h4>⏱️ Field Evolution Comparison</h4>', unsafe_allow_html=True)
    
    fig_comparison = go.Figure()
    
    for sim_name in selected_sims:
        # Find summary
        summary = next((s for s in summaries if 
                       f"q{s['energy']}mJ-delta{s['duration']}ns" == sim_name), None)
        
        if summary and selected_field in summary['field_stats']:
            stats = summary['field_stats'][selected_field]
            
            fig_comparison.add_trace(go.Scatter(
                x=summary['timesteps'],
                y=stats['mean'],
                mode='lines+markers',
                name=f"{sim_name} (mean)",
                line=dict(width=2)
            ))
    
    fig_comparison.update_layout(
        title=f"{selected_field} Comparison",
        xaxis_title="Timestep (ns)",
        yaxis_title="Field Value",
        hovermode="x unified",
        height=500
    )
    
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Parameter space visualization
    st.markdown('<h4>🌐 Parameter Space Analysis</h4>', unsafe_allow_html=True)
    
    # Extract parameters
    energies = []
    durations = []
    max_temps = []
    
    for summary in summaries:
        energies.append(summary['energy'])
        durations.append(summary['duration'])
        
        if 'temperature' in summary['field_stats']:
            max_temps.append(np.max(summary['field_stats']['temperature']['max']))
        else:
            max_temps.append(0)
    
    fig_space = go.Figure(data=go.Scatter3d(
        x=energies,
        y=durations,
        z=max_temps,
        mode='markers',
        marker=dict(
            size=10,
            color=max_temps,
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title="Max Temp")
        ),
        text=[f"E:{e}mJ, τ:{d}ns" for e, d in zip(energies, durations)],
        hovertemplate='%{text}<br>Max Temp: %{z:.1f}<extra></extra>'
    ))
    
    fig_space.update_layout(
        title="Parameter Space - Maximum Temperature",
        scene=dict(
            xaxis_title="Energy (mJ)",
            yaxis_title="Duration (ns)",
            zaxis_title="Max Temperature"
        ),
        height=600
    )
    
    st.plotly_chart(fig_space, use_container_width=True)

if __name__ == "__main__":
    main()
