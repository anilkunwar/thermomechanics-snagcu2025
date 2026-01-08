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
        
        if not folders:
            st.warning(f"No simulation folders found in {FEA_SOLUTIONS_DIR}")
            st.info("Expected folder pattern: q{energy}mJ-delta{duration}ns")
            return simulations, summaries
        
        for folder in folders:
            name = os.path.basename(folder)
            energy, duration = _self.parse_folder_name(name)
            if energy is None:
                continue
                
            vtu_files = sorted(glob.glob(os.path.join(folder, "a_t????.vtu")))
            if not vtu_files:
                st.warning(f"No VTU files found in {name}")
                continue
            
            # Load first file to get structure
            try:
                mesh0 = meshio.read(vtu_files[0])
                
                # Check if mesh has point data
                if not mesh0.point_data:
                    st.warning(f"No point data found in {vtu_files[0]}")
                    continue
                
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
                        try:
                            mesh = meshio.read(vtu_files[t])
                            for key in sim_data['field_info'].keys():
                                if key in mesh.point_data:
                                    fields[key][t] = mesh.point_data[key].astype(np.float32)
                        except Exception as e:
                            st.warning(f"Error loading timestep {t} in {name}: {e}")
                            continue
                    
                    sim_data.update({
                        'points': points,
                        'fields': fields,
                        'triangles': triangles
                    })
                
                # Create summary statistics (for interpolation/extrapolation)
                summary = _self.extract_summary_statistics(vtu_files, energy, duration, name)
                summaries.append(summary)
                
                simulations[name] = sim_data
                
            except Exception as e:
                st.warning(f"Error loading {name}: {str(e)}")
                continue
        
        return simulations, summaries
    
    def extract_summary_statistics(self, vtu_files, energy, duration, name):
        """Extract summary statistics from VTU files"""
        summary = {
            'name': name,
            'energy': energy,
            'duration': duration,
            'timesteps': [],
            'field_stats': {}
        }
        
        for idx, vtu_file in enumerate(vtu_files):
            try:
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
                        summary['field_stats'][field_name]['min'].append(float(np.nanmin(data)))
                        summary['field_stats'][field_name]['max'].append(float(np.nanmax(data)))
                        summary['field_stats'][field_name]['mean'].append(float(np.nanmean(data)))
                        summary['field_stats'][field_name]['std'].append(float(np.nanstd(data)))
                        summary['field_stats'][field_name]['q25'].append(float(np.nanpercentile(data, 25)))
                        summary['field_stats'][field_name]['q50'].append(float(np.nanpercentile(data, 50)))
                        summary['field_stats'][field_name]['q75'].append(float(np.nanpercentile(data, 75)))
                    else:
                        # For vector fields, take magnitude
                        magnitude = np.linalg.norm(data, axis=1)
                        summary['field_stats'][field_name]['min'].append(float(np.nanmin(magnitude)))
                        summary['field_stats'][field_name]['max'].append(float(np.nanmax(magnitude)))
                        summary['field_stats'][field_name]['mean'].append(float(np.nanmean(magnitude)))
                        summary['field_stats'][field_name]['std'].append(float(np.nanstd(magnitude)))
            except Exception as e:
                st.warning(f"Error processing {vtu_file}: {e}")
                continue
        
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
        self.fitted = False
        
    def load_summaries(self, summaries):
        """Load summary statistics from data loader"""
        self.source_db = summaries
        
        if not summaries:
            return
        
        # Prepare data for scaling
        all_embeddings = []
        for summary in summaries:
            for t in summary['timesteps']:
                emb = self._compute_embedding(summary['energy'], summary['duration'], t)
                all_embeddings.append(emb)
        
        if all_embeddings:
            all_embeddings = np.array(all_embeddings)
            if len(all_embeddings) > 1:
                self.scaler.fit(all_embeddings)
                self.fitted = True
    
    def _compute_embedding(self, energy, duration, time):
        """Compute physics-aware embedding with enhanced features"""
        # Log-energy scaling
        logE = np.log1p(energy)
        
        # Dimensionless parameters
        energy_density = energy / max(duration, 1e-6)
        time_ratio = time / max(duration, 1e-3)
        
        # Thermal diffusion proxy
        thermal_diffusion = time * 0.1 / max(duration, 1e-6)
        
        # Strain rate proxy
        strain_rate = energy_density / max(time, 1e-6)
        
        return np.array([
            logE,
            duration,
            time,
            energy_density,
            time_ratio,
            thermal_diffusion,
            strain_rate
        ], dtype=np.float32)
    
    def _normalize_embedding(self, embedding):
        """Normalize embedding using fitted scaler"""
        if self.fitted and len(embedding.shape) == 2:
            return self.scaler.transform(embedding)
        elif self.fitted:
            return self.scaler.transform([embedding])[0]
        return embedding
    
    def _multi_head_attention(self, query_embedding, source_embeddings, values):
        """Multi-head attention mechanism inspired by transformers"""
        n_sources = len(source_embeddings)
        if n_sources == 0:
            return None, None
        
        weights = np.zeros(n_sources)
        
        try:
            # Normalize embeddings
            query_norm = self._normalize_embedding(query_embedding)
            source_norm = self._normalize_embedding(source_embeddings)
            
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
                if self.spatial_weight > 0 and n_sources > 1:
                    # Add spatial correlation (simplified - in practice would use actual coordinates)
                    spatial_corr = 1.0 / (1.0 + np.arange(n_sources) / n_sources)
                    scores = (1 - self.spatial_weight) * scores + self.spatial_weight * spatial_corr
                
                weights += scores / self.n_heads
            
            # Softmax normalization
            weights = np.exp(weights - np.max(weights))
            weights = weights / (np.sum(weights) + 1e-10)
            
            # Weighted prediction
            if len(values) > 0 and values.ndim > 1:
                prediction = np.sum(weights[:, np.newaxis] * values, axis=0)
            elif len(values) > 0:
                prediction = np.sum(weights * values)
            else:
                prediction = np.zeros(1)
            
            return prediction, weights
            
        except Exception as e:
            st.error(f"Error in attention mechanism: {e}")
            return None, None
    
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
                    if idx < len(stats['mean']):
                        field_vals.extend([
                            stats['mean'][idx],
                            stats['max'][idx],
                            stats['std'][idx]
                        ])
                if field_vals:
                    all_values.append(field_vals)
        
        if not all_embeddings or not all_values:
            return None
        
        all_embeddings = np.array(all_embeddings)
        all_values = np.array(all_values)
        
        # Apply multi-head attention
        prediction, attention_weights = self._multi_head_attention(
            query_embedding, all_embeddings, all_values
        )
        
        if prediction is None:
            return None
        
        # Reconstruct field statistics from prediction
        result = {
            'prediction': prediction,
            'attention_weights': attention_weights,
            'confidence': np.max(attention_weights) if attention_weights is not None else 0.0,
            'field_predictions': {}
        }
        
        # Map predictions back to field statistics
        field_idx = 0
        for summary in self.source_db:
            for field in summary['field_stats']:
                if field not in result['field_predictions'] and field_idx + 2 < len(prediction):
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
        
        # Initialize field predictions structure
        common_fields = set()
        for summary in self.source_db:
            common_fields.update(summary['field_stats'].keys())
        
        for field in common_fields:
            results['field_predictions'][field] = {
                'mean': [], 'max': [], 'std': []
            }
        
        for t in time_points:
            pred = self.predict_field_statistics(energy_query, duration_query, t)
            if pred and 'field_predictions' in pred:
                for field in pred['field_predictions']:
                    if field in results['field_predictions']:
                        for stat in ['mean', 'max', 'std']:
                            results['field_predictions'][field][stat].append(
                                pred['field_predictions'][field][stat]
                            )
                if 'attention_weights' in pred and pred['attention_weights'] is not None:
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
        
        if not summaries:
            return go.Figure()
        
        # Root
        labels.append("All Simulations")
        parents.append("")
        values.append(len(summaries))
        
        for summary in summaries:
            # Energy level
            energy_label = f"Energy: {summary['energy']}mJ"
            if energy_label not in labels:
                labels.append(energy_label)
                parents.append("All Simulations")
                values.append(1)
            
            # Duration level
            duration_label = f"Duration: {summary['duration']}ns"
            sim_label = f"Sim: {summary['name']}"
            labels.append(sim_label)
            parents.append(energy_label)
            values.append(1)
            
            # Field statistics
            if selected_field in summary['field_stats']:
                stats = summary['field_stats'][selected_field]
                if stats['max']:
                    avg_max = np.mean(stats['max'])
                    field_label = f"{selected_field}: {avg_max:.1f}"
                    labels.append(field_label)
                    parents.append(sim_label)
                    values.append(avg_max)
        
        if len(labels) <= 1:
            return go.Figure()
        
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
        # Define default fields to look for
        possible_fields = ['temperature', 'displacement', 'principal stress', 'stress', 'strain']
        stats = ['mean', 'max']
        
        fig = go.Figure()
        
        # Find common fields across selected simulations
        common_fields = []
        for field in possible_fields:
            if any(field in summary['field_stats'] for summary in summaries 
                   if f"q{summary['energy']}mJ-delta{summary['duration']}ns" in simulation_names):
                common_fields.append(field)
        
        if not common_fields:
            # If no common fields, try to find any field
            for summary in summaries:
                if summary['field_stats']:
                    common_fields = list(summary['field_stats'].keys())[:3]
                    break
        
        if not common_fields:
            return fig
        
        for sim_name in simulation_names:
            # Find summary
            summary = next((s for s in summaries if 
                          f"q{s['energy']}mJ-delta{s['duration']}ns" in sim_name), None)
            if not summary:
                continue
            
            r_values = []
            theta_values = []
            
            for field in common_fields[:5]:  # Limit to 5 fields for clarity
                if field in summary['field_stats']:
                    stats_data = summary['field_stats'][field]
                    for stat in stats:
                        if stat in stats_data and stats_data[stat]:
                            # Use average across timesteps
                            avg_value = np.mean(stats_data[stat])
                            r_values.append(avg_value)
                            theta_values.append(f"{field[:10]}<br>{stat}")
            
            if r_values:  # Only add trace if we have data
                fig.add_trace(go.Scatterpolar(
                    r=r_values,
                    theta=theta_values,
                    fill='toself',
                    name=sim_name[:20]  # Truncate long names
                ))
        
        if fig.data:  # Only update layout if we have traces
            # Find maximum value across all traces
            max_values = []
            for trace in fig.data:
                if hasattr(trace, 'r') and trace.r:
                    max_values.append(max(trace.r))
            
            if max_values:
                max_value = max(max_values)
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, max_value * 1.1]
                        )
                    ),
                    showlegend=True,
                    title="Simulation Comparison Radar Chart",
                    height=500
                )
        
        return fig
    
    @staticmethod
    def create_attention_heatmap(attention_weights, source_simulations):
        """Create heatmap of attention weights"""
        if attention_weights is None or len(attention_weights) == 0:
            return go.Figure()
        
        fig = go.Figure(data=go.Heatmap(
            z=attention_weights.reshape(1, -1),
            x=[f"Source {i+1}" for i in range(len(attention_weights))],
            y=['Attention'],
            colorscale='Viridis',
            colorbar=dict(title="Attention Weight"),
            hovertemplate='Source: %{x}<br>Weight: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Attention Weights Distribution",
            xaxis_title="Source Simulations",
            yaxis_title="",
            height=200
        )
        
        return fig
    
    @staticmethod
    def create_field_evolution_chart(summaries, simulation_names, selected_field):
        """Create line chart showing field evolution over time"""
        fig = go.Figure()
        
        for sim_name in simulation_names:
            # Find summary
            summary = next((s for s in summaries if 
                          f"q{s['energy']}mJ-delta{s['duration']}ns" in sim_name), None)
            
            if summary and selected_field in summary['field_stats']:
                stats = summary['field_stats'][selected_field]
                
                if stats['mean'] and summary['timesteps']:
                    fig.add_trace(go.Scatter(
                        x=summary['timesteps'],
                        y=stats['mean'],
                        mode='lines+markers',
                        name=f"{sim_name} (mean)",
                        line=dict(width=2)
                    ))
        
        if fig.data:
            fig.update_layout(
                title=f"{selected_field} Evolution Comparison",
                xaxis_title="Timestep (ns)",
                yaxis_title="Field Value",
                hovermode="x unified",
                height=500
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
    .warning-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #FF9800;
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
        
        if st.button("🔄 Load All Simulations", type="primary"):
            with st.spinner("Loading simulation data..."):
                simulations, summaries = st.session_state.data_loader.load_all_simulations(
                    load_full_mesh=load_full_data
                )
                st.session_state.simulations = simulations
                st.session_state.summaries = summaries
                
                if simulations:
                    st.session_state.extrapolator.load_summaries(summaries)
                    st.session_state.data_loaded = True
                    st.success(f"✅ Loaded {len(simulations)} simulations")
                    
                    # Display loaded fields
                    if simulations:
                        first_sim = next(iter(simulations.values()))
                        if 'field_info' in first_sim:
                            fields = list(first_sim['field_info'].keys())
                            st.info(f"**Fields available:** {', '.join(fields[:5])}{'...' if len(fields) > 5 else ''}")
                else:
                    st.session_state.data_loaded = False
                    st.error("❌ No simulations loaded. Check the data directory.")
        
        if st.session_state.data_loaded:
            st.divider()
            st.markdown("### 📈 Loaded Data")
            st.info(f"**Simulations:** {len(st.session_state.simulations)}")
            
            if st.session_state.summaries:
                # Show available fields
                all_fields = set()
                for summary in st.session_state.summaries:
                    all_fields.update(summary['field_stats'].keys())
                
                if all_fields:
                    st.info(f"**Available Fields:** {len(all_fields)}")
                    with st.expander("View all fields"):
                        for field in sorted(all_fields):
                            st.write(f"• {field}")
    
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
        st.markdown('<div class="warning-box">Please load simulations first using the "Load All Simulations" button in the sidebar.</div>', 
                   unsafe_allow_html=True)
        
        # Show expected directory structure
        with st.expander("📁 Expected Directory Structure"):
            st.code("""
fea_solutions/
├── q0p5mJ-delta4p2ns/
│   ├── a_t0001.vtu
│   ├── a_t0002.vtu
│   ├── a_t0003.vtu
│   └── ...
├── q1p0mJ-delta2p0ns/
│   ├── a_t0001.vtu
│   ├── a_t0002.vtu
│   └── ...
└── ...
            """)
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
        st.metric("Timesteps", sim['n_timesteps'])
    
    if 'field_info' not in sim or not sim['field_info']:
        st.error("No field data available for this simulation.")
        return
    
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
            key="viewer_timestep_slider",
            help="Select the timestep to visualize (0-indexed)"
        )
    
    # Main 3D visualization
    if 'points' in sim and 'fields' in sim and field in sim['fields']:
        pts = sim['points']
        kind, _ = sim['field_info'][field]
        raw = sim['fields'][field][timestep]
        
        if kind == "scalar":
            values = raw
            label = field
        else:
            values = np.linalg.norm(raw, axis=1)
            label = f"{field} (magnitude)"
        
        # Remove NaN values for visualization
        valid_mask = ~np.isnan(values)
        if not np.any(valid_mask):
            st.error("No valid data points for visualization.")
            return
        
        pts_valid = pts[valid_mask]
        values_valid = values[valid_mask]
        
        # Create 3D plot
        if sim.get('triangles') is not None and len(pts_valid) > 0:
            tri = sim['triangles']
            # Filter triangles to only include valid points
            valid_triangles = []
            for triangle in tri:
                if all(valid_mask[triangle]):
                    valid_triangles.append(triangle)
            
            if valid_triangles:
                valid_triangles = np.array(valid_triangles)
                mesh_data = go.Mesh3d(
                    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                    i=valid_triangles[:, 0], j=valid_triangles[:, 1], k=valid_triangles[:, 2],
                    intensity=values,
                    colorscale="Viridis",
                    intensitymode='vertex',
                    colorbar=dict(title=label),
                    opacity=0.85,
                    hovertemplate='Value: %{intensity:.2f}<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}'
                )
            else:
                # Fallback to scatter plot if no valid triangles
                mesh_data = go.Scatter3d(
                    x=pts_valid[:, 0], y=pts_valid[:, 1], z=pts_valid[:, 2],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=values_valid,
                        colorscale="Viridis",
                        opacity=0.8,
                        colorbar=dict(title=label)
                    ),
                    hovertemplate='Value: %{marker.color:.2f}<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}'
                )
        else:
            # Scatter plot for point cloud
            mesh_data = go.Scatter3d(
                x=pts_valid[:, 0], y=pts_valid[:, 1], z=pts_valid[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=values_valid,
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
        
        # Field statistics
        st.markdown('<h3 class="sub-header">📊 Field Statistics</h3>', 
                   unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Min", f"{np.nanmin(values):.3f}")
        with col2:
            st.metric("Max", f"{np.nanmax(values):.3f}")
        with col3:
            st.metric("Mean", f"{np.nanmean(values):.3f}")
        with col4:
            st.metric("Std Dev", f"{np.nanstd(values):.3f}")
    
    # Field statistics over time
    st.markdown('<h3 class="sub-header">📈 Field Evolution Over Time</h3>', 
               unsafe_allow_html=True)
    
    # Find corresponding summary
    summary = next((s for s in st.session_state.summaries if s['name'] == sim_name), None)
    
    if summary and field in summary['field_stats']:
        stats = summary['field_stats'][field]
        
        fig_time = go.Figure()
        
        if stats['mean'] and summary['timesteps']:
            fig_time.add_trace(go.Scatter(
                x=summary['timesteps'],
                y=stats['mean'],
                mode='lines+markers',
                name='Mean',
                line=dict(color='blue', width=2)
            ))
        
        if stats['max'] and summary['timesteps']:
            fig_time.add_trace(go.Scatter(
                x=summary['timesteps'],
                y=stats['max'],
                mode='lines+markers',
                name='Max',
                line=dict(color='red', width=2)
            ))
        
        if fig_time.data:
            fig_time.update_layout(
                title=f"{field} Statistics Over Time",
                xaxis_title="Timestep (ns)",
                yaxis_title="Field Value",
                hovermode="x unified",
                height=400
            )
            
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.info("No time series data available for this field.")
    else:
        st.info("No summary statistics available for this simulation.")

def render_interpolation_extrapolation():
    """Render the interpolation/extrapolation interface"""
    st.markdown('<h2 class="sub-header">🔮 Interpolation/Extrapolation Engine</h2>', 
               unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.markdown('<div class="warning-box">Please load simulations first using the "Load All Simulations" button in the sidebar.</div>', 
                   unsafe_allow_html=True)
        return
    
    st.markdown("""
    <div class="info-box">
    <strong>Physics-Informed Attention Mechanism:</strong> This engine uses a transformer-inspired 
    attention mechanism with spatial locality regulation to interpolate and extrapolate 
    simulation results. The model learns from existing FEA simulations and can predict 
    outcomes for new parameter combinations.
    </div>
    """, unsafe_allow_html=True)
    
    # Display loaded simulations info
    with st.expander("📋 Loaded Simulations Info"):
        if st.session_state.summaries:
            df_simulations = pd.DataFrame([{
                'Name': s['name'],
                'Energy (mJ)': s['energy'],
                'Duration (ns)': s['duration'],
                'Timesteps': len(s['timesteps']),
                'Fields': ', '.join(list(s['field_stats'].keys())[:3]) + ('...' if len(s['field_stats']) > 3 else '')
            } for s in st.session_state.summaries])
            st.dataframe(df_simulations, use_container_width=True)
    
    # Query parameters
    st.markdown('<h3 class="sub-header">🎯 Query Parameters</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        # Get range from loaded data
        if st.session_state.summaries:
            min_energy = min(s['energy'] for s in st.session_state.summaries)
            max_energy = max(s['energy'] for s in st.session_state.summaries)
        else:
            min_energy, max_energy = 0.1, 50.0
            
        energy_query = st.number_input(
            "Energy (mJ)",
            min_value=float(min_energy * 0.5),
            max_value=float(max_energy * 2.0),
            value=float((min_energy + max_energy) / 2),
            step=0.5,
            key="interp_energy",
            help=f"Loaded range: {min_energy:.1f} - {max_energy:.1f} mJ"
        )
    
    with col2:
        if st.session_state.summaries:
            min_duration = min(s['duration'] for s in st.session_state.summaries)
            max_duration = max(s['duration'] for s in st.session_state.summaries)
        else:
            min_duration, max_duration = 0.5, 20.0
            
        duration_query = st.number_input(
            "Pulse Duration (ns)",
            min_value=float(min_duration * 0.5),
            max_value=float(max_duration * 2.0),
            value=float((min_duration + max_duration) / 2),
            step=0.5,
            key="interp_duration",
            help=f"Loaded range: {min_duration:.1f} - {max_duration:.1f} ns"
        )
    
    with col3:
        max_time = st.number_input(
            "Max Prediction Time (ns)",
            min_value=1,
            max_value=100,
            value=20,
            step=1,
            key="interp_maxtime"
        )
    
    # Time points for prediction
    time_points = np.arange(1, max_time + 1)
    
    # Model parameters
    with st.expander("⚙️ Model Parameters"):
        col1, col2, col3 = st.columns(3)
        with col1:
            sigma_param = st.slider(
                "Sigma Parameter",
                min_value=0.1,
                max_value=1.0,
                value=0.3,
                step=0.05,
                key="interp_sigma",
                help="Controls the width of attention kernel"
            )
        with col2:
            spatial_weight = st.slider(
                "Spatial Locality Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                key="interp_spatial",
                help="Weight for spatial correlation"
            )
        with col3:
            n_heads = st.slider(
                "Number of Attention Heads",
                min_value=1,
                max_value=8,
                value=4,
                step=1,
                key="interp_heads",
                help="More heads capture different aspects of similarity"
            )
        
        st.session_state.extrapolator.sigma_param = sigma_param
        st.session_state.extrapolator.spatial_weight = spatial_weight
        st.session_state.extrapolator.n_heads = n_heads
    
    if st.button("🚀 Run Prediction", type="primary"):
        with st.spinner("Running physics-informed prediction..."):
            # Get predictions
            results = st.session_state.extrapolator.predict_time_series(
                energy_query, duration_query, time_points
            )
            
            if results and 'field_predictions' in results and results['field_predictions']:
                # Visualize predictions
                st.markdown('<h3 class="sub-header">📊 Prediction Results</h3>', 
                           unsafe_allow_html=True)
                
                # Determine which fields to plot
                available_fields = list(results['field_predictions'].keys())
                
                if not available_fields:
                    st.warning("No field predictions available.")
                    return
                
                # Create subplots based on available fields
                n_fields = min(len(available_fields), 4)
                n_rows = (n_fields + 1) // 2
                
                fig_pred = make_subplots(
                    rows=n_rows, cols=2,
                    subplot_titles=[f"{field}" for field in available_fields[:n_fields]],
                    vertical_spacing=0.15,
                    horizontal_spacing=0.1
                )
                
                for idx, field in enumerate(available_fields[:n_fields]):
                    row = (idx // 2) + 1
                    col = (idx % 2) + 1
                    
                    if 'max' in results['field_predictions'][field]:
                        fig_pred.add_trace(
                            go.Scatter(
                                x=time_points,
                                y=results['field_predictions'][field]['max'],
                                mode='lines+markers',
                                name=f'{field} (max)',
                                line=dict(width=3)
                            ),
                            row=row, col=col
                        )
                    
                    if 'mean' in results['field_predictions'][field]:
                        fig_pred.add_trace(
                            go.Scatter(
                                x=time_points,
                                y=results['field_predictions'][field]['mean'],
                                mode='lines+markers',
                                name=f'{field} (mean)',
                                line=dict(width=2, dash='dash')
                            ),
                            row=row, col=col
                        )
                
                fig_pred.update_layout(
                    height=300 * n_rows,
                    showlegend=True,
                    title_text="Field Predictions Over Time"
                )
                
                for i in range(1, n_rows * 2 + 1, 2):
                    fig_pred.update_xaxes(title_text="Time (ns)", row=(i+1)//2, col=1)
                    fig_pred.update_yaxes(title_text="Value", row=(i+1)//2, col=1)
                
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Confidence metrics
                st.markdown('<h4 class="sub-header">📈 Confidence Metrics</h4>', 
                           unsafe_allow_html=True)
                
                if results['attention_maps']:
                    avg_confidence = [np.max(w) if w is not None else 0.0 for w in results['attention_maps']]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        avg_conf = np.mean(avg_confidence[-5:]) if len(avg_confidence) >= 5 else np.mean(avg_confidence)
                        st.metric("Average Confidence (last 5 steps)", f"{avg_conf:.3f}")
                        
                        if avg_conf < 0.3:
                            st.warning("⚠️ Low confidence: query is far from training data")
                        elif avg_conf < 0.6:
                            st.info("ℹ️ Moderate confidence: query is in extrapolation region")
                        else:
                            st.success("✅ High confidence: query is well-supported by training data")
                    
                    with col2:
                        # Plot confidence over time
                        fig_conf = go.Figure()
                        fig_conf.add_trace(go.Scatter(
                            x=time_points,
                            y=avg_confidence,
                            mode='lines+markers',
                            line=dict(color='orange', width=2)
                        ))
                        fig_conf.update_layout(
                            title="Prediction Confidence Over Time",
                            xaxis_title="Time (ns)",
                            yaxis_title="Confidence",
                            height=300
                        )
                        st.plotly_chart(fig_conf, use_container_width=True)
                
                # Attention heatmap
                if results['attention_maps'] and len(results['attention_maps']) > 0:
                    st.markdown('<h4 class="sub-header">🧠 Attention Weights</h4>', 
                               unsafe_allow_html=True)
                    
                    # Create heatmap for first timestep
                    heatmap_fig = st.session_state.visualizer.create_attention_heatmap(
                        results['attention_maps'][0],
                        st.session_state.summaries
                    )
                    if heatmap_fig.data:
                        st.plotly_chart(heatmap_fig, use_container_width=True)
                
                # Display results table
                st.markdown('<h4 class="sub-header">📋 Detailed Predictions</h4>', 
                           unsafe_allow_html=True)
                
                data_rows = []
                for idx, t in enumerate(time_points):
                    row = {'Time (ns)': t}
                    for field in available_fields:
                        if field in results['field_predictions']:
                            if 'max' in results['field_predictions'][field] and idx < len(results['field_predictions'][field]['max']):
                                row[f'{field}_max'] = results['field_predictions'][field]['max'][idx]
                            if 'mean' in results['field_predictions'][field] and idx < len(results['field_predictions'][field]['mean']):
                                row[f'{field}_mean'] = results['field_predictions'][field]['mean'][idx]
                    if results['attention_maps'] and idx < len(results['attention_maps']):
                        row['confidence'] = np.max(results['attention_maps'][idx]) if results['attention_maps'][idx] is not None else 0.0
                    data_rows.append(row)
                
                if data_rows:
                    df_results = pd.DataFrame(data_rows)
                    
                    # Format the dataframe
                    format_dict = {}
                    for col in df_results.columns:
                        if col != 'Time (ns)':
                            format_dict[col] = "{:.3f}"
                    
                    st.dataframe(df_results.style.format(format_dict), use_container_width=True)
                    
                    # Export option
                    csv = df_results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Predictions as CSV",
                        data=csv,
                        file_name=f"predictions_E{energy_query}_tau{duration_query}.csv",
                        mime="text/csv",
                        type="secondary"
                    )
            else:
                st.error("Prediction failed. No results generated. Check input parameters and loaded data.")

def render_comparative_analysis():
    """Render comparative analysis interface"""
    st.markdown('<h2 class="sub-header">📊 Comparative Analysis</h2>', 
               unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.markdown('<div class="warning-box">Please load simulations first using the "Load All Simulations" button in the sidebar.</div>', 
                   unsafe_allow_html=True)
        return
    
    simulations = st.session_state.simulations
    summaries = st.session_state.summaries
    
    if not summaries:
        st.error("No summary data available for analysis.")
        return
    
    # Simulation selection for comparison
    st.markdown('<h3 class="sub-header">🔍 Select Simulations for Comparison</h3>', 
               unsafe_allow_html=True)
    
    available_simulations = sorted(simulations.keys())
    selected_sims = st.multiselect(
        "Select simulations",
        available_simulations,
        default=available_simulations[:min(3, len(available_simulations))] if available_simulations else [],
        help="Select up to 5 simulations for comparison"
    )
    
    if not selected_sims:
        st.info("Please select at least one simulation for comparison.")
        return
    
    # Limit to 5 simulations for clarity
    if len(selected_sims) > 5:
        st.warning(f"Limited to first 5 simulations for clarity. Showing: {', '.join(selected_sims[:5])}")
        selected_sims = selected_sims[:5]
    
    # Field selection
    st.markdown('<h3 class="sub-header">📈 Select Field for Analysis</h3>', 
               unsafe_allow_html=True)
    
    # Find available fields across selected simulations
    available_fields = set()
    for sim_name in selected_sims:
        sim = simulations.get(sim_name)
        if sim and 'field_info' in sim:
            available_fields.update(sim['field_info'].keys())
    
    if not available_fields:
        # Try to get fields from summaries
        for sim_name in selected_sims:
            summary = next((s for s in summaries if s['name'] == sim_name), None)
            if summary:
                available_fields.update(summary['field_stats'].keys())
    
    if not available_fields:
        st.error("No field data available for the selected simulations.")
        return
    
    selected_field = st.selectbox(
        "Select field",
        sorted(available_fields),
        key="comparison_field"
    )
    
    # Create comparison visualizations
    st.markdown('<h3 class="sub-header">📊 Comparison Visualizations</h3>', 
               unsafe_allow_html=True)
    
    # Create tabs for different visualization types
    tab1, tab2, tab3 = st.tabs(["📈 Sunburst Chart", "🎯 Radar Chart", "⏱️ Evolution Comparison"])
    
    with tab1:
        st.markdown(f"##### Simulation Hierarchy - {selected_field}")
        sunburst_fig = st.session_state.visualizer.create_sunburst_chart(
            summaries, selected_field
        )
        if sunburst_fig.data:
            st.plotly_chart(sunburst_fig, use_container_width=True)
        else:
            st.info(f"No {selected_field} data available for sunburst chart.")
    
    with tab2:
        st.markdown("##### Radar Chart Comparison")
        radar_fig = st.session_state.visualizer.create_radar_chart(
            summaries, selected_sims
        )
        if radar_fig.data:
            st.plotly_chart(radar_fig, use_container_width=True)
        else:
            st.info("Insufficient data for radar chart comparison.")
    
    with tab3:
        st.markdown(f"##### {selected_field} Evolution Over Time")
        evolution_fig = st.session_state.visualizer.create_field_evolution_chart(
            summaries, selected_sims, selected_field
        )
        if evolution_fig.data:
            st.plotly_chart(evolution_fig, use_container_width=True)
        else:
            st.info(f"No {selected_field} evolution data available for selected simulations.")
    
    # Parameter space analysis
    st.markdown('<h3 class="sub-header">🌐 Parameter Space Analysis</h3>', 
               unsafe_allow_html=True)
    
    # Extract parameters for 3D visualization
    energies = []
    durations = []
    max_vals = []
    sim_names = []
    
    for summary in summaries:
        if summary['name'] in selected_sims and selected_field in summary['field_stats']:
            energies.append(summary['energy'])
            durations.append(summary['duration'])
            sim_names.append(summary['name'])
            
            stats = summary['field_stats'][selected_field]
            if stats['max']:
                max_vals.append(np.max(stats['max']))
            else:
                max_vals.append(0)
    
    if energies and durations and max_vals:
        fig_space = go.Figure(data=go.Scatter3d(
            x=energies,
            y=durations,
            z=max_vals,
            mode='markers+text',
            marker=dict(
                size=12,
                color=max_vals,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title=f"Max {selected_field}")
            ),
            text=sim_names,
            textposition="top center",
            hovertemplate='<b>%{text}</b><br>Energy: %{x:.2f} mJ<br>Duration: %{y:.2f} ns<br>Max Value: %{z:.2f}<extra></extra>'
        ))
        
        fig_space.update_layout(
            title=f"Parameter Space - Maximum {selected_field}",
            scene=dict(
                xaxis_title="Energy (mJ)",
                yaxis_title="Duration (ns)",
                zaxis_title=f"Max {selected_field}"
            ),
            height=600
        )
        
        st.plotly_chart(fig_space, use_container_width=True)
        
        # Add insights
        with st.expander("💡 Insights from Parameter Space"):
            if len(energies) > 1:
                # Calculate correlations
                energy_array = np.array(energies)
                duration_array = np.array(durations)
                max_array = np.array(max_vals)
                
                # Correlation between energy and max value
                if len(energy_array) > 1 and len(max_array) > 1:
                    energy_corr = np.corrcoef(energy_array, max_array)[0, 1]
                    st.write(f"**Energy vs Max {selected_field} correlation:** {energy_corr:.3f}")
                    
                    if energy_corr > 0.7:
                        st.info("Strong positive correlation: Higher energy leads to higher values")
                    elif energy_corr > 0.3:
                        st.info("Moderate positive correlation")
                    elif energy_corr < -0.7:
                        st.info("Strong negative correlation: Higher energy leads to lower values")
                    elif energy_corr < -0.3:
                        st.info("Moderate negative correlation")
                    else:
                        st.info("Weak correlation: Energy doesn't strongly affect this field")
                
                # Find extremal points
                max_idx = np.argmax(max_array)
                min_idx = np.argmin(max_array)
                
                st.write("**Extremal Points:**")
                st.write(f"• Highest {selected_field}: {sim_names[max_idx]} (Energy: {energies[max_idx]} mJ, Duration: {durations[max_idx]} ns)")
                st.write(f"• Lowest {selected_field}: {sim_names[min_idx]} (Energy: {energies[min_idx]} mJ, Duration: {durations[min_idx]} ns)")
    
    # Comparative statistics table
    st.markdown('<h3 class="sub-header">📋 Comparative Statistics</h3>', 
               unsafe_allow_html=True)
    
    stats_data = []
    for sim_name in selected_sims:
        summary = next((s for s in summaries if s['name'] == sim_name), None)
        if summary and selected_field in summary['field_stats']:
            stats = summary['field_stats'][selected_field]
            
            if stats['mean'] and stats['max']:
                row = {
                    'Simulation': sim_name,
                    'Energy (mJ)': summary['energy'],
                    'Duration (ns)': summary['duration'],
                    f'Mean {selected_field}': np.mean(stats['mean']),
                    f'Max {selected_field}': np.max(stats['max']),
                    f'Std Dev {selected_field}': np.mean(stats['std']) if stats['std'] else 0
                }
                stats_data.append(row)
    
    if stats_data:
        df_stats = pd.DataFrame(stats_data)
        
        # Add highlighting for max values
        def highlight_max(s):
            is_max = s == s.max()
            return ['background-color: #C8E6C9' if v else '' for v in is_max]
        
        numeric_cols = df_stats.select_dtypes(include=[np.number]).columns
        styled_df = df_stats.style.apply(highlight_max, subset=numeric_cols)
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Export option
        csv = df_stats.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Comparison as CSV",
            data=csv,
            file_name=f"comparison_{selected_field}.csv",
            mime="text/csv"
        )
    else:
        st.info(f"No {selected_field} statistics available for selected simulations.")

if __name__ == "__main__":
    main()
