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
from streamlit.runtime.scripter_runtime.script_runner import get_script_run_ctx
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
    """Enhanced data loader with robust error handling and progress tracking"""
   
    def __init__(self):
        self.simulations = {}
        self.summaries = []
        self.field_statistics = {}
       
    def parse_folder_name(self, folder: str):
        """Parse folder name to extract energy and duration"""
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
       
        progress_bar = st.progress(0)
        total_folders = len(folders)
       
        for folder_idx, folder in enumerate(folders):
            name = os.path.basename(folder)
            energy, duration = _self.parse_folder_name(name)
            if energy is None:
                st.warning(f"Invalid folder name: {name}")
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
                   
                    # Load all timesteps with progress
                    timestep_progress = st.progress(0)
                    for t in range(1, len(vtu_files)):
                        try:
                            mesh = meshio.read(vtu_files[t])
                            for key in sim_data['field_info'].keys():
                                if key in mesh.point_data:
                                    fields[key][t] = mesh.point_data[key].astype(np.float32)
                        except Exception as e:
                            st.warning(f"Error loading timestep {t} in {name}: {e}")
                        timestep_progress.progress((t + 1) / len(vtu_files))
                   
                    sim_data.update({
                        'points': points,
                        'fields': fields,
                        'triangles': triangles
                    })
               
                # Create summary statistics
                summary = _self.extract_summary_statistics(vtu_files, energy, duration, name)
                summaries.append(summary)
               
                simulations[name] = sim_data
               
            except Exception as e:
                st.warning(f"Error loading {name}: {str(e)}")
                continue
           
            progress_bar.progress((folder_idx + 1) / total_folders)
       
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
                timestep = idx + 1
                summary['timesteps'].append(timestep)
               
                for field_name in mesh.point_data.keys():
                    data = mesh.point_data[field_name]
                   
                    if field_name not in summary['field_stats']:
                        summary['field_stats'][field_name] = {
                            'min': [], 'max': [], 'mean': [], 'std': [],
                            'q25': [], 'q50': [], 'q75': []
                        }
                   
                    if data.ndim == 1:
                        values = data
                    else:
                        values = np.linalg.norm(data, axis=1)
                   
                    clean_values = values[~np.isnan(values)]
                    if len(clean_values) > 0:
                        stats = summary['field_stats'][field_name]
                        stats['min'].append(np.min(clean_values))
                        stats['max'].append(np.max(clean_values))
                        stats['mean'].append(np.mean(clean_values))
                        stats['std'].append(np.std(clean_values))
                        stats['q25'].append(np.percentile(clean_values, 25))
                        stats['q50'].append(np.percentile(clean_values, 50))
                        stats['q75'].append(np.percentile(clean_values, 75))
                    else:
                        stats = summary['field_stats'][field_name]
                        for key in stats:
                            stats[key].append(np.nan)
            except Exception as e:
                st.warning(f"Error processing {vtu_file}: {e}")
                for field_name in summary['field_stats']:
                    for key in summary['field_stats'][field_name]:
                        summary['field_stats'][field_name][key].append(np.nan)
       
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
        self.field_names = []
       
    def load_summaries(self, summaries):
        """Load summary statistics from data loader"""
        self.source_db = summaries
       
        # Extract all field names
        self.field_names = list(set.union(*(set(s['field_stats'].keys()) for s in summaries)))
        self.field_names.sort()
       
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
            np.random.seed(head) # For reproducibility
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
                for field in self.field_names:
                    stats = summary['field_stats'].get(field, {})
                    field_vals.extend([
                        stats.get('mean', [0])[idx],
                        stats.get('max', [0])[idx],
                        stats.get('std', [0])[idx],
                        stats.get('q25', [0])[idx],
                        stats.get('q50', [0])[idx],
                        stats.get('q75', [0])[idx]
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
        stats_per_field = 6  # mean, max, std, q25, q50, q75
        field_idx = 0
        for field in self.field_names:
            result['field_predictions'][field] = {
                'mean': prediction[field_idx],
                'max': prediction[field_idx + 1],
                'std': prediction[field_idx + 2],
                'q25': prediction[field_idx + 3],
                'q50': prediction[field_idx + 4],
                'q75': prediction[field_idx + 5]
            }
            field_idx += 6
       
        return result
   
    def predict_time_series(self, energy_query, duration_query, time_points):
        """Predict over a series of time points"""
        results = {
            'time_points': time_points,
            'field_predictions': {},
            'attention_maps': [],
            'confidences': []
        }
       
        for field in self.field_names:
            results['field_predictions'][field] = {
                'mean': [], 'max': [], 'std': [], 'q25': [], 'q50': [], 'q75': []
            }
       
        for t in time_points:
            pred = self.predict_field_statistics(energy_query, duration_query, t)
            if pred:
                for field in self.field_names:
                    field_pred = pred['field_predictions'].get(field, {})
                    for stat in ['mean', 'max', 'std', 'q25', 'q50', 'q75']:
                        results['field_predictions'][field][stat].append(
                            field_pred.get(stat, np.nan)
                        )
                results['attention_maps'].append(pred['attention_weights'])
                results['confidences'].append(pred['confidence'])
       
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
                avg_max = np.nanmean(summary['field_stats'][selected_field]['max'])
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
                        avg_value = np.nanmean(summary['field_stats'][field][stat])
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
   
    @staticmethod
    def create_attention_graph(attention_weights, source_simulations, threshold=0.1):
        """Create graph visualization of attention connections"""
        G = nx.Graph()
       
        # Add nodes
        G.add_node("Query", type='query')
        for idx, sim in enumerate(source_simulations):
            G.add_node(f"Sim{idx+1}", type='source', energy=sim['energy'], duration=sim['duration'])
           
            if attention_weights[idx] > threshold:
                G.add_edge("Query", f"Sim{idx+1}", weight=attention_weights[idx])
       
        # Layout
        pos = nx.spring_layout(G)
       
        # Create Plotly figure
        edge_traces = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_traces.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=edge[2]['weight']*10, color='gray'),
                hoverinfo='none',
                mode='lines'
            ))
       
        node_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            mode='markers+text',
            marker=dict(
                size=[20 if n=='Query' else 15 for n in G.nodes()],
                color=['red' if n=='Query' else 'blue' for n in G.nodes()]
            ),
            text=[n for n in G.nodes()],
            hoverinfo='text'
        )
       
        fig = go.Figure(data=edge_traces + [node_trace])
        fig.update_layout(
            title="Attention Connection Graph",
            showlegend=False,
            height=400
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
            fields = list(set.union(*(set(s['field_info'].keys()) for s in st.session_state.simulations.values())))
            st.info(f"**Fields:** {', '.join(fields)}")
   
    # Main content based on selected mode
    if app_mode == "Data Viewer":
        render_data_viewer()
    elif app_mode == "Interpolation/Extrapolation":
        render_interpolation_extrapolation()
    elif app_mode == "Comparative Analysis":
        render_comparative_analysis()
if __name__ == "__main__":
    main()
