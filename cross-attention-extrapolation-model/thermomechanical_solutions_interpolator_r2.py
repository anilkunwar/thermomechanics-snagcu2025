# advanced_fea_attention_platform.py
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
                    clean_data = data[~np.isnan(data)]
                    if clean_data.size > 0:
                        summary['field_stats'][field_name]['min'].append(np.min(clean_data))
                        summary['field_stats'][field_name]['max'].append(np.max(clean_data))
                        summary['field_stats'][field_name]['mean'].append(np.mean(clean_data))
                        summary['field_stats'][field_name]['std'].append(np.std(clean_data))
                        summary['field_stats'][field_name]['q25'].append(np.percentile(clean_data, 25))
                        summary['field_stats'][field_name]['q50'].append(np.percentile(clean_data, 50))
                        summary['field_stats'][field_name]['q75'].append(np.percentile(clean_data, 75))
                    else:
                        for key in ['min', 'max', 'mean', 'std', 'q25', 'q50', 'q75']:
                            summary['field_stats'][field_name][key].append(0.0)
                else:
                    # For vector fields, take magnitude
                    magnitude = np.linalg.norm(data, axis=1)
                    clean_mag = magnitude[~np.isnan(magnitude)]
                    if clean_mag.size > 0:
                        summary['field_stats'][field_name]['min'].append(np.min(clean_mag))
                        summary['field_stats'][field_name]['max'].append(np.max(clean_mag))
                        summary['field_stats'][field_name]['mean'].append(np.mean(clean_mag))
                        summary['field_stats'][field_name]['std'].append(np.std(clean_mag))
                        summary['field_stats'][field_name]['q25'].append(np.percentile(clean_mag, 25))
                        summary['field_stats'][field_name]['q50'].append(np.percentile(clean_mag, 50))
                        summary['field_stats'][field_name]['q75'].append(np.percentile(clean_mag, 75))
                    else:
                        for key in ['min', 'max', 'mean', 'std', 'q25', 'q50', 'q75']:
                            summary['field_stats'][field_name][key].append(0.0)
        
        return summary

# =============================================
# TRANSFORMER-INSPIRED ATTENTION MECHANISM WITH SPATIAL LOCALITY
# =============================================
class PhysicsInformedAttentionExtrapolator:
    """Enhanced extrapolator with spatial locality regulation and hybrid attention"""
    
    def __init__(self, sigma_param=0.3, spatial_weight=0.5, n_heads=4):
        self.sigma_param = sigma_param
        self.spatial_weight = spatial_weight
        self.n_heads = n_heads
        self.source_db = []
        self.scaler = StandardScaler()
        self.source_embeddings = None
        self.source_values = None
        self.source_metadata = None  # For spatial indexing
        
    def load_summaries(self, summaries):
        """Load summary statistics from data loader"""
        self.source_db = summaries
        
        # Prepare data for scaling and attention
        all_embeddings = []
        all_values = []
        all_metadata = []
        
        for summary_idx, summary in enumerate(summaries):
            for timestep_idx, t in enumerate(summary['timesteps']):
                emb = self._compute_embedding(summary['energy'], summary['duration'], t)
                all_embeddings.append(emb)
                
                # Collect field statistics values (flattened)
                field_vals = []
                for field in sorted(summary['field_stats'].keys()):  # Ensure consistent order
                    stats = summary['field_stats'][field]
                    # Use mean, max, std as representative stats
                    field_vals.extend([
                        stats['mean'][timestep_idx],
                        stats['max'][timestep_idx],
                        stats['std'][timestep_idx]
                    ])
                all_values.append(field_vals)
                
                # Store metadata for spatial correlation
                all_metadata.append({
                    'summary_idx': summary_idx,
                    'timestep_idx': timestep_idx,
                    'energy': summary['energy'],
                    'duration': summary['duration'],
                    'time': t
                })
        
        if all_embeddings:
            all_embeddings = np.array(all_embeddings)
            self.scaler.fit(all_embeddings)
            self.source_embeddings = self.scaler.transform(all_embeddings)
            self.source_values = np.array(all_values)
            self.source_metadata = all_metadata
    
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
        
        # Additional physics features
        power = energy / (duration + 1e-6)
        cooling_rate = 1.0 / (time + 1e-6)
        
        return np.array([
            logE,
            duration,
            time,
            energy_density,
            time_ratio,
            thermal_diffusion,
            strain_rate,
            power,
            cooling_rate
        ], dtype=np.float32)
    
    def _compute_spatial_correlation(self, query_meta, source_metadata):
        """Compute pseudo-spatial correlation based on parameter proximity"""
        # In true spatial attention, this would use actual (x,y,z) coordinates
        # Here we use parameter-space proximity as a proxy for "spatial" locality
        correlations = []
        for meta in source_metadata:
            # Compute parameter distance
            e_dist = abs(query_meta['energy'] - meta['energy'])
            d_dist = abs(query_meta['duration'] - meta['duration'])
            t_dist = abs(query_meta['time'] - meta['time'])
            
            # Normalize distances (rough scaling)
            e_norm = e_dist / 20.0  # max energy ~20mJ
            d_norm = d_dist / 10.0  # max duration ~10ns
            t_norm = t_dist / 50.0  # max time ~50ns
            
            total_dist = e_norm + d_norm + t_norm
            # Higher correlation for closer parameters
            correlation = np.exp(-total_dist)
            correlations.append(correlation)
        
        return np.array(correlations)
    
    def _multi_head_attention(self, query_embedding, query_meta):
        """Multi-head attention mechanism inspired by transformers with spatial regulation"""
        if self.source_embeddings is None or len(self.source_embeddings) == 0:
            n_fields = len(next(iter(self.source_db[0]['field_stats'].keys()), []))
            n_stats = 3  # mean, max, std
            n_total = n_fields * n_stats if n_fields > 0 else 3
            return np.zeros(n_total), np.zeros(0)
        
        # Normalize query embedding
        query_emb_norm = self.scaler.transform([query_embedding])[0]
        
        n_sources = len(self.source_embeddings)
        head_weights = np.zeros((self.n_heads, n_sources))
        
        # Multi-head attention with different projections
        for head in range(self.n_heads):
            # Use deterministic random projection per head
            np.random.seed(42 + head)
            proj_dim = min(5, query_emb_norm.shape[0])
            proj_matrix = np.random.randn(query_emb_norm.shape[0], proj_dim)
            
            query_proj = query_emb_norm @ proj_matrix
            source_proj = self.source_embeddings @ proj_matrix
            
            # Compute attention scores (Gaussian kernel)
            diff = query_proj[np.newaxis, :] - source_proj
            dist_sq = np.sum(diff ** 2, axis=1)
            scores = np.exp(-dist_sq / (2 * self.sigma_param ** 2))
            
            # Apply spatial locality regulation
            if self.spatial_weight > 0:
                spatial_corr = self._compute_spatial_correlation(query_meta, self.source_metadata)
                # Hybrid attention: combine parameter similarity and spatial correlation
                scores = (1 - self.spatial_weight) * scores + self.spatial_weight * spatial_corr
            
            head_weights[head] = scores
        
        # Average across heads
        avg_weights = np.mean(head_weights, axis=0)
        
        # Apply softmax for final attention weights
        # For numerical stability
        max_weight = np.max(avg_weights)
        exp_weights = np.exp(avg_weights - max_weight)
        attention_weights = exp_weights / (np.sum(exp_weights) + 1e-12)
        
        # Weighted prediction
        prediction = np.sum(attention_weights[:, np.newaxis] * self.source_values, axis=0)
        
        return prediction, attention_weights
    
    def predict_field_statistics(self, energy_query, duration_query, time_query):
        """Predict field statistics for given parameters"""
        if not self.source_db:
            return None
        
        query_embedding = self._compute_embedding(energy_query, duration_query, time_query)
        query_meta = {
            'energy': energy_query,
            'duration': duration_query,
            'time': time_query
        }
        
        prediction, attention_weights = self._multi_head_attention(query_embedding, query_meta)
        
        # Reconstruct field statistics from prediction
        result = {
            'prediction': prediction,
            'attention_weights': attention_weights,
            'confidence': float(np.max(attention_weights)) if len(attention_weights) > 0 else 0.0,
            'field_predictions': {}
        }
        
        # Map predictions back to field statistics
        field_names = sorted(next(iter(self.source_db[0]['field_stats'].keys()), []))
        if not field_names:
            # Fallback field names if none detected
            field_names = ['temperature', 'displacement', 'principal stress']
        
        for i, field in enumerate(field_names):
            start_idx = i * 3
            if start_idx + 2 < len(prediction):
                result['field_predictions'][field] = {
                    'mean': float(prediction[start_idx]),
                    'max': float(prediction[start_idx + 1]),
                    'std': float(prediction[start_idx + 2])
                }
        
        return result
    
    def predict_time_series(self, energy_query, duration_query, time_points):
        """Predict over a series of time points"""
        results = {
            'time_points': time_points,
            'field_predictions': {
                'temperature': {'mean': [], 'max': [], 'std': []},
                'displacement': {'mean': [], 'max': [], 'std': []},
                'principal stress': {'mean': [], 'max': [], 'std': []}
            },
            'attention_maps': [],
            'confidence_scores': []
        }
        
        for t in time_points:
            pred = self.predict_field_statistics(energy_query, duration_query, t)
            if pred:
                for field in results['field_predictions']:
                    if field in pred['field_predictions']:
                        stats = pred['field_predictions'][field]
                        results['field_predictions'][field]['mean'].append(stats['mean'])
                        results['field_predictions'][field]['max'].append(stats['max'])
                        results['field_predictions'][field]['std'].append(stats['std'])
                    else:
                        # Append NaN if field not predicted
                        results['field_predictions'][field]['mean'].append(np.nan)
                        results['field_predictions'][field]['max'].append(np.nan)
                        results['field_predictions'][field]['std'].append(np.nan)
                
                results['attention_maps'].append(pred['attention_weights'])
                results['confidence_scores'].append(pred['confidence'])
            else:
                # Handle prediction failure
                for field in results['field_predictions']:
                    results['field_predictions'][field]['mean'].append(np.nan)
                    results['field_predictions'][field]['max'].append(np.nan)
                    results['field_predictions'][field]['std'].append(np.nan)
                results['attention_maps'].append(np.zeros(0))
                results['confidence_scores'].append(0.0)
        
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
            energy_label = f"E:{summary['energy']}mJ"
            labels.append(energy_label)
            parents.append("All Simulations")
            values.append(1)
            
            # Duration level
            duration_label = f"τ:{summary['duration']}ns"
            labels.append(duration_label)
            parents.append(energy_label)
            values.append(1)
            
            # Field statistics
            if selected_field in summary['field_stats']:
                avg_max = np.mean(summary['field_stats'][selected_field]['max'])
                field_label = f"{selected_field}"
                labels.append(field_label)
                parents.append(duration_label)
                values.append(avg_max if avg_max > 0 else 1e-6)  # Avoid zero
        
        # Handle empty or invalid values
        values = [max(v, 1e-6) for v in values]
        
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
        stats = ['mean', 'max']
        
        fig = go.Figure()
        
        for sim_name in simulation_names:
            # Find summary
            summary = None
            for s in summaries:
                if f"q{s['energy']}mJ-delta{s['duration']}ns" == sim_name:
                    summary = s
                    break
            
            if not summary:
                continue
            
            r_values = []
            theta_values = []
            
            for field in fields:
                if field in summary['field_stats']:
                    for stat in stats:
                        avg_value = np.mean(summary['field_stats'][field][stat])
                        r_values.append(avg_value if avg_value > 0 else 1e-6)
                        theta_values.append(f"{field}<br>{stat}")
                else:
                    # Default values if field not present
                    for stat in stats:
                        r_values.append(1e-6)
                        theta_values.append(f"{field}<br>{stat}")
            
            fig.add_trace(go.Scatterpolar(
                r=r_values,
                theta=theta_values,
                fill='toself',
                name=sim_name
            ))
        
        if fig.data:
            max_r = max([max(trace.r) for trace in fig.data if len(trace.r) > 0])
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max_r * 1.1] if max_r > 0 else [0, 1]
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
        if len(attention_weights) == 0:
            # Return empty figure
            fig = go.Figure()
            fig.update_layout(
                title="No attention weights available",
                height=200
            )
            return fig
        
        # Create labels for x-axis
        x_labels = []
        for i, sim in enumerate(source_simulations):
            x_labels.append(f"Sim{i+1}: E{sim['energy']}mJ τ{sim['duration']}ns")
        
        # Reshape attention weights for heatmap (single row)
        z_data = [attention_weights]
        
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=x_labels,
            y=['Attention Weight'],
            colorscale='Viridis',
            colorbar=dict(title="Weight"),
            hovertemplate='Simulation: %{x}<br>Attention: %{z:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Attention Weights Distribution",
            xaxis_title="Source Simulations",
            yaxis_title="",
            height=250,
            xaxis=dict(tickangle=45)
        )
        
        return fig
    
    @staticmethod
    def create_attention_network(attention_weights, source_simulations, top_k=10):
        """Create network graph showing top attention connections"""
        if len(attention_weights) == 0:
            fig = go.Figure()
            fig.update_layout(title="No attention data available", height=400)
            return fig
        
        # Get top-k most attended sources
        top_indices = np.argsort(attention_weights)[-top_k:][::-1]
        top_weights = attention_weights[top_indices]
        top_sims = [source_simulations[i] for i in top_indices]
        
        # Create network
        G = nx.Graph()
        
        # Add query node
        G.add_node("Query", size=50, color='red')
        
        # Add top source nodes
        for i, (sim, weight) in enumerate(zip(top_sims, top_weights)):
            node_id = f"Sim{i}"
            G.add_node(node_id, 
                      size=30 * weight / max(top_weights),
                      color='blue',
                      energy=sim['energy'],
                      duration=sim['duration'],
                      weight=weight)
            G.add_edge("Query", node_id, weight=weight)
        
        # Create positions
        pos = nx.spring_layout(G, seed=42)
        
        # Extract edge traces
        edge_x = []
        edge_y = []
        edge_weights = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(G.edges[edge]['weight'])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='gray'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Extract node traces
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            if node == "Query":
                node_text.append("Query Node")
                node_size.append(50)
                node_color.append('red')
            else:
                idx = int(node.replace("Sim", ""))
                sim = top_sims[idx]
                weight = top_weights[idx]
                node_text.append(f"E:{sim['energy']}mJ<br>τ:{sim['duration']}ns<br>Weight:{weight:.3f}")
                node_size.append(30 * weight / max(top_weights))
                node_color.append('blue')
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2)
            ),
            text=node_text,
            textposition="middle center"
        )
        
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title=f"Top {top_k} Attention Connections",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
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
    
    # Custom CSS for better appearance
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
        border-bottom: 2px solid #E0E0E5;
        padding-bottom: 0.5rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #2196F3;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px;
        color: #424242;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5;
        color: white;
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
            if st.session_state.simulations:
                first_sim = next(iter(st.session_state.simulations.values()))
                fields = list(first_sim['field_info'].keys())
                st.info(f"**Fields:** {', '.join(fields)}")
    
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
        if simulations:
            sim_name = st.selectbox(
                "Select Simulation",
                sorted(simulations.keys()),
                key="viewer_sim_select"
            )
        else:
            st.error("No simulations loaded")
            return
    
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
        
        # Handle NaN values
        if kind == "scalar":
            values = np.where(np.isnan(raw), 0, raw)
            label = field
        else:
            magnitude = np.linalg.norm(raw, axis=1)
            values = np.where(np.isnan(magnitude), 0, magnitude)
            label = f"{field} (magnitude)"
        
        # Create 3D plot
        if sim.get('triangles') is not None and len(sim['triangles']) > 0:
            tri = sim['triangles']
            mesh_data = go.Mesh3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                i=tri[:, 0], j=tri[:, 1], k=tri[:, 2],
                intensity=values,
                colorscale="Viridis",
                intensitymode='vertex',
                colorbar=dict(title=label),
                opacity=0.85,
                hovertemplate='Value: %{intensity:.2f}<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
            )
        else:
            # Fallback to point cloud
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
                hovertemplate='Value: %{marker.color:.2f}<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
            )
        
        fig = go.Figure(data=mesh_data)
        fig.update_layout(
            title=f"{label} at Timestep {timestep + 1} - {sim_name}",
            scene=dict(
                aspectmode="data",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=600,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Field statistics over time
    st.markdown('<h3 class="sub-header">📈 Field Evolution Over Time</h3>', 
                unsafe_allow_html=True)
    
    # Find corresponding summary
    summary = None
    for s in st.session_state.summaries:
        if f"q{s['energy']}mJ-delta{s['duration']}ns" == sim_name:
            summary = s
            break
    
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
        fig_time.add_trace(go.Scatter(
            x=summary['timesteps'],
            y=stats['std'],
            mode='lines+markers',
            name='Std Dev',
            line=dict(color='green', width=2)
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
    st.markdown('<h2 class="sub-header">🔮 Physics-Informed Attention Extrapolation</h2>', 
                unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load simulations first using the sidebar button.")
        return
    
    st.markdown("""
    <div class="info-box">
    <strong>Transformer-Inspired Attention with Spatial Locality:</strong> This engine implements a 
    multi-head attention mechanism that learns from existing FEA simulations. The spatial locality 
    regulator ensures that predictions are influenced more by simulations with similar physical 
    parameters, providing smooth interpolation and graceful extrapolation with quantified uncertainty.
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
    with st.expander("⚙️ Attention Mechanism Parameters"):
        col1, col2, col3 = st.columns(3)
        with col1:
            sigma_param = st.slider(
                "Sigma (Kernel Width)",
                min_value=0.1,
                max_value=1.0,
                value=0.3,
                step=0.05,
                key="interp_sigma",
                help="Controls how sharply attention focuses on similar parameters"
            )
        with col2:
            spatial_weight = st.slider(
                "Spatial Locality Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                key="interp_spatial",
                help="Balance between parameter similarity (0) and spatial correlation (1)"
            )
        with col3:
            n_heads = st.slider(
                "Number of Attention Heads",
                min_value=1,
                max_value=8,
                value=4,
                step=1,
                key="interp_heads",
                help="Ensemble of attention mechanisms for robustness"
            )
        
        # Update extrapolator parameters
        st.session_state.extrapolator.sigma_param = sigma_param
        st.session_state.extrapolator.spatial_weight = spatial_weight
        st.session_state.extrapolator.n_heads = n_heads
    
    if st.button("🚀 Run Physics-Informed Prediction", type="primary"):
        with st.spinner("Running multi-head attention prediction with spatial locality..."):
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
                                   'Mean Displacement', 'Prediction Confidence'),
                    vertical_spacing=0.15,
                    horizontal_spacing=0.1
                )
                
                # Temperature
                temp_max = results['field_predictions']['temperature']['max']
                if any(not np.isnan(x) for x in temp_max):
                    fig_pred.add_trace(
                        go.Scatter(
                            x=time_points,
                            y=temp_max,
                            mode='lines+markers',
                            name='Max Temp',
                            line=dict(color='red', width=3)
                        ),
                        row=1, col=1
                    )
                
                # Stress
                stress_max = results['field_predictions']['principal stress']['max']
                if any(not np.isnan(x) for x in stress_max):
                    fig_pred.add_trace(
                        go.Scatter(
                            x=time_points,
                            y=stress_max,
                            mode='lines+markers',
                            name='Max Stress',
                            line=dict(color='blue', width=3)
                        ),
                        row=1, col=2
                    )
                
                # Displacement
                disp_mean = results['field_predictions']['displacement']['mean']
                if any(not np.isnan(x) for x in disp_mean):
                    fig_pred.add_trace(
                        go.Scatter(
                            x=time_points,
                            y=disp_mean,
                            mode='lines+markers',
                            name='Mean Disp',
                            line=dict(color='green', width=3)
                        ),
                        row=2, col=1
                    )
                
                # Confidence
                conf_scores = results['confidence_scores']
                fig_pred.add_trace(
                    go.Scatter(
                        x=time_points,
                        y=conf_scores,
                        mode='lines+markers',
                        name='Confidence',
                        line=dict(color='orange', width=3)
                    ),
                    row=2, col=2
                )
                
                fig_pred.update_layout(height=600, showlegend=True, title_text="Field Predictions Over Time")
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Attention visualization
                if results['attention_maps'] and len(results['attention_maps'][0]) > 0:
                    st.markdown('<h4 class="sub-header">🧠 Attention Mechanism Visualization</h4>', 
                               unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Attention heatmap for first timestep
                        heatmap_fig = st.session_state.visualizer.create_attention_heatmap(
                            results['attention_maps'][0],
                            st.session_state.summaries
                        )
                        st.plotly_chart(heatmap_fig, use_container_width=True)
                    
                    with col2:
                        # Attention network graph
                        network_fig = st.session_state.visualizer.create_attention_network(
                            results['attention_maps'][0],
                            st.session_state.summaries,
                            top_k=8
                        )
                        st.plotly_chart(network_fig, use_container_width=True)
                
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
                            row[f'{field}_std'] = results['field_predictions'][field]['std'][idx]
                    row['confidence'] = results['confidence_scores'][idx]
                    data_rows.append(row)
                
                df_results = pd.DataFrame(data_rows)
                st.dataframe(df_results.style.format("{:.3f}"), use_container_width=True)
                
                # Export option
                csv = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Predictions (CSV)",
                    data=csv,
                    file_name=f"fealaser_predictions_E{energy_query:.1f}mJ_tau{duration_query:.1f}ns.csv",
                    mime="text/csv"
                )
                
                # Confidence warning
                avg_conf = np.mean([c for c in conf_scores if not np.isnan(c)])
                if avg_conf < 0.3:
                    st.warning(f"⚠️ **Low confidence** (avg={avg_conf:.2f}): Query parameters are far from training data. Extrapolation risk is high.")
            else:
                st.error("Prediction failed. Check input parameters and ensure simulations are loaded.")

def render_comparative_analysis():
    """Render comparative analysis interface"""
    st.markdown('<h2 class="sub-header">📊 Advanced Comparative Analysis</h2>', 
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
        default=list(simulations.keys())[:min(3, len(simulations))] if simulations else []
    )
    
    if not selected_sims:
        st.info("Please select at least one simulation for comparison.")
        return
    
    # Field selection
    available_fields = set()
    for sim_name in selected_sims:
        if sim_name in simulations:
            available_fields.update(simulations[sim_name]['field_info'].keys())
    
    if available_fields:
        selected_field = st.selectbox(
            "Select field for analysis",
            list(available_fields)
        )
    else:
        selected_field = 'temperature'
        st.warning("No fields detected, using default field.")
    
    # Create comparison plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h4>📈 Hierarchical Sunburst Chart</h4>', unsafe_allow_html=True)
        try:
            sunburst_fig = st.session_state.visualizer.create_sunburst_chart(
                summaries, selected_field
            )
            st.plotly_chart(sunburst_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Sunburst chart error: {e}")
    
    with col2:
        st.markdown('<h4>🎯 Multi-Field Radar Comparison</h4>', unsafe_allow_html=True)
        try:
            radar_fig = st.session_state.visualizer.create_radar_chart(
                summaries, selected_sims[:5]  # Limit to 5 for clarity
            )
            st.plotly_chart(radar_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Radar chart error: {e}")
    
    # Detailed field comparison over time
    st.markdown('<h4>⏱️ Field Evolution Comparison</h4>', unsafe_allow_html=True)
    
    fig_comparison = go.Figure()
    
    for sim_name in selected_sims:
        # Find summary
        summary = None
        for s in summaries:
            if f"q{s['energy']}mJ-delta{s['duration']}ns" == sim_name:
                summary = s
                break
        
        if summary and selected_field in summary['field_stats']:
            stats = summary['field_stats'][selected_field]
            
            fig_comparison.add_trace(go.Scatter(
                x=summary['timesteps'],
                y=stats['mean'],
                mode='lines+markers',
                name=f"{sim_name} (mean)",
                line=dict(width=2)
            ))
            fig_comparison.add_trace(go.Scatter(
                x=summary['timesteps'],
                y=stats['max'],
                mode='lines+markers',
                name=f"{sim_name} (max)",
                line=dict(width=2, dash='dash')
            ))
    
    fig_comparison.update_layout(
        title=f"{selected_field} Comparison Across Simulations",
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
    max_stresses = []
    
    for summary in summaries:
        energies.append(summary['energy'])
        durations.append(summary['duration'])
        
        if 'temperature' in summary['field_stats']:
            max_temps.append(np.max(summary['field_stats']['temperature']['max']))
        else:
            max_temps.append(0)
        
        if 'principal stress' in summary['field_stats']:
            max_stresses.append(np.max(summary['field_stats']['principal stress']['max']))
        else:
            max_stresses.append(0)
    
    # Create parameter space plots
    col1, col2 = st.columns(2)
    
    with col1:
        fig_temp = go.Figure(data=go.Scatter3d(
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
        
        fig_temp.update_layout(
            title="Parameter Space - Maximum Temperature",
            scene=dict(
                xaxis_title="Energy (mJ)",
                yaxis_title="Duration (ns)",
                zaxis_title="Max Temperature"
            ),
            height=500
        )
        
        st.plotly_chart(fig_temp, use_container_width=True)
    
    with col2:
        fig_stress = go.Figure(data=go.Scatter3d(
            x=energies,
            y=durations,
            z=max_stresses,
            mode='markers',
            marker=dict(
                size=10,
                color=max_stresses,
                colorscale='Plasma',
                opacity=0.8,
                colorbar=dict(title="Max Stress")
            ),
            text=[f"E:{e}mJ, τ:{d}ns" for e, d in zip(energies, durations)],
            hovertemplate='%{text}<br>Max Stress: %{z:.1f}<extra></extra>'
        ))
        
        fig_stress.update_layout(
            title="Parameter Space - Maximum Stress",
            scene=dict(
                xaxis_title="Energy (mJ)",
                yaxis_title="Duration (ns)",
                zaxis_title="Max Stress (GPa)"
            ),
            height=500
        )
        
        st.plotly_chart(fig_stress, use_container_width=True)

if __name__ == "__main__":
    main()
