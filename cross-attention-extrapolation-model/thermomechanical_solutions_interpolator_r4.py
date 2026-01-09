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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from scipy.spatial.distance import cdist

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
    """Enhanced data loader with comprehensive field extraction"""
    
    def __init__(self):
        self.simulations = {}
        self.summaries = []
        self.field_statistics = {}
        self.available_fields = set()
        
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
                
                # Create simulation entry
                sim_data = {
                    'name': name,
                    'energy_mJ': energy,
                    'duration_ns': duration,
                    'n_timesteps': len(vtu_files),
                    'vtu_files': vtu_files,
                    'field_info': {},
                    'has_mesh': False
                }
                
                if load_full_mesh:
                    # Full mesh loading
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
                        _self.available_fields.add(key)
                    
                    # Load remaining timesteps
                    for t in range(1, len(vtu_files)):
                        try:
                            mesh = meshio.read(vtu_files[t])
                            for key in sim_data['field_info']:
                                if key in mesh.point_data:
                                    fields[key][t] = mesh.point_data[key].astype(np.float32)
                        except Exception as e:
                            st.warning(f"Error loading timestep {t} in {name}: {e}")
                    
                    sim_data.update({
                        'points': points,
                        'fields': fields,
                        'triangles': triangles,
                        'has_mesh': True
                    })
                
                # Create summary statistics
                summary = _self.extract_summary_statistics(vtu_files, energy, duration, name)
                summaries.append(summary)
                
                simulations[name] = sim_data
                
            except Exception as e:
                st.warning(f"Error loading {name}: {str(e)}")
                continue
            
            progress_bar.progress((folder_idx + 1) / len(folders))
        
        progress_bar.empty()
        status_text.empty()
        
        if simulations:
            st.success(f"✅ Loaded {len(simulations)} simulations with {len(_self.available_fields)} unique fields")
        else:
            st.error("❌ No simulations loaded successfully")
        
        return simulations, summaries
    
    def extract_summary_statistics(self, vtu_files, energy, duration, name):
        """Extract comprehensive summary statistics from VTU files"""
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
                            'q25': [], 'q50': [], 'q75': [], 'percentiles': []
                        }
                    
                    if data.ndim == 1:
                        clean_data = data[~np.isnan(data)]
                        if clean_data.size > 0:
                            summary['field_stats'][field_name]['min'].append(float(np.min(clean_data)))
                            summary['field_stats'][field_name]['max'].append(float(np.max(clean_data)))
                            summary['field_stats'][field_name]['mean'].append(float(np.mean(clean_data)))
                            summary['field_stats'][field_name]['std'].append(float(np.std(clean_data)))
                            summary['field_stats'][field_name]['q25'].append(float(np.percentile(clean_data, 25)))
                            summary['field_stats'][field_name]['q50'].append(float(np.percentile(clean_data, 50)))
                            summary['field_stats'][field_name]['q75'].append(float(np.percentile(clean_data, 75)))
                            # Store full percentiles for detailed analysis
                            percentiles = np.percentile(clean_data, [10, 25, 50, 75, 90])
                            summary['field_stats'][field_name]['percentiles'].append(percentiles)
                        else:
                            for key in ['min', 'max', 'mean', 'std', 'q25', 'q50', 'q75']:
                                summary['field_stats'][field_name][key].append(0.0)
                            summary['field_stats'][field_name]['percentiles'].append(np.zeros(5))
                    else:
                        # Vector field - compute magnitude statistics
                        magnitude = np.linalg.norm(data, axis=1)
                        clean_mag = magnitude[~np.isnan(magnitude)]
                        if clean_mag.size > 0:
                            summary['field_stats'][field_name]['min'].append(float(np.min(clean_mag)))
                            summary['field_stats'][field_name]['max'].append(float(np.max(clean_mag)))
                            summary['field_stats'][field_name]['mean'].append(float(np.mean(clean_mag)))
                            summary['field_stats'][field_name]['std'].append(float(np.std(clean_mag)))
                            summary['field_stats'][field_name]['q25'].append(float(np.percentile(clean_mag, 25)))
                            summary['field_stats'][field_name]['q50'].append(float(np.percentile(clean_mag, 50)))
                            summary['field_stats'][field_name]['q75'].append(float(np.percentile(clean_mag, 75)))
                            percentiles = np.percentile(clean_mag, [10, 25, 50, 75, 90])
                            summary['field_stats'][field_name]['percentiles'].append(percentiles)
                        else:
                            for key in ['min', 'max', 'mean', 'std', 'q25', 'q50', 'q75']:
                                summary['field_stats'][field_name][key].append(0.0)
                            summary['field_stats'][field_name]['percentiles'].append(np.zeros(5))
            except Exception as e:
                st.warning(f"Error processing {vtu_file}: {e}")
                continue
        
        return summary

# =============================================
# ENHANCED ATTENTION MECHANISM WITH PHYSICS-AWARE EMBEDDINGS
# =============================================
class EnhancedPhysicsInformedAttentionExtrapolator:
    """Advanced extrapolator with physics-aware embeddings and multi-head attention"""
    
    def __init__(self, sigma_param=0.3, spatial_weight=0.5, n_heads=4, temperature=1.0):
        self.sigma_param = sigma_param
        self.spatial_weight = spatial_weight
        self.n_heads = n_heads
        self.temperature = temperature
        self.source_db = []
        self.embedding_scaler = StandardScaler()
        self.value_scaler = StandardScaler()
        self.source_embeddings = []
        self.source_values = []
        self.source_metadata = []
        self.fitted = False
        
    def load_summaries(self, summaries):
        """Load summary statistics and prepare for attention mechanism"""
        self.source_db = summaries
        
        if not summaries:
            return
        
        # Prepare embeddings and values
        all_embeddings = []
        all_values = []
        metadata = []
        
        for summary_idx, summary in enumerate(summaries):
            for timestep_idx, t in enumerate(summary['timesteps']):
                # Compute physics-aware embedding
                emb = self._compute_physics_embedding(
                    summary['energy'], 
                    summary['duration'], 
                    t
                )
                all_embeddings.append(emb)
                
                # Extract field values (flattened)
                field_vals = []
                for field in sorted(summary['field_stats'].keys()):
                    stats = summary['field_stats'][field]
                    # Use mean, max, std as representative features
                    if timestep_idx < len(stats['mean']):
                        field_vals.extend([
                            stats['mean'][timestep_idx],
                            stats['max'][timestep_idx],
                            stats['std'][timestep_idx]
                        ])
                    else:
                        field_vals.extend([0.0, 0.0, 0.0])
                
                all_values.append(field_vals)
                
                # Store metadata for spatial correlations
                metadata.append({
                    'summary_idx': summary_idx,
                    'timestep_idx': timestep_idx,
                    'energy': summary['energy'],
                    'duration': summary['duration'],
                    'time': t,
                    'name': summary['name']
                })
        
        if all_embeddings and all_values:
            all_embeddings = np.array(all_embeddings)
            all_values = np.array(all_values)
            
            # Scale embeddings
            self.embedding_scaler.fit(all_embeddings)
            self.source_embeddings = self.embedding_scaler.transform(all_embeddings)
            
            # Scale values (for stable attention)
            self.value_scaler.fit(all_values)
            self.source_values = all_values
            
            self.source_metadata = metadata
            self.fitted = True
            
            st.info(f"✅ Prepared {len(all_embeddings)} embeddings with {all_embeddings.shape[1]} features")
    
    def _compute_physics_embedding(self, energy, duration, time):
        """Compute comprehensive physics-aware embedding"""
        # Basic physical parameters
        logE = np.log1p(energy)
        power = energy / max(duration, 1e-6)
        energy_density = energy / (duration * duration + 1e-6)
        
        # Dimensionless parameters
        time_ratio = time / max(duration, 1e-3)
        heating_rate = power / max(time, 1e-6)
        cooling_rate = 1.0 / (time + 1e-6)
        
        # Thermal diffusion proxies
        thermal_diffusion = np.sqrt(time * 0.1) / max(duration, 1e-3)
        thermal_penetration = np.sqrt(time) / 10.0
        
        # Strain rate proxies
        strain_rate = energy_density / (time + 1e-6)
        stress_rate = power / (time + 1e-6)
        
        # Combined features
        return np.array([
            logE,
            duration,
            time,
            power,
            energy_density,
            time_ratio,
            heating_rate,
            cooling_rate,
            thermal_diffusion,
            thermal_penetration,
            strain_rate,
            stress_rate,
            np.log1p(power),
            np.log1p(time)
        ], dtype=np.float32)
    
    def _compute_spatial_similarity(self, query_meta, source_metas):
        """Compute spatial similarity based on parameter proximity"""
        similarities = []
        for meta in source_metas:
            # Normalized differences
            e_diff = abs(query_meta['energy'] - meta['energy']) / 50.0
            d_diff = abs(query_meta['duration'] - meta['duration']) / 20.0
            t_diff = abs(query_meta['time'] - meta['time']) / 50.0
            
            # Combined similarity (inverse distance)
            total_diff = np.sqrt(e_diff**2 + d_diff**2 + t_diff**2)
            similarity = np.exp(-total_diff / self.sigma_param)
            similarities.append(similarity)
        
        return np.array(similarities)
    
    def _multi_head_attention(self, query_embedding, query_meta):
        """Multi-head attention mechanism with spatial regulation"""
        if not self.fitted or len(self.source_embeddings) == 0:
            return None, None
        
        # Normalize query embedding
        query_norm = self.embedding_scaler.transform([query_embedding])[0]
        
        n_sources = len(self.source_embeddings)
        head_weights = np.zeros((self.n_heads, n_sources))
        
        # Multi-head attention
        for head in range(self.n_heads):
            np.random.seed(42 + head)  # Deterministic but different per head
            proj_dim = min(8, query_norm.shape[0])
            proj_matrix = np.random.randn(query_norm.shape[0], proj_dim)
            
            # Project embeddings
            query_proj = query_norm @ proj_matrix
            source_proj = self.source_embeddings @ proj_matrix
            
            # Compute attention scores
            distances = np.linalg.norm(query_proj - source_proj, axis=1)
            scores = np.exp(-distances**2 / (2 * self.sigma_param**2))
            
            # Apply spatial regulation if enabled
            if self.spatial_weight > 0:
                spatial_sim = self._compute_spatial_similarity(query_meta, self.source_metadata)
                scores = (1 - self.spatial_weight) * scores + self.spatial_weight * spatial_sim
            
            head_weights[head] = scores
        
        # Combine head weights
        avg_weights = np.mean(head_weights, axis=0)
        
        # Apply temperature scaling
        if self.temperature != 1.0:
            avg_weights = avg_weights ** (1.0 / self.temperature)
        
        # Softmax normalization
        max_weight = np.max(avg_weights)
        exp_weights = np.exp(avg_weights - max_weight)
        attention_weights = exp_weights / (np.sum(exp_weights) + 1e-12)
        
        # Weighted prediction (using original values, not scaled)
        if len(self.source_values) > 0:
            prediction = np.sum(attention_weights[:, np.newaxis] * self.source_values, axis=0)
        else:
            prediction = np.zeros(1)
        
        return prediction, attention_weights
    
    def predict_field_statistics(self, energy_query, duration_query, time_query):
        """Predict field statistics for given parameters"""
        if not self.fitted:
            return None
        
        # Compute query embedding and metadata
        query_embedding = self._compute_physics_embedding(energy_query, duration_query, time_query)
        query_meta = {
            'energy': energy_query,
            'duration': duration_query,
            'time': time_query
        }
        
        # Apply attention mechanism
        prediction, attention_weights = self._multi_head_attention(query_embedding, query_meta)
        
        if prediction is None:
            return None
        
        # Reconstruct field predictions
        result = {
            'prediction': prediction,
            'attention_weights': attention_weights,
            'confidence': float(np.max(attention_weights)) if len(attention_weights) > 0 else 0.0,
            'field_predictions': {}
        }
        
        # Map predictions back to fields
        if self.source_db:
            # Get field order from first summary
            field_order = sorted(self.source_db[0]['field_stats'].keys())
            n_stats_per_field = 3  # mean, max, std
            
            for i, field in enumerate(field_order):
                start_idx = i * n_stats_per_field
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
            'field_predictions': {},
            'attention_maps': [],
            'confidence_scores': []
        }
        
        # Initialize field predictions structure
        if self.source_db:
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
                        stats = pred['field_predictions'][field]
                        results['field_predictions'][field]['mean'].append(stats['mean'])
                        results['field_predictions'][field]['max'].append(stats['max'])
                        results['field_predictions'][field]['std'].append(stats['std'])
                
                results['attention_maps'].append(pred['attention_weights'])
                results['confidence_scores'].append(pred['confidence'])
            else:
                # Fill with NaN if prediction failed
                for field in results['field_predictions']:
                    results['field_predictions'][field]['mean'].append(np.nan)
                    results['field_predictions'][field]['max'].append(np.nan)
                    results['field_predictions'][field]['std'].append(np.nan)
                results['attention_maps'].append(np.array([]))
                results['confidence_scores'].append(0.0)
        
        return results

# =============================================
# ADVANCED VISUALIZATION COMPONENTS WITH EXTENDED COLORMAPS
# =============================================
class EnhancedVisualizer:
    """Comprehensive visualization with extended colormaps and advanced features"""
    
    # Extended colormaps for better visualization
    COLORSCALES = {
        'temperature': ['#2c0078', '#4402a7', '#5e04d1', '#7b0ef6', '#9a38ff', '#b966ff', '#d691ff', '#f2bcff'],
        'stress': ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600'],
        'displacement': ['#004c6d', '#346888', '#5886a5', '#7aa6c2', '#9dc6e0', '#c1e7ff'],
        'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    }
    
    # Extended colormap options for Plotly
    EXTENDED_COLORMAPS = [
        'Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis',
        'Rainbow', 'Jet', 'Hot', 'Cool', 'Portland',
        'Bluered', 'Electric', 'Thermal', 'Balance',
        'Brwnyl', 'Darkmint', 'Emrld', 'Mint', 'Oranges',
        'Purp', 'Purples', 'Sunset', 'Sunsetdark', 'Teal',
        'Tealgrn', 'Twilight', 'Burg', 'Burgyl'
    ]
    
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
                        avg_max = np.mean(stats['max'])
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
                        avg_value = np.mean(stats['mean'])
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
                
                # Add circular grid lines
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
    def create_attention_heatmap_3d(attention_weights, source_metadata):
        """Create 3D heatmap of attention weights"""
        if len(attention_weights) == 0:
            return go.Figure()
        
        # Extract metadata for 3D coordinates
        energies = []
        durations = []
        times = []
        
        for meta in source_metadata:
            energies.append(meta['energy'])
            durations.append(meta['duration'])
            times.append(meta['time'])
        
        energies = np.array(energies)
        durations = np.array(durations)
        times = np.array(times)
        
        # Create 3D scatter plot with attention weights as color
        fig = go.Figure(data=go.Scatter3d(
            x=energies,
            y=durations,
            z=times,
            mode='markers',
            marker=dict(
                size=10,
                color=attention_weights,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(
                    title="Attention Weight",
                    thickness=20,
                    len=0.5
                ),
                showscale=True
            ),
            text=[f"E: {e:.1f} mJ<br>τ: {d:.1f} ns<br>t: {t:.1f} ns<br>Weight: {w:.4f}" 
                  for e, d, t, w in zip(energies, durations, times, attention_weights)],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title="3D Attention Weight Distribution",
            scene=dict(
                xaxis_title="Energy (mJ)",
                yaxis_title="Duration (ns)",
                zaxis_title="Time (ns)",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            height=600,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig
    
    @staticmethod
    def create_attention_network(attention_weights, source_metadata, top_k=10):
        """Create network graph of attention relationships"""
        if len(attention_weights) == 0 or len(source_metadata) == 0:
            return go.Figure()
        
        # Aggregate attention by simulation
        sim_attention = {}
        for idx, (weight, meta) in enumerate(zip(attention_weights, source_metadata)):
            sim_key = meta['name']
            if sim_key not in sim_attention:
                sim_attention[sim_key] = []
            sim_attention[sim_key].append(weight)
        
        # Average attention per simulation
        avg_attention = {k: np.mean(v) for k, v in sim_attention.items()}
        
        # Get top-k simulations
        sorted_sims = sorted(avg_attention.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        if not sorted_sims:
            return go.Figure()
        
        # Create network graph
        G = nx.Graph()
        G.add_node("QUERY", size=50, color='red', label="Query")
        
        # Add simulation nodes
        for i, (sim_name, weight) in enumerate(sorted_sims):
            node_id = f"SIM_{i}"
            # Find metadata for this simulation
            sim_meta = next((m for m in source_metadata if m['name'] == sim_name), None)
            
            G.add_node(node_id,
                      size=30 * weight / max(avg_attention.values()),
                      color='blue',
                      label=sim_name,
                      energy=sim_meta['energy'] if sim_meta else 0,
                      duration=sim_meta['duration'] if sim_meta else 0,
                      weight=weight)
            
            # Add edge from query to simulation
            G.add_edge("QUERY", node_id, weight=weight, width=3 * weight)
        
        # Spring layout for node positions
        pos = nx.spring_layout(G, seed=42, k=2)
        
        # Create edge traces
        edge_x = []
        edge_y = []
        edge_text = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            weight = G[edge[0]][edge[1]]['weight']
            edge_text.append(f"Attention: {weight:.3f}")
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='gray'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node traces
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            if node == "QUERY":
                node_text.append("QUERY")
                node_size.append(30)
                node_color.append('red')
            else:
                sim_data = G.nodes[node]
                energy = sim_data.get('energy', 0)
                duration = sim_data.get('duration', 0)
                weight = sim_data.get('weight', 0)
                
                node_text.append(
                    f"Simulation: {sim_data['label']}<br>"
                    f"Energy: {energy:.1f} mJ<br>"
                    f"Duration: {duration:.1f} ns<br>"
                    f"Attention: {weight:.3f}"
                )
                node_size.append(sim_data['size'] + 10)
                node_color.append('blue')
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=[n if n == "QUERY" else f"Sim{i}" for i, n in enumerate(G.nodes()) if n != "QUERY"],
            textposition="middle center",
            hoverinfo='text',
            hovertext=node_text,
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='white')
            )
        )
        
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title=f"Attention Network (Top {len(sorted_sims)} Simulations)",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            height=500,
            plot_bgcolor='white',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
    @staticmethod
    def create_field_evolution_comparison(summaries, simulation_names, selected_field, target_sim=None):
        """Create enhanced field evolution comparison plot"""
        fig = go.Figure()
        
        for sim_name in simulation_names:
            summary = next((s for s in summaries if s['name'] == sim_name), None)
            
            if summary and selected_field in summary['field_stats']:
                stats = summary['field_stats'][selected_field]
                
                # Highlight target simulation
                line_width = 4 if target_sim and sim_name == target_sim else 2
                line_dash = 'solid' if target_sim and sim_name == target_sim else 'dash'
                
                # Plot mean
                fig.add_trace(go.Scatter(
                    x=summary['timesteps'],
                    y=stats['mean'],
                    mode='lines+markers',
                    name=f"{sim_name} (mean)",
                    line=dict(width=line_width, dash=line_dash),
                    opacity=0.8
                ))
                
                # Add confidence band (mean ± std)
                if stats['std']:
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
# MAIN INTEGRATED APPLICATION
# =============================================
def main():
    st.set_page_config(
        page_title="Enhanced FEA Laser Simulation Platform",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="🔬"
    )
    
    # Custom CSS with enhanced styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(90deg, #1E88E5, #4A00E0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 800;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3498db;
        font-weight: 600;
    }
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    .warning-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    .success-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f8f9fa;
        border-radius: 8px 8px 0 0;
        color: #495057;
        font-weight: 600;
        padding: 0 24px;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e9ecef;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5;
        color: white;
    }
    .stButton > button {
        background: linear-gradient(90deg, #1E88E5, #4A00E0);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🔬 Enhanced FEA Laser Simulation Platform</h1>', 
                unsafe_allow_html=True)
    
    # Initialize session state
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = UnifiedFEADataLoader()
        st.session_state.extrapolator = EnhancedPhysicsInformedAttentionExtrapolator()
        st.session_state.visualizer = EnhancedVisualizer()
        st.session_state.data_loaded = False
        st.session_state.current_mode = "Data Viewer"
        st.session_state.selected_colormap = "Viridis"
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Navigation")
        app_mode = st.radio(
            "Select Mode",
            ["Data Viewer", "Interpolation/Extrapolation", "Comparative Analysis"],
            index=["Data Viewer", "Interpolation/Extrapolation", "Comparative Analysis"].index(
                st.session_state.current_mode if 'current_mode' in st.session_state else "Data Viewer"
            ),
            key="nav_mode"
        )
        
        st.session_state.current_mode = app_mode
        
        st.markdown("---")
        st.markdown("### 📊 Data Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            load_full_data = st.checkbox("Load Full Mesh", value=True, 
                                        help="Load complete mesh data for 3D visualization")
        with col2:
            st.session_state.selected_colormap = st.selectbox(
                "Colormap",
                EnhancedVisualizer.EXTENDED_COLORMAPS,
                index=0
            )
        
        if st.button("🔄 Load All Simulations", type="primary", use_container_width=True):
            with st.spinner("Loading simulation data..."):
                simulations, summaries = st.session_state.data_loader.load_all_simulations(
                    load_full_mesh=load_full_data
                )
                st.session_state.simulations = simulations
                st.session_state.summaries = summaries
                
                if simulations and summaries:
                    st.session_state.extrapolator.load_summaries(summaries)
                    st.session_state.data_loaded = True
                    
                    # Store available fields
                    st.session_state.available_fields = set()
                    for summary in summaries:
                        st.session_state.available_fields.update(summary['field_stats'].keys())
        
        if st.session_state.data_loaded:
            st.markdown("---")
            st.markdown("### 📈 Loaded Data")
            
            with st.expander("Data Overview", expanded=True):
                st.metric("Simulations", len(st.session_state.simulations))
                st.metric("Available Fields", len(st.session_state.available_fields))
                
                if st.session_state.summaries:
                    energies = [s['energy'] for s in st.session_state.summaries]
                    durations = [s['duration'] for s in st.session_state.summaries]
                    st.metric("Energy Range", f"{min(energies):.1f} - {max(energies):.1f} mJ")
                    st.metric("Duration Range", f"{min(durations):.1f} - {max(durations):.1f} ns")
    
    # Main content based on selected mode
    if app_mode == "Data Viewer":
        render_data_viewer()
    elif app_mode == "Interpolation/Extrapolation":
        render_interpolation_extrapolation()
    elif app_mode == "Comparative Analysis":
        render_comparative_analysis()

def render_data_viewer():
    """Render the enhanced data visualization interface"""
    st.markdown('<h2 class="sub-header">📁 Data Viewer</h2>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.markdown("""
        <div class="warning-box">
        <h3>⚠️ No Data Loaded</h3>
        <p>Please load simulations first using the "Load All Simulations" button in the sidebar.</p>
        <p>Ensure your data follows this structure:</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📁 Expected Directory Structure"):
            st.code("""
fea_solutions/
├── q0p5mJ-delta4p2ns/        # Energy: 0.5 mJ, Duration: 4.2 ns
│   ├── a_t0001.vtu           # Timestep 1
│   ├── a_t0002.vtu           # Timestep 2
│   ├── a_t0003.vtu           # Timestep 3
│   └── ...
├── q1p0mJ-delta2p0ns/        # Energy: 1.0 mJ, Duration: 2.0 ns
│   ├── a_t0001.vtu
│   ├── a_t0002.vtu
│   └── ...
└── q2p0mJ-delta1p0ns/        # Energy: 2.0 mJ, Duration: 1.0 ns
    ├── a_t0001.vtu
    └── ...
            """)
        return
    
    simulations = st.session_state.simulations
    
    # Simulation selection
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        sim_name = st.selectbox(
            "Select Simulation",
            sorted(simulations.keys()),
            key="viewer_sim_select",
            help="Choose a simulation to visualize"
        )
    
    sim = simulations[sim_name]
    
    with col2:
        st.metric("Energy", f"{sim['energy_mJ']:.2f} mJ")
    with col3:
        st.metric("Duration", f"{sim['duration_ns']:.2f} ns")
    
    if not sim.get('has_mesh', False):
        st.warning("This simulation was loaded without mesh data. Please reload with 'Load Full Mesh' enabled.")
        return
    
    if 'field_info' not in sim or not sim['field_info']:
        st.error("No field data available for this simulation.")
        return
    
    # Field and timestep selection
    col1, col2, col3 = st.columns(3)
    with col1:
        field = st.selectbox(
            "Select Field",
            sorted(sim['field_info'].keys()),
            key="viewer_field_select",
            help="Choose a field to visualize"
        )
    with col2:
        timestep = st.slider(
            "Timestep",
            0, sim['n_timesteps'] - 1, 0,
            key="viewer_timestep_slider",
            help="Select timestep to display"
        )
    with col3:
        colormap = st.selectbox(
            "Colormap",
            EnhancedVisualizer.EXTENDED_COLORMAPS,
            index=EnhancedVisualizer.EXTENDED_COLORMAPS.index(st.session_state.selected_colormap),
            key="viewer_colormap"
        )
    
    # Main 3D visualization
    if 'points' in sim and 'fields' in sim and field in sim['fields']:
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
        
        # Create 3D visualization
        if sim.get('triangles') is not None and len(sim['triangles']) > 0:
            tri = sim['triangles']
            
            # Check triangle validity
            valid_triangles = []
            for triangle in tri:
                if all(idx < len(pts) for idx in triangle):
                    valid_triangles.append(triangle)
            
            if valid_triangles:
                valid_triangles = np.array(valid_triangles)
                mesh_data = go.Mesh3d(
                    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                    i=valid_triangles[:, 0], j=valid_triangles[:, 1], k=valid_triangles[:, 2],
                    intensity=values,
                    colorscale=colormap,
                    intensitymode='vertex',
                    colorbar=dict(
                        title=dict(text=label, font=dict(size=14)),
                        thickness=20,
                        len=0.75
                    ),
                    opacity=0.9,
                    lighting=dict(
                        ambient=0.8,
                        diffuse=0.8,
                        specular=0.5,
                        roughness=0.5
                    ),
                    lightposition=dict(x=100, y=200, z=300),
                    hovertemplate='<b>Value:</b> %{intensity:.3f}<br>' +
                                 '<b>X:</b> %{x:.3f}<br>' +
                                 '<b>Y:</b> %{y:.3f}<br>' +
                                 '<b>Z:</b> %{z:.3f}<extra></extra>'
                )
            else:
                # Fallback to scatter plot
                mesh_data = go.Scatter3d(
                    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=values,
                        colorscale=colormap,
                        opacity=0.8,
                        colorbar=dict(
                            title=dict(text=label, font=dict(size=14)),
                            thickness=20,
                            len=0.75
                        ),
                        showscale=True
                    ),
                    hovertemplate='<b>Value:</b> %{marker.color:.3f}<br>' +
                                 '<b>X:</b> %{x:.3f}<br>' +
                                 '<b>Y:</b> %{y:.3f}<br>' +
                                 '<b>Z:</b> %{z:.3f}<extra></extra>'
                )
        else:
            # Scatter plot for point cloud
            mesh_data = go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    color=values,
                    colorscale=colormap,
                    opacity=0.8,
                    colorbar=dict(
                        title=dict(text=label, font=dict(size=14)),
                        thickness=20,
                        len=0.75
                    ),
                    showscale=True
                ),
                hovertemplate='<b>Value:</b> %{marker.color:.3f}<br>' +
                             '<b>X:</b> %{x:.3f}<br>' +
                             '<b>Y:</b> %{y:.3f}<br>' +
                             '<b>Z:</b> %{z:.3f}<extra></extra>'
            )
        
        fig = go.Figure(data=mesh_data)
        fig.update_layout(
            title=dict(
                text=f"{label} at Timestep {timestep + 1}<br><sub>{sim_name}</sub>",
                font=dict(size=20)
            ),
            scene=dict(
                aspectmode="data",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    up=dict(x=0, y=0, z=1)
                ),
                xaxis=dict(
                    title="X",
                    gridcolor="lightgray",
                    showbackground=True,
                    backgroundcolor="white"
                ),
                yaxis=dict(
                    title="Y",
                    gridcolor="lightgray",
                    showbackground=True,
                    backgroundcolor="white"
                ),
                zaxis=dict(
                    title="Z",
                    gridcolor="lightgray",
                    showbackground=True,
                    backgroundcolor="white"
                )
            ),
            height=700,
            margin=dict(l=0, r=0, t=80, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Field statistics
        st.markdown('<h3 class="sub-header">📊 Field Statistics</h3>', unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Min", f"{np.min(values):.3f}")
        with col2:
            st.metric("Max", f"{np.max(values):.3f}")
        with col3:
            st.metric("Mean", f"{np.mean(values):.3f}")
        with col4:
            st.metric("Std Dev", f"{np.std(values):.3f}")
        with col5:
            st.metric("Range", f"{np.max(values) - np.min(values):.3f}")
    
    # Field evolution over time
    st.markdown('<h3 class="sub-header">📈 Field Evolution Over Time</h3>', unsafe_allow_html=True)
    
    # Find corresponding summary
    summary = next((s for s in st.session_state.summaries if s['name'] == sim_name), None)
    
    if summary and field in summary['field_stats']:
        stats = summary['field_stats'][field]
        
        fig_time = go.Figure()
        
        if stats['mean']:
            # Mean line
            fig_time.add_trace(go.Scatter(
                x=summary['timesteps'],
                y=stats['mean'],
                mode='lines',
                name='Mean',
                line=dict(color='blue', width=3)
            ))
            
            # Confidence band (mean ± std)
            if stats['std']:
                y_upper = np.array(stats['mean']) + np.array(stats['std'])
                y_lower = np.array(stats['mean']) - np.array(stats['std'])
                
                fig_time.add_trace(go.Scatter(
                    x=summary['timesteps'] + summary['timesteps'][::-1],
                    y=np.concatenate([y_upper, y_lower[::-1]]),
                    fill='toself',
                    fillcolor='rgba(0, 100, 255, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False,
                    name='± Std Dev'
                ))
            
            # Max line
            if stats['max']:
                fig_time.add_trace(go.Scatter(
                    x=summary['timesteps'],
                    y=stats['max'],
                    mode='lines',
                    name='Maximum',
                    line=dict(color='red', width=2, dash='dash')
                ))
        
        fig_time.update_layout(
            title=dict(
                text=f"{field} Statistics Over Time",
                font=dict(size=18)
            ),
            xaxis_title="Timestep (ns)",
            yaxis_title=f"{field} Value",
            hovermode="x unified",
            height=400,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        st.plotly_chart(fig_time, use_container_width=True)
        
        # Percentile analysis
        with st.expander("📊 Detailed Percentile Analysis"):
            if 'percentiles' in stats and stats['percentiles']:
                percentiles_data = np.array(stats['percentiles'])
                
                fig_percentiles = go.Figure()
                
                percentile_labels = ['10th', '25th', '50th', '75th', '90th']
                colors = ['lightblue', 'blue', 'darkblue', 'red', 'darkred']
                
                for i in range(5):
                    fig_percentiles.add_trace(go.Scatter(
                        x=summary['timesteps'],
                        y=percentiles_data[:, i],
                        mode='lines',
                        name=percentile_labels[i],
                        line=dict(color=colors[i], width=2)
                    ))
                
                fig_percentiles.update_layout(
                    title="Field Value Percentiles Over Time",
                    xaxis_title="Timestep (ns)",
                    yaxis_title=f"{field} Value",
                    height=400
                )
                
                st.plotly_chart(fig_percentiles, use_container_width=True)
    else:
        st.info(f"No time series data available for {field}")

def render_interpolation_extrapolation():
    """Render the enhanced interpolation/extrapolation interface"""
    st.markdown('<h2 class="sub-header">🔮 Interpolation/Extrapolation Engine</h2>', 
               unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.markdown("""
        <div class="warning-box">
        <h3>⚠️ No Data Loaded</h3>
        <p>Please load simulations first to enable interpolation/extrapolation capabilities.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown("""
    <div class="info-box">
    <h3>🧠 Physics-Informed Attention Mechanism</h3>
    <p>This engine uses a <strong>transformer-inspired multi-head attention mechanism</strong> with <strong>spatial locality regulation</strong> to interpolate and extrapolate simulation results. The model learns from existing FEA simulations and can predict outcomes for new parameter combinations with quantified confidence.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display loaded simulations summary
    with st.expander("📋 Loaded Simulations Summary", expanded=True):
        if st.session_state.summaries:
            df_summary = pd.DataFrame([{
                'Simulation': s['name'],
                'Energy (mJ)': s['energy'],
                'Duration (ns)': s['duration'],
                'Timesteps': len(s['timesteps']),
                'Fields': ', '.join(sorted(s['field_stats'].keys())[:3]) + ('...' if len(s['field_stats']) > 3 else '')
            } for s in st.session_state.summaries])
            
            st.dataframe(
                df_summary.style.format({
                    'Energy (mJ)': '{:.2f}',
                    'Duration (ns)': '{:.2f}'
                }).background_gradient(subset=['Energy (mJ)', 'Duration (ns)'], cmap='Blues'),
                use_container_width=True,
                height=300
            )
    
    # Query parameters
    st.markdown('<h3 class="sub-header">🎯 Query Parameters</h3>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        # Get parameter ranges from loaded data
        if st.session_state.summaries:
            energies = [s['energy'] for s in st.session_state.summaries]
            min_energy, max_energy = min(energies), max(energies)
        else:
            min_energy, max_energy = 0.1, 50.0
        
        energy_query = st.number_input(
            "Energy (mJ)",
            min_value=float(min_energy * 0.5),
            max_value=float(max_energy * 2.0),
            value=float((min_energy + max_energy) / 2),
            step=0.1,
            key="interp_energy",
            help=f"Training range: {min_energy:.1f} - {max_energy:.1f} mJ"
        )
    
    with col2:
        if st.session_state.summaries:
            durations = [s['duration'] for s in st.session_state.summaries]
            min_duration, max_duration = min(durations), max(durations)
        else:
            min_duration, max_duration = 0.5, 20.0
        
        duration_query = st.number_input(
            "Pulse Duration (ns)",
            min_value=float(min_duration * 0.5),
            max_value=float(max_duration * 2.0),
            value=float((min_duration + max_duration) / 2),
            step=0.1,
            key="interp_duration",
            help=f"Training range: {min_duration:.1f} - {max_duration:.1f} ns"
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
    
    with col4:
        time_resolution = st.selectbox(
            "Time Resolution",
            ["1 ns", "2 ns", "5 ns", "10 ns"],
            index=0,
            key="interp_resolution"
        )
    
    # Generate time points
    time_step_map = {"1 ns": 1, "2 ns": 2, "5 ns": 5, "10 ns": 10}
    time_step = time_step_map[time_resolution]
    time_points = np.arange(1, max_time + 1, time_step)
    
    # Model parameters
    with st.expander("⚙️ Attention Mechanism Parameters", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            sigma_param = st.slider(
                "Kernel Width (σ)",
                min_value=0.1,
                max_value=1.0,
                value=0.3,
                step=0.05,
                key="interp_sigma",
                help="Controls the attention focus width"
            )
        with col2:
            spatial_weight = st.slider(
                "Spatial Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                key="interp_spatial",
                help="Weight for spatial locality regulation"
            )
        with col3:
            n_heads = st.slider(
                "Attention Heads",
                min_value=1,
                max_value=8,
                value=4,
                step=1,
                key="interp_heads",
                help="Number of parallel attention heads"
            )
        with col4:
            temperature = st.slider(
                "Temperature",
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                step=0.1,
                key="interp_temp",
                help="Softmax temperature for attention weights"
            )
        
        # Update extrapolator parameters
        st.session_state.extrapolator.sigma_param = sigma_param
        st.session_state.extrapolator.spatial_weight = spatial_weight
        st.session_state.extrapolator.n_heads = n_heads
        st.session_state.extrapolator.temperature = temperature
    
    # Run prediction
    if st.button("🚀 Run Physics-Informed Prediction", type="primary", use_container_width=True):
        with st.spinner("Running multi-head attention prediction with spatial locality regulation..."):
            results = st.session_state.extrapolator.predict_time_series(
                energy_query, duration_query, time_points
            )
            
            if results and 'field_predictions' in results and results['field_predictions']:
                st.markdown("""
                <div class="success-box">
                <h3>✅ Prediction Successful</h3>
                <p>Physics-informed predictions generated using multi-head attention with spatial locality regulation.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Visualization tabs
                tab1, tab2, tab3, tab4 = st.tabs(["📈 Predictions", "🧠 Attention", "🌐 3D Analysis", "📊 Details"])
                
                with tab1:
                    render_prediction_results(results, time_points, energy_query, duration_query)
                
                with tab2:
                    render_attention_visualization(results, energy_query, duration_query, time_points)
                
                with tab3:
                    render_3d_analysis(results, time_points, energy_query, duration_query)
                
                with tab4:
                    render_detailed_results(results, time_points, energy_query, duration_query)
            else:
                st.error("Prediction failed. Please check input parameters and ensure sufficient training data.")

def render_prediction_results(results, time_points, energy_query, duration_query):
    """Render prediction results visualization"""
    # Determine which fields to plot
    available_fields = list(results['field_predictions'].keys())
    
    if not available_fields:
        st.warning("No field predictions available.")
        return
    
    # Create subplots
    n_fields = min(len(available_fields), 4)
    fig = make_subplots(
        rows=n_fields, cols=1,
        subplot_titles=[f"Predicted {field}" for field in available_fields[:n_fields]],
        vertical_spacing=0.1,
        shared_xaxes=True
    )
    
    for idx, field in enumerate(available_fields[:n_fields]):
        row = idx + 1
        
        # Plot mean prediction
        if results['field_predictions'][field]['mean']:
            fig.add_trace(
                go.Scatter(
                    x=time_points,
                    y=results['field_predictions'][field]['mean'],
                    mode='lines+markers',
                    name=f'{field} (mean)',
                    line=dict(width=3, color='blue'),
                    fillcolor='rgba(0, 0, 255, 0.1)'
                ),
                row=row, col=1
            )
        
        # Add confidence band (mean ± std)
        if (results['field_predictions'][field]['mean'] and 
            results['field_predictions'][field]['std']):
            
            mean_vals = results['field_predictions'][field]['mean']
            std_vals = results['field_predictions'][field]['std']
            
            y_upper = np.array(mean_vals) + np.array(std_vals)
            y_lower = np.array(mean_vals) - np.array(std_vals)
            
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([time_points, time_points[::-1]]),
                    y=np.concatenate([y_upper, y_lower[::-1]]),
                    fill='toself',
                    fillcolor='rgba(0, 0, 255, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False,
                    name=f'{field} ± std'
                ),
                row=row, col=1
            )
    
    fig.update_layout(
        height=300 * n_fields,
        title_text=f"Field Predictions (E={energy_query:.1f} mJ, τ={duration_query:.1f} ns)",
        showlegend=True,
        hovermode="x unified"
    )
    
    # Update y-axes
    for i in range(1, n_fields + 1):
        fig.update_yaxes(title_text="Value", row=i, col=1)
    
    fig.update_xaxes(title_text="Time (ns)", row=n_fields, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Confidence plot
    if results['confidence_scores']:
        fig_conf = go.Figure()
        fig_conf.add_trace(go.Scatter(
            x=time_points,
            y=results['confidence_scores'],
            mode='lines+markers',
            line=dict(color='orange', width=3),
            fill='tozeroy',
            fillcolor='rgba(255, 165, 0, 0.2)',
            name='Prediction Confidence'
        ))
        
        fig_conf.update_layout(
            title="Prediction Confidence Over Time",
            xaxis_title="Time (ns)",
            yaxis_title="Confidence",
            height=400,
            yaxis_range=[0, 1]
        )
        
        st.plotly_chart(fig_conf, use_container_width=True)
        
        # Confidence insights
        avg_conf = np.mean(results['confidence_scores'])
        min_conf = np.min(results['confidence_scores'])
        max_conf = np.max(results['confidence_scores'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Confidence", f"{avg_conf:.3f}")
        with col2:
            st.metric("Minimum Confidence", f"{min_conf:.3f}")
        with col3:
            st.metric("Maximum Confidence", f"{max_conf:.3f}")
        
        if avg_conf < 0.3:
            st.warning("⚠️ **Low Confidence**: Query parameters are far from training data. Extrapolation risk is high.")
        elif avg_conf < 0.6:
            st.info("ℹ️ **Moderate Confidence**: Query parameters are in extrapolation region.")
        else:
            st.success("✅ **High Confidence**: Query parameters are well-supported by training data.")

def render_attention_visualization(results, energy_query, duration_query, time_points):
    """Render attention mechanism visualizations"""
    if not results['attention_maps'] or len(results['attention_maps'][0]) == 0:
        st.info("No attention data available.")
        return
    
    st.markdown('<h4 class="sub-header">🧠 Attention Mechanism Visualization</h4>', unsafe_allow_html=True)
    
    # Select timestep for attention visualization
    selected_timestep_idx = st.slider(
        "Select timestep for attention visualization",
        0, len(time_points) - 1, 0,
        key="attention_timestep"
    )
    
    attention_weights = results['attention_maps'][selected_timestep_idx]
    selected_time = time_points[selected_timestep_idx]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 3D attention heatmap
        st.markdown("##### 3D Attention Distribution")
        heatmap_3d = st.session_state.visualizer.create_attention_heatmap_3d(
            attention_weights,
            st.session_state.extrapolator.source_metadata
        )
        if heatmap_3d.data:
            st.plotly_chart(heatmap_3d, use_container_width=True)
        else:
            st.info("Insufficient data for 3D heatmap")
    
    with col2:
        # Attention network
        st.markdown("##### Attention Network")
        network_fig = st.session_state.visualizer.create_attention_network(
            attention_weights,
            st.session_state.extrapolator.source_metadata,
            top_k=8
        )
        if network_fig.data:
            st.plotly_chart(network_fig, use_container_width=True)
        else:
            st.info("Insufficient data for network visualization")
    
    # Attention weight distribution
    st.markdown("##### Attention Weight Distribution")
    
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=attention_weights,
        nbinsx=30,
        marker_color='skyblue',
        opacity=0.7,
        name='Attention Weights'
    ))
    
    fig_dist.update_layout(
        title=f"Attention Weight Distribution at t={selected_time} ns",
        xaxis_title="Attention Weight",
        yaxis_title="Frequency",
        height=400
    )
    
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Top attention sources
    if len(attention_weights) > 0:
        # Get top attention sources
        top_indices = np.argsort(attention_weights)[-10:][::-1]
        
        st.markdown("##### Top 10 Attention Sources")
        
        top_sources_data = []
        for idx in top_indices:
            if idx < len(st.session_state.extrapolator.source_metadata):
                meta = st.session_state.extrapolator.source_metadata[idx]
                top_sources_data.append({
                    'Simulation': meta['name'],
                    'Energy (mJ)': meta['energy'],
                    'Duration (ns)': meta['duration'],
                    'Time (ns)': meta['time'],
                    'Attention Weight': attention_weights[idx]
                })
        
        if top_sources_data:
            df_top = pd.DataFrame(top_sources_data)
            st.dataframe(
                df_top.style.format({
                    'Energy (mJ)': '{:.2f}',
                    'Duration (ns)': '{:.2f}',
                    'Time (ns)': '{:.1f}',
                    'Attention Weight': '{:.4f}'
                }).background_gradient(subset=['Attention Weight'], cmap='YlOrRd'),
                use_container_width=True
            )

def render_3d_analysis(results, time_points, energy_query, duration_query):
    """Render 3D analysis visualizations"""
    st.markdown('<h4 class="sub-header">🌐 3D Parameter Space Analysis</h4>', unsafe_allow_html=True)
    
    # Create 3D parameter space visualization
    if st.session_state.summaries:
        # Extract training data
        train_energies = []
        train_durations = []
        train_max_temps = []
        train_max_stresses = []
        
        for summary in st.session_state.summaries:
            train_energies.append(summary['energy'])
            train_durations.append(summary['duration'])
            
            if 'temperature' in summary['field_stats']:
                train_max_temps.append(np.max(summary['field_stats']['temperature']['max']))
            else:
                train_max_temps.append(0)
            
            if 'principal stress' in summary['field_stats']:
                train_max_stresses.append(np.max(summary['field_stats']['principal stress']['max']))
            else:
                train_max_stresses.append(0)
        
        # Add query point
        query_max_temp = np.max(results['field_predictions'].get('temperature', {}).get('max', [0]))
        query_max_stress = np.max(results['field_predictions'].get('principal stress', {}).get('max', [0]))
        
        # Create 3D scatter plot
        col1, col2 = st.columns(2)
        
        with col1:
            # Temperature parameter space
            fig_temp = go.Figure()
            
            # Training points
            fig_temp.add_trace(go.Scatter3d(
                x=train_energies,
                y=train_durations,
                z=train_max_temps,
                mode='markers',
                marker=dict(
                    size=8,
                    color=train_max_temps,
                    colorscale='Viridis',
                    opacity=0.7,
                    colorbar=dict(title="Max Temp")
                ),
                name='Training Data',
                hovertemplate='Training<br>Energy: %{x:.1f} mJ<br>Duration: %{y:.1f} ns<br>Max Temp: %{z:.1f}<extra></extra>'
            ))
            
            # Query point
            fig_temp.add_trace(go.Scatter3d(
                x=[energy_query],
                y=[duration_query],
                z=[query_max_temp],
                mode='markers',
                marker=dict(
                    size=12,
                    color='red',
                    symbol='diamond'
                ),
                name='Query Point',
                hovertemplate='Query<br>Energy: %{x:.1f} mJ<br>Duration: %{y:.1f} ns<br>Pred Temp: %{z:.1f}<extra></extra>'
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
            # Stress parameter space
            fig_stress = go.Figure()
            
            # Training points
            fig_stress.add_trace(go.Scatter3d(
                x=train_energies,
                y=train_durations,
                z=train_max_stresses,
                mode='markers',
                marker=dict(
                    size=8,
                    color=train_max_stresses,
                    colorscale='Plasma',
                    opacity=0.7,
                    colorbar=dict(title="Max Stress")
                ),
                name='Training Data',
                hovertemplate='Training<br>Energy: %{x:.1f} mJ<br>Duration: %{y:.1f} ns<br>Max Stress: %{z:.1f}<extra></extra>'
            ))
            
            # Query point
            fig_stress.add_trace(go.Scatter3d(
                x=[energy_query],
                y=[duration_query],
                z=[query_max_stress],
                mode='markers',
                marker=dict(
                    size=12,
                    color='red',
                    symbol='diamond'
                ),
                name='Query Point',
                hovertemplate='Query<br>Energy: %{x:.1f} mJ<br>Duration: %{y:.1f} ns<br>Pred Stress: %{z:.1f}<extra></extra>'
            ))
            
            fig_stress.update_layout(
                title="Parameter Space - Maximum Stress",
                scene=dict(
                    xaxis_title="Energy (mJ)",
                    yaxis_title="Duration (ns)",
                    zaxis_title="Max Stress"
                ),
                height=500
            )
            
            st.plotly_chart(fig_stress, use_container_width=True)

def render_detailed_results(results, time_points, energy_query, duration_query):
    """Render detailed prediction results"""
    st.markdown('<h4 class="sub-header">📊 Detailed Prediction Results</h4>', unsafe_allow_html=True)
    
    # Create results table
    data_rows = []
    for idx, t in enumerate(time_points):
        row = {'Time (ns)': t}
        
        for field in results['field_predictions']:
            if field in results['field_predictions']:
                if idx < len(results['field_predictions'][field]['mean']):
                    row[f'{field}_mean'] = results['field_predictions'][field]['mean'][idx]
                    row[f'{field}_max'] = results['field_predictions'][field]['max'][idx]
                    row[f'{field}_std'] = results['field_predictions'][field]['std'][idx]
        
        if idx < len(results['confidence_scores']):
            row['confidence'] = results['confidence_scores'][idx]
        
        data_rows.append(row)
    
    if data_rows:
        df_results = pd.DataFrame(data_rows)
        
        # Format numeric columns
        format_dict = {}
        for col in df_results.columns:
            if col != 'Time (ns)':
                format_dict[col] = "{:.3f}"
        
        # Display with highlighting
        styled_df = df_results.style.format(format_dict)
        
        # Add conditional formatting for confidence
        def highlight_confidence(val):
            if isinstance(val, (int, float)):
                if val < 0.3:
                    return 'background-color: #ffcccc'
                elif val < 0.6:
                    return 'background-color: #fff4cc'
                else:
                    return 'background-color: #ccffcc'
            return ''
        
        if 'confidence' in df_results.columns:
            styled_df = styled_df.applymap(highlight_confidence, subset=['confidence'])
        
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Statistics summary
        st.markdown("##### 📈 Prediction Statistics")
        
        if 'confidence' in df_results.columns:
            conf_stats = df_results['confidence'].describe()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean Confidence", f"{conf_stats['mean']:.3f}")
            with col2:
                st.metric("Min Confidence", f"{conf_stats['min']:.3f}")
            with col3:
                st.metric("Max Confidence", f"{conf_stats['max']:.3f}")
            with col4:
                st.metric("Std Dev", f"{conf_stats['std']:.3f}")
        
        # Export options
        st.markdown("##### 💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV export
            csv = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"predictions_E{energy_query:.1f}mJ_tau{duration_query:.1f}ns.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # JSON export
            json_str = df_results.to_json(orient='records', indent=2)
            st.download_button(
                label="📥 Download as JSON",
                data=json_str.encode('utf-8'),
                file_name=f"predictions_E{energy_query:.1f}mJ_tau{duration_query:.1f}ns.json",
                mime="application/json",
                use_container_width=True
            )

def render_comparative_analysis():
    """Render enhanced comparative analysis interface"""
    st.markdown('<h2 class="sub-header">📊 Comparative Analysis</h2>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.markdown("""
        <div class="warning-box">
        <h3>⚠️ No Data Loaded</h3>
        <p>Please load simulations first to enable comparative analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    simulations = st.session_state.simulations
    summaries = st.session_state.summaries
    
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
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Sunburst", "🎯 Radar", "⏱️ Evolution", "🌐 3D Analysis"])
    
    with tab1:
        # Sunburst chart
        st.markdown("##### 📊 Hierarchical Sunburst Chart")
        sunburst_fig = st.session_state.visualizer.create_sunburst_chart(
            summaries,
            selected_field,
            highlight_sim=target_simulation
        )
        if sunburst_fig.data:
            st.plotly_chart(sunburst_fig, use_container_width=True)
        else:
            st.info("Insufficient data for sunburst chart")
    
    with tab2:
        # Radar chart
        st.markdown("##### 🎯 Multi-Field Radar Comparison")
        radar_fig = st.session_state.visualizer.create_radar_chart(
            summaries,
            visualization_sims,
            target_sim=target_simulation
        )
        if radar_fig.data:
            st.plotly_chart(radar_fig, use_container_width=True)
        else:
            st.info("Insufficient data for radar chart")
    
    with tab3:
        # Field evolution comparison
        st.markdown("##### ⏱️ Field Evolution Over Time")
        evolution_fig = st.session_state.visualizer.create_field_evolution_comparison(
            summaries,
            visualization_sims,
            selected_field,
            target_sim=target_simulation
        )
        if evolution_fig.data:
            st.plotly_chart(evolution_fig, use_container_width=True)
        else:
            st.info(f"No {selected_field} data available for selected simulations")
    
    with tab4:
        # 3D parameter space analysis
        st.markdown("##### 🌐 3D Parameter Space Analysis")
        
        if summaries:
            # Extract data for 3D plot
            energies = []
            durations = []
            max_vals = []
            sim_names = []
            is_target = []
            
            for summary in summaries:
                if summary['name'] in visualization_sims and selected_field in summary['field_stats']:
                    energies.append(summary['energy'])
                    durations.append(summary['duration'])
                    sim_names.append(summary['name'])
                    is_target.append(summary['name'] == target_simulation)
                    
                    stats = summary['field_stats'][selected_field]
                    if stats['max']:
                        max_vals.append(np.max(stats['max']))
                    else:
                        max_vals.append(0)
            
            if energies and durations and max_vals:
                # Create 3D scatter plot
                fig_3d = go.Figure()
                
                # Separate target and comparison points
                target_indices = [i for i, target in enumerate(is_target) if target]
                comp_indices = [i for i, target in enumerate(is_target) if not target]
                
                # Comparison points
                if comp_indices:
                    fig_3d.add_trace(go.Scatter3d(
                        x=[energies[i] for i in comp_indices],
                        y=[durations[i] for i in comp_indices],
                        z=[max_vals[i] for i in comp_indices],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=[max_vals[i] for i in comp_indices],
                            colorscale='Viridis',
                            opacity=0.7,
                            colorbar=dict(title=f"Max {selected_field}")
                        ),
                        name='Comparison Sims',
                        text=[sim_names[i] for i in comp_indices],
                        hovertemplate='%{text}<br>Energy: %{x:.1f} mJ<br>Duration: %{y:.1f} ns<br>Value: %{z:.1f}<extra></extra>'
                    ))
                
                # Target point
                if target_indices:
                    for idx in target_indices:
                        fig_3d.add_trace(go.Scatter3d(
                            x=[energies[idx]],
                            y=[durations[idx]],
                            z=[max_vals[idx]],
                            mode='markers',
                            marker=dict(
                                size=15,
                                color='red',
                                symbol='diamond'
                            ),
                            name='Target Sim',
                            text=sim_names[idx],
                            hovertemplate='<b>%{text}</b><br>Energy: %{x:.1f} mJ<br>Duration: %{y:.1f} ns<br>Value: %{z:.1f}<extra></extra>'
                        ))
                
                fig_3d.update_layout(
                    title=f"Parameter Space - {selected_field}",
                    scene=dict(
                        xaxis_title="Energy (mJ)",
                        yaxis_title="Duration (ns)",
                        zaxis_title=f"Max {selected_field}"
                    ),
                    height=600
                )
                
                st.plotly_chart(fig_3d, use_container_width=True)
                
                # Add convex hull for parameter space
                if len(energies) >= 3:
                    st.markdown("##### 📏 Parameter Space Coverage")
                    
                    # Calculate convex hull or bounding box
                    min_e, max_e = min(energies), max(energies)
                    min_d, max_d = min(durations), max(durations)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Energy Range", f"{min_e:.1f} - {max_e:.1f} mJ")
                    with col2:
                        st.metric("Duration Range", f"{min_d:.1f} - {max_d:.1f} ns")
                    with col3:
                        area = (max_e - min_e) * (max_d - min_d)
                        st.metric("Parameter Space Area", f"{area:.1f} mJ·ns")
    
    # Comparative statistics table
    st.markdown('<h3 class="sub-header">📋 Comparative Statistics</h3>', unsafe_allow_html=True)
    
    stats_data = []
    for sim_name in visualization_sims:
        summary = next((s for s in summaries if s['name'] == sim_name), None)
        if summary and selected_field in summary['field_stats']:
            stats = summary['field_stats'][selected_field]
            
            if stats['mean'] and stats['max']:
                row = {
                    'Simulation': sim_name,
                    'Type': 'Target' if sim_name == target_simulation else 'Comparison',
                    'Energy (mJ)': summary['energy'],
                    'Duration (ns)': summary['duration'],
                    f'Mean {selected_field}': np.mean(stats['mean']),
                    f'Max {selected_field}': np.max(stats['max']),
                    f'Std Dev {selected_field}': np.mean(stats['std']) if stats['std'] else 0,
                    'Peak Timestep': np.argmax(stats['max']) + 1 if stats['max'] else 0
                }
                stats_data.append(row)
    
    if stats_data:
        df_stats = pd.DataFrame(stats_data)
        
        # Apply styling
        def highlight_target(row):
            if row['Type'] == 'Target':
                return ['background-color: #ffcccc'] * len(row)
            return [''] * len(row)
        
        styled_df = df_stats.style.apply(highlight_target, axis=1)
        
        # Format numeric columns
        format_dict = {}
        for col in df_stats.columns:
            if col not in ['Simulation', 'Type']:
                if 'Mean' in col or 'Max' in col or 'Std Dev' in col:
                    format_dict[col] = "{:.3f}"
                elif 'Energy' in col or 'Duration' in col:
                    format_dict[col] = "{:.2f}"
        
        styled_df = styled_df.format(format_dict)
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Export options
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df_stats.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Comparison as CSV",
                data=csv,
                file_name=f"comparison_{selected_field}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Generate insights
            with st.expander("💡 Analysis Insights"):
                if len(stats_data) > 1:
                    # Calculate relative differences
                    target_data = [row for row in stats_data if row['Type'] == 'Target'][0]
                    comp_data = [row for row in stats_data if row['Type'] == 'Comparison']
                    
                    if target_data and comp_data:
                        st.write("**Target vs Comparison Simulations:**")
                        
                        for comp in comp_data:
                            energy_diff = abs(target_data['Energy (mJ)'] - comp['Energy (mJ)']) / target_data['Energy (mJ)']
                            duration_diff = abs(target_data['Duration (ns)'] - comp['Duration (ns)']) / target_data['Duration (ns)']
                            field_diff = abs(target_data[f'Mean {selected_field}'] - comp[f'Mean {selected_field}']) / target_data[f'Mean {selected_field}']
                            
                            st.write(f"- **{comp['Simulation']}**: "
                                   f"Energy diff: {energy_diff:.1%}, "
                                   f"Duration diff: {duration_diff:.1%}, "
                                   f"Field diff: {field_diff:.1%}")

if __name__ == "__main__":
    main()
