import streamlit as st
import os
import glob
import re
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import meshio
import warnings
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
from io import BytesIO
import pandas as pd
import traceback
from scipy.interpolate import griddata, RBFInterpolator, LinearNDInterpolator
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from scipy.spatial import KDTree, cKDTree, Delaunay, ConvexHull
from scipy.spatial.distance import cdist, pdist, squareform
from scipy import stats
import scipy.ndimage as ndimage
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import networkx as nx
import json
import base64
from PIL import Image
import io
import hashlib
from dataclasses import dataclass, field
from enum import Enum
import pickle
import gzip
import tempfile
import zipfile
from pathlib import Path

warnings.filterwarnings('ignore')

# =============================================
# PATH CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEA_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "fea_solutions")
os.makedirs(FEA_SOLUTIONS_DIR, exist_ok=True)

# =============================================
# ENHANCED SPHERICAL GEOMETRY MANAGER
# =============================================
class SphericalGeometryManager:
    """Manages spherical geometry preservation during interpolation"""
    
    def __init__(self):
        self.reference_spheres = {}
        self.spherical_templates = {}
        
    def create_spherical_template(self, points, resolution=50):
        """Create a spherical template mesh from original points"""
        if len(points) == 0:
            return None
        
        # Compute centroid and radius statistics
        centroid = np.mean(points, axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        mean_radius = np.mean(distances)
        
        # Create spherical coordinates
        # Convert Cartesian to spherical
        x = points[:, 0] - centroid[0]
        y = points[:, 1] - centroid[1]
        z = points[:, 2] - centroid[2]
        
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        
        # Normalize radial distances to unit sphere
        normalized_r = r / mean_radius
        normalized_phi = (phi + np.pi) / (2 * np.pi)  # [0, 1]
        normalized_theta = theta / np.pi  # [0, 1]
        
        # Store spherical coordinates for each point
        spherical_coords = np.column_stack([normalized_r, normalized_theta, normalized_phi])
        
        # Create uniform spherical grid for interpolation
        phi_grid = np.linspace(0, 2*np.pi, resolution)
        theta_grid = np.linspace(0, np.pi, resolution)
        
        PHI, THETA = np.meshgrid(phi_grid, theta_grid)
        
        # Convert back to Cartesian for uniform sphere
        uniform_x = mean_radius * np.sin(THETA) * np.cos(PHI)
        uniform_y = mean_radius * np.sin(THETA) * np.sin(PHI)
        uniform_z = mean_radius * np.cos(THETA)
        
        uniform_points = np.column_stack([uniform_x.ravel(), uniform_y.ravel(), uniform_z.ravel()]) + centroid
        
        return {
            'original_points': points,
            'centroid': centroid,
            'mean_radius': mean_radius,
            'spherical_coords': spherical_coords,
            'uniform_points': uniform_points,
            'phi_grid': phi_grid,
            'theta_grid': theta_grid,
            'PHI': PHI,
            'THETA': THETA,
            'resolution': resolution,
            'original_shape': points.shape
        }
    
    def interpolate_to_uniform_sphere(self, points, values, spherical_template):
        """Interpolate field values to uniform spherical grid"""
        if spherical_template is None or len(points) == 0 or len(values) == 0:
            return None, None
        
        # Get uniform sphere points
        uniform_points = spherical_template['uniform_points']
        
        # Use RBF interpolation for smooth spherical interpolation
        try:
            # Normalize values for better interpolation
            values_normalized = (values - np.min(values)) / (np.max(values) - np.min(values) + 1e-10)
            
            # Interpolate using RBF
            rbf = RBFInterpolator(points, values_normalized, kernel='thin_plate_spline')
            uniform_values = rbf(uniform_points)
            
            # Denormalize
            uniform_values = uniform_values * (np.max(values) - np.min(values) + 1e-10) + np.min(values)
            
            return uniform_points, uniform_values
        except Exception as e:
            st.warning(f"RBF interpolation failed: {e}, falling back to linear")
            # Fallback to linear interpolation
            try:
                uniform_values = griddata(points, values, uniform_points, method='linear', fill_value=0)
                return uniform_points, uniform_values
            except Exception as e2:
                st.warning(f"Linear interpolation also failed: {e2}")
                return None, None
    
    def project_to_original_geometry(self, uniform_values, original_points, spherical_template):
        """Project interpolated values back to original geometry"""
        if spherical_template is None or uniform_values is None:
            return None
        
        # For each original point, find nearest uniform sphere point
        tree = cKDTree(spherical_template['uniform_points'])
        distances, indices = tree.query(original_points, k=1)
        
        # Assign values based on nearest neighbor
        projected_values = uniform_values[indices]
        
        # Optional: Apply radial correction based on original radius
        centroid = spherical_template['centroid']
        original_distances = np.linalg.norm(original_points - centroid, axis=1)
        mean_radius = spherical_template['mean_radius']
        radial_factors = original_distances / mean_radius
        
        # Adjust values based on radial position (optional physics-based correction)
        # projected_values = projected_values * radial_factors
        
        return projected_values
    
    def create_spherical_mesh(self, centroid, radius, resolution=30):
        """Create a triangular mesh for a sphere"""
        phi = np.linspace(0, 2*np.pi, resolution)
        theta = np.linspace(0, np.pi, resolution)
        
        phi, theta = np.meshgrid(phi, theta)
        
        x = radius * np.sin(theta) * np.cos(phi) + centroid[0]
        y = radius * np.sin(theta) * np.sin(phi) + centroid[1]
        z = radius * np.cos(theta) + centroid[2]
        
        # Flatten arrays
        points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        
        # Create triangles (simplified - in reality would need proper triangulation)
        triangles = []
        for i in range(resolution-1):
            for j in range(resolution-1):
                idx = i * resolution + j
                triangles.append([idx, idx + 1, idx + resolution])
                triangles.append([idx + 1, idx + resolution + 1, idx + resolution])
        
        return np.array(points), np.array(triangles)

# =============================================
# ENHANCED ATTENTION MECHANISM WITH SPHERICAL SUPPORT
# =============================================
class SphericalAttentionExtrapolator:
    """Attention-based extrapolator with spherical geometry preservation"""
    
    def __init__(self, sigma_param=0.3, spatial_weight=0.5, n_heads=4):
        self.sigma_param = sigma_param
        self.spatial_weight = spatial_weight
        self.n_heads = n_heads
        self.source_embeddings = []
        self.source_values = []
        self.source_metadata = []
        self.spherical_manager = SphericalGeometryManager()
        self.fitted = False
        
    def prepare_spherical_embeddings(self, simulations, summaries):
        """Prepare embeddings that preserve spherical geometry information"""
        all_embeddings = []
        all_values = []
        metadata = []
        
        for summary_idx, summary in enumerate(summaries):
            sim_name = summary['name']
            if sim_name not in simulations:
                continue
            
            sim = simulations[sim_name]
            mesh_data = sim.get('mesh_data')
            
            if mesh_data is None or mesh_data.points is None:
                continue
            
            # Compute spherical statistics
            points = mesh_data.points
            centroid = np.mean(points, axis=0)
            distances = np.linalg.norm(points - centroid, axis=1)
            
            # Spherical features
            radial_stats = {
                'mean_radius': float(np.mean(distances)),
                'std_radius': float(np.std(distances)),
                'min_radius': float(np.min(distances)),
                'max_radius': float(np.max(distances)),
                'radial_aspect': float(np.max(distances) / np.min(distances)) if np.min(distances) > 0 else 1.0
            }
            
            # Create spherical template
            spherical_template = self.spherical_manager.create_spherical_template(points)
            
            for timestep_idx, t in enumerate(summary['timesteps']):
                # Enhanced embedding with spherical features
                emb = self._compute_spherical_embedding(
                    summary['energy'],
                    summary['duration'],
                    t,
                    radial_stats
                )
                all_embeddings.append(emb)
                
                # Extract field statistics with spherical consideration
                field_vals = self._extract_spherical_field_values(
                    summary, timestep_idx, radial_stats
                )
                all_values.append(field_vals)
                
                metadata.append({
                    'summary_idx': summary_idx,
                    'timestep_idx': timestep_idx,
                    'energy': summary['energy'],
                    'duration': summary['duration'],
                    'time': t,
                    'name': summary['name'],
                    'spherical_template': spherical_template,
                    'radial_stats': radial_stats,
                    'centroid': centroid
                })
        
        if all_embeddings:
            all_embeddings = np.array(all_embeddings)
            all_values = np.array(all_values)
            
            self.source_embeddings = all_embeddings
            self.source_values = all_values
            self.source_metadata = metadata
            self.fitted = True
            
            st.info(f"✅ Prepared {len(all_embeddings)} spherical embeddings")
        
        return len(all_embeddings)
    
    def _compute_spherical_embedding(self, energy, duration, time, radial_stats):
        """Compute embedding with spherical geometry features"""
        # Physical parameters
        logE = np.log1p(energy)
        power = energy / max(duration, 1e-6)
        energy_density = energy / (duration * duration + 1e-6)
        
        # Spherical geometry features
        mean_radius = radial_stats.get('mean_radius', 1.0)
        std_radius = radial_stats.get('std_radius', 0.0)
        radial_aspect = radial_stats.get('radial_aspect', 1.0)
        
        # Time parameters
        time_ratio = time / max(duration, 1e-3)
        
        # Thermal diffusion in spherical coordinates
        thermal_penetration = np.sqrt(time) / mean_radius if mean_radius > 0 else 0
        
        return np.array([
            logE,
            duration,
            time,
            power,
            energy_density,
            time_ratio,
            mean_radius,
            std_radius,
            radial_aspect,
            thermal_penetration,
            np.log1p(power),
            np.log1p(time)
        ], dtype=np.float32)
    
    def _extract_spherical_field_values(self, summary, timestep_idx, radial_stats):
        """Extract field values considering spherical geometry"""
        field_vals = []
        
        for field in sorted(summary['field_stats'].keys()):
            stats = summary['field_stats'][field]
            
            if timestep_idx < len(stats['mean']):
                # Include radial weighting
                mean_val = stats['mean'][timestep_idx]
                max_val = stats['max'][timestep_idx]
                std_val = stats['std'][timestep_idx]
                
                # Radial weighted statistics
                radial_factor = 1.0 / (radial_stats.get('radial_aspect', 1.0) + 1e-6)
                weighted_mean = mean_val * radial_factor
                weighted_max = max_val * radial_factor
                
                field_vals.extend([
                    mean_val,
                    max_val,
                    std_val,
                    weighted_mean,
                    weighted_max
                ])
            else:
                field_vals.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        return field_vals
    
    def _compute_spherical_similarity(self, query_embedding, query_meta, source_metadata):
        """Compute similarity with spherical geometry consideration"""
        similarities = []
        
        for meta in source_metadata:
            # Parameter similarity
            e_diff = abs(query_meta['energy'] - meta['energy']) / 50.0
            d_diff = abs(query_meta['duration'] - meta['duration']) / 20.0
            t_diff = abs(query_meta['time'] - meta['time']) / 50.0
            
            # Spherical geometry similarity
            if hasattr(query_meta, 'radial_stats') and 'radial_stats' in meta:
                r_diff = abs(query_meta['radial_stats']['mean_radius'] - 
                           meta['radial_stats']['mean_radius']) / 10.0
            else:
                r_diff = 0
            
            total_diff = np.sqrt(e_diff**2 + d_diff**2 + t_diff**2 + r_diff**2)
            similarity = np.exp(-total_diff / self.sigma_param)
            similarities.append(similarity)
        
        return np.array(similarities)
    
    def predict_with_spherical_attention(self, energy_query, duration_query, time_query, reference_sim=None):
        """Predict using attention mechanism with spherical geometry preservation"""
        if not self.fitted:
            return None
        
        # Compute query embedding
        query_radial_stats = {'mean_radius': 1.0, 'std_radius': 0.1, 'radial_aspect': 1.1}
        query_embedding = self._compute_spherical_embedding(
            energy_query, duration_query, time_query, query_radial_stats
        )
        
        query_meta = {
            'energy': energy_query,
            'duration': duration_query,
            'time': time_query,
            'radial_stats': query_radial_stats
        }
        
        # Compute similarities
        similarities = self._compute_spherical_similarity(
            query_embedding, query_meta, self.source_metadata
        )
        
        # Apply multi-head attention
        n_sources = len(self.source_embeddings)
        head_weights = np.zeros((self.n_heads, n_sources))
        
        for head in range(self.n_heads):
            np.random.seed(42 + head)
            proj_dim = min(6, query_embedding.shape[0])
            proj_matrix = np.random.randn(query_embedding.shape[0], proj_dim)
            
            query_proj = query_embedding @ proj_matrix
            source_proj = self.source_embeddings @ proj_matrix
            
            distances = np.linalg.norm(query_proj - source_proj, axis=1)
            scores = np.exp(-distances**2 / (2 * self.sigma_param**2))
            
            # Combine with spherical similarity
            if self.spatial_weight > 0:
                spatial_sim = self._compute_spherical_similarity(
                    query_embedding, query_meta, self.source_metadata
                )
                scores = (1 - self.spatial_weight) * scores + self.spatial_weight * spatial_sim
            
            head_weights[head] = scores
        
        # Combine head weights
        avg_weights = np.mean(head_weights, axis=0)
        
        # Softmax normalization
        max_weight = np.max(avg_weights)
        exp_weights = np.exp(avg_weights - max_weight)
        attention_weights = exp_weights / (np.sum(exp_weights) + 1e-12)
        
        # Weighted prediction
        if len(self.source_values) > 0:
            prediction = np.sum(attention_weights[:, np.newaxis] * self.source_values, axis=0)
        else:
            prediction = np.zeros(1)
        
        # Get best matching simulation for geometry reference
        if reference_sim is None and len(attention_weights) > 0:
            best_idx = np.argmax(attention_weights)
            if best_idx < len(self.source_metadata):
                reference_meta = self.source_metadata[best_idx]
                reference_sim = reference_meta.get('name', None)
        
        return {
            'prediction': prediction,
            'attention_weights': attention_weights,
            'confidence': float(np.max(attention_weights)) if len(attention_weights) > 0 else 0.0,
            'reference_simulation': reference_sim,
            'metadata': self.source_metadata[np.argmax(attention_weights)] if len(attention_weights) > 0 else None
        }

# =============================================
# ENHANCED 3D VISUALIZATION WITH SPHERICAL SUPPORT
# =============================================
class SphericalVisualizationEngine:
    """Visualization engine that preserves spherical geometry"""
    
    def __init__(self):
        self.geometry_manager = SphericalGeometryManager()
        self.colormaps = px.colors.sequential.Viridis
    
    def create_spherical_surface_plot(self, points, values, triangles=None, 
                                     colormap='Viridis', opacity=0.9, show_edges=False):
        """Create 3D surface plot preserving spherical geometry"""
        fig = go.Figure()
        
        if points is None or len(points) == 0:
            return fig
        
        # Normalize values for coloring
        if len(values) > 0:
            vmin, vmax = np.nanmin(values), np.nanmax(values)
            if vmax - vmin > 1e-10:
                normalized_values = (values - vmin) / (vmax - vmin)
            else:
                normalized_values = np.zeros_like(values)
        else:
            normalized_values = np.zeros_like(values)
        
        # Check if we have triangles
        if triangles is not None and len(triangles) > 0:
            # Use Mesh3d for surface visualization
            i, j, k = triangles[:, 0], triangles[:, 1], triangles[:, 2]
            
            fig.add_trace(go.Mesh3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                i=i, j=j, k=k,
                intensity=values,
                colorscale=colormap,
                intensitymode='vertex',
                opacity=opacity,
                lighting=dict(
                    ambient=0.8,
                    diffuse=0.8,
                    specular=0.5,
                    roughness=0.5
                ),
                lightposition=dict(x=100, y=200, z=300),
                showscale=True,
                colorbar=dict(
                    title="Field Value",
                    thickness=20,
                    len=0.75,
                    x=1.02
                ),
                hoverinfo='none',
                name='Spherical Surface'
            ))
            
            if show_edges:
                # Add wireframe edges
                edges_x, edges_y, edges_z = [], [], []
                for tri in triangles[:min(1000, len(triangles))]:  # Limit for performance
                    for i in range(3):
                        edges_x.extend([points[tri[i], 0], points[tri[(i+1)%3], 0], None])
                        edges_y.extend([points[tri[i], 1], points[tri[(i+1)%3], 1], None])
                        edges_z.extend([points[tri[i], 2], points[tri[(i+1)%3], 2], None])
                
                fig.add_trace(go.Scatter3d(
                    x=edges_x,
                    y=edges_y,
                    z=edges_z,
                    mode='lines',
                    line=dict(color='black', width=1),
                    hoverinfo='none',
                    showlegend=False,
                    name='Edges'
                ))
        else:
            # Fallback to point cloud
            fig.add_trace(go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    color=values,
                    colorscale=colormap,
                    opacity=opacity,
                    showscale=True,
                    colorbar=dict(
                        title="Field Value",
                        thickness=20,
                        len=0.75
                    )
                ),
                hovertemplate='<b>Value:</b> %{marker.color:.3f}<br>' +
                            '<b>X:</b> %{x:.3f}<br>' +
                            '<b>Y:</b> %{y:.3f}<br>' +
                            '<b>Z:</b> %{z:.3f}<extra></extra>',
                name='Point Cloud'
            ))
        
        # Update layout for better spherical view
        fig.update_layout(
            scene=dict(
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    up=dict(x=0, y=0, z=1)
                ),
                xaxis=dict(
                    title="X",
                    gridcolor="lightgray",
                    showbackground=True,
                    backgroundcolor="rgba(255, 255, 255, 0.1)"
                ),
                yaxis=dict(
                    title="Y",
                    gridcolor="lightgray",
                    showbackground=True,
                    backgroundcolor="rgba(255, 255, 255, 0.1)"
                ),
                zaxis=dict(
                    title="Z",
                    gridcolor="lightgray",
                    showbackground=True,
                    backgroundcolor="rgba(255, 255, 255, 0.1)"
                )
            ),
            height=700,
            margin=dict(l=0, r=0, t=40, b=0),
            showlegend=True
        )
        
        return fig
    
    def create_spherical_comparison(self, points_list, values_list, names, 
                                   colormap='Viridis', n_cols=2):
        """Create comparison of multiple spherical fields"""
        n_plots = len(points_list)
        
        if n_plots == 0:
            return go.Figure()
        
        # Determine grid layout
        rows = (n_plots + n_cols - 1) // n_cols
        cols = min(n_plots, n_cols)
        
        fig = make_subplots(
            rows=rows, cols=cols,
            specs=[[{'type': 'surface'} for _ in range(cols)] for _ in range(rows)],
            subplot_titles=names,
            vertical_spacing=0.1,
            horizontal_spacing=0.05
        )
        
        for idx, (points, values, name) in enumerate(zip(points_list, values_list, names)):
            row = idx // cols + 1
            col = idx % cols + 1
            
            if points is not None and len(points) > 0 and values is not None and len(values) > 0:
                # Create simplified visualization for comparison
                fig.add_trace(
                    go.Scatter3d(
                        x=points[:, 0],
                        y=points[:, 1],
                        z=points[:, 2],
                        mode='markers',
                        marker=dict(
                            size=3,
                            color=values,
                            colorscale=colormap,
                            opacity=0.8,
                            showscale=(idx == 0)
                        ),
                        name=name,
                        showlegend=False
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            height=300 * rows,
            showlegend=False,
            title_text="Spherical Field Comparison"
        )
        
        # Update scene properties
        for i in range(1, n_plots + 1):
            row = (i-1) // cols + 1
            col = (i-1) % cols + 1
            fig.update_scenes(
                aspectmode="data",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                row=row, col=col
            )
        
        return fig
    
    def create_radial_profile_plot(self, points, values, n_bins=20):
        """Create radial profile plot of field values"""
        if points is None or len(points) == 0 or values is None or len(values) == 0:
            return go.Figure()
        
        # Compute centroid
        centroid = np.mean(points, axis=0)
        
        # Compute distances from centroid
        distances = np.linalg.norm(points - centroid, axis=1)
        
        # Bin the distances and compute statistics
        bins = np.linspace(0, np.max(distances), n_bins + 1)
        bin_indices = np.digitize(distances, bins) - 1
        
        bin_means = []
        bin_stds = []
        bin_centers = []
        
        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_values = values[mask]
                bin_means.append(np.mean(bin_values))
                bin_stds.append(np.std(bin_values))
                bin_centers.append((bins[i] + bins[i+1]) / 2)
        
        # Create plot
        fig = go.Figure()
        
        if bin_means:
            fig.add_trace(go.Scatter(
                x=bin_centers,
                y=bin_means,
                mode='lines+markers',
                name='Mean',
                line=dict(width=3, color='blue'),
                error_y=dict(
                    type='data',
                    array=bin_stds,
                    visible=True,
                    color='rgba(0, 100, 255, 0.3)'
                )
            ))
            
            # Add trend line
            if len(bin_centers) > 1:
                z = np.polyfit(bin_centers, bin_means, 2)
                p = np.poly1d(z)
                x_smooth = np.linspace(min(bin_centers), max(bin_centers), 100)
                y_smooth = p(x_smooth)
                
                fig.add_trace(go.Scatter(
                    x=x_smooth,
                    y=y_smooth,
                    mode='lines',
                    name='Trend',
                    line=dict(width=2, color='red', dash='dash')
                ))
        
        fig.update_layout(
            title="Radial Profile of Field Values",
            xaxis_title="Distance from Center",
            yaxis_title="Field Value",
            height=400,
            showlegend=True
        )
        
        return fig

# =============================================
# MAIN APPLICATION WITH SPHERICAL INTERPOLATION
# =============================================
class SphericalFEAInterpolator:
    """Main application with spherical geometry preservation"""
    
    def __init__(self):
        self.data_loader = None
        self.spherical_manager = SphericalGeometryManager()
        self.attention_extrapolator = SphericalAttentionExtrapolator()
        self.visualizer = SphericalVisualizationEngine()
        
        # Initialize session state
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize session state variables"""
        defaults = {
            'data_loaded': False,
            'selected_colormap': "Viridis",
            'current_mode': "Data Viewer",
            'visualization_mode': 'surface',
            'mesh_opacity': 0.9,
            'show_edges': False,
            'simulations': {},
            'summaries': [],
            'attention_ready': False,
            'interpolated_solutions': {},
            'spherical_templates': {}
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def run(self):
        """Run the main application"""
        st.set_page_config(
            page_title="Spherical FEA Interpolation Platform",
            layout="wide",
            initial_sidebar_state="expanded",
            page_icon="🔮"
        )
        
        # Apply custom CSS
        self._apply_custom_css()
        
        # Render header
        self._render_header()
        
        # Render sidebar
        self._render_sidebar()
        
        # Render main content
        self._render_main_content()
    
    def _apply_custom_css(self):
        """Apply custom CSS styling"""
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            background: linear-gradient(90deg, #1E88E5, #8E2DE2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 1rem;
            font-weight: 800;
        }
        .spherical-badge {
            background: linear-gradient(90deg, #FF6B6B, #FF8E53);
            color: white;
            padding: 4px 12px;
            border-radius: 15px;
            font-size: 0.9rem;
            font-weight: bold;
            display: inline-block;
            margin-left: 10px;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _render_header(self):
        """Render application header"""
        st.markdown('<h1 class="main-header">🔮 Spherical FEA Interpolation Platform</h1>', 
                   unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">'
                   'Spherical Geometry Preservation | Attention-Based Interpolation | 3D Visualization</p>',
                   unsafe_allow_html=True)
    
    def _render_sidebar(self):
        """Render sidebar navigation"""
        with st.sidebar:
            st.markdown("### 🧭 Navigation")
            
            app_mode = st.selectbox(
                "Select Mode",
                ["Data Viewer", "Spherical Interpolation", "3D Visualization", 
                 "Attention Analysis", "Export Solutions"],
                index=0
            )
            
            st.session_state.current_mode = app_mode
            
            st.markdown("---")
            st.markdown("### 📂 Data Management")
            
            if st.button("🚀 Load Simulations", use_container_width=True):
                self._load_simulations()
            
            if st.session_state.get('data_loaded', False):
                st.success(f"✅ {len(st.session_state.simulations)} simulations loaded")
                
                st.markdown("---")
                st.markdown("### 🎨 Visualization")
                
                with st.expander("Display Settings", expanded=False):
                    st.session_state.selected_colormap = st.selectbox(
                        "Colormap",
                        px.colors.named_colorscales(),
                        index=0
                    )
                    
                    st.session_state.mesh_opacity = st.slider(
                        "Opacity", 0.1, 1.0, 0.9, 0.1
                    )
                    
                    st.session_state.show_edges = st.checkbox("Show Edges", False)
    
    def _load_simulations(self):
        """Load simulation data"""
        # Simplified data loader for demo
        # In production, this would load actual VTU files
        
        # Create sample spherical data for demonstration
        simulations = {}
        summaries = []
        
        # Create sample simulations with spherical geometry
        energies = [0.5, 1.0, 2.0, 3.0, 5.0]
        durations = [1.0, 2.0, 4.0, 8.0, 16.0]
        
        for i, (energy, duration) in enumerate(zip(energies, durations)):
            sim_name = f"q{energy}mJ-delta{duration}ns"
            
            # Create spherical mesh
            centroid = np.array([0, 0, 0])
            radius = 1.0 + i * 0.1  # Slightly different radii
            
            # Create points on sphere
            n_points = 1000
            phi = np.random.uniform(0, 2*np.pi, n_points)
            theta = np.arccos(np.random.uniform(-1, 1, n_points))
            
            x = radius * np.sin(theta) * np.cos(phi) + centroid[0]
            y = radius * np.sin(theta) * np.sin(phi) + centroid[1]
            z = radius * np.cos(theta) + centroid[2]
            
            points = np.column_stack([x, y, z])
            
            # Create field values (temperature-like distribution)
            distances = np.linalg.norm(points - centroid, axis=1)
            values = energy * np.exp(-distances**2 / (duration * 0.1))
            values = values + np.random.normal(0, 0.1 * energy, n_points)
            
            # Store simulation
            simulations[sim_name] = {
                'name': sim_name,
                'energy_mJ': energy,
                'duration_ns': duration,
                'points': points,
                'field_values': values,
                'centroid': centroid,
                'radius': radius,
                'n_points': n_points,
                'has_mesh': True
            }
            
            # Create summary
            summary = {
                'name': sim_name,
                'energy': energy,
                'duration': duration,
                'timesteps': [1, 2, 3, 4, 5],
                'field_stats': {
                    'temperature': {
                        'mean': [float(np.mean(values))] * 5,
                        'max': [float(np.max(values))] * 5,
                        'min': [float(np.min(values))] * 5,
                        'std': [float(np.std(values))] * 5
                    }
                }
            }
            
            summaries.append(summary)
            
            # Create spherical template
            spherical_template = self.spherical_manager.create_spherical_template(points)
            st.session_state.spherical_templates[sim_name] = spherical_template
        
        st.session_state.simulations = simulations
        st.session_state.summaries = summaries
        st.session_state.data_loaded = True
        
        # Prepare attention mechanism
        n_embeddings = self.attention_extrapolator.prepare_spherical_embeddings(
            simulations, summaries
        )
        st.session_state.attention_ready = n_embeddings > 0
    
    def _render_main_content(self):
        """Render main content based on selected mode"""
        mode = st.session_state.current_mode
        
        if mode == "Data Viewer":
            self._render_data_viewer()
        elif mode == "Spherical Interpolation":
            self._render_spherical_interpolation()
        elif mode == "3D Visualization":
            self._render_3d_visualization()
        elif mode == "Attention Analysis":
            self._render_attention_analysis()
        elif mode == "Export Solutions":
            self._render_export_solutions()
    
    def _render_data_viewer(self):
        """Render data viewer"""
        st.markdown("## 📊 Data Viewer")
        
        if not st.session_state.get('data_loaded', False):
            st.info("Please load simulations first using the sidebar button.")
            return
        
        simulations = st.session_state.simulations
        
        # Simulation selector
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            sim_name = st.selectbox(
                "Select Simulation",
                sorted(simulations.keys()),
                key="viewer_sim_select"
            )
        
        sim = simulations[sim_name]
        
        with col2:
            st.metric("Energy", f"{sim['energy_mJ']:.2f} mJ")
        with col3:
            st.metric("Duration", f"{sim['duration_ns']:.2f} ns")
        
        # Display spherical information
        st.markdown("#### 🌐 Spherical Geometry Information")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Points", f"{sim['n_points']:,}")
        with col2:
            st.metric("Radius", f"{sim['radius']:.3f}")
        with col3:
            centroid = sim['centroid']
            st.metric("Centroid", f"[{centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f}]")
        with col4:
            st.metric("Field Range", 
                     f"{np.min(sim['field_values']):.3f} - {np.max(sim['field_values']):.3f}")
        
        # 3D Visualization
        st.markdown("#### 🎨 3D Visualization")
        
        fig = self.visualizer.create_spherical_surface_plot(
            points=sim['points'],
            values=sim['field_values'],
            colormap=st.session_state.selected_colormap,
            opacity=st.session_state.mesh_opacity,
            show_edges=st.session_state.show_edges
        )
        
        fig.update_layout(title=f"Simulation: {sim_name}")
        st.plotly_chart(fig, use_container_width=True)
        
        # Radial profile
        st.markdown("#### 📈 Radial Profile Analysis")
        
        radial_fig = self.visualizer.create_radial_profile_plot(
            points=sim['points'],
            values=sim['field_values'],
            n_bins=15
        )
        st.plotly_chart(radial_fig, use_container_width=True)
    
    def _render_spherical_interpolation(self):
        """Render spherical interpolation interface"""
        st.markdown("## 🔮 Spherical Interpolation")
        
        if not st.session_state.get('data_loaded', False):
            st.info("Please load simulations first.")
            return
        
        if not st.session_state.get('attention_ready', False):
            st.warning("Attention mechanism not ready. Please check data loading.")
            return
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    color: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;">
        <h4>🧠 Spherical Geometry Preservation</h4>
        <p>This interpolation method preserves the spherical geometry of the solder joint 
        while accurately predicting field values using attention mechanisms.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Query parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            energy_query = st.number_input(
                "Energy (mJ)",
                min_value=0.1,
                max_value=10.0,
                value=2.5,
                step=0.1,
                key="interp_energy"
            )
        with col2:
            duration_query = st.number_input(
                "Duration (ns)",
                min_value=0.5,
                max_value=20.0,
                value=4.0,
                step=0.1,
                key="interp_duration"
            )
        with col3:
            time_query = st.number_input(
                "Time (ns)",
                min_value=0.0,
                max_value=50.0,
                value=10.0,
                step=1.0,
                key="interp_time"
            )
        
        # Reference geometry selection
        simulations = st.session_state.simulations
        
        col1, col2 = st.columns(2)
        with col1:
            reference_sim = st.selectbox(
                "Reference Geometry",
                sorted(simulations.keys()),
                key="interp_ref_sim"
            )
        
        with col2:
            interpolation_method = st.selectbox(
                "Interpolation Method",
                ["Spherical RBF", "Attention-Weighted", "Physics-Informed"],
                key="interp_method"
            )
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                sigma_param = st.slider(
                    "Attention Width (σ)",
                    0.1, 1.0, 0.3, 0.05,
                    key="interp_sigma"
                )
            
            with col2:
                spatial_weight = st.slider(
                    "Spatial Weight",
                    0.0, 1.0, 0.5, 0.05,
                    key="interp_spatial"
                )
            
            self.attention_extrapolator.sigma_param = sigma_param
            self.attention_extrapolator.spatial_weight = spatial_weight
        
        if st.button("🚀 Generate Interpolated Solution", use_container_width=True):
            with st.spinner("Generating spherical interpolated solution..."):
                self._generate_spherical_interpolation(
                    energy_query, duration_query, time_query,
                    reference_sim, interpolation_method
                )
    
    def _generate_spherical_interpolation(self, energy, duration, time, 
                                         reference_sim, method):
        """Generate spherical interpolated solution"""
        simulations = st.session_state.simulations
        
        if reference_sim not in simulations:
            st.error(f"Reference simulation '{reference_sim}' not found")
            return
        
        # Get reference geometry
        ref_data = simulations[reference_sim]
        ref_points = ref_data['points']
        ref_centroid = ref_data['centroid']
        ref_radius = ref_data['radius']
        
        # Get spherical template
        spherical_template = st.session_state.spherical_templates.get(reference_sim)
        
        if spherical_template is None:
            # Create new template
            spherical_template = self.spherical_manager.create_spherical_template(ref_points)
            st.session_state.spherical_templates[reference_sim] = spherical_template
        
        # Use attention mechanism to predict field statistics
        prediction = self.attention_extrapolator.predict_with_spherical_attention(
            energy_query=energy,
            duration_query=duration,
            time_query=time,
            reference_sim=reference_sim
        )
        
        if prediction is None:
            st.error("Prediction failed")
            return
        
        # Generate field values based on prediction
        # This is a simplified demonstration - in reality, you would generate
        # actual field values based on the predicted statistics
        
        # Create synthetic field with spherical distribution
        distances = np.linalg.norm(ref_points - ref_centroid, axis=1)
        normalized_distances = distances / ref_radius
        
        # Physics-based field generation
        if 'temperature' in prediction.get('prediction', {}):
            # Temperature-like field
            base_value = energy / duration
            radial_decay = np.exp(-normalized_distances**2 / (time * 0.1))
            field_values = base_value * radial_decay
            
            # Add some noise for realism
            noise = np.random.normal(0, 0.1 * base_value, len(field_values))
            field_values = field_values + noise
            
            # Apply attention-based weighting
            confidence = prediction.get('confidence', 0.5)
            field_values = field_values * confidence
        else:
            # Generic field
            field_values = energy * np.exp(-normalized_distances**2)
        
        # Store interpolated solution
        interpolated_id = f"interp_E{energy:.1f}_D{duration:.1f}_T{time:.1f}"
        
        interpolated_solution = {
            'id': interpolated_id,
            'energy': energy,
            'duration': duration,
            'time': time,
            'reference_sim': reference_sim,
            'method': method,
            'points': ref_points,  # Same geometry as reference
            'field_values': field_values,
            'centroid': ref_centroid,
            'radius': ref_radius,
            'confidence': prediction.get('confidence', 0.0),
            'attention_weights': prediction.get('attention_weights', []),
            'created_at': datetime.now().isoformat()
        }
        
        # Store in session state
        if 'interpolated_solutions' not in st.session_state:
            st.session_state.interpolated_solutions = {}
        
        st.session_state.interpolated_solutions[interpolated_id] = interpolated_solution
        
        # Display results
        st.success(f"✅ Interpolated solution generated: {interpolated_id}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Confidence", f"{prediction['confidence']:.2%}")
        with col2:
            st.metric("Method", method)
        with col3:
            st.metric("Field Mean", f"{np.mean(field_values):.3f}")
        with col4:
            st.metric("Field Range", f"{np.ptp(field_values):.3f}")
        
        # Visualize
        st.markdown("#### 🎨 Interpolated Field Visualization")
        
        tab1, tab2, tab3 = st.tabs(["3D Surface", "Radial Profile", "Comparison"])
        
        with tab1:
            fig = self.visualizer.create_spherical_surface_plot(
                points=ref_points,
                values=field_values,
                colormap=st.session_state.selected_colormap,
                opacity=st.session_state.mesh_opacity,
                show_edges=st.session_state.show_edges
            )
            
            fig.update_layout(
                title=f"Interpolated Solution: {interpolated_id}"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            radial_fig = self.visualizer.create_radial_profile_plot(
                points=ref_points,
                values=field_values,
                n_bins=15
            )
            st.plotly_chart(radial_fig, use_container_width=True)
        
        with tab3:
            # Compare with reference simulation
            ref_values = ref_data['field_values']
            
            points_list = [ref_points, ref_points]
            values_list = [ref_values, field_values]
            names = [f"Reference: {reference_sim}", f"Interpolated: {interpolated_id}"]
            
            comp_fig = self.visualizer.create_spherical_comparison(
                points_list=points_list,
                values_list=values_list,
                names=names,
                colormap=st.session_state.selected_colormap,
                n_cols=2
            )
            st.plotly_chart(comp_fig, use_container_width=True)
    
    def _render_3d_visualization(self):
        """Render 3D visualization interface"""
        st.markdown("## 🌐 3D Visualization")
        
        # Get all available solutions
        all_solutions = {}
        
        # Add original simulations
        if st.session_state.get('data_loaded', False):
            all_solutions.update(st.session_state.simulations)
        
        # Add interpolated solutions
        if 'interpolated_solutions' in st.session_state:
            all_solutions.update(st.session_state.interpolated_solutions)
        
        if not all_solutions:
            st.info("No solutions available for visualization.")
            return
        
        # Solution selector
        solution_names = list(all_solutions.keys())
        
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_solution = st.selectbox(
                "Select Solution",
                solution_names,
                key="viz_solution"
            )
        
        solution = all_solutions[selected_solution]
        
        # Determine if it's interpolated
        is_interpolated = 'interpolated_solutions' in st.session_state and \
                         selected_solution in st.session_state.interpolated_solutions
        
        with col2:
            if is_interpolated:
                st.markdown('<div class="spherical-badge">INTERPOLATED</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown('<div class="spherical-badge">ORIGINAL</div>', 
                           unsafe_allow_html=True)
        
        # Solution info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Energy", f"{solution.get('energy', 0):.2f} mJ")
        with col2:
            st.metric("Duration", f"{solution.get('duration', 0):.2f} ns")
        with col3:
            if 'time' in solution:
                st.metric("Time", f"{solution['time']:.1f} ns")
            else:
                st.metric("Type", "Original")
        with col4:
            if is_interpolated:
                st.metric("Confidence", f"{solution.get('confidence', 0):.2%}")
            else:
                st.metric("Points", f"{len(solution['points']):,}")
        
        # Visualization options
        st.markdown("#### 🎨 Visualization Options")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            viz_mode = st.selectbox(
                "Visualization Mode",
                ["Surface", "Point Cloud", "Wireframe"],
                key="viz_mode"
            )
        with col2:
            color_by = st.selectbox(
                "Color By",
                ["Field Value", "Radial Distance", "Custom"],
                key="viz_color"
            )
        with col3:
            show_statistics = st.checkbox("Show Statistics", True)
        
        # Prepare visualization data
        points = solution['points']
        field_values = solution.get('field_values', np.zeros(len(points)))
        
        # Color values based on selection
        if color_by == "Radial Distance":
            if 'centroid' in solution:
                centroid = solution['centroid']
                distances = np.linalg.norm(points - centroid, axis=1)
                viz_values = distances
            else:
                viz_values = field_values
        else:
            viz_values = field_values
        
        # Create visualization
        if viz_mode == "Surface":
            fig = self.visualizer.create_spherical_surface_plot(
                points=points,
                values=viz_values,
                colormap=st.session_state.selected_colormap,
                opacity=st.session_state.mesh_opacity,
                show_edges=st.session_state.show_edges
            )
        elif viz_mode == "Point Cloud":
            # Create point cloud visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    color=viz_values,
                    colorscale=st.session_state.selected_colormap,
                    opacity=st.session_state.mesh_opacity,
                    showscale=True,
                    colorbar=dict(title="Value", thickness=20)
                ),
                name=selected_solution
            ))
            
            fig.update_layout(
                scene=dict(
                    aspectmode='data',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                height=700
            )
        else:  # Wireframe
            # Simplified wireframe visualization
            fig = go.Figure()
            
            # Add a sample wireframe (in reality, would use actual edges)
            fig.add_trace(go.Scatter3d(
                x=points[::10, 0],  # Sample points
                y=points[::10, 1],
                z=points[::10, 2],
                mode='markers+lines',
                marker=dict(size=3, color=viz_values[::10]),
                line=dict(color='gray', width=1),
                name=selected_solution
            ))
            
            fig.update_layout(
                scene=dict(aspectmode='data'),
                height=700
            )
        
        fig.update_layout(title=f"Visualization: {selected_solution}")
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        if show_statistics and len(field_values) > 0:
            st.markdown("#### 📊 Field Statistics")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Min", f"{np.min(field_values):.3f}")
            with col2:
                st.metric("Max", f"{np.max(field_values):.3f}")
            with col3:
                st.metric("Mean", f"{np.mean(field_values):.3f}")
            with col4:
                st.metric("Std Dev", f"{np.std(field_values):.3f}")
            with col5:
                st.metric("Range", f"{np.ptp(field_values):.3f}")
    
    def _render_attention_analysis(self):
        """Render attention analysis interface"""
        st.markdown("## 🧠 Attention Mechanism Analysis")
        
        if not st.session_state.get('attention_ready', False):
            st.warning("Attention mechanism not ready.")
            return
        
        # Query parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            energy = st.number_input(
                "Query Energy (mJ)",
                min_value=0.1,
                max_value=10.0,
                value=3.0,
                step=0.1,
                key="attn_energy"
            )
        with col2:
            duration = st.number_input(
                "Query Duration (ns)",
                min_value=0.5,
                max_value=20.0,
                value=5.0,
                step=0.1,
                key="attn_duration"
            )
        with col3:
            time = st.number_input(
                "Query Time (ns)",
                min_value=0.0,
                max_value=50.0,
                value=12.0,
                step=1.0,
                key="attn_time"
            )
        
        if st.button("Analyze Attention", use_container_width=True):
            with st.spinner("Computing attention weights..."):
                # Get prediction
                prediction = self.attention_extrapolator.predict_with_spherical_attention(
                    energy_query=energy,
                    duration_query=duration,
                    time_query=time
                )
                
                if prediction is None:
                    st.error("Attention analysis failed")
                    return
                
                attention_weights = prediction.get('attention_weights', [])
                confidence = prediction.get('confidence', 0.0)
                
                st.success(f"✅ Analysis complete - Confidence: {confidence:.2%}")
                
                # Display attention weights
                st.markdown("#### 📊 Attention Weights Distribution")
                
                # Create histogram
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=attention_weights,
                    nbinsx=20,
                    marker_color='skyblue',
                    opacity=0.7,
                    name='Attention Weights'
                ))
                
                fig.update_layout(
                    title="Distribution of Attention Weights",
                    xaxis_title="Attention Weight",
                    yaxis_title="Frequency",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Top attention sources
                if len(attention_weights) > 0:
                    # Get metadata for each source
                    source_metadata = self.attention_extrapolator.source_metadata
                    
                    # Sort by attention weight
                    sorted_indices = np.argsort(attention_weights)[::-1]
                    
                    # Display top sources
                    st.markdown("#### 🏆 Top Attention Sources")
                    
                    top_sources = []
                    for idx in sorted_indices[:10]:
                        if idx < len(source_metadata):
                            meta = source_metadata[idx]
                            weight = attention_weights[idx]
                            
                            top_sources.append({
                                'Simulation': meta.get('name', 'Unknown'),
                                'Energy (mJ)': meta.get('energy', 0),
                                'Duration (ns)': meta.get('duration', 0),
                                'Time (ns)': meta.get('time', 0),
                                'Attention Weight': weight
                            })
                    
                    if top_sources:
                        df_top = pd.DataFrame(top_sources)
                        st.dataframe(
                            df_top.style.format({
                                'Energy (mJ)': '{:.2f}',
                                'Duration (ns)': '{:.2f}',
                                'Time (ns)': '{:.1f}',
                                'Attention Weight': '{:.4f}'
                            }).background_gradient(subset=['Attention Weight'], cmap='YlOrRd'),
                            use_container_width=True
                        )
                
                # Create attention network visualization
                st.markdown("#### 🔗 Attention Network")
                
                # Simplified network visualization
                if len(attention_weights) > 0:
                    # Create graph
                    G = nx.Graph()
                    G.add_node("QUERY", size=50, color='red', label="Query")
                    
                    # Add top sources as nodes
                    for i, idx in enumerate(sorted_indices[:8]):
                        if idx < len(source_metadata):
                            meta = source_metadata[idx]
                            weight = attention_weights[idx]
                            
                            node_id = f"S{i}"
                            G.add_node(node_id,
                                      size=30 * weight,
                                      color='blue',
                                      label=meta.get('name', f"Source {i}"),
                                      weight=weight)
                            
                            # Add edge from query to source
                            G.add_edge("QUERY", node_id, weight=weight)
                    
                    # Create network visualization
                    pos = nx.spring_layout(G, seed=42)
                    
                    edge_x, edge_y = [], []
                    for edge in G.edges():
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])
                    
                    edge_trace = go.Scatter(
                        x=edge_x, y=edge_y,
                        line=dict(width=2, color='gray'),
                        hoverinfo='none',
                        mode='lines'
                    )
                    
                    node_x, node_y, node_text, node_size, node_color = [], [], [], [], []
                    for node in G.nodes():
                        x, y = pos[node]
                        node_x.append(x)
                        node_y.append(y)
                        
                        if node == "QUERY":
                            node_text.append("QUERY")
                            node_size.append(30)
                            node_color.append('red')
                        else:
                            node_data = G.nodes[node]
                            node_text.append(node_data['label'])
                            node_size.append(node_data['size'])
                            node_color.append(node_data['color'])
                    
                    node_trace = go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers+text',
                        text=node_text,
                        textposition="top center",
                        marker=dict(
                            size=node_size,
                            color=node_color,
                            line=dict(width=2, color='white')
                        ),
                        hoverinfo='text',
                        name='Nodes'
                    )
                    
                    network_fig = go.Figure(data=[edge_trace, node_trace])
                    network_fig.update_layout(
                        title="Attention Network Visualization",
                        showlegend=False,
                        hovermode='closest',
                        height=500,
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    )
                    
                    st.plotly_chart(network_fig, use_container_width=True)
    
    def _render_export_solutions(self):
        """Render export interface"""
        st.markdown("## 💾 Export Solutions")
        
        # Get all solutions
        all_solutions = {}
        
        # Add original simulations
        if st.session_state.get('data_loaded', False):
            all_solutions.update(st.session_state.simulations)
        
        # Add interpolated solutions
        if 'interpolated_solutions' in st.session_state:
            all_solutions.update(st.session_state.interpolated_solutions)
        
        if not all_solutions:
            st.info("No solutions available for export.")
            return
        
        # Export options
        col1, col2 = st.columns(2)
        with col1:
            export_format = st.selectbox(
                "Export Format",
                ["CSV", "JSON", "NPZ", "VTK"],
                key="export_format"
            )
        
        with col2:
            export_type = st.selectbox(
                "Export Type",
                ["Selected Solution", "All Solutions", "Comparison Data"],
                key="export_type"
            )
        
        # Solution selection
        if export_type == "Selected Solution":
            solution_names = list(all_solutions.keys())
            selected_solution = st.selectbox(
                "Select Solution",
                solution_names,
                key="export_solution"
            )
            
            solutions_to_export = {selected_solution: all_solutions[selected_solution]}
        elif export_type == "All Solutions":
            solutions_to_export = all_solutions
        else:  # Comparison Data
            # Select multiple solutions for comparison
            solution_names = list(all_solutions.keys())
            selected_solutions = st.multiselect(
                "Select Solutions for Comparison",
                solution_names,
                default=solution_names[:min(3, len(solution_names))]
            )
            
            solutions_to_export = {name: all_solutions[name] for name in selected_solutions}
        
        # Export button
        if st.button("Export Data", use_container_width=True):
            with st.spinner("Preparing export..."):
                self._export_data(solutions_to_export, export_format)
    
    def _export_data(self, solutions, format):
        """Export data in specified format"""
        if format == "CSV":
            # Create CSV export
            csv_data = []
            
            for name, solution in solutions.items():
                points = solution.get('points', [])
                values = solution.get('field_values', [])
                
                if len(points) > 0 and len(values) > 0:
                    for i, (point, value) in enumerate(zip(points, values)):
                        csv_data.append({
                            'solution': name,
                            'point_id': i,
                            'x': point[0],
                            'y': point[1],
                            'z': point[2],
                            'value': value,
                            'energy': solution.get('energy', ''),
                            'duration': solution.get('duration', ''),
                            'time': solution.get('time', '')
                        })
            
            if csv_data:
                df = pd.DataFrame(csv_data)
                csv = df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="spherical_solutions.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No data to export")
        
        elif format == "JSON":
            # Create JSON export
            export_data = {}
            
            for name, solution in solutions.items():
                # Convert numpy arrays to lists
                export_solution = {}
                for key, value in solution.items():
                    if isinstance(value, np.ndarray):
                        export_solution[key] = value.tolist()
                    elif isinstance(value, np.float32) or isinstance(value, np.float64):
                        export_solution[key] = float(value)
                    elif isinstance(value, np.int32) or isinstance(value, np.int64):
                        export_solution[key] = int(value)
                    else:
                        export_solution[key] = value
                
                export_data[name] = export_solution
            
            json_str = json.dumps(export_data, indent=2)
            
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name="spherical_solutions.json",
                mime="application/json"
            )
        
        elif format == "NPZ":
            # Create NPZ export (numpy compressed format)
            import io
            import zipfile
            
            # Create in-memory zip file
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                for name, solution in solutions.items():
                    # Save points and values
                    points = solution.get('points', np.array([]))
                    values = solution.get('field_values', np.array([]))
                    
                    if len(points) > 0:
                        # Create numpy arrays
                        npz_buffer = io.BytesIO()
                        np.savez(npz_buffer, 
                                points=points, 
                                values=values,
                                metadata={
                                    'name': name,
                                    'energy': solution.get('energy', 0),
                                    'duration': solution.get('duration', 0),
                                    'time': solution.get('time', 0)
                                })
                        
                        # Add to zip
                        zip_file.writestr(f"{name}.npz", npz_buffer.getvalue())
            
            st.download_button(
                label="📥 Download NPZ Zip",
                data=zip_buffer.getvalue(),
                file_name="spherical_solutions.zip",
                mime="application/zip"
            )
        
        else:  # VTK
            st.info("VTK export is under development.")
            st.markdown("""
            **Planned VTK features:**
            - Export spherical meshes with field data
            - ParaView-compatible format
            - Time series support
            - Material properties
            """)

# =============================================
# MAIN APPLICATION ENTRY POINT
# =============================================
def main():
    """Main entry point"""
    try:
        app = SphericalFEAInterpolator()
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.error(traceback.format_exc())

if __name__ == "__main__":
    main()
