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
from scipy.interpolate import griddata, RBFInterpolator, LinearNDInterpolator, CloughTocher2DInterpolator
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from scipy.spatial import KDTree, cKDTree, Delaunay, distance
from scipy.spatial.distance import cdist, pdist, squareform
from scipy import stats
import scipy.ndimage as ndimage
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import networkx as nx
import json
import base64
import hashlib
import tempfile
import zipfile
from pathlib import Path
import pickle

warnings.filterwarnings('ignore')

# =============================================
# CONSTANTS AND CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEA_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "fea_solutions")
INTERPOLATED_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "interpolated_solutions")
os.makedirs(FEA_SOLUTIONS_DIR, exist_ok=True)
os.makedirs(INTERPOLATED_SOLUTIONS_DIR, exist_ok=True)

# =============================================
# ENHANCED MESH DATA WITH SPHERICAL GEOMETRY
# =============================================
class EnhancedMeshData:
    """Enhanced mesh data with spherical geometry preservation"""
    
    def __init__(self):
        self.points = None
        self.triangles = None
        self.normals = None
        self.curvature = None
        self.radial_distances = None
        self.angular_coordinates = None
        self.spherical_harmonics = None
        self.metadata = {}
        self.is_spherical = False
        
    def compute_spherical_coordinates(self, center=None):
        """Convert Cartesian coordinates to spherical coordinates"""
        if self.points is None or len(self.points) == 0:
            return
        
        if center is None:
            center = np.mean(self.points, axis=0)
        
        # Compute radial distances
        self.radial_distances = np.linalg.norm(self.points - center, axis=1)
        
        # Compute angular coordinates (theta, phi)
        centered_points = self.points - center
        self.angular_coordinates = np.zeros((len(self.points), 2))
        
        # Theta (polar angle): 0 to pi
        self.angular_coordinates[:, 0] = np.arccos(centered_points[:, 2] / (self.radial_distances + 1e-10))
        
        # Phi (azimuthal angle): 0 to 2pi
        self.angular_coordinates[:, 1] = np.arctan2(centered_points[:, 1], centered_points[:, 0])
        self.angular_coordinates[:, 1][self.angular_coordinates[:, 1] < 0] += 2 * np.pi
        
        # Check if geometry is approximately spherical
        radius_std = np.std(self.radial_distances)
        radius_mean = np.mean(self.radial_distances)
        self.is_spherical = radius_std / radius_mean < 0.2  # Less than 20% variation
        
        self.metadata['center'] = center.tolist()
        self.metadata['avg_radius'] = float(radius_mean)
        self.metadata['radius_std'] = float(radius_std)
        self.metadata['is_spherical'] = self.is_spherical
        
    def compute_normals(self):
        """Compute vertex normals for the mesh"""
        if self.triangles is None or len(self.triangles) == 0:
            return
        
        normals = np.zeros_like(self.points)
        face_normals = np.zeros((len(self.triangles), 3))
        
        for i, tri in enumerate(self.triangles):
            v0, v1, v2 = self.points[tri[0]], self.points[tri[1]], self.points[tri[2]]
            face_normals[i] = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(face_normals[i])
            if norm > 0:
                face_normals[i] /= norm
            
            normals[tri[0]] += face_normals[i]
            normals[tri[1]] += face_normals[i]
            normals[tri[2]] += face_normals[i]
        
        # Normalize vertex normals
        norms = np.linalg.norm(normals, axis=1)
        norms[norms == 0] = 1
        self.normals = normals / norms[:, np.newaxis]
        
    def compute_curvature(self):
        """Compute curvature using normal variation"""
        if self.normals is None:
            self.compute_normals()
        
        if self.normals is None:
            return
        
        # Build KD-tree for neighbor search
        tree = cKDTree(self.points)
        
        curvature = np.zeros(len(self.points))
        for i, point in enumerate(self.points):
            # Find nearest neighbors
            distances, indices = tree.query(point, k=10)
            
            # Compute normal variation
            neighbor_normals = self.normals[indices[1:]]  # Exclude self
            normal_variation = np.arccos(np.clip(
                np.dot(self.normals[i], neighbor_normals.T), -1.0, 1.0
            ))
            
            curvature[i] = np.mean(normal_variation)
        
        self.curvature = curvature
        
    def resample_to_spherical_grid(self, n_theta=50, n_phi=100):
        """Resample mesh data to regular spherical grid"""
        if self.angular_coordinates is None:
            self.compute_spherical_coordinates()
        
        # Create regular spherical grid
        theta_grid = np.linspace(0, np.pi, n_theta)
        phi_grid = np.linspace(0, 2 * np.pi, n_phi)
        
        # Create meshgrid
        THETA, PHI = np.meshgrid(theta_grid, phi_grid)
        
        # Convert to Cartesian for points on unit sphere
        X = np.sin(THETA) * np.cos(PHI)
        Y = np.sin(THETA) * np.sin(PHI)
        Z = np.cos(THETA)
        
        grid_points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        
        return grid_points, THETA, PHI
        
    def interpolate_to_spherical_grid(self, values, n_theta=50, n_phi=100):
        """Interpolate field values to regular spherical grid"""
        if self.angular_coordinates is None:
            self.compute_spherical_coordinates()
        
        # Create regular spherical grid
        theta_grid = np.linspace(0, np.pi, n_theta)
        phi_grid = np.linspace(0, 2 * np.pi, n_phi)
        THETA, PHI = np.meshgrid(theta_grid, phi_grid)
        
        # Flatten the grid
        grid_theta_phi = np.stack([THETA.ravel(), PHI.ravel()], axis=1)
        
        # Interpolate using barycentric coordinates on sphere
        grid_values = griddata(
            self.angular_coordinates,
            values,
            grid_theta_phi,
            method='linear',
            fill_value=0
        )
        
        return grid_values.reshape(THETA.shape), THETA, PHI

# =============================================
# SPHERICAL HARMONICS INTERPOLATOR
# =============================================
class SphericalHarmonicsInterpolator:
    """Spherical harmonics-based interpolation for spherical geometries"""
    
    def __init__(self, max_degree=10):
        self.max_degree = max_degree
        self.coefficients = None
        self.theta_grid = None
        self.phi_grid = None
        
    def fit(self, theta, phi, values):
        """Fit spherical harmonics to scattered data"""
        from scipy.special import sph_harm
        
        # Create grid for spherical harmonics
        m = np.arange(-self.max_degree, self.max_degree + 1)
        n = np.abs(m)
        
        # Compute spherical harmonics basis
        Y = []
        for l in range(self.max_degree + 1):
            for m in range(-l, l + 1):
                Y.append(sph_harm(m, l, phi, theta))
        
        Y = np.array(Y).T  # Shape: (n_points, n_basis)
        
        # Solve for coefficients using least squares
        self.coefficients, _, _, _ = np.linalg.lstsq(Y, values, rcond=None)
        
    def predict(self, theta, phi):
        """Predict values at new spherical coordinates"""
        from scipy.special import sph_harm
        
        Y = []
        idx = 0
        for l in range(self.max_degree + 1):
            for m in range(-l, l + 1):
                Y.append(sph_harm(m, l, phi, theta))
        
        Y = np.array(Y).T
        return Y @ self.coefficients

# =============================================
# ENHANCED PHYSICS-AWARE ATTENTION MECHANISM
# =============================================
class EnhancedPhysicsAwareAttention:
    """Enhanced attention mechanism with physics-aware embeddings and spherical geometry support"""
    
    def __init__(self, n_heads=4, temperature=1.0, spatial_weight=0.5):
        self.n_heads = n_heads
        self.temperature = temperature
        self.spatial_weight = spatial_weight
        self.embeddings = {}
        self.scaler = StandardScaler()
        self.source_db = []
        self.attention_weights = []
        self.attention_history = []
        
    def create_physics_aware_embedding(self, energy, duration, time, mesh_data):
        """Create comprehensive physics-aware embedding including geometric features"""
        
        # Basic physical parameters
        logE = np.log1p(energy)
        power = energy / max(duration, 1e-6)
        fluence = energy / (duration ** 2 + 1e-6)
        
        # Temporal features
        time_ratio = time / max(duration, 1e-3)
        heating_rate = power / max(time, 1e-6)
        
        # Geometric features from mesh
        geometric_features = []
        if mesh_data and hasattr(mesh_data, 'metadata'):
            geometric_features.extend([
                mesh_data.metadata.get('avg_radius', 0),
                mesh_data.metadata.get('radius_std', 0),
                float(mesh_data.is_spherical)
            ])
            
            if hasattr(mesh_data, 'curvature') and mesh_data.curvature is not None:
                geometric_features.extend([
                    np.mean(mesh_data.curvature),
                    np.std(mesh_data.curvature),
                    np.max(mesh_data.curvature)
                ])
        
        # Combine all features
        base_features = [
            logE, duration, time, power, fluence,
            time_ratio, heating_rate,
            np.log1p(power), np.log1p(fluence)
        ]
        
        embedding = np.array(base_features + geometric_features, dtype=np.float32)
        
        # Normalize
        if len(self.source_db) > 0:
            # Update scaler if we have data
            all_embeddings = []
            for sim_data in self.source_db:
                if 'embedding' in sim_data:
                    all_embeddings.append(sim_data['embedding'])
            
            if len(all_embeddings) > 0:
                self.scaler.fit(np.array(all_embeddings))
                embedding = self.scaler.transform([embedding])[0]
        
        return embedding
    
    def compute_similarity(self, query_embedding, source_embeddings):
        """Compute attention-based similarity with multi-head mechanism"""
        
        n_sources = len(source_embeddings)
        if n_sources == 0:
            return np.array([])
        
        # Multi-head attention
        head_similarities = np.zeros((self.n_heads, n_sources))
        
        for head in range(self.n_heads):
            # Different random projection for each head
            np.random.seed(head * 42)
            proj_dim = min(8, query_embedding.shape[0])
            proj_matrix = np.random.randn(query_embedding.shape[0], proj_dim)
            
            # Project embeddings
            query_proj = query_embedding @ proj_matrix
            source_proj = source_embeddings @ proj_matrix
            
            # Compute cosine similarity
            query_norm = np.linalg.norm(query_proj)
            source_norms = np.linalg.norm(source_proj, axis=1)
            
            similarities = np.dot(source_proj, query_proj) / (source_norms * query_norm + 1e-10)
            
            # Apply temperature scaling
            similarities = similarities ** (1.0 / self.temperature)
            
            head_similarities[head] = similarities
        
        # Combine head similarities
        combined_similarities = np.mean(head_similarities, axis=0)
        
        # Apply spatial weighting if available
        if self.spatial_weight > 0 and hasattr(self, 'spatial_distances'):
            spatial_sim = 1.0 / (1.0 + self.spatial_distances)
            combined_similarities = (1 - self.spatial_weight) * combined_similarities + self.spatial_weight * spatial_sim
        
        # Softmax normalization
        max_sim = np.max(combined_similarities)
        exp_sim = np.exp(combined_similarities - max_sim)
        attention_weights = exp_sim / (np.sum(exp_sim) + 1e-10)
        
        return attention_weights
    
    def update_attention_history(self, query_params, attention_weights, source_indices):
        """Update attention history for visualization"""
        self.attention_history.append({
            'query': query_params,
            'weights': attention_weights.copy(),
            'source_indices': source_indices.copy(),
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 100 entries
        if len(self.attention_history) > 100:
            self.attention_history = self.attention_history[-100:]

# =============================================
# SPHERICAL FIELD INTERPOLATOR
# =============================================
class SphericalFieldInterpolator:
    """Field interpolator with spherical geometry preservation"""
    
    def __init__(self, method='spherical_rbf'):
        self.method = method
        self.interpolator = None
        self.reference_mesh = None
        self.spherical_coords = None
        
    def prepare_interpolation(self, source_mesh, source_values):
        """Prepare interpolator with source data"""
        self.reference_mesh = source_mesh
        
        # Compute spherical coordinates
        if source_mesh.angular_coordinates is None:
            source_mesh.compute_spherical_coordinates()
        
        self.spherical_coords = source_mesh.angular_coordinates
        
        # Choose interpolation method
        if self.method == 'spherical_rbf':
            # Use spherical RBF interpolation
            self.interpolator = RBFInterpolator(
                self.spherical_coords,
                source_values,
                kernel='thin_plate_spline',
                epsilon=0.1
            )
        elif self.method == 'barycentric':
            # Create Delaunay triangulation on sphere
            self.interpolator = LinearNDInterpolator(
                self.spherical_coords,
                source_values,
                fill_value=0
            )
        elif self.method == 'spherical_harmonics':
            self.interpolator = SphericalHarmonicsInterpolator(max_degree=10)
            theta, phi = source_mesh.angular_coordinates[:, 0], source_mesh.angular_coordinates[:, 1]
            self.interpolator.fit(theta, phi, source_values)
        
    def interpolate(self, target_mesh, method=None):
        """Interpolate field to target mesh"""
        if self.interpolator is None:
            raise ValueError("Interpolator not prepared. Call prepare_interpolation first.")
        
        if target_mesh.angular_coordinates is None:
            target_mesh.compute_spherical_coordinates()
        
        target_coords = target_mesh.angular_coordinates
        
        if method is None:
            method = self.method
        
        if method in ['spherical_rbf', 'barycentric']:
            # Use the prepared interpolator
            interpolated_values = self.interpolator(target_coords)
        elif method == 'spherical_harmonics':
            theta, phi = target_coords[:, 0], target_coords[:, 1]
            interpolated_values = self.interpolator.predict(theta, phi)
        else:
            raise ValueError(f"Unknown interpolation method: {method}")
        
        return interpolated_values
    
    def interpolate_to_query(self, query_params, source_simulations, attention_weights, field_name):
        """Interpolate field for query parameters using attention-weighted combination"""
        
        if len(source_simulations) == 0:
            raise ValueError("No source simulations provided")
        
        # Get reference mesh from first source simulation
        ref_mesh = source_simulations[0]['mesh_data']
        
        # Initialize arrays for weighted combination
        n_points = len(ref_mesh.points)
        weighted_sum = np.zeros(n_points)
        weight_sum = 0
        
        for sim_data, weight in zip(source_simulations, attention_weights):
            if weight < 1e-6:  # Skip very low weights
                continue
                
            # Get field data from source simulation
            if field_name in sim_data['mesh_data'].fields:
                # For simplicity, take mean over time
                field_data = sim_data['mesh_data'].fields[field_name]
                if field_data.ndim == 3:  # Vector field
                    source_values = np.linalg.norm(field_data.mean(axis=0), axis=1)
                else:  # Scalar field
                    source_values = field_data.mean(axis=0)
                
                # Prepare interpolator for this source
                temp_interpolator = SphericalFieldInterpolator(method=self.method)
                temp_interpolator.prepare_interpolation(sim_data['mesh_data'], source_values)
                
                # Interpolate to reference mesh
                interpolated_values = temp_interpolator.interpolate(ref_mesh)
                
                # Add to weighted sum
                weighted_sum += interpolated_values * weight
                weight_sum += weight
        
        if weight_sum > 0:
            return weighted_sum / weight_sum
        else:
            raise ValueError("No valid source simulations with significant weights")

# =============================================
# ENHANCED ATTENTION-BASED EXTRAPOLATOR
# =============================================
class EnhancedAttentionExtrapolator:
    """Enhanced extrapolator with attention mechanism and spherical geometry support"""
    
    def __init__(self, attention_mechanism=None, interpolator=None):
        self.attention = attention_mechanism or EnhancedPhysicsAwareAttention()
        self.interpolator = interpolator or SphericalFieldInterpolator()
        self.source_simulations = []
        self.source_embeddings = []
        self.source_metadata = []
        
    def load_simulation_data(self, simulations, summaries):
        """Load simulation data and create embeddings"""
        self.source_simulations = []
        self.source_embeddings = []
        self.source_metadata = []
        
        for summary in summaries:
            sim_name = summary['name']
            if sim_name not in simulations:
                continue
                
            sim_data = simulations[sim_name]
            mesh_data = sim_data.get('mesh_data')
            
            if mesh_data is None:
                continue
            
            # Create embedding for each timestep
            for t in range(sim_data['n_timesteps']):
                embedding = self.attention.create_physics_aware_embedding(
                    energy=summary['energy'],
                    duration=summary['duration'],
                    time=t + 1,
                    mesh_data=mesh_data
                )
                
                self.source_embeddings.append(embedding)
                self.source_simulations.append(sim_data)
                self.source_metadata.append({
                    'name': sim_name,
                    'energy': summary['energy'],
                    'duration': summary['duration'],
                    'timestep': t,
                    'mesh_data': mesh_data
                })
        
        if self.source_embeddings:
            self.source_embeddings = np.array(self.source_embeddings)
            st.info(f"✅ Prepared {len(self.source_embeddings)} embeddings")
    
    def predict_field(self, energy_query, duration_query, time_query, 
                     reference_mesh, field_name, n_neighbors=5):
        """Predict field distribution for query parameters"""
        
        # Create query embedding
        query_embedding = self.attention.create_physics_aware_embedding(
            energy=energy_query,
            duration=duration_query,
            time=time_query,
            mesh_data=reference_mesh
        )
        
        # Compute attention weights
        attention_weights = self.attention.compute_similarity(
            query_embedding,
            self.source_embeddings
        )
        
        if len(attention_weights) == 0:
            raise ValueError("No source data available for prediction")
        
        # Get top neighbors
        top_indices = np.argsort(attention_weights)[-n_neighbors:][::-1]
        top_weights = attention_weights[top_indices]
        top_simulations = [self.source_simulations[i] for i in top_indices]
        
        # Normalize weights
        top_weights = top_weights / np.sum(top_weights)
        
        # Update attention history
        self.attention.update_attention_history(
            query_params={'energy': energy_query, 'duration': duration_query, 'time': time_query},
            attention_weights=top_weights,
            source_indices=top_indices
        )
        
        # Interpolate field using attention-weighted combination
        predicted_field = self.interpolator.interpolate_to_query(
            query_params={'energy': energy_query, 'duration': duration_query, 'time': time_query},
            source_simulations=top_simulations,
            attention_weights=top_weights,
            field_name=field_name
        )
        
        result = {
            'predicted_field': predicted_field,
            'attention_weights': top_weights,
            'source_indices': top_indices,
            'source_simulations': [s['name'] for s in top_simulations],
            'confidence': float(np.max(top_weights)),
            'method': self.interpolator.method
        }
        
        return result
    
    def predict_time_series(self, energy_query, duration_query, time_points,
                           reference_mesh, field_name):
        """Predict field evolution over time"""
        
        results = {
            'time_points': time_points,
            'predictions': [],
            'attention_weights': [],
            'confidence_scores': []
        }
        
        for t in time_points:
            try:
                prediction = self.predict_field(
                    energy_query=energy_query,
                    duration_query=duration_query,
                    time_query=t,
                    reference_mesh=reference_mesh,
                    field_name=field_name
                )
                
                results['predictions'].append(prediction['predicted_field'])
                results['attention_weights'].append(prediction['attention_weights'])
                results['confidence_scores'].append(prediction['confidence'])
                
            except Exception as e:
                st.warning(f"Error predicting at time {t}: {str(e)}")
                # Fill with zeros if prediction fails
                results['predictions'].append(np.zeros(len(reference_mesh.points)))
                results['attention_weights'].append(np.array([]))
                results['confidence_scores'].append(0.0)
        
        return results

# =============================================
# SPHERICAL VISUALIZATION ENGINE
# =============================================
class SphericalVisualizationEngine:
    """Visualization engine specialized for spherical geometries"""
    
    @staticmethod
    def create_spherical_visualization(mesh_data, field_values, config):
        """Create spherical visualization with proper geometry"""
        
        fig = go.Figure()
        
        mode = config.get('mode', 'surface')
        colormap = config.get('colormap', 'viridis')
        opacity = config.get('opacity', 0.9)
        show_colorbar = config.get('show_colorbar', True)
        
        points = mesh_data.points
        triangles = mesh_data.triangles
        
        if mode == 'spherical_surface':
            # Create spherical surface plot
            if triangles is not None and len(triangles) > 0:
                fig.add_trace(go.Mesh3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    i=triangles[:, 0],
                    j=triangles[:, 1],
                    k=triangles[:, 2],
                    intensity=field_values,
                    colorscale=colormap,
                    intensitymode='vertex',
                    opacity=opacity,
                    lighting=dict(
                        ambient=0.8,
                        diffuse=0.8,
                        specular=0.5,
                        roughness=0.5
                    ),
                    showscale=show_colorbar,
                    colorbar=dict(
                        title=config.get('colorbar_title', 'Field Value'),
                        thickness=20,
                        len=0.75
                    ),
                    hoverinfo='skip'
                ))
        
        elif mode == 'spherical_points':
            # Create spherical point cloud
            fig.add_trace(go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=field_values,
                    colorscale=colormap,
                    opacity=opacity,
                    showscale=show_colorbar,
                    colorbar=dict(
                        title=config.get('colorbar_title', 'Field Value'),
                        thickness=20,
                        len=0.75
                    )
                ),
                hoverinfo='skip'
            ))
        
        elif mode == 'spherical_wireframe':
            # Create spherical wireframe
            if triangles is not None and len(triangles) > 0:
                edges = set()
                for tri in triangles[:min(1000, len(triangles))]:
                    for i in range(3):
                        edge = tuple(sorted((tri[i], tri[(i+1)%3])))
                        edges.add(edge)
                
                edge_x, edge_y, edge_z = [], [], []
                for edge in list(edges)[:500]:
                    edge_x.extend([points[edge[0], 0], points[edge[1], 0], None])
                    edge_y.extend([points[edge[0], 1], points[edge[1], 1], None])
                    edge_z.extend([points[edge[0], 2], points[edge[1], 2], None])
                
                fig.add_trace(go.Scatter3d(
                    x=edge_x, y=edge_y, z=edge_z,
                    mode='lines',
                    line=dict(
                        color='gray',
                        width=1
                    ),
                    hoverinfo='none'
                ))
        
        # Update layout for spherical view
        fig.update_layout(
            scene=dict(
                aspectmode='data',
                camera=dict(
                    eye=dict(x=2, y=2, z=2),
                    up=dict(x=0, y=0, z=1)
                ),
                xaxis=dict(
                    showgrid=True,
                    gridcolor='lightgray',
                    showbackground=True,
                    backgroundcolor='white'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='lightgray',
                    showbackground=True,
                    backgroundcolor='white'
                ),
                zaxis=dict(
                    showgrid=True,
                    gridcolor='lightgray',
                    showbackground=True,
                    backgroundcolor='white'
                )
            ),
            height=config.get('height', 700),
            margin=dict(l=0, r=0, t=40, b=0),
            title=config.get('title', 'Spherical Field Visualization')
        )
        
        return fig
    
    @staticmethod
    def create_attention_visualization(attention_weights, source_metadata, query_params):
        """Create visualization of attention mechanism"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Attention Weights', 'Parameter Space', 
                          'Source Distribution', 'Weight Evolution'],
            specs=[[{'type': 'bar'}, {'type': 'scatter3d'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        # 1. Attention weights bar chart
        if len(attention_weights) > 0:
            source_names = [f"Source {i}" for i in range(len(attention_weights))]
            fig.add_trace(
                go.Bar(
                    x=source_names,
                    y=attention_weights,
                    name='Attention Weights',
                    marker_color='skyblue'
                ),
                row=1, col=1
            )
        
        # 2. Parameter space 3D scatter
        energies = []
        durations = []
        timesteps = []
        weights = []
        
        for meta, weight in zip(source_metadata, attention_weights):
            energies.append(meta.get('energy', 0))
            durations.append(meta.get('duration', 0))
            timesteps.append(meta.get('timestep', 0))
            weights.append(weight)
        
        if energies and durations and timesteps:
            fig.add_trace(
                go.Scatter3d(
                    x=energies,
                    y=durations,
                    z=timesteps,
                    mode='markers',
                    marker=dict(
                        size=[w * 20 for w in weights],
                        color=weights,
                        colorscale='Viridis',
                        showscale=True
                    ),
                    text=[f"Weight: {w:.3f}" for w in weights],
                    hoverinfo='text'
                ),
                row=1, col=2
            )
            
            # Add query point
            fig.add_trace(
                go.Scatter3d(
                    x=[query_params.get('energy', 0)],
                    y=[query_params.get('duration', 0)],
                    z=[query_params.get('time', 0)],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color='red',
                        symbol='star'
                    ),
                    name='Query Point'
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Attention Mechanism Visualization"
        )
        
        return fig

# =============================================
# ENHANCED DATA LOADER WITH SPHERICAL SUPPORT
# =============================================
class EnhancedSphericalDataLoader:
    """Enhanced data loader with spherical geometry support"""
    
    def __init__(self):
        self.simulations = {}
        self.summaries = []
        self.reference_mesh = None
        self.common_fields = set()
        
    def parse_folder_name(self, folder: str):
        """Parse folder name to extract parameters"""
        name = os.path.basename(folder)
        
        # Try different patterns
        patterns = [
            r"q([\dp\.]+)mJ-delta([\dp\.]+)ns",
            r"E_([\d\.]+)_D_([\d\.]+)",
            r"energy_([\d\.]+)_duration_([\d\.]+)",
            r"([\d\.]+)mJ_([\d\.]+)ns"
        ]
        
        for pattern in patterns:
            match = re.match(pattern, name)
            if match:
                e, d = match.groups()
                e = float(e.replace("p", ".")) if "p" in e else float(e)
                d = float(d.replace("p", ".")) if "p" in d else float(d)
                return e, d
        
        return None, None
    
    def load_vtu_file(self, filepath):
        """Load VTU file with enhanced error handling"""
        try:
            mesh = meshio.read(filepath)
            return mesh
        except Exception as e:
            st.warning(f"Error reading {filepath}: {str(e)}")
            return None
    
    def extract_mesh_data(self, mesh):
        """Extract mesh data with spherical geometry features"""
        if mesh is None:
            return None, None, {}
        
        points = mesh.points.astype(np.float32)
        
        # Find triangles
        triangles = None
        for cell_block in mesh.cells:
            if cell_block.type == "triangle":
                triangles = cell_block.data.astype(np.int32)
                break
        
        # Extract point data
        point_data = {}
        for key, data in mesh.point_data.items():
            point_data[key] = data.astype(np.float32)
        
        # Create enhanced mesh data
        enhanced_mesh = EnhancedMeshData()
        enhanced_mesh.points = points
        enhanced_mesh.triangles = triangles
        
        # Compute spherical features
        enhanced_mesh.compute_spherical_coordinates()
        enhanced_mesh.compute_normals()
        enhanced_mesh.compute_curvature()
        
        return enhanced_mesh, point_data
    
    @st.cache_resource(ttl=3600, show_spinner=True)
    def load_all_simulations(_self, load_full_mesh=True):
        """Load all simulations with spherical geometry support"""
        simulations = {}
        summaries = []
        
        # Find simulation folders
        folders = []
        for pattern in ["q*mJ-delta*ns", "E_*_D_*", "energy_*_duration_*", "*mJ_*ns"]:
            folders.extend(glob.glob(os.path.join(FEA_SOLUTIONS_DIR, pattern)))
        
        folders = list(set(folders))
        
        if not folders:
            st.warning(f"No simulation folders found in {FEA_SOLUTIONS_DIR}")
            return simulations, summaries
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for folder_idx, folder in enumerate(folders):
            folder_name = os.path.basename(folder)
            energy, duration = _self.parse_folder_name(folder_name)
            
            if energy is None or duration is None:
                continue
            
            # Find VTU files
            vtu_files = sorted(glob.glob(os.path.join(folder, "*.vtu")))
            if not vtu_files:
                vtu_files = sorted(glob.glob(os.path.join(folder, "*.vtk")))
            
            if not vtu_files:
                continue
            
            status_text.text(f"Loading {folder_name}... ({len(vtu_files)} files)")
            
            try:
                # Load first file to get mesh structure
                mesh0 = _self.load_vtu_file(vtu_files[0])
                if mesh0 is None:
                    continue
                
                # Extract mesh data
                mesh_data, point_data = _self.extract_mesh_data(mesh0)
                if mesh_data is None:
                    continue
                
                # Create simulation data structure
                sim_data = {
                    'name': folder_name,
                    'energy_mJ': energy,
                    'duration_ns': duration,
                    'n_timesteps': len(vtu_files),
                    'vtu_files': vtu_files,
                    'field_info': {},
                    'mesh_data': mesh_data,
                    'has_mesh': True,
                    'is_interpolated': False
                }
                
                # Initialize field arrays
                n_points = len(mesh_data.points)
                for key, data in point_data.items():
                    if data.ndim == 1:
                        sim_data['field_info'][key] = ("scalar", 1)
                        mesh_data.fields[key] = np.full((len(vtu_files), n_points), np.nan, dtype=np.float32)
                        mesh_data.fields[key][0] = data
                    elif data.ndim == 2:
                        dims = data.shape[1]
                        sim_data['field_info'][key] = ("vector", dims)
                        mesh_data.fields[key] = np.full((len(vtu_files), n_points, dims), np.nan, dtype=np.float32)
                        mesh_data.fields[key][0] = data
                
                # Load remaining timesteps
                for t in range(1, len(vtu_files)):
                    try:
                        mesh = _self.load_vtu_file(vtu_files[t])
                        if mesh is None:
                            continue
                        
                        _, point_data_t = _self.extract_mesh_data(mesh)
                        
                        for key in sim_data['field_info']:
                            if key in point_data_t:
                                mesh_data.fields[key][t] = point_data_t[key]
                    except Exception as e:
                        st.warning(f"Error loading timestep {t} in {folder_name}: {e}")
                
                simulations[folder_name] = sim_data
                
                # Create summary
                summary = _self.extract_summary_statistics(sim_data, energy, duration, folder_name)
                summaries.append(summary)
                
                # Set as reference mesh if not already set
                if _self.reference_mesh is None:
                    _self.reference_mesh = mesh_data
                
            except Exception as e:
                st.error(f"Error processing {folder_name}: {str(e)}")
                continue
            
            progress_bar.progress((folder_idx + 1) / len(folders))
        
        progress_bar.empty()
        status_text.empty()
        
        if simulations:
            st.success(f"✅ Loaded {len(simulations)} simulations")
            
            # Determine common fields
            if simulations:
                field_counts = {}
                for sim in simulations.values():
                    for field in sim['field_info'].keys():
                        field_counts[field] = field_counts.get(field, 0) + 1
                
                _self.common_fields = {field for field, count in field_counts.items() 
                                     if count == len(simulations)}
        
        return simulations, summaries
    
    def extract_summary_statistics(self, sim_data, energy, duration, name):
        """Extract summary statistics with spherical features"""
        summary = {
            'name': name,
            'energy': energy,
            'duration': duration,
            'mesh_stats': sim_data['mesh_data'].metadata,
            'field_stats': {},
            'timesteps': list(range(1, sim_data['n_timesteps'] + 1))
        }
        
        mesh_data = sim_data['mesh_data']
        
        for field_name in sim_data['field_info'].keys():
            if field_name not in mesh_data.fields:
                continue
            
            field_data = mesh_data.fields[field_name]
            summary['field_stats'][field_name] = {
                'min': [], 'max': [], 'mean': [], 'std': [],
                'q25': [], 'q50': [], 'q75': []
            }
            
            for t in range(field_data.shape[0]):
                if field_data[t].ndim == 1:
                    values = field_data[t]
                elif field_data[t].ndim == 2:
                    values = np.linalg.norm(field_data[t], axis=1)
                else:
                    continue
                
                valid_values = values[~np.isnan(values)]
                
                if len(valid_values) > 0:
                    summary['field_stats'][field_name]['min'].append(float(np.min(valid_values)))
                    summary['field_stats'][field_name]['max'].append(float(np.max(valid_values)))
                    summary['field_stats'][field_name]['mean'].append(float(np.mean(valid_values)))
                    summary['field_stats'][field_name]['std'].append(float(np.std(valid_values)))
                    summary['field_stats'][field_name]['q25'].append(float(np.percentile(valid_values, 25)))
                    summary['field_stats'][field_name]['q50'].append(float(np.percentile(valid_values, 50)))
                    summary['field_stats'][field_name]['q75'].append(float(np.percentile(valid_values, 75)))
                else:
                    for stat in ['min', 'max', 'mean', 'std', 'q25', 'q50', 'q75']:
                        summary['field_stats'][field_name][stat].append(0.0)
        
        return summary

# =============================================
# INTERPOLATED SOLUTIONS MANAGER (ENHANCED)
# =============================================
class EnhancedInterpolatedSolutionsManager:
    """Manager for interpolated solutions with spherical geometry preservation"""
    
    def __init__(self):
        self.interpolated_simulations = {}
        self.interpolated_summaries = []
        
    def create_interpolated_solution(self, sim_name, field_name, predicted_field, 
                                   reference_mesh, query_params, method, confidence):
        """Create a new interpolated solution with spherical geometry"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        interpolated_id = f"interp_{sim_name}_{field_name}_{method}_{timestamp}"
        
        # Create enhanced mesh data for interpolated solution
        mesh_data = EnhancedMeshData()
        mesh_data.points = reference_mesh.points.copy()
        mesh_data.triangles = reference_mesh.triangles.copy() if reference_mesh.triangles is not None else None
        
        # Compute spherical features
        mesh_data.compute_spherical_coordinates()
        mesh_data.compute_normals()
        
        # Store predicted field
        mesh_data.fields = {field_name: predicted_field.reshape(1, -1)}
        
        # Create interpolated simulation data
        interpolated_sim = {
            'name': interpolated_id,
            'original_simulation': sim_name,
            'field': field_name,
            'mesh_data': mesh_data,
            'query_params': query_params,
            'interpolation_method': method,
            'confidence': confidence,
            'energy_mJ': query_params.get('energy', 0),
            'duration_ns': query_params.get('duration', 0),
            'has_mesh': True,
            'is_interpolated': True,
            'created_at': timestamp,
            'n_timesteps': 1
        }
        
        # Store in session state
        self.interpolated_simulations[interpolated_id] = interpolated_sim
        
        # Create summary
        summary = self.create_interpolated_summary(interpolated_sim)
        self.interpolated_summaries.append(summary)
        
        # Save to disk
        self.save_interpolated_solution(interpolated_sim)
        
        return interpolated_sim
    
    def create_interpolated_summary(self, interpolated_sim):
        """Create summary for interpolated solution"""
        mesh_data = interpolated_sim['mesh_data']
        field_name = interpolated_sim['field']
        
        if field_name in mesh_data.fields:
            field_data = mesh_data.fields[field_name][0]
            valid_values = field_data[~np.isnan(field_data)]
            
            if len(valid_values) > 0:
                percentiles = np.percentile(valid_values, [10, 25, 50, 75, 90])
            else:
                percentiles = np.zeros(5)
        else:
            valid_values = np.array([])
            percentiles = np.zeros(5)
        
        summary = {
            'name': interpolated_sim['name'],
            'energy': interpolated_sim['energy_mJ'],
            'duration': interpolated_sim['duration_ns'],
            'original_simulation': interpolated_sim['original_simulation'],
            'interpolation_method': interpolated_sim['interpolation_method'],
            'is_interpolated': True,
            'confidence': interpolated_sim['confidence'],
            'mesh_stats': mesh_data.metadata,
            'field_stats': {
                field_name: {
                    'min': [float(np.min(valid_values))] if len(valid_values) > 0 else [0.0],
                    'max': [float(np.max(valid_values))] if len(valid_values) > 0 else [0.0],
                    'mean': [float(np.mean(valid_values))] if len(valid_values) > 0 else [0.0],
                    'std': [float(np.std(valid_values))] if len(valid_values) > 0 else [0.0],
                    'q25': [percentiles[1]] if len(percentiles) > 0 else [0.0],
                    'q50': [percentiles[2]] if len(percentiles) > 0 else [0.0],
                    'q75': [percentiles[3]] if len(percentiles) > 0 else [0.0]
                }
            },
            'timesteps': [1]
        }
        
        return summary
    
    def save_interpolated_solution(self, interpolated_sim):
        """Save interpolated solution to disk"""
        save_path = os.path.join(INTERPOLATED_SOLUTIONS_DIR, f"{interpolated_sim['name']}.pkl")
        
        # Prepare data for saving
        save_data = {
            'name': interpolated_sim['name'],
            'original_simulation': interpolated_sim['original_simulation'],
            'field': interpolated_sim['field'],
            'points': interpolated_sim['mesh_data'].points,
            'triangles': interpolated_sim['mesh_data'].triangles,
            'field_data': interpolated_sim['mesh_data'].fields[interpolated_sim['field']][0],
            'query_params': interpolated_sim['query_params'],
            'interpolation_method': interpolated_sim['interpolation_method'],
            'confidence': interpolated_sim['confidence'],
            'energy_mJ': interpolated_sim['energy_mJ'],
            'duration_ns': interpolated_sim['duration_ns'],
            'created_at': interpolated_sim['created_at'],
            'metadata': interpolated_sim['mesh_data'].metadata
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
    
    def load_interpolated_solutions(self):
        """Load interpolated solutions from disk"""
        pkl_files = glob.glob(os.path.join(INTERPOLATED_SOLUTIONS_DIR, "interp_*.pkl"))
        
        self.interpolated_simulations = {}
        self.interpolated_summaries = []
        
        for pkl_file in pkl_files:
            try:
                with open(pkl_file, 'rb') as f:
                    save_data = pickle.load(f)
                
                # Reconstruct mesh data
                mesh_data = EnhancedMeshData()
                mesh_data.points = save_data['points']
                mesh_data.triangles = save_data['triangles']
                mesh_data.metadata = save_data.get('metadata', {})
                mesh_data.fields = {save_data['field']: save_data['field_data'].reshape(1, -1)}
                
                # Recompute spherical features
                mesh_data.compute_spherical_coordinates()
                mesh_data.compute_normals()
                
                # Create interpolated simulation
                interpolated_sim = {
                    'name': save_data['name'],
                    'original_simulation': save_data['original_simulation'],
                    'field': save_data['field'],
                    'mesh_data': mesh_data,
                    'query_params': save_data['query_params'],
                    'interpolation_method': save_data['interpolation_method'],
                    'confidence': save_data['confidence'],
                    'energy_mJ': save_data['energy_mJ'],
                    'duration_ns': save_data['duration_ns'],
                    'has_mesh': True,
                    'is_interpolated': True,
                    'created_at': save_data['created_at'],
                    'n_timesteps': 1
                }
                
                self.interpolated_simulations[save_data['name']] = interpolated_sim
                
                # Create summary
                summary = self.create_interpolated_summary(interpolated_sim)
                self.interpolated_summaries.append(summary)
                
            except Exception as e:
                st.warning(f"Error loading interpolated solution {pkl_file}: {e}")
        
        return len(self.interpolated_simulations)
    
    def get_combined_simulations(self):
        """Get combined original and interpolated simulations"""
        combined = {}
        
        # Add original simulations from session state
        if 'simulations' in st.session_state:
            combined.update(st.session_state.simulations)
        
        # Add interpolated simulations
        combined.update(self.interpolated_simulations)
        
        return combined
    
    def get_combined_summaries(self):
        """Get combined original and interpolated summaries"""
        combined = []
        
        # Add original summaries
        if 'summaries' in st.session_state:
            combined.extend(st.session_state.summaries)
        
        # Add interpolated summaries
        combined.extend(self.interpolated_summaries)
        
        return combined

# =============================================
# MAIN APPLICATION WITH SPHERICAL INTERPOLATION
# =============================================
class SphericalInterpolationApp:
    """Main application with spherical interpolation support"""
    
    def __init__(self):
        self.data_loader = EnhancedSphericalDataLoader()
        self.interp_manager = EnhancedInterpolatedSolutionsManager()
        self.visualizer = SphericalVisualizationEngine()
        self.extrapolator = None
        
        # Initialize session state
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize session state variables"""
        defaults = {
            'data_loaded': False,
            'current_mode': 'Data Explorer',
            'selected_colormap': 'viridis',
            'visualization_mode': 'spherical_surface',
            'mesh_opacity': 0.9,
            'interpolation_method': 'spherical_rbf',
            'n_attention_heads': 4,
            'spatial_weight': 0.3,
            'temperature': 1.0,
            'simulations': {},
            'summaries': []
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def run(self):
        """Run the application"""
        st.set_page_config(
            page_title="Spherical FEA Interpolation Platform",
            layout="wide",
            initial_sidebar_state="expanded",
            page_icon="🔮"
        )
        
        # Apply custom CSS
        self.apply_custom_css()
        
        # Render header
        self.render_header()
        
        # Render sidebar
        self.render_sidebar()
        
        # Render main content
        self.render_main_content()
    
    def apply_custom_css(self):
        """Apply custom CSS styling"""
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            background: linear-gradient(90deg, #1E88E5, #4A00E0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 1rem;
            font-weight: 800;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #2c3e50;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            padding-bottom: 0.3rem;
            border-bottom: 2px solid #3498db;
            font-weight: 600;
        }
        .info-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .warning-box {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .metric-card {
            background: white;
            padding: 0.8rem;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin: 0.3rem;
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render_header(self):
        """Render application header"""
        st.markdown('<h1 class="main-header">🔮 Spherical FEA Interpolation Platform</h1>', unsafe_allow_html=True)
        st.markdown("""
        <p style="text-align: center; color: #666; margin-bottom: 2rem;">
        Physics-Aware Attention Mechanism with Spherical Geometry Preservation
        </p>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar"""
        with st.sidebar:
            st.markdown("### 🧭 Navigation")
            
            app_mode = st.selectbox(
                "Select Mode",
                ["Data Explorer", "Spherical Interpolation", "3D Visualization", "Attention Analysis"],
                index=0
            )
            
            st.session_state.current_mode = app_mode
            
            st.markdown("---")
            st.markdown("### 📂 Data Management")
            
            if st.button("🚀 Load Simulations", use_container_width=True):
                self.load_simulations()
            
            if st.session_state.get('data_loaded', False):
                simulations = st.session_state.simulations
                st.success(f"✅ {len(simulations)} simulations loaded")
                
                st.markdown("---")
                st.markdown("### ⚙️ Settings")
                
                with st.expander("Visualization Settings", expanded=False):
                    st.session_state.visualization_mode = st.selectbox(
                        "Visualization Mode",
                        ["spherical_surface", "spherical_points", "spherical_wireframe"]
                    )
                    
                    st.session_state.selected_colormap = st.selectbox(
                        "Colormap",
                        ["viridis", "plasma", "inferno", "magma", "cividis", "rainbow"]
                    )
                    
                    st.session_state.mesh_opacity = st.slider("Opacity", 0.1, 1.0, 0.9, 0.1)
                
                with st.expander("Interpolation Settings", expanded=False):
                    st.session_state.interpolation_method = st.selectbox(
                        "Interpolation Method",
                        ["spherical_rbf", "spherical_harmonics", "barycentric"]
                    )
                    
                    st.session_state.n_attention_heads = st.slider("Attention Heads", 1, 8, 4)
                    st.session_state.spatial_weight = st.slider("Spatial Weight", 0.0, 1.0, 0.3, 0.1)
                    st.session_state.temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
    
    def load_simulations(self):
        """Load simulation data"""
        with st.spinner("Loading simulations..."):
            try:
                simulations, summaries = self.data_loader.load_all_simulations(load_full_mesh=True)
                
                if simulations:
                    st.session_state.simulations = simulations
                    st.session_state.summaries = summaries
                    st.session_state.data_loaded = True
                    
                    # Initialize extrapolator
                    self.extrapolator = EnhancedAttentionExtrapolator(
                        attention_mechanism=EnhancedPhysicsAwareAttention(
                            n_heads=st.session_state.n_attention_heads,
                            temperature=st.session_state.temperature,
                            spatial_weight=st.session_state.spatial_weight
                        ),
                        interpolator=SphericalFieldInterpolator(
                            method=st.session_state.interpolation_method
                        )
                    )
                    
                    # Load simulation data into extrapolator
                    self.extrapolator.load_simulation_data(simulations, summaries)
                    
                    st.session_state.extrapolator = self.extrapolator
                    
                    # Load interpolated solutions
                    num_interpolated = self.interp_manager.load_interpolated_solutions()
                    if num_interpolated > 0:
                        st.info(f"📊 Loaded {num_interpolated} interpolated solutions")
                    
                else:
                    st.error("❌ No simulations loaded. Check data directory.")
                    
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
    
    def render_main_content(self):
        """Render main content based on selected mode"""
        mode = st.session_state.current_mode
        
        if mode == "Data Explorer":
            self.render_data_explorer()
        elif mode == "Spherical Interpolation":
            self.render_spherical_interpolation()
        elif mode == "3D Visualization":
            self.render_3d_visualization()
        elif mode == "Attention Analysis":
            self.render_attention_analysis()
    
    def render_data_explorer(self):
        """Render data explorer"""
        st.markdown('<h2 class="sub-header">📊 Data Explorer</h2>', unsafe_allow_html=True)
        
        if not st.session_state.get('data_loaded', False):
            self.show_data_not_loaded()
            return
        
        simulations = st.session_state.simulations
        
        # Simulation selector
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            sim_name = st.selectbox(
                "Select Simulation",
                sorted(simulations.keys()),
                key="explorer_sim"
            )
        
        sim = simulations[sim_name]
        
        with col2:
            st.metric("Energy", f"{sim['energy_mJ']:.2f} mJ")
        with col3:
            st.metric("Duration", f"{sim['duration_ns']:.2f} ns")
        
        # Check if spherical
        mesh_data = sim['mesh_data']
        if hasattr(mesh_data, 'is_spherical') and mesh_data.is_spherical:
            st.info(f"🌐 Spherical geometry detected (radius: {mesh_data.metadata.get('avg_radius', 0):.3f})")
        
        # Field selection
        if sim['field_info']:
            field = st.selectbox(
                "Select Field",
                sorted(sim['field_info'].keys()),
                key="explorer_field"
            )
            
            # Timestep selection
            n_timesteps = sim['n_timesteps']
            timestep = st.slider(
                "Timestep",
                0, max(0, n_timesteps - 1), 0,
                key="explorer_timestep"
            )
            
            # Get field values
            if field in mesh_data.fields:
                field_data = mesh_data.fields[field][timestep]
                
                if sim['field_info'][field][0] == "vector":
                    field_data = np.linalg.norm(field_data, axis=1)
                
                # Create visualization
                config = {
                    'mode': st.session_state.visualization_mode,
                    'colormap': st.session_state.selected_colormap,
                    'opacity': st.session_state.mesh_opacity,
                    'title': f"{field} at Timestep {timestep + 1}",
                    'height': 600
                }
                
                fig = self.visualizer.create_spherical_visualization(mesh_data, field_data, config)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show field statistics
                st.markdown("#### 📈 Field Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Min", f"{np.min(field_data):.3f}")
                with col2:
                    st.metric("Max", f"{np.max(field_data):.3f}")
                with col3:
                    st.metric("Mean", f"{np.mean(field_data):.3f}")
                with col4:
                    st.metric("Std", f"{np.std(field_data):.3f}")
        else:
            st.warning("No field data available for this simulation")
    
    def render_spherical_interpolation(self):
        """Render spherical interpolation interface"""
        st.markdown('<h2 class="sub-header">🔮 Spherical Interpolation</h2>', unsafe_allow_html=True)
        
        if not st.session_state.get('data_loaded', False):
            self.show_data_not_loaded()
            return
        
        st.markdown("""
        <div class="info-box">
        <h4>🌐 Spherical Attention Interpolation</h4>
        <p>This system uses physics-aware attention mechanisms to interpolate and extrapolate 
        field distributions on spherical geometries. The interpolation preserves the spherical 
        geometry while using attention weights to combine information from similar simulations.</p>
        </div>
        """, unsafe_allow_html=True)
        
        simulations = st.session_state.simulations
        
        if not simulations:
            st.warning("No simulations loaded")
            return
        
        # Query parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            energy_query = st.number_input(
                "Energy (mJ)",
                min_value=0.1,
                max_value=100.0,
                value=5.25,
                step=0.01,
                key="interp_energy"
            )
        
        with col2:
            duration_query = st.number_input(
                "Pulse Duration (ns)",
                min_value=0.1,
                max_value=50.0,
                value=2.9,
                step=0.01,
                key="interp_duration"
            )
        
        with col3:
            time_query = st.number_input(
                "Time (ns)",
                min_value=0.0,
                max_value=100.0,
                value=10.0,
                step=0.1,
                key="interp_time"
            )
        
        # Reference simulation selection
        ref_sim_name = st.selectbox(
            "Reference Geometry",
            sorted(simulations.keys()),
            key="interp_ref_sim"
        )
        
        ref_sim = simulations[ref_sim_name]
        ref_mesh = ref_sim['mesh_data']
        
        # Field selection
        if ref_sim['field_info']:
            field_name = st.selectbox(
                "Field to Predict",
                sorted(ref_sim['field_info'].keys()),
                key="interp_field"
            )
        else:
            st.warning("No fields available in reference simulation")
            return
        
        # Prediction options
        col1, col2 = st.columns(2)
        with col1:
            n_neighbors = st.slider("Number of Neighbors", 1, 10, 5)
        
        with col2:
            prediction_method = st.selectbox(
                "Prediction Method",
                ["spherical_rbf", "spherical_harmonics", "barycentric"],
                index=0
            )
        
        if st.button("🚀 Generate Prediction", use_container_width=True):
            with st.spinner("Generating physics-aware prediction..."):
                try:
                    # Get extrapolator from session state
                    extrapolator = st.session_state.get('extrapolator')
                    if extrapolator is None:
                        st.error("Extrapolator not initialized. Please reload simulations.")
                        return
                    
                    # Update interpolator method
                    extrapolator.interpolator.method = prediction_method
                    
                    # Generate prediction
                    prediction = extrapolator.predict_field(
                        energy_query=energy_query,
                        duration_query=duration_query,
                        time_query=time_query,
                        reference_mesh=ref_mesh,
                        field_name=field_name,
                        n_neighbors=n_neighbors
                    )
                    
                    # Create interpolated solution
                    interpolated_sim = self.interp_manager.create_interpolated_solution(
                        sim_name=ref_sim_name,
                        field_name=field_name,
                        predicted_field=prediction['predicted_field'],
                        reference_mesh=ref_mesh,
                        query_params={
                            'energy': energy_query,
                            'duration': duration_query,
                            'time': time_query
                        },
                        method=prediction_method,
                        confidence=prediction['confidence']
                    )
                    
                    st.success(f"✅ Prediction generated with confidence: {prediction['confidence']:.3f}")
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Show prediction visualization
                        config = {
                            'mode': 'spherical_surface',
                            'colormap': st.session_state.selected_colormap,
                            'opacity': st.session_state.mesh_opacity,
                            'title': f"Predicted {field_name} - E={energy_query}mJ, τ={duration_query}ns",
                            'height': 500
                        }
                        
                        fig = self.visualizer.create_spherical_visualization(
                            ref_mesh,
                            prediction['predicted_field'],
                            config
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Show attention weights
                        st.markdown("#### 🧠 Attention Weights")
                        source_names = prediction['source_simulations']
                        weights = prediction['attention_weights']
                        
                        for name, weight in zip(source_names, weights):
                            st.progress(float(weight), text=f"{name}: {weight:.3f}")
                        
                        # Show statistics
                        st.markdown("#### 📊 Prediction Statistics")
                        pred_values = prediction['predicted_field']
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Min", f"{np.min(pred_values):.3f}")
                        with col2:
                            st.metric("Max", f"{np.max(pred_values):.3f}")
                        with col3:
                            st.metric("Mean", f"{np.mean(pred_values):.3f}")
                        with col4:
                            st.metric("Std", f"{np.std(pred_values):.3f}")
                    
                    # Show attention visualization
                    st.markdown("#### 🔍 Attention Mechanism Analysis")
                    attention_fig = self.visualizer.create_attention_visualization(
                        prediction['attention_weights'],
                        [extrapolator.source_metadata[i] for i in prediction['source_indices']],
                        {'energy': energy_query, 'duration': duration_query, 'time': time_query}
                    )
                    st.plotly_chart(attention_fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
                    st.error(traceback.format_exc())
    
    def render_3d_visualization(self):
        """Render 3D visualization interface"""
        st.markdown('<h2 class="sub-header">🌐 3D Visualization</h2>', unsafe_allow_html=True)
        
        # Get combined simulations
        combined_simulations = self.interp_manager.get_combined_simulations()
        
        if not combined_simulations:
            self.show_data_not_loaded()
            return
        
        # Simulation selection
        sim_names = sorted(combined_simulations.keys())
        display_names = []
        for name in sim_names:
            sim = combined_simulations[name]
            if sim.get('is_interpolated', False):
                display_names.append(f"{name} 🌟")
            else:
                display_names.append(name)
        
        selected_display = st.selectbox(
            "Select Simulation",
            display_names,
            key="viz_sim_select"
        )
        
        # Extract actual name
        if ' 🌟' in selected_display:
            sim_name = selected_display.split(' 🌟')[0]
            is_interpolated = True
        else:
            sim_name = selected_display
            is_interpolated = False
        
        sim = combined_simulations[sim_name]
        
        # Show simulation info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Energy", f"{sim['energy_mJ']:.2f} mJ")
        with col2:
            st.metric("Duration", f"{sim['duration_ns']:.2f} ns")
        with col3:
            if is_interpolated:
                st.metric("Type", "Interpolated")
            else:
                st.metric("Type", "Original")
        
        if is_interpolated:
            st.info(f"Interpolation method: {sim.get('interpolation_method', 'Unknown')} | Confidence: {sim.get('confidence', 0):.3f}")
        
        # Field and visualization selection
        mesh_data = sim['mesh_data']
        
        if hasattr(sim, 'field_info') and sim['field_info']:
            field = st.selectbox(
                "Select Field",
                sorted(sim['field_info'].keys()),
                key="viz_field"
            )
            
            # For interpolated simulations, only one timestep
            if sim.get('n_timesteps', 1) > 1:
                timestep = st.slider(
                    "Timestep",
                    0, sim['n_timesteps'] - 1, 0,
                    key="viz_timestep"
                )
            else:
                timestep = 0
            
            # Get field values
            if field in mesh_data.fields:
                field_data = mesh_data.fields[field][timestep]
                
                # Handle vector fields
                if hasattr(sim, 'field_info') and sim['field_info'][field][0] == "vector":
                    field_data = np.linalg.norm(field_data, axis=1)
                
                # Create visualization
                config = {
                    'mode': st.session_state.visualization_mode,
                    'colormap': st.session_state.selected_colormap,
                    'opacity': st.session_state.mesh_opacity,
                    'title': f"{field} - {sim_name}",
                    'height': 700
                }
                
                fig = self.visualizer.create_spherical_visualization(mesh_data, field_data, config)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show mesh statistics
                st.markdown("#### 📐 Mesh Statistics")
                if hasattr(mesh_data, 'metadata'):
                    metadata = mesh_data.metadata
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Points", f"{len(mesh_data.points):,}")
                    with col2:
                        if mesh_data.triangles is not None:
                            st.metric("Triangles", f"{len(mesh_data.triangles):,}")
                        else:
                            st.metric("Triangles", "N/A")
                    with col3:
                        if 'avg_radius' in metadata:
                            st.metric("Avg Radius", f"{metadata['avg_radius']:.3f}")
                    with col4:
                        if 'is_spherical' in metadata:
                            st.metric("Spherical", "Yes" if metadata['is_spherical'] else "No")
            else:
                st.warning(f"Field '{field}' not available in this simulation")
        else:
            st.warning("No field data available for this simulation")
    
    def render_attention_analysis(self):
        """Render attention mechanism analysis"""
        st.markdown('<h2 class="sub-header">🧠 Attention Mechanism Analysis</h2>', unsafe_allow_html=True)
        
        if not st.session_state.get('data_loaded', False):
            self.show_data_not_loaded()
            return
        
        extrapolator = st.session_state.get('extrapolator')
        if extrapolator is None:
            st.warning("Extrapolator not initialized. Please generate predictions first.")
            return
        
        attention = extrapolator.attention
        
        if not hasattr(attention, 'attention_history') or len(attention.attention_history) == 0:
            st.info("No attention history available. Generate predictions first.")
            return
        
        # Show attention history
        st.markdown("#### 📜 Attention History")
        
        history_df = []
        for entry in attention.attention_history[-10:]:  # Show last 10 entries
            history_df.append({
                'Timestamp': entry['timestamp'],
                'Energy': entry['query']['energy'],
                'Duration': entry['query']['duration'],
                'Time': entry['query']['time'],
                'Max Weight': np.max(entry['weights']),
                'Avg Weight': np.mean(entry['weights'])
            })
        
        if history_df:
            st.dataframe(pd.DataFrame(history_df), use_container_width=True)
        
        # Parameter space analysis
        st.markdown("#### 🌐 Parameter Space Analysis")
        
        # Get all query points from history
        query_points = []
        query_weights = []
        
        for entry in attention.attention_history:
            query_points.append([
                entry['query']['energy'],
                entry['query']['duration'],
                entry['query']['time']
            ])
            query_weights.append(np.max(entry['weights']))
        
        if query_points:
            query_points = np.array(query_points)
            query_weights = np.array(query_weights)
            
            # Create 3D scatter plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter3d(
                x=query_points[:, 0],
                y=query_points[:, 1],
                z=query_points[:, 2],
                mode='markers',
                marker=dict(
                    size=10,
                    color=query_weights,
                    colorscale='Viridis',
                    opacity=0.8,
                    showscale=True,
                    colorbar=dict(title="Max Attention Weight")
                ),
                text=[f"E: {e:.1f}, τ: {d:.1f}, t: {t:.1f}" for e, d, t in query_points],
                hoverinfo='text'
            ))
            
            fig.update_layout(
                title="Query Points in Parameter Space",
                scene=dict(
                    xaxis_title="Energy (mJ)",
                    yaxis_title="Duration (ns)",
                    zaxis_title="Time (ns)"
                ),
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Source distribution analysis
        st.markdown("#### 📊 Source Distribution")
        
        if hasattr(extrapolator, 'source_metadata') and extrapolator.source_metadata:
            source_energies = []
            source_durations = []
            
            for meta in extrapolator.source_metadata:
                source_energies.append(meta.get('energy', 0))
                source_durations.append(meta.get('duration', 0))
            
            fig = px.scatter(
                x=source_energies,
                y=source_durations,
                title="Source Simulations in Parameter Space",
                labels={'x': 'Energy (mJ)', 'y': 'Duration (ns)'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def show_data_not_loaded(self):
        """Show data not loaded message"""
        st.markdown("""
        <div class="warning-box">
        <h3>⚠️ No Data Loaded</h3>
        <p>Please load simulations first using the "Load Simulations" button in the sidebar.</p>
        <p>Ensure your data follows the expected structure with spherical VTU files.</p>
        </div>
        """, unsafe_allow_html=True)

# =============================================
# MAIN APPLICATION ENTRY POINT
# =============================================
def main():
    """Main entry point"""
    try:
        app = SphericalInterpolationApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.error(traceback.format_exc())

if __name__ == "__main__":
    main()
