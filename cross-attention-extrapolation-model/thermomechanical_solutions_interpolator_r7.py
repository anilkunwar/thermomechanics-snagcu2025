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
from scipy.spatial import KDTree, cKDTree, Delaunay
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

warnings.filterwarnings('ignore')

# =============================================
# DATA CLASSES AND ENUMS
# =============================================
class FieldType(Enum):
    SCALAR = "scalar"
    VECTOR = "vector"
    TENSOR = "tensor"

@dataclass
class FieldInfo:
    name: str
    field_type: FieldType
    dims: int
    units: str = ""
    description: str = ""
    physical_range: Optional[Tuple[float, float]] = None
    is_positive_definite: bool = False

@dataclass
class MeshMetadata:
    num_points: int
    num_cells: int
    num_triangles: int
    num_tetrahedra: int
    bbox: Dict[str, List[float]]
    centroid: List[float]
    volume: Optional[float] = None
    surface_area: Optional[float] = None
    avg_edge_length: Optional[float] = None
    min_angle: Optional[float] = None
    max_angle: Optional[float] = None

# =============================================
# CACHING DECORATORS
# =============================================
def cache_to_session_state(func):
    """Cache expensive computations to session state"""
    def wrapper(*args, **kwargs):
        # Create cache key from function name and arguments
        arg_str = str(args) + str(kwargs)
        cache_key = f"{func.__name__}_{hashlib.md5(arg_str.encode()).hexdigest()[:16]}"
        
        if cache_key in st.session_state:
            return st.session_state[cache_key]
        
        result = func(*args, **kwargs)
        st.session_state[cache_key] = result
        return result
    return wrapper

# =============================================
# ENHANCED MESH DATA STRUCTURES WITH OPTIMIZATION
# =============================================
class EnhancedMeshData:
    """Container for enhanced mesh data with spatial indexing and topological features"""
    
    def __init__(self):
        self.points = None
        self.triangles = None
        self.cells = None
        self.fields = {}
        self.field_info = {}
        self.metadata = MeshMetadata(0, 0, 0, 0, {"min": [0,0,0], "max": [0,0,0]}, [0,0,0])
        self.spatial_tree = None
        self.region_labels = None
        self.mesh_stats = {}
        self.connectivity_graph = None
        self.normals = None
        self.curvature = None
        self._cached_areas = None
        self._cached_edges = None
        
    def build_spatial_index(self):
        """Build KDTree for spatial queries"""
        if self.points is not None and self.spatial_tree is None:
            self.spatial_tree = cKDTree(self.points)
        return self.spatial_tree
    
    def compute_mesh_features(self):
        """Compute comprehensive mesh features"""
        if self.points is None:
            return
        
        # Basic statistics
        min_coords = np.min(self.points, axis=0)
        max_coords = np.max(self.points, axis=0)
        self.metadata.bbox = {'min': min_coords.tolist(), 'max': max_coords.tolist()}
        self.metadata.centroid = np.mean(self.points, axis=0).tolist()
        self.metadata.num_points = len(self.points)
        
        # Compute distances from centroid
        if self.points.shape[0] > 0:
            centroid = self.metadata.centroid
            distances = np.linalg.norm(self.points - centroid, axis=1)
            self.mesh_stats['radial_stats'] = {
                'min': float(np.min(distances)),
                'max': float(np.max(distances)),
                'mean': float(np.mean(distances)),
                'std': float(np.std(distances))
            }
        
        # Compute topological features
        if self.triangles is not None and len(self.triangles) > 0:
            self.metadata.num_triangles = len(self.triangles)
            areas = self.compute_triangle_areas()
            self.metadata.surface_area = float(np.sum(areas))
            self.mesh_stats['triangle_areas'] = {
                'min': float(np.min(areas)),
                'max': float(np.max(areas)),
                'mean': float(np.mean(areas)),
                'std': float(np.std(areas))
            }
            
            # Compute triangle quality metrics
            qualities = self.compute_triangle_quality(areas)
            self.mesh_stats['triangle_quality'] = qualities
            
            # Compute normals
            self.compute_normals()
            
            # Compute curvature
            self.compute_curvature()
        
        # Build connectivity graph
        self.build_connectivity_graph()
    
    @cache_to_session_state
    def compute_triangle_areas(self):
        """Compute areas of triangles with caching"""
        if self.triangles is None or self.points is None:
            return np.array([])
        
        if self._cached_areas is not None:
            return self._cached_areas
        
        areas = []
        for tri in self.triangles:
            if tri[0] < len(self.points) and tri[1] < len(self.points) and tri[2] < len(self.points):
                v0 = self.points[tri[0]]
                v1 = self.points[tri[1]]
                v2 = self.points[tri[2]]
                
                # Compute area using cross product
                v0v1 = v1 - v0
                v0v2 = v2 - v0
                area = 0.5 * np.linalg.norm(np.cross(v0v1, v0v2))
                areas.append(area)
        
        self._cached_areas = np.array(areas)
        return self._cached_areas
    
    def compute_triangle_quality(self, areas=None):
        """Compute triangle quality metrics"""
        if areas is None:
            areas = self.compute_triangle_areas()
        
        if len(areas) == 0:
            return {}
        
        # Compute edge lengths
        edge_lengths = []
        for tri in self.triangles:
            if tri[0] < len(self.points) and tri[1] < len(self.points) and tri[2] < len(self.points):
                v0, v1, v2 = self.points[tri[0]], self.points[tri[1]], self.points[tri[2]]
                edges = [
                    np.linalg.norm(v1 - v0),
                    np.linalg.norm(v2 - v1),
                    np.linalg.norm(v0 - v2)
                ]
                edge_lengths.extend(edges)
        
        edge_lengths = np.array(edge_lengths)
        
        return {
            'aspect_ratio': float(np.max(areas) / np.min(areas)) if np.min(areas) > 0 else 0,
            'edge_length_stats': {
                'min': float(np.min(edge_lengths)),
                'max': float(np.max(edge_lengths)),
                'mean': float(np.mean(edge_lengths)),
                'std': float(np.std(edge_lengths))
            }
        }
    
    def compute_normals(self):
        """Compute vertex normals"""
        if self.triangles is None or self.points is None:
            return
        
        normals = np.zeros_like(self.points)
        
        # Compute face normals
        face_normals = np.zeros((len(self.triangles), 3))
        for i, tri in enumerate(self.triangles):
            v0, v1, v2 = self.points[tri[0]], self.points[tri[1]], self.points[tri[2]]
            face_normals[i] = np.cross(v1 - v0, v2 - v0)
            face_normals[i] /= (np.linalg.norm(face_normals[i]) + 1e-10)
        
        # Accumulate to vertices
        for i, tri in enumerate(self.triangles):
            normals[tri[0]] += face_normals[i]
            normals[tri[1]] += face_normals[i]
            normals[tri[2]] += face_normals[i]
        
        # Normalize
        norms = np.linalg.norm(normals, axis=1)
        norms[norms == 0] = 1
        self.normals = normals / norms[:, np.newaxis]
    
    def compute_curvature(self):
        """Compute mean curvature at vertices"""
        if self.normals is None or self.triangles is None:
            self.compute_normals()
        
        if self.normals is None:
            return
        
        # Simple curvature approximation using Laplacian of normals
        curvatures = np.zeros(len(self.points))
        
        # Build adjacency
        adjacency = [[] for _ in range(len(self.points))]
        for tri in self.triangles:
            for i in range(3):
                for j in range(3):
                    if i != j:
                        adjacency[tri[i]].append(tri[j])
        
        # Compute discrete mean curvature
        for i in range(len(self.points)):
            if adjacency[i]:
                neighbors = list(set(adjacency[i]))
                if neighbors:
                    # Weighted average of normal differences
                    diff_sum = 0
                    for j in neighbors:
                        diff_sum += np.linalg.norm(self.normals[i] - self.normals[j])
                    curvatures[i] = diff_sum / len(neighbors)
        
        self.curvature = curvatures
    
    def build_connectivity_graph(self):
        """Build graph representation of mesh connectivity"""
        if self.triangles is None:
            return
        
        self.connectivity_graph = nx.Graph()
        
        # Add nodes
        for i in range(len(self.points)):
            self.connectivity_graph.add_node(i, pos=self.points[i])
        
        # Add edges from triangles
        edges_set = set()
        for tri in self.triangles:
            for i in range(3):
                edge = tuple(sorted((tri[i], tri[(i+1)%3])))
                if edge not in edges_set:
                    v1, v2 = self.points[edge[0]], self.points[edge[1]]
                    weight = np.linalg.norm(v1 - v2)
                    self.connectivity_graph.add_edge(edge[0], edge[1], weight=weight)
                    edges_set.add(edge)
        
        self.mesh_stats['graph_stats'] = {
            'num_edges': len(edges_set),
            'avg_degree': np.mean([d for _, d in self.connectivity_graph.degree()]),
            'connected_components': nx.number_connected_components(self.connectivity_graph)
        }
    
    def get_edge_list(self):
        """Get list of edges from connectivity graph"""
        if self._cached_edges is None and self.connectivity_graph is not None:
            edges = []
            for edge in self.connectivity_graph.edges():
                edges.append([self.points[edge[0]], self.points[edge[1]]])
            self._cached_edges = np.array(edges)
        return self._cached_edges
    
    def segment_regions(self, n_regions=5, method='kmeans'):
        """Segment mesh into spatial regions using different methods"""
        if self.points is None:
            return
        
        if method == 'kmeans':
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_regions, random_state=42, n_init=10)
            self.region_labels = kmeans.fit_predict(self.points)
        
        elif method == 'spectral':
            from sklearn.cluster import SpectralClustering
            # Use adjacency matrix for spectral clustering
            if self.connectivity_graph is not None:
                adj_matrix = nx.adjacency_matrix(self.connectivity_graph)
                spectral = SpectralClustering(n_clusters=n_regions, 
                                            affinity='precomputed',
                                            random_state=42)
                self.region_labels = spectral.fit_predict(adj_matrix)
            else:
                # Fallback to KMeans
                self.segment_regions(n_regions, method='kmeans')
                return
        
        elif method == 'dbscan':
            from sklearn.cluster import DBSCAN
            dbscan = DBSCAN(eps=0.1, min_samples=5)
            self.region_labels = dbscan.fit_predict(self.points)
        
        # Compute region statistics
        self.region_stats = {}
        for region_id in np.unique(self.region_labels):
            if region_id >= 0:  # DBSCAN might have -1 for noise
                region_mask = self.region_labels == region_id
                if np.any(region_mask):
                    region_points = self.points[region_mask]
                    self.region_stats[region_id] = {
                        'size': int(np.sum(region_mask)),
                        'centroid': np.mean(region_points, axis=0).tolist(),
                        'bbox': {
                            'min': np.min(region_points, axis=0).tolist(),
                            'max': np.max(region_points, axis=0).tolist()
                        },
                        'avg_curvature': float(np.mean(self.curvature[region_mask])) if self.curvature is not None else 0
                    }
    
    def resample_to_grid(self, resolution=50):
        """Resample mesh to regular grid for visualization"""
        if self.points is None:
            return None
        
        # Create grid
        min_coords = np.min(self.points, axis=0)
        max_coords = np.max(self.points, axis=0)
        
        # For 3D, we might want 2D slices or 3D grid
        # Here we create a 2D grid in XY plane at mean Z
        mean_z = np.mean(self.points[:, 2])
        x = np.linspace(min_coords[0], max_coords[0], resolution)
        y = np.linspace(min_coords[1], max_coords[1], resolution)
        X, Y = np.meshgrid(x, y)
        
        # Points at mean Z
        grid_points = np.vstack([X.ravel(), Y.ravel(), np.ones(X.ravel().shape) * mean_z]).T
        
        return grid_points, X, Y, mean_z

# =============================================
# ADVANCED FIELD PREDICTOR WITH MACHINE LEARNING
# =============================================
class PhysicsInformedPredictor:
    """Physics-informed field predictor with multiple interpolation methods"""
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.methods = {
            'rbf': self._predict_rbf,
            'kriging': self._predict_kriging,
            'neural': self._predict_neural,
            'physics': self._predict_physics_informed
        }
        self.trained_models = {}
        
    def predict_field(self, energy, duration, time, mesh_data, field_name, 
                     method='rbf', use_physics_constraints=True):
        """Predict field using specified method"""
        if method not in self.methods:
            method = 'rbf'
        
        # Get similar simulations
        similar_sims = self._find_similar_simulations(energy, duration, time)
        
        if not similar_sims:
            return self._predict_synthetic(energy, duration, time, mesh_data, field_name)
        
        # Use selected prediction method
        predictor_func = self.methods[method]
        prediction = predictor_func(energy, duration, time, mesh_data, 
                                   field_name, similar_sims)
        
        # Apply physics constraints if needed
        if use_physics_constraints:
            prediction = self._apply_physics_constraints(prediction, field_name, mesh_data)
        
        return prediction
    
    def _find_similar_simulations(self, energy, duration, time, n_neighbors=5):
        """Find most similar simulations based on parameters"""
        similarities = []
        
        for sim_name, sim_data in self.data_loader.simulations.items():
            # Compute parameter similarity
            energy_sim = sim_data['energy_mJ']
            duration_sim = sim_data['duration_ns']
            
            # Normalize differences
            energy_diff = abs(energy - energy_sim) / max(energy, energy_sim)
            duration_diff = abs(duration - duration_sim) / max(duration, duration_sim)
            
            # Time similarity (find closest timestep)
            if 'field_info' in sim_data and sim_data['field_info']:
                n_timesteps = sim_data['n_timesteps']
                time_idx = min(max(int(time), 0), n_timesteps - 1)
                time_sim = time_idx / n_timesteps  # Normalized time
                time_query = time / max(time, 1.0)
                time_diff = abs(time_query - time_sim)
            else:
                time_diff = 0
            
            # Combined similarity score
            similarity = 1.0 / (1.0 + energy_diff + duration_diff + time_diff)
            similarities.append((similarity, sim_name, sim_data))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        return similarities[:n_neighbors]
    
    def _predict_rbf(self, energy, duration, time, mesh_data, field_name, similar_sims):
        """Predict using Radial Basis Function interpolation"""
        # Collect source data
        source_points = []
        source_values = []
        
        for similarity, sim_name, sim_data in similar_sims:
            if field_name in sim_data['mesh_data'].fields:
                field_data = sim_data['mesh_data'].fields[field_name]
                
                # Find appropriate timestep
                n_timesteps = field_data.shape[0]
                time_idx = min(max(int(time), 0), n_timesteps - 1)
                
                values = field_data[time_idx]
                if values.ndim == 2:
                    values = np.linalg.norm(values, axis=1)
                
                # Use a subset of points for efficiency
                if len(values) > 1000:
                    indices = np.random.choice(len(values), 1000, replace=False)
                    values = values[indices]
                    points = sim_data['mesh_data'].points[indices]
                else:
                    points = sim_data['mesh_data'].points
                
                source_points.append(points)
                source_values.append(values)
        
        if not source_points:
            return self._predict_synthetic(energy, duration, time, mesh_data, field_name)
        
        # Combine source data
        all_points = np.vstack(source_points)
        all_values = np.concatenate(source_values)
        
        # Handle different mesh geometries using barycentric coordinates
        target_points = mesh_data.points
        
        # Build KDTree for source points
        tree = cKDTree(all_points)
        
        # Find nearest neighbors and interpolate
        distances, indices = tree.query(target_points, k=4)
        
        # Inverse distance weighting
        weights = 1.0 / (distances + 1e-10)
        weights = weights / np.sum(weights, axis=1, keepdims=True)
        
        predicted = np.sum(all_values[indices] * weights, axis=1)
        
        return {
            'values': predicted,
            'method': 'rbf_idw',
            'confidence': float(np.mean(similar_sims[0][0])),
            'similarity_scores': [s[0] for s in similar_sims]
        }
    
    def _predict_kriging(self, energy, duration, time, mesh_data, field_name, similar_sims):
        """Predict using Gaussian Process/Kriging"""
        # Simplified kriging implementation
        # In production, use GaussianProcessRegressor from sklearn
        
        # For now, use RBF with uncertainty estimation
        rbf_pred = self._predict_rbf(energy, duration, time, mesh_data, field_name, similar_sims)
        
        # Estimate uncertainty based on similarity variance
        similarities = np.array([s[0] for s in similar_sims])
        uncertainty = 1.0 - np.mean(similarities)
        
        rbf_pred['uncertainty'] = float(uncertainty)
        rbf_pred['method'] = 'kriging_approx'
        
        return rbf_pred
    
    def _predict_neural(self, energy, duration, time, mesh_data, field_name, similar_sims):
        """Predict using neural network (simplified)"""
        # This would typically use a trained neural network
        # For demonstration, we use a weighted combination of similar fields
        
        predictions = []
        weights = []
        
        for similarity, sim_name, sim_data in similar_sims:
            if field_name in sim_data['mesh_data'].fields:
                field_data = sim_data['mesh_data'].fields[field_name]
                n_timesteps = field_data.shape[0]
                time_idx = min(max(int(time), 0), n_timesteps - 1)
                
                values = field_data[time_idx]
                if values.ndim == 2:
                    values = np.linalg.norm(values, axis=1)
                
                # Resample to target mesh if needed
                if len(values) != len(mesh_data.points):
                    # Use nearest neighbor resampling
                    tree = cKDTree(sim_data['mesh_data'].points)
                    distances, indices = tree.query(mesh_data.points, k=1)
                    values = values[indices]
                
                predictions.append(values)
                weights.append(similarity)
        
        if not predictions:
            return self._predict_synthetic(energy, duration, time, mesh_data, field_name)
        
        # Weighted average
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        weighted_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, weights):
            weighted_pred += pred * weight
        
        return {
            'values': weighted_pred,
            'method': 'neural_weighted',
            'confidence': float(np.mean(weights)),
            'weights': weights.tolist()
        }
    
    def _predict_physics_informed(self, energy, duration, time, mesh_data, field_name, similar_sims):
        """Predict using physics-informed constraints"""
        # Start with neural prediction
        base_pred = self._predict_neural(energy, duration, time, mesh_data, field_name, similar_sims)
        
        # Apply physics-based corrections
        values = base_pred['values']
        
        # Example physics constraints:
        # 1. Smoothness constraint (Laplacian smoothing)
        if mesh_data.connectivity_graph is not None:
            laplacian = nx.laplacian_matrix(mesh_data.connectivity_graph)
            laplacian = laplacian.toarray()
            smoothed = values - 0.1 * laplacian.dot(values)
            values = smoothed
        
        # 2. Boundary conditions (if known)
        # Could enforce zero values at boundaries, etc.
        
        base_pred['values'] = values
        base_pred['method'] = 'physics_informed'
        
        return base_pred
    
    def _predict_synthetic(self, energy, duration, time, mesh_data, field_name):
        """Create synthetic prediction when no similar simulations exist"""
        points = mesh_data.points
        centroid = np.mean(points, axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        distances_norm = distances / (np.max(distances) + 1e-10)
        
        # Physics-inspired synthetic field
        if field_name.lower() in ['temperature', 'heat']:
            # Gaussian heat distribution
            pattern = np.exp(-(distances_norm**2) / 0.2) * energy
            pattern = pattern * (1 - np.exp(-time / max(duration, 1.0)))
        
        elif field_name.lower() in ['stress', 'pressure']:
            # Radial stress pattern
            pattern = energy * (1 - distances_norm) / (time + 1.0)
        
        elif field_name.lower() in ['displacement']:
            # Sinusoidal displacement pattern
            pattern = energy * np.sin(distances_norm * np.pi) * np.exp(-time / duration)
        
        else:
            # Generic pattern
            pattern = energy * np.exp(-distances_norm) * np.sin(time)
        
        # Add noise for realism
        noise = np.random.normal(0, 0.1 * energy, len(points))
        pattern = pattern + noise
        
        return {
            'values': pattern,
            'method': 'synthetic_physics',
            'confidence': 0.3,
            'warning': 'No similar simulations found'
        }
    
    def _apply_physics_constraints(self, prediction, field_name, mesh_data):
        """Apply physical constraints to prediction"""
        values = prediction['values']
        
        # Non-negativity for certain fields
        if field_name.lower() in ['temperature', 'stress', 'pressure', 'density']:
            values = np.maximum(values, 0)
        
        # Maximum limits (could be material-specific)
        if field_name.lower() == 'temperature':
            # Assume maximum temperature limit (e.g., melting point)
            values = np.minimum(values, 2000)  # Example limit
        
        # Smoothing for physical consistency
        if mesh_data.connectivity_graph is not None and len(values) > 0:
            # Apply Laplacian smoothing
            adj_matrix = nx.adjacency_matrix(mesh_data.connectivity_graph)
            degree_matrix = sparse.diags([d for _, d in mesh_data.connectivity_graph.degree()])
            laplacian = degree_matrix - adj_matrix
            
            # Convert to dense for small matrices, sparse for large
            if len(values) < 10000:
                laplacian = laplacian.toarray()
                values = values - 0.05 * laplacian.dot(values)
            else:
                # Use sparse solver for large matrices
                values = values - 0.05 * laplacian.dot(values)
        
        prediction['values'] = values
        prediction['physics_constrained'] = True
        
        return prediction

# =============================================
# ADVANCED VISUALIZATION ENGINE
# =============================================
class AdvancedVisualizationEngine:
    """Advanced visualization with multiple rendering modes and optimizations"""
    
    VISUALIZATION_MODES = {
        'surface': "Surface Mesh",
        'wireframe': "Wireframe",
        'points': "Point Cloud",
        'volume': "Volume Render",
        'streamlines': "Streamlines",
        'slices': "Cross Sections"
    }
    
    LIGHTING_PRESETS = {
        'default': {'ambient': 0.8, 'diffuse': 0.8, 'specular': 0.5, 'roughness': 0.5},
        'bright': {'ambient': 1.0, 'diffuse': 1.0, 'specular': 0.8, 'roughness': 0.3},
        'soft': {'ambient': 0.9, 'diffuse': 0.6, 'specular': 0.2, 'roughness': 0.7},
        'dramatic': {'ambient': 0.5, 'diffuse': 0.9, 'specular': 0.9, 'roughness': 0.2}
    }
    
    @staticmethod
    def create_advanced_mesh_visualization(mesh_data, field_values, config):
        """Create advanced mesh visualization with multiple options"""
        fig = go.Figure()
        
        mode = config.get('mode', 'surface')
        colormap = config.get('colormap', 'Viridis')
        opacity = config.get('opacity', 0.9)
        show_colorbar = config.get('show_colorbar', True)
        lighting = config.get('lighting', 'default')
        
        pts = mesh_data.points
        triangles = mesh_data.triangles
        
        if mode == 'surface':
            fig = AdvancedVisualizationEngine._create_surface_visualization(
                pts, triangles, field_values, colormap, opacity, lighting
            )
        
        elif mode == 'wireframe':
            fig = AdvancedVisualizationEngine._create_wireframe_visualization(
                pts, triangles, field_values, colormap, opacity
            )
        
        elif mode == 'points':
            fig = AdvancedVisualizationEngine._create_point_cloud_visualization(
                pts, field_values, colormap, opacity
            )
        
        elif mode == 'volume':
            fig = AdvancedVisualizationEngine._create_volume_visualization(
                mesh_data, field_values, colormap
            )
        
        # Add annotations and enhancements
        if config.get('show_annotations', True):
            fig = AdvancedVisualizationEngine._add_annotations(fig, mesh_data, field_values)
        
        if config.get('show_statistics', True):
            fig = AdvancedVisualizationEngine._add_statistics_overlay(fig, field_values)
        
        # Update layout
        fig.update_layout(
            scene=dict(
                aspectmode="data",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    up=dict(x=0, y=0, z=1)
                ),
                xaxis=dict(showbackground=False, visible=False),
                yaxis=dict(showbackground=False, visible=False),
                zaxis=dict(showbackground=False, visible=False)
            ),
            height=config.get('height', 700),
            margin=dict(l=0, r=0, t=30, b=0),
            showlegend=config.get('show_legend', False)
        )
        
        return fig
    
    @staticmethod
    def _create_surface_visualization(pts, triangles, field_values, colormap, opacity, lighting_preset):
        """Create surface mesh visualization"""
        fig = go.Figure()
        
        lighting = AdvancedVisualizationEngine.LIGHTING_PRESETS.get(lighting_preset, 
                                                                   AdvancedVisualizationEngine.LIGHTING_PRESETS['default'])
        
        if triangles is not None and len(triangles) > 0:
            # Use Mesh3d for surface
            fig.add_trace(go.Mesh3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
                intensity=field_values,
                colorscale=colormap,
                intensitymode='vertex',
                opacity=opacity,
                lighting=lighting,
                flatshading=False,
                showscale=True,
                hoverinfo='skip',
                name="Field Surface"
            ))
        
        return fig
    
    @staticmethod
    def _create_wireframe_visualization(pts, triangles, field_values, colormap, opacity):
        """Create wireframe visualization"""
        fig = go.Figure()
        
        if triangles is not None and len(triangles) > 0:
            # Create edges from triangles
            edges = set()
            for tri in triangles[:min(2000, len(triangles))]:  # Limit for performance
                for i in range(3):
                    edge = tuple(sorted((tri[i], tri[(i+1)%3])))
                    edges.add(edge)
            
            # Create line segments for edges
            edge_x, edge_y, edge_z = [], [], []
            for edge in list(edges)[:1000]:  # Limit edges
                if edge[0] < len(pts) and edge[1] < len(pts):
                    edge_x.extend([pts[edge[0], 0], pts[edge[1], 0], None])
                    edge_y.extend([pts[edge[0], 1], pts[edge[1], 1], None])
                    edge_z.extend([pts[edge[0], 2], pts[edge[1], 2], None])
            
            # Color edges by average field value at endpoints
            edge_colors = []
            for edge in list(edges)[:1000]:
                if edge[0] < len(field_values) and edge[1] < len(field_values):
                    avg_value = (field_values[edge[0]] + field_values[edge[1]]) / 2
                    edge_colors.extend([avg_value, avg_value, None])
            
            fig.add_trace(go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=dict(
                    width=1,
                    color=edge_colors,
                    colorscale=colormap
                ),
                opacity=opacity,
                hoverinfo='none',
                name="Wireframe"
            ))
        
        return fig
    
    @staticmethod
    def _create_point_cloud_visualization(pts, field_values, colormap, opacity):
        """Create point cloud visualization"""
        fig = go.Figure()
        
        # Use sampling for large point clouds
        if len(pts) > 10000:
            indices = np.random.choice(len(pts), 10000, replace=False)
            pts = pts[indices]
            field_values = field_values[indices]
        
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=field_values,
                colorscale=colormap,
                opacity=opacity,
                showscale=True
            ),
            hoverinfo='skip',
            name="Point Cloud"
        ))
        
        return fig
    
    @staticmethod
    def _create_volume_visualization(mesh_data, field_values, colormap):
        """Create volume rendering using slices"""
        fig = go.Figure()
        
        # Create slices along different axes
        pts = mesh_data.points
        min_coords = np.min(pts, axis=0)
        max_coords = np.max(pts, axis=0)
        
        # Create 3 orthogonal slices
        slice_positions = [
            (min_coords[0] + max_coords[0]) / 2,  # X slice
            (min_coords[1] + max_coords[1]) / 2,  # Y slice
            (min_coords[2] + max_coords[2]) / 2   # Z slice
        ]
        
        for i, (axis, pos) in enumerate(zip(['X', 'Y', 'Z'], slice_positions)):
            # Create slice
            if axis == 'X':
                mask = np.abs(pts[:, 0] - pos) < (max_coords[0] - min_coords[0]) * 0.01
                x, y, z = pos * np.ones_like(pts[mask, 1]), pts[mask, 1], pts[mask, 2]
            elif axis == 'Y':
                mask = np.abs(pts[:, 1] - pos) < (max_coords[1] - min_coords[1]) * 0.01
                x, y, z = pts[mask, 0], pos * np.ones_like(pts[mask, 0]), pts[mask, 2]
            else:  # Z
                mask = np.abs(pts[:, 2] - pos) < (max_coords[2] - min_coords[2]) * 0.01
                x, y, z = pts[mask, 0], pts[mask, 1], pos * np.ones_like(pts[mask, 0])
            
            slice_values = field_values[mask]
            
            if len(slice_values) > 0:
                fig.add_trace(go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=slice_values,
                        colorscale=colormap,
                        opacity=0.6
                    ),
                    name=f"{axis}-Slice",
                    hoverinfo='skip'
                ))
        
        return fig
    
    @staticmethod
    def _add_annotations(fig, mesh_data, field_values):
        """Add annotations to visualization"""
        if mesh_data.points is None or len(mesh_data.points) == 0:
            return fig
        
        # Add centroid marker
        centroid = np.mean(mesh_data.points, axis=0)
        centroid_value = field_values[np.argmin(np.linalg.norm(mesh_data.points - centroid, axis=1))]
        
        fig.add_trace(go.Scatter3d(
            x=[centroid[0]], y=[centroid[1]], z=[centroid[2]],
            mode='markers+text',
            marker=dict(size=8, color='red', symbol='diamond'),
            text=[f"Centroid: {centroid_value:.3f}"],
            textposition="top center",
            name="Centroid"
        ))
        
        # Add min and max value markers
        min_idx = np.argmin(field_values)
        max_idx = np.argmax(field_values)
        
        for idx, label, color in [(min_idx, "Min", "blue"), (max_idx, "Max", "yellow")]:
            fig.add_trace(go.Scatter3d(
                x=[mesh_data.points[idx, 0]],
                y=[mesh_data.points[idx, 1]],
                z=[mesh_data.points[idx, 2]],
                mode='markers+text',
                marker=dict(size=8, color=color, symbol='circle'),
                text=[f"{label}: {field_values[idx]:.3f}"],
                textposition="top center",
                name=label
            ))
        
        return fig
    
    @staticmethod
    def _add_statistics_overlay(fig, field_values):
        """Add statistics as annotation"""
        stats_text = (
            f"Mean: {np.mean(field_values):.3f}<br>"
            f"Std: {np.std(field_values):.3f}<br>"
            f"Min: {np.min(field_values):.3f}<br>"
            f"Max: {np.max(field_values):.3f}<br>"
            f"Range: {np.max(field_values) - np.min(field_values):.3f}"
        )
        
        fig.add_annotation(
            x=0.02, y=0.98,
            xref="paper", yref="paper",
            text=stats_text,
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=10)
        )
        
        return fig
    
    @staticmethod
    def create_comparison_dashboard(mesh_data_list, field_values_list, names, config):
        """Create dashboard for comparing multiple simulations"""
        n_plots = len(mesh_data_list)
        
        if n_plots == 0:
            return go.Figure()
        
        # Determine grid layout
        if n_plots <= 2:
            cols = n_plots
            rows = 1
        elif n_plots <= 4:
            cols = 2
            rows = (n_plots + 1) // 2
        else:
            cols = 3
            rows = (n_plots + 2) // 3
        
        fig = make_subplots(
            rows=rows, cols=cols,
            specs=[[{'type': 'surface'} for _ in range(cols)] for _ in range(rows)],
            subplot_titles=names,
            vertical_spacing=0.1,
            horizontal_spacing=0.05
        )
        
        for idx, (mesh_data, field_values, name) in enumerate(zip(mesh_data_list, field_values_list, names)):
            row = idx // cols + 1
            col = idx % cols + 1
            
            pts = mesh_data.points
            triangles = mesh_data.triangles
            
            if triangles is not None and len(triangles) > 0:
                fig.add_trace(
                    go.Mesh3d(
                        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                        i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
                        intensity=field_values,
                        colorscale=config.get('colormap', 'Viridis'),
                        intensitymode='vertex',
                        opacity=config.get('opacity', 0.8),
                        showscale=(idx == 0)  # Only show colorbar on first plot
                    ),
                    row=row, col=col
                )
        
        # Update layout
        fig.update_layout(
            height=300 * rows,
            showlegend=False,
            title_text="Simulation Comparison Dashboard"
        )
        
        # Update scene properties for each subplot
        for i in range(1, n_plots + 1):
            fig.update_scenes(
                aspectmode="data",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                row=(i-1)//cols + 1, col=(i-1)%cols + 1
            )
        
        return fig
    
    @staticmethod
    def create_time_evolution_plot(field_evolution, time_points, config):
        """Create time evolution plot of field statistics"""
        fig = go.Figure()
        
        # Compute statistics over time
        means = [np.nanmean(f) if f is not None else np.nan for f in field_evolution]
        maxs = [np.nanmax(f) if f is not None else np.nan for f in field_evolution]
        mins = [np.nanmin(f) if f is not None else np.nan for f in field_evolution]
        stds = [np.nanstd(f) if f is not None else np.nan for f in field_evolution]
        
        # Create traces
        fig.add_trace(go.Scatter(
            x=time_points, y=means,
            mode='lines+markers',
            name='Mean',
            line=dict(width=3, color='blue'),
            fillcolor='rgba(0, 100, 255, 0.2)',
            fill='tozeroy'
        ))
        
        fig.add_trace(go.Scatter(
            x=time_points, y=maxs,
            mode='lines',
            name='Maximum',
            line=dict(width=1, color='red', dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=time_points, y=mins,
            mode='lines',
            name='Minimum',
            line=dict(width=1, color='green', dash='dash')
        ))
        
        # Add confidence interval (mean ± std)
        upper_bound = [m + s for m, s in zip(means, stds)]
        lower_bound = [m - s for m, s in zip(means, stds)]
        
        fig.add_trace(go.Scatter(
            x=time_points + time_points[::-1],
            y=upper_bound + lower_bound[::-1],
            fill='toself',
            fillcolor='rgba(0, 100, 255, 0.1)',
            line=dict(color='rgba(255, 255, 255, 0)'),
            showlegend=False,
            name='Mean ± Std'
        ))
        
        fig.update_layout(
            title=config.get('title', 'Field Evolution Over Time'),
            xaxis_title='Time',
            yaxis_title='Field Value',
            hovermode='x unified',
            height=400,
            showlegend=True
        )
        
        return fig

# =============================================
# ENHANCED DATA ANALYTICS ENGINE
# =============================================
class DataAnalyticsEngine:
    """Engine for advanced data analytics on simulation data"""
    
    @staticmethod
    def perform_pca_analysis(simulations, field_name, n_components=3):
        """Perform PCA on field data across simulations"""
        # Collect field data from all simulations
        all_data = []
        sim_names = []
        
        for sim_name, sim_data in simulations.items():
            if field_name in sim_data['mesh_data'].fields:
                field_data = sim_data['mesh_data'].fields[field_name]
                
                # Take mean over timesteps for each point
                if field_data.ndim == 3:  # Vector field over time
                    field_data = np.linalg.norm(field_data, axis=2)
                
                # Mean over time
                if field_data.ndim == 2:
                    field_data = np.mean(field_data, axis=0)
                
                all_data.append(field_data)
                sim_names.append(sim_name)
        
        if not all_data:
            return None
        
        # Ensure all arrays have same length (resample if needed)
        min_length = min(len(d) for d in all_data)
        all_data_resampled = []
        
        for data in all_data:
            if len(data) > min_length:
                # Randomly sample to min_length
                indices = np.random.choice(len(data), min_length, replace=False)
                all_data_resampled.append(data[indices])
            else:
                all_data_resampled.append(data)
        
        X = np.vstack(all_data_resampled)
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        
        # Create DataFrame for visualization
        df_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
        df_pca['Simulation'] = sim_names
        
        # Add simulation parameters
        energies = []
        durations = []
        
        for sim_name in sim_names:
            sim = simulations[sim_name]
            energies.append(sim['energy_mJ'])
            durations.append(sim['duration_ns'])
        
        df_pca['Energy'] = energies
        df_pca['Duration'] = durations
        
        return {
            'pca_data': df_pca,
            'explained_variance': pca.explained_variance_ratio_,
            'components': pca.components_,
            'mean': pca.mean_
        }
    
    @staticmethod
    def perform_cluster_analysis(simulations, field_name, n_clusters=3):
        """Perform clustering analysis on simulation data"""
        # Extract feature vectors (statistics for each simulation)
        features = []
        sim_names = []
        
        for sim_name, sim_data in simulations.items():
            if field_name in sim_data['mesh_data'].fields:
                field_data = sim_data['mesh_data'].fields[field_name]
                
                # Compute statistics
                if field_data.ndim == 3:
                    field_data = np.linalg.norm(field_data, axis=2)
                
                stats_vector = [
                    np.mean(field_data),
                    np.std(field_data),
                    np.max(field_data),
                    np.min(field_data),
                    np.median(field_data),
                    stats.skew(field_data.flatten()),
                    stats.kurtosis(field_data.flatten())
                ]
                
                # Add simulation parameters
                stats_vector.extend([
                    sim_data['energy_mJ'],
                    sim_data['duration_ns'],
                    sim_data['n_timesteps']
                ])
                
                features.append(stats_vector)
                sim_names.append(sim_name)
        
        if not features:
            return None
        
        X = np.array(features)
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # Perform PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create results
        results = {
            'cluster_labels': labels,
            'cluster_centers': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_,
            'pca_data': X_pca,
            'simulation_names': sim_names,
            'feature_names': ['mean', 'std', 'max', 'min', 'median', 'skew', 'kurtosis', 
                            'energy', 'duration', 'timesteps']
        }
        
        return results
    
    @staticmethod
    def compute_correlation_matrix(simulations, field_names):
        """Compute correlation matrix between fields across simulations"""
        # Collect statistics for each field and simulation
        all_stats = {}
        
        for field_name in field_names:
            field_stats = []
            
            for sim_name, sim_data in simulations.items():
                if field_name in sim_data['mesh_data'].fields:
                    field_data = sim_data['mesh_data'].fields[field_name]
                    
                    if field_data.ndim == 3:
                        field_data = np.linalg.norm(field_data, axis=2)
                    
                    # Compute mean statistic
                    field_stats.append(np.mean(field_data))
            
            if field_stats:
                all_stats[field_name] = field_stats
        
        if not all_stats:
            return None
        
        # Create DataFrame
        df = pd.DataFrame(all_stats)
        
        # Compute correlation matrix
        corr_matrix = df.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Field Correlation Matrix",
            height=500,
            xaxis_title="Fields",
            yaxis_title="Fields"
        )
        
        return {
            'correlation_matrix': corr_matrix,
            'heatmap_figure': fig,
            'data_frame': df
        }
    
    @staticmethod
    def analyze_sensitivity(simulations, field_name):
        """Analyze sensitivity of field to simulation parameters"""
        # Collect data
        energies = []
        durations = []
        field_means = []
        field_maxs = []
        
        for sim_name, sim_data in simulations.items():
            if field_name in sim_data['mesh_data'].fields:
                field_data = sim_data['mesh_data'].fields[field_name]
                
                if field_data.ndim == 3:
                    field_data = np.linalg.norm(field_data, axis=2)
                
                energies.append(sim_data['energy_mJ'])
                durations.append(sim_data['duration_ns'])
                field_means.append(np.mean(field_data))
                field_maxs.append(np.max(field_data))
        
        if not energies:
            return None
        
        # Create DataFrame
        df = pd.DataFrame({
            'energy': energies,
            'duration': durations,
            'mean': field_means,
            'max': field_maxs
        })
        
        # Compute correlations
        corr_with_energy_mean = df['energy'].corr(df['mean'])
        corr_with_duration_mean = df['duration'].corr(df['mean'])
        corr_with_energy_max = df['energy'].corr(df['max'])
        corr_with_duration_max = df['duration'].corr(df['max'])
        
        # Create scatter plots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Mean vs Energy', 'Mean vs Duration', 
                          'Max vs Energy', 'Max vs Duration']
        )
        
        # Mean vs Energy
        fig.add_trace(
            go.Scatter(x=df['energy'], y=df['mean'], mode='markers',
                      name='Mean vs Energy'),
            row=1, col=1
        )
        
        # Mean vs Duration
        fig.add_trace(
            go.Scatter(x=df['duration'], y=df['mean'], mode='markers',
                      name='Mean vs Duration'),
            row=1, col=2
        )
        
        # Max vs Energy
        fig.add_trace(
            go.Scatter(x=df['energy'], y=df['max'], mode='markers',
                      name='Max vs Energy'),
            row=2, col=1
        )
        
        # Max vs Duration
        fig.add_trace(
            go.Scatter(x=df['duration'], y=df['max'], mode='markers',
                      name='Max vs Duration'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False, title_text=f"Sensitivity Analysis: {field_name}")
        fig.update_xaxes(title_text="Energy (mJ)", row=1, col=1)
        fig.update_xaxes(title_text="Duration (ns)", row=1, col=2)
        fig.update_xaxes(title_text="Energy (mJ)", row=2, col=1)
        fig.update_xaxes(title_text="Duration (ns)", row=2, col=2)
        fig.update_yaxes(title_text="Mean Value", row=1, col=1)
        fig.update_yaxes(title_text="Mean Value", row=1, col=2)
        fig.update_yaxes(title_text="Max Value", row=2, col=1)
        fig.update_yaxes(title_text="Max Value", row=2, col=2)
        
        return {
            'data': df,
            'correlations': {
                'energy_mean': corr_with_energy_mean,
                'duration_mean': corr_with_duration_mean,
                'energy_max': corr_with_energy_max,
                'duration_max': corr_with_duration_max
            },
            'figure': fig
        }

# =============================================
# ENHANCED APPLICATION WITH ALL FEATURES
# =============================================
class EnhancedFEAPlatform:
    """Enhanced FEA Platform with all features integrated"""
    
    def __init__(self):
        self.data_loader = EnhancedFEADataLoader()
        self.visualizer = AdvancedVisualizationEngine()
        self.predictor = None
        self.analytics = DataAnalyticsEngine()
        
        # Initialize session state
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize session state variables"""
        defaults = {
            'data_loaded': False,
            'selected_colormap': "Viridis",
            'current_mode': "Data Viewer",
            'visualization_mode': 'surface',
            'lighting_preset': 'default',
            'mesh_opacity': 0.9,
            'show_statistics': True,
            'show_annotations': True,
            'prediction_method': 'rbf',
            'use_physics_constraints': True,
            'analytics_clusters': 3,
            'analytics_components': 3
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def run(self):
        """Main application entry point"""
        st.set_page_config(
            page_title="Advanced FEA Laser Simulation Platform",
            layout="wide",
            initial_sidebar_state="expanded",
            page_icon="🔬",
            menu_items={
                'Get Help': 'https://github.com/your-repo',
                'Report a bug': "https://github.com/your-repo/issues",
                'About': "# Advanced FEA Platform\nPhysics-informed simulation analysis"
            }
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
        """Apply enhanced CSS styling"""
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            background: linear-gradient(90deg, #1E88E5, #4A00E0, #8E2DE2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 1rem;
            font-weight: 800;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
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
        .feature-card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            margin-bottom: 1.5rem;
            border-left: 5px solid #1E88E5;
            transition: transform 0.3s ease;
        }
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.12);
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
        .warning-card {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .info-card {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #f8f9fa;
            border-radius: 8px 8px 0 0;
            padding: 10px 20px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #e9ecef;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1E88E5;
            color: white;
            box-shadow: 0 2px 10px rgba(30, 136, 229, 0.3);
        }
        .stButton > button {
            background: linear-gradient(90deg, #1E88E5, #4A00E0);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
            width: 100%;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(30, 136, 229, 0.4);
        }
        .data-table {
            background: white;
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _render_header(self):
        """Render application header"""
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.markdown('<h1 class="main-header">🔬 Advanced FEA Laser Simulation Platform</h1>', 
                       unsafe_allow_html=True)
            st.markdown("""
            <p style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">
            Physics-Informed Analysis | Machine Learning Predictions | Advanced Visualization
            </p>
            """, unsafe_allow_html=True)
    
    def _render_sidebar(self):
        """Render enhanced sidebar"""
        with st.sidebar:
            st.image("https://img.icons8.com/color/96/000000/engineering.png", width=80)
            
            st.markdown("### 🧭 Navigation")
            app_mode = st.selectbox(
                "Select Mode",
                ["Data Explorer", "Predictive Analysis", "Comparative Analytics", 
                 "Advanced Visualization", "Export & Reports"],
                index=0,
                key="nav_mode"
            )
            
            st.session_state.current_mode = app_mode
            
            st.markdown("---")
            st.markdown("### 📂 Data Management")
            
            if st.button("🚀 Load All Simulations", use_container_width=True):
                self._load_simulations()
            
            if st.session_state.get('data_loaded', False):
                st.success(f"✅ {len(st.session_state.simulations)} simulations loaded")
                
                st.markdown("---")
                st.markdown("### 🎨 Visualization Settings")
                
                with st.expander("Display Configuration", expanded=False):
                    st.session_state.visualization_mode = st.selectbox(
                        "Visualization Mode",
                        list(AdvancedVisualizationEngine.VISUALIZATION_MODES.keys()),
                        format_func=lambda x: AdvancedVisualizationEngine.VISUALIZATION_MODES[x]
                    )
                    
                    st.session_state.selected_colormap = st.selectbox(
                        "Colormap",
                        px.colors.named_colorscales(),
                        index=px.colors.named_colorscales().index('Viridis')
                    )
                    
                    st.session_state.lighting_preset = st.selectbox(
                        "Lighting Preset",
                        list(AdvancedVisualizationEngine.LIGHTING_PRESETS.keys())
                    )
                    
                    st.session_state.mesh_opacity = st.slider("Opacity", 0.1, 1.0, 0.9, 0.1)
                
                with st.expander("Analysis Settings", expanded=False):
                    st.session_state.prediction_method = st.selectbox(
                        "Prediction Method",
                        ["rbf", "kriging", "neural", "physics"]
                    )
                    
                    st.session_state.use_physics_constraints = st.checkbox(
                        "Apply Physics Constraints", value=True
                    )
                    
                    st.session_state.analytics_clusters = st.slider(
                        "Clusters for Analytics", 2, 10, 3
                    )
                    
                    st.session_state.analytics_components = st.slider(
                        "PCA Components", 2, 5, 3
                    )
    
    def _load_simulations(self):
        """Load simulations with progress tracking"""
        with st.spinner("Loading simulation data..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                simulations, summaries = self.data_loader.load_all_simulations(load_full_mesh=True)
                
                if simulations:
                    st.session_state.simulations = simulations
                    st.session_state.summaries = summaries
                    st.session_state.data_loaded = True
                    
                    # Initialize predictor
                    self.predictor = PhysicsInformedPredictor(self.data_loader)
                    st.session_state.predictor = self.predictor
                    
                    # Display success message
                    st.success("✅ Data loaded successfully!")
                    
                    # Show data summary
                    with st.expander("📊 Data Summary", expanded=True):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Simulations", len(simulations))
                        with col2:
                            st.metric("Total Fields", len(self.data_loader.available_fields))
                        with col3:
                            st.metric("Common Fields", len(self.data_loader.common_fields))
                
                progress_bar.progress(100)
                status_text.text("Complete!")
                
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
                st.error(traceback.format_exc())
            finally:
                progress_bar.empty()
                status_text.empty()
    
    def _render_main_content(self):
        """Render main content based on selected mode"""
        app_mode = st.session_state.current_mode
        
        if app_mode == "Data Explorer":
            self._render_data_explorer()
        elif app_mode == "Predictive Analysis":
            self._render_predictive_analysis()
        elif app_mode == "Comparative Analytics":
            self._render_comparative_analytics()
        elif app_mode == "Advanced Visualization":
            self._render_advanced_visualization()
        elif app_mode == "Export & Reports":
            self._render_export_reports()
    
    def _render_data_explorer(self):
        """Render enhanced data explorer"""
        st.markdown('<h2 class="sub-header">📊 Data Explorer</h2>', unsafe_allow_html=True)
        
        if not st.session_state.get('data_loaded', False):
            self._show_data_not_loaded()
            return
        
        simulations = st.session_state.simulations
        
        # Simulation selector
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            sim_name = st.selectbox(
                "Select Simulation",
                sorted(simulations.keys()),
                key="explorer_sim_select"
            )
        
        sim = simulations[sim_name]
        
        with col2:
            st.metric("Energy", f"{sim['energy_mJ']:.2f} mJ")
        with col3:
            st.metric("Duration", f"{sim['duration_ns']:.2f} ns")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["3D Visualization", "Field Analysis", "Mesh Statistics", "Time Series"])
        
        with tab1:
            self._render_3d_visualization(sim)
        
        with tab2:
            self._render_field_analysis(sim)
        
        with tab3:
            self._render_mesh_statistics(sim)
        
        with tab4:
            self._render_time_series(sim)
    
    def _render_3d_visualization(self, sim):
        """Render 3D visualization"""
        col1, col2 = st.columns([3, 1])
        
        with col1:
            field = st.selectbox(
                "Select Field",
                sorted(sim['field_info'].keys()),
                key="viz_field_select"
            )
        
        with col2:
            timestep = st.slider(
                "Timestep",
                0, sim['n_timesteps'] - 1, 0,
                key="viz_timestep"
            )
        
        # Get field values
        mesh_data = sim['mesh_data']
        field_data = mesh_data.fields[field][timestep]
        
        if field_data.ndim == 2:
            field_data = np.linalg.norm(field_data, axis=1)
        
        # Create visualization configuration
        config = {
            'mode': st.session_state.visualization_mode,
            'colormap': st.session_state.selected_colormap,
            'opacity': st.session_state.mesh_opacity,
            'lighting': st.session_state.lighting_preset,
            'show_statistics': st.session_state.show_statistics,
            'show_annotations': st.session_state.show_annotations,
            'title': f"{field} at Timestep {timestep + 1} - {sim['name']}"
        }
        
        # Create visualization
        fig = self.visualizer.create_advanced_mesh_visualization(
            mesh_data, field_data, config
        )
        
        st.plotly_chart(fig, use_container_width=True, use_container_width=True)
    
    def _render_field_analysis(self, sim):
        """Render field analysis"""
        st.markdown("#### 📈 Field Statistics Analysis")
        
        field = st.selectbox(
            "Select Field for Analysis",
            sorted(sim['field_info'].keys()),
            key="analysis_field"
        )
        
        # Compute statistics across timesteps
        mesh_data = sim['mesh_data']
        field_data = mesh_data.fields[field]
        
        if field_data.ndim == 3:
            field_data = np.linalg.norm(field_data, axis=2)
        
        # Compute statistics
        means = np.mean(field_data, axis=1)
        stds = np.std(field_data, axis=1)
        maxs = np.max(field_data, axis=1)
        mins = np.min(field_data, axis=1)
        
        # Create statistics plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(range(len(means))),
            y=means,
            mode='lines+markers',
            name='Mean',
            line=dict(width=2, color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=list(range(len(means))),
            y=means + stds,
            mode='lines',
            name='Mean + Std',
            line=dict(width=1, color='blue', dash='dash'),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=list(range(len(means))),
            y=means - stds,
            mode='lines',
            name='Mean - Std',
            line=dict(width=1, color='blue', dash='dash'),
            fill='tonexty',
            fillcolor='rgba(0, 100, 255, 0.1)',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=list(range(len(maxs))),
            y=maxs,
            mode='lines',
            name='Maximum',
            line=dict(width=1, color='red')
        ))
        
        fig.add_trace(go.Scatter(
            x=list(range(len(mins))),
            y=mins,
            mode='lines',
            name='Minimum',
            line=dict(width=1, color='green')
        ))
        
        fig.update_layout(
            title=f"{field} Statistics Over Time",
            xaxis_title="Timestep",
            yaxis_title="Field Value",
            height=400,
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Overall Mean", f"{np.mean(means):.3f}")
        with col2:
            st.metric("Overall Std", f"{np.mean(stds):.3f}")
        with col3:
            st.metric("Peak Value", f"{np.max(maxs):.3f}")
        with col4:
            st.metric("Dynamic Range", f"{np.max(maxs) - np.min(mins):.3f}")
    
    def _render_mesh_statistics(self, sim):
        """Render mesh statistics"""
        mesh_data = sim['mesh_data']
        
        st.markdown("#### 📐 Mesh Properties")
        
        if hasattr(mesh_data, 'mesh_stats'):
            stats = mesh_data.mesh_stats
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Geometrical Properties**")
                
                metrics = [
                    ("Surface Area", stats.get('surface_area', 0)),
                    ("Triangle Count", stats.get('triangle_count', 0)),
                    ("Avg Triangle Area", stats.get('avg_triangle_area', 0)),
                    ("Min Triangle Area", stats.get('triangle_areas', {}).get('min', 0)),
                    ("Max Triangle Area", stats.get('triangle_areas', {}).get('max', 0))
                ]
                
                for name, value in metrics:
                    if value:
                        st.metric(name, f"{value:.6f}")
            
            with col2:
                st.markdown("**Quality Metrics**")
                
                if 'triangle_quality' in stats:
                    quality = stats['triangle_quality']
                    
                    if 'aspect_ratio' in quality:
                        st.metric("Aspect Ratio", f"{quality['aspect_ratio']:.3f}")
                    
                    if 'edge_length_stats' in quality:
                        edge_stats = quality['edge_length_stats']
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Min Edge", f"{edge_stats['min']:.6f}")
                        with col_b:
                            st.metric("Max Edge", f"{edge_stats['max']:.6f}")
        
        # Show mesh connectivity if available
        if hasattr(mesh_data, 'connectivity_graph'):
            st.markdown("#### 🔗 Connectivity Graph")
            
            graph_stats = mesh_data.mesh_stats.get('graph_stats', {})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Number of Edges", graph_stats.get('num_edges', 0))
            with col2:
                st.metric("Average Degree", f"{graph_stats.get('avg_degree', 0):.2f}")
            with col3:
                st.metric("Connected Components", graph_stats.get('connected_components', 0))
    
    def _render_time_series(self, sim):
        """Render time series analysis"""
        field = st.selectbox(
            "Select Field for Time Analysis",
            sorted(sim['field_info'].keys()),
            key="time_field"
        )
        
        # Create time evolution plot
        mesh_data = sim['mesh_data']
        field_data = mesh_data.fields[field]
        
        if field_data.ndim == 3:
            field_data = np.linalg.norm(field_data, axis=2)
        
        time_points = list(range(field_data.shape[0]))
        
        config = {
            'title': f"{field} Time Evolution"
        }
        
        fig = self.visualizer.create_time_evolution_plot(
            list(field_data), time_points, config
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add time controls
        col1, col2, col3 = st.columns(3)
        with col1:
            start_time = st.number_input("Start Time", 0, len(time_points)-1, 0)
        with col2:
            end_time = st.number_input("End Time", 0, len(time_points)-1, len(time_points)-1)
        with col3:
            if st.button("Animate Time Series"):
                self._animate_time_series(sim, field, start_time, end_time)
    
    def _animate_time_series(self, sim, field, start, end):
        """Animate time series"""
        # This would create an animation
        st.info("Animation feature would generate a GIF/video of the time series")
    
    def _render_predictive_analysis(self):
        """Render predictive analysis interface"""
        st.markdown('<h2 class="sub-header">🔮 Predictive Analysis</h2>', unsafe_allow_html=True)
        
        if not st.session_state.get('data_loaded', False):
            self._show_data_not_loaded()
            return
        
        st.markdown("""
        <div class="info-card">
        <h4>🧠 Physics-Informed Predictions</h4>
        <p>This module uses machine learning and physics-based constraints to predict field distributions 
        for new parameter combinations. Multiple interpolation methods are available.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Query parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            energy_query = st.number_input(
                "Energy (mJ)",
                min_value=0.1,
                max_value=100.0,
                value=5.0,
                step=0.1,
                key="pred_energy"
            )
        with col2:
            duration_query = st.number_input(
                "Pulse Duration (ns)",
                min_value=0.5,
                max_value=50.0,
                value=4.0,
                step=0.1,
                key="pred_duration"
            )
        with col3:
            time_query = st.number_input(
                "Time (ns)",
                min_value=0.0,
                max_value=100.0,
                value=10.0,
                step=0.1,
                key="pred_time"
            )
        
        # Field and reference selection
        simulations = st.session_state.simulations
        
        col1, col2 = st.columns(2)
        with col1:
            if self.data_loader.common_fields:
                field = st.selectbox(
                    "Field to Predict",
                    sorted(self.data_loader.common_fields),
                    key="pred_field"
                )
            else:
                st.warning("No common fields available")
                return
        
        with col2:
            ref_sim = st.selectbox(
                "Reference Geometry",
                sorted(simulations.keys()),
                key="pred_ref_sim"
            )
        
        # Prediction options
        with st.expander("⚙️ Prediction Options", expanded=False):
            method = st.selectbox(
                "Prediction Method",
                ["rbf", "kriging", "neural", "physics"],
                index=["rbf", "kriging", "neural", "physics"].index(
                    st.session_state.prediction_method
                )
            )
            
            use_physics = st.checkbox(
                "Apply Physics Constraints",
                value=st.session_state.use_physics_constraints
            )
            
            n_similar = st.slider("Number of Similar Simulations", 1, 10, 5)
        
        if st.button("🚀 Generate Prediction", use_container_width=True):
            with st.spinner("Generating prediction..."):
                self._generate_prediction(
                    energy_query, duration_query, time_query,
                    field, ref_sim, method, use_physics, n_similar
                )
    
    def _generate_prediction(self, energy, duration, time, field, ref_sim, 
                           method, use_physics, n_similar):
        """Generate and display prediction"""
        simulations = st.session_state.simulations
        
        if ref_sim not in simulations:
            st.error(f"Reference simulation '{ref_sim}' not found")
            return
        
        mesh_data = simulations[ref_sim]['mesh_data']
        
        # Generate prediction
        prediction = self.predictor.predict_field(
            energy, duration, time, mesh_data, field,
            method=method, use_physics_constraints=use_physics
        )
        
        if prediction is None:
            st.error("Failed to generate prediction")
            return
        
        # Display results
        st.success(f"✅ Prediction generated with {method.upper()} method")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Confidence", f"{prediction.get('confidence', 0):.2%}")
        with col2:
            st.metric("Method", prediction.get('method', 'unknown'))
        with col3:
            st.metric("Mean Value", f"{np.mean(prediction['values']):.3f}")
        with col4:
            st.metric("Range", f"{np.ptp(prediction['values']):.3f}")
        
        # Create visualization
        config = {
            'mode': st.session_state.visualization_mode,
            'colormap': st.session_state.selected_colormap,
            'opacity': st.session_state.mesh_opacity,
            'lighting': st.session_state.lighting_preset,
            'show_statistics': True,
            'show_annotations': True,
            'title': f"Predicted {field} - E={energy}mJ, τ={duration}ns, t={time}ns"
        }
        
        fig = self.visualizer.create_advanced_mesh_visualization(
            mesh_data, prediction['values'], config
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show prediction details
        with st.expander("📋 Prediction Details", expanded=False):
            if 'similarity_scores' in prediction:
                st.write("**Similarity Scores:**")
                for i, score in enumerate(prediction['similarity_scores']):
                    st.write(f"  Source {i+1}: {score:.3f}")
            
            if 'warning' in prediction:
                st.warning(prediction['warning'])
    
    def _render_comparative_analytics(self):
        """Render comparative analytics"""
        st.markdown('<h2 class="sub-header">📈 Comparative Analytics</h2>', unsafe_allow_html=True)
        
        if not st.session_state.get('data_loaded', False):
            self._show_data_not_loaded()
            return
        
        simulations = st.session_state.simulations
        
        # Analytics type selection
        analytics_type = st.selectbox(
            "Select Analysis Type",
            ["PCA Analysis", "Cluster Analysis", "Correlation Matrix", "Sensitivity Analysis"],
            key="analytics_type"
        )
        
        if analytics_type == "PCA Analysis":
            self._render_pca_analysis(simulations)
        elif analytics_type == "Cluster Analysis":
            self._render_cluster_analysis(simulations)
        elif analytics_type == "Correlation Matrix":
            self._render_correlation_analysis(simulations)
        elif analytics_type == "Sensitivity Analysis":
            self._render_sensitivity_analysis(simulations)
    
    def _render_pca_analysis(self, simulations):
        """Render PCA analysis"""
        field = st.selectbox(
            "Select Field for PCA",
            sorted(self.data_loader.common_fields),
            key="pca_field"
        )
        
        if st.button("Run PCA Analysis", use_container_width=True):
            with st.spinner("Performing PCA..."):
                results = self.analytics.perform_pca_analysis(
                    simulations, field, 
                    n_components=st.session_state.analytics_components
                )
                
                if results:
                    # Plot PCA results
                    df_pca = results['pca_data']
                    
                    fig = px.scatter_3d(
                        df_pca, x='PC1', y='PC2', z='PC3',
                        color='Energy',
                        size='Duration',
                        hover_name='Simulation',
                        title=f"PCA of {field} across simulations"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show explained variance
                    col1, col2, col3 = st.columns(3)
                    for i, var in enumerate(results['explained_variance']):
                        with [col1, col2, col3][i]:
                            st.metric(f"PC{i+1} Variance", f"{var:.2%}")
    
    def _render_cluster_analysis(self, simulations):
        """Render cluster analysis"""
        field = st.selectbox(
            "Select Field for Clustering",
            sorted(self.data_loader.common_fields),
            key="cluster_field"
        )
        
        n_clusters = st.slider("Number of Clusters", 2, 10, 
                              st.session_state.analytics_clusters)
        
        if st.button("Run Cluster Analysis", use_container_width=True):
            with st.spinner("Performing clustering..."):
                results = self.analytics.perform_cluster_analysis(
                    simulations, field, n_clusters=n_clusters
                )
                
                if results:
                    # Plot clustering results
                    fig = px.scatter(
                        x=results['pca_data'][:, 0],
                        y=results['pca_data'][:, 1],
                        color=results['cluster_labels'].astype(str),
                        hover_name=results['simulation_names'],
                        title=f"Clustering of {field} across simulations"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show cluster statistics
                    st.write("**Cluster Distribution:**")
                    unique, counts = np.unique(results['cluster_labels'], return_counts=True)
                    for cluster, count in zip(unique, counts):
                        st.write(f"  Cluster {cluster}: {count} simulations")
    
    def _render_correlation_analysis(self, simulations):
        """Render correlation analysis"""
        available_fields = list(self.data_loader.common_fields)[:10]  # Limit to 10 for clarity
        selected_fields = st.multiselect(
            "Select Fields for Correlation Analysis",
            available_fields,
            default=available_fields[:5] if available_fields else []
        )
        
        if len(selected_fields) >= 2:
            if st.button("Compute Correlations", use_container_width=True):
                with st.spinner("Computing correlations..."):
                    results = self.analytics.compute_correlation_matrix(
                        simulations, selected_fields
                    )
                    
                    if results:
                        st.plotly_chart(results['heatmap_figure'], use_container_width=True)
                        
                        # Show top correlations
                        corr_matrix = results['correlation_matrix']
                        st.write("**Strongest Correlations:**")
                        
                        # Find top correlations
                        correlations = []
                        for i in range(len(corr_matrix.columns)):
                            for j in range(i+1, len(corr_matrix.columns)):
                                corr = abs(corr_matrix.iloc[i, j])
                                if corr > 0.5:  # Only show strong correlations
                                    correlations.append((
                                        corr_matrix.columns[i],
                                        corr_matrix.columns[j],
                                        corr_matrix.iloc[i, j]
                                    ))
                        
                        correlations.sort(key=lambda x: abs(x[2]), reverse=True)
                        
                        for field1, field2, corr in correlations[:10]:
                            st.write(f"  {field1} ↔ {field2}: {corr:.3f}")
    
    def _render_sensitivity_analysis(self, simulations):
        """Render sensitivity analysis"""
        field = st.selectbox(
            "Select Field for Sensitivity Analysis",
            sorted(self.data_loader.common_fields),
            key="sensitivity_field"
        )
        
        if st.button("Analyze Sensitivity", use_container_width=True):
            with st.spinner("Analyzing sensitivity..."):
                results = self.analytics.analyze_sensitivity(simulations, field)
                
                if results:
                    st.plotly_chart(results['figure'], use_container_width=True)
                    
                    # Display correlation coefficients
                    st.write("**Correlation Coefficients:**")
                    for param, value in results['correlations'].items():
                        param_name = param.replace('_', ' ').title()
                        st.metric(param_name, f"{value:.3f}")
    
    def _render_advanced_visualization(self):
        """Render advanced visualization interface"""
        st.markdown('<h2 class="sub-header">🎨 Advanced Visualization</h2>', unsafe_allow_html=True)
        
        if not st.session_state.get('data_loaded', False):
            self._show_data_not_loaded()
            return
        
        simulations = st.session_state.simulations
        
        # Visualization type selection
        viz_type = st.selectbox(
            "Visualization Type",
            ["Comparison Dashboard", "Time Animation", "Multiple Cross-Sections", "3D Volume"],
            key="adv_viz_type"
        )
        
        if viz_type == "Comparison Dashboard":
            self._render_comparison_dashboard(simulations)
        elif viz_type == "Time Animation":
            self._render_time_animation(simulations)
        elif viz_type == "Multiple Cross-Sections":
            self._render_multiple_cross_sections(simulations)
        elif viz_type == "3D Volume":
            self._render_3d_volume(simulations)
    
    def _render_comparison_dashboard(self, simulations):
        """Render comparison dashboard"""
        selected_sims = st.multiselect(
            "Select simulations to compare",
            sorted(simulations.keys()),
            default=list(simulations.keys())[:min(4, len(simulations))]
        )
        
        if len(selected_sims) < 2:
            st.info("Select at least 2 simulations for comparison")
            return
        
        field = st.selectbox(
            "Select Field for Comparison",
            sorted(self.data_loader.common_fields),
            key="comparison_field"
        )
        
        timestep = st.slider(
            "Timestep",
            0, 50, 0,  # Adjust max based on available timesteps
            key="comparison_timestep"
        )
        
        if st.button("Generate Comparison", use_container_width=True):
            # Prepare data for comparison
            mesh_data_list = []
            field_values_list = []
            names = []
            
            for sim_name in selected_sims:
                sim = simulations[sim_name]
                mesh_data = sim['mesh_data']
                
                if field in mesh_data.fields:
                    field_data = mesh_data.fields[field]
                    actual_timestep = min(timestep, field_data.shape[0] - 1)
                    
                    values = field_data[actual_timestep]
                    if values.ndim == 2:
                        values = np.linalg.norm(values, axis=1)
                    
                    mesh_data_list.append(mesh_data)
                    field_values_list.append(values)
                    names.append(f"{sim_name} (t={actual_timestep})")
            
            if mesh_data_list:
                config = {
                    'colormap': st.session_state.selected_colormap,
                    'opacity': st.session_state.mesh_opacity
                }
                
                fig = self.visualizer.create_comparison_dashboard(
                    mesh_data_list, field_values_list, names, config
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_time_animation(self, simulations):
        """Render time animation interface"""
        st.info("Time animation feature would create an animated visualization of field evolution")
        # Implementation would create animated GIF or video
    
    def _render_multiple_cross_sections(self, simulations):
        """Render multiple cross-sections"""
        sim_name = st.selectbox(
            "Select Simulation",
            sorted(simulations.keys()),
            key="cross_section_sim"
        )
        
        sim = simulations[sim_name]
        mesh_data = sim['mesh_data']
        
        field = st.selectbox(
            "Select Field",
            sorted(sim['field_info'].keys()),
            key="cross_section_field"
        )
        
        timestep = st.slider(
            "Timestep",
            0, sim['n_timesteps'] - 1, 0,
            key="cross_section_timestep"
        )
        
        # Get field values
        field_data = mesh_data.fields[field][timestep]
        if field_data.ndim == 2:
            field_data = np.linalg.norm(field_data, axis=1)
        
        # Create multiple cross-sections
        n_slices = st.slider("Number of slices", 1, 5, 3)
        axis = st.selectbox("Slice Axis", ["X", "Y", "Z"])
        
        # Determine slice positions
        if mesh_data.points is not None:
            if axis == 'X':
                coord_range = (np.min(mesh_data.points[:, 0]), np.max(mesh_data.points[:, 0]))
            elif axis == 'Y':
                coord_range = (np.min(mesh_data.points[:, 1]), np.max(mesh_data.points[:, 1]))
            else:
                coord_range = (np.min(mesh_data.points[:, 2]), np.max(mesh_data.points[:, 2]))
            
            slice_positions = np.linspace(coord_range[0], coord_range[1], n_slices + 2)[1:-1]
            
            # Create subplots
            fig = make_subplots(
                rows=1, cols=n_slices,
                subplot_titles=[f"{axis}={pos:.3f}" for pos in slice_positions]
            )
            
            for i, pos in enumerate(slice_positions):
                # Create slice visualization (simplified)
                # In practice, you would create actual cross-section visualizations
                col = i + 1
                
                # Simplified: just show slice positions
                fig.add_trace(
                    go.Scatter(
                        x=[pos], y=[0],
                        mode='markers',
                        marker=dict(size=10, color='red'),
                        showlegend=False
                    ),
                    row=1, col=col
                )
            
            fig.update_layout(height=300, title_text=f"{field} Cross-Sections along {axis}")
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_3d_volume(self, simulations):
        """Render 3D volume visualization"""
        st.info("3D volume rendering feature would create volumetric visualizations")
        # Implementation would use volume rendering techniques
    
    def _render_export_reports(self):
        """Render export and reports interface"""
        st.markdown('<h2 class="sub-header">💾 Export & Reports</h2>', unsafe_allow_html=True)
        
        if not st.session_state.get('data_loaded', False):
            self._show_data_not_loaded()
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Data Export")
            
            export_format = st.selectbox(
                "Export Format",
                ["CSV", "JSON", "Excel", "VTK", "Pickle"]
            )
            
            export_type = st.selectbox(
                "Export Type",
                ["Field Data", "Statistics", "Mesh Geometry", "All Data"]
            )
            
            if st.button("Export Data", use_container_width=True):
                self._export_data(export_format, export_type)
        
        with col2:
            st.markdown("#### 📈 Report Generation")
            
            report_type = st.selectbox(
                "Report Type",
                ["Summary Report", "Comparative Analysis", "Prediction Report", "Full Analysis"]
            )
            
            include_plots = st.checkbox("Include Plots", value=True)
            include_data = st.checkbox("Include Raw Data", value=False)
            
            if st.button("Generate Report", use_container_width=True):
                self._generate_report(report_type, include_plots, include_data)
    
    def _export_data(self, format, type):
        """Export data in specified format"""
        st.info(f"Export feature would save {type} data in {format} format")
        # Implementation would create and download files
    
    def _generate_report(self, type, include_plots, include_data):
        """Generate analysis report"""
        st.info(f"Report generation feature would create a {type} report")
        # Implementation would generate PDF/HTML reports
    
    def _show_data_not_loaded(self):
        """Show data not loaded warning"""
        st.markdown("""
        <div class="warning-card">
        <h3>⚠️ No Data Loaded</h3>
        <p>Please load simulations first using the "Load All Simulations" button in the sidebar.</p>
        </div>
        """, unsafe_allow_html=True)

# =============================================
# MAIN APPLICATION
# =============================================
def main():
    """Main application entry point"""
    try:
        app = EnhancedFEAPlatform()
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.error(traceback.format_exc())

if __name__ == "__main__":
    main()
