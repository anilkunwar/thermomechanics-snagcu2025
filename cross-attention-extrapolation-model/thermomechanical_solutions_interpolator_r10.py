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
from scipy.spatial import KDTree, cKDTree, Delaunay, SphericalVoronoi
from scipy.spatial.distance import cdist, pdist, squareform
from scipy import stats
import scipy.ndimage as ndimage
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
from scipy.spatial.transform import Rotation as R
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
# CONSTANTS AND CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEA_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "fea_solutions")
INTERPOLATED_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "interpolated_solutions")
os.makedirs(FEA_SOLUTIONS_DIR, exist_ok=True)
os.makedirs(INTERPOLATED_SOLUTIONS_DIR, exist_ok=True)

# =============================================
# SPHERICAL GEOMETRY UTILITIES
# =============================================
class SphericalGeometry:
    """Utilities for handling spherical geometry and preserving spherical topology"""
    
    @staticmethod
    def create_spherical_grid(resolution=100, radius=1.0):
        """Create a spherical grid using Fibonacci sphere sampling for even distribution"""
        # Fibonacci sphere algorithm for even point distribution
        points = []
        phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle in radians
        
        for i in range(resolution):
            y = 1 - (i / (resolution - 1)) * 2  # y goes from 1 to -1
            radius_at_y = np.sqrt(1 - y * y)  # Radius at y
            
            theta = phi * i  # Golden angle increment
            
            x = np.cos(theta) * radius_at_y
            z = np.sin(theta) * radius_at_y
            
            points.append([x, y, z])
        
        points = np.array(points) * radius
        
        # Create Delaunay triangulation for sphere surface
        # Project to 2D for triangulation
        spherical_coords = SphericalGeometry.cartesian_to_spherical(points)
        triangles = SphericalGeometry.create_spherical_triangulation(points, spherical_coords)
        
        return points, triangles
    
    @staticmethod
    def create_spherical_triangulation(points, spherical_coords=None):
        """Create triangulation for spherical surface using convex hull or Delaunay"""
        if spherical_coords is None:
            spherical_coords = SphericalGeometry.cartesian_to_spherical(points)
        
        # Use convex hull for triangulation
        try:
            hull = Delaunay(points)
            triangles = hull.simplices
            # Filter triangles that are too large (likely crossing through sphere)
            valid_triangles = []
            for tri in triangles:
                # Check if triangle center is near surface
                center = np.mean(points[tri], axis=0)
                center_dist = np.linalg.norm(center)
                # Accept triangles whose centers are near the surface
                if 0.8 <= center_dist <= 1.2:  # Allow some tolerance
                    valid_triangles.append(tri)
            
            return np.array(valid_triangles) if valid_triangles else triangles
        except:
            # Fallback: use KDTree to find nearest neighbors
            tree = cKDTree(points)
            triangles = []
            for i, point in enumerate(points):
                distances, indices = tree.query(point, k=4)  # Self + 3 neighbors
                indices = indices[1:]  # Remove self
                for j in range(len(indices)):
                    for k in range(j+1, len(indices)):
                        triangles.append([i, indices[j], indices[k]])
            
            return np.array(triangles[:len(points)*2])  # Limit number of triangles
    
    @staticmethod
    def cartesian_to_spherical(points):
        """Convert Cartesian coordinates to spherical coordinates (r, theta, phi)"""
        r = np.linalg.norm(points, axis=1)
        theta = np.arccos(points[:, 2] / (r + 1e-10))
        phi = np.arctan2(points[:, 1], points[:, 0])
        return np.vstack([r, theta, phi]).T
    
    @staticmethod
    def spherical_to_cartesian(spherical_coords):
        """Convert spherical coordinates to Cartesian coordinates"""
        r, theta, phi = spherical_coords[:, 0], spherical_coords[:, 1], spherical_coords[:, 2]
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return np.vstack([x, y, z]).T
    
    @staticmethod
    def resample_to_spherical_surface(source_points, source_values, target_resolution=100):
        """Resample field values to a new spherical surface"""
        # Create target spherical grid
        target_points, target_triangles = SphericalGeometry.create_spherical_grid(
            resolution=target_resolution,
            radius=np.mean(np.linalg.norm(source_points, axis=1))
        )
        
        # Convert both sets to spherical coordinates
        source_spherical = SphericalGeometry.cartesian_to_spherical(source_points)
        
        # Use RBF interpolation in spherical coordinates
        # Normalize spherical coordinates for interpolation
        source_theta_phi = source_spherical[:, 1:]  # Use theta and phi only
        target_theta_phi = SphericalGeometry.cartesian_to_spherical(target_points)[:, 1:]
        
        # Ensure values are 1D
        if source_values.ndim > 1:
            if source_values.shape[1] == 3:  # Vector field
                # Interpolate each component separately
                target_values = np.zeros((len(target_points), 3))
                for i in range(3):
                    rbf = RBFInterpolator(source_theta_phi, source_values[:, i], kernel='thin_plate_spline')
                    target_values[:, i] = rbf(target_theta_phi)
                return target_points, target_values, target_triangles
            else:
                # Use magnitude for vector fields
                source_values = np.linalg.norm(source_values, axis=1)
        
        # Scalar field interpolation
        rbf = RBFInterpolator(source_theta_phi, source_values, kernel='thin_plate_spline')
        target_values = rbf(target_theta_phi)
        
        return target_points, target_values, target_triangles
    
    @staticmethod
    def interpolate_on_sphere(source_points, source_values, target_points, method='rbf'):
        """Interpolate values from one spherical surface to another"""
        # Convert to spherical coordinates
        source_spherical = SphericalGeometry.cartesian_to_spherical(source_points)
        target_spherical = SphericalGeometry.cartesian_to_spherical(target_points)
        
        # Use theta and phi for interpolation (ignore radius for spherical surface)
        source_coords = source_spherical[:, 1:]
        target_coords = target_spherical[:, 1:]
        
        if method == 'rbf':
            # Use RBF interpolation with periodic boundary conditions
            # Handle periodic nature of phi (0 to 2π)
            source_coords_mod = source_coords.copy()
            target_coords_mod = target_coords.copy()
            
            # Ensure phi is in [0, 2π)
            source_coords_mod[:, 1] = np.mod(source_coords_mod[:, 1], 2*np.pi)
            target_coords_mod[:, 1] = np.mod(target_coords_mod[:, 1], 2*np.pi)
            
            # For points near 0/2π boundary, duplicate points with offset
            boundary_mask = source_coords_mod[:, 1] < 0.2*np.pi
            if np.any(boundary_mask):
                source_coords_ext = np.vstack([
                    source_coords_mod,
                    source_coords_mod[boundary_mask] + np.array([0, 2*np.pi])
                ])
                source_values_ext = np.concatenate([
                    source_values,
                    source_values[boundary_mask]
                ])
            else:
                source_coords_ext = source_coords_mod
                source_values_ext = source_values
            
            rbf = RBFInterpolator(source_coords_ext, source_values_ext, kernel='thin_plate_spline')
            return rbf(target_coords_mod)
        
        elif method == 'linear':
            # Linear interpolation on sphere using barycentric coordinates
            return griddata(source_coords, source_values, target_coords, method='linear', fill_value=0)
        
        else:
            # Default to nearest
            return griddata(source_coords, source_values, target_coords, method='nearest', fill_value=0)

# =============================================
# SPHERICAL MESH DATA STRUCTURE
# =============================================
class SphericalMeshData:
    """Enhanced mesh data structure optimized for spherical geometry"""
    
    def __init__(self):
        self.points = None
        self.triangles = None
        self.fields = {}
        self.field_info = {}
        self.normals = None
        self.curvature = None
        self.radial_distances = None
        self.spherical_coords = None
        self.region_labels = None
        self.connectivity_graph = None
        
    def initialize_from_points(self, points, triangles=None):
        """Initialize mesh from points, automatically detecting spherical topology"""
        self.points = points.astype(np.float32)
        
        if triangles is not None:
            self.triangles = triangles.astype(np.int32)
        else:
            # Auto-generate triangles for spherical surface
            self.triangles = self._create_spherical_triangulation()
        
        # Compute spherical coordinates
        self.spherical_coords = SphericalGeometry.cartesian_to_spherical(self.points)
        
        # Compute normals (outward from sphere center)
        self.normals = self.points / (np.linalg.norm(self.points, axis=1, keepdims=True) + 1e-10)
        
        # Compute radial distances
        self.radial_distances = np.linalg.norm(self.points, axis=1)
        
        # Compute curvature (for sphere, curvature is 1/radius)
        avg_radius = np.mean(self.radial_distances)
        self.curvature = np.ones_like(self.radial_distances) / avg_radius
        
        # Build connectivity
        self._build_connectivity()
    
    def _create_spherical_triangulation(self):
        """Create triangulation for spherical point cloud"""
        if self.points is None or len(self.points) < 4:
            return np.array([], dtype=np.int32)
        
        # Use convex hull for triangulation
        try:
            hull = Delaunay(self.points)
            triangles = hull.simplices
            
            # Filter triangles that make sense for spherical surface
            valid_triangles = []
            for tri in triangles:
                # Check if triangle is on spherical surface
                triangle_points = self.points[tri]
                centroid = np.mean(triangle_points, axis=0)
                centroid_dist = np.linalg.norm(centroid)
                avg_radius = np.mean(self.radial_distances) if hasattr(self, 'radial_distances') else 1.0
                
                # Accept triangles near the spherical surface
                if 0.7 * avg_radius <= centroid_dist <= 1.3 * avg_radius:
                    valid_triangles.append(tri)
            
            return np.array(valid_triangles, dtype=np.int32)
        except:
            # Fallback: create triangles from nearest neighbors
            return self._create_triangles_from_neighbors()
    
    def _create_triangles_from_neighbors(self):
        """Create triangles by connecting nearest neighbors"""
        tree = cKDTree(self.points)
        triangles_set = set()
        
        for i in range(len(self.points)):
            # Find k nearest neighbors
            distances, indices = tree.query(self.points[i], k=7)
            neighbors = indices[1:]  # Exclude self
            
            # Create triangles with neighbors
            for j in range(len(neighbors)):
                for k in range(j+1, len(neighbors)):
                    tri = tuple(sorted([i, neighbors[j], neighbors[k]]))
                    triangles_set.add(tri)
        
        triangles = np.array(list(triangles_set), dtype=np.int32)
        
        # Limit number of triangles
        max_triangles = len(self.points) * 3
        if len(triangles) > max_triangles:
            triangles = triangles[:max_triangles]
        
        return triangles
    
    def _build_connectivity(self):
        """Build connectivity graph for mesh"""
        if self.triangles is None or len(self.triangles) == 0:
            self.connectivity_graph = nx.Graph()
            return
        
        G = nx.Graph()
        
        # Add nodes
        for i in range(len(self.points)):
            G.add_node(i, pos=self.points[i])
        
        # Add edges from triangles
        edges_set = set()
        for tri in self.triangles:
            for i in range(3):
                edge = tuple(sorted((tri[i], tri[(i+1)%3])))
                if edge not in edges_set:
                    v1, v2 = self.points[edge[0]], self.points[edge[1]]
                    weight = np.linalg.norm(v1 - v2)
                    G.add_edge(edge[0], edge[1], weight=weight)
                    edges_set.add(edge)
        
        self.connectivity_graph = G
    
    def add_field(self, name, values, field_type="scalar"):
        """Add a field to the mesh"""
        self.fields[name] = values.astype(np.float32)
        self.field_info[name] = {
            "type": field_type,
            "dims": 1 if values.ndim == 1 else values.shape[1],
            "shape": values.shape
        }
    
    def resample_to_uniform_sphere(self, resolution=100):
        """Resample mesh to uniform spherical grid"""
        if self.points is None or len(self.points) == 0:
            return None
        
        # Create uniform spherical grid
        uniform_points, uniform_triangles = SphericalGeometry.create_spherical_grid(
            resolution=resolution,
            radius=np.mean(self.radial_distances)
        )
        
        # Create new mesh
        new_mesh = SphericalMeshData()
        new_mesh.initialize_from_points(uniform_points, uniform_triangles)
        
        # Interpolate fields
        for field_name, field_values in self.fields.items():
            if field_values.ndim == 1:  # Scalar field
                interpolated_values = SphericalGeometry.interpolate_on_sphere(
                    self.points, field_values, uniform_points, method='rbf'
                )
                new_mesh.add_field(field_name, interpolated_values, "scalar")
            elif field_values.ndim == 2 and field_values.shape[1] == 3:  # Vector field
                # Interpolate each component
                interpolated_values = np.zeros((len(uniform_points), 3))
                for i in range(3):
                    interpolated_values[:, i] = SphericalGeometry.interpolate_on_sphere(
                        self.points, field_values[:, i], uniform_points, method='rbf'
                    )
                new_mesh.add_field(field_name, interpolated_values, "vector")
        
        return new_mesh
    
    def compute_field_statistics(self, field_name):
        """Compute statistics for a field"""
        if field_name not in self.fields:
            return None
        
        values = self.fields[field_name]
        
        if values.ndim == 1:  # Scalar
            valid_values = values[~np.isnan(values)]
            if len(valid_values) == 0:
                return None
            
            return {
                "min": float(np.min(valid_values)),
                "max": float(np.max(valid_values)),
                "mean": float(np.mean(valid_values)),
                "std": float(np.std(valid_values)),
                "median": float(np.median(valid_values)),
                "q25": float(np.percentile(valid_values, 25)),
                "q75": float(np.percentile(valid_values, 75))
            }
        else:  # Vector
            magnitudes = np.linalg.norm(values, axis=1)
            valid_magnitudes = magnitudes[~np.isnan(magnitudes)]
            if len(valid_magnitudes) == 0:
                return None
            
            return {
                "min_mag": float(np.min(valid_magnitudes)),
                "max_mag": float(np.max(valid_magnitudes)),
                "mean_mag": float(np.mean(valid_magnitudes)),
                "std_mag": float(np.std(valid_magnitudes)),
                "median_mag": float(np.median(valid_magnitudes))
            }

# =============================================
# SPHERICAL INTERPOLATION MANAGER
# =============================================
class SphericalInterpolationManager:
    """Manager for spherical geometry-aware interpolation"""
    
    def __init__(self):
        self.interpolated_meshes = {}
        self.interpolation_history = []
        
    def create_interpolated_solution(self, source_mesh, source_field, target_energy, target_duration,
                                    method='spherical_rbf', resolution=100):
        """Create interpolated solution on spherical geometry"""
        
        # Generate unique ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        solution_id = f"interp_E{target_energy:.2f}_D{target_duration:.2f}_{timestamp}"
        
        # Resample to uniform sphere if needed
        if resolution != len(source_mesh.points):
            resampled_mesh = source_mesh.resample_to_uniform_sphere(resolution=resolution)
        else:
            resampled_mesh = source_mesh
        
        # Apply physics-aware scaling based on energy and duration
        scaled_mesh = self._apply_physics_scaling(resampled_mesh, source_field, 
                                                 target_energy, target_duration)
        
        # Store interpolated solution
        self.interpolated_meshes[solution_id] = {
            'id': solution_id,
            'mesh': scaled_mesh,
            'source_field': source_field,
            'energy': target_energy,
            'duration': target_duration,
            'method': method,
            'resolution': resolution,
            'created_at': timestamp,
            'is_interpolated': True
        }
        
        # Add to history
        self.interpolation_history.append({
            'id': solution_id,
            'energy': target_energy,
            'duration': target_duration,
            'method': method,
            'field': source_field,
            'timestamp': timestamp
        })
        
        # Save to disk
        self._save_interpolated_solution(solution_id)
        
        return solution_id
    
    def _apply_physics_scaling(self, mesh, field_name, target_energy, target_duration):
        """Apply physics-based scaling to field values based on energy and duration"""
        # Create a copy of the mesh
        scaled_mesh = SphericalMeshData()
        scaled_mesh.initialize_from_points(mesh.points.copy(), mesh.triangles.copy())
        
        # For each field in the mesh
        for field in mesh.fields:
            field_values = mesh.fields[field].copy()
            
            # Simple physics scaling model
            # Temperature scaling: proportional to energy/duration ratio
            if 'temp' in field.lower() or 'temperature' in field.lower():
                # Assume reference values at 1mJ, 1ns
                energy_scale = target_energy / 1.0
                duration_scale = 1.0 / max(target_duration, 0.1)
                scale_factor = np.sqrt(energy_scale * duration_scale)  # Geometric mean
                field_values = field_values * scale_factor
            
            # Stress scaling: proportional to energy
            elif 'stress' in field.lower() or 'pressure' in field.lower():
                energy_scale = target_energy / 1.0
                field_values = field_values * energy_scale
            
            # Displacement scaling: proportional to energy/duration^2
            elif 'disp' in field.lower() or 'deformation' in field.lower():
                energy_scale = target_energy / 1.0
                duration_scale = 1.0 / max(target_duration**2, 0.01)
                scale_factor = energy_scale * duration_scale
                field_values = field_values * scale_factor
            
            scaled_mesh.add_field(field, field_values, 
                                mesh.field_info[field]['type'] if field in mesh.field_info else 'scalar')
        
        return scaled_mesh
    
    def _save_interpolated_solution(self, solution_id):
        """Save interpolated solution to disk"""
        if solution_id not in self.interpolated_meshes:
            return
        
        solution = self.interpolated_meshes[solution_id]
        mesh = solution['mesh']
        
        save_path = os.path.join(INTERPOLATED_SOLUTIONS_DIR, f"{solution_id}.npz")
        
        # Prepare data for saving
        save_data = {
            'points': mesh.points,
            'triangles': mesh.triangles if mesh.triangles is not None else np.array([]),
            'metadata': {
                'id': solution_id,
                'energy': solution['energy'],
                'duration': solution['duration'],
                'method': solution['method'],
                'resolution': solution['resolution'],
                'created_at': solution['created_at'],
                'is_interpolated': True,
                'field_info': mesh.field_info
            }
        }
        
        # Add fields
        for field_name, field_values in mesh.fields.items():
            save_data[f'field_{field_name}'] = field_values
        
        np.savez_compressed(save_path, **save_data)
    
    def load_interpolated_solutions(self):
        """Load all saved interpolated solutions"""
        npz_files = glob.glob(os.path.join(INTERPOLATED_SOLUTIONS_DIR, "*.npz"))
        
        for npz_file in npz_files:
            try:
                data = np.load(npz_file, allow_pickle=True)
                
                # Extract data
                points = data['points']
                triangles = data['triangles'] if 'triangles' in data and len(data['triangles']) > 0 else None
                metadata = data['metadata'].item()
                
                # Create mesh
                mesh = SphericalMeshData()
                mesh.initialize_from_points(points, triangles)
                
                # Add fields
                field_info = metadata.get('field_info', {})
                for key in data:
                    if key.startswith('field_'):
                        field_name = key[6:]  # Remove 'field_' prefix
                        field_values = data[key]
                        field_type = field_info.get(field_name, {}).get('type', 'scalar')
                        mesh.add_field(field_name, field_values, field_type)
                
                # Store solution
                self.interpolated_meshes[metadata['id']] = {
                    'id': metadata['id'],
                    'mesh': mesh,
                    'energy': metadata['energy'],
                    'duration': metadata['duration'],
                    'method': metadata['method'],
                    'resolution': metadata['resolution'],
                    'created_at': metadata['created_at'],
                    'is_interpolated': True
                }
                
            except Exception as e:
                st.warning(f"Error loading interpolated solution {npz_file}: {e}")
        
        return len(self.interpolated_meshes)
    
    def get_solution(self, solution_id):
        """Get interpolated solution by ID"""
        return self.interpolated_meshes.get(solution_id)
    
    def get_all_solutions(self):
        """Get all interpolated solutions"""
        return list(self.interpolated_meshes.values())

# =============================================
# SPHERICAL VISUALIZATION ENGINE
# =============================================
class SphericalVisualizationEngine:
    """Visualization engine optimized for spherical geometry"""
    
    COLORMAPS = [
        'Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis',
        'Rainbow', 'Jet', 'Hot', 'Cool', 'Portland',
        'Bluered', 'Electric', 'Thermal', 'Balance',
        'Brwnyl', 'Darkmint', 'Emrld', 'Mint', 'Oranges',
        'Purp', 'Purples', 'Sunset', 'Sunsetdark', 'Teal',
        'Tealgrn', 'Twilight', 'Burg', 'Burgyl'
    ]
    
    @staticmethod
    def create_spherical_visualization(mesh, field_name, config=None):
        """Create 3D visualization of spherical mesh with field values"""
        if config is None:
            config = {}
        
        # Default configuration
        default_config = {
            'colormap': 'Viridis',
            'opacity': 0.9,
            'show_wireframe': False,
            'wireframe_opacity': 0.2,
            'show_points': False,
            'point_size': 3,
            'lighting': {
                'ambient': 0.8,
                'diffuse': 0.8,
                'specular': 0.5,
                'roughness': 0.5
            },
            'show_colorbar': True,
            'colorbar_title': field_name,
            'title': f"Spherical Field: {field_name}",
            'height': 700
        }
        
        # Update with user config
        config = {**default_config, **config}
        
        fig = go.Figure()
        
        # Get field values
        if field_name not in mesh.fields:
            st.error(f"Field '{field_name}' not found in mesh")
            return fig
        
        field_values = mesh.fields[field_name]
        
        # Handle vector fields by computing magnitude
        if field_values.ndim == 2 and field_values.shape[1] == 3:
            display_values = np.linalg.norm(field_values, axis=1)
            colorbar_title = f"{field_name} (Magnitude)"
        else:
            display_values = field_values
            colorbar_title = field_name
        
        # Create spherical surface mesh
        if mesh.triangles is not None and len(mesh.triangles) > 0:
            # Use mesh triangles for surface
            fig.add_trace(go.Mesh3d(
                x=mesh.points[:, 0],
                y=mesh.points[:, 1],
                z=mesh.points[:, 2],
                i=mesh.triangles[:, 0],
                j=mesh.triangles[:, 1],
                k=mesh.triangles[:, 2],
                intensity=display_values,
                colorscale=config['colormap'],
                intensitymode='vertex',
                opacity=config['opacity'],
                lighting=config['lighting'],
                flatshading=False,
                showscale=config['show_colorbar'],
                colorbar=dict(
                    title=colorbar_title,
                    thickness=20,
                    len=0.75,
                    tickfont=dict(size=12)
                ),
                name=field_name,
                hoverinfo='skip'
            ))
        else:
            # Fallback to point cloud
            fig.add_trace(go.Scatter3d(
                x=mesh.points[:, 0],
                y=mesh.points[:, 1],
                z=mesh.points[:, 2],
                mode='markers',
                marker=dict(
                    size=config['point_size'],
                    color=display_values,
                    colorscale=config['colormap'],
                    opacity=config['opacity'],
                    showscale=config['show_colorbar'],
                    colorbar=dict(
                        title=colorbar_title,
                        thickness=20,
                        len=0.75
                    )
                ),
                name=field_name,
                hoverinfo='skip'
            ))
        
        # Add wireframe if requested
        if config['show_wireframe'] and mesh.triangles is not None:
            edges = SphericalVisualizationEngine._extract_wireframe_edges(mesh)
            fig.add_trace(go.Scatter3d(
                x=edges['x'],
                y=edges['y'],
                z=edges['z'],
                mode='lines',
                line=dict(
                    color='black',
                    width=1
                ),
                opacity=config['wireframe_opacity'],
                showlegend=False,
                hoverinfo='none'
            ))
        
        # Add points if requested
        if config['show_points']:
            fig.add_trace(go.Scatter3d(
                x=mesh.points[:, 0],
                y=mesh.points[:, 1],
                z=mesh.points[:, 2],
                mode='markers',
                marker=dict(
                    size=config['point_size'],
                    color='rgba(0, 0, 0, 0.3)',
                    symbol='circle'
                ),
                showlegend=False,
                hoverinfo='none'
            ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=config['title'],
                font=dict(size=20),
                x=0.5,
                xanchor='center'
            ),
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
                    backgroundcolor="white",
                    showticklabels=True
                ),
                yaxis=dict(
                    title="Y",
                    gridcolor="lightgray",
                    showbackground=True,
                    backgroundcolor="white",
                    showticklabels=True
                ),
                zaxis=dict(
                    title="Z",
                    gridcolor="lightgray",
                    showbackground=True,
                    backgroundcolor="white",
                    showticklabels=True
                )
            ),
            height=config['height'],
            margin=dict(l=0, r=0, t=60, b=0),
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def _extract_wireframe_edges(mesh):
        """Extract wireframe edges from mesh triangles"""
        if mesh.triangles is None:
            return {'x': [], 'y': [], 'z': []}
        
        edges_set = set()
        for tri in mesh.triangles:
            for i in range(3):
                edge = tuple(sorted((tri[i], tri[(i+1)%3])))
                edges_set.add(edge)
        
        edges_x, edges_y, edges_z = [], [], []
        for edge in edges_set:
            edges_x.extend([mesh.points[edge[0], 0], mesh.points[edge[1], 0], None])
            edges_y.extend([mesh.points[edge[0], 1], mesh.points[edge[1], 1], None])
            edges_z.extend([mesh.points[edge[0], 2], mesh.points[edge[1], 2], None])
        
        return {'x': edges_x, 'y': edges_y, 'z': edges_z}
    
    @staticmethod
    def create_comparison_visualization(mesh1, mesh2, field_name, title1="Original", title2="Interpolated"):
        """Create side-by-side comparison visualization"""
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'surface'}, {'type': 'surface'}]],
            subplot_titles=[title1, title2],
            horizontal_spacing=0.05
        )
        
        # Get field values
        field_values1 = mesh1.fields[field_name] if field_name in mesh1.fields else None
        field_values2 = mesh2.fields[field_name] if field_name in mesh2.fields else None
        
        if field_values1 is None or field_values2 is None:
            st.error(f"Field '{field_name}' not found in both meshes")
            return fig
        
        # Handle vector fields
        if field_values1.ndim == 2 and field_values1.shape[1] == 3:
            display_values1 = np.linalg.norm(field_values1, axis=1)
            display_values2 = np.linalg.norm(field_values2, axis=1)
        else:
            display_values1 = field_values1
            display_values2 = field_values2
        
        # Normalize colorscale across both plots for fair comparison
        all_values = np.concatenate([display_values1, display_values2])
        vmin, vmax = np.nanmin(all_values), np.nanmax(all_values)
        
        # Plot first mesh
        if mesh1.triangles is not None and len(mesh1.triangles) > 0:
            fig.add_trace(go.Mesh3d(
                x=mesh1.points[:, 0],
                y=mesh1.points[:, 1],
                z=mesh1.points[:, 2],
                i=mesh1.triangles[:, 0],
                j=mesh1.triangles[:, 1],
                k=mesh1.triangles[:, 2],
                intensity=display_values1,
                colorscale='Viridis',
                intensitymode='vertex',
                opacity=0.9,
                lighting=dict(ambient=0.8, diffuse=0.8, specular=0.5, roughness=0.5),
                flatshading=False,
                showscale=False,
                name=title1
            ), row=1, col=1)
        else:
            fig.add_trace(go.Scatter3d(
                x=mesh1.points[:, 0],
                y=mesh1.points[:, 1],
                z=mesh1.points[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=display_values1,
                    colorscale='Viridis',
                    opacity=0.9,
                    cmin=vmin,
                    cmax=vmax
                ),
                name=title1
            ), row=1, col=1)
        
        # Plot second mesh
        if mesh2.triangles is not None and len(mesh2.triangles) > 0:
            fig.add_trace(go.Mesh3d(
                x=mesh2.points[:, 0],
                y=mesh2.points[:, 1],
                z=mesh2.points[:, 2],
                i=mesh2.triangles[:, 0],
                j=mesh2.triangles[:, 1],
                k=mesh2.triangles[:, 2],
                intensity=display_values2,
                colorscale='Viridis',
                intensitymode='vertex',
                opacity=0.9,
                lighting=dict(ambient=0.8, diffuse=0.8, specular=0.5, roughness=0.5),
                flatshading=False,
                showscale=True,
                colorbar=dict(
                    title=field_name,
                    thickness=20,
                    len=0.75,
                    x=1.02
                ),
                name=title2
            ), row=1, col=2)
        else:
            fig.add_trace(go.Scatter3d(
                x=mesh2.points[:, 0],
                y=mesh2.points[:, 1],
                z=mesh2.points[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=display_values2,
                    colorscale='Viridis',
                    opacity=0.9,
                    cmin=vmin,
                    cmax=vmax,
                    showscale=True,
                    colorbar=dict(
                        title=field_name,
                        thickness=20,
                        len=0.75,
                        x=1.02
                    )
                ),
                name=title2
            ), row=1, col=2)
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Comparison: {field_name}",
                font=dict(size=24),
                x=0.5,
                xanchor='center'
            ),
            height=600,
            showlegend=False,
            scene=dict(
                aspectmode='data',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            scene2=dict(
                aspectmode='data',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            )
        )
        
        return fig
    
    @staticmethod
    def create_field_statistics_plot(mesh, field_name):
        """Create statistics visualization for field"""
        if field_name not in mesh.fields:
            return None
        
        field_values = mesh.fields[field_name]
        
        if field_values.ndim == 2 and field_values.shape[1] == 3:
            # Vector field - plot magnitude
            magnitudes = np.linalg.norm(field_values, axis=1)
            values = magnitudes[~np.isnan(magnitudes)]
        else:
            # Scalar field
            values = field_values[~np.isnan(field_values)]
        
        if len(values) == 0:
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Histogram", "Radial Distribution"],
            horizontal_spacing=0.2
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=values,
                nbinsx=50,
                name="Distribution",
                marker_color='skyblue',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Add vertical lines for statistics
        stats = {
            'Mean': np.mean(values),
            'Median': np.median(values),
            'Std Dev': np.std(values)
        }
        
        for stat_name, stat_value in stats.items():
            fig.add_vline(
                x=stat_value,
                line_dash="dash",
                line_color="red",
                annotation_text=stat_name,
                annotation_position="top right",
                row=1, col=1
            )
        
        # Radial distribution (if mesh has radial distances)
        if hasattr(mesh, 'radial_distances'):
            radial_distances = mesh.radial_distances[~np.isnan(field_values)]
            valid_values = values[:len(radial_distances)]  # Ensure same length
            
            fig.add_trace(
                go.Scatter(
                    x=radial_distances,
                    y=valid_values,
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=valid_values,
                        colorscale='Viridis',
                        opacity=0.6,
                        showscale=False
                    ),
                    name="Radial Profile"
                ),
                row=1, col=2
            )
            
            # Add trend line
            if len(valid_values) > 10:
                z = np.polyfit(radial_distances, valid_values, 3)
                p = np.poly1d(z)
                x_fit = np.linspace(np.min(radial_distances), np.max(radial_distances), 100)
                y_fit = p(x_fit)
                
                fig.add_trace(
                    go.Scatter(
                        x=x_fit,
                        y=y_fit,
                        mode='lines',
                        line=dict(color='red', width=2),
                        name="Trend"
                    ),
                    row=1, col=2
                )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Field Statistics: {field_name}",
                font=dict(size=20),
                x=0.5,
                xanchor='center'
            ),
            height=400,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Field Value", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        
        if hasattr(mesh, 'radial_distances'):
            fig.update_xaxes(title_text="Radial Distance", row=1, col=2)
            fig.update_yaxes(title_text="Field Value", row=1, col=2)
        
        return fig

# =============================================
# ATTENTION-BASED INTERPOLATION WITH SPATIAL LOCALITY
# =============================================
class AttentionInterpolator:
    """Physics-informed attention mechanism for spherical interpolation"""
    
    def __init__(self, n_heads=4, spatial_weight=0.7, temperature=1.0):
        self.n_heads = n_heads
        self.spatial_weight = spatial_weight
        self.temperature = temperature
        self.source_embeddings = []
        self.source_meshes = []
        self.source_metadata = []
        self.fitted = False
        
    def load_training_data(self, meshes, metadata_list):
        """Load training meshes and metadata for attention mechanism"""
        self.source_meshes = meshes
        self.source_metadata = metadata_list
        
        # Create embeddings for each mesh
        for i, (mesh, metadata) in enumerate(zip(meshes, metadata_list)):
            embedding = self._create_mesh_embedding(mesh, metadata)
            self.source_embeddings.append(embedding)
        
        self.fitted = True
        return len(self.source_embeddings)
    
    def _create_mesh_embedding(self, mesh, metadata):
        """Create embedding vector for a mesh"""
        # Extract features from mesh
        if hasattr(mesh, 'radial_distances'):
            radial_stats = [
                np.mean(mesh.radial_distances),
                np.std(mesh.radial_distances),
                np.min(mesh.radial_distances),
                np.max(mesh.radial_distances)
            ]
        else:
            radial_stats = [0, 0, 0, 0]
        
        # Extract features from metadata
        energy = metadata.get('energy', 0)
        duration = metadata.get('duration', 0)
        
        # Physics-based features
        power = energy / max(duration, 0.1)
        energy_density = energy / (max(duration, 0.1) ** 2)
        
        # Create embedding vector
        embedding = np.array([
            energy,
            duration,
            power,
            energy_density,
            np.log1p(energy),
            np.log1p(duration),
            *radial_stats
        ])
        
        return embedding
    
    def interpolate_field(self, query_energy, query_duration, query_mesh, field_name, 
                         method='attention_weighted'):
        """Interpolate field using attention mechanism"""
        if not self.fitted or len(self.source_meshes) == 0:
            return self._fallback_interpolation(query_mesh, field_name)
        
        # Create query embedding
        query_metadata = {'energy': query_energy, 'duration': query_duration}
        query_embedding = self._create_mesh_embedding(query_mesh, query_metadata)
        
        # Compute attention weights
        attention_weights = self._compute_attention_weights(query_embedding)
        
        if method == 'attention_weighted':
            # Weighted combination of source fields
            interpolated_field = self._weighted_field_interpolation(
                attention_weights, query_mesh, field_name
            )
        elif method == 'attention_guided_rbf':
            # Use attention to select neighbors for RBF interpolation
            interpolated_field = self._attention_guided_rbf(
                attention_weights, query_mesh, field_name
            )
        else:
            interpolated_field = self._fallback_interpolation(query_mesh, field_name)
        
        return interpolated_field, attention_weights
    
    def _compute_attention_weights(self, query_embedding):
        """Compute attention weights using multi-head attention"""
        n_sources = len(self.source_embeddings)
        
        if n_sources == 0:
            return np.array([])
        
        # Multi-head attention
        head_weights = np.zeros((self.n_heads, n_sources))
        
        for head in range(self.n_heads):
            # Random projection for each head
            np.random.seed(42 + head)
            proj_dim = min(8, len(query_embedding))
            proj_matrix = np.random.randn(len(query_embedding), proj_dim)
            
            # Project embeddings
            query_proj = query_embedding @ proj_matrix
            source_projs = np.array([emb @ proj_matrix for emb in self.source_embeddings])
            
            # Compute attention scores (cosine similarity)
            query_norm = np.linalg.norm(query_proj)
            source_norms = np.linalg.norm(source_projs, axis=1)
            
            similarities = np.zeros(n_sources)
            for i in range(n_sources):
                if query_norm > 0 and source_norms[i] > 0:
                    similarities[i] = np.dot(query_proj, source_projs[i]) / (query_norm * source_norms[i])
                else:
                    similarities[i] = 0
            
            # Apply spatial locality regulation
            if self.spatial_weight > 0:
                spatial_sim = self._compute_spatial_similarity(query_embedding)
                similarities = (1 - self.spatial_weight) * similarities + self.spatial_weight * spatial_sim
            
            head_weights[head] = similarities
        
        # Combine head weights
        avg_weights = np.mean(head_weights, axis=0)
        
        # Apply temperature scaling
        if self.temperature != 1.0:
            avg_weights = np.power(avg_weights, 1.0/self.temperature)
        
        # Softmax normalization
        max_weight = np.max(avg_weights)
        exp_weights = np.exp(avg_weights - max_weight)
        attention_weights = exp_weights / (np.sum(exp_weights) + 1e-12)
        
        return attention_weights
    
    def _compute_spatial_similarity(self, query_embedding):
        """Compute spatial similarity based on parameter space"""
        n_sources = len(self.source_embeddings)
        spatial_sim = np.zeros(n_sources)
        
        for i in range(n_sources):
            # Compare energy and duration (first two features)
            energy_diff = abs(query_embedding[0] - self.source_embeddings[i][0]) / max(abs(query_embedding[0]), 1)
            duration_diff = abs(query_embedding[1] - self.source_embeddings[i][1]) / max(abs(query_embedding[1]), 1)
            
            # Combined similarity (inverse distance)
            total_diff = np.sqrt(energy_diff**2 + duration_diff**2)
            spatial_sim[i] = np.exp(-total_diff)
        
        return spatial_sim
    
    def _weighted_field_interpolation(self, attention_weights, query_mesh, field_name):
        """Weighted interpolation of fields from source meshes"""
        # Find common field across source meshes
        source_fields = []
        valid_indices = []
        
        for i, mesh in enumerate(self.source_meshes):
            if field_name in mesh.fields:
                source_fields.append(mesh.fields[field_name])
                valid_indices.append(i)
        
        if len(source_fields) == 0:
            return self._fallback_interpolation(query_mesh, field_name)
        
        # Normalize weights for valid sources
        valid_weights = attention_weights[valid_indices]
        valid_weights = valid_weights / np.sum(valid_weights)
        
        # Align all fields to query mesh geometry
        aligned_fields = []
        for field in source_fields:
            # Resample field to query mesh geometry
            aligned_field = SphericalGeometry.interpolate_on_sphere(
                self.source_meshes[0].points,  # Reference geometry
                field if field.ndim == 1 else np.linalg.norm(field, axis=1),
                query_mesh.points,
                method='rbf'
            )
            aligned_fields.append(aligned_field)
        
        # Weighted combination
        interpolated = np.zeros_like(aligned_fields[0])
        for field, weight in zip(aligned_fields, valid_weights):
            interpolated += field * weight
        
        return interpolated
    
    def _attention_guided_rbf(self, attention_weights, query_mesh, field_name):
        """Use attention weights to select neighbors for RBF interpolation"""
        # Select top-k sources based on attention weights
        k = min(5, len(attention_weights))
        top_indices = np.argsort(attention_weights)[-k:]
        
        # Collect source points and values
        source_points_list = []
        source_values_list = []
        
        for idx in top_indices:
            if idx < len(self.source_meshes) and field_name in self.source_meshes[idx].fields:
                mesh = self.source_meshes[idx]
                field_values = mesh.fields[field_name]
                
                if field_values.ndim == 2:
                    field_values = np.linalg.norm(field_values, axis=1)
                
                source_points_list.append(mesh.points)
                source_values_list.append(field_values)
        
        if len(source_points_list) == 0:
            return self._fallback_interpolation(query_mesh, field_name)
        
        # Combine all sources
        all_source_points = np.vstack(source_points_list)
        all_source_values = np.concatenate(source_values_list)
        
        # RBF interpolation
        query_points = query_mesh.points
        
        # Use spherical coordinates for interpolation
        source_spherical = SphericalGeometry.cartesian_to_spherical(all_source_points)[:, 1:]
        query_spherical = SphericalGeometry.cartesian_to_spherical(query_points)[:, 1:]
        
        rbf = RBFInterpolator(source_spherical, all_source_values, kernel='thin_plate_spline')
        interpolated = rbf(query_spherical)
        
        return interpolated
    
    def _fallback_interpolation(self, query_mesh, field_name):
        """Fallback interpolation method"""
        # Create synthetic field based on radial distance
        radial_distances = np.linalg.norm(query_mesh.points, axis=1)
        normalized_distances = radial_distances / np.max(radial_distances)
        
        # Different patterns for different field types
        if 'temp' in field_name.lower():
            # Gaussian temperature profile
            interpolated = np.exp(-10 * (normalized_distances - 0.5)**2)
        elif 'stress' in field_name.lower():
            # Radial stress pattern
            interpolated = 1 - normalized_distances
        else:
            # Sinusoidal pattern
            interpolated = np.sin(2 * np.pi * normalized_distances)
        
        return interpolated
    
    def visualize_attention(self, attention_weights):
        """Visualize attention weights"""
        if len(attention_weights) == 0:
            return None
        
        fig = go.Figure()
        
        # Create bar chart of attention weights
        sources = [f"Source {i+1}" for i in range(len(attention_weights))]
        
        fig.add_trace(go.Bar(
            x=sources,
            y=attention_weights,
            marker_color='skyblue',
            opacity=0.7,
            name='Attention Weights'
        ))
        
        # Add source metadata as hover text
        hover_text = []
        for i, metadata in enumerate(self.source_metadata):
            if i < len(attention_weights):
                energy = metadata.get('energy', 'N/A')
                duration = metadata.get('duration', 'N/A')
                hover_text.append(f"Energy: {energy}mJ<br>Duration: {duration}ns<br>Weight: {attention_weights[i]:.4f}")
        
        fig.update_traces(
            hovertemplate='%{x}<br>%{customdata}',
            customdata=hover_text
        )
        
        fig.update_layout(
            title=dict(
                text="Attention Weights Distribution",
                font=dict(size=20),
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="Source Simulation",
            yaxis_title="Attention Weight",
            height=400,
            showlegend=False
        )
        
        return fig

# =============================================
# MAIN APPLICATION
# =============================================
class SphericalFEAApplication:
    """Main application for spherical FEA analysis and interpolation"""
    
    def __init__(self):
        self.data_loader = None
        self.interpolation_manager = SphericalInterpolationManager()
        self.visualization_engine = SphericalVisualizationEngine()
        self.attention_interpolator = AttentionInterpolator()
        self.loaded_meshes = {}
        self.loaded_metadata = {}
        
        # Initialize session state
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize session state variables"""
        if 'app_initialized' not in st.session_state:
            st.session_state.app_initialized = True
            st.session_state.data_loaded = False
            st.session_state.loaded_simulations = {}
            st.session_state.current_mode = "Data Explorer"
            st.session_state.selected_colormap = "Viridis"
            st.session_state.visualization_config = {
                'opacity': 0.9,
                'show_wireframe': False,
                'show_points': False
            }
    
    def run(self):
        """Run the main application"""
        st.set_page_config(
            page_title="Spherical FEA Laser Simulation Platform",
            layout="wide",
            initial_sidebar_state="expanded",
            page_icon="🔬"
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
        .info-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1.5rem 0;
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        }
        .warning-card {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1.5rem 0;
        }
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
            border-left: 5px solid #1E88E5;
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
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(30, 136, 229, 0.4);
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _render_header(self):
        """Render application header"""
        st.markdown('<h1 class="main-header">🔬 Spherical FEA Laser Simulation Platform</h1>', 
                   unsafe_allow_html=True)
        st.markdown("""
        <p style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">
        Spherical Geometry Preservation | Physics-Informed Interpolation | Attention-Based Spatial Locality
        </p>
        """, unsafe_allow_html=True)
    
    def _render_sidebar(self):
        """Render sidebar with navigation and controls"""
        with st.sidebar:
            st.markdown("### 🧭 Navigation")
            
            app_mode = st.selectbox(
                "Select Mode",
                ["Data Explorer", "Interpolation", "Comparative Analysis", "Advanced Visualization"],
                key="app_mode"
            )
            
            st.session_state.current_mode = app_mode
            
            st.markdown("---")
            st.markdown("### 📂 Data Management")
            
            if st.button("📥 Load Simulation Data", use_container_width=True):
                self._load_simulation_data()
            
            # Load interpolated solutions
            if st.button("🔄 Load Interpolated Solutions", use_container_width=True):
                num_solutions = self.interpolation_manager.load_interpolated_solutions()
                if num_solutions > 0:
                    st.success(f"Loaded {num_solutions} interpolated solutions")
                else:
                    st.info("No interpolated solutions found")
            
            if st.session_state.data_loaded:
                st.markdown("---")
                st.markdown("### 🎨 Visualization Settings")
                
                with st.expander("Display Settings", expanded=False):
                    st.session_state.selected_colormap = st.selectbox(
                        "Colormap",
                        SphericalVisualizationEngine.COLORMAPS,
                        index=0
                    )
                    
                    st.session_state.visualization_config['opacity'] = st.slider(
                        "Opacity", 0.1, 1.0, 0.9, 0.1
                    )
                    
                    st.session_state.visualization_config['show_wireframe'] = st.checkbox(
                        "Show Wireframe", False
                    )
                    
                    st.session_state.visualization_config['show_points'] = st.checkbox(
                        "Show Points", False
                    )
                
                st.markdown("---")
                st.markdown("### ⚙️ Interpolation Settings")
                
                with st.expander("Attention Parameters", expanded=False):
                    n_heads = st.slider("Attention Heads", 1, 8, 4)
                    spatial_weight = st.slider("Spatial Weight", 0.0, 1.0, 0.7)
                    temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
                    
                    self.attention_interpolator.n_heads = n_heads
                    self.attention_interpolator.spatial_weight = spatial_weight
                    self.attention_interpolator.temperature = temperature
    
    def _load_simulation_data(self):
        """Load simulation data from VTU files"""
        with st.spinner("Loading simulation data..."):
            try:
                # Find all simulation folders
                folders = glob.glob(os.path.join(FEA_SOLUTIONS_DIR, "q*mJ-delta*ns"))
                
                if not folders:
                    st.error(f"No simulation folders found in {FEA_SOLUTIONS_DIR}")
                    return
                
                progress_bar = st.progress(0)
                
                for i, folder in enumerate(folders):
                    folder_name = os.path.basename(folder)
                    
                    # Parse energy and duration from folder name
                    match = re.match(r"q([\dp\.]+)mJ-delta([\dp\.]+)ns", folder_name)
                    if not match:
                        continue
                    
                    energy = float(match.group(1).replace("p", "."))
                    duration = float(match.group(2).replace("p", "."))
                    
                    # Find VTU files
                    vtu_files = sorted(glob.glob(os.path.join(folder, "*.vtu")))
                    if not vtu_files:
                        continue
                    
                    # Load first VTU file to get mesh structure
                    try:
                        mesh_data = meshio.read(vtu_files[0])
                        
                        # Extract points and triangles
                        points = mesh_data.points.astype(np.float32)
                        
                        # Find triangles
                        triangles = None
                        for cell_block in mesh_data.cells:
                            if cell_block.type == "triangle":
                                triangles = cell_block.data.astype(np.int32)
                                break
                        
                        # Create spherical mesh
                        spherical_mesh = SphericalMeshData()
                        spherical_mesh.initialize_from_points(points, triangles)
                        
                        # Extract fields
                        for field_name, field_data in mesh_data.point_data.items():
                            spherical_mesh.add_field(
                                field_name, 
                                field_data.astype(np.float32),
                                "scalar" if field_data.ndim == 1 else "vector"
                            )
                        
                        # Store mesh and metadata
                        mesh_id = f"{folder_name}_t0"
                        self.loaded_meshes[mesh_id] = spherical_mesh
                        self.loaded_metadata[mesh_id] = {
                            'name': folder_name,
                            'energy': energy,
                            'duration': duration,
                            'timestep': 0,
                            'folder': folder
                        }
                        
                    except Exception as e:
                        st.warning(f"Error loading {folder_name}: {e}")
                    
                    progress_bar.progress((i + 1) / len(folders))
                
                progress_bar.empty()
                
                if self.loaded_meshes:
                    st.session_state.data_loaded = True
                    st.session_state.loaded_simulations = {
                        mesh_id: metadata['name'] 
                        for mesh_id, metadata in self.loaded_metadata.items()
                    }
                    
                    # Initialize attention interpolator
                    meshes = list(self.loaded_meshes.values())
                    metadata_list = list(self.loaded_metadata.values())
                    self.attention_interpolator.load_training_data(meshes, metadata_list)
                    
                    st.success(f"✅ Loaded {len(self.loaded_meshes)} simulations")
                else:
                    st.error("❌ No simulations loaded successfully")
                    
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                st.error(traceback.format_exc())
    
    def _render_main_content(self):
        """Render main content based on selected mode"""
        app_mode = st.session_state.current_mode
        
        if app_mode == "Data Explorer":
            self._render_data_explorer()
        elif app_mode == "Interpolation":
            self._render_interpolation()
        elif app_mode == "Comparative Analysis":
            self._render_comparative_analysis()
        elif app_mode == "Advanced Visualization":
            self._render_advanced_visualization()
    
    def _render_data_explorer(self):
        """Render data explorer interface"""
        st.markdown('<h2 class="sub-header">📊 Data Explorer</h2>', unsafe_allow_html=True)
        
        if not st.session_state.data_loaded:
            self._show_data_not_loaded()
            return
        
        # Simulation selection
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            simulation_options = list(st.session_state.loaded_simulations.items())
            selected_option = st.selectbox(
                "Select Simulation",
                options=[name for _, name in simulation_options],
                format_func=lambda x: x
            )
            
            # Find selected mesh ID
            selected_mesh_id = None
            for mesh_id, name in simulation_options:
                if name == selected_option:
                    selected_mesh_id = mesh_id
                    break
        
        if selected_mesh_id is None:
            st.warning("No simulation selected")
            return
        
        mesh = self.loaded_meshes[selected_mesh_id]
        metadata = self.loaded_metadata[selected_mesh_id]
        
        with col2:
            st.metric("Energy", f"{metadata['energy']:.2f} mJ")
        with col3:
            st.metric("Duration", f"{metadata['duration']:.2f} ns")
        
        # Field selection
        available_fields = list(mesh.fields.keys())
        if not available_fields:
            st.warning("No fields available in this simulation")
            return
        
        selected_field = st.selectbox("Select Field", available_fields)
        
        # Visualization
        config = {
            'colormap': st.session_state.selected_colormap,
            'opacity': st.session_state.visualization_config['opacity'],
            'show_wireframe': st.session_state.visualization_config['show_wireframe'],
            'show_points': st.session_state.visualization_config['show_points'],
            'title': f"{metadata['name']} - {selected_field}"
        }
        
        fig = self.visualization_engine.create_spherical_visualization(mesh, selected_field, config)
        st.plotly_chart(fig, use_container_width=True)
        
        # Field statistics
        stats_fig = self.visualization_engine.create_field_statistics_plot(mesh, selected_field)
        if stats_fig:
            st.plotly_chart(stats_fig, use_container_width=True)
        
        # Mesh information
        with st.expander("📐 Mesh Information", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Number of Points", len(mesh.points))
            
            with col2:
                if mesh.triangles is not None:
                    st.metric("Number of Triangles", len(mesh.triangles))
                else:
                    st.metric("Number of Triangles", 0)
            
            with col3:
                if hasattr(mesh, 'radial_distances'):
                    avg_radius = np.mean(mesh.radial_distances)
                    st.metric("Average Radius", f"{avg_radius:.4f}")
    
    def _render_interpolation(self):
        """Render interpolation interface"""
        st.markdown('<h2 class="sub-header">🔮 Physics-Informed Interpolation</h2>', 
                   unsafe_allow_html=True)
        
        if not st.session_state.data_loaded:
            self._show_data_not_loaded()
            return
        
        st.markdown("""
        <div class="info-card">
        <h3>🧠 Attention-Based Spherical Interpolation</h3>
        <p>This system uses multi-head attention with spatial locality regulation to interpolate 
        field values on spherical geometry. The interpolation preserves spherical topology 
        while applying physics-informed scaling based on energy and duration parameters.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Query parameters
        col1, col2 = st.columns(2)
        
        with col1:
            energy_query = st.number_input(
                "Target Energy (mJ)",
                min_value=0.1,
                max_value=100.0,
                value=5.25,
                step=0.1,
                key="interp_energy"
            )
        
        with col2:
            duration_query = st.number_input(
                "Target Duration (ns)",
                min_value=0.1,
                max_value=50.0,
                value=2.90,
                step=0.1,
                key="interp_duration"
            )
        
        # Source simulation selection
        available_simulations = list(st.session_state.loaded_simulations.items())
        if not available_simulations:
            st.warning("No source simulations available")
            return
        
        source_option = st.selectbox(
            "Source Simulation (for geometry)",
            options=[name for _, name in available_simulations],
            key="interp_source"
        )
        
        # Find source mesh
        source_mesh_id = None
        for mesh_id, name in available_simulations:
            if name == source_option:
                source_mesh_id = mesh_id
                break
        
        if source_mesh_id is None:
            st.warning("Source simulation not found")
            return
        
        source_mesh = self.loaded_meshes[source_mesh_id]
        
        # Field selection
        available_fields = list(source_mesh.fields.keys())
        if not available_fields:
            st.warning("No fields available in source simulation")
            return
        
        selected_field = st.selectbox("Field to Interpolate", available_fields, key="interp_field")
        
        # Interpolation parameters
        with st.expander("⚙️ Interpolation Parameters", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                interpolation_method = st.selectbox(
                    "Interpolation Method",
                    ["attention_weighted", "attention_guided_rbf"],
                    key="interp_method"
                )
            
            with col2:
                resolution = st.slider(
                    "Mesh Resolution",
                    min_value=50,
                    max_value=500,
                    value=100,
                    step=50,
                    key="interp_resolution"
                )
        
        if st.button("🚀 Generate Interpolated Solution", use_container_width=True):
            with st.spinner("Generating interpolated solution..."):
                try:
                    # Generate interpolated solution ID
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    solution_id = f"interp_E{energy_query:.2f}_D{duration_query:.2f}_{timestamp}"
                    
                    # Create target mesh (resampled source geometry)
                    target_mesh = source_mesh.resample_to_uniform_sphere(resolution=resolution)
                    
                    # Use attention interpolator
                    interpolated_field, attention_weights = self.attention_interpolator.interpolate_field(
                        energy_query, duration_query, target_mesh, selected_field, interpolation_method
                    )
                    
                    # Add field to target mesh
                    target_mesh.add_field(selected_field, interpolated_field, "scalar")
                    
                    # Create interpolated solution
                    self.interpolation_manager.interpolated_meshes[solution_id] = {
                        'id': solution_id,
                        'mesh': target_mesh,
                        'source_field': selected_field,
                        'energy': energy_query,
                        'duration': duration_query,
                        'method': interpolation_method,
                        'resolution': resolution,
                        'created_at': timestamp,
                        'is_interpolated': True
                    }
                    
                    st.success(f"✅ Interpolated solution created: {solution_id}")
                    
                    # Display results
                    self._display_interpolation_results(
                        solution_id, source_mesh, target_mesh, 
                        selected_field, attention_weights
                    )
                    
                except Exception as e:
                    st.error(f"Error generating interpolated solution: {str(e)}")
                    st.error(traceback.format_exc())
    
    def _display_interpolation_results(self, solution_id, source_mesh, target_mesh, field_name, attention_weights):
        """Display interpolation results"""
        # Visualization tabs
        tab1, tab2, tab3 = st.tabs(["📊 Interpolated Field", "🔄 Comparison", "🧠 Attention"])
        
        with tab1:
            # Show interpolated field
            config = {
                'colormap': st.session_state.selected_colormap,
                'opacity': st.session_state.visualization_config['opacity'],
                'title': f"Interpolated {field_name} (E={target_mesh.energy}, D={target_mesh.duration})"
            }
            
            fig = self.visualization_engine.create_spherical_visualization(target_mesh, field_name, config)
            st.plotly_chart(fig, use_container_width=True)
            
            # Field statistics
            stats = target_mesh.compute_field_statistics(field_name)
            if stats:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean", f"{stats['mean']:.3f}")
                with col2:
                    st.metric("Std Dev", f"{stats['std']:.3f}")
                with col3:
                    st.metric("Min", f"{stats['min']:.3f}")
                with col4:
                    st.metric("Max", f"{stats['max']:.3f}")
        
        with tab2:
            # Comparison visualization
            if field_name in source_mesh.fields:
                comparison_fig = self.visualization_engine.create_comparison_visualization(
                    source_mesh, target_mesh, field_name,
                    title1="Original", title2="Interpolated"
                )
                st.plotly_chart(comparison_fig, use_container_width=True)
            else:
                st.warning(f"Field '{field_name}' not found in source mesh for comparison")
        
        with tab3:
            # Attention visualization
            if attention_weights is not None and len(attention_weights) > 0:
                attention_fig = self.attention_interpolator.visualize_attention(attention_weights)
                if attention_fig:
                    st.plotly_chart(attention_fig, use_container_width=True)
                
                # Show top attention sources
                top_indices = np.argsort(attention_weights)[-3:][::-1]
                st.markdown("**Top 3 Attention Sources:**")
                
                for i, idx in enumerate(top_indices):
                    if idx < len(self.attention_interpolator.source_metadata):
                        metadata = self.attention_interpolator.source_metadata[idx]
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(f"Source {i+1} Energy", f"{metadata.get('energy', 0):.2f} mJ")
                        with col2:
                            st.metric(f"Source {i+1} Duration", f"{metadata.get('duration', 0):.2f} ns")
                        with col3:
                            st.metric(f"Attention Weight", f"{attention_weights[idx]:.4f}")
    
    def _render_comparative_analysis(self):
        """Render comparative analysis interface"""
        st.markdown('<h2 class="sub-header">📈 Comparative Analysis</h2>', unsafe_allow_html=True)
        
        if not st.session_state.data_loaded and len(self.interpolation_manager.interpolated_meshes) == 0:
            self._show_data_not_loaded()
            return
        
        # Collect all available solutions
        all_solutions = {}
        
        # Add original simulations
        for mesh_id, mesh in self.loaded_meshes.items():
            metadata = self.loaded_metadata.get(mesh_id, {})
            all_solutions[mesh_id] = {
                'type': 'original',
                'mesh': mesh,
                'metadata': metadata,
                'name': metadata.get('name', 'Unknown'),
                'energy': metadata.get('energy', 0),
                'duration': metadata.get('duration', 0)
            }
        
        # Add interpolated solutions
        for solution_id, solution in self.interpolation_manager.interpolated_meshes.items():
            all_solutions[solution_id] = {
                'type': 'interpolated',
                'mesh': solution['mesh'],
                'metadata': solution,
                'name': solution_id,
                'energy': solution['energy'],
                'duration': solution['duration']
            }
        
        if not all_solutions:
            st.warning("No solutions available for comparison")
            return
        
        # Solution selection
        solution_options = [(sid, f"{data['name']} ({data['type']})") 
                           for sid, data in all_solutions.items()]
        
        selected_solutions = st.multiselect(
            "Select solutions for comparison",
            options=[name for _, name in solution_options],
            default=[name for _, name in solution_options[:min(3, len(solution_options))]],
            key="comparison_solutions"
        )
        
        if len(selected_solutions) < 2:
            st.info("Select at least 2 solutions for comparison")
            return
        
        # Find common field
        common_fields = set()
        for sol_name in selected_solutions:
            for sol_id, disp_name in solution_options:
                if disp_name == sol_name:
                    mesh = all_solutions[sol_id]['mesh']
                    common_fields.update(mesh.fields.keys())
                    break
        
        if not common_fields:
            st.warning("No common fields found in selected solutions")
            return
        
        selected_field = st.selectbox("Select field for comparison", list(common_fields), key="comparison_field")
        
        # Create comparison visualization
        if st.button("📊 Generate Comparison", use_container_width=True):
            with st.spinner("Generating comparison..."):
                # Collect meshes and names for selected solutions
                comparison_meshes = []
                comparison_names = []
                
                for sol_name in selected_solutions:
                    for sol_id, disp_name in solution_options:
                        if disp_name == sol_name:
                            solution_data = all_solutions[sol_id]
                            if selected_field in solution_data['mesh'].fields:
                                comparison_meshes.append(solution_data['mesh'])
                                name_suffix = "(Interpolated)" if solution_data['type'] == 'interpolated' else ""
                                comparison_names.append(f"{solution_data['name']} {name_suffix}")
                            break
                
                if len(comparison_meshes) < 2:
                    st.warning("Selected field not available in all solutions")
                    return
                
                # Create grid of visualizations
                n_plots = len(comparison_meshes)
                cols = min(3, n_plots)
                rows = (n_plots + cols - 1) // cols
                
                fig = make_subplots(
                    rows=rows, cols=cols,
                    specs=[[{'type': 'surface'} for _ in range(cols)] for _ in range(rows)],
                    subplot_titles=comparison_names,
                    vertical_spacing=0.1,
                    horizontal_spacing=0.05
                )
                
                # Normalize colorscale across all plots
                all_values = []
                for mesh in comparison_meshes:
                    if selected_field in mesh.fields:
                        field_values = mesh.fields[selected_field]
                        if field_values.ndim == 2:
                            field_values = np.linalg.norm(field_values, axis=1)
                        all_values.extend(field_values.tolist())
                
                vmin, vmax = np.nanmin(all_values), np.nanmax(all_values)
                
                # Add each mesh to subplot
                for idx, (mesh, name) in enumerate(zip(comparison_meshes, comparison_names)):
                    row = idx // cols + 1
                    col = idx % cols + 1
                    
                    field_values = mesh.fields[selected_field]
                    if field_values.ndim == 2:
                        display_values = np.linalg.norm(field_values, axis=1)
                    else:
                        display_values = field_values
                    
                    if mesh.triangles is not None and len(mesh.triangles) > 0:
                        fig.add_trace(
                            go.Mesh3d(
                                x=mesh.points[:, 0],
                                y=mesh.points[:, 1],
                                z=mesh.points[:, 2],
                                i=mesh.triangles[:, 0],
                                j=mesh.triangles[:, 1],
                                k=mesh.triangles[:, 2],
                                intensity=display_values,
                                colorscale=st.session_state.selected_colormap,
                                intensitymode='vertex',
                                opacity=st.session_state.visualization_config['opacity'],
                                lighting=dict(ambient=0.8, diffuse=0.8, specular=0.5, roughness=0.5),
                                flatshading=False,
                                showscale=(idx == 0),
                                cmin=vmin,
                                cmax=vmax,
                                colorbar=dict(
                                    title=selected_field,
                                    thickness=20,
                                    len=0.75,
                                    x=1.02 if col == cols else None
                                )
                            ),
                            row=row, col=col
                        )
                    else:
                        fig.add_trace(
                            go.Scatter3d(
                                x=mesh.points[:, 0],
                                y=mesh.points[:, 1],
                                z=mesh.points[:, 2],
                                mode='markers',
                                marker=dict(
                                    size=3,
                                    color=display_values,
                                    colorscale=st.session_state.selected_colormap,
                                    opacity=st.session_state.visualization_config['opacity'],
                                    cmin=vmin,
                                    cmax=vmax,
                                    showscale=(idx == 0)
                                )
                            ),
                            row=row, col=col
                        )
                
                fig.update_layout(
                    height=400 * rows,
                    title_text=f"Comparison: {selected_field}",
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add statistics table
                st.markdown("### 📊 Field Statistics Comparison")
                
                stats_data = []
                for mesh, name in zip(comparison_meshes, comparison_names):
                    stats = mesh.compute_field_statistics(selected_field)
                    if stats:
                        stats_data.append({
                            'Solution': name,
                            'Mean': f"{stats['mean']:.3f}",
                            'Std Dev': f"{stats['std']:.3f}",
                            'Min': f"{stats['min']:.3f}",
                            'Max': f"{stats['max']:.3f}",
                            'Range': f"{stats['max'] - stats['min']:.3f}"
                        })
                
                if stats_data:
                    df_stats = pd.DataFrame(stats_data)
                    st.dataframe(df_stats, use_container_width=True)
    
    def _render_advanced_visualization(self):
        """Render advanced visualization interface"""
        st.markdown('<h2 class="sub-header">🎨 Advanced Visualization</h2>', unsafe_allow_html=True)
        
        if not st.session_state.data_loaded and len(self.interpolation_manager.interpolated_meshes) == 0:
            self._show_data_not_loaded()
            return
        
        # Visualization type selection
        viz_type = st.selectbox(
            "Visualization Type",
            ["Radial Cross-Section", "Field Evolution", "3D Parameter Space", "Spherical Harmonics"],
            key="adv_viz_type"
        )
        
        if viz_type == "Radial Cross-Section":
            self._render_radial_cross_section()
        elif viz_type == "Field Evolution":
            self._render_field_evolution()
        elif viz_type == "3D Parameter Space":
            self._render_3d_parameter_space()
        elif viz_type == "Spherical Harmonics":
            self._render_spherical_harmonics()
    
    def _render_radial_cross_section(self):
        """Render radial cross-section visualization"""
        # Get all available solutions
        all_meshes = {}
        
        # Original meshes
        for mesh_id, mesh in self.loaded_meshes.items():
            metadata = self.loaded_metadata.get(mesh_id, {})
            all_meshes[metadata.get('name', mesh_id)] = {
                'mesh': mesh,
                'type': 'original',
                'energy': metadata.get('energy', 0),
                'duration': metadata.get('duration', 0)
            }
        
        # Interpolated meshes
        for solution_id, solution in self.interpolation_manager.interpolated_meshes.items():
            all_meshes[solution_id] = {
                'mesh': solution['mesh'],
                'type': 'interpolated',
                'energy': solution['energy'],
                'duration': solution['duration']
            }
        
        if not all_meshes:
            st.warning("No meshes available for visualization")
            return
        
        # Mesh selection
        selected_mesh_name = st.selectbox(
            "Select mesh for cross-section",
            list(all_meshes.keys()),
            key="cross_section_mesh"
        )
        
        selected_mesh_data = all_meshes[selected_mesh_name]
        mesh = selected_mesh_data['mesh']
        
        # Field selection
        available_fields = list(mesh.fields.keys())
        if not available_fields:
            st.warning("No fields available in selected mesh")
            return
        
        selected_field = st.selectbox("Select field", available_fields, key="cross_section_field")
        
        # Create radial cross-section plot
        if hasattr(mesh, 'radial_distances') and selected_field in mesh.fields:
            radial_distances = mesh.radial_distances
            field_values = mesh.fields[selected_field]
            
            if field_values.ndim == 2:
                field_values = np.linalg.norm(field_values, axis=1)
            
            # Sort by radial distance
            sort_idx = np.argsort(radial_distances)
            sorted_radial = radial_distances[sort_idx]
            sorted_values = field_values[sort_idx]
            
            # Create plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=sorted_radial,
                y=sorted_values,
                mode='markers',
                marker=dict(
                    size=4,
                    color=sorted_values,
                    colorscale=st.session_state.selected_colormap,
                    opacity=0.7,
                    showscale=True,
                    colorbar=dict(title=selected_field, thickness=20)
                ),
                name="Radial Profile"
            ))
            
            # Add moving average
            window_size = min(50, len(sorted_radial) // 10)
            if window_size > 1:
                moving_avg = np.convolve(sorted_values, np.ones(window_size)/window_size, mode='valid')
                valid_radial = sorted_radial[window_size-1:]
                
                fig.add_trace(go.Scatter(
                    x=valid_radial,
                    y=moving_avg,
                    mode='lines',
                    line=dict(color='red', width=3),
                    name="Moving Average"
                ))
            
            fig.update_layout(
                title=f"Radial Cross-Section: {selected_field}",
                xaxis_title="Radial Distance",
                yaxis_title=f"{selected_field} Value",
                height=500,
                showlegend=True,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_field_evolution(self):
        """Render field evolution visualization"""
        st.info("⏳ Field evolution visualization is under development")
        st.markdown("""
        This feature will visualize how fields evolve over time in the simulations.
        
        **Planned capabilities:**
        - Time series animation of field evolution
        - Comparison of field evolution across different simulations
        - Extraction of key temporal features
        """)
    
    def _render_3d_parameter_space(self):
        """Render 3D parameter space visualization"""
        # Collect all solutions
        solutions_data = []
        
        # Original simulations
        for mesh_id, metadata in self.loaded_metadata.items():
            if mesh_id in self.loaded_meshes:
                mesh = self.loaded_meshes[mesh_id]
                if mesh.fields:
                    # Use first available field
                    field_name = list(mesh.fields.keys())[0]
                    field_values = mesh.fields[field_name]
                    
                    if field_values.ndim == 2:
                        field_values = np.linalg.norm(field_values, axis=1)
                    
                    solutions_data.append({
                        'name': metadata['name'],
                        'energy': metadata['energy'],
                        'duration': metadata['duration'],
                        'type': 'original',
                        'mean_value': np.mean(field_values) if len(field_values) > 0 else 0
                    })
        
        # Interpolated solutions
        for solution_id, solution in self.interpolation_manager.interpolated_meshes.items():
            mesh = solution['mesh']
            if mesh.fields:
                field_name = list(mesh.fields.keys())[0]
                field_values = mesh.fields[field_name]
                
                if field_values.ndim == 2:
                    field_values = np.linalg.norm(field_values, axis=1)
                
                solutions_data.append({
                    'name': solution_id,
                    'energy': solution['energy'],
                    'duration': solution['duration'],
                    'type': 'interpolated',
                    'mean_value': np.mean(field_values) if len(field_values) > 0 else 0
                })
        
        if len(solutions_data) < 3:
            st.warning("Need at least 3 solutions for parameter space visualization")
            return
        
        # Create 3D scatter plot
        fig = go.Figure()
        
        # Separate original and interpolated
        original_data = [d for d in solutions_data if d['type'] == 'original']
        interpolated_data = [d for d in solutions_data if d['type'] == 'interpolated']
        
        # Plot original simulations
        if original_data:
            energies = [d['energy'] for d in original_data]
            durations = [d['duration'] for d in original_data]
            mean_values = [d['mean_value'] for d in original_data]
            names = [d['name'] for d in original_data]
            
            fig.add_trace(go.Scatter3d(
                x=energies,
                y=durations,
                z=mean_values,
                mode='markers',
                marker=dict(
                    size=8,
                    color='blue',
                    opacity=0.8,
                    symbol='circle'
                ),
                name='Original Simulations',
                text=names,
                hovertemplate='<b>%{text}</b><br>Energy: %{x:.2f} mJ<br>Duration: %{y:.2f} ns<br>Mean Value: %{z:.3f}<extra></extra>'
            ))
        
        # Plot interpolated solutions
        if interpolated_data:
            energies = [d['energy'] for d in interpolated_data]
            durations = [d['duration'] for d in interpolated_data]
            mean_values = [d['mean_value'] for d in interpolated_data]
            names = [d['name'] for d in interpolated_data]
            
            fig.add_trace(go.Scatter3d(
                x=energies,
                y=durations,
                z=mean_values,
                mode='markers',
                marker=dict(
                    size=10,
                    color='magenta',
                    opacity=0.9,
                    symbol='diamond'
                ),
                name='Interpolated Solutions',
                text=names,
                hovertemplate='<b>%{text}</b><br>Energy: %{x:.2f} mJ<br>Duration: %{y:.2f} ns<br>Mean Value: %{z:.3f}<extra></extra>'
            ))
        
        fig.update_layout(
            title="3D Parameter Space",
            scene=dict(
                xaxis_title="Energy (mJ)",
                yaxis_title="Duration (ns)",
                zaxis_title="Mean Field Value"
            ),
            height=600,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_spherical_harmonics(self):
        """Render spherical harmonics analysis"""
        st.info("🌀 Spherical harmonics analysis is under development")
        st.markdown("""
        This feature will perform spherical harmonics decomposition of fields on the sphere.
        
        **Planned capabilities:**
        - Decomposition of scalar fields into spherical harmonics
        - Visualization of harmonic components
        - Frequency analysis of field variations
        """)
    
    def _show_data_not_loaded(self):
        """Show data not loaded message"""
        st.markdown("""
        <div class="warning-card">
        <h3>⚠️ No Data Loaded</h3>
        <p>Please load simulations first using the "Load Simulation Data" button in the sidebar.</p>
        <p>Ensure your data follows this structure:</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📁 Expected Directory Structure", expanded=True):
            st.code("""
fea_solutions/
├── q0p5mJ-delta4p2ns/        # Energy: 0.5 mJ, Duration: 4.2 ns
│   ├── a_t0001.vtu           # Timestep 1
│   ├── a_t0002.vtu           # Timestep 2
│   └── ...
├── q1p0mJ-delta2p0ns/        # Energy: 1.0 mJ, Duration: 2.0 ns
│   ├── a_t0001.vtu
│   ├── a_t0002.vtu
│   └── ...
└── q2p0mJ-delta1p0ns/        # Energy: 2.0 mJ, Duration: 1.0 ns
    ├── a_t0001.vtu
    └── ...
            """)

# =============================================
# MAIN ENTRY POINT
# =============================================
def main():
    """Main application entry point"""
    try:
        app = SphericalFEAApplication()
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.error(traceback.format_exc())

if __name__ == "__main__":
    main()
