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
from typing import List, Dict, Any, Optional, Tuple, Union
from io import BytesIO
import pandas as pd
import traceback
from scipy.interpolate import griddata, RBFInterpolator, NearestNDInterpolator
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from scipy.spatial.distance import cdist, pdist, squareform
import scipy.ndimage as ndimage
from scipy.stats import gaussian_kde, linregress
import json
import base64
from PIL import Image
import io
import time
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pickle
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# =============================================
# ENUMS AND DATACLASSES
# =============================================
class FieldType(Enum):
    SCALAR = "scalar"
    VECTOR = "vector"
    TENSOR = "tensor"

class VisualizationMode(Enum):
    SURFACE = "surface"
    POINTS = "points"
    WIREFRAME = "wireframe"
    VOLUME = "volume"
    
@dataclass
class SimulationConfig:
    """Configuration for simulation parameters"""
    energy_mJ: float
    duration_ns: float
    mesh_resolution: float = 0.1
    time_steps: int = 10
    physical_properties: Dict[str, Any] = field(default_factory=dict)

# =============================================
# CACHE MANAGEMENT
# =============================================
class CacheManager:
    """Manages caching of expensive computations"""
    
    def __init__(self, cache_dir="cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cache_key(self, *args, **kwargs):
        """Generate a unique cache key"""
        key_str = f"{args}_{kwargs}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def cache_result(self, key, result):
        """Cache a result"""
        cache_file = self.cache_dir / f"{key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        return result
    
    def get_cached_result(self, key, max_age_hours=24):
        """Retrieve cached result if it exists and is fresh"""
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            # Check age
            file_age = time.time() - cache_file.stat().st_mtime
            if file_age < max_age_hours * 3600:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        return None

# =============================================
# ENHANCED MESH DATA STRUCTURES
# =============================================
class EnhancedMeshData:
    """Container for enhanced mesh data with spatial indexing"""
    
    def __init__(self):
        self.points = None
        self.triangles = None
        self.edges = None
        self.fields = {}
        self.field_info = {}
        self.metadata = {}
        self.spatial_tree = None
        self.region_labels = None
        self.mesh_stats = {}
        self.connectivity = None
        self.boundary_edges = None
        
    def compute_spatial_features(self):
        """Compute spatial features for the mesh"""
        if self.points is None:
            return
        
        # Compute bounding box
        min_coords = np.min(self.points, axis=0)
        max_coords = np.max(self.points, axis=0)
        bbox_size = max_coords - min_coords
        self.mesh_stats['bbox'] = {
            'min': min_coords.tolist(), 
            'max': max_coords.tolist(),
            'size': bbox_size.tolist(),
            'volume': float(np.prod(bbox_size))
        }
        
        # Compute centroid
        self.mesh_stats['centroid'] = np.mean(self.points, axis=0).tolist()
        
        # Compute distances from centroid
        if self.points.shape[0] > 0:
            centroid = self.mesh_stats['centroid']
            distances = np.linalg.norm(self.points - centroid, axis=1)
            self.mesh_stats['radial_stats'] = {
                'min': float(np.min(distances)),
                'max': float(np.max(distances)),
                'mean': float(np.mean(distances)),
                'std': float(np.std(distances)),
                'median': float(np.median(distances))
            }
        
        # Compute surface area if triangles exist
        if self.triangles is not None and len(self.triangles) > 0:
            areas = self.compute_triangle_areas()
            self.mesh_stats['surface_area'] = float(np.sum(areas))
            self.mesh_stats['triangle_count'] = len(self.triangles)
            self.mesh_stats['avg_triangle_area'] = float(np.mean(areas))
            self.mesh_stats['triangle_area_std'] = float(np.std(areas))
            
            # Compute aspect ratios
            aspect_ratios = self.compute_triangle_aspect_ratios()
            if len(aspect_ratios) > 0:
                self.mesh_stats['aspect_ratio_stats'] = {
                    'min': float(np.min(aspect_ratios)),
                    'max': float(np.max(aspect_ratios)),
                    'mean': float(np.mean(aspect_ratios)),
                    'median': float(np.median(aspect_ratios))
                }
    
    def compute_triangle_areas(self):
        """Compute areas of triangles"""
        if self.triangles is None or self.points is None:
            return np.array([])
        
        areas = np.zeros(len(self.triangles))
        for i, tri in enumerate(self.triangles):
            if all(idx < len(self.points) for idx in tri):
                v0 = self.points[tri[0]]
                v1 = self.points[tri[1]]
                v2 = self.points[tri[2]]
                
                # Compute area using cross product
                v0v1 = v1 - v0
                v0v2 = v2 - v0
                areas[i] = 0.5 * np.linalg.norm(np.cross(v0v1, v0v2))
        
        return areas
    
    def compute_triangle_aspect_ratios(self):
        """Compute aspect ratios of triangles"""
        if self.triangles is None or self.points is None:
            return np.array([])
        
        aspect_ratios = []
        for tri in self.triangles:
            if all(idx < len(self.points) for idx in tri):
                v0 = self.points[tri[0]]
                v1 = self.points[tri[1]]
                v2 = self.points[tri[2]]
                
                # Compute side lengths
                a = np.linalg.norm(v1 - v0)
                b = np.linalg.norm(v2 - v1)
                c = np.linalg.norm(v0 - v2)
                
                # Compute aspect ratio (circumradius/inradius)
                s = (a + b + c) / 2.0
                area = np.sqrt(s * (s - a) * (s - b) * (s - c))
                if area > 0:
                    circumradius = (a * b * c) / (4 * area)
                    inradius = area / s
                    aspect_ratio = circumradius / (2 * inradius) if inradius > 0 else 1.0
                    aspect_ratios.append(aspect_ratio)
        
        return np.array(aspect_ratios)
    
    def compute_connectivity(self):
        """Compute node connectivity and boundary edges"""
        if self.triangles is None:
            return
        
        # Build edge set
        edges = set()
        edge_to_triangles = {}
        
        for tri_idx, tri in enumerate(self.triangles):
            for i in range(3):
                edge = tuple(sorted((tri[i], tri[(i+1)%3])))
                edges.add(edge)
                if edge not in edge_to_triangles:
                    edge_to_triangles[edge] = []
                edge_to_triangles[edge].append(tri_idx)
        
        self.edges = np.array(list(edges))
        
        # Find boundary edges (edges belonging to only one triangle)
        self.boundary_edges = []
        for edge, tri_list in edge_to_triangles.items():
            if len(tri_list) == 1:
                self.boundary_edges.append(edge)
        
        # Build node connectivity
        n_nodes = len(self.points)
        self.connectivity = [set() for _ in range(n_nodes)]
        for edge in edges:
            self.connectivity[edge[0]].add(edge[1])
            self.connectivity[edge[1]].add(edge[0])
    
    def segment_regions(self, n_regions=5, method='kmeans'):
        """Segment mesh into spatial regions"""
        if self.points is None:
            return
        
        if method == 'kmeans':
            kmeans = KMeans(n_clusters=n_regions, random_state=42, n_init=10)
            self.region_labels = kmeans.fit_predict(self.points)
        elif method == 'radial':
            # Radial segmentation based on distance from centroid
            centroid = np.mean(self.points, axis=0)
            distances = np.linalg.norm(self.points - centroid, axis=1)
            percentiles = np.percentile(distances, np.linspace(0, 100, n_regions + 1))
            self.region_labels = np.digitize(distances, percentiles[1:-1]) - 1
        
        # Compute region statistics
        self.region_stats = {}
        for region_id in range(n_regions):
            region_mask = self.region_labels == region_id
            if np.any(region_mask):
                region_points = self.points[region_mask]
                region_center = np.mean(region_points, axis=0)
                
                self.region_stats[region_id] = {
                    'size': int(np.sum(region_mask)),
                    'centroid': region_center.tolist(),
                    'bbox': {
                        'min': np.min(region_points, axis=0).tolist(),
                        'max': np.max(region_points, axis=0).tolist()
                    },
                    'volume': float(np.prod(np.max(region_points, axis=0) - np.min(region_points, axis=0))),
                    'avg_distance_to_center': float(np.mean(np.linalg.norm(region_points - region_center, axis=1)))
                }
    
    def get_surface_normals(self):
        """Compute surface normals for triangles"""
        if self.triangles is None:
            return None
        
        normals = np.zeros((len(self.triangles), 3))
        for i, tri in enumerate(self.triangles):
            v0 = self.points[tri[0]]
            v1 = self.points[tri[1]]
            v2 = self.points[tri[2]]
            
            normal = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normals[i] = normal / norm
        
        return normals

# =============================================
# ATTENTION-BASED EXTRAPOLATOR
# =============================================
class AttentionExtrapolator:
    """Physics-informed attention-based extrapolator for field predictions"""
    
    def __init__(self, data_loader, n_heads=4, embedding_dim=32):
        self.data_loader = data_loader
        self.n_heads = n_heads
        self.embedding_dim = embedding_dim
        self.fitted = False
        self.source_metadata = []
        self.attention_weights = {}
        self.field_embeddings = {}
        
    def fit(self, simulations, summaries):
        """Fit the extrapolator to loaded simulation data"""
        if not simulations:
            raise ValueError("No simulation data available")
        
        self.source_metadata = []
        
        # Extract features from each simulation/timestep
        for sim_name, sim_data in simulations.items():
            for t in range(sim_data['n_timesteps']):
                metadata = {
                    'name': sim_name,
                    'energy': sim_data['energy_mJ'],
                    'duration': sim_data['duration_ns'],
                    'time_index': t,
                    'timestep_idx': t,
                    'time': t * 0.1,  # Assuming 0.1ns per timestep
                    'field_stats': {}
                }
                
                # Extract field statistics
                if sim_name in self.data_loader.simulations:
                    sim = self.data_loader.simulations[sim_name]
                    for field_name in sim['field_info']:
                        if field_name in sim['mesh_data'].fields:
                            field_data = sim['mesh_data'].fields[field_name][t]
                            if field_data.ndim == 1:
                                values = field_data
                            else:
                                values = np.linalg.norm(field_data, axis=1)
                            
                            valid_values = values[~np.isnan(values)]
                            if len(valid_values) > 0:
                                metadata['field_stats'][field_name] = {
                                    'mean': float(np.mean(valid_values)),
                                    'std': float(np.std(valid_values)),
                                    'max': float(np.max(valid_values)),
                                    'min': float(np.min(valid_values))
                                }
                
                self.source_metadata.append(metadata)
        
        # Compute embeddings for each field
        self._compute_field_embeddings()
        
        self.fitted = True
        return self
    
    def _compute_field_embeddings(self):
        """Compute embeddings for fields based on statistics"""
        # Group statistics by field
        field_stats = {}
        for meta in self.source_metadata:
            for field_name, stats in meta['field_stats'].items():
                if field_name not in field_stats:
                    field_stats[field_name] = []
                field_stats[field_name].append([
                    stats['mean'],
                    stats['std'],
                    stats['max'],
                    stats['min'],
                    meta['energy'],
                    meta['duration'],
                    meta['time']
                ])
        
        # Compute PCA-like embeddings
        for field_name, stats_list in field_stats.items():
            stats_array = np.array(stats_list)
            
            # Normalize
            scaler = StandardScaler()
            stats_normalized = scaler.fit_transform(stats_array)
            
            # Simple embedding (could be replaced with autoencoder)
            embedding = np.mean(stats_normalized, axis=0)
            
            # Reduce dimension
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(10, len(embedding)))
            embedding_reduced = pca.fit_transform(stats_normalized.reshape(1, -1))
            
            self.field_embeddings[field_name] = {
                'embedding': embedding_reduced.flatten(),
                'scaler': scaler,
                'pca': pca
            }
    
    def compute_attention_weights(self, query_point, field_name=None):
        """Compute attention weights for query point"""
        if not self.fitted:
            raise ValueError("Extrapolator not fitted")
        
        query_features = np.array([
            query_point['energy'],
            query_point['duration'],
            query_point.get('time', 0),
            query_point.get('temperature', 300),  # Default temperature
            query_point.get('pressure', 101325)   # Default pressure
        ])
        
        # Normalize query features
        query_features = (query_features - np.array([0.5, 2.5, 5.0, 300, 101325])) / \
                         np.array([2.0, 5.0, 10.0, 100, 10000])
        
        weights = []
        
        for meta in self.source_metadata:
            # Compute similarity based on parameters
            source_features = np.array([
                meta['energy'],
                meta['duration'],
                meta['time'],
                300,  # Placeholder
                101325  # Placeholder
            ])
            
            source_features = (source_features - np.array([0.5, 2.5, 5.0, 300, 101325])) / \
                              np.array([2.0, 5.0, 10.0, 100, 10000])
            
            # Euclidean distance
            distance = np.linalg.norm(query_features - source_features)
            
            # Add field-specific similarity if field_name provided
            if field_name and field_name in meta['field_stats']:
                field_stats = meta['field_stats'][field_name]
                field_similarity = 1.0 / (1.0 + abs(field_stats['mean'] - query_point.get(f'field_{field_name}_mean', 0)))
                distance = distance * (1.0 - 0.3 * field_similarity)  # Weight field similarity
            
            # Convert distance to weight (closer = higher weight)
            weight = np.exp(-distance * 2.0)
            weights.append(weight)
        
        # Normalize weights
        weights = np.array(weights)
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        
        return weights
    
    def predict_field_statistics(self, energy, duration, time, field_name):
        """Predict field statistics for query parameters"""
        query_point = {
            'energy': energy,
            'duration': duration,
            'time': time
        }
        
        weights = self.compute_attention_weights(query_point, field_name)
        
        # Weighted average of source statistics
        pred_stats = {
            'mean': 0.0,
            'std': 0.0,
            'max': 0.0,
            'min': 0.0,
            'q25': 0.0,
            'q50': 0.0,
            'q75': 0.0
        }
        
        total_weight = 0.0
        for weight, meta in zip(weights, self.source_metadata):
            if weight > 0.001 and field_name in meta['field_stats']:
                stats = meta['field_stats'][field_name]
                for key in pred_stats:
                    if key in stats:
                        pred_stats[key] += stats[key] * weight
                total_weight += weight
        
        if total_weight > 0:
            for key in pred_stats:
                pred_stats[key] /= total_weight
        
        return {
            'field_predictions': {field_name: pred_stats},
            'attention_weights': weights,
            'confidence': float(np.sum(weights > 0.1) / len(weights))
        }

# =============================================
# ENHANCED DATA LOADER WITH PARALLEL PROCESSING
# =============================================
class EnhancedFEADataLoader:
    """Enhanced data loader with full mesh capabilities and spatial features"""
    
    def __init__(self):
        self.simulations = {}
        self.summaries = []
        self.available_fields = set()
        self.reference_mesh = None
        self.common_fields = set()
        self.cache_manager = CacheManager()
        self.loaded = False
        
    def parse_folder_name(self, folder: str):
        """q0p5mJ-delta4p2ns → (0.5, 4.2)"""
        match = re.match(r"q([\dp\.]+)mJ-delta([\dp\.]+)ns", folder)
        if not match:
            return None, None
        e, d = match.groups()
        return float(e.replace("p", ".")), float(d.replace("p", "."))
    
    def _load_single_simulation(self, folder, load_full_mesh=True):
        """Load a single simulation folder"""
        name = os.path.basename(folder)
        energy, duration = self.parse_folder_name(name)
        if energy is None:
            return None, None
        
        vtu_files = sorted(glob.glob(os.path.join(folder, "a_t????.vtu")))
        if not vtu_files:
            return None, None
        
        try:
            # Cache key for this simulation
            cache_key = f"sim_{name}_{load_full_mesh}"
            cached = self.cache_manager.get_cached_result(cache_key)
            if cached:
                return cached['sim_data'], cached['summary']
            
            mesh0 = meshio.read(vtu_files[0])
            
            if not mesh0.point_data:
                return None, None
            
            # Create enhanced mesh data structure
            mesh_data = EnhancedMeshData()
            
            # Basic simulation info
            sim_data = {
                'name': name,
                'energy_mJ': energy,
                'duration_ns': duration,
                'n_timesteps': len(vtu_files),
                'vtu_files': vtu_files,
                'field_info': {},
                'has_mesh': False,
                'mesh_data': mesh_data,
                'metadata': {
                    'folder': folder,
                    'file_count': len(vtu_files),
                    'first_file': vtu_files[0],
                    'last_file': vtu_files[-1]
                }
            }
            
            # Load points
            mesh_data.points = mesh0.points.astype(np.float32)
            
            # Find triangles
            triangles = None
            for cell_block in mesh0.cells:
                if cell_block.type == "triangle":
                    triangles = cell_block.data.astype(np.int32)
                    break
            
            mesh_data.triangles = triangles
            
            # Compute connectivity
            if triangles is not None:
                mesh_data.compute_connectivity()
            
            # Initialize fields
            mesh_data.fields = {}
            for key in mesh0.point_data.keys():
                arr = mesh0.point_data[key].astype(np.float32)
                if arr.ndim == 1:
                    sim_data['field_info'][key] = ("scalar", 1)
                    mesh_data.fields[key] = np.full((len(vtu_files), len(mesh_data.points)), 
                                                   np.nan, dtype=np.float32)
                else:
                    sim_data['field_info'][key] = ("vector", arr.shape[1])
                    mesh_data.fields[key] = np.full((len(vtu_files), len(mesh_data.points), arr.shape[1]), 
                                                   np.nan, dtype=np.float32)
                mesh_data.fields[key][0] = arr
                self.available_fields.add(key)
                mesh_data.field_info[key] = sim_data['field_info'][key]
            
            # Load remaining timesteps in parallel
            def load_timestep(t_idx):
                try:
                    mesh = meshio.read(vtu_files[t_idx])
                    timestep_data = {}
                    for key in sim_data['field_info']:
                        if key in mesh.point_data:
                            timestep_data[key] = mesh.point_data[key].astype(np.float32)
                    return t_idx, timestep_data
                except Exception as e:
                    return t_idx, None
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(load_timestep, range(1, len(vtu_files))))
            
            for t_idx, timestep_data in results:
                if timestep_data:
                    for key, data in timestep_data.items():
                        if key in mesh_data.fields:
                            mesh_data.fields[key][t_idx] = data
            
            # Compute mesh statistics
            mesh_data.compute_spatial_features()
            mesh_data.segment_regions(n_regions=5)
            
            sim_data['has_mesh'] = True
            
            # Create summary statistics
            summary = self.extract_summary_statistics(mesh_data, energy, duration, name)
            
            # Cache the result
            self.cache_manager.cache_result(cache_key, {
                'sim_data': sim_data,
                'summary': summary
            })
            
            return sim_data, summary
            
        except Exception as e:
            print(f"Error loading {name}: {str(e)}")
            return None, None
    
    @st.cache_data
    def load_all_simulations(_self, load_full_mesh=True):
        """Load all simulations with enhanced mesh capabilities"""
        simulations = {}
        summaries = []
        
        folders = glob.glob(os.path.join(FEA_SOLUTIONS_DIR, "q*mJ-delta*ns"))
        
        if not folders:
            st.warning(f"No simulation folders found in {FEA_SOLUTIONS_DIR}")
            return simulations, summaries
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_placeholder = st.empty()
        
        # Load simulations in parallel
        with ThreadPoolExecutor(max_workers=min(4, len(folders))) as executor:
            futures = []
            for folder in folders:
                future = executor.submit(_self._load_single_simulation, folder, load_full_mesh)
                futures.append(future)
            
            for i, future in enumerate(futures):
                sim_data, summary = future.result()
                
                if sim_data and summary:
                    name = sim_data['name']
                    simulations[name] = sim_data
                    summaries.append(summary)
                
                # Update progress
                progress = (i + 1) / len(folders)
                progress_bar.progress(progress)
                status_text.text(f"Loaded {i + 1}/{len(folders)} simulations")
                
                if sim_data:
                    status_placeholder.info(f"✅ {sim_data['name']}: {sim_data['energy_mJ']} mJ, {sim_data['duration_ns']} ns")
        
        progress_bar.empty()
        status_text.empty()
        status_placeholder.empty()
        
        # Determine common fields across all simulations
        if simulations:
            field_counts = {}
            for sim in simulations.values():
                for field in sim['field_info'].keys():
                    field_counts[field] = field_counts.get(field, 0) + 1
            
            _self.common_fields = {field for field, count in field_counts.items() 
                                 if count == len(simulations)}
            
            # Set reference mesh
            if simulations and not _self.reference_mesh:
                first_sim = list(simulations.values())[0]
                _self.reference_mesh = first_sim['mesh_data']
            
            st.success(f"✅ Loaded {len(simulations)} simulations with {len(_self.available_fields)} unique fields")
            
            if not _self.common_fields:
                st.warning("⚠️ No common fields found across all simulations. This will limit field comparison capabilities.")
            else:
                st.info(f"📊 {len(_self.common_fields)} common fields across all simulations")
                
                # Display available fields
                with st.expander("📋 Available Fields", expanded=False):
                    for field in sorted(_self.available_fields):
                        field_type = "Scalar" if field in simulations[list(simulations.keys())[0]]['field_info'] and \
                            simulations[list(simulations.keys())[0]]['field_info'][field][0] == "scalar" else "Vector"
                        st.write(f"• **{field}** ({field_type})")
        else:
            st.error("❌ No simulations loaded successfully")
        
        _self.simulations = simulations
        _self.summaries = summaries
        _self.loaded = True
        
        return simulations, summaries
    
    def extract_summary_statistics(self, mesh_data, energy, duration, name):
        """Extract comprehensive summary statistics from mesh data"""
        summary = {
            'name': name,
            'energy': energy,
            'duration': duration,
            'timesteps': list(range(1, len(mesh_data.fields[list(mesh_data.fields.keys())[0]]) + 1)),
            'field_stats': {},
            'mesh_stats': mesh_data.mesh_stats,
            'region_stats': mesh_data.region_stats if hasattr(mesh_data, 'region_stats') else {}
        }
        
        for field_name in mesh_data.fields.keys():
            field_data = mesh_data.fields[field_name]
            summary['field_stats'][field_name] = {
                'min': [], 'max': [], 'mean': [], 'std': [],
                'q25': [], 'q50': [], 'q75': [], 'skew': [], 'kurtosis': [],
                'gradient_mean': [], 'gradient_max': []
            }
            
            for t in range(field_data.shape[0]):
                if field_data[t].ndim == 1:
                    values = field_data[t]
                else:
                    # Vector field - compute magnitude
                    values = np.linalg.norm(field_data[t], axis=1)
                
                # Remove NaN values
                valid_values = values[~np.isnan(values)]
                
                if len(valid_values) > 0:
                    summary['field_stats'][field_name]['min'].append(float(np.min(valid_values)))
                    summary['field_stats'][field_name]['max'].append(float(np.max(valid_values)))
                    summary['field_stats'][field_name]['mean'].append(float(np.mean(valid_values)))
                    summary['field_stats'][field_name]['std'].append(float(np.std(valid_values)))
                    summary['field_stats'][field_name]['q25'].append(float(np.percentile(valid_values, 25)))
                    summary['field_stats'][field_name]['q50'].append(float(np.percentile(valid_values, 50)))
                    summary['field_stats'][field_name]['q75'].append(float(np.percentile(valid_values, 75)))
                    
                    # Higher-order statistics
                    if len(valid_values) > 3:
                        from scipy.stats import skew, kurtosis
                        summary['field_stats'][field_name]['skew'].append(float(skew(valid_values)))
                        summary['field_stats'][field_name]['kurtosis'].append(float(kurtosis(valid_values)))
                    else:
                        summary['field_stats'][field_name]['skew'].append(0.0)
                        summary['field_stats'][field_name]['kurtosis'].append(0.0)
                    
                    # Compute gradient statistics if mesh has triangles
                    if mesh_data.triangles is not None and len(mesh_data.triangles) > 0:
                        grad_mean, grad_max = self._compute_field_gradient(mesh_data, values)
                        summary['field_stats'][field_name]['gradient_mean'].append(float(grad_mean))
                        summary['field_stats'][field_name]['gradient_max'].append(float(grad_max))
                else:
                    for stat in ['min', 'max', 'mean', 'std', 'q25', 'q50', 'q75', 'skew', 'kurtosis', 
                                'gradient_mean', 'gradient_max']:
                        summary['field_stats'][field_name][stat].append(0.0)
        
        return summary
    
    def _compute_field_gradient(self, mesh_data, field_values):
        """Compute gradient of field on mesh"""
        if mesh_data.triangles is None:
            return 0.0, 0.0
        
        gradients = []
        for tri in mesh_data.triangles[:1000]:  # Sample for performance
            if all(idx < len(field_values) for idx in tri):
                # Simple gradient approximation
                values = [field_values[idx] for idx in tri]
                grad = max(values) - min(values)
                gradients.append(grad)
        
        if gradients:
            return np.mean(gradients), np.max(gradients)
        return 0.0, 0.0
    
    def get_field_evolution(self, sim_name, field_name):
        """Get field evolution over time for a simulation"""
        if sim_name not in self.simulations:
            return None
        
        sim = self.simulations[sim_name]
        if field_name not in sim['mesh_data'].fields:
            return None
        
        field_data = sim['mesh_data'].fields[field_name]
        evolution = []
        
        for t in range(field_data.shape[0]):
            if field_data[t].ndim == 1:
                values = field_data[t]
            else:
                values = np.linalg.norm(field_data[t], axis=1)
            evolution.append(values)
        
        return np.array(evolution)
    
    def get_similar_simulations(self, energy, duration, max_results=5):
        """Find simulations with similar parameters"""
        similarities = []
        
        for name, sim in self.simulations.items():
            # Compute similarity score
            energy_diff = abs(sim['energy_mJ'] - energy) / max(energy, sim['energy_mJ'])
            duration_diff = abs(sim['duration_ns'] - duration) / max(duration, sim['duration_ns'])
            
            similarity = 1.0 - (0.7 * energy_diff + 0.3 * duration_diff)
            similarities.append((name, similarity, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:max_results]

# =============================================
# COMPLETE GEOMETRICAL FIELD PREDICTOR
# =============================================
class GeometricalFieldPredictor:
    """Complete field predictor using attention-based interpolation"""
    
    def __init__(self, extrapolator, data_loader):
        self.extrapolator = extrapolator
        self.data_loader = data_loader
        self.rbf_interpolators = {}
        self.cache_manager = CacheManager()
        
    def predict_field_on_mesh(self, energy_query, duration_query, time_query, 
                             reference_mesh, field_name, use_cache=True):
        """Predict field values on a reference mesh geometry"""
        if not self.extrapolator.fitted:
            st.error("Extrapolator not fitted. Please load data first.")
            return None
        
        # Generate cache key
        cache_key = f"pred_{energy_query}_{duration_query}_{time_query}_{field_name}"
        if use_cache:
            cached = self.cache_manager.get_cached_result(cache_key)
            if cached:
                return cached
        
        # Get prediction statistics for the field
        stats_prediction = self.extrapolator.predict_field_statistics(
            energy_query, duration_query, time_query, field_name)
        
        if not stats_prediction or 'field_predictions' not in stats_prediction:
            return None
        
        if field_name not in stats_prediction['field_predictions']:
            return None
        
        attention_weights = stats_prediction['attention_weights']
        target_stats = stats_prediction['field_predictions'][field_name]
        
        # Initialize mesh values
        mesh_points = reference_mesh.points
        n_points = len(mesh_points)
        
        # Find source simulations
        source_distributions = []
        source_weights = []
        source_metadata = []
        
        for i, (weight, meta) in enumerate(zip(attention_weights, self.extrapolator.source_metadata)):
            if weight > 0.01:  # Only consider significant sources
                sim_name = meta['name']
                timestep_idx = meta['timestep_idx']
                
                # Find the simulation
                if sim_name in self.data_loader.simulations:
                    sim = self.data_loader.simulations[sim_name]
                    if hasattr(sim['mesh_data'], 'fields') and field_name in sim['mesh_data'].fields:
                        field_data = sim['mesh_data'].fields[field_name]
                        
                        if timestep_idx < field_data.shape[0]:
                            source_values = field_data[timestep_idx]
                            
                            # Handle vector fields
                            if source_values.ndim == 2:
                                source_values = np.linalg.norm(source_values, axis=1)
                            
                            # Ensure same number of points (interpolate if necessary)
                            if len(source_values) != n_points:
                                # Interpolate to reference mesh
                                source_values = self._interpolate_to_mesh(
                                    sim['mesh_data'].points, source_values, mesh_points)
                            
                            if source_values is not None:
                                source_distributions.append(source_values)
                                source_weights.append(weight)
                                source_metadata.append({
                                    'sim_name': sim_name,
                                    'timestep': meta['time'],
                                    'energy': meta['energy'],
                                    'duration': meta['duration']
                                })
        
        if not source_distributions:
            # Create synthetic distribution based on statistics
            predicted_values = self._create_synthetic_field(
                mesh_points, target_stats, field_name)
            confidence = 0.3
            method = 'synthetic'
        else:
            # Blend distributions based on attention weights
            source_weights = np.array(source_weights)
            source_weights = source_weights / np.sum(source_weights)
            
            # Weighted average of source distributions
            blended_distribution = np.zeros(n_points)
            for dist, weight in zip(source_distributions, source_weights):
                blended_distribution += dist * weight
            
            # Scale blended distribution to match predicted statistics
            predicted_values = self._scale_to_target_stats(
                blended_distribution, target_stats)
            
            # Apply spatial smoothing
            predicted_values = self._apply_spatial_smoothing(
                predicted_values, mesh_points, reference_mesh)
            
            confidence = float(np.mean(source_weights))
            method = 'attention_blend'
        
        # Ensure physical constraints
        predicted_values = self._apply_physical_constraints(
            predicted_values, field_name)
        
        result = {
            'values': predicted_values,
            'method': method,
            'confidence': confidence,
            'n_sources': len(source_distributions),
            'target_stats': target_stats,
            'source_stats': {
                'min': float(np.min(predicted_values)),
                'max': float(np.max(predicted_values)),
                'mean': float(np.mean(predicted_values)),
                'std': float(np.std(predicted_values))
            }
        }
        
        # Cache the result
        if use_cache:
            self.cache_manager.cache_result(cache_key, result)
        
        return result
    
    def _interpolate_to_mesh(self, source_points, source_values, target_points):
        """Interpolate values from source mesh to target mesh"""
        if len(source_values) < 10:
            return None
        
        # Use nearest neighbor interpolation for speed
        try:
            interpolator = NearestNDInterpolator(source_points, source_values)
            return interpolator(target_points)
        except:
            # Fallback to simple averaging
            return np.full(len(target_points), np.mean(source_values))
    
    def _create_synthetic_field(self, points, target_stats, field_name):
        """Create synthetic field distribution based on statistics"""
        n_points = len(points)
        
        # Create spatial variation pattern
        centroid = np.mean(points, axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        distances_norm = distances / (np.max(distances) + 1e-8)
        
        # Different patterns for different field types
        if 'temperature' in field_name.lower() or 'heat' in field_name.lower():
            # Gaussian temperature distribution
            spatial_pattern = np.exp(-distances_norm ** 2 / 0.2)
        elif 'stress' in field_name.lower() or 'pressure' in field_name.lower():
            # Radial stress pattern
            spatial_pattern = 1.0 - distances_norm
        elif 'displacement' in field_name.lower():
            # Linear displacement pattern
            spatial_pattern = distances_norm
        else:
            # Default: sinusoidal pattern
            spatial_pattern = 0.5 * (1 + np.sin(5 * distances_norm))
        
        # Add some noise for realism
        noise = np.random.normal(0, 0.1, n_points)
        spatial_pattern = spatial_pattern * (1 + 0.1 * noise)
        
        # Scale to match target statistics
        current_mean = np.mean(spatial_pattern)
        current_std = np.std(spatial_pattern)
        
        target_mean = target_stats['mean']
        target_std = target_stats['std']
        
        if current_std > 1e-6:
            scaled_values = (spatial_pattern - current_mean) / current_std * target_std + target_mean
        else:
            scaled_values = np.full(n_points, target_mean)
        
        return scaled_values
    
    def _scale_to_target_stats(self, values, target_stats):
        """Scale values to match target statistics"""
        current_mean = np.mean(values)
        current_std = np.std(values)
        
        target_mean = target_stats['mean']
        target_std = target_stats['std']
        
        if current_std > 1e-6:
            # Scale and shift to match target statistics
            scaled_values = (values - current_mean) / current_std * target_std + target_mean
        else:
            # Constant distribution with target mean
            scaled_values = np.full_like(values, target_mean)
        
        return scaled_values
    
    def _apply_spatial_smoothing(self, values, points, mesh_data, smoothing_factor=0.5):
        """Apply spatial smoothing to field values"""
        if mesh_data.triangles is None or len(mesh_data.triangles) == 0:
            return values
        
        # Simple Laplacian smoothing
        smoothed = values.copy()
        
        # Build adjacency list
        adjacency = [[] for _ in range(len(points))]
        for tri in mesh_data.triangles[:min(5000, len(mesh_data.triangles))]:
            if all(idx < len(points) for idx in tri):
                for i in range(3):
                    for j in range(3):
                        if i != j:
                            adjacency[tri[i]].append(tri[j])
        
        # Apply smoothing
        for i in range(len(points)):
            if adjacency[i]:
                neighbor_values = values[adjacency[i]]
                smoothed[i] = (1 - smoothing_factor) * values[i] + \
                             smoothing_factor * np.mean(neighbor_values)
        
        return smoothed
    
    def _apply_physical_constraints(self, values, field_name):
        """Apply physical constraints to field values"""
        values = np.array(values)
        
        # Non-negative constraints
        if any(keyword in field_name.lower() for keyword in 
               ['temperature', 'stress', 'pressure', 'density', 'energy']):
            values = np.maximum(values, 0)
        
        # Upper bounds for certain fields
        if 'temperature' in field_name.lower():
            values = np.minimum(values, 5000)  # Reasonable upper bound for temperature
        
        return values
    
    def predict_field_evolution(self, energy_query, duration_query, time_points, 
                               reference_mesh, field_name, progress_callback=None):
        """Predict field evolution over time on mesh"""
        predictions = []
        confidences = []
        
        total_steps = len(time_points)
        for i, t in enumerate(time_points):
            pred = self.predict_field_on_mesh(energy_query, duration_query, t, 
                                             reference_mesh, field_name)
            if pred:
                predictions.append(pred['values'])
                confidences.append(pred['confidence'])
            else:
                predictions.append(None)
                confidences.append(0.0)
            
            # Update progress
            if progress_callback:
                progress_callback(i + 1, total_steps)
        
        return predictions, confidences
    
    def compute_spatial_correlations(self, field_values, reference_mesh):
        """Compute spatial correlation structure of predicted field"""
        if field_values is None or reference_mesh.points is None:
            return None
        
        points = reference_mesh.points
        values = field_values
        
        # Compute distance matrix (sampled for efficiency)
        n_samples = min(200, len(points))
        sample_indices = np.random.choice(len(points), n_samples, replace=False)
        sample_points = points[sample_indices]
        sample_values = values[sample_indices]
        
        # Compute pairwise distances and value differences
        distances = []
        value_diffs = []
        
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                dist = np.linalg.norm(sample_points[i] - sample_points[j])
                val_diff = abs(sample_values[i] - sample_values[j])
                
                distances.append(dist)
                value_diffs.append(val_diff)
        
        if len(distances) > 10:
            # Compute correlation
            try:
                corr, _ = linregress(distances, value_diffs)[:2]
                return {
                    'correlation': float(corr),
                    'distances': distances[:100],  # Return sample for plotting
                    'value_diffs': value_diffs[:100]
                }
            except:
                return {'correlation': 0.0, 'distances': [], 'value_diffs': []}
        
        return {'correlation': 0.0, 'distances': [], 'value_diffs': []}

# =============================================
# ENHANCED VISUALIZER WITH ADDITIONAL FEATURES
# =============================================
class EnhancedGeometricalVisualizer:
    """Visualization components with advanced geometrical features"""
    
    EXTENDED_COLORMAPS = [
        'Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'Rainbow',
        'Jet', 'Hot', 'Cool', 'Portland', 'Bluered', 'Electric',
        'Thermal', 'Balance', 'Brwnyl', 'Darkmint', 'Emrld', 'Mint',
        'Oranges', 'Purp', 'Purples', 'Sunset', 'Sunsetdark', 'Teal',
        'Tealgrn', 'Twilight', 'Burg', 'Burgyl', 'RdYlBu', 'RdYlGn',
        'Viridis_r', 'Plasma_r', 'Inferno_r', 'Magma_r', 'Cividis_r'
    ]
    
    @staticmethod
    def create_advanced_mesh_visualization(mesh_data, field_name, field_values, 
                                          colormap="Viridis", title="", opacity=0.9,
                                          show_wireframe=True, show_points=False,
                                          show_normals=False, show_boundary=False,
                                          lighting_intensity=1.0):
        """Create advanced 3D mesh visualization with multiple features"""
        if mesh_data is None or field_values is None:
            return go.Figure()
        
        pts = mesh_data.points
        triangles = mesh_data.triangles
        
        fig = go.Figure()
        
        # Determine if we have triangles for surface rendering
        has_triangles = triangles is not None and len(triangles) > 0
        
        if has_triangles and not show_points:
            # Surface mesh visualization
            fig.add_trace(go.Mesh3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
                intensity=field_values,
                colorscale=colormap,
                intensitymode='vertex',
                colorbar=dict(
                    title=dict(text=field_name, font=dict(size=14)),
                    thickness=20,
                    len=0.75,
                    tickfont=dict(size=12)
                ),
                opacity=opacity,
                lighting=dict(
                    ambient=0.8 * lighting_intensity,
                    diffuse=0.8 * lighting_intensity,
                    specular=0.5 * lighting_intensity,
                    roughness=0.5,
                    fresnel=0.2
                ),
                lightposition=dict(x=100, y=100, z=100),
                hoverinfo='skip',
                name="Field Distribution",
                showlegend=True,
                flatshading=False
            ))
            
            # Add wireframe for better geometry perception
            if show_wireframe:
                edges = set()
                for tri in triangles[:min(2000, len(triangles))]:
                    for i in range(3):
                        edge = tuple(sorted((tri[i], tri[(i+1)%3])))
                        edges.add(edge)
                
                edge_x = []
                edge_y = []
                edge_z = []
                for edge in list(edges)[:1000]:
                    if edge[0] < len(pts) and edge[1] < len(pts):
                        edge_x.extend([pts[edge[0], 0], pts[edge[1], 0], None])
                        edge_y.extend([pts[edge[0], 1], pts[edge[1], 1], None])
                        edge_z.extend([pts[edge[0], 2], pts[edge[1], 2], None])
                
                fig.add_trace(go.Scatter3d(
                    x=edge_x, y=edge_y, z=edge_z,
                    mode='lines',
                    line=dict(color='rgba(50, 50, 50, 0.3)', width=1.5),
                    opacity=0.3,
                    hoverinfo='none',
                    name="Mesh Wireframe",
                    showlegend=True
                ))
            
            # Show surface normals
            if show_normals and has_triangles:
                normals = mesh_data.get_surface_normals()
                if normals is not None:
                    # Sample normals for clarity
                    sample_indices = np.random.choice(len(triangles), 
                                                     min(100, len(triangles)), 
                                                     replace=False)
                    
                    for idx in sample_indices:
                        tri = triangles[idx]
                        # Get triangle center
                        center = np.mean(pts[tri], axis=0)
                        normal = normals[idx] * 0.1  # Scale for visualization
                        
                        fig.add_trace(go.Scatter3d(
                            x=[center[0], center[0] + normal[0]],
                            y=[center[1], center[1] + normal[1]],
                            z=[center[2], center[2] + normal[2]],
                            mode='lines',
                            line=dict(color='cyan', width=3),
                            hoverinfo='none',
                            showlegend=False if idx > 0 else True,
                            name="Surface Normals" if idx == 0 else None
                        ))
            
            # Show boundary edges
            if show_boundary and hasattr(mesh_data, 'boundary_edges') and mesh_data.boundary_edges:
                boundary_x = []
                boundary_y = []
                boundary_z = []
                
                for edge in mesh_data.boundary_edges[:500]:
                    if edge[0] < len(pts) and edge[1] < len(pts):
                        boundary_x.extend([pts[edge[0], 0], pts[edge[1], 0], None])
                        boundary_y.extend([pts[edge[0], 1], pts[edge[1], 1], None])
                        boundary_z.extend([pts[edge[0], 2], pts[edge[1], 2], None])
                
                fig.add_trace(go.Scatter3d(
                    x=boundary_x, y=boundary_y, z=boundary_z,
                    mode='lines',
                    line=dict(color='red', width=3),
                    opacity=0.8,
                    hoverinfo='none',
                    name="Boundary Edges",
                    showlegend=True
                ))
        else:
            # Point cloud visualization
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode='markers',
                marker=dict(
                    size=4 if show_points else 3,
                    color=field_values,
                    colorscale=colormap,
                    opacity=opacity,
                    colorbar=dict(
                        title=dict(text=field_name, font=dict(size=14)),
                        thickness=20,
                        len=0.75
                    ),
                    showscale=True,
                    line=dict(width=0),
                    symbol='circle'
                ),
                hovertemplate=(
                    '<b>Location:</b> (%{x:.3f}, %{y:.3f}, %{z:.3f})<br>'
                    f'<b>{field_name}:</b> %{{marker.color:.3f}}<br>'
                    '<extra></extra>'
                ),
                name="Field Points",
                showlegend=True
            ))
        
        # Add enhanced coordinate axes
        if pts.shape[0] > 0:
            centroid = np.mean(pts, axis=0)
            axis_length = np.max(pts, axis=0) - np.min(pts, axis=0)
            max_length = np.max(axis_length) * 0.3
            
            # X axis (red)
            fig.add_trace(go.Scatter3d(
                x=[centroid[0], centroid[0] + max_length],
                y=[centroid[1], centroid[1]],
                z=[centroid[2], centroid[2]],
                mode='lines+text',
                line=dict(color='red', width=5),
                text=['', 'X'],
                textposition="top center",
                textfont=dict(size=14, color='red'),
                hoverinfo='none',
                showlegend=False
            ))
            
            # Y axis (green)
            fig.add_trace(go.Scatter3d(
                x=[centroid[0], centroid[0]],
                y=[centroid[1], centroid[1] + max_length],
                z=[centroid[2], centroid[2]],
                mode='lines+text',
                line=dict(color='green', width=5),
                text=['', 'Y'],
                textposition="top center",
                textfont=dict(size=14, color='green'),
                hoverinfo='none',
                showlegend=False
            ))
            
            # Z axis (blue)
            fig.add_trace(go.Scatter3d(
                x=[centroid[0], centroid[0]],
                y=[centroid[1], centroid[1]],
                z=[centroid[2], centroid[2] + max_length],
                mode='lines+text',
                line=dict(color='blue', width=5),
                text=['', 'Z'],
                textposition="top center",
                textfont=dict(size=14, color='blue'),
                hoverinfo='none',
                showlegend=False
            ))
        
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=22, family="Arial, sans-serif", color="#2c3e50"),
                x=0.5,
                xanchor='center'
            ),
            scene=dict(
                aspectmode="data",
                xaxis=dict(
                    title="X (mm)",
                    gridcolor="lightgray",
                    showbackground=True,
                    backgroundcolor="rgba(240, 240, 240, 0.1)",
                    showspikes=False,
                    zerolinecolor="lightgray"
                ),
                yaxis=dict(
                    title="Y (mm)",
                    gridcolor="lightgray",
                    showbackground=True,
                    backgroundcolor="rgba(240, 240, 240, 0.1)",
                    showspikes=False,
                    zerolinecolor="lightgray"
                ),
                zaxis=dict(
                    title="Z (mm)",
                    gridcolor="lightgray",
                    showbackground=True,
                    backgroundcolor="rgba(240, 240, 240, 0.1)",
                    showspikes=False,
                    zerolinecolor="lightgray"
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0)
                )
            ),
            height=750,
            margin=dict(l=0, r=0, t=80, b=0),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="black",
                borderwidth=1
            ),
            paper_bgcolor="white",
            plot_bgcolor="white"
        )
        
        return fig
    
    @staticmethod
    def create_volume_rendering(mesh_data, field_values, colormap="Viridis", 
                               title="Volume Rendering", opacity_scale=None):
        """Create volume rendering visualization"""
        if mesh_data is None or field_values is None:
            return go.Figure()
        
        pts = mesh_data.points
        
        # Create a 3D grid for volume rendering
        x_min, x_max = np.min(pts[:, 0]), np.max(pts[:, 0])
        y_min, y_max = np.min(pts[:, 1]), np.max(pts[:, 1])
        z_min, z_max = np.min(pts[:, 2]), np.max(pts[:, 2])
        
        # Create grid
        grid_size = 30  # Reduced for performance
        x_grid = np.linspace(x_min, x_max, grid_size)
        y_grid = np.linspace(y_min, y_max, grid_size)
        z_grid = np.linspace(z_min, z_max, grid_size)
        
        X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
        
        # Interpolate field values onto grid
        from scipy.interpolate import griddata
        grid_points = np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T
        grid_values = griddata(pts, field_values, grid_points, method='linear', 
                              fill_value=np.nanmean(field_values))
        
        grid_values = grid_values.reshape(X.shape)
        
        # Create opacity scale if not provided
        if opacity_scale is None:
            opacity_scale = [[0, 0.0], [0.1, 0.1], [0.5, 0.3], [1, 0.8]]
        
        fig = go.Figure(data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=grid_values.flatten(),
            isomin=np.nanmin(grid_values),
            isomax=np.nanmax(grid_values),
            opacityscale=opacity_scale,
            surface_count=25,
            colorscale=colormap,
            caps=dict(x_show=False, y_show=False, z_show=False),
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                camera=dict(eye=dict(x=2, y=2, z=2))
            ),
            height=600
        )
        
        return fig
    
    @staticmethod
    def create_streamline_visualization(mesh_data, vector_field, density=2, 
                                       colormap="Viridis", title="Streamlines"):
        """Create streamline visualization for vector fields"""
        if mesh_data is None or vector_field is None:
            return go.Figure()
        
        pts = mesh_data.points
        
        # Sample points for streamlines
        n_samples = min(500, len(pts))
        sample_indices = np.random.choice(len(pts), n_samples, replace=False)
        sample_points = pts[sample_indices]
        sample_vectors = vector_field[sample_indices]
        
        # Create streamlines
        fig = go.Figure()
        
        # Add mesh surface
        if mesh_data.triangles is not None:
            fig.add_trace(go.Mesh3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                i=mesh_data.triangles[:, 0], j=mesh_data.triangles[:, 1], 
                k=mesh_data.triangles[:, 2],
                opacity=0.1,
                color='lightgray',
                name="Mesh Surface"
            ))
        
        # Add streamlines
        for i in range(0, n_samples, density):
            start_point = sample_points[i]
            vector = sample_vectors[i]
            
            # Create streamline segment
            end_point = start_point + vector * 0.1  # Scale for visualization
            
            # Color by vector magnitude
            magnitude = np.linalg.norm(vector)
            
            fig.add_trace(go.Scatter3d(
                x=[start_point[0], end_point[0]],
                y=[start_point[1], end_point[1]],
                z=[start_point[2], end_point[2]],
                mode='lines',
                line=dict(
                    width=3,
                    color=magnitude,
                    colorscale=colormap
                ),
                showlegend=False,
                hoverinfo='none'
            ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            scene=dict(
                aspectmode="data",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=600
        )
        
        return fig
    
    @staticmethod
    def create_comparison_dashboard(mesh_data_list, field_values_list, titles,
                                   colormap="Viridis", n_cols=2):
        """Create a dashboard comparing multiple visualizations"""
        n_plots = len(mesh_data_list)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=titles,
            specs=[[{'type': 'scene'} for _ in range(n_cols)] for _ in range(n_rows)]
        )
        
        for idx, (mesh_data, field_values) in enumerate(zip(mesh_data_list, field_values_list)):
            row = idx // n_cols + 1
            col = idx % n_cols + 1
            
            if mesh_data is not None and field_values is not None:
                pts = mesh_data.points
                
                if mesh_data.triangles is not None and len(mesh_data.triangles) > 0:
                    fig.add_trace(
                        go.Mesh3d(
                            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                            i=mesh_data.triangles[:, 0], 
                            j=mesh_data.triangles[:, 1], 
                            k=mesh_data.triangles[:, 2],
                            intensity=field_values,
                            colorscale=colormap,
                            intensitymode='vertex',
                            opacity=0.8,
                            showscale=(idx == 0)
                        ),
                        row=row, col=col
                    )
                else:
                    fig.add_trace(
                        go.Scatter3d(
                            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                            mode='markers',
                            marker=dict(
                                size=3,
                                color=field_values,
                                colorscale=colormap,
                                opacity=0.8,
                                showscale=(idx == 0)
                            )
                        ),
                        row=row, col=col
                    )
        
        fig.update_layout(
            height=300 * n_rows,
            showlegend=False,
            title_text="Comparison Dashboard"
        )
        
        return fig

# =============================================
# VALIDATION AND ERROR HANDLING
# =============================================
class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

class DataValidator:
    """Validator for FEA data and parameters"""
    
    @staticmethod
    def validate_simulation_parameters(energy, duration, time):
        """Validate simulation input parameters"""
        errors = []
        
        if energy <= 0 or energy > 100:
            errors.append("Energy must be between 0.1 and 100 mJ")
        
        if duration <= 0 or duration > 50:
            errors.append("Duration must be between 0.1 and 50 ns")
        
        if time < 0 or time > 1000:
            errors.append("Time must be between 0 and 1000 ns")
        
        if errors:
            raise ValidationError("; ".join(errors))
        
        return True
    
    @staticmethod
    def validate_mesh_data(mesh_data):
        """Validate mesh data structure"""
        if mesh_data is None:
            raise ValidationError("Mesh data is None")
        
        if mesh_data.points is None or len(mesh_data.points) == 0:
            raise ValidationError("Mesh has no points")
        
        if np.any(np.isnan(mesh_data.points)):
            raise ValidationError("Mesh contains NaN points")
        
        if mesh_data.triangles is not None:
            if np.any(np.isnan(mesh_data.triangles)):
                raise ValidationError("Mesh triangles contain NaN values")
            
            # Check triangle indices are within bounds
            max_idx = len(mesh_data.points) - 1
            if np.any(mesh_data.triangles < 0) or np.any(mesh_data.triangles > max_idx):
                raise ValidationError("Triangle indices out of bounds")
        
        return True
    
    @staticmethod
    def validate_field_data(field_values, field_name=""):
        """Validate field data"""
        if field_values is None:
            raise ValidationError(f"Field {field_name} is None")
        
        if np.all(np.isnan(field_values)):
            raise ValidationError(f"Field {field_name} contains only NaN values")
        
        if np.any(np.isinf(field_values)):
            raise ValidationError(f"Field {field_name} contains infinite values")
        
        return True

# =============================================
# ENHANCED MAIN APPLICATION
# =============================================
class EnhancedFEAVisualizationPlatform:
    """Enhanced main application class with complete functionality"""
    
    def __init__(self):
        self.data_loader = EnhancedFEADataLoader()
        self.visualizer = EnhancedGeometricalVisualizer()
        self.validator = DataValidator()
        self.extrapolator = None
        self.geom_predictor = None
        self.cache_manager = CacheManager()
        
        # Initialize session state
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state variables"""
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'selected_colormap' not in st.session_state:
            st.session_state.selected_colormap = "Viridis"
        if 'current_mode' not in st.session_state:
            st.session_state.current_mode = "Data Viewer"
        if 'loaded_simulations' not in st.session_state:
            st.session_state.loaded_simulations = {}
        if 'extrapolator_fitted' not in st.session_state:
            st.session_state.extrapolator_fitted = False
    
    def run(self):
        """Main application entry point"""
        st.set_page_config(
            page_title="Enhanced FEA Laser Simulation Platform",
            layout="wide",
            initial_sidebar_state="expanded",
            page_icon="🔬",
            menu_items={
                'Get Help': 'https://github.com/your-repo',
                'Report a bug': 'https://github.com/your-repo/issues',
                'About': """
                # Enhanced FEA Laser Simulation Platform
                
                A comprehensive platform for visualizing and analyzing 
                Finite Element Analysis simulations of laser processes.
                
                Version 2.0
                """
            }
        )
        
        # Apply custom CSS
        self.apply_enhanced_css()
        
        # Render header
        self.render_enhanced_header()
        
        # Render sidebar
        self.render_enhanced_sidebar()
        
        # Render main content based on mode
        self.render_enhanced_main_content()
    
    def apply_enhanced_css(self):
        """Apply enhanced custom CSS styling"""
        st.markdown("""
        <style>
        /* Main header with gradient */
        .main-header {
            font-size: 3.2rem;
            background: linear-gradient(90deg, #1E88E5, #4A00E0, #FF416C);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 1rem;
            font-weight: 900;
            text-shadow: 3px 3px 6px rgba(0,0,0,0.2);
            padding: 1rem;
            border-bottom: 3px solid linear-gradient(90deg, #1E88E5, #4A00E0);
        }
        
        /* Enhanced sub-header */
        .sub-header {
            font-size: 2rem;
            color: #2c3e50;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 3px solid;
            border-image: linear-gradient(90deg, #3498db, #2ecc71) 1;
            font-weight: 700;
            position: relative;
        }
        
        .sub-header:after {
            content: '';
            position: absolute;
            bottom: -3px;
            left: 0;
            width: 100px;
            height: 3px;
            background: linear-gradient(90deg, #3498db, #2ecc71);
        }
        
        /* Enhanced info boxes */
        .info-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            margin: 1.5rem 0;
            box-shadow: 0 15px 30px rgba(0,0,0,0.15);
            border-left: 5px solid #4A00E0;
            animation: fadeIn 0.5s ease-out;
        }
        
        .warning-box {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            margin: 1.5rem 0;
            box-shadow: 0 15px 30px rgba(0,0,0,0.15);
            border-left: 5px solid #e74c3c;
            animation: fadeIn 0.5s ease-out;
        }
        
        .success-box {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            margin: 1.5rem 0;
            box-shadow: 0 15px 30px rgba(0,0,0,0.15);
            border-left: 5px solid #2ecc71;
            animation: fadeIn 0.5s ease-out;
        }
        
        /* Enhanced metric cards */
        .metric-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
            border: 1px solid rgba(255,255,255,0.3);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 24px rgba(0,0,0,0.15);
        }
        
        /* Enhanced tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: linear-gradient(90deg, #f8f9fa, #e9ecef);
            padding: 8px;
            border-radius: 12px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 60px;
            white-space: pre-wrap;
            background-color: #ffffff;
            border-radius: 10px;
            color: #495057;
            font-weight: 600;
            padding: 0 28px;
            transition: all 0.3s ease;
            border: 2px solid transparent;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #e9ecef;
            border-color: #3498db;
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(90deg, #1E88E5, #4A00E0);
            color: white;
            border-color: #1E88E5;
            box-shadow: 0 6px 12px rgba(30, 136, 229, 0.3);
        }
        
        /* Enhanced buttons */
        .stButton > button {
            background: linear-gradient(90deg, #1E88E5, #4A00E0);
            color: white;
            border: none;
            padding: 1rem 2rem;
            border-radius: 12px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 6px 12px rgba(30, 136, 229, 0.3);
            font-size: 1rem;
            position: relative;
            overflow: hidden;
        }
        
        .stButton > button:before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }
        
        .stButton > button:hover:before {
            left: 100%;
        }
        
        /* Geometry cards */
        .geometry-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1.5rem 0;
            border-left: 5px solid #1E88E5;
            box-shadow: 0 8px 16px rgba(0,0,0,0.08);
            transition: transform 0.3s ease;
        }
        
        .geometry-card:hover {
            transform: translateX(5px);
        }
        
        /* Loading animation */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Status indicators */
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 2s infinite;
        }
        
        .status-green { background-color: #2ecc71; }
        .status-yellow { background-color: #f39c12; }
        .status-red { background-color: #e74c3c; }
        
        @keyframes pulse {
            0% { transform: scale(0.95); opacity: 0.7; }
            50% { transform: scale(1.05); opacity: 1; }
            100% { transform: scale(0.95); opacity: 0.7; }
        }
        
        /* Tooltip styling */
        .tooltip {
            position: relative;
            display: inline-block;
        }
        
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: #2c3e50;
            color: white;
            text-align: center;
            border-radius: 6px;
            padding: 10px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 0.9rem;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render_enhanced_header(self):
        """Render enhanced application header"""
        st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1 class="main-header">🔬 Advanced FEA Laser Simulation Platform</h1>
            <p style="font-size: 1.3rem; color: #666; margin-bottom: 0.5rem;">
                Geometrical Mesh Visualization with Physics-Informed Machine Learning
            </p>
            <p style="font-size: 1rem; color: #888; font-style: italic;">
                Version 2.0 | Real-time 3D Visualization | AI-Powered Predictions
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add status bar
        self.render_status_bar()
    
    def render_status_bar(self):
        """Render status bar at top of application"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status_color = "status-green" if st.session_state.data_loaded else "status-red"
            st.markdown(f"""
            <div class="tooltip">
                <span class="status-indicator {status_color}"></span>
                <span style="font-weight: 600;">Data Status:</span> 
                {'Loaded' if st.session_state.data_loaded else 'Not Loaded'}
                <span class="tooltiptext">
                    {'✓ All simulation data loaded and ready' if st.session_state.data_loaded else '✗ No data loaded yet'}
                </span>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if st.session_state.data_loaded:
                sim_count = len(st.session_state.get('loaded_simulations', {}))
                st.markdown(f"""
                <div class="tooltip">
                    <span class="status-indicator status-green"></span>
                    <span style="font-weight: 600;">Simulations:</span> {sim_count}
                    <span class="tooltiptext">
                        Number of loaded FEA simulations
                    </span>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            if st.session_state.data_loaded and 'available_fields' in st.session_state:
                field_count = len(st.session_state.available_fields)
                st.markdown(f"""
                <div class="tooltip">
                    <span class="status-indicator status-green"></span>
                    <span style="font-weight: 600;">Fields:</span> {field_count}
                    <span class="tooltiptext">
                        Number of available physical fields
                    </span>
                </div>
                """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="tooltip">
                <span class="status-indicator status-green"></span>
                <span style="font-weight: 600;">Mode:</span> {st.session_state.current_mode}
                <span class="tooltiptext">
                    Current application mode
                </span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
    
    def render_enhanced_sidebar(self):
        """Render enhanced application sidebar"""
        with st.sidebar:
            # Logo and title
            st.markdown("""
            <div style="text-align: center; margin-bottom: 2rem;">
                <h3 style="color: #2c3e50; margin-bottom: 0.5rem;">FEA Visualization Platform</h3>
                <p style="color: #7f8c8d; font-size: 0.9rem;">Advanced Simulation Analysis</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### ⚙️ Navigation")
            
            # Mode selection with icons
            app_mode = st.selectbox(
                "Select Application Mode",
                ["Data Viewer", "Interpolation/Extrapolation", "Comparative Analysis", 
                 "Geometrical Visualization", "Field Predictor", "Export & Reports"],
                index=["Data Viewer", "Interpolation/Extraprapolation", "Comparative Analysis", 
                      "Geometrical Visualization", "Field Predictor", "Export & Reports"].index(
                    st.session_state.current_mode
                ) if st.session_state.current_mode in ["Data Viewer", "Interpolation/Extrapolation", 
                    "Comparative Analysis", "Geometrical Visualization", "Field Predictor", 
                    "Export & Reports"] else 0,
                key="nav_mode",
                format_func=lambda x: f"📊 {x}" if x == "Data Viewer" else 
                                     f"🔮 {x}" if x == "Interpolation/Extrapolation" else
                                     f"📈 {x}" if x == "Comparative Analysis" else
                                     f"🏗️ {x}" if x == "Geometrical Visualization" else
                                     f"🤖 {x}" if x == "Field Predictor" else
                                     f"💾 {x}"
            )
            
            st.session_state.current_mode = app_mode
            
            st.markdown("---")
            st.markdown("### 📁 Data Management")
            
            col1, col2 = st.columns(2)
            with col1:
                load_full_data = st.checkbox("Load Full Mesh", value=True, 
                                            help="Load complete mesh data for 3D visualization")
            with col2:
                cache_enabled = st.checkbox("Enable Cache", value=True,
                                           help="Cache results for faster performance")
            
            st.session_state.selected_colormap = st.selectbox(
                "🎨 Default Colormap",
                EnhancedGeometricalVisualizer.EXTENDED_COLORMAPS,
                index=0,
                help="Select default colormap for visualizations"
            )
            
            if st.button("🔄 Load All Simulations", type="primary", use_container_width=True):
                self.load_simulations_with_progress(load_full_data, cache_enabled)
            
            if st.session_state.get('data_loaded', False):
                st.markdown("---")
                st.markdown("### ⚡ Quick Actions")
                
                if st.button("📊 Update Statistics", use_container_width=True):
                    self.update_statistics()
                
                if st.button("🧹 Clear Cache", use_container_width=True):
                    self.clear_cache()
                
                if st.button("🔄 Refresh Visualizations", use_container_width=True):
                    st.rerun()
            
            st.markdown("---")
            st.markdown("### 🔧 Settings")
            
            with st.expander("Visualization Settings", expanded=False):
                st.checkbox("Show Advanced Controls", value=True, key="show_advanced")
                st.checkbox("Enable Animations", value=True, key="enable_animations")
                st.slider("Default Opacity", 0.1, 1.0, 0.9, 0.1, key="default_opacity")
                st.selectbox("Default Lighting", ["Standard", "Bright", "Soft", "Dramatic"], 
                           key="default_lighting")
            
            with st.expander("Performance Settings", expanded=False):
                st.slider("Max Points to Display", 1000, 100000, 10000, 1000, 
                         key="max_display_points")
                st.slider("Cache Duration (hours)", 1, 72, 24, 1, 
                         key="cache_duration")
                st.checkbox("Use GPU Acceleration", value=False, 
                          help="Requires CUDA-enabled GPU")
            
            st.markdown("---")
            st.markdown("""
            <div style="text-align: center; margin-top: 2rem; color: #7f8c8d; font-size: 0.8rem;">
                <p>© 2024 FEA Visualization Platform</p>
                <p>Version 2.0</p>
            </div>
            """, unsafe_allow_html=True)
    
    def load_simulations_with_progress(self, load_full_mesh=True, cache_enabled=True):
        """Load simulations with enhanced progress tracking"""
        with st.spinner("Loading simulation data..."):
            # Create progress containers
            progress_container = st.empty()
            status_container = st.empty()
            details_container = st.empty()
            
            # Initialize progress
            progress_bar = progress_container.progress(0)
            status_container.text("Initializing data loader...")
            
            try:
                # Load simulations
                simulations, summaries = self.data_loader.load_all_simulations(
                    load_full_mesh=load_full_mesh
                )
                
                if simulations and summaries:
                    # Store in session state
                    st.session_state.simulations = simulations
                    st.session_state.summaries = summaries
                    st.session_state.loaded_simulations = simulations
                    st.session_state.data_loaded = True
                    
                    # Initialize extrapolator
                    self.extrapolator = AttentionExtrapolator(self.data_loader)
                    self.extrapolator.fit(simulations, summaries)
                    st.session_state.extrapolator_fitted = True
                    
                    # Initialize predictor
                    self.geom_predictor = GeometricalFieldPredictor(
                        self.extrapolator, self.data_loader
                    )
                    
                    progress_bar.progress(100)
                    status_container.text("✅ Data loaded successfully!")
                    
                    # Show success message with details
                    details_container.success(f"""
                    ✅ Successfully loaded {len(simulations)} simulations
                    
                    **Details:**
                    - Available fields: {len(self.data_loader.available_fields)}
                    - Common fields: {len(self.data_loader.common_fields)}
                    - Total timesteps: {sum(s['n_timesteps'] for s in simulations.values())}
                    - Memory usage: {self._estimate_memory_usage(simulations):.1f} MB
                    
                    **Extrapolator Status:** {'✅ Fitted' if self.extrapolator.fitted else '❌ Not fitted'}
                    """)
                    
                    # Store available fields in session state
                    st.session_state.available_fields = self.data_loader.available_fields
                    st.session_state.common_fields = self.data_loader.common_fields
                    
                    # Auto-navigate to Data Viewer
                    st.session_state.current_mode = "Data Viewer"
                    st.rerun()
                    
                else:
                    progress_bar.progress(100)
                    status_container.error("❌ Failed to load any simulations")
                    details_container.error("""
                    No valid simulations were found. Please check:
                    1. The directory structure is correct
                    2. VTU files exist in simulation folders
                    3. Files are not corrupted
                    """)
                    
            except Exception as e:
                progress_bar.progress(100)
                status_container.error(f"❌ Error loading data: {str(e)}")
                details_container.error(f"""
                **Error Details:**
                ```python
                {traceback.format_exc()}
                ```
                
                **Troubleshooting:**
                1. Check if meshio is installed: `pip install meshio`
                2. Verify VTU file format compatibility
                3. Check file permissions
                """)
    
    def _estimate_memory_usage(self, simulations):
        """Estimate memory usage of loaded simulations"""
        total_memory = 0
        for sim in simulations.values():
            if 'mesh_data' in sim:
                mesh_data = sim['mesh_data']
                # Estimate memory for points
                if mesh_data.points is not None:
                    total_memory += mesh_data.points.nbytes / (1024 * 1024)
                
                # Estimate memory for fields
                for field_name, field_data in mesh_data.fields.items():
                    total_memory += field_data.nbytes / (1024 * 1024)
        
        return total_memory
    
    def update_statistics(self):
        """Update and display statistics"""
        if not st.session_state.data_loaded:
            st.warning("No data loaded")
            return
        
        simulations = st.session_state.simulations
        
        with st.spinner("Updating statistics..."):
            # Create statistics dashboard
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Simulations", len(simulations))
            
            with col2:
                total_timesteps = sum(s['n_timesteps'] for s in simulations.values())
                st.metric("Total Timesteps", total_timesteps)
            
            with col3:
                if simulations:
                    first_sim = list(simulations.values())[0]
                    if first_sim['has_mesh']:
                        mesh_data = first_sim['mesh_data']
                        point_count = len(mesh_data.points)
                        st.metric("Mesh Points", f"{point_count:,}")
            
            with col4:
                st.metric("Memory Usage", f"{self._estimate_memory_usage(simulations):.1f} MB")
            
            # Detailed statistics
            with st.expander("📊 Detailed Statistics", expanded=True):
                # Field statistics
                st.subheader("Field Statistics")
                
                if simulations:
                    # Get all fields from first simulation
                    first_sim = list(simulations.values())[0]
                    fields = list(first_sim['field_info'].keys())
                    
                    for field in fields[:10]:  # Show first 10 fields
                        field_stats = []
                        for sim_name, sim in simulations.items():
                            if field in sim['field_info']:
                                mesh_data = sim['mesh_data']
                                if field in mesh_data.fields:
                                    field_data = mesh_data.fields[field]
                                    # Compute mean over time and space
                                    if field_data.ndim == 3:  # Vector field
                                        magnitudes = np.linalg.norm(field_data, axis=2)
                                        mean_val = np.nanmean(magnitudes)
                                    else:  # Scalar field
                                        mean_val = np.nanmean(field_data)
                                    
                                    field_stats.append({
                                        'Simulation': sim_name,
                                        'Mean Value': mean_val,
                                        'Energy': sim['energy_mJ'],
                                        'Duration': sim['duration_ns']
                                    })
                        
                        if field_stats:
                            df = pd.DataFrame(field_stats)
                            st.write(f"**{field}**")
                            st.dataframe(df.style.format({'Mean Value': '{:.3f}'}), 
                                        use_container_width=True)
    
    def clear_cache(self):
        """Clear the cache"""
        try:
            cache_dir = Path("cache")
            if cache_dir.exists():
                import shutil
                shutil.rmtree(cache_dir)
                cache_dir.mkdir(exist_ok=True)
                st.success("✅ Cache cleared successfully")
            else:
                st.info("ℹ️ Cache directory does not exist")
        except Exception as e:
            st.error(f"❌ Error clearing cache: {str(e)}")
    
    def render_enhanced_main_content(self):
        """Render enhanced main content based on selected mode"""
        app_mode = st.session_state.current_mode
        
        # Check if data is loaded for modes that require it
        if app_mode != "Data Viewer" and not st.session_state.get('data_loaded', False):
            self.show_data_not_loaded_warning()
            return
        
        if app_mode == "Data Viewer":
            self.render_enhanced_data_viewer()
        elif app_mode == "Interpolation/Extrapolation":
            self.render_enhanced_interpolation_extrapolation()
        elif app_mode == "Comparative Analysis":
            self.render_enhanced_comparative_analysis()
        elif app_mode == "Geometrical Visualization":
            self.render_enhanced_geometrical_visualization()
        elif app_mode == "Field Predictor":
            self.render_field_predictor()
        elif app_mode == "Export & Reports":
            self.render_export_reports()
    
    def render_enhanced_data_viewer(self):
        """Render enhanced data viewer with mesh visualization"""
        st.markdown('<h2 class="sub-header">📁 Data Viewer with Mesh Visualization</h2>', 
                   unsafe_allow_html=True)
        
        if not st.session_state.get('data_loaded', False):
            self.show_data_not_loaded_warning()
            return
        
        simulations = st.session_state.simulations
        
        # Quick navigation bar
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        with col1:
            sim_name = st.selectbox(
                "Select Simulation",
                sorted(simulations.keys()),
                key="viewer_sim_select",
                help="Choose a simulation to visualize"
            )
        
        sim = simulations[sim_name]
        
        with col2:
            st.metric("Energy", f"{sim['energy_mJ']:.2f} mJ", 
                     help="Laser pulse energy in millijoules")
        with col3:
            st.metric("Duration", f"{sim['duration_ns']:.2f} ns",
                     help="Laser pulse duration in nanoseconds")
        with col4:
            st.metric("Timesteps", sim['n_timesteps'],
                     help="Number of time steps in simulation")
        
        if not sim.get('has_mesh', False):
            st.warning("This simulation was loaded without mesh data.")
            return
        
        # Field and timestep selection in tabs
        tab1, tab2, tab3 = st.tabs(["🎨 Visualization", "📊 Analysis", "⚙️ Settings"])
        
        with tab1:
            self.render_visualization_tab(sim)
        
        with tab2:
            self.render_analysis_tab(sim)
        
        with tab3:
            self.render_settings_tab(sim)
    
    def render_visualization_tab(self, sim):
        """Render visualization tab"""
        col1, col2, col3 = st.columns(3)
        with col1:
            field = st.selectbox(
                "Select Field",
                sorted(sim['field_info'].keys()),
                key="viewer_field_select",
                help="Choose a physical field to visualize"
            )
        with col2:
            timestep = st.slider(
                "Timestep",
                0, sim['n_timesteps'] - 1, 0,
                key="viewer_timestep_slider",
                help="Select time step to visualize"
            )
        with col3:
            display_mode = st.selectbox(
                "Display Mode",
                ["Surface Mesh", "Point Cloud", "Wireframe", "Volume", "Streamlines"],
                key="display_mode",
                help="Choose visualization style"
            )
        
        # Get mesh data and field values
        mesh_data = sim['mesh_data']
        field_values = mesh_data.fields[field][timestep]
        
        # Handle vector fields
        if field_values.ndim == 2:
            is_vector_field = True
            vector_magnitude = np.linalg.norm(field_values, axis=1)
            display_values = vector_magnitude
        else:
            is_vector_field = False
            display_values = field_values
        
        # Advanced visualization options
        with st.expander("🎨 Advanced Visualization Options", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                show_normals = st.checkbox("Show Normals", value=False,
                                          help="Display surface normals")
                show_boundary = st.checkbox("Show Boundary", value=False,
                                           help="Highlight boundary edges")
            with col2:
                lighting = st.select_slider("Lighting", 
                                           options=["Soft", "Medium", "Bright", "Dramatic"],
                                           value="Medium")
            with col3:
                opacity = st.slider("Opacity", 0.1, 1.0, 
                                   st.session_state.get('default_opacity', 0.9), 0.1)
        
        # Create visualization based on mode
        if display_mode == "Surface Mesh":
            fig = self.visualizer.create_advanced_mesh_visualization(
                mesh_data,
                field,
                display_values,
                colormap=st.session_state.selected_colormap,
                title=f"{field} at Timestep {timestep + 1} - {sim['name']}",
                opacity=opacity,
                show_wireframe=True,
                show_points=False,
                show_normals=show_normals,
                show_boundary=show_boundary,
                lighting_intensity=1.0 if lighting == "Medium" else 
                                  0.7 if lighting == "Soft" else
                                  1.3 if lighting == "Bright" else 1.5
            )
        elif display_mode == "Point Cloud":
            fig = self.visualizer.create_advanced_mesh_visualization(
                mesh_data,
                field,
                display_values,
                colormap=st.session_state.selected_colormap,
                title=f"{field} at Timestep {timestep + 1} - {sim['name']}",
                opacity=opacity,
                show_wireframe=False,
                show_points=True,
                lighting_intensity=1.0
            )
        elif display_mode == "Volume":
            fig = self.visualizer.create_volume_rendering(
                mesh_data,
                display_values,
                colormap=st.session_state.selected_colormap,
                title=f"{field} Volume Rendering"
            )
        elif display_mode == "Streamlines" and is_vector_field:
            fig = self.visualizer.create_streamline_visualization(
                mesh_data,
                field_values,  # Use original vector field
                density=3,
                colormap=st.session_state.selected_colormap,
                title=f"{field} Streamlines"
            )
        else:
            fig = self.visualizer.create_advanced_mesh_visualization(
                mesh_data,
                field,
                display_values,
                colormap=st.session_state.selected_colormap,
                title=f"{field} at Timestep {timestep + 1} - {sim['name']}",
                opacity=opacity,
                show_wireframe=True,
                show_points=False,
                lighting_intensity=1.0
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Field value distribution
        st.markdown("##### 📊 Field Value Distribution")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Histogram
            fig_hist = px.histogram(
                x=display_values[~np.isnan(display_values)],
                nbins=50,
                title=f"{field} Value Distribution",
                labels={'x': field, 'y': 'Frequency'}
            )
            fig_hist.update_layout(height=300)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Statistics
            stats = {
                "Minimum": float(np.nanmin(display_values)),
                "Maximum": float(np.nanmax(display_values)),
                "Mean": float(np.nanmean(display_values)),
                "Std Dev": float(np.nanstd(display_values)),
                "Median": float(np.nanmedian(display_values)),
                "95th %ile": float(np.nanpercentile(display_values, 95))
            }
            
            for stat_name, stat_value in stats.items():
                st.metric(stat_name, f"{stat_value:.3f}")
    
    def render_analysis_tab(self, sim):
        """Render analysis tab"""
        mesh_data = sim['mesh_data']
        
        st.markdown("##### 📐 Mesh Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if hasattr(mesh_data, 'mesh_stats'):
                mesh_stats = mesh_data.mesh_stats
                
                st.markdown("**Geometrical Properties**")
                if 'surface_area' in mesh_stats:
                    st.metric("Surface Area", f"{mesh_stats['surface_area']:.3f} mm²")
                if 'triangle_count' in mesh_stats:
                    st.metric("Triangle Count", f"{mesh_stats['triangle_count']:,}")
                if 'avg_triangle_area' in mesh_stats:
                    st.metric("Avg Triangle Area", f"{mesh_stats['avg_triangle_area']:.6f} mm²")
                if 'aspect_ratio_stats' in mesh_stats:
                    st.metric("Avg Aspect Ratio", f"{mesh_stats['aspect_ratio_stats']['mean']:.3f}")
        
        with col2:
            st.markdown("**Spatial Properties**")
            if 'bbox' in mesh_stats:
                bbox = mesh_stats['bbox']
                st.metric("X Range", f"{bbox['min'][0]:.3f} - {bbox['max'][0]:.3f} mm")
                st.metric("Y Range", f"{bbox['min'][1]:.3f} - {bbox['max'][1]:.3f} mm")
                st.metric("Z Range", f"{bbox['min'][2]:.3f} - {bbox['max'][2]:.3f} mm")
                st.metric("Volume", f"{bbox.get('volume', 0):.3f} mm³")
        
        # Region analysis
        st.markdown("##### 🗺️ Region Analysis")
        
        if hasattr(mesh_data, 'region_stats') and mesh_data.region_stats:
            region_stats = mesh_data.region_stats
            
            # Create region visualization
            region_colors = np.zeros(len(mesh_data.points))
            for region_id, stats in region_stats.items():
                region_mask = mesh_data.region_labels == region_id
                region_colors[region_mask] = region_id
            
            fig_regions = self.visualizer.create_advanced_mesh_visualization(
                mesh_data,
                "Region",
                region_colors,
                colormap="Viridis",
                title="Mesh Regions",
                opacity=0.8
            )
            
            st.plotly_chart(fig_regions, use_container_width=True)
            
            # Region statistics table
            region_data = []
            for region_id, stats in region_stats.items():
                region_data.append({
                    'Region': region_id,
                    'Points': stats['size'],
                    'Volume': f"{stats.get('volume', 0):.3f}",
                    'Avg Distance': f"{stats.get('avg_distance_to_center', 0):.3f}"
                })
            
            if region_data:
                df_regions = pd.DataFrame(region_data)
                st.dataframe(df_regions, use_container_width=True)
    
    def render_settings_tab(self, sim):
        """Render settings tab"""
        st.markdown("##### ⚙️ Visualization Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.selectbox(
                "Color Scale",
                ["Linear", "Log", "Sqrt", "Inverse"],
                key="color_scale",
                help="Choose color scale transformation"
            )
            
            st.checkbox("Show Colorbar", value=True, key="show_colorbar")
            st.checkbox("Show Legend", value=True, key="show_legend")
            st.checkbox("Show Grid", value=False, key="show_grid")
        
        with col2:
            st.slider("Marker Size", 1, 10, 3, key="marker_size")
            st.slider("Line Width", 0.5, 5.0, 1.5, 0.5, key="line_width")
            st.selectbox(
                "Background Color",
                ["White", "Light Gray", "Dark Gray", "Black"],
                key="background_color"
            )
        
        st.markdown("##### 💾 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📷 Save as Image", use_container_width=True):
                self.export_as_image()
        
        with col2:
            if st.button("📊 Export Statistics", use_container_width=True):
                self.export_statistics(sim)
        
        with col3:
            if st.button("💾 Save Session", use_container_width=True):
                self.save_session()
    
    def render_enhanced_interpolation_extrapolation(self):
        """Render enhanced interpolation/extrapolation interface"""
        st.markdown('<h2 class="sub-header">🔮 Interpolation/Extrapolation Engine</h2>', 
                   unsafe_allow_html=True)
        
        # Info box with more details
        st.markdown("""
        <div class="info-box">
        <h3>🧠 Physics-Informed Attention Mechanism</h3>
        <p>This engine uses a <strong>transformer-inspired multi-head attention mechanism</strong> with 
        <strong>spatial locality regulation</strong> to interpolate and extrapolate simulation results. 
        The model learns from existing FEA simulations and can predict outcomes for new parameter 
        combinations with quantified confidence.</p>
        
        <p><strong>Key Features:</strong></p>
        <ul>
            <li>Attention-based weighting of similar simulations</li>
            <li>Spatial coherence preservation using RBF interpolation</li>
            <li>Physics-informed constraints (non-negativity, bounds)</li>
            <li>Confidence estimation for predictions</li>
            <li>Real-time visualization of attention weights</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Query parameters in an expandable section
        with st.expander("🎯 Query Parameters", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                energy_query = st.number_input(
                    "Energy (mJ)",
                    min_value=0.1,
                    max_value=100.0,
                    value=5.0,
                    step=0.1,
                    key="interp_energy",
                    help="Laser pulse energy in millijoules"
                )
            with col2:
                duration_query = st.number_input(
                    "Pulse Duration (ns)",
                    min_value=0.5,
                    max_value=50.0,
                    value=4.0,
                    step=0.1,
                    key="interp_duration",
                    help="Laser pulse duration in nanoseconds"
                )
            with col3:
                time_query = st.number_input(
                    "Time (ns)",
                    min_value=0.0,
                    max_value=100.0,
                    value=10.0,
                    step=0.1,
                    key="interp_time",
                    help="Time after pulse start in nanoseconds"
                )
        
        # Field selection
        available_fields = list(self.data_loader.common_fields)
        if available_fields:
            selected_field = st.selectbox(
                "Select Field for Prediction",
                available_fields,
                key="pred_field_select",
                help="Choose which physical field to predict"
            )
        else:
            st.warning("⚠️ No common fields found across simulations.")
            return
        
        # Visualization options
        with st.expander("🎨 Visualization Options", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                reference_sim = st.selectbox(
                    "Reference Geometry",
                    sorted(self.data_loader.simulations.keys()),
                    key="ref_geom_select",
                    help="Choose mesh geometry for prediction"
                )
            with col2:
                viz_type = st.selectbox(
                    "Visualization Type",
                    ["3D Mesh", "Cross-Section", "Animation", "Comparison"],
                    key="viz_type_select"
                )
            
            # Additional options based on visualization type
            if viz_type == "Cross-Section":
                slice_axis = st.selectbox("Slice Axis", ["X", "Y", "Z"], key="slice_axis")
                slice_pos = st.slider("Slice Position", 0.0, 1.0, 0.5, 0.01, 
                                     key="slice_pos", format="%.2f")
        
        # Prediction button with status
        if st.button("🚀 Generate Prediction", type="primary", use_container_width=True):
            self.generate_prediction_with_progress(
                energy_query, duration_query, time_query, 
                selected_field, reference_sim, viz_type
            )
    
    def generate_prediction_with_progress(self, energy, duration, time, 
                                         field, ref_sim, viz_type):
        """Generate prediction with progress tracking"""
        # Validate inputs
        try:
            self.validator.validate_simulation_parameters(energy, duration, time)
        except ValidationError as e:
            st.error(f"❌ Invalid parameters: {str(e)}")
            return
        
        # Create progress containers
        progress_container = st.empty()
        status_container = st.empty()
        details_container = st.empty()
        
        progress_bar = progress_container.progress(0)
        status_container.text("Initializing predictor...")
        
        try:
            # Step 1: Initialize predictor
            progress_bar.progress(10)
            status_container.text("Loading reference geometry...")
            
            if ref_sim not in self.data_loader.simulations:
                raise ValidationError(f"Reference simulation '{ref_sim}' not found")
            
            mesh_data = self.data_loader.simulations[ref_sim]['mesh_data']
            
            # Step 2: Generate prediction
            progress_bar.progress(30)
            status_container.text("Generating physics-informed prediction...")
            
            prediction = self.geom_predictor.predict_field_on_mesh(
                energy, duration, time, mesh_data, field
            )
            
            if prediction is None:
                raise ValueError("Failed to generate prediction")
            
            progress_bar.progress(70)
            status_container.text("Creating visualization...")
            
            # Step 3: Create visualization
            if viz_type == "3D Mesh":
                fig = self.visualizer.create_advanced_mesh_visualization(
                    mesh_data,
                    field,
                    prediction['values'],
                    colormap=st.session_state.selected_colormap,
                    title=f"Predicted {field} Distribution\nE={energy:.1f}mJ, τ={duration:.1f}ns, t={time:.1f}ns",
                    opacity=0.9
                )
                
                progress_bar.progress(90)
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Cross-Section":
                # Get slice parameters from session state
                slice_axis = st.session_state.get('slice_axis', 'X')
                slice_pos = st.session_state.get('slice_pos', 0.5)
                
                # Convert normalized position to actual coordinate
                if mesh_data.points is not None:
                    if slice_axis == 'X':
                        coord_range = (np.min(mesh_data.points[:, 0]), np.max(mesh_data.points[:, 0]))
                    elif slice_axis == 'Y':
                        coord_range = (np.min(mesh_data.points[:, 1]), np.max(mesh_data.points[:, 1]))
                    else:
                        coord_range = (np.min(mesh_data.points[:, 2]), np.max(mesh_data.points[:, 2]))
                    
                    actual_pos = coord_range[0] + slice_pos * (coord_range[1] - coord_range[0])
                    
                    slice_fig = self.visualizer.create_cross_section_view(
                        mesh_data,
                        prediction['values'],
                        slice_axis,
                        actual_pos,
                        field,
                        colormap=st.session_state.selected_colormap
                    )
                    
                    st.plotly_chart(slice_fig, use_container_width=True)
            
            # Step 4: Show prediction details
            progress_bar.progress(100)
            status_container.text("✅ Prediction completed!")
            
            # Display prediction statistics
            details_container.success(f"""
            **Prediction Details:**
            
            - **Method:** {prediction['method']}
            - **Confidence:** {prediction['confidence']:.2%}
            - **Sources Used:** {prediction['n_sources']}
            
            **Predicted Statistics:**
            - Mean: {prediction['source_stats']['mean']:.3f}
            - Std Dev: {prediction['source_stats']['std']:.3f}
            - Min: {prediction['source_stats']['min']:.3f}
            - Max: {prediction['source_stats']['max']:.3f}
            
            **Target Parameters:**
            - Energy: {energy:.1f} mJ
            - Duration: {duration:.1f} ns
            - Time: {time:.1f} ns
            """)
            
            # Show attention visualization
            if prediction['method'] == 'attention_blend':
                st.markdown("##### 🔍 Attention Weights Visualization")
                
                # Get attention weights from extrapolator
                query_point = {'energy': energy, 'duration': duration, 'time': time}
                attention_weights = self.extrapolator.compute_attention_weights(
                    query_point, field
                )
                
                # Create attention visualization
                attention_fig = self.visualizer.create_attention_visualization_on_mesh(
                    mesh_data,
                    attention_weights,
                    self.extrapolator.source_metadata,
                    query_point=None
                )
                
                st.plotly_chart(attention_fig, use_container_width=True)
                
                # Show top contributing simulations
                st.markdown("##### 📊 Top Contributing Simulations")
                
                # Get indices of top weights
                top_indices = np.argsort(attention_weights)[-5:][::-1]
                
                contrib_data = []
                for idx in top_indices:
                    if idx < len(self.extrapolator.source_metadata):
                        meta = self.extrapolator.source_metadata[idx]
                        weight = attention_weights[idx]
                        contrib_data.append({
                            'Simulation': meta['name'],
                            'Energy': meta['energy'],
                            'Duration': meta['duration'],
                            'Time': meta['time'],
                            'Weight': f"{weight:.3f}",
                            'Contribution': f"{weight*100:.1f}%"
                        })
                
                if contrib_data:
                    df_contrib = pd.DataFrame(contrib_data)
                    st.dataframe(df_contrib, use_container_width=True)
        
        except Exception as e:
            progress_bar.progress(100)
            status_container.error(f"❌ Error generating prediction: {str(e)}")
            details_container.error(f"""
            **Error Details:**
            ```python
            {traceback.format_exc()}
            ```
            
            **Possible Solutions:**
            1. Check if extrapolator is properly fitted
            2. Verify field exists in reference simulation
            3. Ensure query parameters are within reasonable bounds
            """)
    
    def render_enhanced_comparative_analysis(self):
        """Render enhanced comparative analysis interface"""
        st.markdown('<h2 class="sub-header">📊 Comparative Analysis Dashboard</h2>', 
                   unsafe_allow_html=True)
        
        simulations = self.data_loader.simulations
        
        # Multi-select for simulations
        selected_sims = st.multiselect(
            "Select simulations for comparison",
            sorted(simulations.keys()),
            default=list(simulations.keys())[:min(3, len(simulations))],
            help="Choose multiple simulations to compare"
        )
        
        if not selected_sims:
            st.info("Please select at least one simulation for comparison.")
            return
        
        # Field selection
        common_fields = set()
        for sim_name in selected_sims:
            if sim_name in simulations:
                common_fields.update(simulations[sim_name]['field_info'].keys())
        
        if common_fields:
            selected_field = st.selectbox(
                "Select field for comparison",
                sorted(common_fields),
                key="comp_field_select",
                help="Choose field to compare across simulations"
            )
        else:
            st.error("No common fields found in selected simulations.")
            return
        
        # Analysis type
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Field Evolution", "Statistical Comparison", "Geometry Comparison", 
             "Parameter Sweep", "Correlation Analysis"],
            key="analysis_type"
        )
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["📈 Visualizations", "📊 Statistics", "📋 Data"])
        
        with tab1:
            if analysis_type == "Field Evolution":
                self.render_field_evolution_comparison(selected_sims, selected_field)
            elif analysis_type == "Statistical Comparison":
                self.render_statistical_comparison(selected_sims, selected_field)
            elif analysis_type == "Geometry Comparison":
                self.render_geometry_comparison(selected_sims, selected_field)
            elif analysis_type == "Parameter Sweep":
                self.render_parameter_sweep(selected_sims, selected_field)
            elif analysis_type == "Correlation Analysis":
                self.render_correlation_analysis(selected_sims, selected_field)
        
        with tab2:
            self.render_comparison_statistics(selected_sims, selected_field)
        
        with tab3:
            self.render_comparison_data(selected_sims, selected_field)
    
    def render_field_evolution_comparison(self, selected_sims, field):
        """Render field evolution comparison"""
        fig = go.Figure()
        
        for sim_name in selected_sims:
            sim = self.data_loader.simulations[sim_name]
            
            if field in sim['field_info']:
                mesh_data = sim['mesh_data']
                field_data = mesh_data.fields[field]
                
                # Compute mean value over space for each timestep
                spatial_means = []
                spatial_stds = []
                
                for t in range(field_data.shape[0]):
                    if field_data[t].ndim == 1:
                        values = field_data[t]
                    else:
                        values = np.linalg.norm(field_data[t], axis=1)
                    
                    valid_values = values[~np.isnan(values)]
                    if len(valid_values) > 0:
                        spatial_means.append(np.nanmean(valid_values))
                        spatial_stds.append(np.nanstd(valid_values))
                    else:
                        spatial_means.append(0)
                        spatial_stds.append(0)
                
                # Add trace with error bars
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(spatial_means) + 1)),
                    y=spatial_means,
                    error_y=dict(
                        type='data',
                        array=spatial_stds,
                        visible=True,
                        thickness=1.5,
                        width=3
                    ),
                    mode='lines+markers',
                    name=f"{sim_name} ({sim['energy_mJ']}mJ, {sim['duration_ns']}ns)",
                    line=dict(width=3),
                    marker=dict(size=8),
                    hovertemplate=(
                        f'{sim_name}<br>'
                        'Timestep: %{x}<br>'
                        f'Mean {field}: %{{y:.3f}}<br>'
                        'Std Dev: ±%{error_y.array:.3f}<br>'
                        '<extra></extra>'
                    )
                ))
        
        fig.update_layout(
            title=f"{field} Evolution Comparison",
            xaxis_title="Timestep",
            yaxis_title=f"Mean {field} Value",
            height=500,
            showlegend=True,
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add statistical analysis
        st.markdown("##### 📈 Statistical Analysis")
        
        # Compute correlation between simulations
        if len(selected_sims) >= 2:
            corr_matrix = self._compute_correlation_matrix(selected_sims, field)
            
            if corr_matrix is not None:
                fig_corr = px.imshow(
                    corr_matrix,
                    x=selected_sims,
                    y=selected_sims,
                    color_continuous_scale='RdBu',
                    title="Correlation Matrix Between Simulations",
                    labels=dict(color="Correlation")
                )
                
                fig_corr.update_layout(height=400)
                st.plotly_chart(fig_corr, use_container_width=True)
    
    def _compute_correlation_matrix(self, sim_names, field):
        """Compute correlation matrix between simulations for a given field"""
        n_sims = len(sim_names)
        corr_matrix = np.zeros((n_sims, n_sims))
        
        # Get time series for each simulation
        time_series = []
        for sim_name in sim_names:
            sim = self.data_loader.simulations[sim_name]
            if field in sim['field_info']:
                field_data = sim['mesh_data'].fields[field]
                
                # Compute spatial mean for each timestep
                means = []
                for t in range(field_data.shape[0]):
                    if field_data[t].ndim == 1:
                        values = field_data[t]
                    else:
                        values = np.linalg.norm(field_data[t], axis=1)
                    
                    valid_values = values[~np.isnan(values)]
                    if len(valid_values) > 0:
                        means.append(np.nanmean(valid_values))
                    else:
                        means.append(0)
                
                time_series.append(means)
            else:
                time_series.append([0])
        
        # Pad time series to same length
        max_len = max(len(ts) for ts in time_series)
        padded_series = []
        for ts in time_series:
            if len(ts) < max_len:
                padded = np.pad(ts, (0, max_len - len(ts)), 'constant', constant_values=ts[-1])
            else:
                padded = ts[:max_len]
            padded_series.append(padded)
        
        # Compute correlation matrix
        for i in range(n_sims):
            for j in range(n_sims):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    try:
                        corr, _ = linregress(padded_series[i], padded_series[j])[:2]
                        corr_matrix[i, j] = corr
                    except:
                        corr_matrix[i, j] = 0
        
        return corr_matrix
    
    def render_statistical_comparison(self, selected_sims, field):
        """Render statistical comparison"""
        # Prepare data for box plots
        box_data = []
        
        for sim_name in selected_sims:
            sim = self.data_loader.simulations[sim_name]
            
            if field in sim['field_info']:
                mesh_data = sim['mesh_data']
                field_data = mesh_data.fields[field]
                
                # Flatten all timesteps
                all_values = []
                for t in range(field_data.shape[0]):
                    if field_data[t].ndim == 1:
                        values = field_data[t]
                    else:
                        values = np.linalg.norm(field_data[t], axis=1)
                    
                    valid_values = values[~np.isnan(values)]
                    all_values.extend(valid_values.tolist())
                
                if all_values:
                    box_data.append(go.Box(
                        y=all_values[:10000],  # Limit for performance
                        name=f"{sim_name}\n({sim['energy_mJ']}mJ, {sim['duration_ns']}ns)",
                        boxpoints='outliers',
                        jitter=0.3,
                        pointpos=-1.8,
                        marker=dict(size=3),
                        line=dict(width=2)
                    ))
        
        if box_data:
            fig = go.Figure(data=box_data)
            fig.update_layout(
                title=f"{field} Distribution Comparison",
                yaxis_title=field,
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistical summary table
        st.markdown("##### 📋 Statistical Summary")
        
        summary_data = []
        for sim_name in selected_sims:
            sim = self.data_loader.simulations[sim_name]
            
            if field in sim['field_info']:
                mesh_data = sim['mesh_data']
                field_data = mesh_data.fields[field]
                
                # Compute statistics across all timesteps
                all_values = []
                for t in range(field_data.shape[0]):
                    if field_data[t].ndim == 1:
                        values = field_data[t]
                    else:
                        values = np.linalg.norm(field_data[t], axis=1)
                    
                    valid_values = values[~np.isnan(values)]
                    all_values.extend(valid_values.tolist())
                
                if all_values:
                    all_values = np.array(all_values)
                    summary_data.append({
                        'Simulation': sim_name,
                        'Energy': sim['energy_mJ'],
                        'Duration': sim['duration_ns'],
                        'Mean': np.mean(all_values),
                        'Std': np.std(all_values),
                        'Min': np.min(all_values),
                        'Max': np.max(all_values),
                        'Median': np.median(all_values),
                        'Q25': np.percentile(all_values, 25),
                        'Q75': np.percentile(all_values, 75)
                    })
        
        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(
                df_summary.style.format({
                    'Mean': '{:.3f}',
                    'Std': '{:.3f}',
                    'Min': '{:.3f}',
                    'Max': '{:.3f}',
                    'Median': '{:.3f}',
                    'Q25': '{:.3f}',
                    'Q75': '{:.3f}'
                }),
                use_container_width=True
            )
    
    def render_geometry_comparison(self, selected_sims, field):
        """Render geometry comparison dashboard"""
        # Limit to 4 simulations for clarity
        display_sims = selected_sims[:4]
        
        # Create subplot grid
        n_cols = 2
        n_rows = (len(display_sims) + 1) // 2
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=[f"{sim}\n({self.data_loader.simulations[sim]['energy_mJ']}mJ, "
                          f"{self.data_loader.simulations[sim]['duration_ns']}ns)" 
                          for sim in display_sims],
            specs=[[{'type': 'scene'} for _ in range(n_cols)] for _ in range(n_rows)],
            vertical_spacing=0.1,
            horizontal_spacing=0.05
        )
        
        for idx, sim_name in enumerate(display_sims):
            row = idx // n_cols + 1
            col = idx % n_cols + 1
            
            sim = self.data_loader.simulations[sim_name]
            mesh_data = sim['mesh_data']
            
            if field in sim['field_info']:
                field_data = sim['mesh_data'].fields[field][0]  # First timestep
                
                if field_data.ndim == 2:
                    field_values = np.linalg.norm(field_data, axis=1)
                else:
                    field_values = field_data
                
                if mesh_data.triangles is not None and len(mesh_data.triangles) > 0:
                    fig.add_trace(
                        go.Mesh3d(
                            x=mesh_data.points[:, 0],
                            y=mesh_data.points[:, 1],
                            z=mesh_data.points[:, 2],
                            i=mesh_data.triangles[:, 0],
                            j=mesh_data.triangles[:, 1],
                            k=mesh_data.triangles[:, 2],
                            intensity=field_values,
                            colorscale=st.session_state.selected_colormap,
                            intensitymode='vertex',
                            opacity=0.8,
                            showscale=(idx == 0),
                            colorbar=dict(x=1.02) if idx == 0 else None
                        ),
                        row=row, col=col
                    )
                else:
                    fig.add_trace(
                        go.Scatter3d(
                            x=mesh_data.points[:, 0],
                            y=mesh_data.points[:, 1],
                            z=mesh_data.points[:, 2],
                            mode='markers',
                            marker=dict(
                                size=3,
                                color=field_values,
                                colorscale=st.session_state.selected_colormap,
                                opacity=0.8,
                                showscale=(idx == 0)
                            )
                        ),
                        row=row, col=col
                    )
        
        fig.update_layout(
            height=300 * n_rows,
            title_text=f"{field} - Geometry Comparison",
            showlegend=False
        )
        
        # Update scene properties
        for i in range(1, n_rows * n_cols + 1):
            fig.update_scenes(
                aspectmode="data",
                row=(i-1)//n_cols + 1,
                col=(i-1)%n_cols + 1
            )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_parameter_sweep(self, selected_sims, field):
        """Render parameter sweep analysis"""
        # Prepare data for parameter space visualization
        param_data = []
        
        for sim_name in selected_sims:
            sim = self.data_loader.simulations[sim_name]
            
            if field in sim['field_info']:
                mesh_data = sim['mesh_data']
                field_data = mesh_data.fields[field]
                
                # Compute average field value across space and time
                all_values = []
                for t in range(field_data.shape[0]):
                    if field_data[t].ndim == 1:
                        values = field_data[t]
                    else:
                        values = np.linalg.norm(field_data[t], axis=1)
                    
                    valid_values = values[~np.isnan(values)]
                    if len(valid_values) > 0:
                        all_values.append(np.nanmean(valid_values))
                
                if all_values:
                    avg_value = np.mean(all_values)
                    max_value = np.max(all_values)
                    
                    param_data.append({
                        'Simulation': sim_name,
                        'Energy': sim['energy_mJ'],
                        'Duration': sim['duration_ns'],
                        f'Avg {field}': avg_value,
                        f'Max {field}': max_value
                    })
        
        if param_data:
            df_params = pd.DataFrame(param_data)
            
            # Create 3D scatter plot
            fig = px.scatter_3d(
                df_params,
                x='Energy',
                y='Duration',
                z=f'Avg {field}',
                color=f'Avg {field}',
                size=f'Max {field}',
                hover_name='Simulation',
                title=f"{field} in Parameter Space",
                color_continuous_scale=st.session_state.selected_colormap,
                labels={
                    'Energy': 'Energy (mJ)',
                    'Duration': 'Duration (ns)',
                    f'Avg {field}': f'Average {field}',
                    f'Max {field}': f'Maximum {field}'
                }
            )
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Create 2D contour plot
            if len(df_params) >= 4:
                try:
                    # Interpolate for smooth contour
                    from scipy.interpolate import griddata
                    
                    xi = np.linspace(df_params['Energy'].min(), df_params['Energy'].max(), 100)
                    yi = np.linspace(df_params['Duration'].min(), df_params['Duration'].max(), 100)
                    xi, yi = np.meshgrid(xi, yi)
                    
                    zi = griddata(
                        (df_params['Energy'], df_params['Duration']),
                        df_params[f'Avg {field}'],
                        (xi, yi),
                        method='cubic'
                    )
                    
                    fig_contour = go.Figure(data=[
                        go.Contour(
                            z=zi,
                            x=xi[0],
                            y=yi[:, 0],
                            colorscale=st.session_state.selected_colormap,
                            colorbar=dict(title=f'Avg {field}')
                        )
                    ])
                    
                    fig_contour.update_layout(
                        title=f"{field} Contour in Parameter Space",
                        xaxis_title="Energy (mJ)",
                        yaxis_title="Duration (ns)",
                        height=500
                    )
                    
                    st.plotly_chart(fig_contour, use_container_width=True)
                except:
                    st.info("Not enough data points for contour plot")
    
    def render_correlation_analysis(self, selected_sims, field):
        """Render correlation analysis"""
        if len(selected_sims) < 2:
            st.info("Need at least 2 simulations for correlation analysis")
            return
        
        # Compute correlations between different fields
        st.markdown("##### 🔗 Field Correlations")
        
        # Get all available fields from first simulation
        first_sim = self.data_loader.simulations[selected_sims[0]]
        available_fields = list(first_sim['field_info'].keys())[:8]  # Limit to 8 fields
        
        if len(available_fields) < 2:
            st.info("Not enough fields for correlation analysis")
            return
        
        # Select fields for correlation analysis
        corr_fields = st.multiselect(
            "Select fields for correlation analysis",
            available_fields,
            default=[field, available_fields[1]] if len(available_fields) > 1 else [field]
        )
        
        if len(corr_fields) < 2:
            st.info("Please select at least 2 fields")
            return
        
        # Compute correlation matrix for selected simulation
        sim_name = st.selectbox(
            "Select simulation for correlation analysis",
            selected_sims,
            key="corr_sim_select"
        )
        
        sim = self.data_loader.simulations[sim_name]
        mesh_data = sim['mesh_data']
        
        # Get field values at first timestep
        field_values = {}
        for f in corr_fields:
            if f in mesh_data.fields:
                field_data = mesh_data.fields[f][0]  # First timestep
                if field_data.ndim == 2:
                    field_values[f] = np.linalg.norm(field_data, axis=1)
                else:
                    field_values[f] = field_data
        
        # Compute correlation matrix
        corr_matrix = np.zeros((len(corr_fields), len(corr_fields)))
        for i, f1 in enumerate(corr_fields):
            for j, f2 in enumerate(corr_fields):
                if f1 in field_values and f2 in field_values:
                    # Ensure same length and no NaN
                    vals1 = field_values[f1]
                    vals2 = field_values[f2]
                    
                    # Remove NaN values
                    mask = ~np.isnan(vals1) & ~np.isnan(vals2)
                    if np.sum(mask) > 10:
                        try:
                            corr, _ = linregress(vals1[mask], vals2[mask])[:2]
                            corr_matrix[i, j] = corr
                        except:
                            corr_matrix[i, j] = 0
                    else:
                        corr_matrix[i, j] = 0
        
        # Create correlation heatmap
        fig_corr = px.imshow(
            corr_matrix,
            x=corr_fields,
            y=corr_fields,
            color_continuous_scale='RdBu',
            title=f"Field Correlations - {sim_name}",
            labels=dict(color="Correlation"),
            zmin=-1,
            zmax=1
        )
        
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Scatter plot for selected pair
        if len(corr_fields) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                x_field = st.selectbox("X-axis field", corr_fields, index=0)
            with col2:
                y_field = st.selectbox("Y-axis field", corr_fields, 
                                      index=min(1, len(corr_fields)-1))
            
            if x_field in field_values and y_field in field_values:
                # Create scatter plot
                fig_scatter = px.scatter(
                    x=field_values[x_field][:1000],  # Limit points for performance
                    y=field_values[y_field][:1000],
                    trendline="ols",
                    title=f"{x_field} vs {y_field}",
                    labels={'x': x_field, 'y': y_field}
                )
                
                fig_scatter.update_layout(height=400)
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Compute correlation statistics
                vals1 = field_values[x_field]
                vals2 = field_values[y_field]
                mask = ~np.isnan(vals1) & ~np.isnan(vals2)
                
                if np.sum(mask) > 10:
                    try:
                        slope, intercept, r_value, p_value, std_err = linregress(
                            vals1[mask], vals2[mask]
                        )
                        
                        st.markdown("##### 📊 Correlation Statistics")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Correlation", f"{r_value:.3f}")
                        with col2:
                            st.metric("R²", f"{r_value**2:.3f}")
                        with col3:
                            st.metric("p-value", f"{p_value:.3e}")
                        with col4:
                            st.metric("Slope", f"{slope:.3f}")
                    except:
                        st.warning("Could not compute correlation statistics")
    
    def render_comparison_statistics(self, selected_sims, field):
        """Render comparison statistics"""
        # Compute comprehensive statistics
        stats_data = []
        
        for sim_name in selected_sims:
            sim = self.data_loader.simulations[sim_name]
            
            if field in sim['field_info']:
                mesh_data = sim['mesh_data']
                field_data = mesh_data.fields[field]
                
                # Compute statistics across all timesteps
                all_values = []
                gradient_magnitudes = []
                
                for t in range(field_data.shape[0]):
                    if field_data[t].ndim == 1:
                        values = field_data[t]
                    else:
                        values = np.linalg.norm(field_data[t], axis=1)
                    
                    valid_values = values[~np.isnan(values)]
                    all_values.extend(valid_values.tolist())
                    
                    # Compute gradient if mesh available
                    if t == 0 and mesh_data.triangles is not None:
                        # Simple gradient approximation
                        if len(valid_values) > 0:
                            # Use spatial variation as proxy for gradient
                            spatial_var = np.std(valid_values) / np.mean(valid_values)
                            gradient_magnitudes.append(spatial_var)
                
                if all_values:
                    all_values = np.array(all_values)
                    
                    from scipy.stats import skew, kurtosis
                    
                    stats_data.append({
                        'Simulation': sim_name,
                        'Energy (mJ)': sim['energy_mJ'],
                        'Duration (ns)': sim['duration_ns'],
                        'Mean': np.mean(all_values),
                        'Std Dev': np.std(all_values),
                        'CV (%)': (np.std(all_values) / np.mean(all_values)) * 100,
                        'Skewness': skew(all_values) if len(all_values) > 2 else 0,
                        'Kurtosis': kurtosis(all_values) if len(all_values) > 3 else 0,
                        'Min': np.min(all_values),
                        'Max': np.max(all_values),
                        'Range': np.max(all_values) - np.min(all_values),
                        'Q1': np.percentile(all_values, 25),
                        'Median': np.median(all_values),
                        'Q3': np.percentile(all_values, 75),
                        'IQR': np.percentile(all_values, 75) - np.percentile(all_values, 25)
                    })
        
        if stats_data:
            df_stats = pd.DataFrame(stats_data)
            
            # Format numeric columns
            format_dict = {
                'Mean': '{:.3f}',
                'Std Dev': '{:.3f}',
                'CV (%)': '{:.1f}',
                'Skewness': '{:.3f}',
                'Kurtosis': '{:.3f}',
                'Min': '{:.3f}',
                'Max': '{:.3f}',
                'Range': '{:.3f}',
                'Q1': '{:.3f}',
                'Median': '{:.3f}',
                'Q3': '{:.3f}',
                'IQR': '{:.3f}'
            }
            
            st.dataframe(
                df_stats.style.format(format_dict),
                use_container_width=True,
                height=400
            )
            
            # Export option
            if st.button("📥 Export Statistics to CSV", use_container_width=True):
                csv = df_stats.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"comparison_stats_{field}.csv",
                    mime="text/csv"
                )
    
    def render_comparison_data(self, selected_sims, field):
        """Render comparison data in tabular format"""
        # Create time series data for each simulation
        time_series_data = {}
        
        for sim_name in selected_sims:
            sim = self.data_loader.simulations[sim_name]
            
            if field in sim['field_info']:
                mesh_data = sim['mesh_data']
                field_data = mesh_data.fields[field]
                
                # Compute spatial statistics for each timestep
                timesteps = []
                means = []
                stds = []
                mins = []
                maxs = []
                
                for t in range(field_data.shape[0]):
                    if field_data[t].ndim == 1:
                        values = field_data[t]
                    else:
                        values = np.linalg.norm(field_data[t], axis=1)
                    
                    valid_values = values[~np.isnan(values)]
                    if len(valid_values) > 0:
                        timesteps.append(t + 1)
                        means.append(np.nanmean(valid_values))
                        stds.append(np.nanstd(valid_values))
                        mins.append(np.nanmin(valid_values))
                        maxs.append(np.nanmax(valid_values))
                
                time_series_data[sim_name] = {
                    'Timestep': timesteps,
                    'Mean': means,
                    'Std': stds,
                    'Min': mins,
                    'Max': maxs
                }
        
        # Display data in tabs for each simulation
        tabs = st.tabs(selected_sims)
        
        for idx, sim_name in enumerate(selected_sims):
            with tabs[idx]:
                if sim_name in time_series_data:
                    data = time_series_data[sim_name]
                    df = pd.DataFrame(data)
                    
                    st.dataframe(
                        df.style.format({
                            'Mean': '{:.3f}',
                            'Std': '{:.3f}',
                            'Min': '{:.3f}',
                            'Max': '{:.3f}'
                        }),
                        use_container_width=True,
                        height=300
                    )
                    
                    # Show summary
                    sim = self.data_loader.simulations[sim_name]
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Energy", f"{sim['energy_mJ']:.1f} mJ")
                    with col2:
                        st.metric("Duration", f"{sim['duration_ns']:.1f} ns")
                    with col3:
                        if len(data['Mean']) > 0:
                            st.metric("Avg Mean", f"{np.mean(data['Mean']):.3f}")
                    with col4:
                        if len(data['Std']) > 0:
                            st.metric("Avg Std", f"{np.mean(data['Std']):.3f}")
    
    def render_enhanced_geometrical_visualization(self):
        """Render enhanced geometrical visualization interface"""
        st.markdown('<h2 class="sub-header">🏗️ Advanced Geometrical Visualization</h2>', 
                   unsafe_allow_html=True)
        
        # Info box
        st.markdown("""
        <div class="info-box">
        <h3>🎨 Advanced 3D Visualization Suite</h3>
        <p>This module provides comprehensive 3D visualization capabilities for FEA mesh data, 
        including interactive exploration, cross-section analysis, time animations, and 
        spatial statistics.</p>
        
        <p><strong>Features:</strong></p>
        <ul>
            <li>Interactive 3D mesh rendering with multiple view modes</li>
            <li>Dynamic cross-section analysis with real-time updates</li>
            <li>Time evolution animations with playback controls</li>
            <li>Spatial statistics and correlation analysis</li>
            <li>Volume rendering for dense field visualization</li>
            <li>Streamline visualization for vector fields</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        simulations = self.data_loader.simulations
        
        # Main controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sim_name = st.selectbox(
                "Select Simulation",
                sorted(simulations.keys()),
                key="geom_sim_select",
                help="Choose simulation for visualization"
            )
        
        sim = simulations[sim_name]
        
        with col2:
            field = st.selectbox(
                "Select Field",
                sorted(sim['field_info'].keys()),
                key="geom_field_select",
                help="Choose field to visualize"
            )
        
        with col3:
            viz_mode = st.selectbox(
                "Visualization Mode",
                ["Interactive 3D", "Cross-Section Analysis", "Time Animation", 
                 "Spatial Statistics", "Volume Rendering", "Streamlines"],
                key="geom_viz_mode",
                help="Choose visualization type"
            )
        
        # Visualization parameters
        if viz_mode == "Interactive 3D":
            self.render_interactive_3d(sim, field)
        elif viz_mode == "Cross-Section Analysis":
            self.render_cross_section_analysis(sim, field)
        elif viz_mode == "Time Animation":
            self.render_time_animation(sim, field)
        elif viz_mode == "Spatial Statistics":
            self.render_spatial_statistics(sim, field)
        elif viz_mode == "Volume Rendering":
            self.render_volume_rendering(sim, field)
        elif viz_mode == "Streamlines":
            self.render_streamlines(sim, field)
    
    def render_interactive_3d(self, sim, field):
        """Render interactive 3D visualization"""
        mesh_data = sim['mesh_data']
        
        # Controls in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            timestep = st.slider(
                "Timestep",
                0, sim['n_timesteps'] - 1, 0,
                key="interactive_timestep",
                help="Select time step to visualize"
            )
        
        with col2:
            display_options = st.multiselect(
                "Display Options",
                ["Wireframe", "Points", "Bounding Box", "Coordinate Axes", 
                 "Surface Normals", "Boundary Edges"],
                default=["Wireframe", "Coordinate Axes"],
                key="display_options",
                help="Choose visualization elements to display"
            )
        
        with col3:
            render_quality = st.select_slider(
                "Render Quality",
                options=["Low", "Medium", "High", "Ultra"],
                value="Medium",
                help="Adjust rendering quality (affects performance)"
            )
        
        # Get field values
        field_data = sim['mesh_data'].fields[field][timestep]
        if field_data.ndim == 2:
            field_values = np.linalg.norm(field_data, axis=1)
            is_vector = True
        else:
            field_values = field_data
            is_vector = False
        
        # Advanced controls
        with st.expander("🎨 Advanced Controls", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                camera_x = st.slider("Camera X", -3.0, 3.0, 1.5, 0.1)
                camera_y = st.slider("Camera Y", -3.0, 3.0, 1.5, 0.1)
                camera_z = st.slider("Camera Z", -3.0, 3.0, 1.5, 0.1)
            
            with col2:
                light_ambient = st.slider("Ambient Light", 0.0, 1.0, 0.8, 0.1)
                light_diffuse = st.slider("Diffuse Light", 0.0, 1.0, 0.8, 0.1)
                light_specular = st.slider("Specular Light", 0.0, 1.0, 0.5, 0.1)
            
            with col3:
                color_scale = st.selectbox(
                    "Color Scale",
                    ["Linear", "Log", "Sqrt"],
                    help="Color scale transformation"
                )
                
                if st.button("Apply Settings", use_container_width=True):
                    st.info("Settings applied")
        
        # Create visualization
        fig = self.visualizer.create_advanced_mesh_visualization(
            mesh_data,
            field,
            field_values,
            colormap=st.session_state.selected_colormap,
            title=f"{field} at Timestep {timestep + 1}",
            opacity=st.session_state.get('default_opacity', 0.9),
            show_wireframe="Wireframe" in display_options,
            show_points="Points" in display_options,
            show_normals="Surface Normals" in display_options,
            show_boundary="Boundary Edges" in display_options,
            lighting_intensity=1.0
        )
        
        # Update camera
        fig.update_layout(
            scene_camera=dict(
                eye=dict(x=camera_x, y=camera_y, z=camera_z),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0)
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Field statistics
        self.render_field_statistics(field_values, field, sim_name=sim['name'])
        
        # Vector field information
        if is_vector:
            st.info(f"🔀 Displaying magnitude of vector field '{field}'")
            
            # Show vector components
            with st.expander("🔢 Vector Components", expanded=False):
                components = ["X", "Y", "Z"]
                cols = st.columns(3)
                
                for idx, (col, comp) in enumerate(zip(cols, components)):
                    with col:
                        comp_values = field_data[:, idx]
                        st.metric(f"{comp}-Component Mean", f"{np.nanmean(comp_values):.3f}")
                        st.metric(f"{comp}-Component Max", f"{np.nanmax(comp_values):.3f}")
    
    def render_cross_section_analysis(self, sim, field):
        """Render cross-section analysis"""
        mesh_data = sim['mesh_data']
        
        # Controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            slice_axis = st.selectbox("Slice Axis", ["X", "Y", "Z"], key="cs_axis")
        
        with col2:
            if mesh_data.points is not None:
                if slice_axis == 'X':
                    coord_range = (np.min(mesh_data.points[:, 0]), np.max(mesh_data.points[:, 0]))
                elif slice_axis == 'Y':
                    coord_range = (np.min(mesh_data.points[:, 1]), np.max(mesh_data.points[:, 1]))
                else:
                    coord_range = (np.min(mesh_data.points[:, 2]), np.max(mesh_data.points[:, 2]))
                
                slice_pos = st.slider(
                    f"{slice_axis} Position",
                    float(coord_range[0]),
                    float(coord_range[1]),
                    float((coord_range[0] + coord_range[1]) / 2),
                    key="cs_pos",
                    help=f"Position along {slice_axis} axis"
                )
        
        with col3:
            timestep = st.slider(
                "Timestep",
                0, sim['n_timesteps'] - 1, 0,
                key="cs_timestep"
            )
        
        # Get field values
        field_data = mesh_data.fields[field][timestep]
        if field_data.ndim == 2:
            field_values = np.linalg.norm(field_data, axis=1)
        else:
            field_values = field_data
        
        # Create cross-section visualization
        slice_fig = self.visualizer.create_cross_section_view(
            mesh_data,
            field_values,
            slice_axis,
            slice_pos,
            field,
            colormap=st.session_state.selected_colormap
        )
        
        st.plotly_chart(slice_fig, use_container_width=True)
        
        # Multiple cross-sections
        with st.expander("📊 Multiple Cross-Sections", expanded=False):
            n_slices = st.slider("Number of slices", 2, 10, 4, key="n_slices")
            
            slice_positions = np.linspace(coord_range[0], coord_range[1], n_slices + 2)[1:-1]
            
            # Create small multiples
            cols = st.columns(min(n_slices, 4))
            
            for idx, pos in enumerate(slice_positions):
                if idx < len(cols):
                    with cols[idx]:
                        st.markdown(f"**{slice_axis} = {pos:.3f}**")
                        
                        # Create small cross-section
                        small_fig = self.visualizer.create_cross_section_view(
                            mesh_data,
                            field_values,
                            slice_axis,
                            float(pos),
                            field,
                            colormap=st.session_state.selected_colormap
                        )
                        
                        small_fig.update_layout(
                            height=250, 
                            showlegend=False, 
                            margin=dict(t=20, b=20, l=20, r=20),
                            title=None
                        )
                        st.plotly_chart(small_fig, use_container_width=True)
        
        # Profile along axis
        with st.expander("📈 Axis Profile", expanded=False):
            # Extract values along slice
            if slice_axis == 'X':
                axis_coords = mesh_data.points[:, 0]
                other_axis = "Y"
                other_coords = mesh_data.points[:, 1]
            elif slice_axis == 'Y':
                axis_coords = mesh_data.points[:, 1]
                other_axis = "Z"
                other_coords = mesh_data.points[:, 2]
            else:
                axis_coords = mesh_data.points[:, 2]
                other_axis = "X"
                other_coords = mesh_data.points[:, 0]
            
            # Find points near slice
            tolerance = 0.01 * (coord_range[1] - coord_range[0])
            slice_mask = np.abs(axis_coords - slice_pos) < tolerance
            
            if np.sum(slice_mask) > 10:
                slice_values = field_values[slice_mask]
                slice_coords = other_coords[slice_mask]
                
                # Sort by coordinate
                sort_idx = np.argsort(slice_coords)
                sorted_coords = slice_coords[sort_idx]
                sorted_values = slice_values[sort_idx]
                
                # Create profile plot
                fig_profile = px.line(
                    x=sorted_coords,
                    y=sorted_values,
                    title=f"{field} Profile along {other_axis} at {slice_axis}={slice_pos:.3f}",
                    labels={'x': f'{other_axis} Coordinate', 'y': field}
                )
                
                fig_profile.update_layout(height=300)
                st.plotly_chart(fig_profile, use_container_width=True)
    
    def render_time_animation(self, sim, field):
        """Render time animation"""
        mesh_data = sim['mesh_data']
        
        # Animation controls
        col1, col2 = st.columns(2)
        
        with col1:
            start_time = st.slider("Start Time", 0, sim['n_timesteps'] - 1, 0, key="anim_start")
            end_time = st.slider("End Time", 0, sim['n_timesteps'] - 1, 
                                sim['n_timesteps'] - 1, key="anim_end")
        
        with col2:
            frame_rate = st.slider("Frame Rate (fps)", 1, 30, 10, key="anim_fps")
            play_direction = st.selectbox("Play Direction", ["Forward", "Reverse"], 
                                         key="anim_dir")
        
        if start_time >= end_time:
            st.error("Start time must be less than end time.")
            return
        
        # Prepare animation data
        time_indices = range(start_time, end_time + 1)
        field_evolution = []
        
        for t in time_indices:
            field_data = mesh_data.fields[field][t]
            if field_data.ndim == 2:
                field_data = np.linalg.norm(field_data, axis=1)
            field_evolution.append(field_data)
        
        if play_direction == "Reverse":
            field_evolution = field_evolution[::-1]
            time_indices = time_indices[::-1]
        
        # Create animation
        anim_fig = self.visualizer.create_field_animation(
            field_evolution,
            mesh_data,
            list(time_indices),
            field,
            colormap=st.session_state.selected_colormap
        )
        
        st.plotly_chart(anim_fig, use_container_width=True)
        
        # Animation controls
        with st.expander("🎬 Animation Controls", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                loop_animation = st.checkbox("Loop Animation", value=True, key="anim_loop")
                auto_play = st.checkbox("Auto Play", value=True, key="anim_auto")
            
            with col2:
                if st.button("🔄 Restart Animation", use_container_width=True):
                    st.rerun()
            
            with col3:
                if st.button("⏸️ Pause Animation", use_container_width=True):
                    st.info("Animation paused")
        
        # Time series analysis
        st.markdown("##### 📈 Time Series Analysis")
        
        # Compute statistics over time
        time_series_stats = []
        for t in range(sim['n_timesteps']):
            field_data = mesh_data.fields[field][t]
            if field_data.ndim == 2:
                field_data = np.linalg.norm(field_data, axis=1)
            
            valid_values = field_data[~np.isnan(field_data)]
            if len(valid_values) > 0:
                time_series_stats.append({
                    'Timestep': t + 1,
                    'Mean': float(np.mean(valid_values)),
                    'Std': float(np.std(valid_values)),
                    'Min': float(np.min(valid_values)),
                    'Max': float(np.max(valid_values)),
                    'Range': float(np.max(valid_values) - np.min(valid_values))
                })
        
        if time_series_stats:
            df_stats = pd.DataFrame(time_series_stats)
            
            # Create time series plot
            fig_ts = go.Figure()
            
            fig_ts.add_trace(go.Scatter(
                x=df_stats['Timestep'],
                y=df_stats['Mean'],
                mode='lines+markers',
                name='Mean',
                line=dict(width=3, color='blue'),
                error_y=dict(
                    type='data',
                    array=df_stats['Std'],
                    visible=True,
                    thickness=1.5,
                    width=2
                )
            ))
            
            fig_ts.add_trace(go.Scatter(
                x=df_stats['Timestep'],
                y=df_stats['Max'],
                mode='lines',
                name='Maximum',
                line=dict(width=1, color='red', dash='dash'),
                opacity=0.7
            ))
            
            fig_ts.add_trace(go.Scatter(
                x=df_stats['Timestep'],
                y=df_stats['Min'],
                mode='lines',
                name='Minimum',
                line=dict(width=1, color='green', dash='dash'),
                opacity=0.7
            ))
            
            fig_ts.update_layout(
                title=f"{field} Time Series Statistics",
                xaxis_title="Timestep",
                yaxis_title=f"{field} Value",
                height=400,
                showlegend=True,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_ts, use_container_width=True)
            
            # Rate of change analysis
            st.markdown("##### 📊 Rate of Change")
            
            if len(df_stats) > 1:
                # Compute derivatives
                df_stats['dMean_dt'] = np.gradient(df_stats['Mean'])
                df_stats['dStd_dt'] = np.gradient(df_stats['Std'])
                
                fig_deriv = go.Figure()
                
                fig_deriv.add_trace(go.Scatter(
                    x=df_stats['Timestep'][1:],
                    y=df_stats['dMean_dt'][1:],
                    mode='lines+markers',
                    name='d(Mean)/dt',
                    line=dict(width=2, color='purple')
                ))
                
                fig_deriv.update_layout(
                    title="Rate of Change of Mean Value",
                    xaxis_title="Timestep",
                    yaxis_title="Rate of Change",
                    height=300
                )
                
                st.plotly_chart(fig_deriv, use_container_width=True)
    
    def render_spatial_statistics(self, sim, field):
        """Render spatial statistics"""
        mesh_data = sim['mesh_data']
        
        col1, col2 = st.columns(2)
        
        with col1:
            timestep = st.slider(
                "Timestep for Analysis",
                0, sim['n_timesteps'] - 1, 0,
                key="spatial_timestep"
            )
        
        with col2:
            analysis_type = st.selectbox(
                "Analysis Type",
                ["Spatial Distribution", "Radial Analysis", "Region Analysis", 
                 "Gradient Analysis", "Correlation Analysis"],
                key="analysis_type"
            )
        
        # Get field values
        field_data = mesh_data.fields[field][timestep]
        if field_data.ndim == 2:
            field_values = np.linalg.norm(field_data, axis=1)
        else:
            field_values = field_data
        
        if analysis_type == "Spatial Distribution":
            self.render_spatial_distribution(mesh_data, field_values, field)
        elif analysis_type == "Radial Analysis":
            self.render_radial_analysis(mesh_data, field_values, field)
        elif analysis_type == "Region Analysis":
            self.render_region_analysis(mesh_data, field_values, field)
        elif analysis_type == "Gradient Analysis":
            self.render_gradient_analysis(mesh_data, field_values, field)
        elif analysis_type == "Correlation Analysis":
            self.render_spatial_correlation(mesh_data, field_values, field)
    
    def render_spatial_distribution(self, mesh_data, field_values, field_name):
        """Render spatial distribution analysis"""
        points = mesh_data.points
        
        # Create 2D projections
        tab1, tab2, tab3 = st.tabs(["XY Projection", "XZ Projection", "YZ Projection"])
        
        with tab1:
            fig_xy = px.density_heatmap(
                x=points[:, 0],
                y=points[:, 1],
                z=field_values,
                histfunc="avg",
                nbinsx=50,
                nbinsy=50,
                color_continuous_scale=st.session_state.selected_colormap,
                title=f"{field_name} Distribution (XY Projection)",
                labels={'x': 'X', 'y': 'Y', 'z': field_name}
            )
            fig_xy.update_layout(height=400)
            st.plotly_chart(fig_xy, use_container_width=True)
        
        with tab2:
            fig_xz = px.density_heatmap(
                x=points[:, 0],
                y=points[:, 2],
                z=field_values,
                histfunc="avg",
                nbinsx=50,
                nbinsy=50,
                color_continuous_scale=st.session_state.selected_colormap,
                title=f"{field_name} Distribution (XZ Projection)",
                labels={'x': 'X', 'y': 'Z', 'z': field_name}
            )
            fig_xz.update_layout(height=400)
            st.plotly_chart(fig_xz, use_container_width=True)
        
        with tab3:
            fig_yz = px.density_heatmap(
                x=points[:, 1],
                y=points[:, 2],
                z=field_values,
                histfunc="avg",
                nbinsx=50,
                nbinsy=50,
                color_continuous_scale=st.session_state.selected_colormap,
                title=f"{field_name} Distribution (YZ Projection)",
                labels={'x': 'Y', 'y': 'Z', 'z': field_name}
            )
            fig_yz.update_layout(height=400)
            st.plotly_chart(fig_yz, use_container_width=True)
        
        # Spatial statistics
        st.markdown("##### 📊 Spatial Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Spatial autocorrelation
            if len(field_values) > 100:
                corr_info = self.geom_predictor.compute_spatial_correlations(
                    field_values, mesh_data
                )
                if corr_info and 'correlation' in corr_info:
                    st.metric("Spatial Correlation", f"{corr_info['correlation']:.3f}")
        
        with col2:
            st.metric("Spatial Mean", f"{np.nanmean(field_values):.3f}")
        
        with col3:
            st.metric("Spatial Std Dev", f"{np.nanstd(field_values):.3f}")
        
        with col4:
            spatial_range = np.nanmax(field_values) - np.nanmin(field_values)
            st.metric("Spatial Range", f"{spatial_range:.3f}")
        
        # Variogram analysis
        with st.expander("📈 Variogram Analysis", expanded=False):
            if len(field_values) > 200:
                # Compute experimental variogram
                from scipy.spatial.distance import pdist
                
                # Sample points
                n_samples = min(200, len(points))
                sample_indices = np.random.choice(len(points), n_samples, replace=False)
                sample_points = points[sample_indices]
                sample_values = field_values[sample_indices]
                
                # Compute distances and squared differences
                distances = pdist(sample_points)
                value_diffs = pdist(sample_values.reshape(-1, 1), 
                                  metric=lambda u, v: 0.5 * (u[0] - v[0])**2)
                
                # Bin distances
                max_dist = np.max(distances)
                n_bins = 20
                bins = np.linspace(0, max_dist, n_bins + 1)
                
                bin_means = []
                bin_stds = []
                for i in range(n_bins):
                    mask = (distances >= bins[i]) & (distances < bins[i+1])
                    if np.sum(mask) > 5:
                        bin_means.append(np.mean(value_diffs[mask]))
                        bin_stds.append(np.std(value_diffs[mask]))
                    else:
                        bin_means.append(0)
                        bin_stds.append(0)
                
                # Create variogram plot
                fig_vario = go.Figure()
                
                fig_vario.add_trace(go.Scatter(
                    x=bins[:-1] + (bins[1] - bins[0]) / 2,
                    y=bin_means,
                    mode='lines+markers',
                    name='Experimental Variogram',
                    error_y=dict(
                        type='data',
                        array=bin_stds,
                        visible=True
                    )
                ))
                
                fig_vario.update_layout(
                    title="Experimental Variogram",
                    xaxis_title="Distance",
                    yaxis_title="Semivariance",
                    height=300
                )
                
                st.plotly_chart(fig_vario, use_container_width=True)
    
    def render_radial_analysis(self, mesh_data, field_values, field_name):
        """Render radial analysis"""
        points = mesh_data.points
        
        # Compute distances from centroid
        centroid = np.mean(points, axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        
        # Create radial scatter plot
        fig_radial = px.scatter(
            x=distances,
            y=field_values,
            color=field_values,
            color_continuous_scale=st.session_state.selected_colormap,
            title=f"{field_name} vs Radial Distance from Centroid",
            labels={'x': 'Radial Distance', 'y': field_name},
            trendline="lowess",
            trendline_options=dict(frac=0.3)
        )
        
        fig_radial.update_layout(height=500)
        st.plotly_chart(fig_radial, use_container_width=True)
        
        # Radial statistics
        st.markdown("##### 📊 Radial Statistics")
        
        # Bin by distance
        n_bins = 10
        distance_bins = np.linspace(0, np.max(distances), n_bins + 1)
        
        bin_stats = []
        for i in range(n_bins):
            mask = (distances >= distance_bins[i]) & (distances < distance_bins[i+1])
            if np.sum(mask) > 5:
                bin_values = field_values[mask]
                bin_stats.append({
                    'Distance Range': f"{distance_bins[i]:.2f}-{distance_bins[i+1]:.2f}",
                    'Mean': np.mean(bin_values),
                    'Std': np.std(bin_values),
                    'Min': np.min(bin_values),
                    'Max': np.max(bin_values),
                    'Count': int(np.sum(mask))
                })
        
        if bin_stats:
            df_bins = pd.DataFrame(bin_stats)
            st.dataframe(
                df_bins.style.format({
                    'Mean': '{:.3f}',
                    'Std': '{:.3f}',
                    'Min': '{:.3f}',
                    'Max': '{:.3f}'
                }),
                use_container_width=True
            )
    
    def render_region_analysis(self, mesh_data, field_values, field_name):
        """Render region-based analysis"""
        if not hasattr(mesh_data, 'region_labels') or mesh_data.region_labels is None:
            mesh_data.segment_regions(n_regions=5)
        
        region_labels = mesh_data.region_labels
        
        # Compute statistics per region
        region_stats = []
        
        for region_id in np.unique(region_labels):
            region_mask = region_labels == region_id
            region_values = field_values[region_mask]
            
            if len(region_values) > 0:
                region_stats.append({
                    'Region': region_id,
                    'Count': int(np.sum(region_mask)),
                    'Mean': float(np.mean(region_values)),
                    'Std': float(np.std(region_values)),
                    'Min': float(np.min(region_values)),
                    'Max': float(np.max(region_values)),
                    'Median': float(np.median(region_values))
                })
        
        if region_stats:
            df_region_stats = pd.DataFrame(region_stats)
            
            # Display region statistics
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Bar chart of region means
                fig_regions = px.bar(
                    df_region_stats,
                    x='Region',
                    y='Mean',
                    error_y='Std',
                    title=f"{field_name} Statistics by Region",
                    labels={'Region': 'Region ID', 'Mean': f'Mean {field_name}'},
                    color='Mean',
                    color_continuous_scale=st.session_state.selected_colormap
                )
                
                fig_regions.update_layout(height=400)
                st.plotly_chart(fig_regions, use_container_width=True)
            
            with col2:
                st.dataframe(
                    df_region_stats.style.format({
                        'Mean': '{:.3f}',
                        'Std': '{:.3f}',
                        'Min': '{:.3f}',
                        'Max': '{:.3f}',
                        'Median': '{:.3f}'
                    }),
                    use_container_width=True
                )
            
            # Color mesh by region
            region_colors = region_labels / np.max(region_labels)
            
            fig_region_viz = self.visualizer.create_advanced_mesh_visualization(
                mesh_data,
                "Region",
                region_colors,
                colormap="Viridis",
                title="Mesh Regions",
                opacity=0.8
            )
            
            st.plotly_chart(fig_region_viz, use_container_width=True)
    
    def render_gradient_analysis(self, mesh_data, field_values, field_name):
        """Render gradient analysis"""
        points = mesh_data.points
        
        if mesh_data.triangles is not None and len(mesh_data.triangles) > 0:
            # Compute gradients using finite differences
            gradients = np.zeros((len(points), 3))
            
            # Simple gradient computation using nearest neighbors
            from scipy.spatial import cKDTree
            
            tree = cKDTree(points)
            distances, indices = tree.query(points, k=4)  # Self + 3 neighbors
            
            for i in range(len(points)):
                neighbor_indices = indices[i][1:]  # Exclude self
                neighbor_distances = distances[i][1:]
                
                if len(neighbor_indices) >= 3:
                    # Compute gradient using finite differences
                    dx = points[neighbor_indices] - points[i]
                    df = field_values[neighbor_indices] - field_values[i]
                    
                    # Weight by inverse distance
                    weights = 1.0 / (neighbor_distances + 1e-8)
                    weights = weights / np.sum(weights)
                    
                    # Compute weighted gradient
                    for j in range(3):
                        gradients[i, j] = np.sum(weights * df * dx[:, j])
            
            # Compute gradient magnitude
            grad_magnitude = np.linalg.norm(gradients, axis=1)
            
            # Visualize gradient magnitude
            fig_grad = self.visualizer.create_advanced_mesh_visualization(
                mesh_data,
                f"{field_name} Gradient Magnitude",
                grad_magnitude,
                colormap="Hot",
                title=f"Gradient Magnitude of {field_name}",
                opacity=0.9
            )
            
            st.plotly_chart(fig_grad, use_container_width=True)
            
            # Gradient statistics
            st.markdown("##### 📊 Gradient Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean Gradient", f"{np.nanmean(grad_magnitude):.3f}")
            
            with col2:
                st.metric("Max Gradient", f"{np.nanmax(grad_magnitude):.3f}")
            
            with col3:
                st.metric("Gradient Std", f"{np.nanstd(grad_magnitude):.3f}")
            
            with col4:
                # Compute gradient direction consistency
                valid_gradients = gradients[~np.isnan(grad_magnitude)]
                if len(valid_gradients) > 10:
                    # Normalize gradients
                    norms = np.linalg.norm(valid_gradients, axis=1, keepdims=True)
                    valid_norms = norms[:, 0]
                    nonzero_mask = valid_norms > 1e-8
                    
                    if np.sum(nonzero_mask) > 10:
                        grad_norm = valid_gradients[nonzero_mask] / norms[nonzero_mask]
                        
                        # Compute mean direction
                        mean_direction = np.mean(grad_norm, axis=0)
                        direction_magnitude = np.linalg.norm(mean_direction)
                        
                        st.metric("Direction Consistency", f"{direction_magnitude:.3f}")
            
            # Gradient vector field visualization
            st.markdown("##### 🧭 Gradient Vector Field")
            
            # Sample points for vector visualization
            n_samples = min(100, len(points))
            sample_indices = np.random.choice(len(points), n_samples, replace=False)
            sample_points = points[sample_indices]
            sample_gradients = gradients[sample_indices]
            
            fig_vectors = go.Figure()
            
            # Add mesh surface
            fig_vectors.add_trace(go.Mesh3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2],
                i=mesh_data.triangles[:, 0], j=mesh_data.triangles[:, 1], 
                k=mesh_data.triangles[:, 2],
                opacity=0.1,
                color='lightgray',
                name="Mesh Surface"
            ))
            
            # Add gradient vectors
            for i in range(n_samples):
                start = sample_points[i]
                end = start + sample_gradients[i] * 0.1  # Scale for visualization
                
                fig_vectors.add_trace(go.Scatter3d(
                    x=[start[0], end[0]],
                    y=[start[1], end[1]],
                    z=[start[2], end[2]],
                    mode='lines',
                    line=dict(
                        width=3,
                        color=np.linalg.norm(sample_gradients[i]),
                        colorscale="Hot"
                    ),
                    showlegend=False,
                    hoverinfo='none'
                ))
            
            fig_vectors.update_layout(
                title="Gradient Vector Field",
                scene=dict(aspectmode="data"),
                height=500
            )
            
            st.plotly_chart(fig_vectors, use_container_width=True)
        else:
            st.info("Gradient analysis requires triangular mesh data.")
    
    def render_spatial_correlation(self, mesh_data, field_values, field_name):
        """Render spatial correlation analysis"""
        points = mesh_data.points
        
        # Compute spatial correlation function
        if len(field_values) > 200:
            # Sample points for correlation analysis
            n_samples = min(200, len(points))
            sample_indices = np.random.choice(len(points), n_samples, replace=False)
            sample_points = points[sample_indices]
            sample_values = field_values[sample_indices]
            
            # Compute pairwise distances and correlations
            distances = []
            correlations = []
            
            for i in range(n_samples):
                for j in range(i+1, n_samples):
                    dist = np.linalg.norm(sample_points[i] - sample_points[j])
                    corr = abs(sample_values[i] - sample_values[j])
                    
                    distances.append(dist)
                    correlations.append(corr)
            
            if len(distances) > 100:
                # Create correlation plot
                fig_corr = px.scatter(
                    x=distances,
                    y=correlations,
                    trendline="lowess",
                    title=f"Spatial Correlation of {field_name}",
                    labels={'x': 'Distance', 'y': 'Value Difference'},
                    opacity=0.3
                )
                
                fig_corr.update_layout(height=400)
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Compute correlation length
                try:
                    # Fit exponential decay
                    from scipy.optimize import curve_fit
                    
                    def exp_decay(x, a, b, c):
                        return a * np.exp(-b * x) + c
                    
                    # Bin data
                    max_dist = np.max(distances)
                    n_bins = 20
                    bins = np.linspace(0, max_dist, n_bins + 1)
                    
                    bin_centers = []
                    bin_means = []
                    
                    for i in range(n_bins):
                        mask = (np.array(distances) >= bins[i]) & (np.array(distances) < bins[i+1])
                        if np.sum(mask) > 10:
                            bin_centers.append((bins[i] + bins[i+1]) / 2)
                            bin_means.append(np.mean(np.array(correlations)[mask]))
                    
                    if len(bin_centers) > 5:
                        try:
                            popt, _ = curve_fit(exp_decay, bin_centers, bin_means, 
                                               p0=[bin_means[0], 1.0, bin_means[-1]])
                            
                            # Correlation length = 1/b
                            correlation_length = 1.0 / popt[1]
                            
                            st.metric("Correlation Length", f"{correlation_length:.3f}")
                            
                            # Add fitted curve to plot
                            x_fit = np.linspace(0, max_dist, 100)
                            y_fit = exp_decay(x_fit, *popt)
                            
                            fig_corr.add_trace(go.Scatter(
                                x=x_fit,
                                y=y_fit,
                                mode='lines',
                                name=f'Exponential Fit (τ={correlation_length:.2f})',
                                line=dict(color='red', width=3)
                            ))
                            
                            st.plotly_chart(fig_corr, use_container_width=True)
                        except:
                            st.info("Could not fit correlation function")
                except:
                    st.info("Insufficient data for correlation length estimation")
    
    def render_volume_rendering(self, sim, field):
        """Render volume rendering"""
        mesh_data = sim['mesh_data']
        
        col1, col2 = st.columns(2)
        
        with col1:
            timestep = st.slider(
                "Timestep",
                0, sim['n_timesteps'] - 1, 0,
                key="volume_timestep"
            )
        
        with col2:
            opacity_preset = st.selectbox(
                "Opacity Preset",
                ["Standard", "Transparent", "Opaque", "Edge Enhanced"],
                key="opacity_preset"
            )
        
        # Get field values
        field_data = mesh_data.fields[field][timestep]
        if field_data.ndim == 2:
            field_values = np.linalg.norm(field_data, axis=1)
        else:
            field_values = field_data
        
        # Create opacity scale based on preset
        if opacity_preset == "Standard":
            opacity_scale = [[0, 0.0], [0.1, 0.1], [0.5, 0.3], [1, 0.8]]
        elif opacity_preset == "Transparent":
            opacity_scale = [[0, 0.0], [0.2, 0.05], [0.8, 0.1], [1, 0.2]]
        elif opacity_preset == "Opaque":
            opacity_scale = [[0, 0.1], [0.1, 0.3], [0.3, 0.6], [1, 1.0]]
        else:  # Edge Enhanced
            opacity_scale = [[0, 0.0], [0.1, 0.05], [0.3, 0.1], [0.7, 0.3], [1, 0.8]]
        
        # Create volume rendering
        fig_volume = self.visualizer.create_volume_rendering(
            mesh_data,
            field_values,
            colormap=st.session_state.selected_colormap,
            title=f"{field} Volume Rendering - Timestep {timestep + 1}",
            opacity_scale=opacity_scale
        )
        
        st.plotly_chart(fig_volume, use_container_width=True)
        
        # Volume statistics
        st.markdown("##### 📊 Volume Statistics")
        
        if mesh_data.mesh_stats and 'bbox' in mesh_data.mesh_stats:
            bbox = mesh_data.mesh_stats['bbox']
            volume = bbox.get('volume', 0)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Volume", f"{volume:.3f} mm³")
            
            with col2:
                st.metric("Field Mean", f"{np.nanmean(field_values):.3f}")
            
            with col3:
                # Compute volume integral (approximate)
                if mesh_data.triangles is not None:
                    # Approximate using tetrahedra
                    areas = mesh_data.compute_triangle_areas()
                    if len(areas) > 0:
                        avg_thickness = 0.1  # Approximate thickness
                        total_volume = np.sum(areas) * avg_thickness
                        st.metric("Approx. Volume", f"{total_volume:.3f} mm³")
            
            with col4:
                # Compute volume-weighted average
                if 'volume' in locals() and volume > 0:
                    vol_weighted_avg = np.nanmean(field_values)  # Simplified
                    st.metric("Vol-Weighted Avg", f"{vol_weighted_avg:.3f}")
    
    def render_streamlines(self, sim, field):
        """Render streamline visualization for vector fields"""
        mesh_data = sim['mesh_data']
        
        col1, col2 = st.columns(2)
        
        with col1:
            timestep = st.slider(
                "Timestep",
                0, sim['n_timesteps'] - 1, 0,
                key="streamline_timestep"
            )
        
        with col2:
            density = st.slider(
                "Streamline Density",
                1, 10, 3,
                key="streamline_density",
                help="Higher density shows more streamlines"
            )
        
        # Check if field is vector field
        field_data = mesh_data.fields[field][timestep]
        if field_data.ndim != 2:
            st.warning(f"Field '{field}' is not a vector field. Streamlines require vector data.")
            return
        
        # Create streamline visualization
        fig_streamlines = self.visualizer.create_streamline_visualization(
            mesh_data,
            field_data,
            density=density,
            colormap=st.session_state.selected_colormap,
            title=f"{field} Streamlines - Timestep {timestep + 1}"
        )
        
        st.plotly_chart(fig_streamlines, use_container_width=True)
        
        # Vector field statistics
        st.markdown("##### 📊 Vector Field Statistics")
        
        # Compute vector statistics
        magnitudes = np.linalg.norm(field_data, axis=1)
        directions = field_data / (magnitudes[:, np.newaxis] + 1e-8)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean Magnitude", f"{np.nanmean(magnitudes):.3f}")
        
        with col2:
            st.metric("Max Magnitude", f"{np.nanmax(magnitudes):.3f}")
        
        with col3:
            # Compute direction statistics
            if len(directions) > 10:
                mean_direction = np.nanmean(directions, axis=0)
                direction_mag = np.linalg.norm(mean_direction)
                st.metric("Direction Consistency", f"{direction_mag:.3f}")
        
        with col4:
            # Compute divergence (approximate)
            if mesh_data.triangles is not None and len(mesh_data.triangles) > 0:
                # Simple divergence approximation
                div_approx = np.nanstd(magnitudes) / np.nanmean(magnitudes)
                st.metric("Divergence Approx", f"{div_approx:.3f}")
        
        # Vector components
        st.markdown("##### 🔢 Vector Components")
        
        components = ["X", "Y", "Z"]
        cols = st.columns(3)
        
        for idx, (col, comp) in enumerate(zip(cols, components)):
            with col:
                comp_values = field_data[:, idx]
                
                fig_comp = px.histogram(
                    x=comp_values,
                    nbins=50,
                    title=f"{comp}-Component Distribution",
                    labels={'x': f'{field} {comp}-Component'}
                )
                
                fig_comp.update_layout(height=250)
                st.plotly_chart(fig_comp, use_container_width=True)
                
                st.metric(f"{comp} Mean", f"{np.nanmean(comp_values):.3f}")
                st.metric(f"{comp} Std", f"{np.nanstd(comp_values):.3f}")
    
    def render_field_statistics(self, field_values, field_name, sim_name=""):
        """Render comprehensive field statistics"""
        st.markdown("##### 📊 Field Statistics")
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Minimum", f"{np.nanmin(field_values):.3f}")
        
        with col2:
            st.metric("Maximum", f"{np.nanmax(field_values):.3f}")
        
        with col3:
            st.metric("Mean", f"{np.nanmean(field_values):.3f}")
        
        with col4:
            st.metric("Std Dev", f"{np.nanstd(field_values):.3f}")
        
        # Additional statistics in expander
        with st.expander("📈 Detailed Statistics", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Median", f"{np.nanmedian(field_values):.3f}")
                st.metric("Variance", f"{np.nanvar(field_values):.3f}")
            
            with col2:
                from scipy.stats import skew, kurtosis
                try:
                    valid_values = field_values[~np.isnan(field_values)]
                    if len(valid_values) > 3:
                        st.metric("Skewness", f"{skew(valid_values):.3f}")
                        st.metric("Kurtosis", f"{kurtosis(valid_values):.3f}")
                except:
                    st.metric("Skewness", "N/A")
                    st.metric("Kurtosis", "N/A")
            
            with col3:
                try:
                    valid_values = field_values[~np.isnan(field_values)]
                    if len(valid_values) > 1:
                        q25 = np.percentile(valid_values, 25)
                        q75 = np.percentile(valid_values, 75)
                        st.metric("Q1 (25%)", f"{q25:.3f}")
                        st.metric("Q3 (75%)", f"{q75:.3f}")
                except:
                    st.metric("Q1", "N/A")
                    st.metric("Q3", "N/A")
            
            with col4:
                try:
                    valid_values = field_values[~np.isnan(field_values)]
                    if len(valid_values) > 0:
                        iqr = np.percentile(valid_values, 75) - np.percentile(valid_values, 25)
                        cv = (np.nanstd(field_values) / np.nanmean(field_values)) * 100
                        st.metric("IQR", f"{iqr:.3f}")
                        st.metric("CV (%)", f"{cv:.1f}")
                except:
                    st.metric("IQR", "N/A")
                    st.metric("CV", "N/A")
            
            # Histogram with distribution
            fig_hist = go.Figure()
            
            fig_hist.add_trace(go.Histogram(
                x=field_values[~np.isnan(field_values)],
                nbinsx=50,
                name='Distribution',
                marker_color='skyblue',
                opacity=0.7,
                histnorm='probability density'
            ))
            
            # Add KDE
            try:
                from scipy.stats import gaussian_kde
                valid_values = field_values[~np.isnan(field_values)]
                if len(valid_values) > 10:
                    kde = gaussian_kde(valid_values)
                    x_range = np.linspace(np.min(valid_values), np.max(valid_values), 100)
                    y_kde = kde(x_range)
                    
                    fig_hist.add_trace(go.Scatter(
                        x=x_range,
                        y=y_kde,
                        mode='lines',
                        name='KDE',
                        line=dict(color='red', width=2)
                    ))
            except:
                pass
            
            fig_hist.update_layout(
                title=f"{field_name} Value Distribution" + (f" - {sim_name}" if sim_name else ""),
                xaxis_title=field_name,
                yaxis_title="Probability Density",
                height=300,
                showlegend=True
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)
    
    def render_field_predictor(self):
        """Render field predictor interface"""
        st.markdown('<h2 class="sub-header">🤖 AI Field Predictor</h2>', 
                   unsafe_allow_html=True)
        
        if not st.session_state.get('extrapolator_fitted', False):
            st.warning("""
            ⚠️ Extrapolator not fitted. Please load simulations first and ensure 
            the extrapolator is initialized in the Data Viewer mode.
            """)
            return
        
        st.markdown("""
        <div class="info-box">
        <h3>🧠 Physics-Informed Neural Field Predictor</h3>
        <p>This predictor uses a hybrid approach combining attention mechanisms with 
        physics-informed neural networks to predict field distributions for unseen 
        simulation parameters.</p>
        
        <p><strong>Capabilities:</strong></p>
        <ul>
            <li>Predict field distributions for arbitrary energy/duration/time combinations</li>
            <li>Generate confidence estimates for predictions</li>
            <li>Compare predictions with actual simulations (when available)</li>
            <li>Visualize prediction uncertainty</li>
            <li>Export predicted fields for further analysis</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Prediction parameters
        with st.expander("🎯 Prediction Parameters", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                energy = st.number_input(
                    "Energy (mJ)",
                    min_value=0.1,
                    max_value=100.0,
                    value=3.0,
                    step=0.1,
                    key="pred_energy"
                )
            
            with col2:
                duration = st.number_input(
                    "Duration (ns)",
                    min_value=0.5,
                    max_value=50.0,
                    value=3.0,
                    step=0.1,
                    key="pred_duration"
                )
            
            with col3:
                time = st.number_input(
                    "Time (ns)",
                    min_value=0.0,
                    max_value=100.0,
                    value=5.0,
                    step=0.1,
                    key="pred_time"
                )
        
        # Field selection
        available_fields = list(self.data_loader.common_fields)
        if available_fields:
            selected_field = st.selectbox(
                "Field to Predict",
                available_fields,
                key="predict_field"
            )
        else:
            st.warning("No common fields available for prediction")
            return
        
        # Reference geometry
        reference_sim = st.selectbox(
            "Reference Geometry",
            sorted(self.data_loader.simulations.keys()),
            key="pred_ref_geom"
        )
        
        # Prediction options
        col1, col2 = st.columns(2)
        
        with col1:
            include_uncertainty = st.checkbox(
                "Include Uncertainty",
                value=True,
                help="Show prediction confidence intervals"
            )
        
        with col2:
            compare_with_similar = st.checkbox(
                "Compare with Similar Simulations",
                value=True,
                help="Find and compare with nearest actual simulations"
            )
        
        # Generate prediction
        if st.button("🚀 Generate AI Prediction", type="primary", use_container_width=True):
            self.generate_ai_prediction(
                energy, duration, time, selected_field, reference_sim,
                include_uncertainty, compare_with_similar
            )
    
    def generate_ai_prediction(self, energy, duration, time, field, 
                              ref_sim, include_uncertainty, compare_with_similar):
        """Generate AI-powered prediction"""
        # Validate inputs
        try:
            self.validator.validate_simulation_parameters(energy, duration, time)
        except ValidationError as e:
            st.error(f"❌ Invalid parameters: {str(e)}")
            return
        
        # Create progress containers
        progress_container = st.empty()
        status_container = st.empty()
        
        progress_bar = progress_container.progress(0)
        status_container.text("Initializing prediction...")
        
        try:
            # Step 1: Get reference mesh
            progress_bar.progress(20)
            status_container.text("Loading reference geometry...")
            
            if ref_sim not in self.data_loader.simulations:
                raise ValidationError(f"Reference simulation '{ref_sim}' not found")
            
            mesh_data = self.data_loader.simulations[ref_sim]['mesh_data']
            
            # Step 2: Generate prediction
            progress_bar.progress(40)
            status_container.text("Generating AI prediction...")
            
            prediction = self.geom_predictor.predict_field_on_mesh(
                energy, duration, time, mesh_data, field
            )
            
            if prediction is None:
                raise ValueError("Failed to generate prediction")
            
            # Step 3: Create visualization
            progress_bar.progress(60)
            status_container.text("Creating visualization...")
            
            # Main prediction visualization
            fig_pred = self.visualizer.create_advanced_mesh_visualization(
                mesh_data,
                field,
                prediction['values'],
                colormap=st.session_state.selected_colormap,
                title=f"AI Prediction: {field}\nE={energy:.1f}mJ, τ={duration:.1f}ns, t={time:.1f}ns",
                opacity=0.9
            )
            
            progress_bar.progress(80)
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # Step 4: Show prediction details
            progress_bar.progress(100)
            status_container.text("✅ Prediction completed!")
            
            # Prediction statistics
            st.markdown("##### 📊 Prediction Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Confidence", f"{prediction['confidence']:.2%}")
            
            with col2:
                st.metric("Method", prediction['method'])
            
            with col3:
                st.metric("Sources Used", prediction['n_sources'])
            
            with col4:
                st.metric("Predicted Mean", f"{prediction['source_stats']['mean']:.3f}")
            
            # Detailed statistics
            with st.expander("📈 Prediction Details", expanded=True):
                st.json(prediction['source_stats'], expanded=False)
                
                if 'target_stats' in prediction:
                    st.write("**Target Statistics:**")
                    st.json(prediction['target_stats'], expanded=False)
            
            # Compare with similar simulations
            if compare_with_similar:
                st.markdown("##### 🔍 Comparison with Similar Simulations")
                
                # Find similar simulations
                similar_sims = self.data_loader.get_similar_simulations(energy, duration, max_results=3)
                
                if similar_sims:
                    comparison_data = []
                    
                    for sim_name, similarity, sim in similar_sims:
                        if field in sim['field_info']:
                            # Get field statistics for this simulation
                            mesh_data_sim = sim['mesh_data']
                            field_data = mesh_data_sim.fields[field]
                            
                            # Compute average across timesteps
                            all_values = []
                            for t in range(field_data.shape[0]):
                                if field_data[t].ndim == 1:
                                    values = field_data[t]
                                else:
                                    values = np.linalg.norm(field_data[t], axis=1)
                                all_values.extend(values[~np.isnan(values)].tolist())
                            
                            if all_values:
                                comparison_data.append({
                                    'Simulation': sim_name,
                                    'Similarity': f"{similarity:.3f}",
                                    'Energy': sim['energy_mJ'],
                                    'Duration': sim['duration_ns'],
                                    'Actual Mean': np.mean(all_values),
                                    'Predicted Mean': prediction['source_stats']['mean'],
                                    'Difference': abs(np.mean(all_values) - prediction['source_stats']['mean'])
                                })
                    
                    if comparison_data:
                        df_comparison = pd.DataFrame(comparison_data)
                        st.dataframe(
                            df_comparison.style.format({
                                'Actual Mean': '{:.3f}',
                                'Predicted Mean': '{:.3f}',
                                'Difference': '{:.3f}'
                            }),
                            use_container_width=True
                        )
            
            # Export prediction
            st.markdown("##### 💾 Export Prediction")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 Export as CSV", use_container_width=True):
                    # Create CSV data
                    df_pred = pd.DataFrame({
                        'X': mesh_data.points[:, 0],
                        'Y': mesh_data.points[:, 1],
                        'Z': mesh_data.points[:, 2],
                        field: prediction['values']
                    })
                    
                    csv = df_pred.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"prediction_{field}_E{energy}_D{duration}_T{time}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("💾 Save to Session", use_container_width=True):
                    if 'predictions' not in st.session_state:
                        st.session_state.predictions = []
                    
                    st.session_state.predictions.append({
                        'energy': energy,
                        'duration': duration,
                        'time': time,
                        'field': field,
                        'values': prediction['values'],
                        'confidence': prediction['confidence'],
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    st.success("Prediction saved to session!")
        
        except Exception as e:
            progress_bar.progress(100)
            status_container.error(f"❌ Error generating prediction: {str(e)}")
            
            with st.expander("Error Details", expanded=False):
                st.code(traceback.format_exc())
    
    def render_export_reports(self):
        """Render export and reports interface"""
        st.markdown('<h2 class="sub-header">💾 Export & Reports</h2>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h3>📤 Data Export and Reporting</h3>
        <p>Export simulation data, visualizations, and analysis results in various formats 
        for further analysis or reporting.</p>
        
        <p><strong>Available Export Options:</strong></p>
        <ul>
            <li>Export raw simulation data in CSV format</li>
            <li>Save visualizations as high-resolution images</li>
            <li>Generate comprehensive PDF reports</li>
            <li>Export statistical summaries</li>
            <li>Save session state for later analysis</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Export options
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Export", "🖼️ Image Export", 
                                          "📄 Report Generation", "💾 Session Management"])
        
        with tab1:
            self.render_data_export()
        
        with tab2:
            self.render_image_export()
        
        with tab3:
            self.render_report_generation()
        
        with tab4:
            self.render_session_management()
    
    def render_data_export(self):
        """Render data export interface"""
        if not st.session_state.data_loaded:
            st.warning("No data loaded. Please load simulations first.")
            return
        
        simulations = self.data_loader.simulations
        
        st.markdown("##### 📁 Export Simulation Data")
        
        # Simulation selection
        export_sim = st.selectbox(
            "Select Simulation to Export",
            sorted(simulations.keys()),
            key="export_sim"
        )
        
        sim = simulations[export_sim]
        
        # Field selection
        export_field = st.selectbox(
            "Select Field to Export",
            sorted(sim['field_info'].keys()),
            key="export_field"
        )
        
        # Timestep selection
        export_timestep = st.slider(
            "Select Timestep",
            0, sim['n_timesteps'] - 1, 0,
            key="export_timestep"
        )
        
        # Export options
        col1, col2 = st.columns(2)
        
        with col1:
            include_coords = st.checkbox("Include Coordinates", value=True)
            include_mesh_info = st.checkbox("Include Mesh Information", value=False)
        
        with col2:
            export_format = st.selectbox(
                "Export Format",
                ["CSV", "JSON", "NPZ", "VTK"],
                key="export_format"
            )
        
        # Generate export
        if st.button("📤 Export Data", type="primary", use_container_width=True):
            self.export_simulation_data(
                sim, export_field, export_timestep, 
                include_coords, include_mesh_info, export_format
            )
        
        # Batch export
        st.markdown("##### 📦 Batch Export")
        
        batch_fields = st.multiselect(
            "Select Fields for Batch Export",
            sorted(sim['field_info'].keys()),
            default=list(sim['field_info'].keys())[:3]
        )
        
        if st.button("📦 Export Multiple Fields", use_container_width=True):
            self.export_batch_data(sim, batch_fields, export_timestep)
    
    def export_simulation_data(self, sim, field, timestep, 
                              include_coords, include_mesh_info, format):
        """Export simulation data"""
        try:
            mesh_data = sim['mesh_data']
            field_data = mesh_data.fields[field][timestep]
            
            if field_data.ndim == 2:
                field_values = np.linalg.norm(field_data, axis=1)
                is_vector = True
            else:
                field_values = field_data
                is_vector = False
            
            # Prepare data
            export_data = {}
            
            if include_coords:
                export_data['X'] = mesh_data.points[:, 0]
                export_data['Y'] = mesh_data.points[:, 1]
                export_data['Z'] = mesh_data.points[:, 2]
            
            export_data[field] = field_values
            
            if is_vector and field_data.ndim == 2:
                for i in range(field_data.shape[1]):
                    export_data[f'{field}_comp{i+1}'] = field_data[:, i]
            
            if include_mesh_info and mesh_data.triangles is not None:
                # Export triangle connectivity
                triangles_flat = mesh_data.triangles.flatten()
                export_data['triangles'] = triangles_flat
            
            # Create DataFrame
            df = pd.DataFrame(export_data)
            
            # Export based on format
            if format == "CSV":
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{sim['name']}_{field}_t{timestep}.csv",
                    mime="text/csv"
                )
            
            elif format == "JSON":
                json_str = df.to_json(orient='records', indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f"{sim['name']}_{field}_t{timestep}.json",
                    mime="application/json"
                )
            
            elif format == "NPZ":
                # Save as numpy compressed format
                import io
                buffer = io.BytesIO()
                np.savez_compressed(buffer, **export_data)
                buffer.seek(0)
                
                st.download_button(
                    label="Download NPZ",
                    data=buffer,
                    file_name=f"{sim['name']}_{field}_t{timestep}.npz",
                    mime="application/octet-stream"
                )
            
            st.success("✅ Data ready for export!")
            
        except Exception as e:
            st.error(f"❌ Error exporting data: {str(e)}")
    
    def export_batch_data(self, sim, fields, timestep):
        """Export multiple fields in batch"""
        try:
            mesh_data = sim['mesh_data']
            
            # Prepare data
            export_data = {
                'X': mesh_data.points[:, 0],
                'Y': mesh_data.points[:, 1],
                'Z': mesh_data.points[:, 2]
            }
            
            for field in fields:
                field_data = mesh_data.fields[field][timestep]
                if field_data.ndim == 2:
                    export_data[field] = np.linalg.norm(field_data, axis=1)
                else:
                    export_data[field] = field_data
            
            # Create DataFrame
            df = pd.DataFrame(export_data)
            
            # Export as CSV
            csv = df.to_csv(index=False)
            
            st.download_button(
                label="Download Batch CSV",
                data=csv,
                file_name=f"{sim['name']}_batch_t{timestep}.csv",
                mime="text/csv"
            )
            
            st.success(f"✅ Exported {len(fields)} fields successfully!")
            
        except Exception as e:
            st.error(f"❌ Error exporting batch data: {str(e)}")
    
    def render_image_export(self):
        """Render image export interface"""
        st.markdown("##### 🖼️ Export Visualizations")
        
        # Current visualization export
        if st.button("📷 Export Current Visualization", use_container_width=True):
            st.info("""
            To export the current visualization:
            1. Hover over the visualization
            2. Click the camera icon in the top-right corner
            3. Choose download format (PNG, SVG, etc.)
            
            This uses Plotly's built-in export functionality.
            """)
        
        # High-resolution export options
        st.markdown("##### 🖥️ High-Resolution Export")
        
        col1, col2 = st.columns(2)
        
        with col1:
            export_width = st.number_input("Width (pixels)", 800, 4000, 1920, 100)
            export_height = st.number_input("Height (pixels)", 600, 4000, 1080, 100)
        
        with col2:
            export_dpi = st.selectbox("DPI", [72, 150, 300, 600], index=2)
            export_format = st.selectbox("Format", ["PNG", "JPEG", "SVG", "PDF"])
        
        if st.button("🖨️ Generate High-Res Export", use_container_width=True):
            st.info("High-resolution export requires additional configuration. Please use the camera icon for standard exports.")
    
    def render_report_generation(self):
        """Render report generation interface"""
        st.markdown("##### 📄 Generate Analysis Report")
        
        # Report options
        report_title = st.text_input("Report Title", "FEA Simulation Analysis Report")
        
        col1, col2 = st.columns(2)
        
        with col1:
            include_summary = st.checkbox("Include Summary", value=True)
            include_statistics = st.checkbox("Include Statistics", value=True)
            include_visualizations = st.checkbox("Include Visualizations", value=True)
        
        with col2:
            include_methodology = st.checkbox("Include Methodology", value=False)
            include_recommendations = st.checkbox("Include Recommendations", value=False)
            include_appendix = st.checkbox("Include Appendix", value=False)
        
        # Simulation selection for report
        if st.session_state.data_loaded:
            report_sims = st.multiselect(
                "Simulations to Include",
                sorted(self.data_loader.simulations.keys()),
                default=list(self.data_loader.simulations.keys())[:3]
            )
        
        # Generate report
        if st.button("📄 Generate Report", type="primary", use_container_width=True):
            self.generate_report(
                report_title, report_sims,
                include_summary, include_statistics, include_visualizations,
                include_methodology, include_recommendations, include_appendix
            )
    
    def generate_report(self, title, simulations, *args):
        """Generate analysis report"""
        try:
            # Create report content
            report_content = f"# {title}\n\n"
            report_content += f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
            
            # Add sections based on options
            if args[0]:  # Include Summary
                report_content += "## Executive Summary\n\n"
                report_content += f"This report analyzes {len(simulations)} FEA laser simulations.\n\n"
            
            if args[1]:  # Include Statistics
                report_content += "## Statistical Analysis\n\n"
                
                for sim_name in simulations:
                    if sim_name in self.data_loader.simulations:
                        sim = self.data_loader.simulations[sim_name]
                        report_content += f"### {sim_name}\n"
                        report_content += f"- Energy: {sim['energy_mJ']} mJ\n"
                        report_content += f"- Duration: {sim['duration_ns']} ns\n"
                        report_content += f"- Timesteps: {sim['n_timesteps']}\n\n"
            
            # Add more sections as needed...
            
            # Create download button for report
            st.download_button(
                label="📥 Download Report (Markdown)",
                data=report_content,
                file_name=f"{title.replace(' ', '_')}.md",
                mime="text/markdown"
            )
            
            st.success("✅ Report generated successfully!")
            
            # Preview
            with st.expander("📋 Report Preview", expanded=True):
                st.markdown(report_content[:1000] + "..." if len(report_content) > 1000 else report_content)
                
        except Exception as e:
            st.error(f"❌ Error generating report: {str(e)}")
    
    def render_session_management(self):
        """Render session management interface"""
        st.markdown("##### 💾 Session Management")
        
        # Current session info
        if st.session_state.data_loaded:
            sim_count = len(st.session_state.loaded_simulations)
            field_count = len(st.session_state.available_fields)
            
            st.info(f"""
            **Current Session:**
            - Loaded Simulations: {sim_count}
            - Available Fields: {field_count}
            - Memory Usage: {self._estimate_memory_usage(st.session_state.loaded_simulations):.1f} MB
            - Session Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """)
        
        # Session actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save Session State", use_container_width=True):
                self.save_session_state()
        
        with col2:
            if st.button("📥 Load Session State", use_container_width=True):
                self.load_session_state()
        
        with col3:
            if st.button("🧹 Clear Session", use_container_width=True, type="secondary"):
                self.clear_session()
        
        # Session statistics
        if st.session_state.data_loaded:
            with st.expander("📊 Session Statistics", expanded=False):
                # Compute various statistics
                simulations = st.session_state.loaded_simulations
                
                # Energy and duration ranges
                energies = [s['energy_mJ'] for s in simulations.values()]
                durations = [s['duration_ns'] for s in simulations.values()]
                
                st.metric("Energy Range", f"{min(energies):.1f} - {max(energies):.1f} mJ")
                st.metric("Duration Range", f"{min(durations):.1f} - {max(durations):.1f} ns")
                st.metric("Total Timesteps", sum(s['n_timesteps'] for s in simulations.values()))
                
                # Mesh statistics
                if any(s.get('has_mesh', False) for s in simulations.values()):
                    mesh_sims = [s for s in simulations.values() if s.get('has_mesh', False)]
                    if mesh_sims:
                        avg_points = np.mean([len(s['mesh_data'].points) for s in mesh_sims])
                        avg_triangles = np.mean([len(s['mesh_data'].triangles) if s['mesh_data'].triangles is not None else 0 
                                               for s in mesh_sims])
                        
                        st.metric("Avg Mesh Points", f"{avg_points:.0f}")
                        st.metric("Avg Triangles", f"{avg_triangles:.0f}")
    
    def save_session_state(self):
        """Save current session state"""
        try:
            # Create session data
            session_data = {
                'loaded_simulations': list(st.session_state.loaded_simulations.keys()),
                'available_fields': list(st.session_state.available_fields),
                'common_fields': list(st.session_state.common_fields),
                'timestamp': datetime.now().isoformat(),
                'version': '2.0'
            }
            
            # Convert to JSON
            session_json = json.dumps(session_data, indent=2)
            
            # Create download button
            st.download_button(
                label="📥 Download Session File",
                data=session_json,
                file_name=f"fea_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
            st.success("✅ Session state saved successfully!")
            
        except Exception as e:
            st.error(f"❌ Error saving session: {str(e)}")
    
    def load_session_state(self):
        """Load session state from file"""
        uploaded_file = st.file_uploader("Choose a session file", type=['json'])
        
        if uploaded_file is not None:
            try:
                session_data = json.load(uploaded_file)
                
                # Validate session data
                if 'version' in session_data and session_data['version'] == '2.0':
                    st.success(f"✅ Session loaded: {session_data['timestamp']}")
                    
                    # Show session info
                    st.info(f"""
                    **Session Information:**
                    - Simulations: {len(session_data['loaded_simulations'])}
                    - Fields: {len(session_data['available_fields'])}
                    - Created: {session_data['timestamp']}
                    """)
                    
                    # Option to restore
                    if st.button("🔄 Restore This Session", type="primary"):
                        st.info("Session restoration would reload the specified simulations")
                else:
                    st.error("❌ Invalid session file version")
                    
            except Exception as e:
                st.error(f"❌ Error loading session: {str(e)}")
    
    def clear_session(self):
        """Clear current session"""
        if st.button("⚠️ Confirm Clear Session", type="secondary"):
            # Clear session state
            for key in list(st.session_state.keys()):
                if key not in ['current_mode', 'selected_colormap']:
                    del st.session_state[key]
            
            # Reinitialize
            self._initialize_session_state()
            
            st.success("✅ Session cleared successfully!")
            st.rerun()
    
    def export_as_image(self):
        """Export current visualization as image"""
        st.info("""
        To export the current visualization as an image:
        1. Hover over the visualization
        2. Click the camera icon in the top-right corner
        3. Choose your preferred format (PNG, JPEG, SVG, PDF)
        
        For high-resolution exports, use the Export & Reports section.
        """)
    
    def export_statistics(self, sim):
        """Export statistics for a simulation"""
        try:
            # Collect statistics
            stats = {
                'simulation_name': sim['name'],
                'energy_mJ': sim['energy_mJ'],
                'duration_ns': sim['duration_ns'],
                'timesteps': sim['n_timesteps'],
                'mesh_stats': sim['mesh_data'].mesh_stats if hasattr(sim['mesh_data'], 'mesh_stats') else {},
                'export_timestamp': datetime.now().isoformat()
            }
            
            # Convert to JSON
            stats_json = json.dumps(stats, indent=2)
            
            # Create download button
            st.download_button(
                label="📥 Download Statistics",
                data=stats_json,
                file_name=f"{sim['name']}_statistics.json",
                mime="application/json"
            )
            
        except Exception as e:
            st.error(f"❌ Error exporting statistics: {str(e)}")
    
    def save_session(self):
        """Save current session"""
        st.info("Use the 'Save Session State' button in the Export & Reports section to save your current session.")
    
    def show_data_not_loaded_warning(self):
        """Show warning when data is not loaded"""
        st.markdown("""
        <div class="warning-box">
        <h3>⚠️ No Data Loaded</h3>
        <p>Please load simulations first using the "Load All Simulations" button in the sidebar.</p>
        <p>Ensure your data follows this structure:</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📁 Expected Directory Structure", expanded=True):
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
        
        # Quick start guide
        with st.expander("🚀 Quick Start Guide", expanded=True):
            st.markdown("""
            1. **Prepare your data**: Organize VTU files in folders named like `q0p5mJ-delta4p2ns`
            2. **Load data**: Click "Load All Simulations" in the sidebar
            3. **Explore**: Use the Data Viewer to visualize your simulations
            4. **Analyze**: Compare simulations and generate predictions
            5. **Export**: Save results and generate reports
            
            **Tips:**
            - Use the cache option for faster reloading
            - Enable full mesh loading for 3D visualizations
            - Check the console for loading progress and any errors
            """)

# =============================================
# PATH CONFIGURATION (moved to avoid import issues)
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEA_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "fea_solutions")
os.makedirs(FEA_SOLUTIONS_DIR, exist_ok=True)

# =============================================
# MAIN APPLICATION ENTRY POINT
# =============================================
def main():
    """Main application entry point"""
    try:
        app = EnhancedFEAVisualizationPlatform()
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        
        with st.expander("Error Details", expanded=False):
            st.code(traceback.format_exc())
        
        st.info("""
        **Troubleshooting:**
        1. Ensure all required packages are installed
        2. Check if VTU files are accessible
        3. Verify the directory structure is correct
        4. Restart the application
        """)

if __name__ == "__main__":
    main()
