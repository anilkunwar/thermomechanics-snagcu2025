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
from scipy.spatial import Delaunay, ConvexHull
import hashlib
import pickle
import time
import json
from collections import defaultdict

warnings.filterwarnings('ignore')

# =============================================
# PATH CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEA_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "fea_solutions")
os.makedirs(FEA_SOLUTIONS_DIR, exist_ok=True)

# =============================================
# ENHANCED CACHE MANAGEMENT
# =============================================
class EnhancedCacheManager:
    """Enhanced cache management with versioning and validation"""
    
    @staticmethod
    def create_cache_key(*args, **kwargs):
        """Create a stable cache key from arguments"""
        key_parts = []
        
        # Add positional args
        for arg in args:
            if isinstance(arg, (str, int, float, bool, type(None))):
                key_parts.append(str(arg))
            elif isinstance(arg, np.ndarray):
                # Create hash for numpy arrays
                key_parts.append(hashlib.md5(arg.tobytes()).hexdigest()[:16])
            elif hasattr(arg, '__dict__'):
                key_parts.append(str(arg.__class__.__name__))
        
        # Add keyword args
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        
        return "_".join(key_parts)
    
    @staticmethod
    def validate_data_consistency(simulations, field_name):
        """Validate that field exists in all required simulations"""
        missing_in = []
        valid_simulations = {}
        
        for name, sim in simulations.items():
            if 'fields' in sim and field_name in sim['fields']:
                valid_simulations[name] = sim
            else:
                missing_in.append(name)
        
        return valid_simulations, missing_in

# =============================================
# FIELD MANAGEMENT SYSTEM
# =============================================
class FieldManager:
    """Manages field discovery, categorization, and metadata"""
    
    def __init__(self):
        self.all_fields = set()
        self.field_categories = {
            'temperature': set(),
            'stress': set(),
            'displacement': set(),
            'strain': set(),
            'velocity': set(),
            'pressure': set(),
            'force': set(),
            'energy': set(),
            'other': set()
        }
        self.field_metadata = {}
        self.field_aliases = {
            'temp': 'temperature',
            'T': 'temperature',
            'stress': 'stress',
            'sigma': 'stress',
            'displ': 'displacement',
            'disp': 'displacement',
            'u': 'displacement',
            'strain': 'strain',
            'epsilon': 'strain',
            'vel': 'velocity',
            'v': 'velocity',
            'press': 'pressure',
            'p': 'pressure',
            'force': 'force',
            'F': 'force',
            'energy': 'energy',
            'E': 'energy'
        }
        
    def categorize_field(self, field_name):
        """Categorize field based on name patterns"""
        field_lower = field_name.lower()
        
        # Temperature fields
        if any(keyword in field_lower for keyword in ['temp', 'temperature', 'T_']):
            category = 'temperature'
            aliases = ['temperature']
        
        # Stress fields
        elif any(keyword in field_lower for keyword in ['stress', 'sigma', 'tau', 's_']):
            if any(suffix in field_lower for suffix in ['_xx', '_yy', '_zz', '_xy', '_yz', '_xz']):
                component = self.extract_component(field_name)
                category = 'stress'
                aliases = ['stress', f'stress{component}']
            else:
                category = 'stress'
                aliases = ['stress']
        
        # Displacement fields
        elif any(keyword in field_lower for keyword in ['displacement', 'disp', 'u_', 'displ']):
            if any(suffix in field_lower for suffix in ['_x', '_y', '_z']):
                component = self.extract_component(field_name)
                category = 'displacement'
                aliases = ['displacement', f'displacement{component}']
            else:
                category = 'displacement'
                aliases = ['displacement']
        
        # Strain fields
        elif any(keyword in field_lower for keyword in ['strain', 'epsilon', 'e_']):
            category = 'strain'
            aliases = ['strain']
        
        # Velocity fields
        elif any(keyword in field_lower for keyword in ['velocity', 'vel', 'v_']):
            category = 'velocity'
            aliases = ['velocity']
        
        # Pressure fields
        elif any(keyword in field_lower for keyword in ['pressure', 'press', 'p_']):
            category = 'pressure'
            aliases = ['pressure']
        
        # Force fields
        elif any(keyword in field_lower for keyword in ['force', 'F_', 'load']):
            category = 'force'
            aliases = ['force']
        
        # Energy fields
        elif any(keyword in field_lower for keyword in ['energy', 'E_', 'work']):
            category = 'energy'
            aliases = ['energy']
        
        # Other fields
        else:
            category = 'other'
            aliases = [field_name]
        
        return category, aliases
    
    def extract_component(self, field_name):
        """Extract component suffix from field name"""
        patterns = [
            ('_xx', '_xx'), ('_yy', '_yy'), ('_zz', '_zz'),
            ('_xy', '_xy'), ('_yz', '_yz'), ('_xz', '_xz'),
            ('_x', '_x'), ('_y', '_y'), ('_z', '_z'),
            ('_1', '_1'), ('_2', '_2'), ('_3', '_3')
        ]
        
        for pattern, suffix in patterns:
            if field_name.endswith(pattern):
                return suffix
        
        return ''
    
    def discover_fields_from_summaries(self, summaries):
        """Discover all fields from simulation summaries"""
        field_info = defaultdict(set)
        
        for summary in summaries:
            for field_name in summary['field_stats'].keys():
                self.all_fields.add(field_name)
                category, aliases = self.categorize_field(field_name)
                self.field_categories[category].add(field_name)
                
                # Store metadata
                if field_name not in self.field_metadata:
                    self.field_metadata[field_name] = {
                        'category': category,
                        'aliases': aliases,
                        'simulations': set(),
                        'timesteps': 0,
                        'has_data': False
                    }
                
                # Track which simulations have this field
                self.field_metadata[field_name]['simulations'].add(summary['name'])
                
                # Check if field has valid data
                stats = summary['field_stats'][field_name]
                if stats['mean'] and len(stats['mean']) > 0:
                    self.field_metadata[field_name]['has_data'] = True
                    self.field_metadata[field_name]['timesteps'] = max(
                        self.field_metadata[field_name]['timesteps'],
                        len(stats['mean'])
                    )
        
        return self.get_field_availability_report()
    
    def get_field_availability_report(self):
        """Generate report of field availability across simulations"""
        report = {
            'total_fields': len(self.all_fields),
            'fields_by_category': {},
            'coverage_stats': {}
        }
        
        # Count fields by category
        for category, fields in self.field_categories.items():
            if fields:
                report['fields_by_category'][category] = {
                    'count': len(fields),
                    'fields': sorted(fields)
                }
        
        # Calculate coverage statistics
        for field_name, metadata in self.field_metadata.items():
            sim_count = len(metadata['simulations'])
            coverage = sim_count / max(len(self.field_metadata), 1)
            
            report['coverage_stats'][field_name] = {
                'simulation_count': sim_count,
                'coverage_percentage': coverage * 100,
                'has_data': metadata['has_data']
            }
        
        return report
    
    def get_fields_by_category(self, category=None):
        """Get fields filtered by category"""
        if category and category in self.field_categories:
            return sorted(self.field_categories[category])
        elif category == 'all':
            return sorted(self.all_fields)
        else:
            # Return all fields grouped by category
            grouped = {}
            for cat, fields in self.field_categories.items():
                if fields:
                    grouped[cat] = sorted(fields)
            return grouped
    
    def suggest_alternative_fields(self, field_name, available_fields):
        """Suggest alternative fields if requested field is not available"""
        suggestions = []
        
        # Try to find by category
        if field_name in self.field_metadata:
            category = self.field_metadata[field_name]['category']
            category_fields = self.field_categories[category]
            suggestions.extend([f for f in category_fields if f in available_fields])
        
        # Try to find by aliases
        for alias in self.field_aliases.keys():
            if alias in field_name.lower():
                target_category = self.field_aliases[alias]
                for cat_field in self.field_categories.get(target_category, []):
                    if cat_field in available_fields:
                        suggestions.append(cat_field)
        
        return list(set(suggestions))[:5]  # Return top 5 suggestions

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
        self.mesh_info = {}
        self.field_manager = FieldManager()
        
    def parse_folder_name(self, folder: str):
        """q0p5mJ-delta4p2ns → (0.5, 4.2)"""
        match = re.match(r"q([\dp\.]+)mJ-delta([\dp\.]+)ns", folder)
        if not match:
            return None, None
        e, d = match.groups()
        return float(e.replace("p", ".")), float(d.replace("p", "."))
    
    @st.cache_data(show_spinner=False, ttl=3600)
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
                    
                    # Enhanced triangle extraction with validation
                    triangles = None
                    for cell_block in mesh0.cells:
                        if cell_block.type in ["triangle", "tetra", "quad"]:
                            triangles = cell_block.data.astype(np.int32)
                            # Validate triangle indices
                            if triangles.max() >= n_pts:
                                st.warning(f"Invalid triangle indices in {name}, creating surface mesh...")
                                triangles = _self._create_surface_mesh(points)
                            break
                    
                    # If no triangles found, create surface mesh
                    if triangles is None:
                        triangles = _self._create_surface_mesh(points)
                    
                    # Initialize fields with enhanced validation
                    fields = {}
                    for key in mesh0.point_data.keys():
                        arr = mesh0.point_data[key].astype(np.float32)
                        if arr.ndim == 1:
                            sim_data['field_info'][key] = ("scalar", 1)
                            fields[key] = np.full((len(vtu_files), n_pts), np.nan, dtype=np.float32)
                        elif arr.ndim == 2 and arr.shape[1] in [2, 3, 6, 9]:
                            # Handle vectors and tensors
                            sim_data['field_info'][key] = ("vector", arr.shape[1])
                            fields[key] = np.full((len(vtu_files), n_pts, arr.shape[1]), np.nan, dtype=np.float32)
                        else:
                            # Skip malformed fields
                            continue
                        
                        fields[key][0] = arr
                        _self.available_fields.add(key)
                    
                    # Load remaining timesteps
                    for t in range(1, len(vtu_files)):
                        try:
                            mesh = meshio.read(vtu_files[t])
                            for key in sim_data['field_info']:
                                if key in mesh.point_data:
                                    arr = mesh.point_data[key].astype(np.float32)
                                    # Validate shape consistency
                                    if arr.shape == fields[key][0].shape:
                                        fields[key][t] = arr
                                    else:
                                        st.warning(f"Shape mismatch for {key} in {name} timestep {t}")
                        except Exception as e:
                            st.warning(f"Error loading timestep {t} in {name}: {e}")
                    
                    sim_data.update({
                        'points': points,
                        'fields': fields,
                        'triangles': triangles,
                        'has_mesh': True,
                        'n_points': n_pts,
                        'n_triangles': len(triangles) if triangles is not None else 0
                    })
                    
                    # Store mesh info for validation
                    _self.mesh_info[name] = {
                        'n_points': n_pts,
                        'n_triangles': len(triangles) if triangles is not None else 0,
                        'bounds': {
                            'x_min': float(points[:, 0].min()),
                            'x_max': float(points[:, 0].max()),
                            'y_min': float(points[:, 1].min()),
                            'y_max': float(points[:, 1].max()),
                            'z_min': float(points[:, 2].min()),
                            'z_max': float(points[:, 2].max())
                        }
                    }
                
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
            
            # Validate mesh consistency
            if load_full_mesh:
                _self._validate_mesh_consistency(simulations)
            
            # Discover and categorize fields
            field_report = _self.field_manager.discover_fields_from_summaries(summaries)
            st.info(f"📊 Discovered {field_report['total_fields']} fields across {len(simulations)} simulations")
        
        return simulations, summaries
    
    def _create_surface_mesh(self, points):
        """Create surface mesh from points using Delaunay triangulation"""
        try:
            # Project points to 2D for triangulation (use XY plane for simplicity)
            points_2d = points[:, :2]
            tri = Delaunay(points_2d)
            return tri.simplices.astype(np.int32)
        except Exception as e:
            st.warning(f"Failed to create surface mesh: {e}")
            return np.array([], dtype=np.int32)
    
    def _validate_mesh_consistency(self, simulations):
        """Validate that all meshes are compatible"""
        if not simulations:
            return
        
        mesh_stats = {}
        for name, sim in simulations.items():
            if sim.get('has_mesh', False):
                mesh_stats[name] = {
                    'n_points': sim.get('n_points', 0),
                    'n_triangles': sim.get('n_triangles', 0)
                }
        
        if len(mesh_stats) < 2:
            return
        
        # Check point count consistency
        point_counts = [stats['n_points'] for stats in mesh_stats.values()]
        if len(set(point_counts)) > 1:
            st.warning(f"⚠️ Inconsistent mesh sizes: Point counts vary from {min(point_counts)} to {max(point_counts)}")
            st.info("Interpolation may be inaccurate with inconsistent mesh sizes.")
    
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
        self.cache = {}
        
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
    
    @st.cache_data(show_spinner=False, max_entries=50, ttl=600)
    def interpolate_full_field_cached(_self, field_name, attention_weights, source_metadata, 
                                     simulation_names, timestep_indices, top_k=5):
        """Cached version of interpolate_full_field for better performance"""
        return _self._interpolate_full_field_impl(field_name, attention_weights, source_metadata,
                                                 simulation_names, timestep_indices, top_k)
    
    def interpolate_full_field(self, field_name, attention_weights, source_metadata, 
                              simulations, top_k=5, use_cache=True):
        """Compute interpolated full field using attention weights with enhanced handling.
        
        Args:
            field_name (str): Field to interpolate (e.g., 'temperature').
            attention_weights (np.array): Weights from _multi_head_attention.
            source_metadata (list): Metadata for sources.
            simulations (dict): Loaded simulations with full fields.
            top_k (int): Use only top K sources by weight.
            use_cache (bool): Whether to use caching.
        
        Returns:
            np.array: Interpolated field values (n_points or n_points x components).
            list: List of used sources with metadata.
        """
        if not self.fitted or len(attention_weights) == 0:
            return None, []
        
        # Prepare cache key if caching is enabled
        if use_cache:
            # Create stable identifiers for cache
            sim_names = []
            timestep_idxs = []
            
            for idx, weight in enumerate(attention_weights):
                if weight < 1e-6:
                    continue
                meta = source_metadata[idx]
                sim_names.append(meta['name'])
                timestep_idxs.append(meta['timestep_idx'])
            
            cache_key = EnhancedCacheManager.create_cache_key(
                field_name,
                tuple(sim_names[:10]),
                tuple(timestep_idxs[:10]),
                top_k
            )
            
            if cache_key in self.cache:
                st.info("Using cached interpolation result")
                return self.cache[cache_key]
        
        # Get top K sources by weight
        sorted_indices = np.argsort(attention_weights)[::-1]
        top_indices = sorted_indices[:min(top_k, len(attention_weights))]
        used_sources = []
        
        # Assume common mesh: Get shape from first simulation with the field
        field_shape = None
        field_type = None
        
        for idx in top_indices:
            if attention_weights[idx] < 1e-6:
                continue
                
            meta = source_metadata[idx]
            sim_name = meta['name']
            
            if sim_name in simulations:
                sim = simulations[sim_name]
                if 'fields' in sim and field_name in sim['fields']:
                    field_shape = sim['fields'][field_name].shape[1:]
                    field_type = sim['field_info'].get(field_name, ("unknown", 0))[0]
                    break
        
        if field_shape is None:
            st.warning(f"Field '{field_name}' not found in any of the top {top_k} source simulations.")
            return None, []
        
        # Initialize interpolated field based on type
        if field_type == "scalar":
            interpolated_field = np.zeros(field_shape, dtype=np.float32)
        else:  # vector or tensor
            interpolated_field = np.zeros(field_shape, dtype=np.float32)
        
        total_weight = 0.0
        
        # Weighted interpolation from top K sources
        for idx in top_indices:
            weight = attention_weights[idx]
            if weight < 1e-6:
                continue
            
            meta = source_metadata[idx]
            sim_name = meta['name']
            timestep_idx = meta['timestep_idx']
            
            if sim_name in simulations:
                sim = simulations[sim_name]
                if 'fields' in sim and field_name in sim['fields']:
                    source_field = sim['fields'][field_name][timestep_idx]
                    
                    # Validate shape consistency
                    if source_field.shape[1:] == field_shape:
                        interpolated_field += weight * source_field
                        total_weight += weight
                        used_sources.append({
                            'name': sim_name,
                            'weight': weight,
                            'energy': meta['energy'],
                            'duration': meta['duration'],
                            'time': meta['time']
                        })
                    else:
                        st.warning(f"Shape mismatch for {field_name} in {sim_name}")
                else:
                    st.warning(f"Field '{field_name}' not found in {sim_name}")
            else:
                st.warning(f"Simulation {sim_name} not found in loaded data")
        
        if total_weight > 0:
            interpolated_field /= total_weight
        else:
            st.error("No valid sources found for interpolation.")
            return None, []
        
        # Cache the result if caching is enabled
        if use_cache:
            self.cache[cache_key] = (interpolated_field, used_sources)
        
        return interpolated_field, used_sources
    
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
    def create_surface_from_points(points, values, resolution=50):
        """Create surface mesh from irregular 3D points using griddata"""
        try:
            # Project points to 2D plane (use PCA or simple projection)
            # For simplicity, use XY plane and interpolate Z values
            x = points[:, 0]
            y = points[:, 1]
            z = points[:, 2]
            
            # Create grid
            xi = np.linspace(x.min(), x.max(), resolution)
            yi = np.linspace(y.min(), y.max(), resolution)
            xi, yi = np.meshgrid(xi, yi)
            
            # Interpolate Z values
            zi = griddata((x, y), z, (xi, yi), method='cubic')
            
            # Interpolate field values
            vi = griddata((x, y), values, (xi, yi), method='cubic')
            
            # Handle NaN values
            zi = np.nan_to_num(zi)
            vi = np.nan_to_num(vi)
            
            return xi, yi, zi, vi
        except Exception as e:
            st.warning(f"Surface creation failed: {e}")
            return None, None, None, None
    
    @staticmethod
    def create_mesh_with_wireframe(points, triangles, values, wireframe=False):
        """Create mesh visualization with optional wireframe"""
        if triangles is not None and len(triangles) > 0:
            # Validate and filter triangles
            valid_mask = np.all(triangles < len(points), axis=1)
            valid_triangles = triangles[valid_mask]
            
            if len(valid_triangles) == 0:
                # No valid triangles, fall back to point cloud
                return go.Scatter3d(
                    x=points[:, 0], y=points[:, 1], z=points[:, 2],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=values,
                        colorscale='Viridis',
                        opacity=0.8,
                        showscale=True
                    )
                )
            
            # Create mesh
            mesh = go.Mesh3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2],
                i=valid_triangles[:, 0], j=valid_triangles[:, 1], k=valid_triangles[:, 2],
                intensity=values,
                colorscale='Viridis',
                intensitymode='vertex',
                opacity=0.8,
                lighting=dict(
                    ambient=0.8,
                    diffuse=0.8,
                    specular=0.5,
                    roughness=0.5
                ),
                name='Surface'
            )
            
            if wireframe:
                # Create wireframe edges
                edges = set()
                for tri in valid_triangles[:1000]:  # Limit for performance
                    edges.add(tuple(sorted([tri[0], tri[1]])))
                    edges.add(tuple(sorted([tri[1], tri[2]])))
                    edges.add(tuple(sorted([tri[2], tri[0]])))
                
                # Create line traces
                lines_x, lines_y, lines_z = [], [], []
                for edge in list(edges)[:2000]:  # Limit for performance
                    lines_x.extend([points[edge[0], 0], points[edge[1], 0], None])
                    lines_y.extend([points[edge[0], 1], points[edge[1], 1], None])
                    lines_z.extend([points[edge[0], 2], points[edge[1], 2], None])
                
                wireframe_trace = go.Scatter3d(
                    x=lines_x, y=lines_y, z=lines_z,
                    mode='lines',
                    line=dict(color='darkgray', width=1),
                    hoverinfo='none',
                    name='Wireframe'
                )
                
                return [mesh, wireframe_trace]
            else:
                return mesh
        else:
            # Fall back to point cloud
            return go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    color=values,
                    colorscale='Viridis',
                    opacity=0.8,
                    showscale=True
                ),
                name='Point Cloud'
            )
    
    @staticmethod
    def create_field_summary_chart(field_manager):
        """Create chart showing field distribution by category"""
        categories = {}
        for category, fields in field_manager.field_categories.items():
            if fields:
                categories[category] = len(fields)
        
        if not categories:
            return go.Figure()
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(categories.keys()),
                y=list(categories.values()),
                text=list(categories.values()),
                textposition='auto',
                marker_color='lightblue'
            )
        ])
        
        fig.update_layout(
            title="Field Distribution by Category",
            xaxis_title="Category",
            yaxis_title="Number of Fields",
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_field_coverage_chart(field_manager):
        """Create chart showing field coverage across simulations"""
        coverage_data = []
        for field_name, metadata in field_manager.field_metadata.items():
            coverage_data.append({
                'field': field_name,
                'simulations': len(metadata['simulations']),
                'has_data': metadata['has_data']
            })
        
        if not coverage_data:
            return go.Figure()
        
        df = pd.DataFrame(coverage_data)
        df = df.sort_values('simulations', ascending=False).head(20)  # Top 20 fields
        
        fig = go.Figure(data=[
            go.Bar(
                x=df['field'],
                y=df['simulations'],
                text=df['simulations'],
                textposition='auto',
                marker_color=['green' if h else 'orange' for h in df['has_data']]
            )
        ])
        
        fig.update_layout(
            title="Field Coverage Across Simulations (Top 20)",
            xaxis_title="Field Name",
            yaxis_title="Number of Simulations",
            height=500,
            xaxis_tickangle=-45
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
    .field-category {
        background-color: #f0f8ff;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 5px;
        border-left: 4px solid #3498db;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🔬 Enhanced FEA Laser Simulation Platform</h1>', 
                unsafe_allow_html=True)
    
    # Initialize session state with enhanced persistence
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = UnifiedFEADataLoader()
        st.session_state.extrapolator = EnhancedPhysicsInformedAttentionExtrapolator()
        st.session_state.visualizer = EnhancedVisualizer()
        st.session_state.data_loaded = False
        st.session_state.current_mode = "Data Viewer"
        st.session_state.selected_colormap = "Viridis"
        st.session_state.debug_mode = False
        st.session_state.last_field = None
        st.session_state.last_timestep = 0
        st.session_state.cache_cleared = False
        st.session_state.field_category_filter = "all"
        st.session_state.show_all_fields = False
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Navigation")
        app_mode = st.radio(
            "Select Mode",
            ["Data Viewer", "Interpolation/Extrapolation", "Comparative Analysis", "Field Explorer"],
            index=["Data Viewer", "Interpolation/Extrapolation", "Comparative Analysis", "Field Explorer"].index(
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
        
        # Debug mode toggle
        st.session_state.debug_mode = st.checkbox("🔧 Debug Mode", value=False,
                                                 help="Show debug information and raw data")
        
        # Field display options
        st.session_state.show_all_fields = st.checkbox("📋 Show All Fields", value=False,
                                                      help="Show all fields in dropdowns (may be slow)")
        
        # Cache management
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Clear Cache", use_container_width=True):
                st.cache_data.clear()
                st.session_state.extrapolator.cache = {}
                st.session_state.cache_cleared = True
                st.success("Cache cleared!")
                time.sleep(1)
                st.rerun()
        
        with col2:
            if st.button("🔄 Load/Reload Data", type="primary", use_container_width=True):
                with st.spinner("Loading all simulation data..."):
                    # Clear cache and reload
                    st.cache_data.clear()
                    simulations, summaries = st.session_state.data_loader.load_all_simulations(
                        load_full_mesh=load_full_data
                    )
                    st.session_state.simulations = simulations
                    st.session_state.summaries = summaries
                    
                    if simulations and summaries:
                        st.session_state.extrapolator.load_summaries(summaries)
                        st.session_state.data_loaded = True
                        
                        # Discover and categorize all fields
                        field_report = st.session_state.data_loader.field_manager.discover_fields_from_summaries(summaries)
                        
                        # Store field information in session state
                        st.session_state.all_fields = sorted(st.session_state.data_loader.field_manager.all_fields)
                        st.session_state.field_categories = st.session_state.data_loader.field_manager.field_categories
                        st.session_state.field_metadata = st.session_state.data_loader.field_manager.field_metadata
                        
                        # Store field coverage report
                        st.session_state.field_coverage = field_report
                        
                        st.success(f"✅ Data loaded successfully! Found {len(st.session_state.all_fields)} unique fields.")
        
        if st.session_state.data_loaded:
            st.markdown("---")
            st.markdown("### 📈 Loaded Data Overview")
            
            with st.expander("📊 Data Statistics", expanded=True):
                st.metric("Simulations", len(st.session_state.simulations))
                st.metric("Total Fields", len(st.session_state.all_fields))
                
                if st.session_state.summaries:
                    energies = [s['energy'] for s in st.session_state.summaries]
                    durations = [s['duration'] for s in st.session_state.summaries]
                    st.metric("Energy Range", f"{min(energies):.1f} - {max(energies):.1f} mJ")
                    st.metric("Duration Range", f"{min(durations):.1f} - {max(durations):.1f} ns")
            
            # Field category filter for dropdowns
            st.markdown("### 🔍 Field Filter")
            category_options = ["all"] + [cat for cat in st.session_state.field_categories.keys() 
                                         if st.session_state.field_categories[cat]]
            st.session_state.field_category_filter = st.selectbox(
                "Filter by Category",
                category_options,
                index=0,
                help="Filter fields by category in dropdowns"
            )
            
            # Field explorer in sidebar
            with st.expander("📋 Available Fields", expanded=False):
                if 'field_coverage' in st.session_state:
                    report = st.session_state.field_coverage
                    st.write(f"**Total Fields:** {report['total_fields']}")
                    
                    for category, info in report['fields_by_category'].items():
                        with st.expander(f"📁 {category.title()} ({info['count']} fields)"):
                            for field in info['fields']:
                                coverage = report['coverage_stats'].get(field, {})
                                sim_count = coverage.get('simulation_count', 0)
                                has_data = coverage.get('has_data', False)
                                
                                status_icon = "✅" if has_data else "⚠️"
                                st.write(f"{status_icon} **{field}** ({sim_count} sims)")
    
    # Main content based on selected mode
    if app_mode == "Data Viewer":
        render_data_viewer()
    elif app_mode == "Interpolation/Extrapolation":
        render_interpolation_extrapolation()
    elif app_mode == "Comparative Analysis":
        render_comparative_analysis()
    elif app_mode == "Field Explorer":
        render_field_explorer()

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
    
    # Enhanced field selection with categories
    st.markdown('<h3 class="sub-header">🎯 Field Selection</h3>', unsafe_allow_html=True)
    
    # Get available fields in this simulation
    sim_fields = sorted(sim['field_info'].keys())
    
    # Filter fields by category if requested
    if st.session_state.field_category_filter != "all":
        category_fields = st.session_state.field_categories.get(
            st.session_state.field_category_filter, set()
        )
        sim_fields = [f for f in sim_fields if f in category_fields]
    
    if not sim_fields:
        st.warning(f"No fields found in category '{st.session_state.field_category_filter}' for this simulation.")
        st.info("Try changing the category filter or selecting a different simulation.")
        return
    
    # Display field selection in columns for better organization
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Field category information
        st.markdown("**Field Information:**")
        st.write(f"**Total fields in sim:** {len(sim_fields)}")
        
        # Show field categories present in this simulation
        field_categories_in_sim = defaultdict(list)
        for field in sim_fields:
            if field in st.session_state.field_metadata:
                category = st.session_state.field_metadata[field]['category']
                field_categories_in_sim[category].append(field)
        
        if field_categories_in_sim:
            st.markdown("**Categories in this simulation:**")
            for category, fields in field_categories_in_sim.items():
                st.write(f"• {category.title()}: {len(fields)} fields")
    
    with col2:
        # Main field selection dropdown
        try:
            # Format field names with category information
            field_options = []
            for field in sim_fields:
                if field in st.session_state.field_metadata:
                    category = st.session_state.field_metadata[field]['category']
                    field_type = sim['field_info'][field][0]
                    display_name = f"{field} ({category}, {field_type})"
                else:
                    display_name = field
                field_options.append((field, display_name))
            
            # Sort by display name
            field_options.sort(key=lambda x: x[1])
            
            # Create selection with enhanced display
            selected_field_display = st.selectbox(
                "Select Field",
                options=[opt[1] for opt in field_options],
                key="viewer_field_select",
                help="Choose a field to visualize"
            )
            
            # Extract actual field name from selection
            field = next(opt[0] for opt in field_options if opt[1] == selected_field_display)
            
            # Store last field for change detection
            if field != st.session_state.get('last_field'):
                st.session_state.last_field = field
                # Clear field-specific cache
                if 'field_cache' in st.session_state:
                    del st.session_state.field_cache
        except (KeyError, StopIteration) as e:
            st.error(f"Error accessing field information: {e}")
            st.info("Try reloading the data or selecting a different simulation.")
            return
    
    with col3:
        # Timestep selection
        timestep = st.slider(
            "Timestep",
            0, sim['n_timesteps'] - 1, 0,
            key="viewer_timestep_slider",
            help="Select timestep to display"
        )
        
        # Visualization options
        colormap = st.selectbox(
            "Colormap",
            EnhancedVisualizer.EXTENDED_COLORMAPS,
            index=EnhancedVisualizer.EXTENDED_COLORMAPS.index(st.session_state.selected_colormap),
            key="viewer_colormap"
        )
        
        # Store last timestep for change detection
        if timestep != st.session_state.get('last_timestep'):
            st.session_state.last_timestep = timestep
    
    # Field information panel
    with st.expander("📋 Field Details", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if field in st.session_state.field_metadata:
                metadata = st.session_state.field_metadata[field]
                st.write(f"**Category:** {metadata['category'].title()}")
                st.write(f"**Simulations with field:** {len(metadata['simulations'])}")
        
        with col2:
            if field in sim['field_info']:
                field_type, components = sim['field_info'][field]
                st.write(f"**Type:** {field_type}")
                st.write(f"**Components:** {components}")
        
        with col3:
            # Quick field statistics
            if field in sim['fields']:
                data = sim['fields'][field][timestep]
                if data.ndim == 1:
                    clean_data = data[~np.isnan(data)]
                    if len(clean_data) > 0:
                        st.write(f"**Min:** {clean_data.min():.3f}")
                        st.write(f"**Max:** {clean_data.max():.3f}")
    
    # Main 3D visualization with enhanced error handling
    try:
        if 'points' in sim and 'fields' in sim and field in sim['fields']:
            pts = sim['points']
            kind, _ = sim['field_info'].get(field, ("scalar", 1))
            raw = sim['fields'][field][timestep]
            
            if kind == "scalar":
                values = np.where(np.isnan(raw), 0, raw)
                label = field
            else:
                # Handle vector fields - compute magnitude
                if raw.ndim == 2:
                    magnitude = np.linalg.norm(raw, axis=1)
                else:
                    magnitude = np.abs(raw)
                values = np.where(np.isnan(magnitude), 0, magnitude)
                label = f"{field} (magnitude)"
            
            # Enhanced surface rendering with wireframe option
            wireframe = st.checkbox("Show Wireframe", value=False, key="viewer_wireframe")
            
            # Create visualization
            mesh_data = st.session_state.visualizer.create_mesh_with_wireframe(
                pts, sim.get('triangles'), values, wireframe
            )
            
            if isinstance(mesh_data, list):
                # Multiple traces (mesh + wireframe)
                fig = go.Figure(data=mesh_data)
            else:
                # Single trace
                fig = go.Figure(data=[mesh_data])
            
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
            
            # Debug information
            if st.session_state.debug_mode:
                with st.expander("🔧 Debug Information"):
                    st.write(f"**Field Type:** {kind}")
                    st.write(f"**Shape:** {raw.shape}")
                    st.write(f"**Triangle Count:** {len(sim.get('triangles', []))}")
                    st.write(f"**Valid Values:** {np.sum(~np.isnan(values))}/{len(values)}")
        
        else:
            st.error(f"Field '{field}' not found in simulation data.")
            
    except Exception as e:
        st.error(f"Error rendering 3D visualization: {str(e)}")
        if st.session_state.debug_mode:
            st.exception(e)
        st.info("Try selecting a different field or timestep.")
    
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
    
    # Advanced model parameters
    with st.expander("⚙️ Advanced Parameters", expanded=False):
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
        
        # Performance options
        col1, col2 = st.columns(2)
        with col1:
            top_k_sources = st.slider(
                "Top K Sources",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                key="interp_top_k",
                help="Use only top K sources by attention weight"
            )
        with col2:
            use_caching = st.checkbox(
                "Use Caching",
                value=True,
                key="interp_caching",
                help="Cache interpolation results for better performance"
            )
        
        # Update extrapolator parameters
        st.session_state.extrapolator.sigma_param = sigma_param
        st.session_state.extrapolator.spatial_weight = spatial_weight
        st.session_state.extrapolator.n_heads = n_heads
        st.session_state.extrapolator.temperature = temperature
    
    # Run prediction
    if st.button("🚀 Run Physics-Informed Prediction", type="primary", use_container_width=True):
        with st.spinner("Running multi-head attention prediction with spatial locality regulation..."):
            try:
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
                    
                    # Store results in session state
                    st.session_state.last_prediction = {
                        'results': results,
                        'energy': energy_query,
                        'duration': duration_query,
                        'time_points': time_points
                    }
                    
                    # Visualization tabs
                    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Predictions", "🧠 Attention", "🌐 3D Analysis", "📊 Details", "🖼️ 3D Rendering"])
                    
                    with tab1:
                        render_prediction_results(results, time_points, energy_query, duration_query)
                    
                    with tab2:
                        render_attention_visualization(results, energy_query, duration_query, time_points)
                    
                    with tab3:
                        render_3d_analysis(results, time_points, energy_query, duration_query)
                    
                    with tab4:
                        render_detailed_results(results, time_points, energy_query, duration_query)
                    
                    with tab5:
                        render_3d_interpolation_visualization(
                            results, time_points, energy_query, duration_query,
                            top_k=top_k_sources, use_caching=use_caching
                        )
                else:
                    st.error("Prediction failed. Please check input parameters and ensure sufficient training data.")
                    
            except Exception as e:
                st.error(f"Prediction failed with error: {str(e)}")
                if st.session_state.debug_mode:
                    st.exception(e)
                st.info("Try adjusting parameters or reloading the data.")

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
        
        st.dataframe(styled_df, use_container_width=True, height=400)

def render_3d_interpolation_visualization(results, time_points, energy_query, duration_query, 
                                         top_k=5, use_caching=True):
    """Enhanced 3D visualization of interpolated/extrapolated fields"""
    st.markdown('<h4 class="sub-header">🖼️ 3D Interpolation Visualization</h4>', unsafe_allow_html=True)
    
    if not st.session_state.get('simulations'):
        st.warning("No full simulations loaded for 3D rendering. Please reload with 'Load Full Mesh' enabled.")
        return
    
    # Check if simulations have full mesh data
    first_sim = next(iter(st.session_state.simulations.values()))
    if not first_sim.get('has_mesh', False):
        st.warning("Full mesh data not available. Please reload simulations with 'Load Full Mesh' enabled.")
        return
    
    # Select field and timestep for 3D rendering
    available_fields = list(results['field_predictions'].keys())
    if not available_fields:
        st.warning("No predicted fields available for 3D rendering.")
        return
    
    # Enhanced field selection with categories
    col1, col2, col3 = st.columns(3)
    
    with col1:
        try:
            # Filter fields by category if requested
            filtered_fields = available_fields
            if st.session_state.field_category_filter != "all":
                category_fields = st.session_state.field_categories.get(
                    st.session_state.field_category_filter, set()
                )
                filtered_fields = [f for f in available_fields if f in category_fields]
            
            if not filtered_fields:
                st.warning(f"No fields found in category '{st.session_state.field_category_filter}' for prediction.")
                st.info("Try changing the category filter or selecting different parameters.")
                return
            
            # Format field names with category information
            field_options = []
            for field in filtered_fields:
                if field in st.session_state.field_metadata:
                    category = st.session_state.field_metadata[field]['category']
                    display_name = f"{field} ({category})"
                else:
                    display_name = field
                field_options.append((field, display_name))
            
            # Sort by display name
            field_options.sort(key=lambda x: x[1])
            
            # Create selection with enhanced display
            selected_field_display = st.selectbox(
                "Select Field for 3D Rendering", 
                options=[opt[1] for opt in field_options],
                key="interp_3d_field",
                help="Choose a field to visualize in 3D"
            )
            
            # Extract actual field name from selection
            selected_field_3d = next(opt[0] for opt in field_options if opt[1] == selected_field_display)
            
        except Exception as e:
            st.error(f"Error selecting field: {str(e)}")
            st.info("Try selecting a different field or check field availability in simulations.")
            return
    
    with col2:
        try:
            timestep_idx_3d = st.slider(
                "Select Timestep for 3D", 
                0, len(time_points) - 1, 0, 
                key="interp_3d_timestep"
            )
            selected_time_3d = time_points[timestep_idx_3d]
        except Exception as e:
            st.error(f"Error selecting timestep: {str(e)}")
            return
    
    with col3:
        # Add visualization options
        render_mode = st.selectbox(
            "Rendering Mode",
            ["Surface Mesh", "Point Cloud", "Wireframe", "Surface + Wireframe"],
            index=0,
            key="render_mode"
        )
        
        # Performance options
        subsample = st.checkbox("Subsample Points", value=False, 
                               help="Subsample points for faster rendering")
    
    # Get attention weights for selected timestep
    if timestep_idx_3d < len(results['attention_maps']):
        attention_weights_3d = results['attention_maps'][timestep_idx_3d]
        
        # Check field availability in simulations
        available_simulations, missing_in = EnhancedCacheManager.validate_data_consistency(
            st.session_state.simulations, selected_field_3d
        )
        
        if missing_in:
            st.warning(f"Field '{selected_field_3d}' not available in {len(missing_in)} simulations.")
        
        # Compute interpolated field with enhanced error handling
        try:
            with st.spinner(f"Interpolating {selected_field_3d} field at t={selected_time_3d} ns..."):
                interpolated_values, used_sources = st.session_state.extrapolator.interpolate_full_field(
                    selected_field_3d,
                    attention_weights_3d,
                    st.session_state.extrapolator.source_metadata,
                    st.session_state.simulations,
                    top_k=top_k,
                    use_cache=use_caching
                )
            
            if interpolated_values is None:
                st.error(f"Failed to interpolate field '{selected_field_3d}'. Check field availability in source simulations.")
                return
            
            # Get common mesh from first simulation
            first_sim = next(iter(st.session_state.simulations.values()))
            pts = first_sim['points']
            triangles = first_sim.get('triangles')
            
            # Subsample if requested
            if subsample and len(pts) > 10000:
                step = max(1, len(pts) // 5000)
                idx = np.arange(0, len(pts), step)
                pts = pts[idx]
                interpolated_values = interpolated_values[idx]
                if triangles is not None:
                    # Keep only triangles where all vertices are in the subsampled set
                    mask = np.isin(triangles, idx).all(axis=1)
                    triangles = triangles[mask]
                    # Remap indices
                    idx_map = {old: new for new, old in enumerate(idx)}
                    triangles = np.vectorize(idx_map.get)(triangles)
            
            # Handle scalar/vector fields
            if interpolated_values.ndim == 1:
                values = np.nan_to_num(interpolated_values)  # Replace NaNs
                label = selected_field_3d
            else:
                # For vector fields, use magnitude
                magnitude = np.linalg.norm(interpolated_values, axis=1)
                values = np.nan_to_num(magnitude)
                label = f"{selected_field_3d} (magnitude)"
            
            # Display confidence information
            if timestep_idx_3d < len(results['confidence_scores']):
                confidence = results['confidence_scores'][timestep_idx_3d]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Interpolation Confidence", f"{confidence:.3f}")
                with col2:
                    st.metric("Field Range", f"{np.min(values):.3f} - {np.max(values):.3f}")
                with col3:
                    st.metric("Active Sources", f"{len(used_sources)}")
            
            # Create 3D visualization based on selected mode
            wireframe = "Wireframe" in render_mode or render_mode == "Surface + Wireframe"
            show_surface = "Surface" in render_mode or render_mode == "Surface + Wireframe"
            
            if show_surface and triangles is not None and len(triangles) > 0:
                # Create surface mesh with optional wireframe
                mesh_data = st.session_state.visualizer.create_mesh_with_wireframe(
                    pts, triangles, values, wireframe
                )
                
                if isinstance(mesh_data, list):
                    fig_3d = go.Figure(data=mesh_data)
                else:
                    fig_3d = go.Figure(data=[mesh_data])
            else:
                # Fall back to point cloud
                fig_3d = go.Figure(data=[
                    go.Scatter3d(
                        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                        mode='markers',
                        marker=dict(
                            size=4,
                            color=values,
                            colorscale=st.session_state.selected_colormap,
                            opacity=0.8,
                            showscale=True
                        ),
                        name='Point Cloud'
                    )
                ])
            
            # Update figure layout
            fig_3d.update_layout(
                title=dict(
                    text=f"Interpolated {label} at t={selected_time_3d} ns<br><sub>E={energy_query:.1f} mJ, τ={duration_query:.1f} ns | Confidence: {confidence:.3f}</sub>",
                    font=dict(size=18)
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
            
            st.plotly_chart(fig_3d, use_container_width=True)
            
            # Additional analysis
            with st.expander("📊 Field Analysis", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogram of field values
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(
                        x=values,
                        nbinsx=50,
                        marker_color='skyblue',
                        opacity=0.7,
                        name='Value Distribution'
                    ))
                    fig_hist.update_layout(
                        title=f"{label} Value Distribution",
                        xaxis_title="Value",
                        yaxis_title="Frequency",
                        height=300
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    # Top contributing sources
                    if used_sources:
                        df_top = pd.DataFrame(used_sources)
                        st.markdown(f"**Top {len(used_sources)} Contributing Sources:**")
                        st.dataframe(
                            df_top.style.format({
                                'weight': '{:.4f}',
                                'energy': '{:.2f}',
                                'duration': '{:.2f}',
                                'time': '{:.1f}'
                            }).background_gradient(subset=['weight'], cmap='YlOrRd'),
                            use_container_width=True,
                            height=200
                        )
            
            # Export options
            st.markdown("##### 💾 Export 3D Data")
            col1, col2 = st.columns(2)
            
            with col1:
                # Export as CSV
                df_export = pd.DataFrame({
                    'X': pts[:, 0],
                    'Y': pts[:, 1],
                    'Z': pts[:, 2],
                    label: values
                })
                csv = df_export.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"interpolated_{label}_E{energy_query:.1f}mJ_tau{duration_query:.1f}ns_t{selected_time_3d}ns.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Export as NPZ
                npz_buffer = BytesIO()
                np.savez_compressed(npz_buffer, 
                                   points=pts, 
                                   values=values, 
                                   field_name=label,
                                   energy=energy_query,
                                   duration=duration_query,
                                   time=selected_time_3d)
                npz_buffer.seek(0)
                
                st.download_button(
                    label="📥 Download as NPZ",
                    data=npz_buffer,
                    file_name=f"interpolated_{label}_E{energy_query:.1f}mJ_tau{duration_query:.1f}ns_t{selected_time_3d}ns.npz",
                    mime="application/octet-stream",
                    use_container_width=True
                )
                
        except Exception as e:
            st.error(f"Error during interpolation or visualization: {str(e)}")
            if st.session_state.debug_mode:
                st.exception(e)
            st.info("Try selecting a different field, adjusting parameters, or clearing the cache.")
    else:
        st.warning("No attention weights available for selected timestep.")

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
    
    # Field selection with enhanced dropdown
    st.markdown('<h3 class="sub-header">📈 Select Field for Analysis</h3>', unsafe_allow_html=True)
    
    # Get available fields from selected simulations
    available_fields = set()
    for sim_name in visualization_sims:
        if sim_name in simulations:
            available_fields.update(simulations[sim_name]['field_info'].keys())
    
    if not available_fields:
        st.error("No field data available for selected simulations.")
        return
    
    # Filter fields by category if requested
    filtered_fields = sorted(available_fields)
    if st.session_state.field_category_filter != "all":
        category_fields = st.session_state.field_categories.get(
            st.session_state.field_category_filter, set()
        )
        filtered_fields = [f for f in filtered_fields if f in category_fields]
    
    if not filtered_fields:
        st.warning(f"No fields found in category '{st.session_state.field_category_filter}' for selected simulations.")
        st.info("Try changing the category filter or selecting different simulations.")
        return
    
    # Enhanced field selection with categories
    field_options = []
    for field in filtered_fields:
        if field in st.session_state.field_metadata:
            category = st.session_state.field_metadata[field]['category']
            display_name = f"{field} ({category})"
        else:
            display_name = field
        field_options.append((field, display_name))
    
    # Sort by display name
    field_options.sort(key=lambda x: x[1])
    
    # Create selection with enhanced display
    selected_field_display = st.selectbox(
        "Select field for analysis",
        options=[opt[1] for opt in field_options],
        key="comparison_field",
        help="Choose a field to compare across simulations"
    )
    
    # Extract actual field name from selection
    selected_field = next(opt[0] for opt in field_options if opt[1] == selected_field_display)
    
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

def render_field_explorer():
    """Render the field explorer interface"""
    st.markdown('<h2 class="sub-header">🔍 Field Explorer</h2>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.markdown("""
        <div class="warning-box">
        <h3>⚠️ No Data Loaded</h3>
        <p>Please load simulations first to explore fields.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Display field statistics
    st.markdown('<h3 class="sub-header">📊 Field Statistics Overview</h3>', unsafe_allow_html=True)
    
    if 'field_coverage' in st.session_state:
        report = st.session_state.field_coverage
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Fields", report['total_fields'])
        with col2:
            st.metric("Simulations", len(st.session_state.simulations))
        with col3:
            # Calculate average fields per simulation
            fields_per_sim = np.mean([len(sim['field_info']) for sim in st.session_state.simulations.values()])
            st.metric("Avg Fields/Sim", f"{fields_per_sim:.1f}")
        with col4:
            # Calculate field coverage
            total_sims = len(st.session_state.simulations)
            avg_coverage = np.mean([stats['coverage_percentage'] 
                                   for stats in report['coverage_stats'].values()])
            st.metric("Avg Coverage", f"{avg_coverage:.1f}%")
    
    # Field distribution by category
    st.markdown('<h3 class="sub-header">📁 Field Distribution by Category</h3>', unsafe_allow_html=True)
    
    fig_dist = st.session_state.visualizer.create_field_summary_chart(
        st.session_state.data_loader.field_manager
    )
    if fig_dist.data:
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Field coverage chart
    st.markdown('<h3 class="sub-header">📈 Field Coverage Across Simulations</h3>', unsafe_allow_html=True)
    
    fig_cov = st.session_state.visualizer.create_field_coverage_chart(
        st.session_state.data_loader.field_manager
    )
    if fig_cov.data:
        st.plotly_chart(fig_cov, use_container_width=True)
    
    # Detailed field exploration
    st.markdown('<h3 class="sub-header">🔍 Field Details Explorer</h3>', unsafe_allow_html=True)
    
    # Field search and filter
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Field search
        search_term = st.text_input("Search fields", "", 
                                   placeholder="Enter field name or keyword")
    
    with col2:
        # Category filter
        category_options = ["all"] + [cat for cat in st.session_state.field_categories.keys() 
                                     if st.session_state.field_categories[cat]]
        selected_category = st.selectbox(
            "Filter by category",
            category_options,
            index=0
        )
    
    with col3:
        # Coverage filter
        min_coverage = st.slider(
            "Minimum coverage (%)",
            0, 100, 0,
            step=5,
            help="Filter fields by minimum simulation coverage"
        )
    
    # Get filtered fields
    filtered_fields = []
    for field_name, metadata in st.session_state.field_metadata.items():
        # Apply category filter
        if selected_category != "all" and metadata['category'] != selected_category:
            continue
        
        # Apply search filter
        if search_term and search_term.lower() not in field_name.lower():
            continue
        
        # Apply coverage filter
        coverage = report['coverage_stats'].get(field_name, {}).get('coverage_percentage', 0)
        if coverage < min_coverage:
            continue
        
        filtered_fields.append((field_name, metadata, coverage))
    
    # Sort fields
    sort_option = st.selectbox(
        "Sort by",
        ["Name (A-Z)", "Name (Z-A)", "Coverage (High-Low)", "Coverage (Low-High)"]
    )
    
    if sort_option == "Name (A-Z)":
        filtered_fields.sort(key=lambda x: x[0])
    elif sort_option == "Name (Z-A)":
        filtered_fields.sort(key=lambda x: x[0], reverse=True)
    elif sort_option == "Coverage (High-Low)":
        filtered_fields.sort(key=lambda x: x[2], reverse=True)
    elif sort_option == "Coverage (Low-High)":
        filtered_fields.sort(key=lambda x: x[2])
    
    # Display filtered fields
    if not filtered_fields:
        st.info("No fields match the selected filters.")
        return
    
    st.write(f"**Found {len(filtered_fields)} fields:**")
    
    # Pagination
    items_per_page = 20
    total_pages = max(1, (len(filtered_fields) + items_per_page - 1) // items_per_page)
    
    page_number = st.number_input(
        "Page",
        min_value=1,
        max_value=total_pages,
        value=1,
        step=1
    )
    
    start_idx = (page_number - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(filtered_fields))
    
    # Display fields for current page
    for i in range(start_idx, end_idx):
        field_name, metadata, coverage = filtered_fields[i]
        
        with st.expander(f"📊 {field_name}", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Category:** {metadata['category'].title()}")
                st.write(f"**Coverage:** {coverage:.1f}%")
            
            with col2:
                st.write(f"**Simulations:** {len(metadata['simulations'])}")
                st.write(f"**Has Data:** {'✅' if metadata['has_data'] else '⚠️'}")
            
            with col3:
                # Quick actions
                if st.button(f"View {field_name}", key=f"view_{field_name}"):
                    # Switch to Data Viewer mode and select a simulation with this field
                    st.session_state.current_mode = "Data Viewer"
                    # Find a simulation with this field
                    for sim_name in metadata['simulations']:
                        if sim_name in st.session_state.simulations:
                            st.session_state.viewer_sim_select = sim_name
                            st.session_state.viewer_field_select = field_name
                            st.rerun()
                            break
    
    # Field export options
    st.markdown("---")
    st.markdown("##### 💾 Export Field Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export as CSV
        export_data = []
        for field_name, metadata, coverage in filtered_fields:
            export_data.append({
                'Field Name': field_name,
                'Category': metadata['category'],
                'Simulation Count': len(metadata['simulations']),
                'Coverage (%)': coverage,
                'Has Data': metadata['has_data'],
                'Simulations': ', '.join(sorted(metadata['simulations'])[:5]) + 
                              ('...' if len(metadata['simulations']) > 5 else '')
            })
        
        df_export = pd.DataFrame(export_data)
        csv = df_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Field List as CSV",
            data=csv,
            file_name="field_inventory.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Export as JSON
        export_json = {}
        for field_name, metadata, coverage in filtered_fields:
            export_json[field_name] = {
                'category': metadata['category'],
                'simulation_count': len(metadata['simulations']),
                'coverage_percentage': coverage,
                'has_data': metadata['has_data'],
                'simulations': list(metadata['simulations'])
            }
        
        json_str = json.dumps(export_json, indent=2)
        st.download_button(
            label="📥 Download Field List as JSON",
            data=json_str.encode('utf-8'),
            file_name="field_inventory.json",
            mime="application/json",
            use_container_width=True
        )

if __name__ == "__main__":
    main()
