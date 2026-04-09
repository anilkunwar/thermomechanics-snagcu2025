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
import tempfile
import base64
import hashlib
import pickle
from functools import lru_cache
from collections import OrderedDict

warnings.filterwarnings('ignore')

# =============================================
# PATH CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEA_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "fea_solutions")
os.makedirs(FEA_SOLUTIONS_DIR, exist_ok=True)

# =============================================
# CACHE MANAGEMENT UTILITIES
# =============================================
class CacheManager:
    """Manages caching of interpolation results to prevent recomputation"""
    
    @staticmethod
    def generate_cache_key(field_name, timestep_idx, energy, duration, time, 
                          sigma_param, spatial_weight, n_heads, temperature,
                          sigma_g, s_E, s_tau, s_t, temporal_weight,  # Enhanced parameters
                          top_k=None, subsample_factor=None):
        """Generate a unique cache key for interpolation parameters"""
        params_str = f"{field_name}_{timestep_idx}_{energy:.2f}_{duration:.2f}_{time:.2f}"
        params_str += f"_{sigma_param:.2f}_{spatial_weight:.2f}_{n_heads}_{temperature:.2f}"
        params_str += f"_{sigma_g:.2f}_{s_E:.2f}_{s_tau:.2f}_{s_t:.2f}_{temporal_weight:.2f}"  # Include ST-DGPA params
        
        if top_k:
            params_str += f"_top{top_k}"
        if subsample_factor:
            params_str += f"_sub{subsample_factor}"
        
        return hashlib.md5(params_str.encode()).hexdigest()[:16]
    
    @staticmethod
    def clear_3d_cache():
        """Clear 3D interpolation cache"""
        if 'interpolation_3d_cache' in st.session_state:
            st.session_state.interpolation_3d_cache = {}
        if 'interpolation_field_history' in st.session_state:
            st.session_state.interpolation_field_history = OrderedDict()
    
    @staticmethod
    def get_cached_interpolation(field_name, timestep_idx, params):
        """Get cached interpolation result"""
        if 'interpolation_3d_cache' not in st.session_state:
            st.session_state.interpolation_3d_cache = {}
        
        cache_key = CacheManager.generate_cache_key(
            field_name, timestep_idx,
            params.get('energy_query', 0),
            params.get('duration_query', 0),
            params.get('time_points', [0])[timestep_idx] if timestep_idx < len(params.get('time_points', [])) else 0,
            params.get('sigma_param', 0.3),
            params.get('spatial_weight', 0.5),
            params.get('n_heads', 4),
            params.get('temperature', 1.0),
            params.get('sigma_g', 0.20),
            params.get('s_E', 10.0),
            params.get('s_tau', 5.0),
            params.get('s_t', 20.0),  # New: time scaling
            params.get('temporal_weight', 0.3),  # New: temporal weight
            params.get('top_k'),
            params.get('subsample_factor')
        )
        
        return st.session_state.interpolation_3d_cache.get(cache_key)
    
    @staticmethod
    def set_cached_interpolation(field_name, timestep_idx, params, interpolated_values):
        """Store interpolation result in cache"""
        if 'interpolation_3d_cache' not in st.session_state:
            st.session_state.interpolation_3d_cache = {}
        
        cache_key = CacheManager.generate_cache_key(
            field_name, timestep_idx,
            params.get('energy_query', 0),
            params.get('duration_query', 0),
            params.get('time_points', [0])[timestep_idx] if timestep_idx < len(params.get('time_points', [])) else 0,
            params.get('sigma_param', 0.3),
            params.get('spatial_weight', 0.5),
            params.get('n_heads', 4),
            params.get('temperature', 1.0),
            params.get('sigma_g', 0.20),
            params.get('s_E', 10.0),
            params.get('s_tau', 5.0),
            params.get('s_t', 20.0),  # New: time scaling
            params.get('temporal_weight', 0.3),  # New: temporal weight
            params.get('top_k'),
            params.get('subsample_factor')
        )
        
        # Store in cache with timestamp for LRU eviction
        st.session_state.interpolation_3d_cache[cache_key] = {
            'interpolated_values': interpolated_values,
            'timestamp': datetime.now().timestamp(),
            'field_name': field_name,
            'timestep_idx': timestep_idx,
            'params': {k: v for k, v in params.items() if k not in ['simulations', 'summaries']}
        }
        
        # Update field history for quick access
        if 'interpolation_field_history' not in st.session_state:
            st.session_state.interpolation_field_history = OrderedDict()
        
        history_key = f"{field_name}_{timestep_idx}"
        st.session_state.interpolation_field_history[history_key] = cache_key
        
        # Limit cache size (keep only last 20 entries)
        if len(st.session_state.interpolation_3d_cache) > 20:
            # Find oldest entry and remove it
            oldest_key = min(
                st.session_state.interpolation_3d_cache.keys(),
                key=lambda k: st.session_state.interpolation_3d_cache[k]['timestamp']
            )
            del st.session_state.interpolation_3d_cache[oldest_key]
        
        # Limit history size
        if len(st.session_state.interpolation_field_history) > 10:
            # Remove oldest from history
            st.session_state.interpolation_field_history.popitem(last=False)

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
    
    @st.cache_data(show_spinner="Loading simulation data...")
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
# SPATIO-TEMPORAL GATED PHYSICS ATTENTION (ST-DGPA)
# =============================================
class SpatioTemporalGatedPhysicsAttentionExtrapolator:
    """Advanced extrapolator with Spatio-Temporal Gated Physics Attention (ST-DGPA)"""
    
    def __init__(self, sigma_param=0.3, spatial_weight=0.5, n_heads=4, temperature=1.0,
                 sigma_g=0.20, s_E=10.0, s_tau=5.0, s_t=20.0, temporal_weight=0.3):
        self.sigma_param = sigma_param
        self.spatial_weight = spatial_weight
        self.n_heads = n_heads
        self.temperature = temperature
        
        # ST-DGPA specific parameters
        self.sigma_g = sigma_g  # Gating kernel width
        self.s_E = s_E          # Energy scaling factor (mJ)
        self.s_tau = s_tau      # Duration scaling factor (ns)
        self.s_t = s_t          # Time scaling factor (ns) - NEW: temporal parameter
        self.temporal_weight = temporal_weight  # Weight for temporal similarity
        
        # Physics parameters for heat transfer characterization
        self.thermal_diffusivity = 1e-5  # m²/s, typical for metals (adjust based on material)
        self.laser_spot_radius = 50e-6   # m, typical laser spot size
        self.characteristic_length = 100e-6  # m, characteristic length for heat diffusion
        
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
                # Compute physics-aware embedding with enhanced temporal features
                emb = self._compute_enhanced_physics_embedding(
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
                
                # Store metadata for spatial and temporal correlations
                metadata.append({
                    'summary_idx': summary_idx,
                    'timestep_idx': timestep_idx,
                    'energy': summary['energy'],
                    'duration': summary['duration'],
                    'time': t,
                    'name': summary['name'],
                    'fourier_number': self._compute_fourier_number(t),  # Heat transfer characterization
                    'thermal_penetration': self._compute_thermal_penetration(t)  # Diffusion depth
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
    
    def _compute_fourier_number(self, time_ns):
        """Compute Fourier number (Fo = αt/L²) for heat transfer characterization"""
        time_s = time_ns * 1e-9  # Convert ns to s
        Fo = self.thermal_diffusivity * time_s / (self.characteristic_length ** 2)
        return Fo
    
    def _compute_thermal_penetration(self, time_ns):
        """Compute thermal penetration depth (δ ~ √(αt))"""
        time_s = time_ns * 1e-9  # Convert ns to s
        penetration = np.sqrt(self.thermal_diffusivity * time_s) * 1e6  # Convert to μm
        return penetration
    
    def _compute_enhanced_physics_embedding(self, energy, duration, time):
        """Compute comprehensive physics-aware embedding with temporal features"""
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
        
        # Heat transfer specific features
        fourier_number = self._compute_fourier_number(time)
        thermal_penetration_depth = self._compute_thermal_penetration(time)
        diffusion_time_scale = time / (duration + 1e-6)
        
        # Temporal phase indicators
        heating_phase = 1.0 if time < duration else 0.0
        cooling_phase = 1.0 if time >= duration else 0.0
        early_time = 1.0 if time < duration * 0.5 else 0.0
        late_time = 1.0 if time > duration * 2.0 else 0.0
        
        # Combined features with enhanced temporal information
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
            fourier_number,
            thermal_penetration_depth,
            diffusion_time_scale,
            heating_phase,
            cooling_phase,
            early_time,
            late_time,
            np.log1p(power),
            np.log1p(time),
            np.sqrt(time),  # Square root for diffusion scaling
            time / (duration + 1e-6),  # Normalized time
        ], dtype=np.float32)
    
    def _compute_ett_gating(self, energy_query, duration_query, time_query, source_metadata=None):
        """Compute the (E, τ, t) gating kernel for ST-DGPA
        
        Extended from DGPA to include temporal parameter:
        φ_i = sqrt( ((E* - E_i)/s_E)^2 + ((τ* - τ_i)/s_τ)^2 + ((t* - t_i)/s_t)^2 )
        gating_i = exp( - (φ_i^2) / (2 * sigma_g^2) )
        """
        if source_metadata is None:
            source_metadata = self.source_metadata
        
        phi_squared = []
        for meta in source_metadata:
            de = (energy_query - meta['energy']) / self.s_E
            dt = (duration_query - meta['duration']) / self.s_tau
            dtime = (time_query - meta['time']) / self.s_t  # NEW: temporal difference
            
            # Apply physics-aware temporal scaling for heat transfer
            if self.temporal_weight > 0:
                # For heat transfer: early times (heating) need tighter temporal matching
                # than late times (diffusion-dominated)
                time_scaling_factor = 1.0 + 0.5 * (time_query / max(duration_query, 1e-6))  # Looser at later times
                dtime = dtime * time_scaling_factor
            
            phi_squared.append(de**2 + dt**2 + dtime**2)
        
        phi_squared = np.array(phi_squared)
        gating = np.exp(-phi_squared / (2 * self.sigma_g**2))
        
        # Normalize gating weights
        gating_sum = np.sum(gating)
        if gating_sum > 0:
            gating = gating / gating_sum
        else:
            gating = np.ones_like(gating) / len(gating)
        
        return gating
    
    def _compute_temporal_similarity(self, query_meta, source_metas):
        """Compute temporal similarity with physics-aware weighting for heat transfer"""
        similarities = []
        
        for meta in source_metas:
            time_diff = abs(query_meta['time'] - meta['time'])
            
            # Physics-aware temporal similarity:
            # 1. For heating phase (t < τ): tight matching
            # 2. For cooling phase (t > τ): looser matching
            if query_meta['time'] < query_meta['duration'] * 1.5:  # Heating/early cooling
                # Tighter matching during heating phase
                temporal_tolerance = max(query_meta['duration'] * 0.1, 1.0)
            else:
                # Looser matching during late cooling/diffusion phase
                temporal_tolerance = max(query_meta['duration'] * 0.3, 3.0)
            
            # Fourier number similarity for heat transfer characterization
            if 'fourier_number' in meta and 'fourier_number' in query_meta:
                fourier_diff = abs(query_meta['fourier_number'] - meta['fourier_number'])
                fourier_similarity = np.exp(-fourier_diff / 0.1)  # Scale by typical Fo range
            else:
                fourier_similarity = 1.0
            
            # Combine time difference and Fourier number similarity
            time_similarity = np.exp(-time_diff / temporal_tolerance)
            combined_similarity = (1 - self.temporal_weight) * time_similarity + \
                                 self.temporal_weight * fourier_similarity
            
            similarities.append(combined_similarity)
        
        return np.array(similarities)
    
    def _compute_spatial_similarity(self, query_meta, source_metas):
        """Compute spatial similarity based on parameter proximity"""
        similarities = []
        for meta in source_metas:
            # Normalized differences
            e_diff = abs(query_meta['energy'] - meta['energy']) / 50.0
            d_diff = abs(query_meta['duration'] - meta['duration']) / 20.0
            
            # Combined similarity (inverse distance)
            total_diff = np.sqrt(e_diff**2 + d_diff**2)
            similarity = np.exp(-total_diff / self.sigma_param)
            similarities.append(similarity)
        
        return np.array(similarities)
    
    def _multi_head_attention_with_gating(self, query_embedding, query_meta):
        """Multi-head attention mechanism with ST-DGPA (Spatio-Temporal Gated Physics Attention)"""
        if not self.fitted or len(self.source_embeddings) == 0:
            return None, None, None, None
        
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
            
            # Apply temporal regulation if enabled
            if self.temporal_weight > 0:
                temporal_sim = self._compute_temporal_similarity(query_meta, self.source_metadata)
                scores = (1 - self.temporal_weight) * scores + self.temporal_weight * temporal_sim
            
            head_weights[head] = scores
        
        # Combine head weights
        avg_weights = np.mean(head_weights, axis=0)
        
        # Apply temperature scaling
        if self.temperature != 1.0:
            avg_weights = avg_weights ** (1.0 / self.temperature)
        
        # Softmax normalization for physics attention
        max_weight = np.max(avg_weights)
        exp_weights = np.exp(avg_weights - max_weight)
        physics_attention = exp_weights / (np.sum(exp_weights) + 1e-12)
        
        # Apply ST-DGPA: Combine physics attention with (E, τ, t) gating
        ett_gating = self._compute_ett_gating(query_meta['energy'], query_meta['duration'], query_meta['time'])
        
        # ST-DGPA formula: w_i = (α_i * gating_i) / (sum_j α_j * gating_j)
        combined_weights = physics_attention * ett_gating
        combined_sum = np.sum(combined_weights)
        
        if combined_sum > 1e-12:
            final_weights = combined_weights / combined_sum
        else:
            # Fallback to physics attention if combined weights are too small
            final_weights = physics_attention
        
        # Weighted prediction (using original values, not scaled)
        if len(self.source_values) > 0:
            prediction = np.sum(final_weights[:, np.newaxis] * self.source_values, axis=0)
        else:
            prediction = np.zeros(1)
        
        return prediction, final_weights, physics_attention, ett_gating
    
    def predict_field_statistics(self, energy_query, duration_query, time_query):
        """Predict field statistics for given parameters using ST-DGPA"""
        if not self.fitted:
            return None
        
        # Compute query embedding and metadata with enhanced temporal features
        query_embedding = self._compute_enhanced_physics_embedding(energy_query, duration_query, time_query)
        query_meta = {
            'energy': energy_query,
            'duration': duration_query,
            'time': time_query,
            'fourier_number': self._compute_fourier_number(time_query),
            'thermal_penetration': self._compute_thermal_penetration(time_query)
        }
        
        # Apply ST-DGPA attention mechanism
        prediction, final_weights, physics_attention, ett_gating = self._multi_head_attention_with_gating(query_embedding, query_meta)
        
        if prediction is None:
            return None
        
        # Reconstruct field predictions
        result = {
            'prediction': prediction,
            'attention_weights': final_weights,
            'physics_attention': physics_attention,
            'ett_gating': ett_gating,
            'confidence': float(np.max(final_weights)) if len(final_weights) > 0 else 0.0,
            'temporal_confidence': self._compute_temporal_confidence(time_query, duration_query),
            'heat_transfer_indicators': self._compute_heat_transfer_indicators(energy_query, duration_query, time_query),
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
    
    def _compute_temporal_confidence(self, time_query, duration_query):
        """Compute confidence in temporal prediction based on heat transfer physics"""
        # Early times have lower confidence due to rapid changes
        # Late times have higher confidence due to diffusion smoothing
        if time_query < duration_query * 0.5:  # Early heating phase
            return 0.6  # Moderate confidence
        elif time_query < duration_query * 1.5:  # Peak and early cooling
            return 0.8  # Higher confidence
        else:  # Late cooling/diffusion
            return 0.9  # Highest confidence (diffusion-dominated)
    
    def _compute_heat_transfer_indicators(self, energy, duration, time):
        """Compute heat transfer characterization indicators"""
        fourier_number = self._compute_fourier_number(time)
        thermal_penetration = self._compute_thermal_penetration(time)
        
        # Phase identification
        if time < duration * 0.3:
            phase = "Early Heating"
            heat_transfer_regime = "Adiabatic-like"
        elif time < duration:
            phase = "Heating"
            heat_transfer_regime = "Conduction-dominated"
        elif time < duration * 2:
            phase = "Early Cooling"
            heat_transfer_regime = "Mixed conduction"
        else:
            phase = "Diffusion Cooling"
            heat_transfer_regime = "Thermal diffusion"
        
        # Dimensionless indicators
        energy_density = energy / duration
        normalized_time = time / max(duration, 1e-6)
        
        return {
            'phase': phase,
            'regime': heat_transfer_regime,
            'fourier_number': fourier_number,
            'thermal_penetration_um': thermal_penetration,
            'normalized_time': normalized_time,
            'energy_density': energy_density
        }
    
    def predict_time_series(self, energy_query, duration_query, time_points):
        """Predict over a series of time points using ST-DGPA"""
        results = {
            'time_points': time_points,
            'field_predictions': {},
            'attention_maps': [],
            'physics_attention_maps': [],
            'ett_gating_maps': [],
            'confidence_scores': [],
            'temporal_confidences': [],
            'heat_transfer_indicators': []
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
                results['physics_attention_maps'].append(pred['physics_attention'])
                results['ett_gating_maps'].append(pred['ett_gating'])
                results['confidence_scores'].append(pred['confidence'])
                results['temporal_confidences'].append(pred['temporal_confidence'])
                results['heat_transfer_indicators'].append(pred['heat_transfer_indicators'])
            else:
                # Fill with NaN if prediction failed
                for field in results['field_predictions']:
                    results['field_predictions'][field]['mean'].append(np.nan)
                    results['field_predictions'][field]['max'].append(np.nan)
                    results['field_predictions'][field]['std'].append(np.nan)
                results['attention_maps'].append(np.array([]))
                results['physics_attention_maps'].append(np.array([]))
                results['ett_gating_maps'].append(np.array([]))
                results['confidence_scores'].append(0.0)
                results['temporal_confidences'].append(0.0)
                results['heat_transfer_indicators'].append({})
        
        return results
    
    def interpolate_full_field(self, field_name, attention_weights, source_metadata, simulations):
        """Compute interpolated full field using ST-DGPA weights."""
        
        if not self.fitted or len(attention_weights) == 0:
            return None
        
        # Assume common mesh: Get shape from first simulation
        first_sim = next(iter(simulations.values()))
        if 'fields' not in first_sim or field_name not in first_sim['fields']:
            return None
        
        field_shape = first_sim['fields'][field_name].shape[1:]  # (n_points) or (n_points, components)
        
        # Initialize interpolated field based on field type
        if len(field_shape) == 0:  # Scalar field
            interpolated_field = np.zeros(first_sim['fields'][field_name].shape[1], dtype=np.float32)
        else:  # Vector field
            interpolated_field = np.zeros(field_shape, dtype=np.float32)
        
        total_weight = 0.0
        n_sources_used = 0
        
        for idx, weight in enumerate(attention_weights):
            if weight < 1e-6:  # Skip negligible weights for efficiency
                continue
            
            meta = source_metadata[idx]
            sim_name = meta['name']
            timestep_idx = meta['timestep_idx']
            
            if sim_name in simulations:
                sim = simulations[sim_name]
                if 'fields' in sim and field_name in sim['fields']:
                    source_field = sim['fields'][field_name][timestep_idx]
                    interpolated_field += weight * source_field
                    total_weight += weight
                    n_sources_used += 1
        
        if total_weight > 0:
            interpolated_field /= total_weight
        else:
            return None
        
        # Store metadata about interpolation
        self.last_interpolation_metadata = {
            'field_name': field_name,
            'n_sources_used': n_sources_used,
            'total_weight': total_weight,
            'max_weight': np.max(attention_weights) if len(attention_weights) > 0 else 0,
            'min_weight': np.min(attention_weights) if len(attention_weights) > 0 else 0,
            'temporal_coherence': self._assess_temporal_coherence(source_metadata, attention_weights)
        }
        
        return interpolated_field
    
    def _assess_temporal_coherence(self, source_metadata, attention_weights):
        """Assess temporal coherence of the interpolation sources"""
        if len(source_metadata) == 0 or len(attention_weights) == 0:
            return 0.0
        
        times = np.array([meta['time'] for meta in source_metadata])
        weighted_times = times * attention_weights
        mean_time = np.sum(weighted_times) / np.sum(attention_weights)
        
        # Compute temporal spread (weighted standard deviation)
        time_diff = times - mean_time
        weighted_variance = np.sum(attention_weights * time_diff**2) / np.sum(attention_weights)
        temporal_spread = np.sqrt(weighted_variance)
        
        # Normalize by typical time scale
        avg_duration = np.mean([meta['duration'] for meta in source_metadata])
        normalized_spread = temporal_spread / max(avg_duration, 1e-6)
        
        # Convert to coherence score (higher is better)
        coherence = np.exp(-normalized_spread)
        
        return float(coherence)
    
    def export_interpolated_vtu(self, field_name, interpolated_values, simulations, output_path):
        """Export interpolated field to VTU file"""
        if interpolated_values is None or len(simulations) == 0:
            return False
        
        try:
            # Get mesh from first simulation
            first_sim = next(iter(simulations.values()))
            points = first_sim['points']
            cells = None
            
            # Get triangles if available
            if 'triangles' in first_sim and first_sim['triangles'] is not None:
                cells = [("triangle", first_sim['triangles'])]
            
            # Create meshio mesh
            if interpolated_values.ndim == 1:
                # Scalar field
                point_data = {field_name: interpolated_values}
            else:
                # Vector field
                point_data = {field_name: interpolated_values}
            
            mesh = meshio.Mesh(points, cells, point_data=point_data)
            mesh.write(output_path)
            return True
        except Exception as e:
            st.error(f"Error exporting VTU: {str(e)}")
            return False

# =============================================
# ADVANCED VISUALIZATION COMPONENTS WITH ST-DGPA ANALYSIS
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
    def create_stdgpa_analysis(results, energy_query, duration_query, time_points):
        """Create ST-DGPA-specific analysis visualizations"""
        if not results or 'attention_maps' not in results or len(results['attention_maps']) == 0:
            return None
        
        # Select a timestep for detailed analysis (middle point)
        timestep_idx = len(time_points) // 2
        time = time_points[timestep_idx]
        
        # FIXED: Updated subplot specs to handle all plot types correctly
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                "ST-DGPA Final Weights", "Physics Attention Only", 
                "(E, τ, t) Gating Only", "ST-DGPA vs Physics Attention",
                "Temporal Coherence Analysis", "Heat Transfer Phase",
                "Parameter Space 3D", "Attention Network", "Weight Evolution"
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.12,
            specs=[
                [{'type': 'xy'}, {'type': 'xy'}, {'type': 'xy'}],
                [{'type': 'xy'}, {'type': 'xy'}, {'type': 'polar'}],  # Changed 'domain' to 'polar' for radar chart
                [{'type': 'scene'}, {'type': 'xy'}, {'type': 'xy'}]   # Changed 'scatter3d' to 'scene' for 3D plot
            ]
        )
        
        # Get weights for selected timestep
        final_weights = results['attention_maps'][timestep_idx]
        physics_attention = results['physics_attention_maps'][timestep_idx]
        ett_gating = results['ett_gating_maps'][timestep_idx]
        
        # 1. Final ST-DGPA weights
        fig.add_trace(
            go.Bar(
                x=list(range(len(final_weights))),
                y=final_weights,
                name='ST-DGPA Weights',
                marker_color='blue',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. Physics attention only
        fig.add_trace(
            go.Bar(
                x=list(range(len(physics_attention))),
                y=physics_attention,
                name='Physics Attention',
                marker_color='green',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. (E, τ, t) gating only
        fig.add_trace(
            go.Bar(
                x=list(range(len(ett_gating))),
                y=ett_gating,
                name='(E, τ, t) Gating',
                marker_color='red',
                showlegend=False
            ),
            row=1, col=3
        )
        
        # 4. Comparison: ST-DGPA vs Physics Attention
        fig.add_trace(
            go.Scatter(
                x=list(range(len(final_weights))),
                y=final_weights,
                mode='lines+markers',
                name='ST-DGPA Weights',
                line=dict(color='blue', width=3)
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(len(physics_attention))),
                y=physics_attention,
                mode='lines+markers',
                name='Physics Attention',
                line=dict(color='green', width=2, dash='dash')
            ),
            row=2, col=1
        )
        
        # 5. Temporal coherence analysis
        if st.session_state.get('summaries'):
            # Extract time differences
            times = []
            weights = []
            for i, weight in enumerate(final_weights):
                if weight > 0.01:  # Only significant weights
                    # Get time from metadata
                    if hasattr(st.session_state.extrapolator, 'source_metadata'):
                        meta = st.session_state.extrapolator.source_metadata[i]
                        times.append(meta['time'])
                        weights.append(weight)
            
            if times and weights:
                fig.add_trace(
                    go.Scatter(
                        x=times,
                        y=weights,
                        mode='markers',
                        marker=dict(
                            size=np.array(weights) * 50,
                            color=weights,
                            colorscale='Viridis',
                            showscale=False
                        ),
                        name='Weight vs Time',
                        showlegend=False
                    ),
                    row=2, col=2
                )
                fig.add_vline(x=time, line_dash="dash", line_color="red", row=2, col=2)
        
        # 6. Heat transfer phase indicator - FIXED: Now using polar subplot correctly
        if 'heat_transfer_indicators' in results and results['heat_transfer_indicators']:
            indicators = results['heat_transfer_indicators'][timestep_idx]
            if indicators:
                # Create radar/polar plot for phase indicators
                categories = ['Heating', 'Cooling', 'Diffusion', 'Adiabatic']
                
                # Use placeholder values or actual values if available
                if 'phase' in indicators:
                    phase = indicators['phase']
                    # Map phase to values
                    if phase == 'Early Heating' or phase == 'Heating':
                        values = [0.9, 0.3, 0.2, 0.1]
                    elif phase == 'Early Cooling':
                        values = [0.4, 0.8, 0.3, 0.1]
                    elif phase == 'Diffusion Cooling':
                        values = [0.2, 0.5, 0.9, 0.2]
                    else:
                        values = [0.7, 0.5, 0.3, 0.2]  # Default
                else:
                    values = [0.7, 0.5, 0.3, 0.2]  # Placeholder values
                
                # Close the loop for radar chart
                values_closed = list(values) + [values[0]]
                categories_closed = list(categories) + [categories[0]]
                
                fig.add_trace(
                    go.Scatterpolar(
                        r=values_closed,
                        theta=categories_closed,
                        fill='toself',
                        name='Heat Transfer',
                        line=dict(color='orange', width=2),
                        fillcolor='rgba(255, 165, 0, 0.5)',
                        showlegend=False
                    ),
                    row=2, col=3
                )
        
        # 7. Parameter space 3D visualization
        if st.session_state.get('summaries'):
            energies = []
            durations = []
            times = []
            weights = []
            
            for i, summary in enumerate(st.session_state.summaries[:10]):  # First 10 sources
                for t_idx, t in enumerate(summary['timesteps'][:5]):  # First 5 timesteps
                    energies.append(summary['energy'])
                    durations.append(summary['duration'])
                    times.append(t)
                    # Use average weight for visualization
                    weights.append(np.mean(final_weights) if i < len(final_weights) else 0.1)
            
            fig.add_trace(
                go.Scatter3d(
                    x=energies,
                    y=durations,
                    z=times,
                    mode='markers',
                    marker=dict(
                        size=np.array(weights) * 20,
                        color=weights,
                        colorscale='Viridis',
                        opacity=0.7,
                        colorbar=dict(title="Weight", x=1.05)
                    ),
                    name='Sources (E, τ, t)',
                    showlegend=False
                ),
                row=3, col=1
            )
            
            # Add query point
            fig.add_trace(
                go.Scatter3d(
                    x=[energy_query],
                    y=[duration_query],
                    z=[time],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color='red',
                        symbol='diamond'
                    ),
                    name='Query Point',
                    showlegend=False
                ),
                row=3, col=1
            )
        
        # 8. Attention network (simplified)
        # Create a simple network visualization
        if len(final_weights) > 5:
            top_indices = np.argsort(final_weights)[-5:]
            top_weights = final_weights[top_indices]
            
            node_x = [0] + list(range(1, 6))
            node_y = [0] + [0] * 5
            node_text = ['Query'] + [f'Source {i+1}' for i in top_indices]
            
            fig.add_trace(
                go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode='markers+text',
                    text=node_text,
                    textposition="top center",
                    marker=dict(
                        size=[30] + list(top_weights * 50),
                        color=['red'] + ['blue'] * 5
                    ),
                    name='Attention Network',
                    showlegend=False
                ),
                row=3, col=2
            )
            
            # Add edges
            for i in range(1, 6):
                fig.add_trace(
                    go.Scatter(
                        x=[0, i],
                        y=[0, 0],
                        mode='lines',
                        line=dict(width=top_weights[i-1] * 10, color='gray'),
                        showlegend=False
                    ),
                    row=3, col=2
                )
        
        # 9. Weight evolution over time (if multiple timesteps)
        if len(results['attention_maps']) > 1:
            # Show how top source weights evolve
            if len(final_weights) > 0:
                top_idx = np.argmax(final_weights)
                weight_evolution = []
                for t_idx in range(len(results['attention_maps'])):
                    if top_idx < len(results['attention_maps'][t_idx]):
                        weight_evolution.append(results['attention_maps'][t_idx][top_idx])
                
                if weight_evolution:
                    fig.add_trace(
                        go.Scatter(
                            x=time_points[:len(weight_evolution)],
                            y=weight_evolution,
                            mode='lines+markers',
                            line=dict(color='purple', width=3),
                            name='Top Source Weight',
                            showlegend=False
                        ),
                        row=3, col=3
                    )
        
        fig.update_layout(
            height=1000,
            title_text=f"ST-DGPA Analysis at t={time} ns (E={energy_query:.1f} mJ, τ={duration_query:.1f} ns)",
            showlegend=True,
            legend=dict(x=1.05, y=1)
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Source Index", row=1, col=1)
        fig.update_yaxes(title_text="Weight", row=1, col=1)
        fig.update_xaxes(title_text="Source Index", row=1, col=2)
        fig.update_yaxes(title_text="Weight", row=1, col=2)
        fig.update_xaxes(title_text="Source Index", row=1, col=3)
        fig.update_yaxes(title_text="Weight", row=1, col=3)
        
        fig.update_xaxes(title_text="Source Index", row=2, col=1)
        fig.update_yaxes(title_text="Weight", row=2, col=1)
        fig.update_xaxes(title_text="Time (ns)", row=2, col=2)
        fig.update_yaxes(title_text="Weight", row=2, col=2)
        
        # Update polar plot layout
        fig.update_polars(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            ),
            angularaxis=dict(
                direction="clockwise"
            ),
            row=2, col=3
        )
        
        # Update 3D scene
        fig.update_scenes(
            xaxis_title="Energy (mJ)",
            yaxis_title="Duration (ns)",
            zaxis_title="Time (ns)",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            row=3, col=1
        )
        
        fig.update_xaxes(title_text="Node", row=3, col=2)
        fig.update_yaxes(title_text="", showticklabels=False, row=3, col=2)
        fig.update_xaxes(title_text="Time (ns)", row=3, col=3)
        fig.update_yaxes(title_text="Weight", row=3, col=3)
        
        return fig
    
    
    @staticmethod
    def create_temporal_analysis(results, time_points):
        """Create temporal-specific analysis visualizations"""
        if not results or 'heat_transfer_indicators' not in results:
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Heat Transfer Phase Evolution",
                "Fourier Number Evolution",
                "Temporal Confidence",
                "Thermal Penetration Depth"
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )
        
        # 1. Heat transfer phase evolution
        phases = []
        for indicators in results['heat_transfer_indicators']:
            if indicators:
                phases.append(indicators.get('phase', 'Unknown'))
        
        # Convert phases to numerical values for plotting
        phase_mapping = {
            'Early Heating': 0,
            'Heating': 1,
            'Early Cooling': 2,
            'Diffusion Cooling': 3
        }
        
        phase_values = [phase_mapping.get(p, 0) for p in phases]
        
        fig.add_trace(
            go.Scatter(
                x=time_points[:len(phase_values)],
                y=phase_values,
                mode='lines+markers',
                line=dict(color='red', width=3),
                name='Phase',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add phase annotations
        for phase_name, phase_val in phase_mapping.items():
            fig.add_hline(y=phase_val, line_dash="dot", line_color="gray", 
                         annotation_text=phase_name, row=1, col=1)
        
        # 2. Fourier number evolution
        fourier_numbers = []
        for indicators in results['heat_transfer_indicators']:
            if indicators:
                fourier_numbers.append(indicators.get('fourier_number', 0))
        
        if fourier_numbers:
            fig.add_trace(
                go.Scatter(
                    x=time_points[:len(fourier_numbers)],
                    y=fourier_numbers,
                    mode='lines+markers',
                    line=dict(color='blue', width=3),
                    name='Fourier Number',
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # 3. Temporal confidence
        if 'temporal_confidences' in results:
            fig.add_trace(
                go.Scatter(
                    x=time_points[:len(results['temporal_confidences'])],
                    y=results['temporal_confidences'],
                    mode='lines+markers',
                    line=dict(color='green', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(0, 255, 0, 0.2)',
                    name='Temporal Confidence',
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # 4. Thermal penetration depth
        penetration_depths = []
        for indicators in results['heat_transfer_indicators']:
            if indicators:
                penetration_depths.append(indicators.get('thermal_penetration_um', 0))
        
        if penetration_depths:
            fig.add_trace(
                go.Scatter(
                    x=time_points[:len(penetration_depths)],
                    y=penetration_depths,
                    mode='lines+markers',
                    line=dict(color='orange', width=3),
                    name='Penetration (μm)',
                    showlegend=False
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=700,
            title_text="Temporal Analysis of Heat Transfer Characteristics",
            showlegend=False
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Time (ns)", row=1, col=1)
        fig.update_yaxes(title_text="Phase", row=1, col=1)
        fig.update_xaxes(title_text="Time (ns)", row=1, col=2)
        fig.update_yaxes(title_text="Fourier Number", row=1, col=2)
        fig.update_xaxes(title_text="Time (ns)", row=2, col=1)
        fig.update_yaxes(title_text="Confidence", row=2, col=1)
        fig.update_xaxes(title_text="Time (ns)", row=2, col=2)
        fig.update_yaxes(title_text="Depth (μm)", row=2, col=2)
        
        return fig
    
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
            title="3D Attention Weight Distribution in (E, τ, t) Space",
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
                      time=sim_meta['time'] if sim_meta else 0,
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
                time = sim_data.get('time', 0)
                weight = sim_data.get('weight', 0)
                
                node_text.append(
                    f"Simulation: {sim_data['label']}<br>"
                    f"Energy: {energy:.1f} mJ<br>"
                    f"Duration: {duration:.1f} ns<br>"
                    f"Time: {time:.1f} ns<br>"
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
        page_title="Enhanced FEA Laser Simulation Platform with ST-DGPA",
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
    .stdgpa-box {
        background: linear-gradient(135deg, #f093fb 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    .heat-transfer-box {
        background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%);
        color: #333;
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
    .interpolation-3d-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .cache-status {
        background: #e8f5e8;
        border-left: 4px solid #4CAF50;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🔬 Enhanced FEA Laser Simulation Platform with ST-DGPA</h1>', 
                unsafe_allow_html=True)
    
    # Initialize session state with enhanced cache management
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = UnifiedFEADataLoader()
        # Use ST-DGPA-enhanced extrapolator
        st.session_state.extrapolator = SpatioTemporalGatedPhysicsAttentionExtrapolator(
            sigma_param=0.3, spatial_weight=0.5, n_heads=4, temperature=1.0,
            sigma_g=0.20, s_E=10.0, s_tau=5.0, s_t=20.0, temporal_weight=0.3
        )
        st.session_state.visualizer = EnhancedVisualizer()
        st.session_state.data_loaded = False
        st.session_state.current_mode = "Data Viewer"
        st.session_state.selected_colormap = "Viridis"
        st.session_state.interpolation_results = None
        st.session_state.interpolation_params = None
        st.session_state.interpolation_3d_cache = {}
        st.session_state.interpolation_field_history = OrderedDict()
        st.session_state.current_3d_field = None
        st.session_state.current_3d_timestep = 0
        st.session_state.last_prediction_id = None
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Navigation")
        app_mode = st.radio(
            "Select Mode",
            ["Data Viewer", "Interpolation/Extrapolation", "Comparative Analysis", "ST-DGPA Analysis", "Heat Transfer Analysis"],
            index=["Data Viewer", "Interpolation/Extrapolation", "Comparative Analysis", "ST-DGPA Analysis", "Heat Transfer Analysis"].index(
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
        
        # Add warning for interpolation mode without full mesh
        if st.session_state.current_mode == "Interpolation/Extrapolation" and not load_full_data:
            st.warning("⚠️ Full mesh loading is required for 3D interpolation visualization. Please enable and reload.")
        
        if st.button("🔄 Load All Simulations", type="primary", use_container_width=True):
            with st.spinner("Loading simulation data..."):
                # Clear caches when reloading data
                CacheManager.clear_3d_cache()
                st.session_state.last_prediction_id = None
                
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
        
        # Cache management controls
        if st.session_state.data_loaded and 'interpolation_3d_cache' in st.session_state:
            st.markdown("---")
            st.markdown("### 🗄️ Cache Management")
            
            with st.expander("Cache Statistics", expanded=False):
                cache_size = len(st.session_state.interpolation_3d_cache)
                history_size = len(st.session_state.interpolation_field_history)
                
                st.metric("Cached Fields", cache_size)
                st.metric("Field History", history_size)
                
                if cache_size > 0:
                    # Show cached fields
                    st.write("**Cached Fields:**")
                    for cache_key, cache_data in list(st.session_state.interpolation_3d_cache.items())[:5]:
                        field_name = cache_data.get('field_name', 'Unknown')
                        timestep = cache_data.get('timestep_idx', 0)
                        st.caption(f"• {field_name} (t={timestep})")
                    
                    if cache_size > 5:
                        st.caption(f"... and {cache_size - 5} more")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Clear Cache", use_container_width=True):
                        CacheManager.clear_3d_cache()
                        st.rerun()
                
                with col2:
                    if st.button("Refresh Stats", use_container_width=True):
                        st.rerun()
        
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
    elif app_mode == "ST-DGPA Analysis":
        render_stdgpa_analysis()
    elif app_mode == "Heat Transfer Analysis":
        render_heat_transfer_analysis()

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
    """Render the enhanced interpolation/extrapolation interface with ST-DGPA"""
    st.markdown('<h2 class="sub-header">🔮 Interpolation/Extrapolation Engine with ST-DGPA</h2>', 
               unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.markdown("""
        <div class="warning-box">
        <h3>⚠️ No Data Loaded</h3>
        <p>Please load simulations first to enable interpolation/extrapolation capabilities.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Check if full mesh is loaded for 3D visualization
    if st.session_state.simulations and not next(iter(st.session_state.simulations.values())).get('has_mesh', False):
        st.markdown("""
        <div class="warning-box">
        <h3>⚠️ Full Mesh Data Required</h3>
        <p>3D interpolation visualization requires full mesh data. Please reload simulations with "Load Full Mesh" enabled in the sidebar.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="stdgpa-box">
    <h3>🧠 Spatio-Temporal Gated Physics Attention (ST-DGPA)</h3>
    <p>This engine uses <strong>Spatio-Temporal Gated Physics Attention (ST-DGPA)</strong> to interpolate and extrapolate simulation results. ST-DGPA extends DGPA to include temporal parameter gating:</p>
    <ol>
    <li><strong>Physics-informed attention</strong>: Multi-head transformer-like attention with enhanced physics-aware embeddings</li>
    <li><strong>Energy-duration-time gating</strong>: Explicit (E, τ, t) proximity kernel that ensures physically meaningful interpolation across time</li>
    <li><strong>Heat transfer characterization</strong>: Incorporates Fourier number and thermal penetration depth for temporal similarity</li>
    </ol>
    <p><strong>ST-DGPA Formula:</strong> w_i = (α_i × gating_i) / (∑_j α_j × gating_j)</p>
    <p>where φ_i = √(((E*-E_i)/s_E)² + ((τ*-τ_i)/s_τ)² + ((t*-t_i)/s_t)²) and gating_i = exp(-φ_i²/(2σ_g²))</p>
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
                'Max Time (ns)': max(s['timesteps']) if s['timesteps'] else 0,
                'Fields': ', '.join(sorted(s['field_stats'].keys())[:3]) + ('...' if len(s['field_stats']) > 3 else ''),
                'Has Full Mesh': 'Yes' if st.session_state.simulations.get(s['name'], {}).get('has_mesh', False) else 'No'
            } for s in st.session_state.summaries])
            
            st.dataframe(
                df_summary.style.format({
                    'Energy (mJ)': '{:.2f}',
                    'Duration (ns)': '{:.2f}',
                    'Max Time (ns)': '{:.0f}'
                }).background_gradient(subset=['Energy (mJ)', 'Duration (ns)', 'Max Time (ns)'], cmap='Blues'),
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
            max_value=200,
            value=50,
            step=1,
            key="interp_maxtime",
            help="Maximum time for prediction (ns)"
        )
    
    with col4:
        time_resolution = st.selectbox(
            "Time Resolution",
            ["1 ns", "2 ns", "5 ns", "10 ns"],
            index=1,
            key="interp_resolution"
        )
    
    # Generate time points
    time_step_map = {"1 ns": 1, "2 ns": 2, "5 ns": 5, "10 ns": 10}
    time_step = time_step_map[time_resolution]
    time_points = np.arange(1, max_time + 1, time_step)
    
    # Model parameters with ST-DGPA-specific parameters
    with st.expander("⚙️ ST-DGPA Attention Parameters", expanded=False):
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
        
        # ST-DGPA-specific parameters
        st.markdown("### 🎯 ST-DGPA Gating Parameters")
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            sigma_g = st.slider(
                "Gating Kernel Width (σ_g)",
                min_value=0.05,
                max_value=1.0,
                value=0.20,
                step=0.05,
                key="interp_sigma_g",
                help="Sharpness of the (E, τ, t) gating kernel"
            )
        with col6:
            s_E = st.slider(
                "Energy Scale (s_E) [mJ]",
                min_value=0.1,
                max_value=50.0,
                value=10.0,
                step=0.5,
                key="interp_s_E",
                help="Scaling factor for energy in gating kernel"
            )
        with col7:
            s_tau = st.slider(
                "Duration Scale (s_τ) [ns]",
                min_value=0.1,
                max_value=20.0,
                value=5.0,
                step=0.5,
                key="interp_s_tau",
                help="Scaling factor for pulse duration in gating kernel"
            )
        with col8:
            s_t = st.slider(
                "Time Scale (s_t) [ns]",
                min_value=1.0,
                max_value=50.0,
                value=20.0,
                step=1.0,
                key="interp_s_t",
                help="Scaling factor for time in gating kernel"
            )
        
        # Temporal weight parameter
        st.markdown("### ⏱️ Temporal Weighting")
        temporal_weight = st.slider(
            "Temporal Similarity Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            key="interp_temporal_weight",
            help="Weight for temporal similarity in attention calculation"
        )
        
        # Update extrapolator parameters
        st.session_state.extrapolator.sigma_param = sigma_param
        st.session_state.extrapolator.spatial_weight = spatial_weight
        st.session_state.extrapolator.n_heads = n_heads
        st.session_state.extrapolator.temperature = temperature
        st.session_state.extrapolator.sigma_g = sigma_g
        st.session_state.extrapolator.s_E = s_E
        st.session_state.extrapolator.s_tau = s_tau
        st.session_state.extrapolator.s_t = s_t
        st.session_state.extrapolator.temporal_weight = temporal_weight
    
    # Heat transfer physics parameters
    with st.expander("🔥 Heat Transfer Physics Parameters", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            thermal_diffusivity = st.number_input(
                "Thermal Diffusivity (m²/s)",
                min_value=1e-7,
                max_value=1e-4,
                value=1e-5,
                format="%.1e",
                key="thermal_diffusivity",
                help="Material thermal diffusivity (typical metals: ~1e-5 m²/s)"
            )
            st.session_state.extrapolator.thermal_diffusivity = thermal_diffusivity
        
        with col2:
            spot_radius = st.number_input(
                "Laser Spot Radius (μm)",
                min_value=1.0,
                max_value=200.0,
                value=50.0,
                key="spot_radius",
                help="Laser spot radius for heat transfer calculations"
            )
            st.session_state.extrapolator.laser_spot_radius = spot_radius * 1e-6
        
        with col3:
            char_length = st.number_input(
                "Characteristic Length (μm)",
                min_value=10.0,
                max_value=500.0,
                value=100.0,
                key="char_length",
                help="Characteristic length for heat diffusion"
            )
            st.session_state.extrapolator.characteristic_length = char_length * 1e-6
        
        # Display derived heat transfer parameters
        fourier_max = st.session_state.extrapolator._compute_fourier_number(max_time)
        penetration_max = st.session_state.extrapolator._compute_thermal_penetration(max_time)
        
        st.info(f"**Derived Parameters:** Max Fourier Number: {fourier_max:.3f}, Max Thermal Penetration: {penetration_max:.1f} μm")
    
    # 3D interpolation specific settings
    with st.expander("🖼️ 3D Interpolation Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            enable_3d = st.checkbox(
                "Enable 3D Field Interpolation",
                value=True,
                help="Enable full 3D field interpolation and visualization"
            )
            optimize_performance = st.checkbox(
                "Optimize Performance",
                value=True,
                help="Use top-K attention weights and subsampling for large meshes"
            )
        
        with col2:
            if optimize_performance:
                top_k = st.slider(
                    "Top-K Sources",
                    min_value=3,
                    max_value=20,
                    value=10,
                    help="Use only top-K sources by attention weight"
                )
                subsample_factor = st.slider(
                    "Mesh Subsampling",
                    min_value=1,
                    max_value=10,
                    value=1,
                    help="Factor for mesh subsampling (1=full resolution)"
                )
    
    # Run prediction
    if st.button("🚀 Run ST-DGPA Prediction", type="primary", use_container_width=True):
        with st.spinner("Running Spatio-Temporal Gated Physics Attention (ST-DGPA) prediction..."):
            # Clear cache for new prediction
            CacheManager.clear_3d_cache()
            
            results = st.session_state.extrapolator.predict_time_series(
                energy_query, duration_query, time_points
            )
            
            if results and 'field_predictions' in results and results['field_predictions']:
                st.session_state.interpolation_results = results
                st.session_state.interpolation_params = {
                    'energy_query': energy_query,
                    'duration_query': duration_query,
                    'time_points': time_points,
                    'sigma_param': sigma_param,
                    'spatial_weight': spatial_weight,
                    'n_heads': n_heads,
                    'temperature': temperature,
                    'sigma_g': sigma_g,
                    's_E': s_E,
                    's_tau': s_tau,
                    's_t': s_t,
                    'temporal_weight': temporal_weight,
                    'thermal_diffusivity': thermal_diffusivity,
                    'top_k': top_k if 'top_k' in locals() and optimize_performance else None,
                    'subsample_factor': subsample_factor if 'subsample_factor' in locals() and optimize_performance else None
                }
                
                # Generate prediction ID for cache tracking
                prediction_id = hashlib.md5(
                    f"{energy_query}_{duration_query}_{sigma_param}_{sigma_g}_{s_t}".encode()
                ).hexdigest()[:8]
                st.session_state.last_prediction_id = prediction_id
                
                st.markdown("""
                <div class="success-box">
                <h3>✅ ST-DGPA Prediction Successful</h3>
                <p>Spatio-Temporal Gated Physics Attention predictions generated with explicit energy-duration-time gating.</p>
                <p><strong>Heat transfer characterized:</strong> Fourier numbers and thermal penetration depths computed.</p>
                <p><strong>Cache initialized:</strong> 3D field interpolations will be cached for faster switching.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Visualization tabs - Updated with ST-DGPA tabs
                tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                    "📈 Predictions", "🧠 ST-DGPA Analysis", "⏱️ Temporal Analysis", 
                    "🌐 3D Analysis", "📊 Details", "🖼️ 3D Rendering", "⚙️ Parameters"
                ])
                
                with tab1:
                    render_prediction_results(results, time_points, energy_query, duration_query)
                
                with tab2:
                    render_stdgpa_attention_visualization(results, energy_query, duration_query, time_points)
                
                with tab3:
                    render_temporal_analysis(results, time_points, energy_query, duration_query)
                
                with tab4:
                    render_3d_analysis(results, time_points, energy_query, duration_query)
                
                with tab5:
                    render_detailed_results(results, time_points, energy_query, duration_query)
                
                with tab6:
                    render_3d_interpolation(results, time_points, energy_query, duration_query, 
                                           enable_3d, optimize_performance, top_k if optimize_performance else None)
                
                with tab7:
                    render_parameter_analysis(results, energy_query, duration_query, time_points)
            else:
                st.error("Prediction failed. Please check input parameters and ensure sufficient training data.")
    
    # If results already exist from previous run, show them
    elif st.session_state.interpolation_results is not None:
        st.markdown(f"""
        <div class="info-box">
        <h3>📊 Previous ST-DGPA Prediction Results Available</h3>
        <p>Showing results from previous ST-DGPA prediction run with temporal gating.</p>
        <p><strong>Prediction ID:</strong> {st.session_state.last_prediction_id or 'N/A'}</p>
        <p><strong>Cached fields:</strong> {len(st.session_state.interpolation_3d_cache)}</p>
        <p><strong>Temporal gating enabled:</strong> Yes (s_t = {st.session_state.extrapolator.s_t:.1f} ns)</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Get params from session state
        params = st.session_state.interpolation_params or {}
        energy_query = params.get('energy_query', 0)
        duration_query = params.get('duration_query', 0)
        time_points = params.get('time_points', [])
        results = st.session_state.interpolation_results
        
        # Visualization tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📈 Predictions", "🧠 ST-DGPA Analysis", "⏱️ Temporal Analysis", 
            "🌐 3D Analysis", "📊 Details", "🖼️ 3D Rendering", "⚙️ Parameters"
        ])
        
        with tab1:
            render_prediction_results(results, time_points, energy_query, duration_query)
        
        with tab2:
            render_stdgpa_attention_visualization(results, energy_query, duration_query, time_points)
        
        with tab3:
            render_temporal_analysis(results, time_points, energy_query, duration_query)
        
        with tab4:
            render_3d_analysis(results, time_points, energy_query, duration_query)
        
        with tab5:
            render_detailed_results(results, time_points, energy_query, duration_query)
        
        with tab6:
            # Get optimization parameters
            enable_3d = True
            optimize_performance = params.get('top_k') is not None
            top_k = params.get('top_k')
            render_3d_interpolation(results, time_points, energy_query, duration_query, 
                                   enable_3d, optimize_performance, top_k)
        
        with tab7:
            render_parameter_analysis(results, energy_query, duration_query, time_points)

def render_stdgpa_attention_visualization(results, energy_query, duration_query, time_points):
    """Render ST-DGPA-specific attention visualizations"""
    if not results.get('physics_attention_maps') or len(results['physics_attention_maps'][0]) == 0:
        st.info("No ST-DGPA attention data available.")
        return
    
    st.markdown('<h4 class="sub-header">🧠 ST-DGPA Attention Analysis</h4>', unsafe_allow_html=True)
    
    # Select timestep for ST-DGPA visualization
    selected_timestep_idx = st.slider(
        "Select timestep for ST-DGPA analysis",
        0, len(time_points) - 1, len(time_points) // 2,
        key="stdgpa_timestep"
    )
    
    final_weights = results['attention_maps'][selected_timestep_idx]
    physics_attention = results['physics_attention_maps'][selected_timestep_idx]
    ett_gating = results['ett_gating_maps'][selected_timestep_idx]
    selected_time = time_points[selected_timestep_idx]
    
    # Create comprehensive ST-DGPA analysis plot
    fig = st.session_state.visualizer.create_stdgpa_analysis(
        results, energy_query, duration_query, time_points
    )
    
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed ST-DGPA analysis
    st.markdown("##### 📊 ST-DGPA Weight Analysis")
    
    if len(final_weights) > 0:
        # Create comparison dataframe
        comparison_data = []
        for i in range(min(15, len(final_weights))):
            comparison_data.append({
                'Source': f"Source {i+1}",
                'Physics Attention': physics_attention[i],
                '(E, τ, t) Gating': ett_gating[i],
                'ST-DGPA Final Weight': final_weights[i],
                'Weight Change (%)': ((final_weights[i] - physics_attention[i]) / physics_attention[i] * 100) if physics_attention[i] > 0 else 0
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Display with highlighting
        styled_df = df_comparison.style.format({
            'Physics Attention': '{:.4f}',
            '(E, τ, t) Gating': '{:.4f}',
            'ST-DGPA Final Weight': '{:.4f}',
            'Weight Change (%)': '{:.1f}%'
        })
        
        # Highlight significant changes
        if 'Weight Change (%)' in df_comparison.columns:
            styled_df = styled_df.background_gradient(
                subset=['Weight Change (%)'], 
                cmap='RdYlGn', 
                vmin=-100, 
                vmax=100
            )
        
        st.dataframe(styled_df, use_container_width=True)
        
        # ST-DGPA statistics
        st.markdown("##### 📈 ST-DGPA Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            max_physics = np.max(physics_attention) if len(physics_attention) > 0 else 0
            st.metric("Max Physics Attention", f"{max_physics:.4f}")
        with col2:
            max_gating = np.max(ett_gating) if len(ett_gating) > 0 else 0
            st.metric("Max (E, τ, t) Gating", f"{max_gating:.4f}")
        with col3:
            max_stdgpa = np.max(final_weights) if len(final_weights) > 0 else 0
            st.metric("Max ST-DGPA Weight", f"{max_stdgpa:.4f}")
        with col4:
            weight_change_avg = np.mean(np.abs(np.array(final_weights) - np.array(physics_attention))) if len(final_weights) > 0 else 0
            st.metric("Avg Weight Change", f"{weight_change_avg:.4f}")
        
        # ST-DGPA effect analysis
        st.markdown("##### 🔍 ST-DGPA Effect Analysis")
        
        # Calculate how much ST-DGPA changes the weights
        if len(physics_attention) > 0 and len(final_weights) > 0:
            # Find which sources were boosted/suppressed
            weight_diffs = final_weights - physics_attention
            boosted_indices = np.where(weight_diffs > 0)[0]
            suppressed_indices = np.where(weight_diffs < 0)[0]
            
            if len(boosted_indices) > 0:
                max_boost_idx = boosted_indices[np.argmax(weight_diffs[boosted_indices])]
                max_boost = weight_diffs[max_boost_idx]
                st.info(f"**ST-DGPA boosted {len(boosted_indices)} sources** (max boost: +{max_boost:.3f} for source {max_boost_idx+1})")
            
            if len(suppressed_indices) > 0:
                max_suppress_idx = suppressed_indices[np.argmin(weight_diffs[suppressed_indices])]
                max_suppress = weight_diffs[max_suppress_idx]
                st.info(f"**ST-DGPA suppressed {len(suppressed_indices)} sources** (max suppression: {max_suppress:.3f} for source {max_suppress_idx+1})")
            
            # Show top 5 sources by ST-DGPA weight with temporal information
            top_indices = np.argsort(final_weights)[-5:][::-1]
            st.write("**Top 5 Sources by ST-DGPA Weight:**")
            for rank, idx in enumerate(top_indices):
                # Get temporal information
                if hasattr(st.session_state.extrapolator, 'source_metadata') and idx < len(st.session_state.extrapolator.source_metadata):
                    meta = st.session_state.extrapolator.source_metadata[idx]
                    time_info = f", t={meta['time']} ns"
                else:
                    time_info = ""
                
                st.write(f"{rank+1}. Source {idx+1}{time_info}: Physics={physics_attention[idx]:.4f}, Gating={ett_gating[idx]:.4f}, ST-DGPA={final_weights[idx]:.4f}")

def render_temporal_analysis(results, time_points, energy_query, duration_query):
    """Render temporal-specific analysis"""
    if not results or 'heat_transfer_indicators' not in results:
        st.info("No temporal analysis data available.")
        return
    
    st.markdown('<h4 class="sub-header">⏱️ Temporal Analysis</h4>', unsafe_allow_html=True)
    
    # Create temporal analysis visualization
    fig = st.session_state.visualizer.create_temporal_analysis(results, time_points)
    
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed temporal metrics
    st.markdown("##### 📊 Temporal Metrics")
    
    # Calculate phase durations
    if results['heat_transfer_indicators']:
        phases = []
        for indicators in results['heat_transfer_indicators']:
            if indicators:
                phases.append(indicators.get('phase', 'Unknown'))
        
        # Count phase occurrences
        phase_counts = {}
        for phase in phases:
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        # Display phase distribution
        col1, col2, col3, col4 = st.columns(4)
        
        phase_list = ['Early Heating', 'Heating', 'Early Cooling', 'Diffusion Cooling']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for idx, phase in enumerate(phase_list):
            count = phase_counts.get(phase, 0)
            percentage = (count / len(phases) * 100) if phases else 0
            
            with [col1, col2, col3, col4][idx % 4]:
                st.metric(
                    f"{phase}",
                    f"{percentage:.1f}%",
                    delta=f"{count} timesteps" if count > 0 else None
                )
        
        # Temporal confidence analysis
        if 'temporal_confidences' in results:
            avg_temporal_conf = np.mean(results['temporal_confidences'])
            min_temporal_conf = np.min(results['temporal_confidences'])
            max_temporal_conf = np.max(results['temporal_confidences'])
            
            st.markdown("##### 📈 Temporal Confidence")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Temporal Confidence", f"{avg_temporal_conf:.3f}")
            with col2:
                st.metric("Minimum Temporal Confidence", f"{min_temporal_conf:.3f}")
            with col3:
                st.metric("Maximum Temporal Confidence", f"{max_temporal_conf:.3f}")
            
            # Interpretation
            if avg_temporal_conf < 0.5:
                st.warning("⚠️ **Low Temporal Confidence**: Time interpolation may be unreliable, especially during heating phases.")
            elif avg_temporal_conf < 0.7:
                st.info("ℹ️ **Moderate Temporal Confidence**: Time interpolation is reasonable but may have artifacts during rapid changes.")
            else:
                st.success("✅ **High Temporal Confidence**: Time interpolation is reliable across all phases.")
        
        # Heat transfer regime analysis
        st.markdown("##### 🔥 Heat Transfer Regime Analysis")
        
        # Create a table of heat transfer indicators
        if results['heat_transfer_indicators']:
            # Sample at key timesteps
            sample_indices = [0, len(time_points)//4, len(time_points)//2, 3*len(time_points)//4, -1]
            sample_data = []
            
            for idx in sample_indices:
                if idx < len(results['heat_transfer_indicators']):
                    indicators = results['heat_transfer_indicators'][idx]
                    if indicators:
                        sample_data.append({
                            'Time (ns)': time_points[idx],
                            'Phase': indicators.get('phase', 'Unknown'),
                            'Regime': indicators.get('regime', 'Unknown'),
                            'Fourier Number': f"{indicators.get('fourier_number', 0):.3f}",
                            'Penetration (μm)': f"{indicators.get('thermal_penetration_um', 0):.1f}",
                            'Norm. Time': f"{indicators.get('normalized_time', 0):.2f}"
                        })
            
            if sample_data:
                df_heat_transfer = pd.DataFrame(sample_data)
                st.dataframe(df_heat_transfer, use_container_width=True)

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
        
        # Add temporal confidence if available
        if 'temporal_confidences' in results:
            fig_conf.add_trace(go.Scatter(
                x=time_points,
                y=results['temporal_confidences'],
                mode='lines+markers',
                line=dict(color='green', width=3, dash='dash'),
                name='Temporal Confidence'
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
        
        # Add query point predictions
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
        
        # Time evolution in 3D
        st.markdown("##### ⏱️ Time Evolution in Parameter Space")
        
        if 'field_predictions' in results and 'temperature' in results['field_predictions']:
            # Create animation of temperature evolution
            temp_evolution = results['field_predictions']['temperature']['mean']
            
            fig_evolution = go.Figure()
            
            # Training points (static)
            fig_evolution.add_trace(go.Scatter3d(
                x=train_energies,
                y=train_durations,
                z=[train_max_temps[0]] * len(train_energies),  # Placeholder z
                mode='markers',
                marker=dict(
                    size=6,
                    color='lightblue',
                    opacity=0.3
                ),
                name='Training Data'
            ))
            
            # Query point evolution
            fig_evolution.add_trace(go.Scatter3d(
                x=[energy_query] * len(time_points),
                y=[duration_query] * len(time_points),
                z=temp_evolution,
                mode='lines+markers',
                marker=dict(
                    size=8,
                    color=temp_evolution,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Temperature")
                ),
                line=dict(
                    color='gray',
                    width=2
                ),
                name='Temperature Evolution'
            ))
            
            fig_evolution.update_layout(
                title="Temperature Evolution Over Time",
                scene=dict(
                    xaxis_title="Energy (mJ)",
                    yaxis_title="Duration (ns)",
                    zaxis_title="Temperature"
                ),
                height=500
            )
            
            st.plotly_chart(fig_evolution, use_container_width=True)

def render_detailed_results(results, time_points, energy_query, duration_query):
    """Render detailed prediction results"""
    st.markdown('<h4 class="sub-header">📊 Detailed Prediction Results</h4>', unsafe_allow_html=True)
    
    # Create results table with enhanced temporal information
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
        
        if idx < len(results.get('temporal_confidences', [])):
            row['temporal_confidence'] = results['temporal_confidences'][idx]
        
        # Add heat transfer indicators
        if idx < len(results.get('heat_transfer_indicators', [])):
            indicators = results['heat_transfer_indicators'][idx]
            if indicators:
                row['phase'] = indicators.get('phase', 'Unknown')
                row['fourier_number'] = indicators.get('fourier_number', 0)
                row['penetration_um'] = indicators.get('thermal_penetration_um', 0)
        
        data_rows.append(row)
    
    if data_rows:
        df_results = pd.DataFrame(data_rows)
        
        # Format numeric columns
        format_dict = {}
        for col in df_results.columns:
            if col not in ['Time (ns)', 'phase']:
                if 'mean' in col or 'max' in col or 'std' in col:
                    format_dict[col] = "{:.3f}"
                elif 'confidence' in col:
                    format_dict[col] = "{:.3f}"
                elif 'fourier_number' in col:
                    format_dict[col] = "{:.4f}"
                elif 'penetration_um' in col:
                    format_dict[col] = "{:.1f}"
        
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
        
        # Apply formatting to confidence columns
        confidence_cols = [col for col in df_results.columns if 'confidence' in col]
        for col in confidence_cols:
            styled_df = styled_df.applymap(highlight_confidence, subset=[col])
        
        # Color phases
        phase_colors = {
            'Early Heating': '#FF6B6B',
            'Heating': '#4ECDC4',
            'Early Cooling': '#45B7D1',
            'Diffusion Cooling': '#96CEB4'
        }
        
        def color_phase(val):
            if val in phase_colors:
                return f'background-color: {phase_colors[val]}; color: white'
            return ''
        
        if 'phase' in df_results.columns:
            styled_df = styled_df.applymap(color_phase, subset=['phase'])
        
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
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CSV export
            csv = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"stdgpa_predictions_E{energy_query:.1f}mJ_tau{duration_query:.1f}ns.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # JSON export
            json_str = df_results.to_json(orient='records', indent=2)
            st.download_button(
                label="📥 Download as JSON",
                data=json_str.encode('utf-8'),
                file_name=f"stdgpa_predictions_E{energy_query:.1f}mJ_tau{duration_query:.1f}ns.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col3:
            # HTML export with formatting
            html = styled_df.to_html()
            st.download_button(
                label="📥 Download as HTML",
                data=html.encode('utf-8'),
                file_name=f"stdgpa_predictions_E{energy_query:.1f}mJ_tau{duration_query:.1f}ns.html",
                mime="text/html",
                use_container_width=True
            )

def render_parameter_analysis(results, energy_query, duration_query, time_points):
    """Render ST-DGPA parameter sensitivity analysis"""
    st.markdown('<h4 class="sub-header">⚙️ ST-DGPA Parameter Sensitivity</h4>', unsafe_allow_html=True)
    
    # Get current parameters
    params = st.session_state.interpolation_params or {}
    
    st.markdown("### Current ST-DGPA Parameters")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("σ_g (Gating Width)", f"{params.get('sigma_g', 0.20):.2f}")
    with col2:
        st.metric("s_E (Energy Scale)", f"{params.get('s_E', 10.0):.1f} mJ")
    with col3:
        st.metric("s_τ (Duration Scale)", f"{params.get('s_tau', 5.0):.1f} ns")
    with col4:
        st.metric("s_t (Time Scale)", f"{params.get('s_t', 20.0):.1f} ns")
    
    # Parameter sensitivity explanation
    with st.expander("📖 ST-DGPA Parameter Guide", expanded=True):
        st.markdown("""
        ### ST-DGPA Parameter Effects
        
        **σ_g (Gating Kernel Width):**
        - **Small values (0.05-0.2):** Sharp gating, only very similar (E, τ, t) simulations contribute
        - **Large values (0.3-1.0):** Broad gating, allows more distant simulations to contribute
        - **Recommended:** 0.15-0.25 for laser processing with temporal gating
        
        **s_E (Energy Scaling Factor):**
        - Controls how energy differences are weighted
        - **Small values:** Energy differences are heavily penalized
        - **Large values:** Energy differences are less important
        - **Recommended:** Set to ~2× standard deviation of training energies
        
        **s_τ (Duration Scaling Factor):**
        - Controls how duration differences are weighted
        - **Small values:** Duration differences are heavily penalized
        - **Large values:** Duration differences are less important
        - **Recommended:** Set to ~2× standard deviation of training durations
        
        **s_t (Time Scaling Factor):**
        - **NEW:** Controls how time differences are weighted
        - **Small values (5-15 ns):** Tight temporal matching, good for heating phase
        - **Large values (20-50 ns):** Looser temporal matching, good for diffusion phase
        - **Recommended:** 15-25 ns for typical laser processing
        
        **temporal_weight:**
        - **NEW:** Weight for temporal similarity in attention calculation
        - **0.0-0.3:** Physics attention dominates
        - **0.3-0.7:** Balanced temporal and physics attention
        - **0.7-1.0:** Temporal similarity dominates
        
        ### Parameter Selection Strategy for Heat Transfer
        
        1. **For heating phase interpolation** (t < τ):
           - Use smaller s_t (10-15 ns)
           - Use moderate σ_g (0.2-0.25)
           - Higher temporal_weight (0.4-0.6)
           
        2. **For cooling phase interpolation** (t > τ):
           - Use larger s_t (20-30 ns)
           - Use moderate σ_g (0.2-0.25)
           - Lower temporal_weight (0.2-0.4)
           
        3. **For uncertainty quantification:**
           - Vary s_t and observe confidence changes
           - Larger s_t → higher temporal confidence but potentially less accurate for heating
        """)
    
    # Quick parameter testing
    st.markdown("### 🧪 Quick Parameter Test")
    
    test_col1, test_col2, test_col3, test_col4 = st.columns(4)
    with test_col1:
        test_sigma_g = st.slider(
            "Test σ_g",
            min_value=0.05,
            max_value=1.0,
            value=params.get('sigma_g', 0.20),
            step=0.05,
            key="test_sigma_g"
        )
    with test_col2:
        test_s_E = st.slider(
            "Test s_E",
            min_value=0.1,
            max_value=50.0,
            value=params.get('s_E', 10.0),
            step=0.5,
            key="test_s_E"
        )
    with test_col3:
        test_s_tau = st.slider(
            "Test s_τ",
            min_value=0.1,
            max_value=20.0,
            value=params.get('s_tau', 5.0),
            step=0.5,
            key="test_s_tau"
        )
    with test_col4:
        test_s_t = st.slider(
            "Test s_t",
            min_value=1.0,
            max_value=50.0,
            value=params.get('s_t', 20.0),
            step=1.0,
            key="test_s_t"
        )
    
    test_temporal_weight = st.slider(
        "Test Temporal Weight",
        min_value=0.0,
        max_value=1.0,
        value=params.get('temporal_weight', 0.3),
        step=0.05,
        key="test_temporal_weight"
    )
    
    if st.button("🔄 Test Parameters", key="test_params"):
        # Temporarily update extrapolator parameters
        original_params = {
            'sigma_g': st.session_state.extrapolator.sigma_g,
            's_E': st.session_state.extrapolator.s_E,
            's_tau': st.session_state.extrapolator.s_tau,
            's_t': st.session_state.extrapolator.s_t,
            'temporal_weight': st.session_state.extrapolator.temporal_weight
        }
        
        # Update with test values
        st.session_state.extrapolator.sigma_g = test_sigma_g
        st.session_state.extrapolator.s_E = test_s_E
        st.session_state.extrapolator.s_tau = test_s_tau
        st.session_state.extrapolator.s_t = test_s_t
        st.session_state.extrapolator.temporal_weight = test_temporal_weight
        
        # Run quick prediction for middle timestep
        middle_time = time_points[len(time_points) // 2]
        test_result = st.session_state.extrapolator.predict_field_statistics(
            energy_query, duration_query, middle_time
        )
        
        # Restore original parameters
        st.session_state.extrapolator.sigma_g = original_params['sigma_g']
        st.session_state.extrapolator.s_E = original_params['s_E']
        st.session_state.extrapolator.s_tau = original_params['s_tau']
        st.session_state.extrapolator.s_t = original_params['s_t']
        st.session_state.extrapolator.temporal_weight = original_params['temporal_weight']
        
        if test_result:
            st.info(f"**Test Results at t={middle_time} ns:**")
            st.write(f"- Confidence: {test_result['confidence']:.3f}")
            st.write(f"- Temporal Confidence: {test_result['temporal_confidence']:.3f}")
            st.write(f"- Max ST-DGPA weight: {np.max(test_result['attention_weights']):.4f}")
            st.write(f"- Top 3 sources contribute: {np.sum(np.sort(test_result['attention_weights'])[-3:]):.1%}")
            
            if 'heat_transfer_indicators' in test_result:
                indicators = test_result['heat_transfer_indicators']
                st.write(f"- Phase: {indicators.get('phase', 'Unknown')}")
                st.write(f"- Fourier Number: {indicators.get('fourier_number', 0):.4f}")
                st.write(f"- Thermal Penetration: {indicators.get('thermal_penetration_um', 0):.1f} μm")

@st.cache_data(show_spinner=False)
def compute_interpolated_field_cached(_extrapolator, field_name, attention_weights, 
                                     source_metadata, simulations, params):
    """Cached version of interpolate_full_field"""
    return _extrapolator.interpolate_full_field(
        field_name, attention_weights, source_metadata, simulations
    )

def render_3d_interpolation(results, time_points, energy_query, duration_query, 
                           enable_3d=True, optimize_performance=False, top_k=10):
    """Render 3D interpolation visualization with enhanced caching"""
    st.markdown('<h4 class="sub-header">🖼️ 3D Field Interpolation with ST-DGPA</h4>', unsafe_allow_html=True)
    
    if not st.session_state.get('simulations'):
        st.warning("No full simulations loaded for 3D rendering. Please reload with 'Load Full Mesh' enabled.")
        return
    
    # Check if any simulation has mesh data
    first_sim = next(iter(st.session_state.simulations.values()))
    if not first_sim.get('has_mesh', False):
        st.error("Simulations were loaded without full mesh data. Please reload with 'Load Full Mesh' enabled.")
        return
    
    # Show cache status
    if 'interpolation_3d_cache' in st.session_state and st.session_state.interpolation_3d_cache:
        cache_size = len(st.session_state.interpolation_3d_cache)
        st.markdown(f"""
        <div class="cache-status">
        <strong>Cache Status:</strong> {cache_size} field(s) cached | 
        <strong>Prediction ID:</strong> {st.session_state.last_prediction_id or 'N/A'} |
        <strong>Temporal Gating:</strong> s_t = {st.session_state.extrapolator.s_t:.1f} ns
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="interpolation-3d-container">
    <h5>3D Field Interpolation Settings</h5>
    <p>Visualize interpolated full field using ST-DGPA-weighted averaging of source simulations with temporal gating.</p>
    <p><strong>Field switching is now cached</strong> - previously computed fields load instantly.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get parameters from session state
    params = st.session_state.interpolation_params or {}
    
    # Select field and timestep for 3D with session state persistence
    available_fields = list(results['field_predictions'].keys())
    
    if not available_fields:
        st.warning("No field predictions available for 3D visualization.")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        # Use session state to remember last selected field
        if 'current_3d_field' not in st.session_state or st.session_state.current_3d_field not in available_fields:
            st.session_state.current_3d_field = available_fields[0]
        
        selected_field_3d = st.selectbox(
            "Select Field for 3D Rendering", 
            available_fields, 
            index=available_fields.index(st.session_state.current_3d_field) if st.session_state.current_3d_field in available_fields else 0,
            key="interp_3d_field",
            help="Choose field to visualize in 3D",
            on_change=lambda: setattr(st.session_state, 'current_3d_field', 
                                    st.session_state.interp_3d_field if 'interp_3d_field' in st.session_state else available_fields[0])
        )
        
        # Update session state
        st.session_state.current_3d_field = selected_field_3d
    
    with col2:
        # Use session state to remember last selected timestep
        if 'current_3d_timestep' not in st.session_state:
            st.session_state.current_3d_timestep = min(len(time_points)//2, len(time_points)-1)
        
        timestep_idx_3d = st.slider(
            "Select Timestep for 3D", 
            0, len(time_points) - 1, 
            st.session_state.current_3d_timestep,
            key="interp_3d_timestep",
            help="Select timestep for 3D visualization",
            on_change=lambda: setattr(st.session_state, 'current_3d_timestep', 
                                    st.session_state.interp_3d_timestep if 'interp_3d_timestep' in st.session_state else 0)
        )
        
        # Update session state
        st.session_state.current_3d_timestep = timestep_idx_3d
    
    selected_time_3d = time_points[timestep_idx_3d]
    attention_weights_3d = results['attention_maps'][timestep_idx_3d]
    physics_attention_3d = results['physics_attention_maps'][timestep_idx_3d]
    ett_gating_3d = results['ett_gating_maps'][timestep_idx_3d]
    confidence_score = results['confidence_scores'][timestep_idx_3d]
    temporal_confidence = results['temporal_confidences'][timestep_idx_3d] if 'temporal_confidences' in results else 0.0
    
    # Display confidence metric and cache info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Prediction Confidence", f"{confidence_score:.3f}")
    with col2:
        st.metric("Temporal Confidence", f"{temporal_confidence:.3f}")
    with col3:
        st.metric("Timestep", f"{selected_time_3d} ns")
    with col4:
        cached = CacheManager.get_cached_interpolation(selected_field_3d, timestep_idx_3d, params) is not None
        cache_status = "✅ Cached" if cached else "🔄 Compute"
        st.metric("Cache Status", cache_status)
    
    # Check cache first
    cached_result = CacheManager.get_cached_interpolation(selected_field_3d, timestep_idx_3d, params)
    
    if cached_result:
        # Load from cache
        interpolated_values = cached_result['interpolated_values']
        cache_source = "cache"
        st.success(f"✅ Loaded {selected_field_3d} from cache (previously computed)")
    else:
        # Performance optimization: filter top-K sources
        if optimize_performance and len(attention_weights_3d) > top_k:
            top_indices = np.argsort(attention_weights_3d)[-top_k:]
            filtered_weights = np.zeros_like(attention_weights_3d)
            filtered_weights[top_indices] = attention_weights_3d[top_indices]
            filtered_weights = filtered_weights / np.sum(filtered_weights)
            attention_weights_3d = filtered_weights
            st.info(f"Using top-{top_k} sources for performance optimization")
        
        # Compute interpolated field
        with st.spinner(f"Computing {selected_field_3d} field at t={selected_time_3d} ns..."):
            interpolated_values = st.session_state.extrapolator.interpolate_full_field(
                selected_field_3d,
                attention_weights_3d,
                st.session_state.extrapolator.source_metadata,
                st.session_state.simulations
            )
        
        if interpolated_values is None:
            st.error("Failed to interpolate field. Ensure full meshes are loaded and field exists.")
            return
        
        # Cache the result
        CacheManager.set_cached_interpolation(selected_field_3d, timestep_idx_3d, params, interpolated_values)
        cache_source = "new computation"
        st.info(f"✅ Computed and cached {selected_field_3d}")
    
    # Get common mesh (from first sim)
    first_sim = next(iter(st.session_state.simulations.values()))
    pts = first_sim['points']
    triangles = first_sim.get('triangles')
    
    # Performance optimization: subsample points
    subsample_factor = params.get('subsample_factor', 1)
    if optimize_performance and subsample_factor > 1 and pts.shape[0] > 10000:
        indices = np.arange(0, pts.shape[0], subsample_factor)
        pts = pts[indices]
        interpolated_values = interpolated_values[indices]
        st.info(f"Subsampled mesh from {pts.shape[0]*subsample_factor} to {pts.shape[0]} points for performance")
    
    # Handle scalar/vector fields
    if interpolated_values.ndim == 1:
        values = np.nan_to_num(interpolated_values)  # Replace NaNs
        label = selected_field_3d
        field_type = "scalar"
    else:
        magnitude = np.linalg.norm(interpolated_values, axis=1)
        values = np.nan_to_num(magnitude)
        label = f"{selected_field_3d} (magnitude)"
        field_type = "vector"
    
    # Create 3D visualization
    st.markdown(f"### 📊 {label} at t={selected_time_3d} ns")
    
    # Show ST-DGPA weight analysis
    with st.expander("🔍 ST-DGPA Weight Analysis for This Timestep", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Max Physics Attention", f"{np.max(physics_attention_3d):.4f}")
        with col2:
            st.metric("Max (E, τ, t) Gating", f"{np.max(ett_gating_3d):.4f}")
        with col3:
            st.metric("Max ST-DGPA Weight", f"{np.max(attention_weights_3d):.4f}")
        with col4:
            # Temporal coherence of sources
            if hasattr(st.session_state.extrapolator, '_assess_temporal_coherence'):
                coherence = st.session_state.extrapolator._assess_temporal_coherence(
                    st.session_state.extrapolator.source_metadata, attention_weights_3d
                )
                st.metric("Temporal Coherence", f"{coherence:.3f}")
        
        # Show top contributing sources with temporal information
        if len(attention_weights_3d) > 0:
            top_indices = np.argsort(attention_weights_3d)[-5:][::-1]
            st.write("**Top 5 Contributing Sources:**")
            for i, idx in enumerate(top_indices):
                meta = st.session_state.extrapolator.source_metadata[idx]
                st.write(f"{i+1}. {meta['name']} (E={meta['energy']:.1f} mJ, τ={meta['duration']:.1f} ns, t={meta['time']:.1f} ns): "
                        f"Physics={physics_attention_3d[idx]:.4f}, "
                        f"Gating={ett_gating_3d[idx]:.4f}, "
                        f"ST-DGPA={attention_weights_3d[idx]:.4f}")
    
    # Show field history if available
    if 'interpolation_field_history' in st.session_state and st.session_state.interpolation_field_history:
        recent_fields = list(st.session_state.interpolation_field_history.keys())[-5:]
        if recent_fields:
            st.caption(f"**Recently viewed:** {', '.join([f.split('_')[0] for f in recent_fields])}")
    
    if triangles is not None and len(triangles) > 0 and pts.shape[0] < 50000:
        # Validate triangles after possible subsampling
        if optimize_performance and subsample_factor > 1:
            # For subsampled points, use point cloud
            mesh_data = go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    color=values,
                    colorscale=st.session_state.selected_colormap,
                    opacity=0.8,
                    colorbar=dict(
                        title=dict(text=label, font=dict(size=12)),
                        thickness=20,
                        len=0.6
                    ),
                    showscale=True
                ),
                hovertemplate='<b>Value:</b> %{marker.color:.3f}<br>' +
                             '<b>X:</b> %{x:.3f}<br>' +
                             '<b>Y:</b> %{y:.3f}<br>' +
                             '<b>Z:</b> %{z:.3f}<extra></extra>'
            )
        else:
            # Use mesh with triangles
            valid_triangles = triangles[np.all(triangles < len(pts), axis=1)]
            if len(valid_triangles) > 0:
                mesh_data = go.Mesh3d(
                    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                    i=valid_triangles[:, 0], j=valid_triangles[:, 1], k=valid_triangles[:, 2],
                    intensity=values,
                    colorscale=st.session_state.selected_colormap,
                    intensitymode='vertex',
                    colorbar=dict(
                        title=dict(text=label, font=dict(size=12)),
                        thickness=20,
                        len=0.6
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
                # Fallback to scatter if triangles are invalid
                mesh_data = go.Scatter3d(
                    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=values,
                        colorscale=st.session_state.selected_colormap,
                        opacity=0.8,
                        colorbar=dict(title=label),
                        showscale=True
                    ),
                    hovertemplate='<b>Value:</b> %{marker.color:.3f}<br>' +
                                 '<b>X:</b> %{x:.3f}<br>' +
                                 '<b>Y:</b> %{y:.3f}<br>' +
                                 '<b>Z:</b> %{z:.3f}<extra></extra>'
                )
    else:
        # Fallback scatter for point cloud or large meshes
        mesh_data = go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode='markers',
            marker=dict(
                size=4,
                color=values,
                colorscale=st.session_state.selected_colormap,
                opacity=0.8,
                colorbar=dict(
                    title=dict(text=label, font=dict(size=12)),
                    thickness=20,
                    len=0.6
                ),
                showscale=True
            ),
            hovertemplate='<b>Value:</b> %{marker.color:.3f}<br>' +
                         '<b>X:</b> %{x:.3f}<br>' +
                         '<b>Y:</b> %{y:.3f}<br>' +
                         '<b>Z:</b> %{z:.3f}<extra></extra>'
        )
    
    fig_3d = go.Figure(data=mesh_data)
    
    # Get heat transfer phase
    heat_transfer_phase = "Unknown"
    if 'heat_transfer_indicators' in results and timestep_idx_3d < len(results['heat_transfer_indicators']):
        indicators = results['heat_transfer_indicators'][timestep_idx_3d]
        if indicators:
            heat_transfer_phase = indicators.get('phase', 'Unknown')
    
    fig_3d.update_layout(
        title=dict(
            text=f"ST-DGPA Interpolated {label} at t={selected_time_3d} ns<br>"
                 f"E={energy_query:.1f} mJ, τ={duration_query:.1f} ns, Phase: {heat_transfer_phase}<br>"
                 f"Confidence: {confidence_score:.3f}, Temporal: {temporal_confidence:.3f}<br>"
                 f"<sub>σ_g={params.get('sigma_g', 0.20):.2f}, s_E={params.get('s_E', 10.0):.1f}, s_τ={params.get('s_tau', 5.0):.1f}, s_t={params.get('s_t', 20.0):.1f}</sub>",
            font=dict(size=14)
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
        margin=dict(l=0, r=0, t=100, b=0)
    )
    
    st.plotly_chart(fig_3d, use_container_width=True)
    
    # Field statistics for interpolated field
    st.markdown("##### 📊 Interpolated Field Statistics")
    
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
    
    # Histogram of interpolated values
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=values,
        nbinsx=50,
        marker_color='skyblue',
        opacity=0.7,
        name='Value Distribution'
    ))
    
    fig_hist.update_layout(
        title=f"Distribution of Interpolated {label} Values",
        xaxis_title=label,
        yaxis_title="Frequency",
        height=400
    )
    
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Export options
    st.markdown("##### 💾 Export Interpolated Field")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export as VTU file
        if st.button("📥 Export as VTU File", use_container_width=True, key="export_vtu_3d"):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.vtu') as tmp:
                success = st.session_state.extrapolator.export_interpolated_vtu(
                    selected_field_3d,
                    interpolated_values,
                    st.session_state.simulations,
                    tmp.name
                )
                
                if success:
                    with open(tmp.name, 'rb') as f:
                        vtu_data = f.read()
                    
                    b64 = base64.b64encode(vtu_data).decode()
                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="stdgpa_interpolated_{selected_field_3d}_t{selected_time_3d}.vtu">Download VTU File</a>'
                    st.markdown(href, unsafe_allow_html=True)
    
    with col2:
        # Export as NPZ file
        if st.button("📥 Export as NPZ File", use_container_width=True, key="export_npz_3d"):
            npz_data = {
                'field_name': selected_field_3d,
                'values': interpolated_values,
                'points': pts,
                'triangles': triangles,
                'metadata': {
                    'energy_mJ': energy_query,
                    'duration_ns': duration_query,
                    'time_ns': selected_time_3d,
                    'confidence': confidence_score,
                    'temporal_confidence': temporal_confidence,
                    'heat_transfer_phase': heat_transfer_phase,
                    'stdgpa_params': {
                        'sigma_g': params.get('sigma_g', 0.20),
                        's_E': params.get('s_E', 10.0),
                        's_tau': params.get('s_tau', 5.0),
                        's_t': params.get('s_t', 20.0),
                        'temporal_weight': params.get('temporal_weight', 0.3)
                    },
                    'cache_source': cache_source
                }
            }
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as tmp:
                np.savez(tmp, **npz_data)
                with open(tmp.name, 'rb') as f:
                    npz_bytes = f.read()
                
                b64 = base64.b64encode(npz_bytes).decode()
                href = f'<a href="data:application/octet-stream;base64,{b64}" download="stdgpa_interpolated_{selected_field_3d}_t{selected_time_3d}.npz">Download NPZ File</a>'
                st.markdown(href, unsafe_allow_html=True)
    
    with col3:
        # Export heat transfer data
        if st.button("📥 Export Heat Transfer Data", use_container_width=True, key="export_heat_3d"):
            # Create comprehensive heat transfer analysis
            if 'heat_transfer_indicators' in results and timestep_idx_3d < len(results['heat_transfer_indicators']):
                indicators = results['heat_transfer_indicators'][timestep_idx_3d]
                
                heat_data = {
                    'time_ns': selected_time_3d,
                    'energy_mJ': energy_query,
                    'duration_ns': duration_query,
                    'field_stats': {
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values))
                    },
                    'heat_transfer_indicators': indicators,
                    'attention_analysis': {
                        'n_sources_used': len([w for w in attention_weights_3d if w > 1e-6]),
                        'max_weight': float(np.max(attention_weights_3d)),
                        'temporal_coherence': float(st.session_state.extrapolator._assess_temporal_coherence(
                            st.session_state.extrapolator.source_metadata, attention_weights_3d
                        ) if hasattr(st.session_state.extrapolator, '_assess_temporal_coherence') else 0.0)
                    }
                }
                
                json_str = json.dumps(heat_data, indent=2)
                st.download_button(
                    label="📥 Download Heat Data",
                    data=json_str.encode('utf-8'),
                    file_name=f"heat_transfer_{selected_field_3d}_t{selected_time_3d}.json",
                    mime="application/json",
                    use_container_width=True
                )
    
    # Quick field switching buttons
    if len(available_fields) > 1:
        st.markdown("##### ⚡ Quick Field Switching")
        
        # Show buttons for other fields
        other_fields = [f for f in available_fields if f != selected_field_3d][:4]
        
        if other_fields:
            cols = st.columns(len(other_fields))
            for idx, field in enumerate(other_fields):
                with cols[idx]:
                    if st.button(f"📊 {field[:10]}...", key=f"quick_switch_{field}", use_container_width=True):
                        # Update session state and rerun
                        st.session_state.current_3d_field = field
                        st.rerun()

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

def render_stdgpa_analysis():
    """Render comprehensive ST-DGPA analysis interface"""
    st.markdown('<h2 class="sub-header">🔬 Spatio-Temporal Gated Physics Attention (ST-DGPA) Analysis</h2>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.markdown("""
        <div class="warning-box">
        <h3>⚠️ No Data Loaded</h3>
        <p>Please load simulations first to enable ST-DGPA analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown("""
    <div class="stdgpa-box">
    <h3>📚 ST-DGPA Theory & Implementation</h3>
    
    **Spatio-Temporal Gated Physics Attention (ST-DGPA)** is an advanced interpolation method for laser FEA simulations that explicitly incorporates energy (E), pulse duration (τ), and time (t) similarity with heat transfer characterization.
    
    ### Core ST-DGPA Formula
    
    For any field **F** (temperature, stress, displacement, etc.):
    
    $$
    \\boxed{\\mathbf{F}(\\boldsymbol{\\theta}^*) = \\sum_{i=1}^{N} w_i(\\boldsymbol{\\theta}^*) \\cdot \\mathbf{F}^{(i)}}
    $$
    
    where $w_i(\\boldsymbol{\\theta}^*)$ are ST-DGPA weights computed as:
    
    $$
    w_i(\\boldsymbol{\\theta}^*) = \\frac{ 
    \\bar{\\alpha}_i(\\boldsymbol{\\theta}^*) \\cdot 
    \\exp\\left( -\\frac{\\phi_i^2}{2\\sigma_g^2} \\right)
    }{
    \\sum_{k=1}^{N} \\bar{\\alpha}_k(\\boldsymbol{\\theta}^*) \\cdot 
    \\exp\\left( -\\frac{\\phi_k^2}{2\\sigma_g^2} \\right)
    }
    $$
    
    with the (E, τ, t) proximity kernel:
    
    $$
    \\phi_i = \\sqrt{ 
    \\left( \\frac{E^* - E_i}{s_E} \\right)^2 + 
    \\left( \\frac{\\tau^* - \\tau_i}{s_\\tau} \\right)^2 + 
    \\left( \\frac{t^* - t_i}{s_t} \\right)^2 
    }
    $$
    
    ### Key Components
    
    1. **Physics Attention** ($\\bar{\\alpha}_i$): Multi-head transformer-inspired attention with enhanced physics-aware embeddings including heat transfer features
    2. **(E, τ, t) Gating**: Gaussian kernel that ensures physically meaningful interpolation across time
    3. **Heat Transfer Characterization**: Incorporates Fourier number ($Fo = \\alpha t / L^2$) and thermal penetration depth ($\\delta \\sim \\sqrt{\\alpha t}$)
    4. **Temporal Weighting**: Explicit control over temporal similarity importance
    </div>
    """, unsafe_allow_html=True)
    
    # ST-DGPA parameter exploration
    st.markdown('<h3 class="sub-header">🔍 ST-DGPA Parameter Explorer</h3>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        explore_sigma_g = st.slider(
            "Explore σ_g",
            min_value=0.05,
            max_value=1.0,
            value=0.20,
            step=0.05,
            key="explore_sigma_g"
        )
    with col2:
        explore_s_E = st.slider(
            "Explore s_E",
            min_value=0.1,
            max_value=50.0,
            value=10.0,
            step=0.5,
            key="explore_s_E"
        )
    with col3:
        explore_s_tau = st.slider(
            "Explore s_τ",
            min_value=0.1,
            max_value=20.0,
            value=5.0,
            step=0.5,
            key="explore_s_tau"
        )
    with col4:
        explore_s_t = st.slider(
            "Explore s_t",
            min_value=1.0,
            max_value=50.0,
            value=20.0,
            step=1.0,
            key="explore_s_t"
        )
    
    explore_temporal_weight = st.slider(
        "Explore Temporal Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        key="explore_temporal_weight"
    )
    
    # Create visualization of ST-DGPA kernel
    st.markdown("##### 📊 (E, τ, t) Gating Kernel Visualization")
    
    # Generate sample data
    if st.session_state.summaries:
        energies = [s['energy'] for s in st.session_state.summaries]
        durations = [s['duration'] for s in st.session_state.summaries]
        
        # Create grid for visualization (E-τ plane)
        e_min, e_max = min(energies), max(energies)
        d_min, d_max = min(durations), max(durations)
        
        e_grid = np.linspace(e_min, e_max, 50)
        d_grid = np.linspace(d_min, d_max, 50)
        E_grid, D_grid = np.meshgrid(e_grid, d_grid)
        
        # Query point (center of grid)
        query_e = (e_min + e_max) / 2
        query_d = (d_min + d_max) / 2
        query_t = 10.0  # Example time
        
        # Compute gating kernel at fixed time
        phi_squared = ((E_grid - query_e) / explore_s_E)**2 + ((D_grid - query_d) / explore_s_tau)**2
        gating = np.exp(-phi_squared / (2 * explore_sigma_g**2))
        
        # Create heatmap
        fig_kernel = go.Figure(data=go.Heatmap(
            z=gating,
            x=e_grid,
            y=d_grid,
            colorscale='Viridis',
            colorbar=dict(title="Gating Weight")
        ))
        
        # Add training points
        fig_kernel.add_trace(go.Scatter(
            x=energies,
            y=durations,
            mode='markers',
            marker=dict(
                size=8,
                color='red',
                symbol='circle',
                line=dict(width=2, color='white')
            ),
            name='Training Simulations'
        ))
        
        # Add query point
        fig_kernel.add_trace(go.Scatter(
            x=[query_e],
            y=[query_d],
            mode='markers',
            marker=dict(
                size=15,
                color='yellow',
                symbol='star'
            ),
            name='Query Point'
        ))
        
        fig_kernel.update_layout(
            title=f"(E, τ) Gating Kernel at t={query_t} ns (σ_g={explore_sigma_g:.2f}, s_E={explore_s_E:.1f}, s_τ={explore_s_tau:.1f}, s_t={explore_s_t:.1f})",
            xaxis_title="Energy (mJ)",
            yaxis_title="Duration (ns)",
            height=500
        )
        
        st.plotly_chart(fig_kernel, use_container_width=True)
        
        # Kernel statistics
        st.markdown("##### 📈 Kernel Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Max Gating", f"{np.max(gating):.3f}")
        with col2:
            st.metric("Min Gating", f"{np.min(gating):.3f}")
        with col3:
            # Effective radius (where gating > 0.5)
            effective_mask = gating > 0.5
            effective_area = np.sum(effective_mask) / gating.size * 100
            st.metric("Effective Area", f"{effective_area:.1f}%")
        with col4:
            # Number of training points in effective area
            effective_points = 0
            for e, d in zip(energies, durations):
                phi = np.sqrt(((e - query_e)/explore_s_E)**2 + ((d - query_d)/explore_s_tau)**2)
                if np.exp(-phi**2/(2*explore_sigma_g**2)) > 0.5:
                    effective_points += 1
            st.metric("Effective Points", f"{effective_points}/{len(energies)}")
    
    # ST-DGPA vs baseline comparison
    st.markdown('<h3 class="sub-header">⚖️ ST-DGPA vs DGPA Comparison</h3>', unsafe_allow_html=True)
    
    # Simulate comparison scenario
    if st.button("🧪 Run ST-DGPA vs DGPA Comparison", use_container_width=True):
        with st.spinner("Running comparison analysis..."):
            
            st.markdown("""
            ### Conceptual Comparison
            
            | Aspect | ST-DGPA (Extended) | DGPA (Original) |
            |--------|-------------------|----------------|
            | **Temporal sensitivity** | Explicit time gating | Implicit via embeddings |
            | **Heat transfer physics** | Fourier number, penetration depth | Basic time features |
            | **Temporal phases** | Heating/cooling phase detection | No phase distinction |
            | **Temporal confidence** | Physics-based confidence scoring | Single confidence score |
            | **Parameter tuning** | Time-specific scaling (s_t) | No time-specific scaling |
            | **Computational cost** | Slightly higher (enhanced features) | Standard DGPA |
            
            ### Key Advantages of ST-DGPA
            
            1. **Temporal interpretability**: Clear separation of temporal phases (heating vs cooling)
            2. **Heat transfer awareness**: Incorporates diffusion physics via Fourier number
            3. **Phase-appropriate gating**: Tighter temporal matching during heating, looser during cooling
            4. **Enhanced embeddings**: Physics-aware temporal features improve attention quality
            5. **Temporal confidence**: Separate confidence metric for time interpolation reliability
            """)
            
            # Create comparison visualization
            fig_compare = go.Figure()
            
            # Simulated weights for comparison
            n_sources = 20
            physics_weights = np.random.dirichlet(np.ones(n_sources))
            et_distances = np.abs(np.random.randn(n_sources))
            time_distances = np.abs(np.random.randn(n_sources)) * 0.5
            
            # DGPA weights (E, τ only)
            sigma_g_dgpa = 0.2
            dgpa_gating = np.exp(-et_distances**2 / (2 * sigma_g_dgpa**2))
            dgpa_weights = (physics_weights * dgpa_gating)
            dgpa_weights = dgpa_weights / np.sum(dgpa_weights)
            
            # ST-DGPA weights (E, τ, t)
            sigma_g_stdgpa = 0.2
            stdgpa_distances = np.sqrt(et_distances**2 + (time_distances/explore_s_t)**2)
            stdgpa_gating = np.exp(-stdgpa_distances**2 / (2 * sigma_g_stdgpa**2))
            stdgpa_weights = (physics_weights * stdgpa_gating)
            stdgpa_weights = stdgpa_weights / np.sum(stdgpa_weights)
            
            fig_compare.add_trace(go.Scatter(
                x=list(range(n_sources)),
                y=physics_weights,
                mode='lines+markers',
                name='Physics Attention Only',
                line=dict(color='green', width=2)
            ))
            
            fig_compare.add_trace(go.Scatter(
                x=list(range(n_sources)),
                y=dgpa_weights,
                mode='lines+markers',
                name='DGPA (E, τ only)',
                line=dict(color='blue', width=2, dash='dash')
            ))
            
            fig_compare.add_trace(go.Scatter(
                x=list(range(n_sources)),
                y=stdgpa_weights,
                mode='lines+markers',
                name='ST-DGPA (E, τ, t)',
                line=dict(color='red', width=3)
            ))
            
            fig_compare.update_layout(
                title="ST-DGPA vs DGPA Weight Distribution",
                xaxis_title="Source Index",
                yaxis_title="Weight",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig_compare, use_container_width=True)
            
            # Calculate differences
            dgpa_change = dgpa_weights - physics_weights
            stdgpa_change = stdgpa_weights - physics_weights
            
            st.info(f"**DGPA changes weights by ±{np.max(np.abs(dgpa_change)):.3f} (avg: {np.mean(np.abs(dgpa_change)):.3f})**")
            st.info(f"**ST-DGPA changes weights by ±{np.max(np.abs(stdgpa_change)):.3f} (avg: {np.mean(np.abs(stdgpa_change)):.3f})**")
    
    # ST-DGPA applications guide
    with st.expander("📖 ST-DGPA Applications & Best Practices", expanded=True):
        st.markdown("""
        ### 🎯 When to Use ST-DGPA
        
        **Highly recommended for:**
        - Laser processing with strong time-dependent effects
        - Heat transfer-dominated simulations
        - Interpolation across different temporal phases
        - Conservative temporal extrapolation needs
        
        **Recommended for:**
        - General laser FEA interpolation with time dependence
        - Multi-phase physical processes
        - Uncertainty quantification in time domain
        
        **Less suitable for:**
        - Steady-state simulations (use DGPA)
        - Very sparse temporal data (< 3 timesteps per simulation)
        - Real-time applications requiring maximum speed
        
        ### ⚙️ Parameter Selection Guide
        
        **Default values (laser FEA with heat transfer):**
        - σ_g = 0.20 (moderate gating)
        - s_E = 10.0 mJ (based on typical energy range)
        - s_τ = 5.0 ns (based on typical duration range)
        - s_t = 20.0 ns (balanced temporal matching)
        - temporal_weight = 0.3 (moderate temporal emphasis)
        
        **Heat transfer-specific tuning:**
        - **For conductive materials** (high α): Use smaller s_t (10-15 ns)
        - **For diffusive regimes** (long times): Use larger s_t (25-40 ns)
        - **For heating phase focus**: Increase temporal_weight (0.4-0.6)
        - **For cooling phase focus**: Decrease temporal_weight (0.2-0.4)
        
        ### 🔬 Advanced Features
        
        **Adaptive temporal scaling:**
        ```python
        # Auto-adjust s_t based on time relative to pulse duration
        if time_query < duration_query:  # Heating phase
            s_t_effective = s_t * 0.7  # Tighter matching
        else:  # Cooling phase
            s_t_effective = s_t * 1.3  # Looser matching
        ```
        
        **Phase-aware gating:**
        - Early heating (t < 0.3τ): Very tight gating (σ_g ≈ 0.15)
        - Heating phase (0.3τ < t < τ): Moderate gating (σ_g ≈ 0.20)
        - Cooling phase (τ < t < 2τ): Standard gating (σ_g ≈ 0.25)
        - Diffusion phase (t > 2τ): Loose gating (σ_g ≈ 0.30)
        
        ### 📊 Validation Metrics for Temporal Interpolation
        
        Monitor:
        1. **Temporal confidence**: Should be > 0.7 for reliable interpolation
        2. **Phase consistency**: Sources should be in similar temporal phases
        3. **Fourier number spread**: Sources should have similar Fo (ΔFo < 0.5)
        4. **Weight temporal coherence**: High weights should cluster in time
        """)
    
    # ST-DGPA code example
    with st.expander("💻 ST-DGPA Implementation Code", expanded=False):
        st.code("""
# ST-DGPA Implementation Snippet
class SpatioTemporalGatedPhysicsAttentionExtrapolator:
    def __init__(self, sigma_g=0.20, s_E=10.0, s_tau=5.0, s_t=20.0, temporal_weight=0.3):
        self.sigma_g = sigma_g
        self.s_E = s_E
        self.s_tau = s_tau
        self.s_t = s_t  # NEW: Time scaling factor
        self.temporal_weight = temporal_weight  # NEW: Temporal weight
    
    def _compute_ett_gating(self, energy_query, duration_query, time_query):
        \"\"\"Compute the (E, τ, t) gating kernel for ST-DGPA\"\"\"
        phi = []
        for meta in self.source_metadata:
            de = (energy_query - meta['energy']) / self.s_E
            dt = (duration_query - meta['duration']) / self.s_tau
            dtime = (time_query - meta['time']) / self.s_t  # NEW: Time difference
            phi.append(np.sqrt(de**2 + dt**2 + dtime**2))
        
        phi = np.array(phi)
        gating = np.exp(-phi**2 / (2 * self.sigma_g**2))
        return gating / (gating.sum() + 1e-12)
    
    def _compute_temporal_similarity(self, query_meta, source_metas):
        \"\"\"Compute temporal similarity with physics-aware weighting\"\"\"
        similarities = []
        for meta in source_metas:
            time_diff = abs(query_meta['time'] - meta['time'])
            
            # Physics-aware temporal similarity
            if query_meta['time'] < query_meta['duration'] * 1.5:
                temporal_tolerance = max(query_meta['duration'] * 0.1, 1.0)
            else:
                temporal_tolerance = max(query_meta['duration'] * 0.3, 3.0)
            
            # Fourier number similarity
            fourier_diff = abs(query_meta['fourier_number'] - meta['fourier_number'])
            fourier_similarity = np.exp(-fourier_diff / 0.1)
            
            # Combine
            time_similarity = np.exp(-time_diff / temporal_tolerance)
            combined = (1 - self.temporal_weight) * time_similarity + 
                      self.temporal_weight * fourier_similarity
            
            similarities.append(combined)
        return np.array(similarities)
    
    def _multi_head_attention_with_gating(self, query_embedding, query_meta):
        \"\"\"ST-DGPA: Combine physics attention with (E, τ, t) gating\"\"\"
        # 1. Compute physics attention
        physics_attention = self._compute_physics_attention(query_embedding, query_meta)
        
        # 2. Apply temporal regulation
        if self.temporal_weight > 0:
            temporal_sim = self._compute_temporal_similarity(query_meta, self.source_metadata)
            physics_attention = (1 - self.temporal_weight) * physics_attention + 
                               self.temporal_weight * temporal_sim
        
        # 3. Compute (E, τ, t) gating
        ett_gating = self._compute_ett_gating(query_meta['energy'], 
                                             query_meta['duration'], 
                                             query_meta['time'])
        
        # 4. ST-DGPA combination
        combined_weights = physics_attention * ett_gating
        final_weights = combined_weights / (combined_weights.sum() + 1e-12)
        
        return final_weights, physics_attention, ett_gating
        """, language="python")

def render_heat_transfer_analysis():
    """Render heat transfer-specific analysis interface"""
    st.markdown('<h2 class="sub-header">🔥 Heat Transfer Analysis</h2>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.markdown("""
        <div class="warning-box">
        <h3>⚠️ No Data Loaded</h3>
        <p>Please load simulations first to enable heat transfer analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown("""
    <div class="heat-transfer-box">
    <h3>🌡️ Heat Transfer Physics in Laser Processing</h3>
    
    This analysis focuses on heat transfer characterization in laser FEA simulations using ST-DGPA.
    
    ### Key Heat Transfer Concepts
    
    1. **Fourier Number** ($Fo = \\frac{\\alpha t}{L^2}$):
       - Dimensionless time characterizing heat diffusion
       - $Fo < 0.1$: Localized heating (conduction-limited)
       - $Fo > 1$: Well-developed diffusion
       
    2. **Thermal Penetration Depth** ($\\delta \\sim \\sqrt{\\alpha t}$):
       - Characteristic distance heat diffuses in time t
       - Important for determining affected volume
       
    3. **Temporal Phases**:
       - **Early Heating** ($t < 0.3\\tau$): Rapid temperature rise, minimal diffusion
       - **Heating** ($0.3\\tau < t < \\tau$): Continued energy deposition
       - **Early Cooling** ($\\tau < t < 2\\tau$): Rapid cooling, significant gradients
       - **Diffusion Cooling** ($t > 2\\tau$): Slow thermal diffusion dominates
       
    4. **Dimensionless Groups**:
       - **Energy Density**: $E/\\tau$ (W)
       - **Normalized Time**: $t/\\tau$
       - **Thermal Diffusion Ratio**: $\\sqrt{\\alpha\\tau}/L$
    </div>
    """, unsafe_allow_html=True)
    
    # Material properties
    st.markdown('<h3 class="sub-header">📊 Material Properties</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        material_type = st.selectbox(
            "Material Type",
            ["Steel", "Aluminum", "Copper", "Titanium", "Custom"],
            index=0,
            key="material_type"
        )
    
    # Set default properties based on material
    material_props = {
        "Steel": {"α": 1.2e-5, "k": 50, "ρ": 7850, "cp": 450},
        "Aluminum": {"α": 8.4e-5, "k": 237, "ρ": 2700, "cp": 900},
        "Copper": {"α": 1.1e-4, "k": 400, "ρ": 8960, "cp": 385},
        "Titanium": {"α": 8.9e-6, "k": 21.9, "ρ": 4500, "cp": 520},
        "Custom": {"α": 1e-5, "k": 50, "ρ": 5000, "cp": 500}
    }
    
    props = material_props[material_type]
    
    with col2:
        thermal_diffusivity = st.number_input(
            "Thermal Diffusivity (m²/s)",
            min_value=1e-7,
            max_value=1e-3,
            value=props["α"],
            format="%.2e",
            key="ht_thermal_diffusivity",
            help="α = k/(ρ·cp)"
        )
    
    with col3:
        laser_radius = st.number_input(
            "Laser Spot Radius (μm)",
            min_value=1.0,
            max_value=500.0,
            value=50.0,
            key="ht_laser_radius"
        )
    
    # Update extrapolator with new properties
    st.session_state.extrapolator.thermal_diffusivity = thermal_diffusivity
    st.session_state.extrapolator.laser_spot_radius = laser_radius * 1e-6
    st.session_state.extrapolator.characteristic_length = laser_radius * 2e-6  # 2× spot radius
    
    # Heat transfer calculations
    st.markdown('<h3 class="sub-header">🧮 Heat Transfer Calculations</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        calc_time = st.number_input(
            "Calculation Time (ns)",
            min_value=1,
            max_value=500,
            value=50,
            key="calc_time"
        )
    
    with col2:
        calc_duration = st.number_input(
            "Pulse Duration (ns)",
            min_value=1,
            max_value=100,
            value=10,
            key="calc_duration"
        )
    
    # Perform calculations
    fourier_number = st.session_state.extrapolator._compute_fourier_number(calc_time)
    penetration_depth = st.session_state.extrapolator._compute_thermal_penetration(calc_time)
    phase = "Unknown"
    
    if calc_time < calc_duration * 0.3:
        phase = "Early Heating"
    elif calc_time < calc_duration:
        phase = "Heating"
    elif calc_time < calc_duration * 2:
        phase = "Early Cooling"
    else:
        phase = "Diffusion Cooling"
    
    # Display results
    st.markdown("##### 📈 Calculation Results")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Fourier Number", f"{fourier_number:.4f}")
    with col2:
        st.metric("Thermal Penetration", f"{penetration_depth:.1f} μm")
    with col3:
        st.metric("Phase", phase)
    with col4:
        normalized_time = calc_time / calc_duration
        st.metric("Normalized Time", f"{normalized_time:.2f}")
    
    # Interpretation
    st.markdown("##### 💡 Interpretation")
    
    interpretation = []
    if fourier_number < 0.1:
        interpretation.append("**Conduction-limited regime**: Heat diffusion is minimal, temperature gradients are steep.")
    elif fourier_number < 1.0:
        interpretation.append("**Transition regime**: Diffusion is developing, gradients are moderating.")
    else:
        interpretation.append("**Diffusion-dominated regime**: Heat has spread significantly, temperature field is smoothing.")
    
    if phase == "Early Heating":
        interpretation.append("**Phase**: Rapid heating with minimal diffusion - high interpolation uncertainty.")
    elif phase == "Heating":
        interpretation.append("**Phase**: Active energy deposition - moderate interpolation uncertainty.")
    elif phase == "Early Cooling":
        interpretation.append("**Phase**: Rapid cooling with developing gradients - good interpolation conditions.")
    else:
        interpretation.append("**Phase**: Slow diffusion cooling - excellent interpolation conditions.")
    
    for item in interpretation:
        st.info(item)
    
    # Heat transfer visualization
    st.markdown('<h3 class="sub-header">📊 Heat Transfer Visualization</h3>', unsafe_allow_html=True)
    
    # Create time evolution plot
    time_range = np.linspace(1, 200, 100)  # 1-200 ns
    fourier_evolution = [st.session_state.extrapolator._compute_fourier_number(t) for t in time_range]
    penetration_evolution = [st.session_state.extrapolator._compute_thermal_penetration(t) for t in time_range]
    
    fig_evolution = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Fourier Number Evolution", "Thermal Penetration Evolution"],
        vertical_spacing=0.15
    )
    
    # Fourier number
    fig_evolution.add_trace(
        go.Scatter(
            x=time_range,
            y=fourier_evolution,
            mode='lines',
            line=dict(color='blue', width=3),
            name='Fourier Number'
        ),
        row=1, col=1
    )
    
    # Add phase regions
    phase_boundaries = [calc_duration * 0.3, calc_duration, calc_duration * 2]
    phase_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    phase_labels = ['Early Heating', 'Heating', 'Early Cooling', 'Diffusion Cooling']
    
    for i in range(len(phase_boundaries) + 1):
        if i == 0:
            x0, x1 = 0, phase_boundaries[0]
        elif i == len(phase_boundaries):
            x0, x1 = phase_boundaries[-1], time_range[-1]
        else:
            x0, x1 = phase_boundaries[i-1], phase_boundaries[i]
        
        fig_evolution.add_vrect(
            x0=x0, x1=x1,
            fillcolor=phase_colors[i],
            opacity=0.2,
            line_width=0,
            row=1, col=1
        )
        
        # Add label
        fig_evolution.add_annotation(
            x=(x0 + x1) / 2,
            y=np.max(fourier_evolution) * 0.9,
            text=phase_labels[i],
            showarrow=False,
            font=dict(size=10),
            row=1, col=1
        )
    
    # Thermal penetration
    fig_evolution.add_trace(
        go.Scatter(
            x=time_range,
            y=penetration_evolution,
            mode='lines',
            line=dict(color='red', width=3),
            name='Penetration Depth'
        ),
        row=2, col=1
    )
    
    fig_evolution.update_layout(
        height=600,
        title_text=f"Heat Transfer Evolution (α = {thermal_diffusivity:.2e} m²/s, Laser Radius = {laser_radius} μm)",
        showlegend=True
    )
    
    fig_evolution.update_xaxes(title_text="Time (ns)", row=1, col=1)
    fig_evolution.update_yaxes(title_text="Fourier Number", row=1, col=1)
    fig_evolution.update_xaxes(title_text="Time (ns)", row=2, col=1)
    fig_evolution.update_yaxes(title_text="Penetration Depth (μm)", row=2, col=1)
    
    st.plotly_chart(fig_evolution, use_container_width=True)
    
    # ST-DGPA for heat transfer
    st.markdown('<h3 class="sub-header">🎯 ST-DGPA for Heat Transfer</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    ### How ST-DGPA Enhances Heat Transfer Interpolation
    
    1. **Phase-Aware Gating**:
       - Tighter temporal matching during heating phases (smaller effective s_t)
       - Looser matching during diffusion phases (larger effective s_t)
       - Prevents mixing incompatible thermal regimes
    
    2. **Physics-Informed Similarity**:
       - Uses Fourier number for dimensionless time comparison
       - Considers thermal penetration depth for spatial scale
       - Incorporates phase information in attention weights
    
    3. **Confidence Metrics**:
       - Temporal confidence based on phase and Fourier number spread
       - Source coherence assessment for temporal consistency
       - Uncertainty quantification for time interpolation
    
    4. **Optimized Parameters**:
       ```python
       # Heat-transfer optimized ST-DGPA parameters
       st_dgpa_heat = SpatioTemporalGatedPhysicsAttentionExtrapolator(
           sigma_g=0.22,          # Slightly broader for thermal diffusion
           s_E=10.0,             # Energy scaling (mJ)
           s_tau=5.0,            # Duration scaling (ns)
           s_t=25.0,             # Time scaling for diffusion (ns)
           temporal_weight=0.4   # Enhanced temporal emphasis
       )
       ```
    
    ### Best Practices for Heat Transfer Applications
    
    1. **Training Data Preparation**:
       - Ensure sufficient temporal resolution (≥ 5 timesteps per simulation)
       - Include simulations covering different phases
       - Balance energy-duration combinations
    
    2. **Query Strategy**:
       - For heating phase queries, prioritize simulations with similar t/τ
       - For cooling phase, include broader temporal range
       - Use Fourier number similarity for dimensionless comparison
    
    3. **Validation**:
       - Check phase consistency of top-weighted sources
       - Verify Fourier number spread is reasonable
       - Validate against known physical limits (energy conservation, max temperature)
    
    4. **Troubleshooting**:
       - **Low temporal confidence**: Increase s_t or check phase mismatch
       - **Over-smoothing**: Decrease s_t or increase temporal_weight
       - **Unphysical results**: Check source coherence and phase consistency
    """)
    
    # Interactive heat transfer exploration
    if st.button("🔬 Explore Heat Transfer with Current Data", use_container_width=True):
        if st.session_state.summaries:
            # Analyze available simulations for heat transfer characteristics
            st.markdown("##### 📋 Simulation Heat Transfer Analysis")
            
            analysis_data = []
            for summary in st.session_state.summaries[:10]:  # First 10 for brevity
                # Calculate characteristic Fourier numbers
                max_time = max(summary['timesteps']) if summary['timesteps'] else 0
                max_fourier = st.session_state.extrapolator._compute_fourier_number(max_time)
                max_penetration = st.session_state.extrapolator._compute_thermal_penetration(max_time)
                
                # Temperature statistics if available
                temp_stats = "N/A"
                if 'temperature' in summary['field_stats']:
                    stats = summary['field_stats']['temperature']
                    if stats['max']:
                        max_temp = np.max(stats['max'])
                        temp_stats = f"{max_temp:.1f}°C"
                
                analysis_data.append({
                    'Simulation': summary['name'],
                    'Energy (mJ)': summary['energy'],
                    'Duration (ns)': summary['duration'],
                    'Max Time (ns)': max_time,
                    'Max Fourier': f"{max_fourier:.3f}",
                    'Max Penetration (μm)': f"{max_penetration:.1f}",
                    'Max Temp': temp_stats,
                    'Phase at Max': "Diffusion" if max_time > summary['duration'] * 2 else "Heating/Cooling"
                })
            
            if analysis_data:
                df_analysis = pd.DataFrame(analysis_data)
                st.dataframe(df_analysis, use_container_width=True)
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_fourier = np.mean([float(d['Max Fourier']) for d in analysis_data])
                    st.metric("Avg Max Fourier", f"{avg_fourier:.3f}")
                with col2:
                    avg_penetration = np.mean([float(d['Max Penetration (μm)']) for d in analysis_data])
                    st.metric("Avg Max Penetration", f"{avg_penetration:.1f} μm")
                with col3:
                    heating_sims = len([d for d in analysis_data if d['Phase at Max'] == "Heating/Cooling"])
                    st.metric("Heating/Cooling Sims", f"{heating_sims}/{len(analysis_data)}")

if __name__ == "__main__":
    main()
