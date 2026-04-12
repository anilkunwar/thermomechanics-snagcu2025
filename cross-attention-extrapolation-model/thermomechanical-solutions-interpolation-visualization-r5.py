#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ENHANCED LASER SOLDERING ST-DGPA PLATFORM WITH POLAR RADAR & SANKEY VISUALIZATION
================================================================================
Complete integrated application for:
- FEA laser soldering simulation loading from VTU files
- ST-DGPA (Spatio-Temporal Gated Physics Attention) interpolation/extrapolation
- Polar Radar Charts: Energy (angular) × Pulse Duration (radial) × Peak T/Stress
- Interactive Sankey Diagrams showing weight decomposition flow
- 3D mesh visualization with caching
- Robust error handling, session state management, and UI controls
- NEW: Export functionality, advanced filtering, responsive design
"""

import streamlit as st
import os
import glob
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import meshio
import warnings
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
from io import BytesIO
import traceback
from scipy.interpolate import griddata, RBFInterpolator
from sklearn.preprocessing import StandardScaler
import tempfile
import base64
import hashlib
from collections import OrderedDict
import json
import time

warnings.filterwarnings('ignore')

# =============================================
# PATH CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEA_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "fea_solutions")
VISUALIZATION_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "visualization_outputs")
TEMP_ANIMATION_DIR = os.path.join(SCRIPT_DIR, "temp_animations")
EXPORT_DIR = os.path.join(SCRIPT_DIR, "exports")

for d in [FEA_SOLUTIONS_DIR, VISUALIZATION_OUTPUT_DIR, TEMP_ANIMATION_DIR, EXPORT_DIR]:
    os.makedirs(d, exist_ok=True)

# =============================================
# 1. CACHE MANAGEMENT UTILITIES
# =============================================
class CacheManager:
    """Manages caching of interpolation results to prevent recomputation"""
    
    @staticmethod
    def generate_cache_key(field_name: str, timestep_idx: int, energy: float, 
                          duration: float, time: float, sigma_param: float,
                          spatial_weight: float, n_heads: int, temperature: float,
                          sigma_g: float, s_E: float, s_tau: float, s_t: float,
                          temporal_weight: float, top_k: Optional[int] = None,
                          subsample_factor: Optional[int] = None) -> str:
        """Generate a unique cache key for interpolation parameters"""
        params_str = f"{field_name}_{timestep_idx}_{energy:.2f}_{duration:.2f}_{time:.2f}"
        params_str += f"_{sigma_param:.2f}_{spatial_weight:.2f}_{n_heads}_{temperature:.2f}"
        params_str += f"_{sigma_g:.2f}_{s_E:.2f}_{s_tau:.2f}_{s_t:.2f}_{temporal_weight:.2f}"
        if top_k: params_str += f"_top{top_k}"
        if subsample_factor: params_str += f"_sub{subsample_factor}"
        return hashlib.md5(params_str.encode()).hexdigest()[:16]
    
    @staticmethod
    def clear_3d_cache():
        """Clear all cached interpolation results"""
        if 'interpolation_3d_cache' in st.session_state:
            st.session_state.interpolation_3d_cache = {}
        if 'interpolation_field_history' in st.session_state:
            st.session_state.interpolation_field_history = OrderedDict()
        st.success("✅ Cache cleared")
    
    @staticmethod
    def get_cached_interpolation(field_name: str, timestep_idx: int, params: Dict) -> Optional[Dict]:
        """Get cached interpolation result if available"""
        if 'interpolation_3d_cache' not in st.session_state:
            st.session_state.interpolation_3d_cache = {}
        
        cache_key = CacheManager.generate_cache_key(
            field_name, timestep_idx,
            params.get('energy_query', 0), params.get('duration_query', 0),
            params.get('time_points', [0])[timestep_idx] if timestep_idx < len(params.get('time_points', [])) else 0,
            params.get('sigma_param', 0.3), params.get('spatial_weight', 0.5),
            params.get('n_heads', 4), params.get('temperature', 1.0),
            params.get('sigma_g', 0.20), params.get('s_E', 10.0),
            params.get('s_tau', 5.0), params.get('s_t', 20.0),
            params.get('temporal_weight', 0.3), params.get('top_k'), params.get('subsample_factor')
        )
        return st.session_state.interpolation_3d_cache.get(cache_key)
    
    @staticmethod
    def set_cached_interpolation(field_name: str, timestep_idx: int, params: Dict, 
                                interpolated_values: np.ndarray):
        """Store interpolation result in cache with LRU eviction"""
        if 'interpolation_3d_cache' not in st.session_state:
            st.session_state.interpolation_3d_cache = {}
        
        cache_key = CacheManager.generate_cache_key(
            field_name, timestep_idx,
            params.get('energy_query', 0), params.get('duration_query', 0),
            params.get('time_points', [0])[timestep_idx] if timestep_idx < len(params.get('time_points', [])) else 0,
            params.get('sigma_param', 0.3), params.get('spatial_weight', 0.5),
            params.get('n_heads', 4), params.get('temperature', 1.0),
            params.get('sigma_g', 0.20), params.get('s_E', 10.0),
            params.get('s_tau', 5.0), params.get('s_t', 20.0),
            params.get('temporal_weight', 0.3), params.get('top_k'), params.get('subsample_factor')
        )
        
        # Store with metadata for cache management
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
        
        # LRU eviction: keep only last 20 entries
        if len(st.session_state.interpolation_3d_cache) > 20:
            oldest_key = min(
                st.session_state.interpolation_3d_cache.keys(),
                key=lambda k: st.session_state.interpolation_3d_cache[k]['timestamp']
            )
            del st.session_state.interpolation_3d_cache[oldest_key]
        
        # Limit history size
        if len(st.session_state.interpolation_field_history) > 10:
            st.session_state.interpolation_field_history.popitem(last=False)

# =============================================
# 2. UNIFIED DATA LOADER
# =============================================
class UnifiedFEADataLoader:
    """Enhanced data loader with comprehensive field extraction and validation"""
    
    def __init__(self):
        self.simulations = {}
        self.summaries = []
        self.available_fields = set()
        self.load_errors = []
    
    def parse_folder_name(self, folder: str) -> Tuple[Optional[float], Optional[float]]:
        """Parse folder name: q0p5mJ-delta4p2ns → (0.5, 4.2)"""
        match = re.match(r"q([\dp\.]+)mJ-delta([\dp\.]+)ns", folder)
        if not match:
            return None, None
        e, d = match.groups()
        return float(e.replace("p", ".")), float(d.replace("p", "."))
    
    @st.cache_data(show_spinner="Loading simulation data...")
    def load_all_simulations(_self, load_full_mesh: bool = True) -> Tuple[Dict, List]:
        """Load all simulations with option for full mesh or summaries only"""
        simulations, summaries = {}, []
        folders = glob.glob(os.path.join(FEA_SOLUTIONS_DIR, "q*mJ-delta*ns"))
        
        if not folders:
            st.warning(f"No simulation folders found in {FEA_SOLUTIONS_DIR}")
            return simulations, summaries
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        _self.load_errors = []
        
        for folder_idx, folder in enumerate(folders):
            name = os.path.basename(folder)
            energy, duration = _self.parse_folder_name(name)
            
            if energy is None:
                _self.load_errors.append(f"Could not parse folder: {name}")
                continue
            
            vtu_files = sorted(glob.glob(os.path.join(folder, "a_t????.vtu")))
            if not vtu_files:
                _self.load_errors.append(f"No VTU files in: {name}")
                continue
            
            status_text.text(f"Loading {name}... ({len(vtu_files)} files)")
            
            try:
                mesh0 = meshio.read(vtu_files[0])
                if not mesh0.point_data:
                    _self.load_errors.append(f"No point data in: {name}")
                    continue
                
                # Create simulation entry
                sim_data = {
                    'name': name,
                    'energy_mJ': energy,
                    'duration_ns': duration,
                    'n_timesteps': len(vtu_files),
                    'vtu_files': vtu_files,
                    'field_info': {},
                    'has_mesh': False,
                    'load_timestamp': datetime.now()
                }
                
                if load_full_mesh:
                    # Full mesh loading for 3D visualization
                    points = mesh0.points.astype(np.float32)
                    n_pts = len(points)
                    
                    # Find triangles for surface rendering
                    triangles = None
                    for cell_block in mesh0.cells:
                        if cell_block.type == "triangle":
                            triangles = cell_block.data.astype(np.int32)
                            break
                    
                    # Initialize fields dictionary
                    fields = {}
                    for key in mesh0.point_data.keys():
                        arr = mesh0.point_data[key].astype(np.float32)
                        
                        # Determine field type: scalar (1D) or vector/tensor (2D+)
                        if arr.ndim == 1:
                            field_type = "scalar"
                            comp = 1
                        else:
                            field_type = "vector" if arr.shape[1] <= 3 else "tensor"
                            comp = arr.shape[1]
                        
                        sim_data['field_info'][key] = (field_type, comp)
                        
                        # Pre-allocate array for all timesteps
                        shape = (len(vtu_files), n_pts) if field_type == "scalar" else (len(vtu_files), n_pts, comp)
                        fields[key] = np.full(shape, np.nan, dtype=np.float32)
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
                
                # Extract summary statistics for quick analysis
                summary = _self.extract_summary_statistics(vtu_files, energy, duration, name)
                summaries.append(summary)
                simulations[name] = sim_data
                
            except Exception as e:
                error_msg = f"Error loading {name}: {str(e)}"
                _self.load_errors.append(error_msg)
                st.warning(error_msg)
                continue
            
            progress_bar.progress((folder_idx + 1) / len(folders))
        
        progress_bar.empty()
        status_text.empty()
        
        # Report results
        if simulations:
            st.success(f"✅ Loaded {len(simulations)} simulations with {len(_self.available_fields)} unique fields")
            if _self.load_errors:
                with st.expander(f"⚠️ {len(_self.load_errors)} loading warnings"):
                    for err in _self.load_errors[:10]:
                        st.caption(err)
                    if len(_self.load_errors) > 10:
                        st.caption(f"... and {len(_self.load_errors) - 10} more")
        else:
            st.error("❌ No simulations loaded successfully")
            if _self.load_errors:
                with st.expander("View errors"):
                    for err in _self.load_errors:
                        st.error(err)
        
        return simulations, summaries
    
    def extract_summary_statistics(self, vtu_files: List[str], energy: float, 
                                duration: float, name: str) -> Dict:
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
                summary['timesteps'].append(idx + 1)
                
                for field_name in mesh.point_data.keys():
                    data = mesh.point_data[field_name]
                    
                    if field_name not in summary['field_stats']:
                        summary['field_stats'][field_name] = {
                            'min': [], 'max': [], 'mean': [], 'std': [],
                            'q25': [], 'q50': [], 'q75': []
                        }
                    
                    # Convert vector/tensor fields to magnitude for statistics
                    if data.ndim == 1:
                        clean_data = data[~np.isnan(data)]
                    else:
                        # Take Euclidean norm over component axes
                        mag = np.linalg.norm(data, axis=1)
                        clean_data = mag[~np.isnan(mag)]
                    
                    if clean_data.size > 0:
                        summary['field_stats'][field_name]['min'].append(float(np.min(clean_data)))
                        summary['field_stats'][field_name]['max'].append(float(np.max(clean_data)))
                        summary['field_stats'][field_name]['mean'].append(float(np.mean(clean_data)))
                        summary['field_stats'][field_name]['std'].append(float(np.std(clean_data)))
                        summary['field_stats'][field_name]['q25'].append(float(np.percentile(clean_data, 25)))
                        summary['field_stats'][field_name]['q50'].append(float(np.percentile(clean_data, 50)))
                        summary['field_stats'][field_name]['q75'].append(float(np.percentile(clean_data, 75)))
                    else:
                        # Fill with zeros if no valid data
                        for key in ['min', 'max', 'mean', 'std', 'q25', 'q50', 'q75']:
                            summary['field_stats'][field_name][key].append(0.0)
                            
            except Exception as e:
                st.warning(f"Error processing {vtu_file}: {e}")
                continue
        
        return summary
    
    def get_field_range(self, field_name: str, simulations: Dict) -> Tuple[float, float]:
        """Get global min/max for a field across all simulations"""
        all_mins, all_maxs = [], []
        for sim in simulations.values():
            if field_name in sim.get('field_stats', {}):
                stats = sim['field_stats'][field_name]
                if stats['min']:
                    all_mins.extend(stats['min'])
                    all_maxs.extend(stats['max'])
        if all_mins and all_maxs:
            return min(all_mins), max(all_maxs)
        return 0.0, 1.0

# =============================================
# 3. ST-DGPA EXTRAPOLATOR
# =============================================
class SpatioTemporalGatedPhysicsAttentionExtrapolator:
    """Advanced extrapolator with Spatio-Temporal Gated Physics Attention (ST-DGPA)"""
    
    def __init__(self, sigma_param: float = 0.3, spatial_weight: float = 0.5,
                 n_heads: int = 4, temperature: float = 1.0,
                 sigma_g: float = 0.20, s_E: float = 10.0, s_tau: float = 5.0,
                 s_t: float = 20.0, temporal_weight: float = 0.3):
        # Core attention parameters
        self.sigma_param = sigma_param
        self.spatial_weight = spatial_weight
        self.n_heads = n_heads
        self.temperature = temperature
        
        # ST-DGPA specific gating parameters
        self.sigma_g = sigma_g      # Gating kernel width
        self.s_E = s_E              # Energy scaling factor (mJ)
        self.s_tau = s_tau          # Duration scaling factor (ns)
        self.s_t = s_t              # Time scaling factor (ns)
        self.temporal_weight = temporal_weight  # Weight for temporal similarity
        
        # Physics parameters for heat transfer characterization
        self.thermal_diffusivity = 1e-5      # m²/s, typical for metals
        self.laser_spot_radius = 50e-6       # m, typical laser spot size
        self.characteristic_length = 100e-6  # m, characteristic length for heat diffusion
        
        # Data containers
        self.source_db = []
        self.embedding_scaler = StandardScaler()
        self.value_scaler = StandardScaler()
        self.source_embeddings = []
        self.source_values = []
        self.source_metadata = []
        self.fitted = False
    
    def load_summaries(self, summaries: List[Dict]):
        """Load summary statistics and prepare for attention mechanism"""
        self.source_db = summaries
        if not summaries:
            return
        
        all_embeddings, all_values, metadata = [], [], []
        
        for summary_idx, summary in enumerate(summaries):
            for timestep_idx, t in enumerate(summary['timesteps']):
                # Compute physics-aware embedding with enhanced temporal features
                emb = self._compute_enhanced_physics_embedding(
                    summary['energy'], summary['duration'], t
                )
                all_embeddings.append(emb)
                
                # Extract field values (flattened) for prediction target
                field_vals = []
                for field in sorted(summary['field_stats'].keys()):
                    stats = summary['field_stats'][field]
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
                    'fourier_number': self._compute_fourier_number(t),
                    'thermal_penetration': self._compute_thermal_penetration(t)
                })
        
        if all_embeddings and all_values:
            all_embeddings = np.array(all_embeddings)
            all_values = np.array(all_values)
            
            # Scale embeddings for stable attention computation
            self.embedding_scaler.fit(all_embeddings)
            self.source_embeddings = self.embedding_scaler.transform(all_embeddings)
            
            # Scale values for stable prediction
            self.value_scaler.fit(all_values)
            self.source_values = all_values
            self.source_metadata = metadata
            self.fitted = True
            
            st.info(f"✅ Prepared {len(all_embeddings)} embeddings with {all_embeddings.shape[1]} features")
    
    def _compute_fourier_number(self, time_ns: float) -> float:
        """Compute Fourier number (Fo = αt/L²) for heat transfer characterization"""
        time_s = time_ns * 1e-9  # Convert ns to s
        return self.thermal_diffusivity * time_s / (self.characteristic_length ** 2)
    
    def _compute_thermal_penetration(self, time_ns: float) -> float:
        """Compute thermal penetration depth (δ ~ √(αt)) in micrometers"""
        time_s = time_ns * 1e-9  # Convert ns to s
        return np.sqrt(self.thermal_diffusivity * time_s) * 1e6  # Convert to μm
    
    def _compute_enhanced_physics_embedding(self, energy: float, duration: float, time: float) -> np.ndarray:
        """Compute comprehensive physics-aware embedding with temporal features"""
        # Basic physical parameters
        logE = np.log1p(energy)
        power = energy / max(duration, 1e-6)
        time_ratio = time / max(duration, 1e-3)
        
        # Heat transfer specific features
        fourier_number = self._compute_fourier_number(time)
        thermal_penetration = self._compute_thermal_penetration(time)
        
        # Temporal phase indicators
        heating_phase = 1.0 if time < duration else 0.0
        cooling_phase = 1.0 if time >= duration else 0.0
        
        # Combined features with enhanced temporal information
        return np.array([
            logE, duration, time, power, time_ratio,
            fourier_number, thermal_penetration,
            heating_phase, cooling_phase,
            np.log1p(power), np.log1p(time), np.sqrt(time)
        ], dtype=np.float32)
    
    def _compute_ett_gating(self, energy_query: float, duration_query: float,
                           time_query: float, source_metadata: Optional[List] = None) -> np.ndarray:
        """Compute the (E, τ, t) gating kernel for ST-DGPA"""
        if source_metadata is None:
            source_metadata = self.source_metadata
        
        phi_squared = []
        for meta in source_metadata:
            de = (energy_query - meta['energy']) / self.s_E
            dt = (duration_query - meta['duration']) / self.s_tau
            dtime = (time_query - meta['time']) / self.s_t
            
            # Apply physics-aware temporal scaling for heat transfer
            if self.temporal_weight > 0:
                # For heat transfer: early times need tighter temporal matching
                time_scaling_factor = 1.0 + 0.5 * (time_query / max(duration_query, 1e-6))
                dtime *= time_scaling_factor
            
            phi_squared.append(de**2 + dt**2 + dtime**2)
        
        phi_squared = np.array(phi_squared)
        gating = np.exp(-phi_squared / (2 * self.sigma_g**2))
        
        # Normalize gating weights
        gating_sum = np.sum(gating)
        return gating / gating_sum if gating_sum > 0 else np.ones_like(gating) / len(gating)
    
    def _compute_temporal_similarity(self, query_meta: Dict, source_metas: List[Dict]) -> np.ndarray:
        """Compute temporal similarity with physics-aware weighting for heat transfer"""
        similarities = []
        for meta in source_metas:
            time_diff = abs(query_meta['time'] - meta['time'])
            
            # Physics-aware temporal similarity
            if query_meta['time'] < query_meta['duration'] * 1.5:
                # Tighter matching during heating phase
                tolerance = max(query_meta['duration'] * 0.1, 1.0)
            else:
                # Looser matching during late cooling/diffusion phase
                tolerance = max(query_meta['duration'] * 0.3, 3.0)
            
            # Fourier number similarity for heat transfer characterization
            fourier_similarity = np.exp(
                -abs(query_meta.get('fourier_number', 0) - meta.get('fourier_number', 0)) / 0.1
            )
            time_similarity = np.exp(-time_diff / tolerance)
            
            # Combine time difference and Fourier number similarity
            similarities.append(
                (1 - self.temporal_weight) * time_similarity +
                self.temporal_weight * fourier_similarity
            )
        
        return np.array(similarities)
    
    def _compute_spatial_similarity(self, query_meta: Dict, source_metas: List[Dict]) -> np.ndarray:
        """Compute spatial similarity based on parameter proximity"""
        similarities = []
        for meta in source_metas:
            e_diff = abs(query_meta['energy'] - meta['energy']) / 50.0
            d_diff = abs(query_meta['duration'] - meta['duration']) / 20.0
            total_diff = np.sqrt(e_diff**2 + d_diff**2)
            similarities.append(np.exp(-total_diff / self.sigma_param))
        return np.array(similarities)
    
    def _multi_head_attention_with_gating(self, query_embedding: np.ndarray,
                                         query_meta: Dict) -> Tuple:
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
        ett_gating = self._compute_ett_gating(
            query_meta['energy'], query_meta['duration'], query_meta['time']
        )
        
        # ST-DGPA formula: w_i = (α_i * gating_i) / (∑_j α_j * gating_j)
        combined_weights = physics_attention * ett_gating
        combined_sum = np.sum(combined_weights)
        final_weights = combined_weights / combined_sum if combined_sum > 1e-12 else physics_attention
        
        # Weighted prediction (using original values, not scaled)
        prediction = np.sum(final_weights[:, np.newaxis] * self.source_values, axis=0) \
            if len(self.source_values) > 0 else np.zeros(1)
        
        return prediction, final_weights, physics_attention, ett_gating
    
    def predict_field_statistics(self, energy_query: float, duration_query: float,
                                time_query: float) -> Optional[Dict]:
        """Predict field statistics for given parameters using ST-DGPA"""
        if not self.fitted:
            return None
        
        # Compute query embedding and metadata with enhanced temporal features
        query_embedding = self._compute_enhanced_physics_embedding(
            energy_query, duration_query, time_query
        )
        query_meta = {
            'energy': energy_query,
            'duration': duration_query,
            'time': time_query,
            'fourier_number': self._compute_fourier_number(time_query),
            'thermal_penetration': self._compute_thermal_penetration(time_query)
        }
        
        # Apply ST-DGPA attention mechanism
        prediction, final_weights, physics_attention, ett_gating = \
            self._multi_head_attention_with_gating(query_embedding, query_meta)
        
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
            'heat_transfer_indicators': self._compute_heat_transfer_indicators(
                energy_query, duration_query, time_query
            ),
            'field_predictions': {}
        }
        
        # Map predictions back to fields
        if self.source_db:
            field_order = sorted(self.source_db[0]['field_stats'].keys())
            for i, field in enumerate(field_order):
                start_idx = i * 3  # mean, max, std
                if start_idx + 2 < len(prediction):
                    result['field_predictions'][field] = {
                        'mean': float(prediction[start_idx]),
                        'max': float(prediction[start_idx + 1]),
                        'std': float(prediction[start_idx + 2])
                    }
        
        return result
    
    def _compute_temporal_confidence(self, time_query: float, duration_query: float) -> float:
        """Compute confidence in temporal prediction based on heat transfer physics"""
        if time_query < duration_query * 0.5:  # Early heating phase
            return 0.6  # Moderate confidence
        elif time_query < duration_query * 1.5:  # Peak and early cooling
            return 0.8  # Higher confidence
        else:  # Late cooling/diffusion
            return 0.9  # Highest confidence (diffusion-dominated)
    
    def _compute_heat_transfer_indicators(self, energy: float, duration: float,
                                         time: float) -> Dict:
        """Compute heat transfer characterization indicators"""
        fourier_number = self._compute_fourier_number(time)
        thermal_penetration = self._compute_thermal_penetration(time)
        
        # Phase identification
        if time < duration * 0.3:
            phase, regime = "Early Heating", "Adiabatic-like"
        elif time < duration:
            phase, regime = "Heating", "Conduction-dominated"
        elif time < duration * 2:
            phase, regime = "Early Cooling", "Mixed conduction"
        else:
            phase, regime = "Diffusion Cooling", "Thermal diffusion"
        
        return {
            'phase': phase,
            'regime': regime,
            'fourier_number': fourier_number,
            'thermal_penetration_um': thermal_penetration,
            'normalized_time': time / max(duration, 1e-6),
            'energy_density': energy / duration
        }
    
    def predict_time_series(self, energy_query: float, duration_query: float,
                           time_points: np.ndarray) -> Dict:
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
            common_fields = set(f for s in self.source_db for f in s['field_stats'].keys())
            for field in common_fields:
                results['field_predictions'][field] = {'mean': [], 'max': [], 'std': []}
        
        for t in time_points:
            pred = self.predict_field_statistics(energy_query, duration_query, t)
            if pred and 'field_predictions' in pred:
                for field in pred['field_predictions']:
                    if field in results['field_predictions']:
                        results['field_predictions'][field]['mean'].append(
                            pred['field_predictions'][field]['mean']
                        )
                        results['field_predictions'][field]['max'].append(
                            pred['field_predictions'][field]['max']
                        )
                        results['field_predictions'][field]['std'].append(
                            pred['field_predictions'][field]['std']
                        )
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
    
    def interpolate_full_field(self, field_name: str, attention_weights: np.ndarray,
                            source_metadata: List[Dict], simulations: Dict) -> Optional[np.ndarray]:
        """Compute interpolated full field using ST-DGPA weights"""
        if not self.fitted or len(attention_weights) == 0:
            return None
        
        # Assume common mesh: Get shape from first simulation
        first_sim = next(iter(simulations.values()))
        if 'fields' not in first_sim or field_name not in first_sim['fields']:
            return None
        
        field_shape = first_sim['fields'][field_name].shape[1:]
        
        # Initialize interpolated field based on field type
        if len(field_shape) == 0:  # Scalar field
            interpolated_field = np.zeros(first_sim['fields'][field_name].shape[1], dtype=np.float32)
        else:  # Vector field
            interpolated_field = np.zeros(field_shape, dtype=np.float32)
        
        total_weight, n_sources_used = 0.0, 0
        for idx, weight in enumerate(attention_weights):
            if weight < 1e-6:  # Skip negligible weights for efficiency
                continue
            meta = source_metadata[idx]
            if meta['name'] in simulations and field_name in simulations[meta['name']]['fields']:
                interpolated_field += weight * simulations[meta['name']]['fields'][field_name][meta['timestep_idx']]
                total_weight += weight
                n_sources_used += 1
        
        return interpolated_field / total_weight if total_weight > 0 else None
    
    def export_interpolated_vtu(self, field_name: str, interpolated_values: np.ndarray,
                               simulations: Dict, output_path: str) -> bool:
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
            mesh = meshio.Mesh(
                points, cells,
                point_data={field_name: interpolated_values}
            )
            mesh.write(output_path)
            return True
        except Exception as e:
            st.error(f"Error exporting VTU: {str(e)}")
            return False

# =============================================
# 4. POLAR RADAR VISUALIZER (FIXED & ENHANCED)
# =============================================
class PolarRadarVisualizer:
    """Creates polar radar charts with Energy (angular), Pulse Duration (radial), and Peak Value (color/size)"""
    
    def __init__(self):
        self.color_scale_temp = 'Inferno'
        self.color_scale_stress = 'Plasma'
        self.target_symbol = 'star-diamond'
        self.target_color = '#00FF7F'
        self.source_symbol = 'circle'
        self.source_color_scale = 'Viridis'
    
    def create_polar_radar_chart(self, df: pd.DataFrame, field_type: str = 'temperature',
                                query_params: Optional[Dict] = None,
                                timestep: int = 1,
                                width: int = 800, height: int = 700,
                                show_legend: bool = True,
                                marker_size_range: Tuple[int, int] = (10, 30)) -> go.Figure:
        """
        Create polar radar chart.
        Angular axis: Energy (mJ)
        Radial axis: Pulse Duration (ns)
        Color/Size: Peak Value (Temperature or Stress)
        """
        if df.empty:
            fig = go.Figure()
            fig.update_layout(title="No data available", width=width, height=height)
            return fig

        # Extract parameters, ensure numeric and finite
        energies = pd.to_numeric(df['Energy'], errors='coerce').dropna().values
        durations = pd.to_numeric(df['Duration'], errors='coerce').dropna().values
        peak_values = pd.to_numeric(df['Peak_Value'], errors='coerce').dropna().values
        sim_names = df.get('Name', [f"Sim {i}" for i in range(len(df))]).tolist()

        if len(energies) == 0 or len(peak_values) == 0:
            fig = go.Figure()
            fig.update_layout(title="No valid numeric data", width=width, height=height)
            return fig

        # Normalize Energy to Angular Axis (0 to 2*pi)
        e_min, e_max = np.nanmin(energies), np.nanmax(energies)
        e_range = e_max - e_min if e_max > e_min else 1.0
        
        # Protect against division by zero or extreme values
        if not np.isfinite(e_range) or e_range < 1e-6:
            e_range = 1.0
            e_min = 0.0

        angles_rad = 2 * np.pi * (energies - e_min) / e_range
        angles_deg = np.degrees(angles_rad)

        # Prepare hover text
        hover_text = []
        for i in range(len(df)):
            if i < len(sim_names) and i < len(energies) and i < len(durations) and i < len(peak_values):
                hover_text.append(
                    f"<b>{sim_names[i]}</b><br>"
                    f"Time Step: {timestep}<br>"
                    f"Energy: {energies[i]:.2f} mJ<br>"
                    f"Duration: {durations[i]:.2f} ns<br>"
                    f"Peak {field_type.capitalize()}: {peak_values[i]:.3f}"
                )

        # Determine color scale and title
        is_temp = 'temp' in field_type.lower()
        c_scale = self.color_scale_temp if is_temp else self.color_scale_stress
        title_field = "Peak Temperature (K)" if is_temp else "Peak von Mises Stress (MPa)"

        # Normalize peak values for marker sizing
        p_min, p_max = np.min(peak_values), np.max(peak_values)
        p_range = p_max - p_min if p_max > p_min else 1.0
        marker_sizes = marker_size_range[0] + (peak_values - p_min) / p_range * (marker_size_range[1] - marker_size_range[0])

        # Create Figure
        fig = go.Figure()

        # Source simulations scatter
        fig.add_trace(go.Scatterpolar(
            r=durations,
            theta=angles_deg,
            mode='markers',
            marker=dict(
                size=marker_sizes,
                color=peak_values,
                colorscale=c_scale,
                colorbar=dict(title=title_field, thickness=20),
                line=dict(width=2, color='white'),
                symbol=self.source_symbol
            ),
            text=hover_text,
            hoverinfo='text',
            name='Source Simulations',
            opacity=0.85
        ))

        # Add Target Query Point if provided
        if query_params:
            q_e = query_params.get('Energy')
            q_d = query_params.get('Duration')
            if q_e is not None and q_d is not None and np.isfinite(q_e) and np.isfinite(q_d):
                q_angle_rad = 2 * np.pi * (q_e - e_min) / e_range
                q_angle_deg = np.degrees(q_angle_rad)
                fig.add_trace(go.Scatterpolar(
                    r=[q_d],
                    theta=[q_angle_deg],
                    mode='markers',
                    marker=dict(
                        size=25,
                        color=self.target_color,
                        symbol=self.target_symbol,
                        line=dict(width=3, color='black')
                    ),
                    name='Target Query',
                    hovertemplate=f"<b>Target Query</b><br>Energy: {q_e:.2f} mJ<br>Duration: {q_d:.2f} ns<extra></extra>"
                ))

        # Build polar layout
        polar_layout = dict(
            radialaxis=dict(
                visible=True,
                title="Pulse Duration (ns)",
                gridcolor="lightgray",
                tickfont=dict(size=12)
            ),
            angularaxis=dict(
                visible=True,
                direction="clockwise",
                rotation=90,
                gridcolor="lightgray",
                tickfont=dict(size=12)
                # NOTE: Plotly does not support 'title' on angularaxis
            ),
            bgcolor="white"
        )

        # Add custom tick labels for energy if range is valid
        if e_range > 1e-6 and np.isfinite(e_min) and np.isfinite(e_max):
            try:
                tick_angles, tick_texts = [], []
                for v in np.linspace(e_min, e_max, 6):
                    angle_rad = 2 * np.pi * (v - e_min) / e_range
                    angle_deg = np.degrees(angle_rad)
                    if 0 <= angle_deg <= 360:
                        tick_angles.append(angle_deg)
                        tick_texts.append(f"{v:.1f}")
                if tick_angles:
                    polar_layout['angularaxis']['tickvals'] = tick_angles
                    polar_layout['angularaxis']['ticktext'] = tick_texts
            except Exception:
                pass  # Skip custom ticks if calculation fails

        # Apply layout with error fallback
        try:
            fig.update_layout(
                title=dict(
                    text=f"Polar Radar: {title_field} at t={timestep} ns<br>"
                         f"<span style='font-size:12px; color:gray;'>Angular: Energy (mJ) • Radial: Pulse Duration (ns)</span>",
                    font=dict(size=18),
                    x=0.5,
                    xanchor='center'
                ),
                polar=polar_layout,
                width=width,
                height=height,
                showlegend=show_legend,
                margin=dict(l=60, r=60, t=80, b=60),
                hovermode='closest'
            )
        except Exception as e:
            st.warning(f"Polar layout error, using fallback: {str(e)}")
            fig.update_layout(
                title=f"Polar Radar: {title_field} at t={timestep} ns",
                width=width,
                height=height,
                showlegend=show_legend,
                margin=dict(l=50, r=50, t=50, b=50)
            )

        return fig
    
    def create_multi_timestep_radar(self, df: pd.DataFrame, field_type: str,
                                   query_params: Optional[Dict],
                                   timesteps: List[int],
                                   width: int = 300, height: int = 300) -> List[go.Figure]:
        """Create multiple radar charts for different timesteps"""
        figures = []
        for t in timesteps:
            # Filter data for this timestep (assuming df has timestep info)
            df_t = df.copy()  # In practice, filter by actual timestep column
            fig = self.create_polar_radar_chart(
                df_t, field_type, query_params, timestep=t,
                width=width, height=height, show_legend=(t == timesteps[0]),
                marker_size_range=(8, 20)
            )
            figures.append(fig)
        return figures

# =============================================
# 5. SANKEY VISUALIZER (FIXED & ENHANCED)
# =============================================
class SankeyVisualizer:
    """Creates Sankey diagrams showing ST-DGPA weight decomposition flow"""
    
    def __init__(self):
        self.defaults = {
            'font_family': 'Arial, sans-serif',
            'font_size': 12,
            'node_thickness': 20,
            'node_pad': 15,
            'width': 1000,
            'height': 700,
            'show_math': True,
            'target_label': 'TARGET',
            'node_colors': {
                'target': '#FF6B6B',
                'source': '#9966FF',
                'components': ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFD93D', '#36A2EB', '#9966FF']
            }
        }

    def create_stdgpa_sankey(self, sources_: List[Dict], query: Dict, 
                            customization: Optional[Dict] = None) -> go.Figure:
        """Create Sankey diagram showing ST-DGPA weight flow from sources through components to target"""
        cfg = {**self.defaults, **(customization or {})}
        
        # Component labels with optional mathematical notation
        if cfg['show_math']:
            comp_labels = [
                'Energy Gate\nφ² = ((E*-Eᵢ)/s_E)²',
                'Duration Gate\nφ² = ((τ*-τᵢ)/s_τ)²',
                'Time Gate\nφ² = ((t*-tᵢ)/s_t)²',
                'Attention\nαᵢ = softmax(QKᵀ/√dₖ)',
                'Refinement\nwᵢ ∝ αᵢ·gatingᵢ',
                'Combined\nwᵢ = αᵢ·gatingᵢ / Σ(...)'
            ]
        else:
            comp_labels = ['Energy Gate', 'Duration Gate', 'Time Gate', 'Attention', 'Refinement', 'Combined']
        
        # Build node labels and colors
        labels = [cfg['target_label']]
        node_colors = [cfg['node_colors']['target']]
        n_sources = len(sources_)
        
        for i in range(n_sources):
            row = sources_[i]
            w = row.get('Combined_Weight', 0)
            opacity = min(0.3 + w * 0.7, 1.0)
            base = cfg['node_colors']['source']
            r, g, b = int(base[1:3], 16), int(base[3:5], 16), int(base[5:7], 16)
            node_colors.append(f'rgba({r},{g},{b},{opacity:.2f})')
            labels.append(f"Sim {i+1}\nE:{row.get('Energy',0):.1f} τ:{row.get('Duration',0):.1f}")
        
        # Add component nodes
        comp_start = len(labels)
        labels.extend(comp_labels)
        node_colors.extend(cfg['node_colors']['components'])
        
        # Build links: sources → components → target
        s_idx, t_idx, vals, l_colors, h_texts = [], [], [], [], []
        
        for i in range(n_sources):
            src = i + 1
            row = sources_[i]
            
            # Compute gate values (scaled for visualization)
            ve = ((row.get('Energy', query['Energy']) - query['Energy']) / 10.0)**2 * 10
            vτ = ((row.get('Duration', query['Duration']) - query['Duration']) / 5.0)**2 * 10
            vt = ((row.get('Time', query['Time']) - query['Time']) / 20.0)**2 * 10
            va = row.get('Attention_Score', 0) * 100
            vr = row.get('Refinement', 0) * 100
            vc = row.get('Combined_Weight', 0) * 100
            
            vals_list = [ve, vτ, vt, va, vr, vc]
            
            for c in range(6):
                s_idx.append(src)
                t_idx.append(comp_start + c)
                vals.append(max(0.01, vals_list[c]))
                l_colors.append(cfg['node_colors']['components'][c].replace('rgb', 'rgba').replace(')', ', 0.5)'))
                
                # Create hover text
                if c == 0:
                    h_texts.append(f"<b>Energy Gate</b><br>S{src} | φ² = ((E*-Eᵢ)/s_E)² = {ve/10:.4f}")
                elif c == 1:
                    h_texts.append(f"<b>Duration Gate</b><br>S{src} | φ² = ((τ*-τᵢ)/s_τ)² = {vτ/10:.4f}")
                elif c == 2:
                    h_texts.append(f"<b>Time Gate</b><br>S{src} | φ² = ((t*-tᵢ)/s_t)² = {vt/10:.4f}")
                elif c == 3:
                    h_texts.append(f"<b>Attention</b><br>S{src} | αᵢ = 1/(1+√φ²)<br>Score: {row.get('Attention_Score',0):.4f}")
                elif c == 4:
                    h_texts.append(f"<b>Refinement</b><br>S{src} | wᵢ ∝ αᵢ·gatingᵢ<br>Ref: {row.get('Refinement',0):.4f}")
                else:
                    h_texts.append(f"<b>Combined Weight</b><br>S{src} | wᵢ = (αᵢ·gatingᵢ)/Σ(...)<br>Weight: {row.get('Combined_Weight',0):.4f}")

        # Aggregate component → target flows
        for c in range(6):
            s_idx.append(comp_start + c)
            t_idx.append(0)
            flow_in = sum(v for s, t, v in zip(s_idx[:-6], t_idx[:-6], vals[:-6]) if t == comp_start + c)
            vals.append(flow_in * 0.5)
            l_colors.append('rgba(153,102,255,0.6)')
            h_texts.append(f"<b>Aggregation</b><br>{comp_labels[c]} → TARGET<br>Total: {flow_in:.3f}")
            
        # Create Sankey diagram
        # NOTE: 'line' property is NOT allowed for Sankey links in Plotly
        fig = go.Figure(go.Sankey(
            node=dict(
                pad=cfg['node_pad'],
                thickness=cfg['node_thickness'],
                line=dict(color="black", width=0.5),
                label=labels,
                color=node_colors,
                font=dict(family=cfg['font_family'], size=cfg['font_size']),
                hovertemplate='<b>%{label}</b><br>Value: %{value:.3f}<extra></extra>'
            ),
            link=dict(
                source=s_idx,
                target=t_idx,
                value=vals,
                color=l_colors,
                hovertext=h_texts,
                hovertemplate='%{hovertext}<extra></extra>'
                # 'line' property removed - not supported for Sankey links
            )
        ))
        
        # Build title with query parameters
        title_text = (f"<b>ST-DGPA Attention Flow</b><br>"
                     f"Query: E={query['Energy']:.2f} mJ, τ={query['Duration']:.2f} ns, t={query['Time']:.2f} ns<br>"
                     f"<sub>σ_g={cfg.get('sigma_g', 0.20):.2f}, s_E={cfg.get('s_E', 10.0):.1f}, s_τ={cfg.get('s_tau', 5.0):.1f}, s_t={cfg.get('s_t', 20.0):.1f}</sub>")
        
        fig.update_layout(
            title=dict(
                text=title_text,
                font=dict(family=cfg['font_family'], size=cfg['font_size']+4),
                x=0.5,
                xanchor='center'
            ),
            font=dict(family=cfg['font_family'], size=cfg['font_size']),
            width=cfg['width'],
            height=cfg['height'],
            plot_bgcolor='rgba(240, 240, 245, 0.9)',
            paper_bgcolor='white',
            margin=dict(t=100, l=50, r=50, b=50),
            hoverlabel=dict(
                font=dict(family=cfg['font_family'], size=cfg['font_size']),
                bgcolor='rgba(44, 62, 80, 0.9)',
                bordercolor='white',
                namelength=-1
            )
        )
        
        return fig
    
    def create_comparison_sankey(self, sources_a: List[Dict], sources_b: List[Dict],
                               query: Dict, label_a: str = "Config A", label_b: str = "Config B",
                               customization: Optional[Dict] = None) -> go.Figure:
        """Create side-by-side Sankey comparison of two configurations"""
        cfg = {**self.defaults, **(customization or {})}
        
        # Create two separate Sankey diagrams
        fig_a = self.create_stdgpa_sankey(sources_a, query, {**cfg, 'width': cfg['width']//2 - 20})
        fig_b = self.create_stdgpa_sankey(sources_b, query, {**cfg, 'width': cfg['width']//2 - 20})
        
        # Combine into subplots
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=1, cols=2, subplot_titles=[label_a, label_b])
        
        # Add traces from both figures (simplified - in practice would need to extract traces)
        # This is a placeholder for the full implementation
        
        fig.update_layout(
            title=f"ST-DGPA Comparison: {label_a} vs {label_b}",
            width=cfg['width'],
            height=cfg['height']
        )
        
        return fig

# =============================================
# 6. ENHANCED VISUALIZER
# =============================================
class EnhancedVisualizer:
    """Comprehensive visualization with extended analysis capabilities"""
    
    @staticmethod
    def create_stdgpa_analysis(results: Dict, energy_query: float, duration_query: float,
                            time_points: np.ndarray) -> Optional[go.Figure]:
        """Create comprehensive ST-DGPA analysis visualization"""
        if not results or 'attention_maps' not in results or len(results['attention_maps']) == 0:
            return None
        
        timestep_idx = len(time_points) // 2
        time = time_points[timestep_idx]
        
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                "ST-DGPA Final Weights", "Physics Attention Only", "(E, τ, t) Gating Only",
                "ST-DGPA vs Physics Attention", "Temporal Coherence Analysis", "Heat Transfer Phase",
                "Parameter Space 3D", "Attention Network", "Weight Evolution"
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.12,
            specs=[
                [{'type': 'xy'}, {'type': 'xy'}, {'type': 'xy'}],
                [{'type': 'xy'}, {'type': 'xy'}, {'type': 'polar'}],
                [{'type': 'scene'}, {'type': 'xy'}, {'type': 'xy'}]
            ]
        )
        
        final_weights = results['attention_maps'][timestep_idx]
        physics_attention = results['physics_attention_maps'][timestep_idx]
        ett_gating = results['ett_gating_maps'][timestep_idx]
        
        # 1. Final ST-DGPA weights
        fig.add_trace(
            go.Bar(x=list(range(len(final_weights))), y=final_weights,
                  marker_color='blue', showlegend=False),
            row=1, col=1
        )
        
        # 2. Physics attention only
        fig.add_trace(
            go.Bar(x=list(range(len(physics_attention))), y=physics_attention,
                  marker_color='green', showlegend=False),
            row=1, col=2
        )
        
        # 3. (E, τ, t) gating only
        fig.add_trace(
            go.Bar(x=list(range(len(ett_gating))), y=ett_gating,
                  marker_color='red', showlegend=False),
            row=1, col=3
        )
        
        # 4. Comparison: ST-DGPA vs Physics Attention
        fig.add_trace(
            go.Scatter(x=list(range(len(final_weights))), y=final_weights,
                      mode='lines+markers', line=dict(color='blue', width=3)),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=list(range(len(physics_attention))), y=physics_attention,
                      mode='lines+markers', line=dict(color='green', width=2, dash='dash')),
            row=2, col=1
        )
        
        # 5. Temporal coherence analysis
        if st.session_state.get('summaries') and hasattr(st.session_state.extrapolator, 'source_metadata'):
            times, weights = [], []
            for i, weight in enumerate(final_weights):
                if weight > 0.01 and i < len(st.session_state.extrapolator.source_metadata):
                    times.append(st.session_state.extrapolator.source_metadata[i]['time'])
                    weights.append(weight)
            if times and weights:
                fig.add_trace(
                    go.Scatter(x=times, y=weights, mode='markers',
                              marker=dict(size=np.array(weights)*50, color=weights,
                                         colorscale='Viridis', showscale=False)),
                    row=2, col=2
                )
                fig.add_vline(x=time, line_dash="dash", line_color="red", row=2, col=2)
        
        # Update layout
        fig.update_layout(
            height=1000,
            title_text=f"ST-DGPA Analysis at t={time} ns (E={energy_query:.1f} mJ, τ={duration_query:.1f} ns)",
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def create_confidence_plot(results: Dict, time_points: np.ndarray) -> go.Figure:
        """Create confidence evolution plot"""
        fig = go.Figure()
        
        if 'confidence_scores' in results and results['confidence_scores']:
            fig.add_trace(go.Scatter(
                x=time_points,
                y=results['confidence_scores'],
                mode='lines+markers',
                name='Prediction Confidence',
                line=dict(color='orange', width=3),
                fill='tozeroy',
                fillcolor='rgba(255, 165, 0, 0.2)'
            ))
        
        if 'temporal_confidences' in results and results['temporal_confidences']:
            fig.add_trace(go.Scatter(
                x=time_points,
                y=results['temporal_confidences'],
                mode='lines+markers',
                name='Temporal Confidence',
                line=dict(color='green', width=3, dash='dash')
            ))
        
        fig.update_layout(
            title="Prediction Confidence Over Time",
            xaxis_title="Time (ns)",
            yaxis_title="Confidence",
            height=400,
            yaxis_range=[0, 1],
            hovermode='x unified'
        )
        
        return fig

# =============================================
# 7. EXPORT UTILITIES
# =============================================
class ExportManager:
    """Handles export of results in various formats"""
    
    @staticmethod
    def export_to_json(data: Dict, filename: Optional[str] = None) -> Tuple[str, str]:
        """Export data to JSON format"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"stdgpa_export_{timestamp}.json"
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj
        
        json_str = json.dumps(convert_numpy(data), indent=2)
        return json_str, filename
    
    @staticmethod
    def export_to_csv(df: pd.DataFrame, filename: Optional[str] = None) -> Tuple[str, str]:
        """Export DataFrame to CSV"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"stdgpa_export_{timestamp}.csv"
        
        csv_str = df.to_csv(index=False)
        return csv_str, filename
    
    @staticmethod
    def export_plotly_figure(fig: go.Figure, format: str = 'png',
                           filename: Optional[str] = None) -> Optional[bytes]:
        """Export Plotly figure to image format"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"stdgpa_plot_{timestamp}.{format}"
            
            if format == 'png':
                return fig.to_image(format='png', width=1200, height=800, scale=2)
            elif format == 'svg':
                return fig.to_image(format='svg', width=1200, height=800)
            elif format == 'html':
                return fig.to_html(include_plotlyjs='cdn').encode('utf-8')
            else:
                st.error(f"Unsupported export format: {format}")
                return None
        except Exception as e:
            st.error(f"Export error: {e}")
            st.info("Tip: Install 'kaleido' for image export: pip install -U kaleido")
            return None

# =============================================
# 8. MAIN APPLICATION
# =============================================
def main():
    # Page configuration
    st.set_page_config(
        page_title="Laser Soldering ST-DGPA Platform",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="🔬"
    )
    
    # Custom CSS for enhanced styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 800;
        color: #1a202c;
        background: linear-gradient(90deg, #1E88E5, #4A00E0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3498db;
        font-weight: 600;
    }
    .info-box {
        background-color: #F0F9FF;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🔬 Laser Soldering ST-DGPA Analysis Platform</h1>', unsafe_allow_html=True)

    # Initialize Session State
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = UnifiedFEADataLoader()
    if 'extrapolator' not in st.session_state:
        st.session_state.extrapolator = SpatioTemporalGatedPhysicsAttentionExtrapolator()
    if 'polar_viz' not in st.session_state:
        st.session_state.polar_viz = PolarRadarVisualizer()
    if 'sankey_viz' not in st.session_state:
        st.session_state.sankey_viz = SankeyVisualizer()
    if 'enhanced_viz' not in st.session_state:
        st.session_state.enhanced_viz = EnhancedVisualizer()
    if 'export_manager' not in st.session_state:
        st.session_state.export_manager = ExportManager()
    
    # Initialize result containers
    if 'interpolation_results' not in st.session_state:
        st.session_state.interpolation_results = None
    if 'interpolation_params' not in st.session_state:
        st.session_state.interpolation_params = None
    if 'polar_query' not in st.session_state:
        st.session_state.polar_query = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False

    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data loading options
        load_full = st.checkbox("Load Full Mesh", value=True,
                               help="Load complete mesh data for 3D visualization (slower)")
        
        if st.button("🔄 Load All Simulations", type="primary", use_container_width=True):
            with st.spinner("Loading simulation data..."):
                sims, summaries = st.session_state.data_loader.load_all_simulations(
                    load_full_mesh=load_full
                )
                st.session_state.simulations = sims
                st.session_state.summaries = summaries
                
                if sims and summaries:
                    st.session_state.extrapolator.load_summaries(summaries)
                    st.session_state.data_loaded = True
                    st.session_state.available_fields = set()
                    for s in summaries:
                        st.session_state.available_fields.update(s['field_stats'].keys())
        
        # Show loaded data summary
        if st.session_state.get('data_loaded') and st.session_state.get('summaries'):
            st.success(f"✅ {len(st.session_state.summaries)} simulations loaded")
            st.markdown("---")
            
            # ST-DGPA Parameters
            st.header("🎯 ST-DGPA Parameters")
            
            col1, col2 = st.columns(2)
            with col1:
                sigma_g = st.slider("σ_g (Gating)", 0.05, 1.0, 0.20, 0.05,
                                   help="Gating kernel width - smaller = sharper gating")
                s_E = st.slider("s_E (Energy)", 0.1, 50.0, 10.0, 0.5,
                               help="Energy scaling factor in mJ")
            with col2:
                s_tau = st.slider("s_τ (Duration)", 0.1, 20.0, 5.0, 0.5,
                                 help="Duration scaling factor in ns")
                s_t = st.slider("s_t (Time)", 1.0, 50.0, 20.0, 1.0,
                               help="Time scaling factor in ns")
            
            temporal_weight = st.slider("Temporal Weight", 0.0, 1.0, 0.3, 0.05,
                                       help="Weight for temporal similarity in attention")
            
            # Update extrapolator parameters
            st.session_state.extrapolator.sigma_g = sigma_g
            st.session_state.extrapolator.s_E = s_E
            st.session_state.extrapolator.s_tau = s_tau
            st.session_state.extrapolator.s_t = s_t
            st.session_state.extrapolator.temporal_weight = temporal_weight
            
            # Cache management
            st.markdown("---")
            st.header("🗄️ Cache Management")
            if st.button("Clear Cache", use_container_width=True):
                CacheManager.clear_3d_cache()
            
            if 'interpolation_3d_cache' in st.session_state:
                cache_size = len(st.session_state.interpolation_3d_cache)
                st.caption(f"Cache size: {cache_size}/20 entries")

    # Main Tabs
    tabs = st.tabs([
        "📊 Data Overview",
        "🔮 Interpolation",
        "🎯 Polar Radar",
        "🕸️ Sankey Diagram",
        "🧠 ST-DGPA Analysis",
        "💾 Export"
    ])
    
    # Check if data is loaded
    if not st.session_state.get('data_loaded') or not st.session_state.get('summaries'):
        st.info("👈 Please load simulations using the sidebar to begin analysis.")
        with st.expander("📁 Expected Directory Structure"):
            st.code("""
            fea_solutions/
            ├── q0p5mJ-delta4p2ns/    # Energy: 0.5 mJ, Duration: 4.2 ns
            │   ├── a_t0001.vtu       # Timestep 1
            │   ├── a_t0002.vtu       # Timestep 2
            │   └── ...
            ├── q1p0mJ-delta2p0ns/    # Energy: 1.0 mJ, Duration: 2.0 ns
            └── ...
            """)
        return
    
    # --- TAB 1: Data Overview ---
    with tabs[0]:
        st.subheader("Loaded Simulations Summary")
        
        # Create summary DataFrame
        df_summary = pd.DataFrame([
            {
                'Name': s['name'],
                'Energy (mJ)': s['energy'],
                'Duration (ns)': s['duration'],
                'Timesteps': len(s['timesteps']),
                'Fields': ', '.join(sorted(s['field_stats'].keys())[:3]) + ('...' if len(s['field_stats']) > 3 else '')
            }
            for s in st.session_state.summaries
        ])
        
        # Display with formatting
        st.dataframe(
            df_summary.style.format({
                'Energy (mJ)': '{:.2f}',
                'Duration (ns)': '{:.2f}'
            }).background_gradient(subset=['Energy (mJ)', 'Duration (ns)'], cmap='Blues'),
            use_container_width=True,
            height=300
        )
        
        # 3D Viewer (if mesh data available)
        if st.session_state.simulations and next(iter(st.session_state.simulations.values())).get('has_mesh'):
            st.markdown("### 🎨 3D Field Viewer")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                sim_name = st.selectbox("Simulation", sorted(st.session_state.simulations.keys()), key="viewer_sim")
            with col2:
                sim = st.session_state.simulations[sim_name]
                field = st.selectbox("Field", sorted(sim['field_info'].keys()), key="viewer_field")
            with col3:
                timestep = st.slider("Timestep", 0, sim['n_timesteps']-1, 0, key="viewer_timestep")
            
            if sim['points'] is not None and field in sim['fields']:
                values = sim['fields'][field][timestep].copy()
                
                # Convert vector/tensor fields to scalar intensity
                if values.ndim >= 2:
                    norm_axes = tuple(range(1, values.ndim))
                    values = np.linalg.norm(values, axis=norm_axes)
                
                values = np.nan_to_num(values, nan=0.0)
                
                # Determine colormap based on field type
                colormap = 'Viridis'
                if 'temp' in field.lower():
                    colormap = 'Inferno'
                elif 'stress' in field.lower():
                    colormap = 'Plasma'
                
                # Create 3D visualization
                if sim['triangles'] is not None and len(sim['triangles']) > 0:
                    fig = go.Figure(go.Mesh3d(
                        x=sim['points'][:,0], y=sim['points'][:,1], z=sim['points'][:,2],
                        i=sim['triangles'][:,0], j=sim['triangles'][:,1], k=sim['triangles'][:,2],
                        intensity=values,
                        colorscale=colormap,
                        intensitymode='vertex',
                        colorbar=dict(title=field),
                        hovertemplate='<b>Value:</b> %{intensity:.3f}<br><b>X:</b> %{x:.3f}<br><b>Y:</b> %{y:.3f}<br><b>Z:</b> %{z:.3f}<extra></extra>'
                    ))
                else:
                    fig = go.Figure(go.Scatter3d(
                        x=sim['points'][:,0], y=sim['points'][:,1], z=sim['points'][:,2],
                        mode='markers',
                        marker=dict(
                            size=3,
                            color=values,
                            colorscale=colormap,
                            colorbar=dict(title=field),
                            showscale=True
                        ),
                        hovertemplate='<b>Value:</b> %{marker.color:.3f}<br><b>X:</b> %{x:.3f}<br><b>Y:</b> %{y:.3f}<br><b>Z:</b> %{z:.3f}<extra></extra>'
                    ))
                
                fig.update_layout(
                    scene=dict(aspectmode="data"),
                    title=f"{field} at Timestep {timestep+1}",
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Field statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("Min", f"{np.min(values):.3f}")
                with col2: st.metric("Max", f"{np.max(values):.3f}")
                with col3: st.metric("Mean", f"{np.mean(values):.3f}")
                with col4: st.metric("Std Dev", f"{np.std(values):.3f}")

    # --- TAB 2: Interpolation ---
    with tabs[1]:
        st.subheader("Run ST-DGPA Interpolation")
        
        # Query parameter inputs
        c1, c2, c3 = st.columns(3)
        with c1:
            # Get energy range from loaded data
            energies = [s['energy'] for s in st.session_state.summaries]
            min_e, max_e = min(energies), max(energies)
            q_E = st.number_input("Energy (mJ)", float(min_e*0.5), float(max_e*2.0), 
                                 float((min_e+max_e)/2), 0.1, key="interp_energy")
        with c2:
            durations = [s['duration'] for s in st.session_state.summaries]
            min_d, max_d = min(durations), max(durations)
            q_τ = st.number_input("Duration (ns)", float(min_d*0.5), float(max_d*2.0),
                                 float((min_d+max_d)/2), 0.1, key="interp_duration")
        with c3:
            max_t = st.number_input("Max Time (ns)", 1, 200, 50, 1, key="interp_maxtime")
        
        time_resolution = st.selectbox("Time Resolution", ["1 ns", "2 ns", "5 ns", "10 ns"], index=1)
        time_step = {"1 ns": 1, "2 ns": 2, "5 ns": 5, "10 ns": 10}[time_resolution]
        time_points = np.arange(1, max_t + 1, time_step)
        
        # Run prediction button
        if st.button("🚀 Run ST-DGPA Prediction", type="primary", use_container_width=True):
            with st.spinner("Computing Spatio-Temporal Gated Physics Attention..."):
                start_time = time.time()
                results = st.session_state.extrapolator.predict_time_series(q_E, q_τ, time_points)
                elapsed = time.time() - start_time
                
                if results and results['field_predictions']:
                    st.session_state.interpolation_results = results
                    st.session_state.interpolation_params = {
                        'energy_query': q_E,
                        'duration_query': q_τ,
                        'time_points': time_points
                    }
                    st.session_state.polar_query = {'Energy': q_E, 'Duration': q_τ}
                    st.success(f"✅ Prediction Complete ({elapsed:.2f}s)")
                else:
                    st.error("❌ Prediction failed. Check parameters and ensure sufficient training data.")
        
        # Display results if available
        if st.session_state.interpolation_results:
            results = st.session_state.interpolation_results
            params = st.session_state.interpolation_params
            
            st.subheader("Prediction Results")
            
            # Field selection
            fields = list(results['field_predictions'].keys())
            if fields:
                selected_fields = st.multiselect("Select Fields to Plot", fields, default=fields[:3])
                
                if selected_fields:
                    # Create prediction plot
                    fig_preds = go.Figure()
                    for f in selected_fields:
                        if results['field_predictions'][f]['mean']:
                            fig_preds.add_trace(go.Scatter(
                                x=params['time_points'],
                                y=results['field_predictions'][f]['mean'],
                                mode='lines',
                                name=f,
                                line=dict(width=2)
                            ))
                            # Add confidence band if std available
                            if results['field_predictions'][f]['std']:
                                mean = np.array(results['field_predictions'][f]['mean'])
                                std = np.array(results['field_predictions'][f]['std'])
                                fig_preds.add_trace(go.Scatter(
                                    x=np.concatenate([params['time_points'], params['time_points'][::-1]]),
                                    y=np.concatenate([mean + std, (mean - std)[::-1]]),
                                    fill='toself',
                                    fillcolor=f'rgba(0, 100, 255, 0.1)',
                                    line=dict(color='rgba(255,255,255,0)'),
                                    showlegend=False,
                                    name=f'{f} ± std'
                                ))
                    
                    fig_preds.update_layout(
                        title="Predicted Mean Values with Confidence Bands",
                        xaxis_title="Time (ns)",
                        yaxis_title="Value",
                        height=400,
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_preds, use_container_width=True)
                
                # Confidence plot
                if results['confidence_scores']:
                    fig_conf = st.session_state.enhanced_viz.create_confidence_plot(results, params['time_points'])
                    st.plotly_chart(fig_conf, use_container_width=True)
                
                # Statistics summary
                st.markdown("##### 📊 Prediction Statistics")
                if 'confidence_scores' in results:
                    conf_scores = results['confidence_scores']
                    col1, col2, col3 = st.columns(3)
                    with col1: st.metric("Avg Confidence", f"{np.mean(conf_scores):.3f}")
                    with col2: st.metric("Min Confidence", f"{np.min(conf_scores):.3f}")
                    with col3: st.metric("Max Confidence", f"{np.max(conf_scores):.3f}")

    # --- TAB 3: Polar Radar ---
    with tabs[2]:
        st.subheader("🎯 Polar Radar Visualization")
        
        # Field and timestep selection
        all_fields = set()
        for s in st.session_state.summaries:
            all_fields.update(s['field_stats'].keys())
        
        col1, col2, col3 = st.columns(3)
        with col1:
            field_type = st.selectbox("Field Type", sorted(all_fields), key="polar_field")
        with col2:
            max_timestep = max(s.get('timesteps', [1])[-1] for s in st.session_state.summaries)
            t_step = st.number_input("Timestep Index", 1, max_timestep, 1, key="polar_timestep")
        with col3:
            show_target = st.checkbox("Show Target Query", value=True, key="polar_show_target")
            show_legend = st.checkbox("Show Legend", value=True, key="polar_legend")
        
        # Build DataFrame for polar chart
        rows = []
        for s in st.session_state.summaries:
            if t_step <= len(s['timesteps']):
                idx = t_step - 1
                peak = s['field_stats'].get(field_type, {}).get('max', [0])
                if idx < len(peak):
                    rows.append({
                        'Name': s['name'],
                        'Energy': s['energy'],
                        'Duration': s['duration'],
                        'Peak_Value': peak[idx]
                    })
        
        df_polar = pd.DataFrame(rows)
        
        # Warning for degenerate energy range
        if df_polar['Energy'].nunique() <= 1:
            st.info("ℹ️ All loaded simulations use the same energy. The polar chart will show a collapsed angular axis.")
        
        # Get query params if showing target
        query_params = None
        if show_target and st.session_state.get('polar_query'):
            query_params = st.session_state.polar_query
        
        # Create polar radar chart
        fig_polar = st.session_state.polar_viz.create_polar_radar_chart(
            df_polar, field_type, query_params, timestep=t_step,
            show_legend=show_legend
        )
        st.plotly_chart(fig_polar, use_container_width=True)
        
        # Multi-timestep comparison
        if st.checkbox("Show Multiple Timesteps", key="polar_multi"):
            st.markdown("##### Multi-Timestep Comparison")
            t_indices = st.multiselect("Select Timesteps", [1, 2, 3, 4, 5], default=[1, 3, 5])
            
            if t_indices:
                cols = st.columns(len(t_indices))
                for col, t_idx in zip(cols, t_indices):
                    if t_idx <= max(len(s.get('timesteps', [])) for s in st.session_state.summaries):
                        # Build data for this timestep
                        rows_t = []
                        for s in st.session_state.summaries:
                            if t_idx <= len(s['timesteps']):
                                peak_t = s['field_stats'].get(field_type, {}).get('max', [0])
                                if t_idx-1 < len(peak_t):
                                    rows_t.append({
                                        'Name': s['name'],
                                        'Energy': s['energy'],
                                        'Duration': s['duration'],
                                        'Peak_Value': peak_t[t_idx-1]
                                    })
                        
                        fig_t = st.session_state.polar_viz.create_polar_radar_chart(
                            pd.DataFrame(rows_t), field_type, query_params,
                            timestep=t_idx, width=300, height=300, show_legend=(t_idx == t_indices[0])
                        )
                        with col:
                            st.plotly_chart(fig_t, use_container_width=True)
                            st.caption(f"Timestep {t_idx}")

    # --- TAB 4: Sankey Diagram ---
    with tabs[3]:
        st.subheader("🕸️ ST-DGPA Sankey Diagram")
        
        if st.session_state.interpolation_results and st.session_state.get('interpolation_params'):
            res = st.session_state.interpolation_results
            params = st.session_state.interpolation_params
            q_E = params['energy_query']
            q_τ = params['duration_query']
            
            # Timestep selector
            t_sel = st.slider("Select Timestep for Sankey", 1, len(res['attention_maps']), 1, key="sankey_timestep")
            t_idx = t_sel - 1
            
            # Prepare sources data for Sankey
            sources_data = []
            weights = res['attention_maps'][t_idx]
            phys_att = res['physics_attention_maps'][t_idx]
            gating = res['ett_gating_maps'][t_idx]
            
            # Map weights back to sources using metadata
            for i in range(len(st.session_state.extrapolator.source_metadata)):
                meta = st.session_state.extrapolator.source_metadata[i]
                if meta['timestep_idx'] == t_idx and i < len(weights):
                    sources_data.append({
                        'Energy': meta['energy'],
                        'Duration': meta['duration'],
                        'Time': meta['time'],
                        'Attention_Score': float(phys_att[i]) if i < len(phys_att) else 0,
                        'Gating': float(gating[i]) if i < len(gating) else 0,
                        'Refinement': float(phys_att[i] * gating[i]) if i < len(phys_att) and i < len(gating) else 0,
                        'Combined_Weight': float(weights[i]) if i < len(weights) else 0
                    })
            
            if sources_data:
                query = {'Energy': q_E, 'Duration': q_τ, 'Time': t_sel}
                customization = {
                    'font_size': 11,
                    'node_thickness': 15,
                    'show_math': st.checkbox("Show Math Formulas", value=True, key="sankey_math")
                }
                
                fig_sankey = st.session_state.sankey_viz.create_stdgpa_sankey(
                    sources_data, query, customization
                )
                st.plotly_chart(fig_sankey, use_container_width=True)
                
                st.info("💡 Hover over flows to see mathematical formulas and weight breakdown.")
                
                # Export options
                with st.expander("💾 Export Sankey Diagram"):
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Export as PNG"):
                            img_bytes = st.session_state.export_manager.export_plotly_figure(
                                fig_sankey, format='png'
                            )
                            if img_bytes:
                                st.download_button(
                                    label="⬇️ Download PNG",
                                    data=img_bytes,
                                    file_name=f"sankey_{q_E:.1f}mJ_{q_τ:.1f}ns_t{t_sel}.png",
                                    mime="image/png"
                                )
                    with col2:
                        if st.button("Export as HTML"):
                            html_bytes = st.session_state.export_manager.export_plotly_figure(
                                fig_sankey, format='html'
                            )
                            if html_bytes:
                                st.download_button(
                                    label="⬇️ Download HTML",
                                    data=html_bytes,
                                    file_name=f"sankey_{q_E:.1f}mJ_{q_τ:.1f}ns_t{t_sel}.html",
                                    mime="text/html"
                                )
            else:
                st.warning("⚠️ No source data matches the selected timestep. Try a different timestep.")
        else:
            st.info("ℹ️ Please run an interpolation first (Tab 2) to see the Sankey diagram.")

    # --- TAB 5: ST-DGPA Analysis ---
    with tabs[4]:
        st.subheader("🧠 ST-DGPA Attention & Physics Analysis")
        
        if st.session_state.interpolation_results and st.session_state.get('interpolation_params'):
            res = st.session_state.interpolation_results
            params = st.session_state.interpolation_params
            
            fig_stdgpa = st.session_state.enhanced_viz.create_stdgpa_analysis(
                res, params['energy_query'], params['duration_query'], params['time_points']
            )
            
            if fig_stdgpa:
                st.plotly_chart(fig_stdgpa, use_container_width=True)
                
                # Export options
                with st.expander("💾 Export Analysis"):
                    if st.button("Export as PNG"):
                        img_bytes = st.session_state.export_manager.export_plotly_figure(
                            fig_stdgpa, format='png'
                        )
                        if img_bytes:
                            st.download_button(
                                label="⬇️ Download PNG",
                                data=img_bytes,
                                file_name=f"stdgpa_analysis_{params['energy_query']:.1f}mJ.png",
                                mime="image/png"
                            )
            else:
                st.info("ℹ️ No attention data available for analysis.")
        else:
            st.info("ℹ️ Please run an interpolation first (Tab 2) to see ST-DGPA analysis.")

    # --- TAB 6: Export ---
    with tabs[5]:
        st.subheader("💾 Export Results")
        
        if st.session_state.interpolation_results and st.session_state.get('interpolation_params'):
            results = st.session_state.interpolation_results
            params = st.session_state.interpolation_params
            
            st.markdown("### Export Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Export as JSON
                export_data = {
                    'metadata': {
                        'generated_at': datetime.now().isoformat(),
                        'query_params': params,
                        'stdgpa_params': {
                            'sigma_g': st.session_state.extrapolator.sigma_g,
                            's_E': st.session_state.extrapolator.s_E,
                            's_tau': st.session_state.extrapolator.s_tau,
                            's_t': st.session_state.extrapolator.s_t,
                            'temporal_weight': st.session_state.extrapolator.temporal_weight
                        }
                    },
                    'results': results
                }
                json_str, json_name = st.session_state.export_manager.export_to_json(export_data)
                st.download_button(
                    label="📥 Download as JSON",
                    data=json_str,
                    file_name=json_name,
                    mime="application/json",
                    use_container_width=True
                )
            
            with col2:
                # Export as CSV
                if results['field_predictions']:
                    # Create DataFrame from predictions
                    csv_data = []
                    for idx, t in enumerate(params['time_points']):
                        row = {'Time (ns)': t}
                        for field in results['field_predictions']:
                            if results['field_predictions'][field]['mean']:
                                row[f'{field}_mean'] = results['field_predictions'][field]['mean'][idx]
                                row[f'{field}_max'] = results['field_predictions'][field]['max'][idx]
                                row[f'{field}_std'] = results['field_predictions'][field]['std'][idx]
                        if idx < len(results['confidence_scores']):
                            row['confidence'] = results['confidence_scores'][idx]
                        csv_data.append(row)
                    
                    if csv_data:
                        df_csv = pd.DataFrame(csv_data)
                        csv_str, csv_name = st.session_state.export_manager.export_to_csv(df_csv)
                        st.download_button(
                            label="📥 Download as CSV",
                            data=csv_str,
                            file_name=csv_name,
                            mime="text/csv",
                            use_container_width=True
                        )
            
            with col3:
                # Export attention weights
                if results['attention_maps']:
                    attention_df = pd.DataFrame(results['attention_maps'])
                    att_csv, att_name = st.session_state.export_manager.export_to_csv(
                        attention_df, filename="attention_weights.csv"
                    )
                    st.download_button(
                        label="📥 Download Attention Weights",
                        data=att_csv,
                        file_name=att_name,
                        mime="text/csv",
                        use_container_width=True
                    )
            
            st.markdown("---")
            st.markdown("### Export Visualizations")
            
            # List available visualizations to export
            viz_options = []
            if st.session_state.get('polar_query'):
                viz_options.append("Polar Radar Chart")
            if results.get('attention_maps'):
                viz_options.append("Sankey Diagram")
            if results.get('attention_maps') and len(results['attention_maps']) > 0:
                viz_options.append("ST-DGPA Analysis")
            
            if viz_options:
                selected_viz = st.multiselect("Select Visualizations to Export", viz_options)
                
                for viz in selected_viz:
                    if viz == "Polar Radar Chart" and st.session_state.get('polar_query'):
                        # Re-create polar chart for export
                        field_type = st.session_state.get('polar_field', 'temperature')
                        t_step = st.session_state.get('polar_timestep', 1)
                        # ... (rebuild df_polar as in Tab 3)
                        st.info(f"Polar Radar export ready for {field_type} at timestep {t_step}")
                    
                    elif viz == "Sankey Diagram" and results.get('attention_maps'):
                        st.info("Sankey Diagram export available in Tab 4")
                    
                    elif viz == "ST-DGPA Analysis" and results.get('attention_maps'):
                        st.info("ST-DGPA Analysis export available in Tab 5")
        else:
            st.info("ℹ️ Please run an interpolation first (Tab 2) to enable export.")

if __name__ == "__main__":
    main()
