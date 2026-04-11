#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ST-DGPA Sankey Visualizer for Pulsed Laser Soldering Simulations
================================================================
Interactive visualization of Spatio-Temporal Gated Physics Attention (ST-DGPA) weights
with full customization options and mathematical explanations on hover.

Features:
- Load FEA simulation data from VTU files
- ST-DGPA interpolation/extrapolation with (E, τ, t) gating
- Interactive Sankey diagram with hover explanations of mathematical formulas
- Full customization: colors, labels, fonts, node sizes, figure dimensions
- Heat transfer physics integration (Fourier number, thermal penetration)
- Cache management for efficient 3D field interpolation
- Export capabilities (VTU, NPZ, JSON, CSV)
"""

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
from scipy.interpolate import griddata, RBFInterpolator, interp1d, CubicSpline, RegularGridInterpolator
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from scipy.spatial.distance import cdist
import tempfile
import base64
import hashlib
import pickle
import json
from functools import lru_cache
from collections import OrderedDict
from dataclasses import dataclass, field
from math import pi, cos, sin
import itertools
import threading
import shutil
import time

warnings.filterwarnings('ignore')

# =============================================
# PATH CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEA_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "fea_solutions")
VISUALIZATION_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "visualization_outputs")
TEMP_ANIMATION_DIR = os.path.join(SCRIPT_DIR, "temp_animations")

os.makedirs(FEA_SOLUTIONS_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_ANIMATION_DIR, exist_ok=True)

# =============================================
# CACHE MANAGEMENT UTILITIES
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
    def get_cached_interpolation(field_name: str, timestep_idx: int, params: Dict) -> Optional[Dict]:
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
            params.get('s_t', 20.0),
            params.get('temporal_weight', 0.3),
            params.get('top_k'),
            params.get('subsample_factor')
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
            params.get('s_t', 20.0),
            params.get('temporal_weight', 0.3),
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
            oldest_key = min(
                st.session_state.interpolation_3d_cache.keys(),
                key=lambda k: st.session_state.interpolation_3d_cache[k]['timestamp']
            )
            del st.session_state.interpolation_3d_cache[oldest_key]
        
        # Limit history size
        if len(st.session_state.interpolation_field_history) > 10:
            st.session_state.interpolation_field_history.popitem(last=False)

# =============================================
# UNIFIED DATA LOADER WITH ENHANCED CAPABILITIES
# =============================================
class UnifiedFEADataLoader:
    """Enhanced data loader with comprehensive field extraction from VTU files"""
    
    def __init__(self):
        self.simulations = {}
        self.summaries = []
        self.field_statistics = {}
        self.available_fields = set()
    
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
                        # Scalar field
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
    
    def __init__(self, sigma_param: float = 0.3, spatial_weight: float = 0.5, 
                 n_heads: int = 4, temperature: float = 1.0,
                 sigma_g: float = 0.20, s_E: float = 10.0, s_tau: float = 5.0, 
                 s_t: float = 20.0, temporal_weight: float = 0.3):
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
        self.thermal_diffusivity = 1e-5  # m²/s, typical for metals
        self.laser_spot_radius = 50e-6   # m, typical laser spot size
        self.characteristic_length = 100e-6  # m, characteristic length for heat diffusion
        
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
                    'fourier_number': self._compute_fourier_number(t),
                    'thermal_penetration': self._compute_thermal_penetration(t)
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
    
    def _compute_fourier_number(self, time_ns: float) -> float:
        """Compute Fourier number (Fo = αt/L²) for heat transfer characterization"""
        time_s = time_ns * 1e-9  # Convert ns to s
        Fo = self.thermal_diffusivity * time_s / (self.characteristic_length ** 2)
        return Fo
    
    def _compute_thermal_penetration(self, time_ns: float) -> float:
        """Compute thermal penetration depth (δ ~ √(αt))"""
        time_s = time_ns * 1e-9  # Convert ns to s
        penetration = np.sqrt(self.thermal_diffusivity * time_s) * 1e6  # Convert to μm
        return penetration
    
    def _compute_enhanced_physics_embedding(self, energy: float, duration: float, time: float) -> np.ndarray:
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
            logE, duration, time, power, energy_density, time_ratio,
            heating_rate, cooling_rate, thermal_diffusion, thermal_penetration,
            strain_rate, stress_rate, fourier_number, thermal_penetration_depth,
            diffusion_time_scale, heating_phase, cooling_phase, early_time, late_time,
            np.log1p(power), np.log1p(time), np.sqrt(time),  # Square root for diffusion scaling
            time / (duration + 1e-6),  # Normalized time
        ], dtype=np.float32)
    
    def _compute_ett_gating(self, energy_query: float, duration_query: float, 
                           time_query: float, source_metadata: Optional[List[Dict]] = None) -> np.ndarray:
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
                time_scaling_factor = 1.0 + 0.5 * (time_query / max(duration_query, 1e-6))
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
    
    def _compute_temporal_similarity(self, query_meta: Dict, source_metas: List[Dict]) -> np.ndarray:
        """Compute temporal similarity with physics-aware weighting for heat transfer"""
        similarities = []
        for meta in source_metas:
            time_diff = abs(query_meta['time'] - meta['time'])
            
            # Physics-aware temporal similarity:
            # 1. For heating phase (t < τ): tight matching
            # 2. For cooling phase (t > τ): looser matching
            if query_meta['time'] < query_meta['duration'] * 1.5:  # Heating/early cooling
                temporal_tolerance = max(query_meta['duration'] * 0.1, 1.0)
            else:  # Late cooling/diffusion phase
                temporal_tolerance = max(query_meta['duration'] * 0.3, 3.0)
            
            # Fourier number similarity for heat transfer characterization
            if 'fourier_number' in meta and 'fourier_number' in query_meta:
                fourier_diff = abs(query_meta['fourier_number'] - meta['fourier_number'])
                fourier_similarity = np.exp(-fourier_diff / 0.1)
            else:
                fourier_similarity = 1.0
            
            # Combine time difference and Fourier number similarity
            time_similarity = np.exp(-time_diff / temporal_tolerance)
            combined_similarity = (1 - self.temporal_weight) * time_similarity + \
                                self.temporal_weight * fourier_similarity
            similarities.append(combined_similarity)
        
        return np.array(similarities)
    
    def _compute_spatial_similarity(self, query_meta: Dict, source_metas: List[Dict]) -> np.ndarray:
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
    
    def _multi_head_attention_with_gating(self, query_embedding: np.ndarray, 
                                         query_meta: Dict) -> Tuple[Optional[np.ndarray], 
                                                                   Optional[np.ndarray],
                                                                   Optional[np.ndarray],
                                                                   Optional[np.ndarray]]:
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
    
    def predict_field_statistics(self, energy_query: float, duration_query: float, 
                                time_query: float) -> Optional[Dict]:
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
        prediction, final_weights, physics_attention, ett_gating = self._multi_head_attention_with_gating(
            query_embedding, query_meta)
        
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
    
    def _compute_temporal_confidence(self, time_query: float, duration_query: float) -> float:
        """Compute confidence in temporal prediction based on heat transfer physics"""
        if time_query < duration_query * 0.5:  # Early heating phase
            return 0.6  # Moderate confidence
        elif time_query < duration_query * 1.5:  # Peak and early cooling
            return 0.8  # Higher confidence
        else:  # Late cooling/diffusion
            return 0.9  # Highest confidence (diffusion-dominated)
    
    def _compute_heat_transfer_indicators(self, energy: float, duration: float, time: float) -> Dict:
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
    
    def interpolate_full_field(self, field_name: str, attention_weights: np.ndarray,
                              source_metadata: List[Dict], simulations: Dict) -> Optional[np.ndarray]:
        """Compute interpolated full field using ST-DGPA weights"""
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
    
    def _assess_temporal_coherence(self, source_metadata: List[Dict], 
                                  attention_weights: np.ndarray) -> float:
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
    def create_stdgpa_analysis(results: Dict, energy_query: float, duration_query: float, 
                            time_points: np.ndarray) -> Optional[go.Figure]:
        """Create ST-DGPA-specific analysis visualizations"""
        if not results or 'attention_maps' not in results or len(results['attention_maps']) == 0:
            return None
        
        # Select a timestep for detailed analysis (middle point)
        timestep_idx = len(time_points) // 2
        time = time_points[timestep_idx]
        
        # Create comprehensive ST-DGPA analysis plot
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
                [{'type': 'xy'}, {'type': 'xy'}, {'type': 'polar'}],
                [{'type': 'scene'}, {'type': 'xy'}, {'type': 'xy'}]
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
            times = []
            weights = []
            for i, weight in enumerate(final_weights):
                if weight > 0.01:
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
        
        # 6. Heat transfer phase indicator
        if 'heat_transfer_indicators' in results and results['heat_transfer_indicators']:
            indicators = results['heat_transfer_indicators'][timestep_idx]
            if indicators:
                categories = ['Heating', 'Cooling', 'Diffusion', 'Adiabatic']
                if 'phase' in indicators:
                    phase = indicators['phase']
                    if phase == 'Early Heating' or phase == 'Heating':
                        values = [0.9, 0.3, 0.2, 0.1]
                    elif phase == 'Early Cooling':
                        values = [0.4, 0.8, 0.3, 0.1]
                    elif phase == 'Diffusion Cooling':
                        values = [0.2, 0.5, 0.9, 0.2]
                    else:
                        values = [0.7, 0.5, 0.3, 0.2]
                else:
                    values = [0.7, 0.5, 0.3, 0.2]
                
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
            for i, summary in enumerate(st.session_state.summaries[:10]):
                for t_idx, t in enumerate(summary['timesteps'][:5]):
                    energies.append(summary['energy'])
                    durations.append(summary['duration'])
                    times.append(t)
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
        
        # 9. Weight evolution over time
        if len(results['attention_maps']) > 1:
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
            radialaxis=dict(visible=True, range=[0, 1]),
            angularaxis=dict(direction="clockwise"),
            row=2, col=3
        )
        
        # Update 3D scene
        fig.update_scenes(
            xaxis_title="Energy (mJ)",
            yaxis_title="Duration (ns)",
            zaxis_title="Time (ns)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            row=3, col=1
        )
        
        fig.update_xaxes(title_text="Node", row=3, col=2)
        fig.update_yaxes(title_text="", showticklabels=False, row=3, col=2)
        fig.update_xaxes(title_text="Time (ns)", row=3, col=3)
        fig.update_yaxes(title_text="Weight", row=3, col=3)
        
        return fig
    
    @staticmethod
    def create_temporal_analysis(results: Dict, time_points: np.ndarray) -> Optional[go.Figure]:
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

# =============================================
# SANKEY DIAGRAM VISUALIZATION WITH MATHEMATICAL EXPLANATIONS
# =============================================
class SankeyVisualizer:
    """Specialized Sankey diagram visualizer for ST-DGPA weight analysis"""
    
    def __init__(self):
        self.default_node_colors = {
            'target': '#FF6B6B',
            'source': '#9966FF',
            'components': ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFD93D', '#36A2EB', '#9966FF']
        }
        self.default_font = {
            'family': 'Arial, sans-serif',
            'size_title': 20,
            'size_labels': 14,
            'size_hover': 12
        }
    
    def create_stdgpa_sankey(self, sources_data: List[Dict], query: Dict,
                           customization: Optional[Dict] = None) -> go.Figure:
        """
        Create Sankey diagram showing flow from Sources → Weight Components → Target
        with interactive mathematical explanations on hover.
        
        Parameters:
        -----------
        sources_data : List[Dict]
            List of source simulation data with computed weights
        query : Dict
            Query parameters (Energy, Duration, Time)
        customization : Dict, optional
            Customization options for colors, fonts, labels, etc.
        """
        # Apply customizations or use defaults
        if customization is None:
            customization = {}
        
        node_colors = customization.get('node_colors', self.default_node_colors.copy())
        font_family = customization.get('font_family', self.default_font['family'])
        font_size = customization.get('font_size', self.default_font['size_labels'])
        node_thickness = customization.get('node_thickness', 20)
        node_pad = customization.get('node_pad', 15)
        show_math = customization.get('show_math_explanations', True)
        component_labels = customization.get('component_labels', None)
        target_label = customization.get('target_label', 'TARGET')
        width = customization.get('width', 1000)
        height = customization.get('height', 700)
        
        # Default component labels with mathematical formulas
        if component_labels is None:
            if show_math:
                component_labels = [
                    'Energy Gate\nφ² = ((E*-Eᵢ)/s_E)²',
                    'Duration Gate\nφ² = ((τ*-τᵢ)/s_τ)²',
                    'Time Gate\nφ² = ((t*-tᵢ)/s_t)²',
                    'Attention\nαᵢ = softmax(Q·Kᵀ/√dₖ)',
                    'Refinement\nwᵢ ∝ αᵢ × gatingᵢ',
                    'Combined\nwᵢ = αᵢ·gatingᵢ / Σ(αⱼ·gatingⱼ)'
                ]
            else:
                component_labels = [
                    'Energy Gate', 'Duration Gate', 'Time Gate',
                    'Attention', 'Refinement', 'Combined'
                ]
        
        # Build node labels
        labels = [target_label]
        node_colors_list = [node_colors['target']]
        
        # Add Sources with detailed hover info
        n_sources = len(sources_data)
        for i in range(n_sources):
            row = sources_data[i]
            source_label = (
                f"Sim {i+1}\n"
                f"E: {row['Energy']:.2f} mJ\n"
                f"τ: {row['Duration']:.2f} ns\n"
                f"t: {row['Time']:.2f} ns\n"
                f"w: {row['Combined_Weight']:.3f}"
            )
            labels.append(source_label)
            # Scale opacity by weight for visual emphasis
            weight = row['Combined_Weight']
            opacity = min(0.3 + weight * 0.7, 1.0)
            node_colors_list.append(f'rgba(153,102,255,{opacity:.2f})')
        
        # Add Components
        component_start = len(labels)
        labels.extend(component_labels)
        node_colors_list.extend(node_colors['components'])
        
        # Link Construction
        sources_idx, targets_idx, values, link_colors_list, hover_texts = [], [], [], [], []
        
        # 1. Sources -> Components (The decomposition)
        for i in range(n_sources):
            src_idx = i + 1  # +1 because target is at index 0
            row = sources_data[i]
            
            # Calculate individual contributions for visualization (scaled for visual clarity)
            val_e = ((row['Energy'] - query['Energy']) / 10.0)**2 * 10
            val_tau = ((row['Duration'] - query['Duration']) / 5.0)**2 * 10
            val_t = ((row['Time'] - query['Time']) / 20.0)**2 * 10
            val_attn = row['Attention_Score'] * 100
            val_ref = row['Refinement'] * 100
            val_comb = row['Combined_Weight'] * 100
            
            # Energy Gate link with mathematical explanation
            sources_idx.append(src_idx)
            targets_idx.append(component_start + 0)
            values.append(max(0.01, val_e))
            link_colors_list.append('rgba(255,107,107,0.5)')
            hover_texts.append(
                f"<b>Energy Gate Contribution</b><br>"
                f"Source: Sim {i+1}<br>"
                f"Formula: φ² = ((E* - Eᵢ) / s_E)²<br>"
                f"E* = {query['Energy']:.2f} mJ, Eᵢ = {row['Energy']:.2f} mJ<br>"
                f"s_E = 10.0 mJ (scaling factor)<br>"
                f"Contribution: {val_e:.3f}"
            )
            
            # Duration Gate link
            sources_idx.append(src_idx)
            targets_idx.append(component_start + 1)
            values.append(max(0.01, val_tau))
            link_colors_list.append('rgba(78,205,196,0.5)')
            hover_texts.append(
                f"<b>Duration Gate Contribution</b><br>"
                f"Source: Sim {i+1}<br>"
                f"Formula: φ² = ((τ* - τᵢ) / s_τ)²<br>"
                f"τ* = {query['Duration']:.2f} ns, τᵢ = {row['Duration']:.2f} ns<br>"
                f"s_τ = 5.0 ns (scaling factor)<br>"
                f"Contribution: {val_tau:.3f}"
            )
            
            # Time Gate link
            sources_idx.append(src_idx)
            targets_idx.append(component_start + 2)
            values.append(max(0.01, val_t))
            link_colors_list.append('rgba(149,225,211,0.5)')
            hover_texts.append(
                f"<b>Time Gate Contribution</b><br>"
                f"Source: Sim {i+1}<br>"
                f"Formula: φ² = ((t* - tᵢ) / s_t)²<br>"
                f"t* = {query['Time']:.2f} ns, tᵢ = {row['Time']:.2f} ns<br>"
                f"s_t = 20.0 ns (scaling factor)<br>"
                f"Contribution: {val_t:.3f}"
            )
            
            # Attention link
            sources_idx.append(src_idx)
            targets_idx.append(component_start + 3)
            values.append(max(0.01, val_attn))
            link_colors_list.append('rgba(255,217,61,0.5)')
            hover_texts.append(
                f"<b>Physics Attention Score</b><br>"
                f"Source: Sim {i+1}<br>"
                f"Formula: αᵢ = softmax(Q·Kᵀ / √dₖ)<br>"
                f"Q = query embedding, K = source key embeddings<br>"
                f"dₖ = embedding dimension, T = temperature<br>"
                f"Score: {row['Attention_Score']:.4f}"
            )
            
            # Refinement link
            sources_idx.append(src_idx)
            targets_idx.append(component_start + 4)
            values.append(max(0.01, val_ref))
            link_colors_list.append('rgba(54,162,235,0.5)')
            hover_texts.append(
                f"<b>Physics Refinement</b><br>"
                f"Source: Sim {i+1}<br>"
                f"Formula: wᵢ ∝ αᵢ × gatingᵢ<br>"
                f"gatingᵢ = exp(-φ² / (2σ_g²))<br>"
                f"σ_g = 0.20 (gating width)<br>"
                f"Refinement: {row['Refinement']:.4f}"
            )
            
            # Combined Weight link
            sources_idx.append(src_idx)
            targets_idx.append(component_start + 5)
            values.append(max(0.01, val_comb))
            link_colors_list.append('rgba(153,102,255,0.5)')
            hover_texts.append(
                f"<b>Final Combined Weight</b><br>"
                f"Source: Sim {i+1}<br>"
                f"Formula: wᵢ = (αᵢ × gatingᵢ) / Σⱼ(αⱼ × gatingⱼ)<br>"
                f"This is the normalized weight used for interpolation<br>"
                f"Weight: {row['Combined_Weight']:.4f} ({row['Combined_Weight']*100:.2f}%)"
            )
        
        # 2. Components -> Target (Aggregation)
        for comp_idx in range(len(component_labels)):
            sources_idx.append(component_start + comp_idx)
            targets_idx.append(0)  # Target index
            
            # Calculate total flow INTO this component from all sources
            flow_in = sum(
                v for s, t, v in zip(sources_idx[:-len(component_labels)], 
                                   targets_idx[:-len(component_labels)], 
                                   values[:-len(component_labels)])
                if t == component_start + comp_idx
            )
            
            # Scale for visual balance (damping factor)
            values.append(flow_in * 0.5)
            link_colors_list.append('rgba(153,102,255,0.6)')
            hover_texts.append(
                f"<b>Component Aggregation</b><br>"
                f"{component_labels[comp_idx]} → {target_label}<br>"
                f"Total contribution from all sources: {flow_in:.3f}<br>"
                f"Scaled for visualization: {flow_in * 0.5:.3f}"
            )
        
        # Create Sankey figure
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=node_pad,
                thickness=node_thickness,
                line=dict(color="black", width=0.5),
                label=labels,
                color=node_colors_list,
                hovertemplate='<b>%{label}</b><br>Value: %{value:.3f}<extra></extra>',
                font=dict(family=font_family, size=font_size)
            ),
            link=dict(
                source=sources_idx,
                target=targets_idx,
                value=values,
                color=link_colors_list,
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=hover_texts,
                line=dict(width=0.5, color='rgba(255,255,255,0.3)')
            ),
            hoverinfo='text'
        )])
        
        # Update layout with customizations
        title_text = (
            f"<b>ST-DGPA Attention Flow</b><br>"
            f"Query: E={query['Energy']:.2f} mJ, τ={query['Duration']:.2f} ns, t={query['Time']:.2f} ns<br>"
            f"<sub>σ_g=0.20, s_E=10.0, s_τ=5.0, s_t=20.0</sub>"
        )
        
        fig.update_layout(
            title=dict(
                text=title_text,
                font=dict(family=font_family, size=font_size+4),
                x=0.5,
                xanchor='center'
            ),
            font=dict(family=font_family, size=font_size),
            width=width,
            height=height,
            plot_bgcolor='rgba(240, 240, 245, 0.9)',
            paper_bgcolor='white',
            margin=dict(t=100, l=50, r=50, b=50),
            hoverlabel=dict(
                font=dict(family=font_family, size=customization.get('hover_font_size', 12)),
                bgcolor='rgba(44, 62, 80, 0.9)',
                bordercolor='white',
                namelength=-1  # Show full text
            )
        )
        
        return fig

# =============================================
# MAIN STREAMLIT APPLICATION
# =============================================
def main():
    st.set_page_config(
        page_title="ST-DGPA Sankey Visualizer for Pulsed Laser Soldering",
        layout="wide",
        page_icon="🔬"
    )
    
    # Custom CSS for enhanced visual appeal
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #1E88E5, #4A00E0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 800;
    }
    .sub-header {
        font-size: 1.5rem;
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
    .stdgpa-box {
        background: linear-gradient(135deg, #f093fb 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
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
    
    st.markdown('<h1 class="main-header">🔬 ST-DGPA Sankey Visualizer</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = UnifiedFEADataLoader()
    if 'extrapolator' not in st.session_state:
        st.session_state.extrapolator = SpatioTemporalGatedPhysicsAttentionExtrapolator(
            sigma_param=0.3, spatial_weight=0.5, n_heads=4, temperature=1.0,
            sigma_g=0.20, s_E=10.0, s_tau=5.0, s_t=20.0, temporal_weight=0.3
        )
    if 'sankey_visualizer' not in st.session_state:
        st.session_state.sankey_visualizer = SankeyVisualizer()
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'interpolation_results' not in st.session_state:
        st.session_state.interpolation_results = None
    if 'interpolation_params' not in st.session_state:
        st.session_state.interpolation_params = None
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # ST-DGPA parameters
        st.markdown("#### ST-DGPA Parameters")
        sigma_g = st.slider("Gating Width (σ_g)", 0.05, 1.0, 0.20, 0.05,
                          help="Controls sharpness of the (E, τ, t) gating kernel")
        s_E = st.slider("Energy Scale (s_E) [mJ]", 0.1, 50.0, 10.0, 0.5,
                       help="Scaling factor for energy differences in gating kernel")
        s_tau = st.slider("Duration Scale (s_τ) [ns]", 0.1, 20.0, 5.0, 0.5,
                         help="Scaling factor for duration differences in gating kernel")
        s_t = st.slider("Time Scale (s_t) [ns]", 1.0, 50.0, 20.0, 1.0,
                       help="Scaling factor for time differences in gating kernel")
        
        # Update extrapolator parameters
        st.session_state.extrapolator.sigma_g = sigma_g
        st.session_state.extrapolator.s_E = s_E
        st.session_state.extrapolator.s_tau = s_tau
        st.session_state.extrapolator.s_t = s_t
        
        st.markdown("---")
        
        # Sankey visualization customization
        st.markdown("#### Sankey Visualization")
        
        # Font settings
        font_family = st.selectbox("Font Family",
                                  ["Arial, sans-serif", "Courier New, monospace", 
                                   "Times New Roman, serif", "Verdana, sans-serif"],
                                  index=0)
        font_size = st.slider("Font Size", 8, 20, 14, 1)
        
        # Node settings
        node_thickness = st.slider("Node Thickness", 10, 40, 20, 2)
        node_pad = st.slider("Node Padding", 5, 30, 15, 1)
        
        # Figure size
        fig_width = st.slider("Figure Width", 600, 1400, 1000, 50)
        fig_height = st.slider("Figure Height", 400, 1000, 700, 50)
        
        # Color customization
        st.markdown("**Node Colors**")
        col1, col2 = st.columns(2)
        with col1:
            target_color = st.color_picker("Target Node", "#FF6B6B")
        with col2:
            source_base_color = st.color_picker("Source Base Color", "#9966FF")
        
        st.markdown("**Component Colors**")
        comp_colors = st.color_picker("Component Base Color", "#4ECDC4")
        
        # Toggle mathematical explanations
        show_math = st.checkbox("Show Mathematical Formulas on Hover", value=True,
                               help="Display ST-DGPA formulas when hovering over diagram elements")
        
        # Custom component labels
        st.markdown("**Component Labels**")
        use_custom_labels = st.checkbox("Use Custom Component Labels", value=False)
        if use_custom_labels:
            component_labels = []
            for i, default in enumerate(['Energy Gate', 'Duration Gate', 'Time Gate', 
                                        'Attention', 'Refinement', 'Combined']):
                label = st.text_input(f"Component {i+1}", value=default, key=f"comp_{i}")
                component_labels.append(label)
        else:
            component_labels = None
        
        st.markdown("---")
        
        # Data settings
        st.markdown("#### Data Settings")
        n_sims = st.slider("Number of Simulations", 2, 20, 4, 1,
                          help="Number of source simulations to generate for demo")
        
        if st.button("🔄 Generate Demo Data", use_container_width=True):
            np.random.seed(42)
            data = {
                'Energy': np.random.uniform(0.5, 8.0, n_sims),
                'Duration': np.random.uniform(2.0, 7.0, n_sims),
                'Time': np.random.uniform(1.0, 10.0, n_sims),
                'Max_Temp': np.random.uniform(500, 1500, n_sims)
            }
            st.session_state.df_sources = pd.DataFrame(data)
            st.success(f"Generated {n_sims} demo simulations!")
        
        uploaded_file = st.file_uploader("📁 Upload CSV", type=["csv"],
                                        help="CSV with columns: Energy, Duration, Time, Max_Temp")
        if uploaded_file is not None:
            try:
                st.session_state.df_sources = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(st.session_state.df_sources)} simulations from CSV!")
            except Exception as e:
                st.error(f"Error loading CSV: {e}")
        
        if 'df_sources' in st.session_state:
            st.markdown("---")
            st.markdown(f"**Loaded Data:** {len(st.session_state.df_sources)} simulations")
            with st.expander("📋 View Data"):
                st.dataframe(st.session_state.df_sources, use_container_width=True)
    
    # Main area: Query and visualization
    if 'df_sources' in st.session_state:
        df = st.session_state.df_sources
        
        st.subheader("🎯 Target Query Parameters")
        col1, col2, col3 = st.columns(3)
        with col1:
            q_energy = st.number_input("Energy (mJ)", 0.1, 20.0, 3.5, 0.1,
                                      help="Target pulse energy for interpolation")
        with col2:
            q_tau = st.number_input("Duration (ns)", 0.1, 20.0, 4.2, 0.1,
                                   help="Target pulse duration for interpolation")
        with col3:
            q_time = st.number_input("Time (ns)", 0.1, 20.0, 5.0, 0.1,
                                    help="Target observation time for interpolation")
        
        query = {'Energy': q_energy, 'Duration': q_tau, 'Time': q_time}
        
        # Prepare customization dictionary
        customization = {
            'font_family': font_family,
            'font_size': font_size,
            'node_thickness': node_thickness,
            'node_pad': node_pad,
            'width': fig_width,
            'height': fig_height,
            'show_math_explanations': show_math,
            'component_labels': component_labels,
            'node_colors': {
                'target': target_color,
                'source': source_base_color,
                'components': []
            }
        }
        
        # Generate component colors from base color
        comp_base = [int(comp_colors[1:3], 16), int(comp_colors[3:5], 16), int(comp_colors[5:7], 16)]
        for i in range(6):
            r = min(255, comp_base[0] + i*20)
            g = min(255, comp_base[1] + i*15)
            b = min(255, comp_base[2] + i*10)
            customization['node_colors']['components'].append(f'rgb({r},{g},{b})')
        
        if st.button("🚀 Compute ST-DGPA Interpolation", type="primary", use_container_width=True):
            with st.spinner("Computing Spatio-Temporal Gated Physics Attention weights..."):
                # Run interpolation
                results = st.session_state.extrapolator.predict_field_statistics(
                    q_energy, q_tau, q_time)
                
                if results:
                    # Prepare sources data for Sankey
                    sources_data = []
                    for i in range(len(df)):
                        row = df.iloc[i]
                        # Compute individual weight components for visualization
                        de = (row['Energy'] - q_energy) / s_E
                        dt = (row['Duration'] - q_tau) / s_tau
                        dtime = (row['Time'] - q_time) / s_t
                        phi_sq = de**2 + dt**2 + dtime**2
                        gating = np.exp(-phi_sq / (2 * sigma_g**2))
                        attention = 1 / (1 + np.sqrt(phi_sq))
                        refinement = gating * attention
                        
                        sources_data.append({
                            'Energy': row['Energy'],
                            'Duration': row['Duration'],
                            'Time': row['Time'],
                            'Gating_Weight': gating,
                            'Attention_Score': attention,
                            'Refinement': refinement,
                            'Combined_Weight': refinement / (refinement + 1e-12)  # Simplified normalization
                        })
                    
                    # Create Sankey diagram
                    fig_sankey = st.session_state.sankey_visualizer.create_stdgpa_sankey(
                        sources_data, query, customization)
                    
                    # Display results
                    st.markdown("### 📊 Top Contributing Sources")
                    top_sources = pd.DataFrame(sources_data).nlargest(5, 'Combined_Weight')[
                        ['Energy', 'Duration', 'Time', 'Combined_Weight', 'Max_Temp']
                    ]
                    st.dataframe(
                        top_sources.style.format({
                            'Energy': '{:.2f}',
                            'Duration': '{:.2f}',
                            'Time': '{:.2f}',
                            'Combined_Weight': '{:.4f}',
                            'Max_Temp': '{:.1f}'
                        }).highlight_max(axis=0, subset=['Combined_Weight'], color='#90EE90'),
                        use_container_width=True
                    )
                    
                    # Prediction example
                    predicted_temp = (pd.DataFrame(sources_data)['Combined_Weight'] * df['Max_Temp']).sum()
                    st.metric("Predicted Max Temperature", f"{predicted_temp:.2f} K")
                    
                    # Sankey Diagram
                    st.markdown("### 🕸️ Attention Weight Flow Diagram")
                    st.markdown("""
                    **How to read this diagram:**
                    - **Left nodes**: Source simulations (colored by their final weight)
                    - **Middle nodes**: Weight components (Energy Gate, Duration Gate, etc.)
                    - **Right node**: Target prediction (aggregated result)
                    - **Flow thickness**: Proportional to contribution magnitude
                    - **Hover over any element**: See mathematical formulas and detailed explanations
                    """)
                    
                    st.plotly_chart(fig_sankey, use_container_width=True)
                    
                    # Additional analysis
                    with st.expander("📈 Weight Distribution Analysis"):
                        # Plot weight distribution
                        fig_dist = go.Figure()
                        fig_dist.add_trace(go.Bar(
                            x=[f"Sim {i+1}" for i in range(len(sources_data))],
                            y=[s['Combined_Weight'] for s in sources_data],
                            marker_color=[customization['node_colors']['source']] * len(sources_data),
                            text=[f"{s['Combined_Weight']*100:.1f}%" for s in sources_data],
                            textposition='auto'
                        ))
                        fig_dist.update_layout(
                            title="Final Weight Distribution Across Sources",
                            xaxis_title="Source Simulation",
                            yaxis_title="Weight",
                            height=400
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)
                        
                        # Show parameter distances
                        st.markdown("**Parameter Distances from Query:**")
                        dist_df = pd.DataFrame({
                            'Source': [f"Sim {i+1}" for i in range(len(sources_data))],
                            'Energy Dist': np.abs(df['Energy'] - q_energy),
                            'Duration Dist': np.abs(df['Duration'] - q_tau),
                            'Time Dist': np.abs(df['Time'] - q_time),
                            'Combined Weight': [s['Combined_Weight'] for s in sources_data]
                        })
                        st.dataframe(
                            dist_df.style.format({
                                'Energy Dist': '{:.2f}',
                                'Duration Dist': '{:.2f}',
                                'Time Dist': '{:.2f}',
                                'Combined Weight': '{:.4f}'
                            }),
                            use_container_width=True
                        )
                    
                    # ST-DGPA formula explanation
                    with st.expander("📐 ST-DGPA Mathematical Formulas", expanded=True):
                        st.markdown("""
                        **Core ST-DGPA Weight Formula:**
                        ```
                        w_i(θ*) = [α_i(θ*) × gating_i(θ*)] / Σ_j[α_j(θ*) × gating_j(θ*)]
                        ```
                        
                        **Where:**
                        - `α_i(θ*)` = Physics attention from transformer-like mechanism
                        - `gating_i(θ*)` = exp(-φ_i² / (2σ_g²)) with φ_i² = Σ[(p*-p_i)/s_p]²
                        - `s_p` = scaling factor for parameter p ∈ {E, τ, t}
                        - `σ_g` = gating kernel width (controls sharpness)
                        
                        **Parameter Proximity Kernel:**
                        ```
                        φ_i = √[((E*-E_i)/s_E)² + ((τ*-τ_i)/s_τ)² + ((t*-t_i)/s_t)²]
                        ```
                        
                        **Physics Attention:**
                        ```
                        α_i = softmax(Q·Kᵀ / √dₖ)  [multi-head transformer attention]
                        ```
                        """)
                
                else:
                    st.error("Prediction failed. Please check input parameters.")
    
    else:
        st.info("👈 Please generate demo data or upload a CSV file in the sidebar to begin.")
        
        # Show example CSV format
        with st.expander("📋 Expected CSV Format"):
            st.markdown("""
            Your CSV file should have the following columns:
            ```csv
            Energy,Duration,Time,Max_Temp
            0.5,2.0,1.0,520.5
            1.2,3.5,2.5,680.3
            2.8,4.2,4.0,890.1
            ...
            ```
            
            **Column descriptions:**
            - `Energy`: Pulse energy in millijoules (mJ)
            - `Duration`: Pulse duration in nanoseconds (ns)
            - `Time`: Observation time in nanoseconds (ns)
            - `Max_Temp`: Maximum temperature in Kelvin (example field for prediction)
            """)

if __name__ == "__main__":
    main()
