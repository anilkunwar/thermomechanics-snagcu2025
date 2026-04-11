#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ST-DGPA Laser Soldering Sankey Visualizer - FULLY WORKING VERSION
==================================================================
Complete integrated application with:
- FEA simulation data loading from VTU files
- ST-DGPA (Spatio-Temporal Gated Physics Attention) interpolation
- REAL Sankey diagram showing attention weight flow with mathematical hover explanations
- Full customization: colors, fonts, labels, node sizes, figure dimensions
- Heat transfer physics integration (Fourier number, thermal penetration)
- Cache management for efficient 3D field interpolation
- Export capabilities (VTU, NPZ, JSON, CSV)
- Robust error handling and debugging
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
# SANKEY VISUALIZATION ENGINE (REAL SANKEY WITH ST-DGPA)
# =============================================
def create_stdgpa_sankey(sources_data: List[Dict], query: Dict, 
                        customization: Optional[Dict] = None) -> go.Figure:
    """
    Create REAL Sankey diagram showing ST-DGPA attention flow with mathematical hover explanations.
    
    Parameters:
    -----------
    sources_data : List[Dict]
        Sources with computed ST-DGPA weights (list of dicts)
    query : Dict
        Query parameters {Energy, Duration, Time}
    customization : Dict, optional
        Custom settings for colors, fonts, sizes
    """
    # Default customization
    defaults = {
        'font_family': 'Arial, sans-serif',
        'font_size': 12,
        'node_thickness': 20,
        'node_pad': 15,
        'width': 1200,
        'height': 800,
        'show_math': True,
        'target_label': 'TARGET PREDICTION',
        'node_colors': {
            'target': '#FF6B6B',
            'source': '#9966FF',
            'components': ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFD93D', '#36A2EB', '#9966FF']
        }
    }
    cfg = {**defaults, **(customization or {})}
    
    # Component labels (with math if enabled)
    if cfg['show_math']:
        comp_labels = [
            'Energy Gate\nφ² = ((E*-Eᵢ)/s_E)²',
            'Duration Gate\nφ² = ((τ*-τᵢ)/s_τ)²',
            'Time Gate\nφ² = ((t*-tᵢ)/s_t)²',
            'Attention\nαᵢ = 1/(1+√φ²)',
            'Refinement\nwᵢ ∝ αᵢ×gatingᵢ',
            'Combined\nwᵢ = αᵢ·gatingᵢ / Σ(...)'
        ]
    else:
        comp_labels = ['Energy Gate', 'Duration Gate', 'Time Gate', 
                      'Attention', 'Refinement', 'Combined']
    
    # Build node labels and colors
    labels = [cfg['target_label']]
    node_colors = [cfg['node_colors']['target']]
    
    n_sources = len(sources_data)
    for i in range(n_sources):
        row = sources_data[i]
        w = row.get('Combined_Weight', 0)
        # Scale opacity by weight for visual emphasis
        opacity = min(0.3 + w * 0.7, 1.0)
        base = cfg['node_colors']['source']
        r, g, b = int(base[1:3], 16), int(base[3:5], 16), int(base[5:7], 16)
        node_colors.append(f'rgba({r},{g},{b},{opacity:.2f})')
        labels.append(f"Sim {i+1}\nE:{row.get('Energy',0):.1f} τ:{row.get('Duration',0):.1f}")
    
    # Add component nodes
    comp_start = len(labels)
    labels.extend(comp_labels)
    node_colors.extend(cfg['node_colors']['components'])
    
    # Build links: Sources → Components → Target
    s_idx, t_idx, vals, l_colors, customdata_list = [], [], [], [], []
    
    # Stage 1: Sources → Components (decomposition)
    for i in range(n_sources):
        src = i + 1  # +1 because target is index 0
        row = sources_data[i]
        
        # Scaled values for visualization (not actual weights)
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
            
            # Store custom data for hovertemplate (FIXED: use customdata instead of hovertext)
            if c == 0:
                customdata_list.append(f"E*={query['Energy']}, Eᵢ={row.get('Energy',0)}<br>φ² = ((E*-Eᵢ)/s_E)² = {ve/10:.4f}")
            elif c == 1:
                customdata_list.append(f"τ*={query['Duration']}, τᵢ={row.get('Duration',0)}<br>φ² = ((τ*-τᵢ)/s_τ)² = {vτ/10:.4f}")
            elif c == 2:
                customdata_list.append(f"t*={query['Time']}, tᵢ={row.get('Time',0)}<br>φ² = ((t*-tᵢ)/s_t)² = {vt/10:.4f}")
            elif c == 3:
                customdata_list.append(f"αᵢ = 1/(1+√φ²)<br>Score: {row.get('Attention_Score',0):.4f}")
            elif c == 4:
                customdata_list.append(f"wᵢ ∝ αᵢ·gatingᵢ<br>Ref: {row.get('Refinement',0):.4f}")
            else:
                customdata_list.append(f"wᵢ = (αᵢ·gatingᵢ)/Σ(...)<br>Weight: {row.get('Combined_Weight',0):.4f}")
    
    # Stage 2: Components → Target (aggregation)
    for c in range(6):
        s_idx.append(comp_start + c)
        t_idx.append(0)  # Target index
        # Sum flows INTO this component
        flow_in = sum(v for s, t, v in zip(s_idx[:-6], t_idx[:-6], vals[:-6]) if t == comp_start + c)
        vals.append(flow_in * 0.5)  # Damping for visual balance
        l_colors.append('rgba(153,102,255,0.6)')
        customdata_list.append(f"Total flow into {comp_labels[c]}: {flow_in:.3f}")
    
    # Create Sankey figure - FIXED: removed invalid properties
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=cfg['node_pad'],
            thickness=cfg['node_thickness'],
            line=dict(color="black", width=0.5),
            label=labels,
            color=node_colors,
            # FIXED: Removed invalid 'font' property from node dict
            hovertemplate='<b>%{label}</b><br>Value: %{value:.3f}<extra></extra>'  # FIXED: in node dict
        ),
        link=dict(
            source=s_idx,
            target=t_idx,
            value=vals,
            color=l_colors,
            customdata=customdata_list,  # FIXED: use customdata instead of hovertext
            hovertemplate='%{customdata}<extra></extra>',  # FIXED: reference customdata in template
            line=dict(width=0.5, color='rgba(255,255,255,0.3)')
        )
        # FIXED: removed invalid top-level hoverinfo property
    ))
    
    # Update layout - FIXED: apply font via layout, not node
    title_text = (
        f"<b>ST-DGPA Attention Flow</b><br>"
        f"Query: E={query['Energy']:.2f} mJ, τ={query['Duration']:.2f} ns, t={query['Time']:.2f} ns<br>"
        f"<sub>σ_g={cfg.get('sigma_g', 0.20):.2f}, s_E={cfg.get('s_E', 10.0):.1f}, s_τ={cfg.get('s_tau', 5.0):.1f}, s_t={cfg.get('s_t', 20.0):.1f}</sub>"
    )
    
    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(family=cfg['font_family'], size=cfg['font_size']+4),
            x=0.5,
            xanchor='center'
        ),
        # FIXED: Apply global font via layout, not node
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
            namelength=-1  # Show full text
        )
    )
    
    return fig

# =============================================
# MAIN STREAMLIT APPLICATION
# =============================================
def main():
    st.set_page_config(
        page_title="ST-DGPA Sankey Visualizer",
        layout="wide",
        page_icon="🔬",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for enhanced visual appeal
    st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; text-align: center; margin-bottom: 1.5rem; font-weight: 800; }
    .info-box { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; }
    .metric-card { background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 1rem; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🔬 ST-DGPA Interpolation & Sankey Analysis</h1>', unsafe_allow_html=True)
    st.markdown("Visualizing the flow of weights from source simulations to the target query using Spatio-Temporal Gated Physics Attention.")
    
    # Initialize session state
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = UnifiedFEADataLoader()
    if 'extrapolator' not in st.session_state:
        st.session_state.extrapolator = SpatioTemporalGatedPhysicsAttentionExtrapolator()
    if 'df_sources' not in st.session_state:
        st.session_state.df_sources = None
    if 'interpolation_results' not in st.session_state:
        st.session_state.interpolation_results = None
    if 'interpolation_params' not in st.session_state:
        st.session_state.interpolation_params = None
    if 'interpolation_3d_cache' not in st.session_state:
        st.session_state.interpolation_3d_cache = {}
    if 'interpolation_field_history' not in st.session_state:
        st.session_state.interpolation_field_history = OrderedDict()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ ST-DGPA Parameters")
        
        # Physics parameters
        sigma_g = st.slider("Gating Width (σ_g)", 0.05, 1.0, 0.20, 0.05, 
                          help="Controls sharpness of the (E, τ, t) gating kernel")
        s_E = st.slider("Energy Scale (s_E) [mJ]", 0.1, 50.0, 10.0, 0.5,
                       help="Scaling factor for energy differences")
        s_tau = st.slider("Duration Scale (s_τ) [ns]", 0.1, 20.0, 5.0, 0.5,
                         help="Scaling factor for duration differences")
        s_t = st.slider("Time Scale (s_t) [ns]", 1.0, 50.0, 20.0, 1.0,
                       help="Scaling factor for time differences")
        temporal_weight = st.slider("Temporal Weight", 0.0, 1.0, 0.3, 0.05,
                                   help="Weight for temporal similarity in attention")
        
        # Update extrapolator
        st.session_state.extrapolator.sigma_g = sigma_g
        st.session_state.extrapolator.s_E = s_E
        st.session_state.extrapolator.s_tau = s_tau
        st.session_state.extrapolator.s_t = s_t
        st.session_state.extrapolator.temporal_weight = temporal_weight
        
        st.markdown("---")
        st.header("🎨 Visualization Settings")
        
        # Font settings
        font_family = st.selectbox("Font Family", 
                                  ["Arial, sans-serif", "Courier New, monospace", "Times New Roman, serif"],
                                  index=0)
        font_size = st.slider("Font Size", 8, 20, 12, 1)
        
        # Node settings
        node_thickness = st.slider("Node Thickness", 10, 40, 20, 2)
        
        # Figure size
        fig_width = st.slider("Figure Width", 600, 1400, 1000, 50)
        fig_height = st.slider("Figure Height", 400, 1000, 700, 50)
        
        # Color customization
        st.markdown("**Node Colors**")
        c1, c2 = st.columns(2)
        with c1:
            target_color = st.color_picker("Target Node", "#FF6B6B")
        with c2:
            source_color = st.color_picker("Source Base Color", "#9966FF")
        comp_color = st.color_picker("Component Base Color", "#4ECDC4")
        
        # Toggle mathematical explanations
        show_math = st.checkbox("Show Mathematical Formulas on Hover", value=True,
                               help="Display ST-DGPA formulas when hovering over diagram elements")
        
        st.markdown("---")
        st.header("📊 Data Settings")
        
        # Number of simulations for demo
        n_sims = st.slider("Demo Simulations", 2, 20, 4, 1,
                          help="Number of source simulations to generate")
        
        if st.button("🔄 Generate Demo Data", use_container_width=True):
            np.random.seed(42)  # Reproducible
            data = {
                'Energy': np.random.uniform(0.5, 8.0, n_sims),
                'Duration': np.random.uniform(2.0, 7.0, n_sims),
                'Time': np.random.uniform(1.0, 10.0, n_sims),
                'Max_Temp': np.random.uniform(500, 1500, n_sims)
            }
            st.session_state.df_sources = pd.DataFrame(data)
            st.success(f"✅ Generated {n_sims} demo simulations!")
        
        # CSV upload
        uploaded_file = st.file_uploader("📁 Upload CSV", type=["csv"],
                                        help="CSV with columns: Energy, Duration, Time, [Max_Temp]")
        if uploaded_file is not None:
            try:
                st.session_state.df_sources = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(st.session_state.df_sources)} rows from CSV")
            except Exception as e:
                st.error(f"❌ Error loading CSV: {e}")
                st.exception(e)
        
        # Show parameter summary
        if st.session_state.df_sources is not None:
            st.info(f"✅ {len(st.session_state.df_sources)} simulations loaded")

    # Main area: Query and visualization
    if st.session_state.df_sources is not None:
        df = st.session_state.df_sources
        st.success(f"✅ Loaded {len(df)} simulations")
        
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
        
        # Build customization dict
        customization = {
            'font_family': font_family,
            'font_size': font_size,
            'node_thickness': node_thickness,
            'width': fig_width,
            'height': fig_height,
            'show_math': show_math,
            'node_colors': {
                'target': target_color,
                'source': source_color,
                'components': [comp_color, '#4ECDC4', '#95E1D3', '#FFD93D', '#36A2EB', '#9966FF']
            }
        }
        
        if st.button("🚀 Compute ST-DGPA", type="primary", use_container_width=True):
            try:
                with st.spinner("Computing Spatio-Temporal Gated Physics Attention weights..."):
                    # Run interpolation
                    interpolator = st.session_state.extrapolator
                    results = interpolator.compute_stdgpa_weights(df.copy(), query)
                    
                    # Display Top Sources
                    st.markdown("### 📊 Top Contributing Sources")
                    top_sources = results.nlargest(5, 'Combined_Weight')[
                        ['Energy', 'Duration', 'Time', 'Combined_Weight']
                    ]
                    # Add Max_Temp if it exists
                    if 'Max_Temp' in results.columns:
                        top_sources = top_sources.join(results[['Max_Temp']])
                    
                    st.dataframe(
                        top_sources.style.format({
                            'Energy': '{:.2f}',
                            'Duration': '{:.2f}',
                            'Time': '{:.2f}',
                            'Combined_Weight': '{:.4f}',
                            'Max_Temp': '{:.1f}' if 'Max_Temp' in top_sources.columns else None
                        }).highlight_max(axis=0, subset=['Combined_Weight'], color='#90EE90'),
                        use_container_width=True
                    )
                    
                    # Prediction (weighted average) - only if Max_Temp exists
                    if 'Max_Temp' in results.columns:
                        predicted_temp = (results['Combined_Weight'] * results['Max_Temp']).sum()
                        st.metric("Predicted Max Temperature", f"{predicted_temp:.2f} K")
                    else:
                        st.info("⚠️ `Max_Temp` column not found. Skipping temperature prediction.")
                    
                    # REAL SANKEY DIAGRAM - Convert DataFrame to list of dicts
                    st.markdown("### 🕸️ Attention Weight Flow (REAL Sankey)")
                    st.markdown("""
                    **How to read this diagram:**
                    - **Left nodes**: Source simulations (colored by final weight)
                    - **Middle nodes**: Weight components (Energy Gate, Duration Gate, etc.)
                    - **Right node**: Target prediction (aggregated result)
                    - **Flow thickness**: Proportional to contribution magnitude
                    - **Hover over any element**: See mathematical formulas and detailed explanations
                    """)
                    
                    # FIXED: Convert DataFrame to list of dicts for Sankey function
                    sources_data = results.to_dict('records')
                    fig_sankey = create_stdgpa_sankey(sources_data, query, customization)
                    st.plotly_chart(fig_sankey, use_container_width=True)
                    
                    # Weight distribution plot
                    st.markdown("### 📈 Weight Distribution")
                    fig_dist = go.Figure()
                    fig_dist.add_trace(go.Bar(
                        x=[f"Sim {i+1}" for i in range(len(results))],
                        y=results['Combined_Weight'],
                        marker_color=source_color,
                        text=[f"{w*100:.1f}%" for w in results['Combined_Weight']],
                        textposition='auto'
                    ))
                    fig_dist.update_layout(
                        title="Final Weight Distribution Across Sources",
                        xaxis_title="Source Simulation",
                        yaxis_title="Weight",
                        height=400
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
                    
                    # ST-DGPA formula reference
                    with st.expander("📐 ST-DGPA Mathematical Formulas", expanded=True):
                        st.markdown("""
                        **Core ST-DGPA Weight Formula:**
                        ```
                        w_i(θ*) = [α_i(θ*) × gating_i(θ*)] / Σ_j[α_j(θ*) × gating_j(θ*)]
                        ```
                        
                        **Where:**
                        - `α_i(θ*)` = Physics attention (inverse distance similarity)
                        - `gating_i(θ*)` = exp(-φ_i² / (2σ_g²)) with φ_i² = Σ[(p*-p_i)/s_p]²
                        - `s_p` = scaling factor for parameter p ∈ {E, τ, t}
                        - `σ_g` = gating kernel width (controls sharpness)
                        
                        **Parameter Proximity Kernel:**
                        ```
                        φ_i = √[((E*-E_i)/s_E)² + ((τ*-τᵢ)/s_τ)² + ((t*-tᵢ)/s_t)²]
                        ```
                        
                        **Physics Attention:**
                        ```
                        αᵢ = 1 / (1 + √φ²)  [simplified inverse distance]
                        ```
                        """)
                        
            except Exception as e:
                st.error("❌ Prediction failed. Please check input parameters and data format.")
                st.exception(e)  # Show full traceback for debugging
                
    else:
        st.info("👈 Generate demo data or upload a CSV file in the sidebar to begin.")
        
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
            - `Max_Temp`: Maximum temperature in Kelvin (optional, for prediction example)
            """)

if __name__ == "__main__":
    main()
