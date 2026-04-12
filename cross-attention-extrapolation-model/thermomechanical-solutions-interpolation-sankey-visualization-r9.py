#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ENHANCED FEA LASER SIMULATION PLATFORM WITH ST-DGPA & CUSTOMIZABLE SANKEY DIAGRAM
====================================================================
🔧 FIXED: Streamlit color_picker now uses hex codes only
🔧 ENHANCED: Hex-to-rgba converter for Plotly Sankey compatibility
🔧 ENHANCED: Alpha/opacity slider for link transparency control
"""
import streamlit as st
import os
import glob
import re
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import meshio
import warnings
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
from io import BytesIO
import pandas as pd
import traceback
from scipy.interpolate import griddata, RBFInterpolator
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from plotly.subplots import make_subplots
import networkx as nx
from scipy.spatial.distance import cdist
import tempfile
import base64
import hashlib
import pickle
from functools import lru_cache
from collections import OrderedDict
import re as regex  # Avoid conflict with 're' module
warnings.filterwarnings('ignore')

# =============================================
# PATH CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEA_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "fea_solutions")
os.makedirs(FEA_SOLUTIONS_DIR, exist_ok=True)

SOURCE_REGISTRY = {}

# =============================================
# 🔧 NEW: COLOR UTILITIES FOR SANKEY COMPATIBILITY
# =============================================
class ColorUtils:
    """Utility functions for color format conversion between Streamlit and Plotly."""
    
    @staticmethod
    def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
        """
        Convert hex color code to rgba string for Plotly.
        
        Args:
            hex_color: Hex color string (e.g., "#2ecc71" or "2ecc71")
            alpha: Opacity value 0.0-1.0
            
        Returns:
            rgba string: "rgba(r, g, b, alpha)"
        """
        # Remove # if present and ensure 6-char hex
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 3:  # Expand shorthand #RGB to #RRGGBB
            hex_color = ''.join([c*2 for c in hex_color])
        if len(hex_color) != 6:
            # Fallback to default green if invalid
            hex_color = "2ecc71"
        
        try:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            # Clamp alpha to valid range
            alpha = max(0.0, min(1.0, alpha))
            return f"rgba({r}, {g}, {b}, {alpha:.2f})"
        except (ValueError, IndexError):
            # Return safe fallback
            return f"rgba(46, 204, 113, {alpha:.2f})"
    
    @staticmethod
    def validate_hex_color(hex_color: str) -> bool:
        """Validate if string is a proper hex color code."""
        hex_color = hex_color.lstrip('#')
        return bool(regex.fullmatch(r'[0-9a-fA-F]{6}', hex_color)) or \
               bool(regex.fullmatch(r'[0-9a-fA-F]{3}', hex_color))
    
    @staticmethod
    def get_safe_color(user_input: str, default: str = "#2ecc71") -> str:
        """Return validated hex color or fallback to default."""
        if ColorUtils.validate_hex_color(user_input):
            return '#' + user_input.lstrip('#')
        return default

# =============================================
# CACHE MANAGEMENT UTILITIES
# =============================================
class CacheManager:
    @staticmethod
    def generate_cache_key(field_name, timestep_idx, energy, duration, time,
                          sigma_param, spatial_weight, n_heads, temperature,
                          sigma_g, s_E, s_tau, s_t, temporal_weight,
                          top_k=None, subsample_factor=None):
        params_str = f"{field_name}_{timestep_idx}_{energy:.2f}_{duration:.2f}_{time:.2f}"
        params_str += f"_{sigma_param:.2f}_{spatial_weight:.2f}_{n_heads}_{temperature:.2f}"
        params_str += f"_{sigma_g:.2f}_{s_E:.2f}_{s_tau:.2f}_{s_t:.2f}_{temporal_weight:.2f}"
        if top_k: params_str += f"_top{top_k}"
        if subsample_factor: params_str += f"_sub{subsample_factor}"
        return hashlib.md5(params_str.encode()).hexdigest()[:16]
    
    @staticmethod
    def clear_3d_cache():
        if 'interpolation_3d_cache' in st.session_state: st.session_state.interpolation_3d_cache = {}
        if 'interpolation_field_history' in st.session_state: st.session_state.interpolation_field_history = OrderedDict()
    
    @staticmethod
    def get_cached_interpolation(field_name, timestep_idx, params):
        if 'interpolation_3d_cache' not in st.session_state: st.session_state.interpolation_3d_cache = {}
        cache_key = CacheManager.generate_cache_key(
            field_name, timestep_idx,
            params.get('energy_query', 0), params.get('duration_query', 0),
            params.get('time_points', [0])[timestep_idx] if timestep_idx < len(params.get('time_points', [])) else 0,
            params.get('sigma_param', 0.3), params.get('spatial_weight', 0.5),
            params.get('n_heads', 4), params.get('temperature', 1.0),
            params.get('sigma_g', 0.20), params.get('s_E', 10.0),
            params.get('s_tau', 5.0), params.get('s_t', 20.0),
            params.get('temporal_weight', 0.3), params.get('top_k'),
            params.get('subsample_factor')
        )
        return st.session_state.interpolation_3d_cache.get(cache_key)
    
    @staticmethod
    def set_cached_interpolation(field_name, timestep_idx, params, interpolated_values):
        if 'interpolation_3d_cache' not in st.session_state: st.session_state.interpolation_3d_cache = {}
        cache_key = CacheManager.generate_cache_key(
            field_name, timestep_idx,
            params.get('energy_query', 0), params.get('duration_query', 0),
            params.get('time_points', [0])[timestep_idx] if timestep_idx < len(params.get('time_points', [])) else 0,
            params.get('sigma_param', 0.3), params.get('spatial_weight', 0.5),
            params.get('n_heads', 4), params.get('temperature', 1.0),
            params.get('sigma_g', 0.20), params.get('s_E', 10.0),
            params.get('s_tau', 5.0), params.get('s_t', 20.0),
            params.get('temporal_weight', 0.3), params.get('top_k'),
            params.get('subsample_factor')
        )
        st.session_state.interpolation_3d_cache[cache_key] = {
            'interpolated_values': interpolated_values,
            'timestamp': datetime.now().timestamp(),
            'field_name': field_name,
            'timestep_idx': timestep_idx,
            'params': {k: v for k, v in params.items() if k not in ['simulations', 'summaries']}
        }
        if 'interpolation_field_history' not in st.session_state: st.session_state.interpolation_field_history = OrderedDict()
        st.session_state.interpolation_field_history[f"{field_name}_{timestep_idx}"] = cache_key
        if len(st.session_state.interpolation_3d_cache) > 20:
            oldest_key = min(st.session_state.interpolation_3d_cache.keys(), key=lambda k: st.session_state.interpolation_3d_cache[k]['timestamp'])
            del st.session_state.interpolation_3d_cache[oldest_key]
        if len(st.session_state.interpolation_field_history) > 10: st.session_state.interpolation_field_history.popitem(last=False)

# =============================================
# UNIFIED DATA LOADER
# =============================================
class UnifiedFEADataLoader:
    def __init__(self):
        self.simulations = {}
        self.summaries = []
        self.field_statistics = {}
        self.available_fields = set()
        self.source_id_counter = 0
        self.global_source_map = {}
    
    def parse_folder_name(self, folder: str):
        name = os.path.basename(folder)
        match = re.match(r"q([\dp\.]+)mJ-delta([\dp\.]+)ns", name)
        if not match: return None, None
        e, d = match.groups()
        return float(e.replace("p", ".")), float(d.replace("p", "."))
    
    @st.cache_data(show_spinner="Loading simulation data...")
    def load_all_simulations(_self, load_full_mesh=True):
        global SOURCE_REGISTRY
        SOURCE_REGISTRY = {}
        
        simulations, summaries = {}, []
        folders = glob.glob(os.path.join(FEA_SOLUTIONS_DIR, "q*mJ-delta*ns"))
        if not folders:
            st.warning(f"No simulation folders found in {FEA_SOLUTIONS_DIR}")
            return simulations, summaries
        
        progress_bar = st.progress(0)
        total_folders = len(folders)
        
        for folder_idx, folder in enumerate(folders):
            name = os.path.basename(folder)
            energy, duration = _self.parse_folder_name(name)
            if energy is None: 
                st.warning(f"⚠️ Could not parse folder name: {name}")
                continue
            
            vtu_files = sorted(glob.glob(os.path.join(folder, "a_t????.vtu")))
            if not vtu_files: 
                st.warning(f"⚠️ No VTU files found in: {folder}")
                continue
            
            try:
                mesh0 = meshio.read(vtu_files[0])
                if not mesh0.point_data: 
                    st.warning(f"⚠️ No point data in: {vtu_files[0]}")
                    continue
                
                sim_data = {
                    'name': name, 
                    'energy_mJ': energy, 
                    'duration_ns': duration, 
                    'n_timesteps': len(vtu_files), 
                    'vtu_files': vtu_files, 
                    'field_info': {}, 
                    'has_mesh': False,
                    'global_id_base': _self.source_id_counter
                }
                
                if load_full_mesh:
                    points = mesh0.points.astype(np.float32)
                    triangles = None
                    for cell_block in mesh0.cells:
                        if cell_block.type == "triangle": 
                            triangles = cell_block.data.astype(np.int32)
                            break
                    
                    fields = {}
                    for key in mesh0.point_data.keys():
                        arr = mesh0.point_data[key].astype(np.float32)
                        sim_data['field_info'][key] = ("scalar", 1) if arr.ndim == 1 else ("vector", arr.shape[1])
                        shape = (len(vtu_files), len(points)) if arr.ndim == 1 else (len(vtu_files), len(points), arr.shape[1])
                        fields[key] = np.full(shape, np.nan, dtype=np.float32)
                        fields[key][0] = arr
                        _self.available_fields.add(key)
                    
                    for t in range(1, len(vtu_files)):
                        try:
                            mesh = meshio.read(vtu_files[t])
                            for key in sim_data['field_info']:
                                if key in mesh.point_data: 
                                    fields[key][t] = mesh.point_data[key].astype(np.float32)
                        except Exception as e:
                            st.warning(f"⚠️ Error loading timestep {t} from {name}: {e}")
                            pass
                    
                    sim_data.update({
                        'points': points, 
                        'fields': fields, 
                        'triangles': triangles, 
                        'has_mesh': True
                    })
                
                summary = _self.extract_summary_statistics(vtu_files, energy, duration, name)
                
                for timestep_idx in range(len(summary['timesteps'])):
                    global_id = _self.source_id_counter
                    _self.global_source_map[(name, timestep_idx)] = global_id
                    SOURCE_REGISTRY[global_id] = {
                        'simulation_name': name,
                        'timestep_idx': timestep_idx,
                        'time': summary['timesteps'][timestep_idx],
                        'energy': energy,
                        'duration': duration,
                        'vtu_file': vtu_files[timestep_idx] if timestep_idx < len(vtu_files) else None
                    }
                    _self.source_id_counter += 1
                
                summaries.append(summary)
                simulations[name] = sim_data
                
            except Exception as e:
                st.error(f"❌ Error loading simulation {folder}: {traceback.format_exc()}")
                continue
            
            progress_bar.progress((folder_idx + 1) / total_folders)
        
        progress_bar.empty()
        
        if simulations: 
            st.success(f"✅ Loaded {len(simulations)} simulations with {len(_self.available_fields)} unique fields")
            st.info(f"🔗 Registered {len(SOURCE_REGISTRY)} source entries for Sankey visualization")
        else:
            st.error("❌ No simulations loaded. Check folder structure and VTU files.")
        
        return simulations, summaries
    
    def extract_summary_statistics(self, vtu_files, energy, duration, name):
        summary = {'name': name, 'energy': energy, 'duration': duration, 'timesteps': [], 'field_stats': {}}
        
        for idx, vtu_file in enumerate(vtu_files):
            try:
                mesh = meshio.read(vtu_file)
                summary['timesteps'].append(idx + 1)
                
                for field_name in mesh.point_data.keys():
                    data = mesh.point_data[field_name]
                    
                    if field_name not in summary['field_stats']:
                        summary['field_stats'][field_name] = {
                            'min': [], 'max': [], 'mean': [], 'std': [], 
                            'q25': [], 'q50': [], 'q75': [], 'percentiles': []
                        }
                    
                    if data.ndim == 1: 
                        clean_data = data[~np.isnan(data)]
                    else: 
                        norms = np.linalg.norm(data, axis=1)
                        clean_data = norms[~np.isnan(norms)]
                    
                    if clean_data.size > 0:
                        for key in ['min', 'max', 'mean', 'std', 'q25', 'q50', 'q75']: 
                            summary['field_stats'][field_name][key].append(float(eval(f"np.{key}(clean_data)")))
                        summary['field_stats'][field_name]['percentiles'].append(np.percentile(clean_data, [10, 25, 50, 75, 90]))
                    else:
                        for key in ['min', 'max', 'mean', 'std', 'q25', 'q50', 'q75']: 
                            summary['field_stats'][field_name][key].append(0.0)
                        summary['field_stats'][field_name]['percentiles'].append(np.zeros(5))
                        
            except Exception as e:
                st.warning(f"⚠️ Error extracting stats from {vtu_file}: {e}")
                continue
        
        return summary
    
    def get_source_metadata_for_sankey(self, global_ids=None):
        if global_ids is None:
            global_ids = list(SOURCE_REGISTRY.keys())
        
        metadata_list = []
        for gid in global_ids:
            if gid not in SOURCE_REGISTRY:
                continue
            
            reg_entry = SOURCE_REGISTRY[gid]
            sim_summary = None
            for s in self.summaries:
                if s['name'] == reg_entry['simulation_name']:
                    sim_summary = s
                    break
            
            if sim_summary:
                timestep_idx = reg_entry['timestep_idx']
                time_val = sim_summary['timesteps'][timestep_idx] if timestep_idx < len(sim_summary['timesteps']) else 0
                
                metadata_list.append({
                    'global_id': gid,
                    'simulation_name': reg_entry['simulation_name'],
                    'energy': reg_entry['energy'],
                    'duration': reg_entry['duration'],
                    'time': time_val,
                    'vtu_file': reg_entry['vtu_file'],
                    'field_stats': sim_summary['field_stats'] if timestep_idx < len(sim_summary['timesteps']) else {}
                })
        
        return metadata_list

# =============================================
# SPATIO-TEMPORAL GATED PHYSICS ATTENTION (ST-DGPA)
# =============================================
class SpatioTemporalGatedPhysicsAttentionExtrapolator:
    def __init__(self, sigma_param=0.3, spatial_weight=0.5, n_heads=4, temperature=1.0,
                 sigma_g=0.20, s_E=10.0, s_tau=5.0, s_t=20.0, temporal_weight=0.3):
        self.sigma_param = sigma_param
        self.spatial_weight = spatial_weight
        self.n_heads = n_heads
        self.temperature = temperature
        self.sigma_g = sigma_g; self.s_E = s_E; self.s_tau = s_tau; self.s_t = s_t
        self.temporal_weight = temporal_weight
        self.thermal_diffusivity = 1e-5
        self.laser_spot_radius = 50e-6
        self.characteristic_length = 100e-6
        self.source_db = []; self.embedding_scaler = StandardScaler()
        self.value_scaler = StandardScaler(); self.source_embeddings = []
        self.source_values = []; self.source_metadata = []; self.fitted = False
        self.data_loader_ref = None
    
    def set_data_loader(self, data_loader):
        self.data_loader_ref = data_loader
    
    def load_summaries(self, summaries):
        self.source_db = summaries
        if not summaries: return
        
        all_embeddings, all_values, metadata = [], [], []
        
        for summary_idx, summary in enumerate(summaries):
            for timestep_idx, t in enumerate(summary['timesteps']):
                all_embeddings.append(self._compute_enhanced_physics_embedding(
                    summary['energy'], summary['duration'], t))
                
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
                
                global_id = None
                if self.data_loader_ref and (summary['name'], timestep_idx) in self.data_loader_ref.global_source_map:
                    global_id = self.data_loader_ref.global_source_map[(summary['name'], timestep_idx)]
                
                metadata.append({
                    'summary_idx': summary_idx, 
                    'timestep_idx': timestep_idx, 
                    'energy': summary['energy'], 
                    'duration': summary['duration'], 
                    'time': t, 
                    'name': summary['name'],
                    'global_id': global_id,
                    'fourier_number': self._compute_fourier_number(t), 
                    'thermal_penetration': self._compute_thermal_penetration(t)
                })
        
        if all_embeddings and all_values:
            all_embeddings, all_values = np.array(all_embeddings), np.array(all_values)
            self.embedding_scaler.fit(all_embeddings); self.source_embeddings = self.embedding_scaler.transform(all_embeddings)
            self.value_scaler.fit(all_values); self.source_values = all_values
            self.source_metadata = metadata; self.fitted = True
            st.info(f"✅ Prepared {len(all_embeddings)} embeddings with {all_embeddings.shape[1]} features")
    
    def _compute_fourier_number(self, time_ns): 
        return self.thermal_diffusivity * (time_ns * 1e-9) / (self.characteristic_length ** 2)
    
    def _compute_thermal_penetration(self, time_ns): 
        return np.sqrt(self.thermal_diffusivity * (time_ns * 1e-9)) * 1e6
    
    def _compute_enhanced_physics_embedding(self, energy, duration, time):
        logE = np.log1p(energy); power = energy / max(duration, 1e-6)
        time_ratio = time / max(duration, 1e-3); fourier_number = self._compute_fourier_number(time)
        thermal_penetration = self._compute_thermal_penetration(time)
        return np.array([logE, duration, time, power, time_ratio, fourier_number, thermal_penetration,
                        1.0 if time < duration else 0.0, 1.0 if time >= duration else 0.0,
                        np.log1p(power), np.log1p(time), np.sqrt(time)], dtype=np.float32)
    
    def _compute_ett_gating(self, energy_query, duration_query, time_query, source_metadata=None):
        if source_metadata is None: source_metadata = self.source_metadata
        phi_squared = []
        
        for meta in source_metadata:
            de = (energy_query - meta['energy']) / self.s_E
            dt = (duration_query - meta['duration']) / self.s_tau
            dtime = (time_query - meta['time']) / self.s_t
            
            if self.temporal_weight > 0:
                time_scaling_factor = 1.0 + 0.5 * (time_query / max(duration_query, 1e-6))
                dtime *= time_scaling_factor
            
            phi_squared.append(de**2 + dt**2 + dtime**2)
        
        phi_squared = np.array(phi_squared)
        gating = np.exp(-phi_squared / (2 * self.sigma_g**2))
        gating_sum = np.sum(gating)
        return gating / gating_sum if gating_sum > 0 else np.ones_like(gating) / len(gating)
    
    def _compute_temporal_similarity(self, query_meta, source_metas):
        similarities = []
        for meta in source_metas:
            time_diff = abs(query_meta['time'] - meta['time'])
            
            if query_meta['time'] < query_meta['duration'] * 1.5: 
                temporal_tolerance = max(query_meta['duration'] * 0.1, 1.0)
            else: 
                temporal_tolerance = max(query_meta['duration'] * 0.3, 3.0)
            
            if 'fourier_number' in meta and 'fourier_number' in query_meta: 
                fourier_similarity = np.exp(-abs(query_meta['fourier_number'] - meta['fourier_number']) / 0.1)
            else: 
                fourier_similarity = 1.0
            
            time_similarity = np.exp(-time_diff / temporal_tolerance)
            similarities.append((1 - self.temporal_weight) * time_similarity + self.temporal_weight * fourier_similarity)
        
        return np.array(similarities)
    
    def _compute_spatial_similarity(self, query_meta, source_metas):
        similarities = []
        for meta in source_metas:
            e_diff = abs(query_meta['energy'] - meta['energy']) / 50.0
            d_diff = abs(query_meta['duration'] - meta['duration']) / 20.0
            similarities.append(np.exp(-np.sqrt(e_diff**2 + d_diff**2) / self.sigma_param))
        return np.array(similarities)
    
    def _multi_head_attention_with_gating(self, query_embedding, query_meta):
        if not self.fitted or len(self.source_embeddings) == 0: return None, None, None, None
        
        query_norm = self.embedding_scaler.transform([query_embedding])[0]
        n_sources = len(self.source_embeddings)
        head_weights = np.zeros((self.n_heads, n_sources))
        
        for head in range(self.n_heads):
            np.random.seed(42 + head)
            proj_dim = min(8, query_norm.shape[0])
            proj_matrix = np.random.randn(query_norm.shape[0], proj_dim)
            
            query_proj = query_norm @ proj_matrix
            source_proj = self.source_embeddings @ proj_matrix
            distances = np.linalg.norm(query_proj - source_proj, axis=1)
            scores = np.exp(-distances**2 / (2 * self.sigma_param**2))
            
            if self.spatial_weight > 0:
                spatial_sim = self._compute_spatial_similarity(query_meta, self.source_metadata)
                scores = (1 - self.spatial_weight) * scores + self.spatial_weight * spatial_sim
            
            if self.temporal_weight > 0:
                temporal_sim = self._compute_temporal_similarity(query_meta, self.source_metadata)
                scores = (1 - self.temporal_weight) * scores + self.temporal_weight * temporal_sim
            
            head_weights[head] = scores
        
        avg_weights = np.mean(head_weights, axis=0)
        if self.temperature != 1.0: avg_weights = avg_weights ** (1.0 / self.temperature)
        
        max_weight = np.max(avg_weights)
        exp_weights = np.exp(avg_weights - max_weight)
        physics_attention = exp_weights / (np.sum(exp_weights) + 1e-12)
        
        ett_gating = self._compute_ett_gating(query_meta['energy'], query_meta['duration'], query_meta['time'])
        combined_weights = physics_attention * ett_gating
        combined_sum = np.sum(combined_weights)
        final_weights = combined_weights / combined_sum if combined_sum > 1e-12 else physics_attention
        
        prediction = np.sum(final_weights[:, np.newaxis] * self.source_values, axis=0) if len(self.source_values) > 0 else np.zeros(1)
        
        return prediction, final_weights, physics_attention, ett_gating
    
    def predict_field_statistics(self, energy_query, duration_query, time_query):
        if not self.fitted: return None
        
        query_embedding = self._compute_enhanced_physics_embedding(energy_query, duration_query, time_query)
        query_meta = {
            'energy': energy_query, 'duration': duration_query, 'time': time_query, 
            'fourier_number': self._compute_fourier_number(time_query), 
            'thermal_penetration': self._compute_thermal_penetration(time_query)
        }
        
        prediction, final_weights, physics_attention, ett_gating = self._multi_head_attention_with_gating(query_embedding, query_meta)
        if prediction is None: return None
        
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
        
        if self.source_db:
            field_order = sorted(self.source_db[0]['field_stats'].keys())
            for i, field in enumerate(field_order):
                start_idx = i * 3
                if start_idx + 2 < len(prediction): 
                    result['field_predictions'][field] = {
                        'mean': float(prediction[start_idx]), 
                        'max': float(prediction[start_idx + 1]), 
                        'std': float(prediction[start_idx + 2])
                    }
        
        return result
    
    def _compute_temporal_confidence(self, time_query, duration_query):
        if time_query < duration_query * 0.5: return 0.6
        elif time_query < duration_query * 1.5: return 0.8
        else: return 0.9
    
    def _compute_heat_transfer_indicators(self, energy, duration, time):
        fourier_number = self._compute_fourier_number(time)
        thermal_penetration = self._compute_thermal_penetration(time)
        
        if time < duration * 0.3: phase, regime = "Early Heating", "Adiabatic-like"
        elif time < duration: phase, regime = "Heating", "Conduction-dominated"
        elif time < duration * 2: phase, regime = "Early Cooling", "Mixed conduction"
        else: phase, regime = "Diffusion Cooling", "Thermal diffusion"
        
        return {
            'phase': phase, 'regime': regime, 
            'fourier_number': fourier_number, 
            'thermal_penetration_um': thermal_penetration, 
            'normalized_time': time / max(duration, 1e-6), 
            'energy_density': energy / duration
        }
    
    def predict_time_series(self, energy_query, duration_query, time_points):
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
        
        if self.source_db:
            for field in set(f for s in self.source_db for f in s['field_stats'].keys()): 
                results['field_predictions'][field] = {'mean': [], 'max': [], 'std': []}
        
        for t in time_points:
            pred = self.predict_field_statistics(energy_query, duration_query, t)
            if pred and 'field_predictions' in pred:
                for field in pred['field_predictions']:
                    if field in results['field_predictions']: 
                        results['field_predictions'][field]['mean'].append(pred['field_predictions'][field]['mean'])
                        results['field_predictions'][field]['max'].append(pred['field_predictions'][field]['max'])
                        results['field_predictions'][field]['std'].append(pred['field_predictions'][field]['std'])
                
                results['attention_maps'].append(pred['attention_weights'])
                results['physics_attention_maps'].append(pred['physics_attention'])
                results['ett_gating_maps'].append(pred['ett_gating'])
                results['confidence_scores'].append(pred['confidence'])
                results['temporal_confidences'].append(pred['temporal_confidence'])
                results['heat_transfer_indicators'].append(pred['heat_transfer_indicators'])
            else:
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
        if not self.fitted or len(attention_weights) == 0: return None
        
        first_sim = next(iter(simulations.values()))
        if 'fields' not in first_sim or field_name not in first_sim['fields']: return None
        
        field_shape = first_sim['fields'][field_name].shape[1:]
        interpolated_field = np.zeros(
            first_sim['fields'][field_name].shape[1] if len(field_shape) == 0 else field_shape, 
            dtype=np.float32
        )
        
        total_weight, n_sources_used = 0.0, 0
        for idx, weight in enumerate(attention_weights):
            if weight < 1e-6: continue
            meta = source_metadata[idx]
            if meta['name'] in simulations and field_name in simulations[meta['name']]['fields']:
                interpolated_field += weight * simulations[meta['name']]['fields'][field_name][meta['timestep_idx']]
                total_weight += weight; n_sources_used += 1
        
        return interpolated_field / total_weight if total_weight > 0 else None
    
    def _assess_temporal_coherence(self, source_metadata, attention_weights):
        if len(source_metadata) == 0 or len(attention_weights) == 0: return 0.0
        
        times = np.array([meta['time'] for meta in source_metadata])
        weighted_times = times * attention_weights
        mean_time = np.sum(weighted_times) / np.sum(attention_weights)
        time_diff = times - mean_time
        weighted_variance = np.sum(attention_weights * time_diff**2) / np.sum(attention_weights)
        temporal_spread = np.sqrt(weighted_variance)
        avg_duration = np.mean([meta['duration'] for meta in source_metadata])
        
        return float(np.exp(-temporal_spread / max(avg_duration, 1e-6)))
    
    def export_interpolated_vtu(self, field_name, interpolated_values, simulations, output_path):
        if interpolated_values is None or len(simulations) == 0: return False
        try:
            first_sim = next(iter(simulations.values()))
            points = first_sim['points']
            cells = None
            if 'triangles' in first_sim and first_sim['triangles'] is not None: 
                cells = [("triangle", first_sim['triangles'])]
            
            mesh = meshio.Mesh(points, cells, point_data={field_name: interpolated_values})
            mesh.write(output_path)
            return True
        except Exception as e: 
            st.error(f"Error exporting VTU: {str(e)}")
            return False

# =============================================
# ADVANCED VISUALIZATION COMPONENTS
# =============================================
class EnhancedVisualizer:
    COLORSCALES = {
        'temperature': ['#2c0078', '#4402a7', '#5e04d1', '#7b0ef6', '#9a38ff', '#b966ff', '#d691ff', '#f2bcff'], 
        'stress': ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600'], 
        'displacement': ['#004c6d', '#346888', '#5886a5', '#7aa6c2', '#9dc6e0', '#c1e7ff'], 
        'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    }
    EXTENDED_COLORMAPS = ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'Rainbow', 'Jet', 'Hot', 'Cool', 'Portland', 'Bluered', 'Electric', 'Thermal', 'Balance', 'Brwnyl', 'Darkmint', 'Emrld', 'Mint', 'Oranges', 'Purp', 'Purples', 'Sunset', 'Sunsetdark', 'Teal', 'Tealgrn', 'Twilight', 'Burg', 'Burgyl']

    @staticmethod
    def create_stdgpa_analysis(results, energy_query, duration_query, time_points):
        if not results or 'attention_maps' not in results or len(results['attention_maps']) == 0: return None
        
        timestep_idx = len(time_points) // 2; time = time_points[timestep_idx]
        fig = make_subplots(
            rows=3, cols=3, 
            subplot_titles=[
                "ST-DGPA Final Weights", "Physics Attention Only", "(E, τ, t) Gating Only", 
                "ST-DGPA vs Physics Attention", "Temporal Coherence Analysis", "Heat Transfer Phase", 
                "Parameter Space 3D", "Attention Network", "Weight Evolution"
            ], 
            vertical_spacing=0.12, horizontal_spacing=0.12, 
            specs=[[{'type': 'xy'}, {'type': 'xy'}, {'type': 'xy'}], 
                   [{'type': 'xy'}, {'type': 'xy'}, {'type': 'polar'}], 
                   [{'type': 'scene'}, {'type': 'xy'}, {'type': 'xy'}]]
        )
        
        final_weights = results['attention_maps'][timestep_idx]
        physics_attention = results['physics_attention_maps'][timestep_idx]
        ett_gating = results['ett_gating_maps'][timestep_idx]
        
        fig.add_trace(go.Bar(x=list(range(len(final_weights))), y=final_weights, name='ST-DGPA Weights', marker_color='blue', showlegend=False), row=1, col=1)
        fig.add_trace(go.Bar(x=list(range(len(physics_attention))), y=physics_attention, name='Physics Attention', marker_color='green', showlegend=False), row=1, col=2)
        fig.add_trace(go.Bar(x=list(range(len(ett_gating))), y=ett_gating, name='(E, τ, t) Gating', marker_color='red', showlegend=False), row=1, col=3)
        
        fig.add_trace(go.Scatter(x=list(range(len(final_weights))), y=final_weights, mode='lines+markers', name='ST-DGPA Weights', line=dict(color='blue', width=3)), row=2, col=1)
        fig.add_trace(go.Scatter(x=list(range(len(physics_attention))), y=physics_attention, mode='lines+markers', name='Physics Attention', line=dict(color='green', width=2, dash='dash')), row=2, col=1)
        
        if st.session_state.get('summaries') and hasattr(st.session_state.extrapolator, 'source_metadata'):
            times, weights = [], []
            for i, weight in enumerate(final_weights):
                if weight > 0.01 and i < len(st.session_state.extrapolator.source_metadata): 
                    times.append(st.session_state.extrapolator.source_metadata[i]['time'])
                    weights.append(weight)
            if times and weights: 
                fig.add_trace(go.Scatter(x=times, y=weights, mode='markers', marker=dict(size=np.array(weights)*50, color=weights, colorscale='Viridis', showscale=False)), row=2, col=2)
                fig.add_vline(x=time, line_dash="dash", line_color="red", row=2, col=2)
        
        if 'heat_transfer_indicators' in results and results['heat_transfer_indicators']:
            indicators = results['heat_transfer_indicators'][timestep_idx]
            if indicators:
                phase = indicators.get('phase', 'Unknown')
                val_map = {'Early Heating': [0.9, 0.3, 0.2, 0.1], 'Heating': [0.9, 0.3, 0.2, 0.1], 'Early Cooling': [0.4, 0.8, 0.3, 0.1], 'Diffusion Cooling': [0.2, 0.5, 0.9, 0.2]}
                values = val_map.get(phase, [0.7, 0.5, 0.3, 0.2])
                values_closed = values + [values[0]]
                categories = ['Heating', 'Cooling', 'Diffusion', 'Adiabatic']
                categories_closed = categories + [categories[0]]
                fig.add_trace(go.Scatterpolar(r=values_closed, theta=categories_closed, fill='toself', fillcolor='rgba(243, 156, 18, 0.35)', line=dict(color='#f39c12', width=3), showlegend=False), row=2, col=3)
        
        fig.update_layout(height=1000, title_text=f"ST-DGPA Analysis at t={time} ns (E={energy_query:.1f} mJ, τ={duration_query:.1f} ns)", showlegend=True, legend=dict(x=1.05, y=1))
        fig.update_polars(radialaxis=dict(visible=True, range=[0, 1]), angularaxis=dict(direction="clockwise"), row=2, col=3)
        return fig

    def create_stdgpa_sankey(self, results, energy_query, duration_query, selected_time,
                            source_metadata, attention_weights, physics_attention,
                            ett_gating, temporal_sim=None, top_k=12,
                            custom_labels=None, font_size=12, font_family="Arial",
                            node_colors=None, link_colors_hex=None, link_alphas=None,
                            node_pad=20, node_thickness=30, show_values=True, 
                            orientation='h', hover_template=None):
        """
        🔧 FIXED: Uses hex colors for Streamlit compatibility, converts to rgba for Plotly
        """
        if len(attention_weights) == 0:
            return go.Figure()
        
        total_sources = len(attention_weights)
        if top_k == -1 or top_k >= total_sources:
            top_k = total_sources
        
        top_indices = np.argsort(attention_weights)[-top_k:][::-1]
        top_weights = attention_weights[top_indices]
        top_physics = physics_attention[top_indices]
        top_gating = ett_gating[top_indices]
        top_temporal = temporal_sim[top_indices] if temporal_sim is not None else np.ones_like(top_weights) * 0.5
        
        # Build labels
        labels = []
        if custom_labels and 'target' in custom_labels:
            labels.append(custom_labels['target'])
        else:
            labels.append("🔮 Target Query")
        
        for i, idx in enumerate(top_indices):
            meta = source_metadata[idx]
            sim_name = meta['name']
            if custom_labels and f'source_{i}' in custom_labels:
                label = custom_labels[f'source_{i}']
            else:
                if len(sim_name) > 25:
                    sim_name = sim_name[:22] + "..."
                label = f"<b>{sim_name}</b><br>E={meta['energy']:.1f}mJ, τ={meta['duration']:.1f}ns<br>t={meta['time']:.1f}ns"
            labels.append(label)
        
        component_start = len(labels)
        component_labels = ["⚛️ Physics Attention", "⚙️ (E,τ,t) Gating", "⏱️ Temporal Sim.", "🎯 ST-DGPA Final"]
        if custom_labels:
            for j, comp_key in enumerate(['physics', 'gating', 'temporal', 'final']):
                if comp_key in custom_labels:
                    component_labels[j] = custom_labels[comp_key]
        labels.extend(component_labels)
        
        source_idx, target_idx, values, link_colors_list = [], [], [], []
        
        # 🔧 FIXED: Default hex colors with alpha values for conversion
        default_link_hex = ["#2ecc71", "#e74c3c", "#3498db", "#9b59b6"]  # Green, Red, Blue, Purple
        default_link_alpha = [0.85, 0.85, 0.85, 0.90]
        
        # Use user-provided or defaults
        link_hex = link_colors_hex if link_colors_hex else default_link_hex
        link_alpha = link_alphas if link_alphas else default_link_alpha
        
        # Ensure lists are properly sized
        while len(link_hex) < 4:
            link_hex.append(default_link_hex[len(link_hex) % len(default_link_hex)])
        while len(link_alpha) < 4:
            link_alpha.append(default_link_alpha[len(link_alpha) % len(default_link_alpha)])
        
        # Connect source nodes to component nodes with rgba conversion
        for i, src_idx in enumerate(top_indices):
            s_node = i + 1
            # Physics attention flow
            source_idx.append(s_node); target_idx.append(component_start)
            values.append(top_physics[i] * 100)
            link_colors_list.append(ColorUtils.hex_to_rgba(link_hex[0], link_alpha[0]))
            # Gating flow
            source_idx.append(s_node); target_idx.append(component_start + 1)
            values.append(top_gating[i] * 100)
            link_colors_list.append(ColorUtils.hex_to_rgba(link_hex[1], link_alpha[1]))
            # Temporal similarity flow
            source_idx.append(s_node); target_idx.append(component_start + 2)
            values.append(top_temporal[i] * 100)
            link_colors_list.append(ColorUtils.hex_to_rgba(link_hex[2], link_alpha[2]))
            # Final ST-DGPA weight flow
            source_idx.append(s_node); target_idx.append(component_start + 3)
            values.append(top_weights[i] * 100)
            link_colors_list.append(ColorUtils.hex_to_rgba(link_hex[3], link_alpha[3]))

        # Connect component nodes to target
        for c in range(4):
            comp_node = component_start + c
            agg_value = sum(v for s, t, v in zip(source_idx, target_idx, values) if t == comp_node)
            if agg_value > 0:
                source_idx.append(comp_node); target_idx.append(0)
                values.append(agg_value * 0.6)
                link_colors_list.append("rgba(149, 165, 166, 0.7)")

        # Node colors (these can stay as hex - Plotly Sankey accepts hex for nodes)
        default_node_colors = ["#FF6B6B"] + ["#9b59b6"] * len(top_indices) + ["#2ecc71", "#e74c3c", "#3498db", "#9b59b6"]
        if node_colors and isinstance(node_colors, list):
            node_colors_final = [ColorUtils.get_safe_color(c) for c in node_colors[:len(labels)]]
            if len(node_colors_final) < len(labels):
                node_colors_final.extend(default_node_colors[len(node_colors_final):])
        else:
            node_colors_final = default_node_colors
        
        if hover_template is None:
            hover_template = '<b>%{label}</b><br>Total weight: %{value:.2f}<extra></extra>'
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=node_pad,
                thickness=node_thickness,
                line=dict(color="black", width=1),
                label=labels,
                color=node_colors_final,
                hovertemplate=hover_template
            ),
            link=dict(
                source=source_idx,
                target=target_idx,
                value=values if show_values else None,
                color=link_colors_list,
                hovertemplate='<b>%{source.label}</b> → <b>%{target.label}</b><br>Flow: %{value:.2f}%<extra></extra>'
            ),
            orientation=orientation
        )])
        
        fig.update_layout(
            title=dict(
                text=f"<b>ST-DGPA Weight Decomposition</b><br>Query: E={energy_query:.1f} mJ, τ={duration_query:.1f} ns, t={selected_time:.1f} ns<br>(Showing top {top_k} of {total_sources} source entries)",
                font=dict(size=22, family=font_family, color="#2c3e50"),
                x=0.5, y=0.95
            ),
            font=dict(family=font_family, size=font_size),
            width=1350,
            height=850,
            plot_bgcolor='rgba(248, 249, 250, 0.95)',
            paper_bgcolor='white',
            margin=dict(t=120, l=50, r=50, b=50)
        )
        return fig

    @staticmethod
    def create_temporal_analysis(results, time_points):
        if not results or 'heat_transfer_indicators' not in results: return None
        
        fig = make_subplots(
            rows=2, cols=2, 
            subplot_titles=["Heat Transfer Phase Evolution", "Fourier Number Evolution", "Temporal Confidence", "Thermal Penetration Depth"], 
            vertical_spacing=0.15, horizontal_spacing=0.15
        )
        
        phases = [ind.get('phase', 'Unknown') for ind in results['heat_transfer_indicators'] if ind]
        phase_mapping = {'Early Heating': 0, 'Heating': 1, 'Early Cooling': 2, 'Diffusion Cooling': 3}
        phase_values = [phase_mapping.get(p, 0) for p in phases]
        
        fig.add_trace(go.Scatter(x=time_points[:len(phase_values)], y=phase_values, mode='lines+markers', line=dict(color='red', width=3)), row=1, col=1)
        for phase_name, phase_val in phase_mapping.items(): 
            fig.add_hline(y=phase_val, line_dash="dot", line_color="gray", annotation_text=phase_name, row=1, col=1)
        
        fourier_numbers = [ind.get('fourier_number', 0) for ind in results['heat_transfer_indicators'] if ind]
        if fourier_numbers: 
            fig.add_trace(go.Scatter(x=time_points[:len(fourier_numbers)], y=fourier_numbers, mode='lines+markers', line=dict(color='blue', width=3)), row=1, col=2)
        
        if 'temporal_confidences' in results: 
            fig.add_trace(go.Scatter(x=time_points[:len(results['temporal_confidences'])], y=results['temporal_confidences'], mode='lines+markers', line=dict(color='green', width=3), fill='tozeroy', fillcolor='rgba(0, 255, 0, 0.2)'), row=2, col=1)
        
        penetration_depths = [ind.get('thermal_penetration_um', 0) for ind in results['heat_transfer_indicators'] if ind]
        if penetration_depths: 
            fig.add_trace(go.Scatter(x=time_points[:len(penetration_depths)], y=penetration_depths, mode='lines+markers', line=dict(color='orange', width=3)), row=2, col=2)
        
        fig.update_layout(height=700, title_text="Temporal Analysis of Heat Transfer Characteristics", showlegend=False)
        return fig

    @staticmethod
    def create_field_time_series(summaries, field_name):
        if not summaries:
            return None
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            subplot_titles=[f"{field_name} - Mean", f"{field_name} - Max", f"{field_name} - Std"],
                            vertical_spacing=0.1)
        for summary in summaries:
            if field_name not in summary['field_stats']:
                continue
            stats = summary['field_stats'][field_name]
            time_vals = summary['timesteps']
            name = summary['name'][:20]
            fig.add_trace(go.Scatter(x=time_vals, y=stats['mean'], mode='lines+markers', name=f"{name} (mean)"), row=1, col=1)
            fig.add_trace(go.Scatter(x=time_vals, y=stats['max'], mode='lines+markers', name=f"{name} (max)"), row=2, col=2)
            fig.add_trace(go.Scatter(x=time_vals, y=stats['std'], mode='lines+markers', name=f"{name} (std)"), row=3, col=3)
        fig.update_layout(height=800, title_text=f"Time Evolution of {field_name}", showlegend=True)
        return fig

    @staticmethod
    def create_field_boxplot(summaries, field_name):
        data = []
        for summary in summaries:
            if field_name not in summary['field_stats']:
                continue
            values = summary['field_stats'][field_name]['mean']
            if values:
                data.append(go.Box(y=values, name=summary['name'][:30], boxmean='sd'))
        if not data:
            return None
        fig = go.Figure(data=data)
        fig.update_layout(title=f"Distribution of {field_name} Mean Across Timesteps", 
                          yaxis_title=field_name, height=500)
        return fig

    @staticmethod
    def create_2d_slice(points, values, plane='xy', z_value=None, resolution=100):
        if points.shape[1] < 3:
            return None
        if plane == 'xy':
            x_vals = np.linspace(points[:,0].min(), points[:,0].max(), resolution)
            y_vals = np.linspace(points[:,1].min(), points[:,1].max(), resolution)
            X, Y = np.meshgrid(x_vals, y_vals)
            Z_target = z_value if z_value is not None else np.mean(points[:,2])
            points_plane = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, Z_target)])
        elif plane == 'xz':
            x_vals = np.linspace(points[:,0].min(), points[:,0].max(), resolution)
            z_vals = np.linspace(points[:,2].min(), points[:,2].max(), resolution)
            X, Z = np.meshgrid(x_vals, z_vals)
            Y_target = z_value if z_value is not None else np.mean(points[:,1])
            points_plane = np.column_stack([X.ravel(), np.full(X.size, Y_target), Z.ravel()])
        elif plane == 'yz':
            y_vals = np.linspace(points[:,1].min(), points[:,1].max(), resolution)
            z_vals = np.linspace(points[:,2].min(), points[:,2].max(), resolution)
            Y, Z = np.meshgrid(y_vals, z_vals)
            X_target = z_value if z_value is not None else np.mean(points[:,0])
            points_plane = np.column_stack([np.full(Y.size, X_target), Y.ravel(), Z.ravel()])
        else:
            return None
        
        from scipy.interpolate import griddata
        slice_vals = griddata(points, values, points_plane, method='linear')
        slice_vals = slice_vals.reshape((resolution, resolution))
        
        if plane == 'xy':
            fig = go.Figure(data=go.Heatmap(z=slice_vals, x=x_vals, y=y_vals, colorscale='Viridis'))
            fig.update_layout(title=f"Slice at Z={Z_target:.3f}", xaxis_title="X", yaxis_title="Y")
        elif plane == 'xz':
            fig = go.Figure(data=go.Heatmap(z=slice_vals, x=x_vals, y=z_vals, colorscale='Viridis'))
            fig.update_layout(title=f"Slice at Y={Y_target:.3f}", xaxis_title="X", yaxis_title="Z")
        else:
            fig = go.Figure(data=go.Heatmap(z=slice_vals, x=y_vals, y=z_vals, colorscale='Viridis'))
            fig.update_layout(title=f"Slice at X={X_target:.3f}", xaxis_title="Y", yaxis_title="Z")
        return fig

    @staticmethod
    def create_field_animation(sim_data, field_name):
        if not sim_data.get('has_mesh', False) or field_name not in sim_data['fields']:
            return None
        frames = []
        points = sim_data['points']
        triangles = sim_data.get('triangles')
        for t in range(sim_data['n_timesteps']):
            values = sim_data['fields'][field_name][t]
            if values.ndim > 1:
                values = np.linalg.norm(values, axis=1)
            if triangles is not None and len(triangles) > 0:
                frame = go.Frame(data=[go.Mesh3d(x=points[:,0], y=points[:,1], z=points[:,2],
                                                 i=triangles[:,0], j=triangles[:,1], k=triangles[:,2],
                                                 intensity=values, colorscale='Viridis',
                                                 name=f"t={t}")])
            else:
                frame = go.Frame(data=[go.Scatter3d(x=points[:,0], y=points[:,1], z=points[:,2],
                                                   mode='markers', marker=dict(size=2, color=values, colorscale='Viridis'))])
            frames.append(frame)
        fig = go.Figure(data=frames[0].data, frames=frames)
        fig.update_layout(
            title=f"Animation of {field_name} over time",
            updatemenus=[dict(type="buttons", buttons=[dict(label="Play", method="animate", args=[None])])],
            scene=dict(aspectmode="data")
        )
        return fig

# =============================================
# UI RENDERING FUNCTIONS
# =============================================
def render_data_viewer():
    st.markdown('<h2 class="sub-header">📁 Data Viewer</h2>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.markdown('<div class="warning-box"><h3>⚠️ No Data Loaded</h3><p>Please load simulations first using the sidebar button.</p></div>', unsafe_allow_html=True)
        return
    
    sim_name = st.selectbox("Select Simulation", sorted(st.session_state.simulations.keys()), key="viewer_sim_select")
    sim = st.session_state.simulations[sim_name]
    
    if not sim.get('has_mesh', False): 
        st.warning("No mesh data. Reload with 'Load Full Mesh'."); return
    
    tab1, tab2, tab3, tab4 = st.tabs(["3D Viewer", "Field Statistics", "2D Slice", "Animation"])
    
    with tab1:
        field = st.selectbox("Select Field", sorted(sim['field_info'].keys()), key="viewer_field_select")
        timestep = st.slider("Timestep", 0, sim['n_timesteps'] - 1, 0, key="viewer_timestep_slider")
        opacity = st.slider("Opacity", 0.0, 1.0, 0.9, 0.05, key="viewer_opacity")
        
        if 'points' in sim and 'fields' in sim and field in sim['fields']:
            pts = sim['points']
            kind, _ = sim['field_info'][field]
            raw = sim['fields'][field][timestep]
            
            values = np.where(np.isnan(raw), 0, raw) if kind == "scalar" else np.where(np.isnan(np.linalg.norm(raw, axis=1)), 0, np.linalg.norm(raw, axis=1))
            
            if sim.get('triangles') is not None and len(sim['triangles']) > 0:
                valid_triangles = [tri for tri in sim['triangles'] if all(idx < len(pts) for idx in tri)]
                if valid_triangles: 
                    mesh_data = go.Mesh3d(
                        x=pts[:,0], y=pts[:,1], z=pts[:,2], 
                        i=np.array(valid_triangles)[:,0], j=np.array(valid_triangles)[:,1], k=np.array(valid_triangles)[:,2], 
                        intensity=values, colorscale=st.session_state.selected_colormap, opacity=opacity, 
                        hovertemplate='<b>Value:</b> %{intensity:.3f}<extra></extra>'
                    )
                else: 
                    mesh_data = go.Scatter3d(
                        x=pts[:,0], y=pts[:,1], z=pts[:,2], mode='markers', 
                        marker=dict(size=4, color=values, colorscale=st.session_state.selected_colormap, opacity=opacity), 
                        hovertemplate='<b>Value:</b> %{marker.color:.3f}<extra></extra>'
                    )
            else: 
                mesh_data = go.Scatter3d(
                    x=pts[:,0], y=pts[:,1], z=pts[:,2], mode='markers', 
                    marker=dict(size=4, color=values, colorscale=st.session_state.selected_colormap, opacity=opacity), 
                    hovertemplate='<b>Value:</b> %{marker.color:.3f}<extra></extra>'
                )
            
            fig = go.Figure(data=mesh_data)
            fig.update_layout(
                title=dict(text=f"{field} at Timestep {timestep+1}<br><sub>{sim_name}</sub>", font=dict(size=20)), 
                scene=dict(aspectmode="data"), 
                height=700
            )
            st.plotly_chart(fig, use_container_width=True)
            
            if st.button("📸 Export Current View as PNG", key="export_3d"):
                img_bytes = fig.to_image(format="png", width=1200, height=800)
                b64 = base64.b64encode(img_bytes).decode()
                href = f'<a href="data:image/png;base64,{b64}" download="fea_plot.png">Download PNG</a>'
                st.markdown(href, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("Field Statistics Overview")
        field_stats = st.selectbox("Field for statistics", sorted(sim['field_info'].keys()), key="stats_field")
        if field_stats in sim['fields']:
            fig_ts = EnhancedVisualizer.create_field_time_series([st.session_state.summaries[sim_name]], field_stats)
            if fig_ts:
                st.plotly_chart(fig_ts, use_container_width=True)
            fig_box = EnhancedVisualizer.create_field_boxplot([st.session_state.summaries[sim_name]], field_stats)
            if fig_box:
                st.plotly_chart(fig_box, use_container_width=True)
    
    with tab3:
        st.subheader("2D Slice Through 3D Field")
        field_slice = st.selectbox("Field for slice", sorted(sim['field_info'].keys()), key="slice_field")
        timestep_slice = st.slider("Timestep for slice", 0, sim['n_timesteps'] - 1, 0, key="slice_timestep")
        plane = st.selectbox("Plane", ['xy', 'xz', 'yz'], key="slice_plane")
        if field_slice in sim['fields']:
            raw_slice = sim['fields'][field_slice][timestep_slice]
            if raw_slice.ndim > 1:
                values_slice = np.linalg.norm(raw_slice, axis=1)
            else:
                values_slice = raw_slice
            fig_slice = EnhancedVisualizer.create_2d_slice(sim['points'], values_slice, plane=plane)
            if fig_slice:
                st.plotly_chart(fig_slice, use_container_width=True)
            else:
                st.warning("Could not create slice (insufficient points or invalid plane).")
    
    with tab4:
        st.subheader("Animate Field Over Time")
        field_anim = st.selectbox("Field for animation", sorted(sim['field_info'].keys()), key="anim_field")
        if field_anim in sim['fields']:
            fig_anim = EnhancedVisualizer.create_field_animation(sim, field_anim)
            if fig_anim:
                st.plotly_chart(fig_anim, use_container_width=True)
            else:
                st.warning("Animation not available for this field.")

def render_interpolation_extrapolation():
    st.markdown('<h2 class="sub-header">🔮 Interpolation/Extrapolation Engine</h2>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded: 
        st.markdown('<div class="warning-box"><h3>⚠️ No Data Loaded</h3></div>', unsafe_allow_html=True)
        return
    
    with st.expander("📋 Loaded Simulations Summary", expanded=True):
        if st.session_state.summaries:
            df_summary = pd.DataFrame([
                {'Simulation': s['name'], 'Energy (mJ)': s['energy'], 'Duration (ns)': s['duration'], 'Timesteps': len(s['timesteps'])} 
                for s in st.session_state.summaries
            ])
            st.dataframe(df_summary, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1: energy_query = st.number_input("Energy (mJ)", 0.1, 20.0, 3.5, 0.1, key="interp_energy")
    with col2: duration_query = st.number_input("Duration (ns)", 0.1, 20.0, 4.2, 0.1, key="interp_duration")
    with col3: max_time = st.number_input("Max Prediction Time (ns)", 1, 200, 50, 1, key="interp_maxtime")
    
    time_points = np.arange(1, max_time + 1, 2)
    sigma_g = st.slider("Gating Width (σ_g)", 0.05, 1.0, 0.20, 0.05, key="interp_sigma_g")
    s_E, s_tau, s_t = 10.0, 5.0, 20.0
    temporal_weight = st.slider("Temporal Weight", 0.0, 1.0, 0.3, 0.05, key="interp_temporal_weight")
    
    st.session_state.extrapolator.sigma_g = sigma_g
    st.session_state.extrapolator.s_E = s_E
    st.session_state.extrapolator.s_tau = s_tau
    st.session_state.extrapolator.s_t = s_t
    st.session_state.extrapolator.temporal_weight = temporal_weight
    
    if st.button("🚀 Run ST-DGPA Prediction", type="primary", use_container_width=True):
        with st.spinner("Computing ST-DGPA..."):
            CacheManager.clear_3d_cache()
            results = st.session_state.extrapolator.predict_time_series(energy_query, duration_query, time_points)
            
            if results and 'field_predictions' in results and results['field_predictions']:
                st.session_state.interpolation_results = results
                st.session_state.interpolation_params = {
                    'energy_query': energy_query, 'duration_query': duration_query, 
                    'time_points': time_points, 'sigma_g': sigma_g, 'temporal_weight': temporal_weight
                }
                st.markdown('<div class="success-box"><h3>✅ Prediction Successful</h3></div>', unsafe_allow_html=True)
                
                tab1, tab2, tab3, tab4 = st.tabs(["📈 Predictions", "🧠 ST-DGPA Analysis & Sankey", "⏱️ Temporal Analysis", "🌐 3D Analysis"])
                with tab1: render_prediction_results(results, time_points, energy_query, duration_query)
                with tab2: render_stdgpa_attention_visualization(results, energy_query, duration_query, time_points)
                with tab3: render_temporal_analysis(results, time_points, energy_query, duration_query)
                with tab4: render_3d_analysis(results, time_points, energy_query, duration_query)
            else: 
                st.error("Prediction failed. Check data loading and parameters.")
    
    elif st.session_state.interpolation_results is not None:
        params = st.session_state.interpolation_params or {}
        energy_query = params.get('energy_query', 0)
        duration_query = params.get('duration_query', 0)
        time_points = params.get('time_points', [])
        results = st.session_state.interpolation_results
        
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Predictions", "🧠 ST-DGPA Analysis & Sankey", "⏱️ Temporal Analysis", "🌐 3D Analysis"])
        with tab1: render_prediction_results(results, time_points, energy_query, duration_query)
        with tab2: render_stdgpa_attention_visualization(results, energy_query, duration_query, time_points)
        with tab3: render_temporal_analysis(results, time_points, energy_query, duration_query)
        with tab4: render_3d_analysis(results, time_points, energy_query, duration_query)

def render_prediction_results(results, time_points, energy_query, duration_query):
    available_fields = list(results['field_predictions'].keys())
    if not available_fields: 
        st.warning("No field predictions."); return
    
    n_fields = min(len(available_fields), 4)
    fig = make_subplots(rows=n_fields, cols=1, subplot_titles=[f"Predicted {field}" for field in available_fields[:n_fields]], vertical_spacing=0.1, shared_xaxes=True)
    
    for idx, field in enumerate(available_fields[:n_fields]):
        row = idx + 1
        if results['field_predictions'][field]['mean']: 
            fig.add_trace(
                go.Scatter(x=time_points, y=results['field_predictions'][field]['mean'], mode='lines', name=f'{field} (mean)', line=dict(width=3, color='blue')), 
                row=row, col=1
            )
    
    fig.update_layout(height=300 * n_fields, title_text=f"Field Predictions (E={energy_query:.1f} mJ, τ={duration_query:.1f} ns)")
    st.plotly_chart(fig, use_container_width=True)
    
    if results['confidence_scores']:
        fig_conf = go.Figure()
        fig_conf.add_trace(go.Scatter(x=time_points, y=results['confidence_scores'], mode='lines+markers', line=dict(color='orange', width=3), fill='tozeroy', fillcolor='rgba(255, 165, 0, 0.2)'))
        fig_conf.update_layout(title="Prediction Confidence Over Time", height=400, yaxis_range=[0, 1])
        st.plotly_chart(fig_conf, use_container_width=True)

def render_stdgpa_attention_visualization(results, energy_query, duration_query, time_points):
    """🔧 FIXED: Sankey customization with hex colors only for Streamlit compatibility."""
    if not results.get('physics_attention_maps') or len(results['physics_attention_maps'][0]) == 0:
        st.info("No ST-DGPA attention data available.")
        return
    
    st.markdown('<h4 class="sub-header">🧠 ST-DGPA Weight Decomposition (Customizable Sankey)</h4>', unsafe_allow_html=True)
    
    col_timestep, col_topk = st.columns([2, 1])
    with col_timestep:
        selected_timestep_idx = st.slider(
            "Select timestep for ST-DGPA analysis",
            0, len(time_points) - 1, len(time_points) // 2,
            key="stdgpa_timestep"
        )
    with col_topk:
        total_sources = len(results['attention_maps'][selected_timestep_idx])
        max_topk = total_sources
        top_k = st.slider(
            "Number of source entries to show in Sankey",
            min_value=5, max_value=max_topk, value=min(15, max_topk),
            help="Show top N source entries by final attention weight. Set to max to show all.",
            key="stdgpa_topk"
        )
    
    final_weights = results['attention_maps'][selected_timestep_idx]
    physics_attention = results['physics_attention_maps'][selected_timestep_idx]
    ett_gating = results['ett_gating_maps'][selected_timestep_idx]
    selected_time = time_points[selected_timestep_idx]
    
    temporal_sim = None
    if st.session_state.extrapolator.temporal_weight > 0 and hasattr(st.session_state.extrapolator, 'source_metadata'):
        query_meta = {
            'time': selected_time, 
            'duration': duration_query, 
            'fourier_number': st.session_state.extrapolator._compute_fourier_number(selected_time)
        }
        temporal_sim = st.session_state.extrapolator._compute_temporal_similarity(
            query_meta, st.session_state.extrapolator.source_metadata
        )

    # 🔧 FIXED: Sankey Customization Panel with HEX COLORS ONLY
    with st.expander("🎨 Customize Sankey Diagram", expanded=False):
        st.info("💡 **Note**: Color pickers use hex codes (#RRGGBB). Opacity is controlled separately.")
        
        st.markdown("##### ✏️ Edit Node Labels")
        
        if 'sankey_custom_labels' not in st.session_state:
            st.session_state.sankey_custom_labels = {}
        
        target_label = st.text_input(
            "Target Query Label", 
            value=st.session_state.sankey_custom_labels.get('target', "🔮 Target Query"),
            key="custom_target_label"
        )
        st.session_state.sankey_custom_labels['target'] = target_label
        
        st.markdown("**Source Node Labels** (edit individual entries):")
        top_indices = np.argsort(final_weights)[-top_k:][::-1]
        
        for i, idx in enumerate(top_indices):
            meta = st.session_state.extrapolator.source_metadata[idx]
            default_label = f"{meta['name'][:25]}... E={meta['energy']:.1f}mJ" if len(meta['name']) > 25 else f"{meta['name']} E={meta['energy']:.1f}mJ"
            custom_key = f'source_{i}'
            current_val = st.session_state.sankey_custom_labels.get(custom_key, default_label)
            new_val = st.text_input(f"Source {i+1} Label", value=current_val, key=f"custom_source_{i}_label")
            st.session_state.sankey_custom_labels[custom_key] = new_val
        
        st.markdown("**Component Node Labels**:")
        comp_cols = st.columns(2)
        comp_keys = ['physics', 'gating', 'temporal', 'final']
        comp_defaults = ["⚛️ Physics Attention", "⚙️ (E,τ,t) Gating", "⏱️ Temporal Sim.", "🎯 ST-DGPA Final"]
        for j, (ckey, cdefault) in enumerate(zip(comp_keys, comp_defaults)):
            col_idx = j % 2
            with comp_cols[col_idx]:
                val = st.text_input(f"{ckey.title()} Label", 
                                   value=st.session_state.sankey_custom_labels.get(ckey, cdefault),
                                   key=f"custom_comp_{ckey}")
                st.session_state.sankey_custom_labels[ckey] = val
        
        st.markdown("---")
        st.markdown("##### 🔤 Font & Typography")
        font_family = st.selectbox("Font Family", ["Arial", "Courier New", "Times New Roman", "Verdana", "Georgia"], 
                                   index=0, key="sankey_font_family")
        font_size = st.slider("Font Size", 8, 24, 12, key="sankey_font_size")
        
        st.markdown("##### 🎨 Node Colors (Hex)")
        node_color_1 = st.color_picker("Target Node Color", "#FF6B6B", key="sankey_node_target")
        node_color_2 = st.color_picker("Source Nodes Color", "#9b59b6", key="sankey_node_source")
        comp_colors = st.columns(4)
        comp_color_keys = ['physics', 'gating', 'temporal', 'final']
        comp_color_defaults = ["#2ecc71", "#e74c3c", "#3498db", "#9b59b6"]
        node_colors_list = [node_color_1] + [node_color_2] * top_k
        for j, (ckey, cdefault) in enumerate(zip(comp_color_keys, comp_color_defaults)):
            with comp_colors[j]:
                cval = st.color_picker(f"{ckey.title()}", cdefault, key=f"sankey_comp_{ckey}")
                node_colors_list.append(cval)
        
        st.markdown("##### 🔗 Link Colors & Opacity (Hex + Alpha)")
        st.caption("Select base color with picker, then adjust transparency with slider")
        
        link_cols = st.columns(4)
        link_keys = ['physics', 'gating', 'temporal', 'final']
        link_hex_defaults = ["#2ecc71", "#e74c3c", "#3498db", "#9b59b6"]
        link_alpha_defaults = [0.85, 0.85, 0.85, 0.90]
        
        link_hex_list = []
        link_alpha_list = []
        
        for j, (lkey, lhex_default, lalpha_default) in enumerate(zip(link_keys, link_hex_defaults, link_alpha_defaults)):
            with link_cols[j]:
                lhex = st.color_picker(f"{lkey.title()} Color", lhex_default, key=f"sankey_link_hex_{lkey}")
                lalpha = st.slider(f"{lkey.title()} Opacity", 0.0, 1.0, lalpha_default, 0.05, key=f"sankey_link_alpha_{lkey}")
                link_hex_list.append(lhex)
                link_alpha_list.append(lalpha)
        
        st.markdown("##### 📐 Layout & Display")
        node_pad = st.slider("Node Padding", 5, 50, 20, key="sankey_node_pad")
        node_thickness = st.slider("Node Thickness", 10, 50, 30, key="sankey_node_thickness")
        show_values = st.checkbox("Show Link Values", value=True, key="sankey_show_values")
        orientation = st.radio("Orientation", ['h', 'v'], horizontal=True, key="sankey_orientation", 
                              format_func=lambda x: "Horizontal" if x == 'h' else "Vertical")
        
        hover_template = st.text_area(
            "Custom Hover Template (Plotly format)",
            value=st.session_state.get('sankey_hover_template', '<b>%{label}</b><br>Weight: %{value:.2f}<extra></extra>'),
            key="sankey_hover_template",
            help="Use Plotly variables: %{label}, %{value}, %{source.label}, %{target.label}"
        )
        st.session_state.sankey_hover_template = hover_template
        
        # Reset button
        if st.button("🔄 Reset to Defaults", key="sankey_reset"):
            st.session_state.sankey_custom_labels = {}
            st.rerun()
    
    # Build customization dict for Sankey creation
    custom_labels = st.session_state.sankey_custom_labels.copy()
    
    # 🔧 FIXED: Create Sankey with hex colors + alpha conversion
    sankey_fig = st.session_state.visualizer.create_stdgpa_sankey(
        results=results, 
        energy_query=energy_query, 
        duration_query=duration_query, 
        selected_time=selected_time,
        source_metadata=st.session_state.extrapolator.source_metadata, 
        attention_weights=final_weights,
        physics_attention=physics_attention, 
        ett_gating=ett_gating, 
        temporal_sim=temporal_sim, 
        top_k=top_k,
        # Customization parameters
        custom_labels=custom_labels,
        font_size=font_size,
        font_family=font_family,
        node_colors=node_colors_list,
        link_colors_hex=link_hex_list,  # 🔧 Hex colors for Streamlit
        link_alphas=link_alpha_list,     # 🔧 Separate alpha values
        node_pad=node_pad,
        node_thickness=node_thickness,
        show_values=show_values,
        orientation=orientation,
        hover_template=hover_template if hover_template.strip() else None
    )
    st.plotly_chart(sankey_fig, use_container_width=True)

    st.markdown("##### 📊 ST-DGPA Weight Analysis (Top Sources)")
    if len(final_weights) > 0:
        top_indices = np.argsort(final_weights)[-min(10, len(final_weights)):][::-1]
        comparison_data = []
        for i, idx in enumerate(top_indices):
            meta = st.session_state.extrapolator.source_metadata[idx]
            comparison_data.append({
                'Global ID (Simulation)': meta['name'],
                'Time (ns)': f"{meta['time']:.1f}",
                'Energy (mJ)': f"{meta['energy']:.1f}",
                'Duration (ns)': f"{meta['duration']:.1f}",
                'Physics Attn': f"{physics_attention[idx]:.4f}",
                '(E,τ,t) Gating': f"{ett_gating[idx]:.4f}",
                'ST-DGPA Final': f"{final_weights[idx]:.4f}"
            })
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
        st.markdown(f"**Total source entries:** {len(final_weights)} | **Top {top_k} shown in Sankey**")

def render_temporal_analysis(results, time_points, energy_query, duration_query):
    if not results or 'heat_transfer_indicators' not in results: 
        st.info("No temporal data available."); return
    
    fig = st.session_state.visualizer.create_temporal_analysis(results, time_points)
    if fig: 
        st.plotly_chart(fig, use_container_width=True)

def render_3d_analysis(results, time_points, energy_query, duration_query):
    st.markdown('<h4 class="sub-header">🌐 3D Parameter Space</h4>', unsafe_allow_html=True)
    
    if not st.session_state.summaries: return
    
    train_energies = [s['energy'] for s in st.session_state.summaries]
    train_durations = [s['duration'] for s in st.session_state.summaries]
    train_max_temps = [
        np.max(s['field_stats'].get('temperature', {}).get('max', [0])) if s['field_stats'].get('temperature') else 0 
        for s in st.session_state.summaries
    ]
    query_max_temp = np.max(results['field_predictions'].get('temperature', {}).get('max', [0]))
    
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter3d(
        x=train_energies, y=train_durations, z=train_max_temps, mode='markers', 
        marker=dict(size=8, color=train_max_temps, colorscale='Viridis'), name='Training'
    ))
    fig_temp.add_trace(go.Scatter3d(
        x=[energy_query], y=[duration_query], z=[query_max_temp], mode='markers', 
        marker=dict(size=12, color='red', symbol='diamond'), name='Query'
    ))
    fig_temp.update_layout(
        title="Parameter Space - Max Temperature", 
        scene=dict(xaxis_title="Energy (mJ)", yaxis_title="Duration (ns)", zaxis_title="Max Temp"), 
        height=500
    )
    st.plotly_chart(fig_temp, use_container_width=True)

# =============================================
# MAIN APPLICATION
# =============================================
def main():
    st.set_page_config(
        page_title="Enhanced FEA Laser Platform with Customizable Sankey", 
        layout="wide", 
        initial_sidebar_state="expanded", 
        page_icon="🔬"
    )
    
    st.markdown("""
    <style>
    .main-header { font-size: 3rem; background: linear-gradient(90deg, #1E88E5, #4A00E0); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; margin-bottom: 1.5rem; font-weight: 800; }
    .sub-header { font-size: 1.8rem; color: #2c3e50; margin-top: 1.5rem; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 3px solid #3498db; font-weight: 600; }
    .warning-box { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 1.5rem; border-radius: 10px; margin: 1.5rem 0; box-shadow: 0 10px 20px rgba(0,0,0,0.1); }
    .success-box { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 1.5rem; border-radius: 10px; margin: 1.5rem 0; box-shadow: 0 10px 20px rgba(0,0,0,0.1); }
    .info-note { background: #e8f4fd; border-left: 4px solid #3498db; padding: 1rem; margin: 1rem 0; border-radius: 0 8px 8px 0; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🔬 Enhanced FEA Laser Simulation Platform</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'data_loader' not in st.session_state: 
        st.session_state.data_loader = UnifiedFEADataLoader()
    if 'extrapolator' not in st.session_state: 
        st.session_state.extrapolator = SpatioTemporalGatedPhysicsAttentionExtrapolator()
    if 'visualizer' not in st.session_state: 
        st.session_state.visualizer = EnhancedVisualizer()
    if 'data_loaded' not in st.session_state: 
        st.session_state.data_loaded = False
    if 'interpolation_results' not in st.session_state: 
        st.session_state.interpolation_results = None
    if 'interpolation_3d_cache' not in st.session_state: 
        st.session_state.interpolation_3d_cache = {}
    if 'selected_colormap' not in st.session_state: 
        st.session_state.selected_colormap = 'Viridis'
    if 'sankey_custom_labels' not in st.session_state:
        st.session_state.sankey_custom_labels = {}
    if 'sankey_hover_template' not in st.session_state:
        st.session_state.sankey_hover_template = '<b>%{label}</b><br>Weight: %{value:.2f}<extra></extra>'
    
    with st.sidebar:
        st.markdown("### ⚙️ Navigation")
        app_mode = st.radio("Select Mode", ["Data Viewer", "Interpolation/Extrapolation"], key="nav_mode")
        st.markdown("---")
        
        if st.button("🔄 Load All Simulations", type="primary", use_container_width=True):
            with st.spinner("Loading simulations and building Sankey registry..."):
                sims, summaries = st.session_state.data_loader.load_all_simulations(load_full_mesh=True)
                st.session_state.simulations = sims
                st.session_state.summaries = summaries
                
                if sims and summaries: 
                    st.session_state.extrapolator.set_data_loader(st.session_state.data_loader)
                    st.session_state.extrapolator.load_summaries(summaries)
                    st.session_state.data_loaded = True
                    st.success("✅ Data loaded successfully! Sankey visualization enabled.")
        
        st.markdown("---")
        st.markdown("### 🎨 Visualization Settings")
        st.session_state.selected_colormap = st.selectbox("Colormap", EnhancedVisualizer.EXTENDED_COLORMAPS, index=0)
        
        st.markdown("### 🎯 Sankey Style Presets")
        if st.button("🌈 Colorful", key="preset_colorful"):
            st.session_state.sankey_custom_labels = {}
            st.rerun()
        if st.button("🔬 Minimal", key="preset_minimal"):
            st.session_state.sankey_custom_labels = {'target': 'Query', 'physics': 'Physics', 'gating': 'Gating', 'temporal': 'Temporal', 'final': 'Final'}
            st.rerun()
    
    if app_mode == "Data Viewer": 
        render_data_viewer()
    elif app_mode == "Interpolation/Extrapolation": 
        render_interpolation_extrapolation()

if __name__ == "__main__":
    main()
