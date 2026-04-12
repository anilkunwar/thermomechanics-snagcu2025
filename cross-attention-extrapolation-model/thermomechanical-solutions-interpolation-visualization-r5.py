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
from typing import List, Dict, Any, Optional, Tuple
from io import BytesIO
import traceback
from scipy.interpolate import griddata, RBFInterpolator
from sklearn.preprocessing import StandardScaler
import tempfile
import base64
import hashlib
from collections import OrderedDict
import json

warnings.filterwarnings('ignore')

# =============================================
# PATH CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEA_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "fea_solutions")
VISUALIZATION_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "visualization_outputs")
TEMP_ANIMATION_DIR = os.path.join(SCRIPT_DIR, "temp_animations")

for d in [FEA_SOLUTIONS_DIR, VISUALIZATION_OUTPUT_DIR, TEMP_ANIMATION_DIR]:
    os.makedirs(d, exist_ok=True)

# =============================================
# 1. CACHE MANAGEMENT UTILITIES
# =============================================
class CacheManager:
    @staticmethod
    def generate_cache_key(field_name, timestep_idx, energy, duration, time,
                          sigma_param, spatial_weight, n_heads, temperature,
                          sigma_g, s_E, s_tau, s_t, temporal_weight,
                          top_k=None, subsample_factor=None) -> str:
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
            params.get('temporal_weight', 0.3), params.get('top_k'), params.get('subsample_factor')
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
            params.get('temporal_weight', 0.3), params.get('top_k'), params.get('subsample_factor')
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
        if len(st.session_state.interpolation_field_history) > 10:
            st.session_state.interpolation_field_history.popitem(last=False)

# =============================================
# 2. UNIFIED DATA LOADER
# =============================================
class UnifiedFEADataLoader:
    def __init__(self):
        self.simulations = {}
        self.summaries = []
        self.available_fields = set()
    
    def parse_folder_name(self, folder: str):
        match = re.match(r"q([\dp\.]+)mJ-delta([\dp\.]+)ns", folder)
        if not match: return None, None
        e, d = match.groups()
        return float(e.replace("p", ".")), float(d.replace("p", "."))
    
    @st.cache_data(show_spinner="Loading simulation data...")
    def load_all_simulations(_self, load_full_mesh=True):
        simulations, summaries = {}, []
        folders = glob.glob(os.path.join(FEA_SOLUTIONS_DIR, "q*mJ-delta*ns"))
        if not folders:
            st.warning(f"No simulation folders found in {FEA_SOLUTIONS_DIR}")
            return simulations, summaries
        progress_bar = st.progress(0)
        status_text = st.empty()
        for folder_idx, folder in enumerate(folders):
            name = os.path.basename(folder)
            energy, duration = _self.parse_folder_name(name)
            if energy is None: continue
            vtu_files = sorted(glob.glob(os.path.join(folder, "a_t????.vtu")))
            if not vtu_files: continue
            status_text.text(f"Loading {name}... ({len(vtu_files)} files)")
            try:
                mesh0 = meshio.read(vtu_files[0])
                if not mesh0.point_data: continue
                sim_data = {
                    'name': name, 'energy_mJ': energy, 'duration_ns': duration,
                    'n_timesteps': len(vtu_files), 'vtu_files': vtu_files,
                    'field_info': {}, 'has_mesh': False
                }
                if load_full_mesh:
                    points = mesh0.points.astype(np.float32)
                    n_pts = len(points)
                    triangles = None
                    for cell_block in mesh0.cells:
                        if cell_block.type == "triangle":
                            triangles = cell_block.data.astype(np.int32)
                            break
                    fields = {}
                    for key in mesh0.point_data.keys():
                        arr = mesh0.point_data[key].astype(np.float32)
                        if arr.ndim == 1:
                            field_type = "scalar"
                            comp = 1
                        else:
                            field_type = "vector"
                            comp = arr.shape[1]
                        sim_data['field_info'][key] = (field_type, comp)
                        shape = (len(vtu_files), n_pts) if field_type == "scalar" else (len(vtu_files), n_pts, comp)
                        fields[key] = np.full(shape, np.nan, dtype=np.float32)
                        fields[key][0] = arr
                        _self.available_fields.add(key)
                    for t in range(1, len(vtu_files)):
                        try:
                            mesh = meshio.read(vtu_files[t])
                            for key in sim_data['field_info']:
                                if key in mesh.point_data:
                                    fields[key][t] = mesh.point_data[key].astype(np.float32)
                        except: pass
                    sim_data.update({'points': points, 'fields': fields, 'triangles': triangles, 'has_mesh': True})
                summaries.append(_self.extract_summary_statistics(vtu_files, energy, duration, name))
                simulations[name] = sim_data
            except Exception as e:
                st.warning(f"Error loading {name}: {str(e)}")
                continue
            progress_bar.progress((folder_idx + 1) / len(folders))
        progress_bar.empty()
        status_text.empty()
        if simulations: st.success(f"✅ Loaded {len(simulations)} simulations with {len(_self.available_fields)} unique fields")
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
                        summary['field_stats'][field_name] = {'min': [], 'max': [], 'mean': [], 'std': []}
                    if data.ndim == 1:
                        clean_data = data[~np.isnan(data)]
                    else:
                        mag = np.linalg.norm(data, axis=1)
                        clean_data = mag[~np.isnan(mag)]
                    if clean_data.size > 0:
                        summary['field_stats'][field_name]['min'].append(float(np.min(clean_data)))
                        summary['field_stats'][field_name]['max'].append(float(np.max(clean_data)))
                        summary['field_stats'][field_name]['mean'].append(float(np.mean(clean_data)))
                        summary['field_stats'][field_name]['std'].append(float(np.std(clean_data)))
                    else:
                        for key in ['min', 'max', 'mean', 'std']:
                            summary['field_stats'][field_name][key].append(0.0)
            except: continue
        return summary

# =============================================
# 3. ST-DGPA EXTRAPOLATOR
# =============================================
class SpatioTemporalGatedPhysicsAttentionExtrapolator:
    def __init__(self, sigma_param=0.3, spatial_weight=0.5, n_heads=4, temperature=1.0,
                 sigma_g=0.20, s_E=10.0, s_tau=5.0, s_t=20.0, temporal_weight=0.3):
        self.sigma_param = sigma_param; self.spatial_weight = spatial_weight
        self.n_heads = n_heads; self.temperature = temperature
        self.sigma_g = sigma_g; self.s_E = s_E; self.s_tau = s_tau
        self.s_t = s_t; self.temporal_weight = temporal_weight
        self.thermal_diffusivity = 1e-5; self.laser_spot_radius = 50e-6
        self.characteristic_length = 100e-6
        self.source_db = []; self.embedding_scaler = StandardScaler()
        self.value_scaler = StandardScaler()
        self.source_embeddings = []; self.source_values = []
        self.source_metadata = []; self.fitted = False
    
    def load_summaries(self, summaries):
        self.source_db = summaries
        if not summaries: return
        all_embeddings, all_values, metadata = [], [], []
        for summary_idx, summary in enumerate(summaries):
            for timestep_idx, t in enumerate(summary['timesteps']):
                emb = self._compute_enhanced_physics_embedding(summary['energy'], summary['duration'], t)
                all_embeddings.append(emb)
                field_vals = []
                for field in sorted(summary['field_stats'].keys()):
                    stats = summary['field_stats'][field]
                    if timestep_idx < len(stats['mean']):
                        field_vals.extend([stats['mean'][timestep_idx], stats['max'][timestep_idx], stats['std'][timestep_idx]])
                    else: field_vals.extend([0.0, 0.0, 0.0])
                all_values.append(field_vals)
                metadata.append({
                    'summary_idx': summary_idx, 'timestep_idx': timestep_idx,
                    'energy': summary['energy'], 'duration': summary['duration'], 'time': t,
                    'name': summary['name'],
                    'fourier_number': self._compute_fourier_number(t),
                    'thermal_penetration': self._compute_thermal_penetration(t)
                })
        if all_embeddings and all_values:
            all_embeddings = np.array(all_embeddings); all_values = np.array(all_values)
            self.embedding_scaler.fit(all_embeddings)
            self.source_embeddings = self.embedding_scaler.transform(all_embeddings)
            self.value_scaler.fit(all_values)
            self.source_values = all_values
            self.source_metadata = metadata
            self.fitted = True
            st.info(f"✅ Prepared {len(all_embeddings)} embeddings with {all_embeddings.shape[1]} features")
    
    def _compute_fourier_number(self, time_ns):
        time_s = time_ns * 1e-9
        return self.thermal_diffusivity * time_s / (self.characteristic_length ** 2)
    
    def _compute_thermal_penetration(self, time_ns):
        time_s = time_ns * 1e-9
        return np.sqrt(self.thermal_diffusivity * time_s) * 1e6
    
    def _compute_enhanced_physics_embedding(self, energy, duration, time):
        logE = np.log1p(energy); power = energy / max(duration, 1e-6)
        energy_density = energy / (duration * duration + 1e-6)
        time_ratio = time / max(duration, 1e-3)
        heating_rate = power / max(time, 1e-6); cooling_rate = 1.0 / (time + 1e-6)
        thermal_diffusion = np.sqrt(time * 0.1) / max(duration, 1e-3)
        thermal_penetration = np.sqrt(time) / 10.0
        strain_rate = energy_density / (time + 1e-6); stress_rate = power / (time + 1e-6)
        fourier_number = self._compute_fourier_number(time)
        thermal_penetration_depth = self._compute_thermal_penetration(time)
        diffusion_time_scale = time / (duration + 1e-6)
        heating_phase = 1.0 if time < duration else 0.0
        cooling_phase = 1.0 if time >= duration else 0.0
        early_time = 1.0 if time < duration * 0.5 else 0.0
        late_time = 1.0 if time > duration * 2.0 else 0.0
        return np.array([
            logE, duration, time, power, energy_density, time_ratio,
            heating_rate, cooling_rate, thermal_diffusion, thermal_penetration,
            strain_rate, stress_rate, fourier_number, thermal_penetration_depth,
            diffusion_time_scale, heating_phase, cooling_phase, early_time, late_time,
            np.log1p(power), np.log1p(time), np.sqrt(time), time / (duration + 1e-6)
        ], dtype=np.float32)
    
    def _compute_ett_gating(self, energy_query, duration_query, time_query, source_metadata=None):
        if source_metadata is None: source_metadata = self.source_metadata
        phi_squared = []
        for meta in source_metadata:
            de = (energy_query - meta['energy']) / self.s_E
            dt = (duration_query - meta['duration']) / self.s_tau
            dtime = (time_query - meta['time']) / self.s_t
            if self.temporal_weight > 0:
                time_scaling_factor = 1.0 + 0.5 * (time_query / max(duration_query, 1e-6))
                dtime = dtime * time_scaling_factor
            phi_squared.append(de**2 + dt**2 + dtime**2)
        phi_squared = np.array(phi_squared)
        gating = np.exp(-phi_squared / (2 * self.sigma_g**2))
        gating_sum = np.sum(gating)
        if gating_sum > 0: gating = gating / gating_sum
        else: gating = np.ones_like(gating) / len(gating)
        return gating
    
    def _compute_temporal_similarity(self, query_meta, source_metas):
        similarities = []
        for meta in source_metas:
            time_diff = abs(query_meta['time'] - meta['time'])
            if query_meta['time'] < query_meta['duration'] * 1.5:
                temporal_tolerance = max(query_meta['duration'] * 0.1, 1.0)
            else: temporal_tolerance = max(query_meta['duration'] * 0.3, 3.0)
            if 'fourier_number' in meta and 'fourier_number' in query_meta:
                fourier_diff = abs(query_meta['fourier_number'] - meta['fourier_number'])
                fourier_similarity = np.exp(-fourier_diff / 0.1)
            else: fourier_similarity = 1.0
            time_similarity = np.exp(-time_diff / temporal_tolerance)
            combined_similarity = (1 - self.temporal_weight) * time_similarity + self.temporal_weight * fourier_similarity
            similarities.append(combined_similarity)
        return np.array(similarities)
    
    def _compute_spatial_similarity(self, query_meta, source_metas):
        similarities = []
        for meta in source_metas:
            e_diff = abs(query_meta['energy'] - meta['energy']) / 50.0
            d_diff = abs(query_meta['duration'] - meta['duration']) / 20.0
            total_diff = np.sqrt(e_diff**2 + d_diff**2)
            similarity = np.exp(-total_diff / self.sigma_param)
            similarities.append(similarity)
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
        if combined_sum > 1e-12: final_weights = combined_weights / combined_sum
        else: final_weights = physics_attention
        if len(self.source_values) > 0: prediction = np.sum(final_weights[:, np.newaxis] * self.source_values, axis=0)
        else: prediction = np.zeros(1)
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
            'prediction': prediction, 'attention_weights': final_weights,
            'physics_attention': physics_attention, 'ett_gating': ett_gating,
            'confidence': float(np.max(final_weights)) if len(final_weights) > 0 else 0.0,
            'temporal_confidence': self._compute_temporal_confidence(time_query, duration_query),
            'heat_transfer_indicators': self._compute_heat_transfer_indicators(energy_query, duration_query, time_query),
            'field_predictions': {}
        }
        if self.source_db:
            field_order = sorted(self.source_db[0]['field_stats'].keys())
            n_stats_per_field = 3
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
            'phase': phase, 'regime': regime, 'fourier_number': fourier_number,
            'thermal_penetration_um': thermal_penetration,
            'normalized_time': time / max(duration, 1e-6), 'energy_density': energy / duration
        }
    
    def predict_time_series(self, energy_query, duration_query, time_points):
        results = {
            'time_points': time_points, 'field_predictions': {},
            'attention_maps': [], 'physics_attention_maps': [], 'ett_gating_maps': [],
            'confidence_scores': [], 'temporal_confidences': [], 'heat_transfer_indicators': []
        }
        if self.source_db:
            common_fields = set()
            for summary in self.source_db: common_fields.update(summary['field_stats'].keys())
            for field in common_fields: results['field_predictions'][field] = {'mean': [], 'max': [], 'std': []}
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
        if len(field_shape) == 0: interpolated_field = np.zeros(first_sim['fields'][field_name].shape[1], dtype=np.float32)
        else: interpolated_field = np.zeros(field_shape, dtype=np.float32)
        total_weight, n_sources_used = 0.0, 0
        for idx, weight in enumerate(attention_weights):
            if weight < 1e-6: continue
            meta = source_metadata[idx]
            sim_name, timestep_idx = meta['name'], meta['timestep_idx']
            if sim_name in simulations:
                sim = simulations[sim_name]
                if 'fields' in sim and field_name in sim['fields']:
                    source_field = sim['fields'][field_name][timestep_idx]
                    interpolated_field += weight * source_field
                    total_weight += weight; n_sources_used += 1
        if total_weight > 0: interpolated_field /= total_weight
        else: return None
        self.last_interpolation_metadata = {
            'field_name': field_name, 'n_sources_used': n_sources_used,
            'total_weight': total_weight,
            'max_weight': np.max(attention_weights) if len(attention_weights) > 0 else 0,
            'min_weight': np.min(attention_weights) if len(attention_weights) > 0 else 0
        }
        return interpolated_field

# =============================================
# 4. ADVANCED VISUALIZER WITH SANKEY & POLAR RADAR
# =============================================
class EnhancedVisualizer:
    EXTENDED_COLORMAPS = ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'Rainbow', 'Jet', 'Hot', 'Cool', 'Portland']
    
    @staticmethod
    def create_stdgpa_sankey(sources_data, query_params, customization=None):
        """
        Create an enhanced Sankey diagram showing ST-DGPA weight decomposition flow.
        Uses customdata instead of hovertext for Plotly compatibility.
        """
        cfg = customization or {}
        node_pad = cfg.get('node_pad', 20); node_thickness = cfg.get('node_thickness', 25)
        font_size = cfg.get('font_size', 12)
        
        # Build node labels and colors
        labels = ['Target Query']
        node_colors = ['#FF6B6B']  # Red for target
        for src in sources_data:
            labels.append(f"Source {src['source_index']+1}\nE={src['Energy']:.1f}mJ\nτ={src['Duration']:.1f}ns")
            node_colors.append(f'rgba(100,149,237,{src["Combined_Weight"]:.2f})')
        
        # Add component nodes
        component_start = len(labels)
        labels.extend(['Energy Gate', 'Duration Gate', 'Time Gate', 'Physics Attention', 'Refinement', 'Final Weight'])
        node_colors.extend(['#4ECDC4', '#95E1D3', '#FFD93D', '#9D4EDD', '#36A2EB', '#9966FF'])
        
        # Build link data
        s_idx, t_idx, vals, l_colors, h_texts = [], [], [], [], []
        
        for i, src in enumerate(sources_data):
            src_node = i + 1  # Source nodes start at index 1 (0 is target)
            
            # Source → Component gates
            for comp_idx, (comp_name, comp_val, comp_color) in enumerate([
                ('Energy', src['Attention_Score'], '#4ECDC4'),
                ('Duration', src['Attention_Score'], '#95E1D3'),
                ('Time', src['Attention_Score'], '#FFD93D')
            ]):
                s_idx.append(src_node); t_idx.append(component_start + comp_idx)
                vals.append(src['Combined_Weight'] * 0.3); l_colors.append(comp_color)
                h_texts.append(f"<b>{src['Name']}</b> → {comp_name} Gate<br>Score: {comp_val:.4f}")
            
            # Component gates → Physics Attention
            for comp_idx in range(3):
                s_idx.append(component_start + comp_idx); t_idx.append(component_start + 3)
                vals.append(src['Combined_Weight'] * 0.1); l_colors.append('#9D4EDD')
                h_texts.append(f"{labels[component_start+comp_idx]} → Physics Attention<br>Weight: {vals[-1]:.4f}")
            
            # Physics Attention → Refinement
            s_idx.append(component_start + 3); t_idx.append(component_start + 4)
            vals.append(src['Combined_Weight'] * 0.5); l_colors.append('#36A2EB')
            h_texts.append(f"Physics Attention → Refinement<br>Gating: {src['Gating']:.4f}")
            
            # Refinement → Final Weight
            s_idx.append(component_start + 4); t_idx.append(component_start + 5)
            vals.append(src['Combined_Weight']); l_colors.append('#9966FF')
            h_texts.append(f"Refinement → Final Weight<br>Combined: {src['Combined_Weight']:.4f}")
            
            # Final Weight → Target
            s_idx.append(component_start + 5); t_idx.append(0)
            vals.append(src['Combined_Weight'] * 0.8); l_colors.append('#FF6B6B')
            h_texts.append(f"<b>Final Contribution</b><br>Source {src['Name']} → Target<br>Weight: {src['Combined_Weight']:.4f}<br>Formula: wᵢ = (αᵢ×gatingᵢ)/Σ(αⱼ×gatingⱼ)")
        
        # ✅ FIXED: Use customdata instead of hovertext
        fig = go.Figure(go.Sankey(
            node=dict(
                pad=node_pad, thickness=node_thickness,
                line=dict(color="black", width=0.5),
                label=labels, color=node_colors,
                hovertemplate='<b>%{label}</b><extra></extra>'
            ),
            link=dict(
                source=s_idx, target=t_idx, value=vals, color=l_colors,
                customdata=h_texts,  # ✅ CORRECT: customdata instead of hovertext
                hovertemplate='%{customdata}<extra></extra>'  # ✅ CORRECT: reference customdata
            )
        ))
        
        fig.update_layout(
            title=dict(
                text=f"ST-DGPA Weight Decomposition Flow<br><sub>Query: E={query_params['Energy']:.1f}mJ, τ={query_params['Duration']:.1f}ns</sub>",
                font=dict(size=font_size+4), x=0.5
            ),
            font=dict(size=font_size),
            width=cfg.get('width', 1400), height=cfg.get('height', 900),
            plot_bgcolor='rgba(240,240,245,0.9)', paper_bgcolor='white',
            margin=dict(t=100, l=50, r=50, b=50),
            hoverlabel=dict(font_size=font_size-1, bgcolor='rgba(44,62,80,0.9)')
        )
        return fig
    
    @staticmethod
    def create_polar_radar_chart(df, field_type, query_params=None, timestep=1, width=800, height=700):
        """Create polar radar chart with Energy (angular) × Duration (radial) × Peak Value"""
        if df.empty: return go.Figure()
        
        # Convert to polar coordinates
        df['theta'] = np.arctan2(df['Energy'] - df['Energy'].mean(), df['Duration'] - df['Duration'].mean())
        df['r'] = np.sqrt((df['Energy'] - df['Energy'].mean())**2 + (df['Duration'] - df['Duration'].mean())**2)
        
        fig = go.Figure()
        
        # Add data points
        fig.add_trace(go.Scatterpolar(
            r=df['r'], theta=df['theta']*180/np.pi,
            mode='markers',
            marker=dict(
                size=df['Peak_Value'] / df['Peak_Value'].max() * 30 + 10,
                color=df['Peak_Value'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=f"Peak {field_type}")
            ),
            name='Simulations',
            hovertemplate='<b>%{text}</b><br>Energy: %{customdata[0]:.1f} mJ<br>Duration: %{customdata[1]:.1f} ns<br>Peak: %{customdata[2]:.3f}<extra></extra>',
            text=df['Name'],
            customdata=df[['Energy', 'Duration', 'Peak_Value']].values
        ))
        
        # Add query point if provided
        if query_params:
            q_theta = np.arctan2(query_params['Energy'] - df['Energy'].mean(), query_params['Duration'] - df['Duration'].mean())
            q_r = np.sqrt((query_params['Energy'] - df['Energy'].mean())**2 + (query_params['Duration'] - df['Duration'].mean())**2)
            fig.add_trace(go.Scatterpolar(
                r=[q_r], theta=[q_theta*180/np.pi],
                mode='markers',
                marker=dict(size=20, color='red', symbol='star', line=dict(width=2, color='white')),
                name='Query Target',
                hovertemplate='<b>Query</b><br>Energy: %{customdata[0]:.1f} mJ<br>Duration: %{customdata[1]:.1f} ns<extra></extra>',
                customdata=[[query_params['Energy'], query_params['Duration']]]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, title='Parameter Distance'),
                angularaxis=dict(direction='clockwise', tickfont=dict(size=10))
            ),
            title=f"Polar Radar: {field_type} at Timestep {timestep}",
            width=width, height=height,
            showlegend=True, legend=dict(x=1.05, y=1)
        )
        return fig

# =============================================
# 5. MAIN APPLICATION
# =============================================
def main():
    st.set_page_config(page_title="Laser Soldering ST-DGPA Platform", layout="wide", page_icon="🔬")
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; color: #1E3A8A; text-align: center; margin-bottom: 1.5rem; font-weight: 800; }
    .sub-header { font-size: 1.5rem; color: #2c3e50; margin: 1rem 0; padding-bottom: 0.5rem; border-bottom: 2px solid #3498db; }
    .info-box { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 8px; margin: 1rem 0; }
    .success-box { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 1rem; border-radius: 8px; margin: 1rem 0; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🔬 Laser Soldering ST-DGPA Platform</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = UnifiedFEADataLoader()
        st.session_state.extrapolator = SpatioTemporalGatedPhysicsAttentionExtrapolator()
        st.session_state.visualizer = EnhancedVisualizer()
        st.session_state.data_loaded = False
        st.session_state.interpolation_results = None
        st.session_state.interpolation_params = None
        st.session_state.polar_query = None
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Controls")
        if st.button("🔄 Load Simulations", type="primary", width="stretch"):
            with st.spinner("Loading..."):
                sims, sums = st.session_state.data_loader.load_all_simulations(load_full_mesh=True)
                if sims and sums:
                    st.session_state.simulations = sims
                    st.session_state.summaries = sums
                    st.session_state.extrapolator.load_summaries(sums)
                    st.session_state.data_loaded = True
                    st.success(f"✅ Loaded {len(sims)} simulations")
                else: st.error("❌ Load failed")
        
        if st.session_state.data_loaded:
            st.markdown("### 📊 Loaded Data")
            st.metric("Simulations", len(st.session_state.simulations))
            if st.session_state.summaries:
                energies = [s['energy'] for s in st.session_state.summaries]
                durations = [s['duration'] for s in st.session_state.summaries]
                st.metric("Energy Range", f"{min(energies):.1f}-{max(energies):.1f} mJ")
                st.metric("Duration Range", f"{min(durations):.1f}-{max(durations):.1f} ns")
    
    # Main tabs
    tabs = st.tabs(["📊 Data Overview", "🔮 Interpolation", "🎯 Polar Radar", "🕸️ Sankey Diagram", "🧠 ST-DGPA Analysis"])
    
    if st.session_state.get('data_loaded') and st.session_state.get('summaries'):
        # --- TAB 1: Data Overview ---
        with tabs[0]:
            st.subheader("Loaded Simulations Summary")
            df_summary = pd.DataFrame([{
                'Name': s['name'], 'Energy (mJ)': s['energy'],
                'Duration (ns)': s['duration'], 'Timesteps': len(s['timesteps'])
            } for s in st.session_state.summaries])
            st.dataframe(df_summary.style.format({'Energy (mJ)': '{:.2f}', 'Duration (ns)': '{:.2f}'}), width="stretch")
            
            # 3D viewer
            if st.session_state.simulations and next(iter(st.session_state.simulations.values())).get('has_mesh'):
                sim_name = st.selectbox("Select Simulation", sorted(st.session_state.simulations.keys()))
                sim = st.session_state.simulations[sim_name]
                field = st.selectbox("Select Field", sorted(sim['field_info'].keys()))
                timestep = st.slider("Timestep", 0, sim['n_timesteps']-1, 0)
                if sim['points'] is not None and field in sim['fields']:
                    values = sim['fields'][field][timestep].copy()
                    if values.ndim >= 2:
                        norm_axes = tuple(range(1, values.ndim))
                        values = np.linalg.norm(values, axis=norm_axes)
                    values = np.nan_to_num(values, nan=0.0)
                    fig = go.Figure(go.Mesh3d(
                        x=sim['points'][:,0], y=sim['points'][:,1], z=sim['points'][:,2],
                        i=sim['triangles'][:,0], j=sim['triangles'][:,1], k=sim['triangles'][:,2],
                        intensity=values, colorscale='Viridis'
                    ))
                    fig.update_layout(scene=dict(aspectmode="data"), height=600)
                    st.plotly_chart(fig, width="stretch")
        
        # --- TAB 2: Interpolation ---
        with tabs[1]:
            st.subheader("Run ST-DGPA Interpolation")
            c1, c2, c3 = st.columns(3)
            with c1: q_E = st.number_input("Energy (mJ)", 0.1, 20.0, 3.5, 0.1)
            with c2: q_τ = st.number_input("Duration (ns)", 0.1, 20.0, 4.2, 0.1)
            with c3: max_t = st.number_input("Max Time (ns)", 1, 200, 50, 1)
            time_points = np.arange(1, max_t + 1, 1)
            
            if st.button("🚀 Run Prediction", type="primary", width="stretch"):
                with st.spinner("Computing ST-DGPA..."):
                    results = st.session_state.extrapolator.predict_time_series(q_E, q_τ, time_points)
                    if results and results['field_predictions']:
                        st.session_state.interpolation_results = results
                        st.session_state.interpolation_params = {'energy_query': q_E, 'duration_query': q_τ, 'time_points': time_points}
                        st.success("✅ Prediction Complete")
                        st.session_state.polar_query = {'Energy': q_E, 'Duration': q_τ}
                    else: st.error("Prediction failed.")
            
            if st.session_state.interpolation_results:
                results = st.session_state.interpolation_results
                st.subheader("Prediction Results")
                fields = list(results['field_predictions'].keys())
                if fields:
                    fig_preds = go.Figure()
                    for f in fields[:3]:
                        if results['field_predictions'][f]['mean']:
                            fig_preds.add_trace(go.Scatter(x=time_points, y=results['field_predictions'][f]['mean'], mode='lines', name=f))
                    fig_preds.update_layout(title="Predicted Mean Values", xaxis_title="Time (ns)", yaxis_title="Value", height=400)
                    st.plotly_chart(fig_preds, width="stretch")
        
        # --- TAB 3: Polar Radar ---
        with tabs[2]:
            st.subheader("Polar Radar Visualization")
            if not st.session_state.get('data_loaded'):
                st.warning("Please load simulations first.")
            else:
                all_fields = set()
                for s in st.session_state.summaries: all_fields.update(s['field_stats'].keys())
                field_type = st.selectbox("Field Type", sorted(all_fields))
                t_step = st.number_input("Timestep Index", 1, max(s.get('timesteps', [1])[-1] for s in st.session_state.summaries), 1)
                show_target = st.checkbox("Show Target Query", value=True)
                
                rows = []
                for s in st.session_state.summaries:
                    if t_step <= len(s['timesteps']):
                        idx = t_step - 1
                        peak = s['field_stats'].get(field_type, {}).get('max', [0])
                        if idx < len(peak):
                            rows.append({'Name': s['name'], 'Energy': s['energy'], 'Duration': s['duration'], 'Peak_Value': peak[idx]})
                df_polar = pd.DataFrame(rows)
                
                if df_polar['Energy'].nunique() <= 1:
                    st.info("All loaded simulations use the same energy. The polar chart will show a collapsed angular axis.")
                
                query_params = st.session_state.polar_query if show_target and st.session_state.get('polar_query') else None
                fig_polar = st.session_state.visualizer.create_polar_radar_chart(df_polar, field_type, query_params, timestep=t_step)
                st.plotly_chart(fig_polar, width="stretch")
                
                if st.checkbox("Show Multiple Timesteps (1, 3, 5)"):
                    c1, c2, c3 = st.columns(3)
                    for col, t_idx in zip([c1, c2, c3], [1, 3, 5]):
                        if t_idx <= max(len(s.get('timesteps', [])) for s in st.session_state.summaries):
                            rows_t = []
                            for s in st.session_state.summaries:
                                if t_idx <= len(s['timesteps']):
                                    peak_t = s['field_stats'].get(field_type, {}).get('max', [0])
                                    if t_idx-1 < len(peak_t): rows_t.append({'Name': s['name'], 'Energy': s['energy'], 'Duration': s['duration'], 'Peak_Value': peak_t[t_idx-1]})
                            fig_t = st.session_state.visualizer.create_polar_radar_chart(pd.DataFrame(rows_t), field_type, query_params, timestep=t_idx, width=300, height=300)
                            with col: st.plotly_chart(fig_t, width="stretch")
        
        # --- TAB 4: Sankey Diagram ---
        with tabs[3]:
            st.subheader("ST-DGPA Sankey Diagram")
            if st.session_state.interpolation_results and st.session_state.get('interpolation_params'):
                res = st.session_state.interpolation_results
                params = st.session_state.interpolation_params
                q_E = params['energy_query']; q_τ = params['duration_query']
                
                t_sel = st.slider("Select Timestep for Sankey", 1, len(res['attention_maps']), 1)
                t_idx = t_sel - 1
                
                sources_data = []
                weights = res['attention_maps'][t_idx]
                phys_att = res['physics_attention_maps'][t_idx]
                gating = res['ett_gating_maps'][t_idx]
                
                for i in range(len(st.session_state.summaries)):
                    meta = st.session_state.extrapolator.source_metadata[i]
                    if meta['timestep_idx'] == t_idx:
                        sources_data.append({
                            'Name': meta['name'], 'Energy': meta['energy'], 'Duration': meta['duration'], 'Time': meta['time'],
                            'Attention_Score': phys_att[i], 'Gating': gating[i],
                            'Refinement': phys_att[i] * gating[i], 'Combined_Weight': weights[i]
                        })
                
                if sources_data:
                    query = {'Energy': q_E, 'Duration': q_τ, 'Time': t_sel}
                    customization = {'font_size': 11, 'node_thickness': 15}
                    fig_sankey = st.session_state.visualizer.create_stdgpa_sankey(sources_data, query, customization)
                    st.plotly_chart(fig_sankey, width="stretch")
                    st.info("Hover over flows to see mathematical formulas and weight breakdown.")
                else:
                    st.warning("No source data matches the selected timestep. Try a different timestep.")
            else:
                st.info("Please run an interpolation first to see the Sankey diagram.")
        
        # --- TAB 5: ST-DGPA Analysis ---
        with tabs[4]:
            st.subheader("ST-DGPA Attention & Physics Analysis")
            if st.session_state.interpolation_results and st.session_state.get('interpolation_params'):
                res = st.session_state.interpolation_results
                params = st.session_state.interpolation_params
                # Simple attention visualization
                t_idx = st.slider("Analysis Timestep", 0, len(res['attention_maps'])-1, 0)
                weights = res['attention_maps'][t_idx]
                if len(weights) > 0:
                    fig = go.Figure(data=go.Bar(x=list(range(len(weights))), y=weights))
                    fig.update_layout(title=f"ST-DGPA Weights at t={params['time_points'][t_idx]} ns", xaxis_title="Source Index", yaxis_title="Weight")
                    st.plotly_chart(fig, width="stretch")
            else:
                st.info("Please run an interpolation first.")
    else:
        st.info("👈 Please load simulations using the sidebar to begin analysis.")

if __name__ == "__main__":
    main()
