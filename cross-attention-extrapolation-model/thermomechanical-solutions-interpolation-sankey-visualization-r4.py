#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ENHANCED FEA LASER SIMULATION PLATFORM WITH ST-DGPA & SANKEY DIAGRAM
====================================================================
- Full ST-DGPA interpolation/extrapolation
- Advanced 3D field rendering with caching
- 🔥 NEW: Physics-aware Sankey diagram for weight decomposition
- Robust error handling & UI improvements
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
                if not mesh0.point_data:
                    st.warning(f"No point data in {name}")
                    continue
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
                        sim_data['field_info'][key] = ("scalar", 1) if arr.ndim == 1 else ("vector", arr.shape[1])
                        shape = (len(vtu_files), n_pts) if arr.ndim == 1 else (len(vtu_files), n_pts, arr.shape[1])
                        fields[key] = np.full(shape, np.nan, dtype=np.float32)
                        fields[key][0] = arr
                        _self.available_fields.add(key)
                    for t in range(1, len(vtu_files)):
                        try:
                            mesh = meshio.read(vtu_files[t])
                            for key in sim_data['field_info']:
                                if key in mesh.point_data: fields[key][t] = mesh.point_data[key].astype(np.float32)
                        except Exception as e:
                            st.warning(f"Error loading timestep {t} in {name}: {e}")
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
                        summary['field_stats'][field_name] = {'min': [], 'max': [], 'mean': [], 'std': [], 'q25': [], 'q50': [], 'q75': [], 'percentiles': []}
                    if data.ndim == 1: clean_data = data[~np.isnan(data)]
                    else: clean_data = np.linalg.norm(data, axis=1)[~np.isnan(np.linalg.norm(data, axis=1))]
                    if clean_data.size > 0:
                        for key in ['min', 'max', 'mean', 'std', 'q25', 'q50', 'q75']: summary['field_stats'][field_name][key].append(float(eval(f"np.{key}(clean_data)")))
                        summary['field_stats'][field_name]['percentiles'].append(np.percentile(clean_data, [10, 25, 50, 75, 90]))
                    else:
                        for key in ['min', 'max', 'mean', 'std', 'q25', 'q50', 'q75']: summary['field_stats'][field_name][key].append(0.0)
                        summary['field_stats'][field_name]['percentiles'].append(np.zeros(5))
            except Exception as e:
                st.warning(f"Error processing {vtu_file}: {e}")
                continue
        return summary

# =============================================
# SPATIO-TEMPORAL GATED PHYSICS ATTENTION (ST-DGPA)
# =============================================
class SpatioTemporalGatedPhysicsAttentionExtrapolator:
    def __init__(self, sigma_param=0.3, spatial_weight=0.5, n_heads=4, temperature=1.0,
                 sigma_g=0.20, s_E=10.0, s_tau=5.0, s_t=20.0, temporal_weight=0.3):
        self.sigma_param = sigma_param; self.spatial_weight = spatial_weight
        self.n_heads = n_heads; self.temperature = temperature
        self.sigma_g = sigma_g; self.s_E = s_E; self.s_tau = s_tau; self.s_t = s_t
        self.temporal_weight = temporal_weight
        self.thermal_diffusivity = 1e-5; self.laser_spot_radius = 50e-6; self.characteristic_length = 100e-6
        self.source_db = []; self.embedding_scaler = StandardScaler()
        self.value_scaler = StandardScaler(); self.source_embeddings = []
        self.source_values = []; self.source_metadata = []; self.fitted = False
    
    def load_summaries(self, summaries):
        self.source_db = summaries
        if not summaries: return
        all_embeddings, all_values, metadata = [], [], []
        for summary_idx, summary in enumerate(summaries):
            for timestep_idx, t in enumerate(summary['timesteps']):
                all_embeddings.append(self._compute_enhanced_physics_embedding(summary['energy'], summary['duration'], t))
                field_vals = []
                for field in sorted(summary['field_stats'].keys()):
                    stats = summary['field_stats'][field]
                    if timestep_idx < len(stats['mean']): field_vals.extend([stats['mean'][timestep_idx], stats['max'][timestep_idx], stats['std'][timestep_idx]])
                    else: field_vals.extend([0.0, 0.0, 0.0])
                all_values.append(field_vals)
                metadata.append({'summary_idx': summary_idx, 'timestep_idx': timestep_idx, 'energy': summary['energy'], 'duration': summary['duration'], 'time': t, 'name': summary['name'], 'fourier_number': self._compute_fourier_number(t), 'thermal_penetration': self._compute_thermal_penetration(t)})
        if all_embeddings and all_values:
            all_embeddings, all_values = np.array(all_embeddings), np.array(all_values)
            self.embedding_scaler.fit(all_embeddings); self.source_embeddings = self.embedding_scaler.transform(all_embeddings)
            self.value_scaler.fit(all_values); self.source_values = all_values
            self.source_metadata = metadata; self.fitted = True
            st.info(f"✅ Prepared {len(all_embeddings)} embeddings with {all_embeddings.shape[1]} features")
    
    def _compute_fourier_number(self, time_ns): return self.thermal_diffusivity * (time_ns * 1e-9) / (self.characteristic_length ** 2)
    def _compute_thermal_penetration(self, time_ns): return np.sqrt(self.thermal_diffusivity * (time_ns * 1e-9)) * 1e6
    
    def _compute_enhanced_physics_embedding(self, energy, duration, time):
        logE = np.log1p(energy); power = energy / max(duration, 1e-6); time_ratio = time / max(duration, 1e-3)
        fourier_number = self._compute_fourier_number(time); thermal_penetration = self._compute_thermal_penetration(time)
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
            if self.temporal_weight > 0: dtime = dtime * (1.0 + 0.5 * (time_query / max(duration_query, 1e-6)))
            phi_squared.append(de**2 + dt**2 + dtime**2)
        phi_squared = np.array(phi_squared)
        gating = np.exp(-phi_squared / (2 * self.sigma_g**2))
        gating_sum = np.sum(gating)
        return gating / gating_sum if gating_sum > 0 else np.ones_like(gating) / len(gating)
    
    def _compute_temporal_similarity(self, query_meta, source_metas):
        similarities = []
        for meta in source_metas:
            time_diff = abs(query_meta['time'] - meta['time'])
            tolerance = max(query_meta['duration'] * 0.1, 1.0) if query_meta['time'] < query_meta['duration'] * 1.5 else max(query_meta['duration'] * 0.3, 3.0)
            fourier_similarity = np.exp(-abs(query_meta.get('fourier_number', 0) - meta.get('fourier_number', 0)) / 0.1)
            time_similarity = np.exp(-time_diff / tolerance)
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
        n_sources = len(self.source_embeddings); head_weights = np.zeros((self.n_heads, n_sources))
        for head in range(self.n_heads):
            np.random.seed(42 + head); proj_dim = min(8, query_norm.shape[0])
            proj_matrix = np.random.randn(query_norm.shape[0], proj_dim)
            query_proj = query_norm @ proj_matrix; source_proj = self.source_embeddings @ proj_matrix
            distances = np.linalg.norm(query_proj - source_proj, axis=1)
            scores = np.exp(-distances**2 / (2 * self.sigma_param**2))
            if self.spatial_weight > 0: scores = (1 - self.spatial_weight) * scores + self.spatial_weight * self._compute_spatial_similarity(query_meta, self.source_metadata)
            if self.temporal_weight > 0: scores = (1 - self.temporal_weight) * scores + self.temporal_weight * self._compute_temporal_similarity(query_meta, self.source_metadata)
            head_weights[head] = scores
        avg_weights = np.mean(head_weights, axis=0)
        if self.temperature != 1.0: avg_weights = avg_weights ** (1.0 / self.temperature)
        max_weight = np.max(avg_weights); exp_weights = np.exp(avg_weights - max_weight)
        physics_attention = exp_weights / (np.sum(exp_weights) + 1e-12)
        ett_gating = self._compute_ett_gating(query_meta['energy'], query_meta['duration'], query_meta['time'])
        combined_weights = physics_attention * ett_gating; combined_sum = np.sum(combined_weights)
        final_weights = combined_weights / combined_sum if combined_sum > 1e-12 else physics_attention
        prediction = np.sum(final_weights[:, np.newaxis] * self.source_values, axis=0) if len(self.source_values) > 0 else np.zeros(1)
        return prediction, final_weights, physics_attention, ett_gating
    
    def predict_field_statistics(self, energy_query, duration_query, time_query):
        if not self.fitted: return None
        query_embedding = self._compute_enhanced_physics_embedding(energy_query, duration_query, time_query)
        query_meta = {'energy': energy_query, 'duration': duration_query, 'time': time_query, 'fourier_number': self._compute_fourier_number(time_query), 'thermal_penetration': self._compute_thermal_penetration(time_query)}
        prediction, final_weights, physics_attention, ett_gating = self._multi_head_attention_with_gating(query_embedding, query_meta)
        if prediction is None: return None
        result = {'prediction': prediction, 'attention_weights': final_weights, 'physics_attention': physics_attention, 'ett_gating': ett_gating,
                  'confidence': float(np.max(final_weights)) if len(final_weights) > 0 else 0.0,
                  'temporal_confidence': self._compute_temporal_confidence(time_query, duration_query),
                  'heat_transfer_indicators': self._compute_heat_transfer_indicators(energy_query, duration_query, time_query), 'field_predictions': {}}
        if self.source_db:
            field_order = sorted(self.source_db[0]['field_stats'].keys())
            for i, field in enumerate(field_order):
                start_idx = i * 3
                if start_idx + 2 < len(prediction): result['field_predictions'][field] = {'mean': float(prediction[start_idx]), 'max': float(prediction[start_idx + 1]), 'std': float(prediction[start_idx + 2])}
        return result
    
    def _compute_temporal_confidence(self, time_query, duration_query):
        if time_query < duration_query * 0.5: return 0.6
        elif time_query < duration_query * 1.5: return 0.8
        else: return 0.9
    
    def _compute_heat_transfer_indicators(self, energy, duration, time):
        fourier_number = self._compute_fourier_number(time); thermal_penetration = self._compute_thermal_penetration(time)
        if time < duration * 0.3: phase, regime = "Early Heating", "Adiabatic-like"
        elif time < duration: phase, regime = "Heating", "Conduction-dominated"
        elif time < duration * 2: phase, regime = "Early Cooling", "Mixed conduction"
        else: phase, regime = "Diffusion Cooling", "Thermal diffusion"
        return {'phase': phase, 'regime': regime, 'fourier_number': fourier_number, 'thermal_penetration_um': thermal_penetration, 'normalized_time': time / max(duration, 1e-6), 'energy_density': energy / duration}
    
    def predict_time_series(self, energy_query, duration_query, time_points):
        results = {'time_points': time_points, 'field_predictions': {}, 'attention_maps': [], 'physics_attention_maps': [], 'ett_gating_maps': [], 'confidence_scores': [], 'temporal_confidences': [], 'heat_transfer_indicators': []}
        if self.source_db:
            for field in set(f for s in self.source_db for f in s['field_stats'].keys()): results['field_predictions'][field] = {'mean': [], 'max': [], 'std': []}
        for t in time_points:
            pred = self.predict_field_statistics(energy_query, duration_query, t)
            if pred and 'field_predictions' in pred:
                for field in pred['field_predictions']:
                    if field in results['field_predictions']: results['field_predictions'][field]['mean'].append(pred['field_predictions'][field]['mean']); results['field_predictions'][field]['max'].append(pred['field_predictions'][field]['max']); results['field_predictions'][field]['std'].append(pred['field_predictions'][field]['std'])
                results['attention_maps'].append(pred['attention_weights']); results['physics_attention_maps'].append(pred['physics_attention']); results['ett_gating_maps'].append(pred['ett_gating']); results['confidence_scores'].append(pred['confidence']); results['temporal_confidences'].append(pred['temporal_confidence']); results['heat_transfer_indicators'].append(pred['heat_transfer_indicators'])
            else:
                for field in results['field_predictions']: results['field_predictions'][field]['mean'].append(np.nan); results['field_predictions'][field]['max'].append(np.nan); results['field_predictions'][field]['std'].append(np.nan)
                results['attention_maps'].append(np.array([])); results['physics_attention_maps'].append(np.array([])); results['ett_gating_maps'].append(np.array([])); results['confidence_scores'].append(0.0); results['temporal_confidences'].append(0.0); results['heat_transfer_indicators'].append({})
        return results
    
    def interpolate_full_field(self, field_name, attention_weights, source_metadata, simulations):
        if not self.fitted or len(attention_weights) == 0: return None
        first_sim = next(iter(simulations.values()))
        if 'fields' not in first_sim or field_name not in first_sim['fields']: return None
        field_shape = first_sim['fields'][field_name].shape[1:]
        interpolated_field = np.zeros(first_sim['fields'][field_name].shape[1] if len(field_shape) == 0 else field_shape, dtype=np.float32)
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
        times = np.array([meta['time'] for meta in source_metadata]); weighted_times = times * attention_weights
        mean_time = np.sum(weighted_times) / np.sum(attention_weights)
        time_diff = times - mean_time; weighted_variance = np.sum(attention_weights * time_diff**2) / np.sum(attention_weights)
        return float(np.exp(-np.sqrt(weighted_variance) / max(np.mean([meta['duration'] for meta in source_metadata]), 1e-6)))
    
    def export_interpolated_vtu(self, field_name, interpolated_values, simulations, output_path):
        if interpolated_values is None or len(simulations) == 0: return False
        try:
            first_sim = next(iter(simulations.values())); points = first_sim['points']; cells = None
            if 'triangles' in first_sim and first_sim['triangles'] is not None: cells = [("triangle", first_sim['triangles'])]
            mesh = meshio.Mesh(points, cells, point_data={field_name: interpolated_values}); mesh.write(output_path); return True
        except Exception as e: st.error(f"Error exporting VTU: {str(e)}"); return False

# =============================================
# ADVANCED VISUALIZATION COMPONENTS WITH ST-DGPA ANALYSIS
# =============================================
class EnhancedVisualizer:
    COLORSCALES = {'temperature': ['#2c0078', '#4402a7', '#5e04d1', '#7b0ef6', '#9a38ff', '#b966ff', '#d691ff', '#f2bcff'], 'stress': ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600'], 'displacement': ['#004c6d', '#346888', '#5886a5', '#7aa6c2', '#9dc6e0', '#c1e7ff'], 'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']}
    EXTENDED_COLORMAPS = ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'Rainbow', 'Jet', 'Hot', 'Cool', 'Portland', 'Bluered', 'Electric', 'Thermal', 'Balance', 'Brwnyl', 'Darkmint', 'Emrld', 'Mint', 'Oranges', 'Purp', 'Purples', 'Sunset', 'Sunsetdark', 'Teal', 'Tealgrn', 'Twilight', 'Burg', 'Burgyl']

    @staticmethod
    def create_stdgpa_analysis(results, energy_query, duration_query, time_points):
        if not results or 'attention_maps' not in results or len(results['attention_maps']) == 0: return None
        timestep_idx = len(time_points) // 2; time = time_points[timestep_idx]
        fig = make_subplots(rows=3, cols=3, subplot_titles=["ST-DGPA Final Weights", "Physics Attention Only", "(E, τ, t) Gating Only", "ST-DGPA vs Physics Attention", "Temporal Coherence Analysis", "Heat Transfer Phase", "Parameter Space 3D", "Attention Network", "Weight Evolution"], vertical_spacing=0.12, horizontal_spacing=0.12, specs=[ [{'type': 'xy'}, {'type': 'xy'}, {'type': 'xy'}], [{'type': 'xy'}, {'type': 'xy'}, {'type': 'polar'}], [{'type': 'scene'}, {'type': 'xy'}, {'type': 'xy'}] ])
        final_weights = results['attention_maps'][timestep_idx]; physics_attention = results['physics_attention_maps'][timestep_idx]; ett_gating = results['ett_gating_maps'][timestep_idx]
        fig.add_trace(go.Bar(x=list(range(len(final_weights))), y=final_weights, name='ST-DGPA Weights', marker_color='blue', showlegend=False), row=1, col=1)
        fig.add_trace(go.Bar(x=list(range(len(physics_attention))), y=physics_attention, name='Physics Attention', marker_color='green', showlegend=False), row=1, col=2)
        fig.add_trace(go.Bar(x=list(range(len(ett_gating))), y=ett_gating, name='(E, τ, t) Gating', marker_color='red', showlegend=False), row=1, col=3)
        fig.add_trace(go.Scatter(x=list(range(len(final_weights))), y=final_weights, mode='lines+markers', name='ST-DGPA Weights', line=dict(color='blue', width=3)), row=2, col=1)
        fig.add_trace(go.Scatter(x=list(range(len(physics_attention))), y=physics_attention, mode='lines+markers', name='Physics Attention', line=dict(color='green', width=2, dash='dash')), row=2, col=1)
        if st.session_state.get('summaries') and hasattr(st.session_state.extrapolator, 'source_metadata'):
            times, weights = [], []
            for i, weight in enumerate(final_weights):
                if weight > 0.01 and i < len(st.session_state.extrapolator.source_metadata): times.append(st.session_state.extrapolator.source_metadata[i]['time']); weights.append(weight)
            if times and weights: fig.add_trace(go.Scatter(x=times, y=weights, mode='markers', marker=dict(size=np.array(weights)*50, color=weights, colorscale='Viridis', showscale=False)), row=2, col=2); fig.add_vline(x=time, line_dash="dash", line_color="red", row=2, col=2)
        if 'heat_transfer_indicators' in results and results['heat_transfer_indicators']:
            indicators = results['heat_transfer_indicators'][timestep_idx]
            if indicators:
                phase = indicators.get('phase', 'Unknown'); values = {'Early Heating': [0.9, 0.3, 0.2, 0.1], 'Heating': [0.9, 0.3, 0.2, 0.1], 'Early Cooling': [0.4, 0.8, 0.3, 0.1], 'Diffusion Cooling': [0.2, 0.5, 0.9, 0.2]}.get(phase, [0.7, 0.5, 0.3, 0.2])
                values_closed = values + [values[0]]; categories_closed = ['Heating', 'Cooling', 'Diffusion', 'Adiabatic'] * 2; categories_closed.pop(4)
                fig.add_trace(go.Scatterpolar(r=values_closed, theta=categories_closed, fill='toself', fillcolor='rgba(255, 165, 0, 0.5)', line=dict(color='#f39c12', width=2), showlegend=False), row=2, col=3)
        if st.session_state.get('summaries'):
            energies, durations, times_3d, weights_3d = [], [], [], []
            for summary in st.session_state.summaries[:10]:
                for t in summary['timesteps'][:5]: energies.append(summary['energy']); durations.append(summary['duration']); times_3d.append(t); weights_3d.append(np.mean(final_weights) if len(final_weights) > 0 else 0.1)
            fig.add_trace(go.Scatter3d(x=energies, y=durations, z=times_3d, mode='markers', marker=dict(size=np.array(weights_3d)*20, color=weights_3d, colorscale='Viridis', opacity=0.7, colorbar=dict(title="Weight", x=1.05)), showlegend=False), row=3, col=1)
            fig.add_trace(go.Scatter3d(x=[energy_query], y=[duration_query], z=[time], mode='markers', marker=dict(size=15, color='red', symbol='diamond'), showlegend=False), row=3, col=1)
        if len(final_weights) > 5:
            top_indices = np.argsort(final_weights)[-5:]; top_weights = final_weights[top_indices]
            fig.add_trace(go.Scatter(x=[0] + list(range(1, 6)), y=[0]*6, mode='markers+text', text=['Query'] + [f'Source {i+1}' for i in top_indices], textposition="top center", marker=dict(size=[30] + list(top_weights*50), color=['red'] + ['blue']*5), showlegend=False), row=3, col=2)
            for i in range(1, 6): fig.add_trace(go.Scatter(x=[0, i], y=[0, 0], mode='lines', line=dict(width=top_weights[i-1]*10, color='gray'), showlegend=False), row=3, col=2)
        if len(results['attention_maps']) > 1:
            top_idx = np.argmax(final_weights); weight_evolution = [results['attention_maps'][t_idx][top_idx] for t_idx in range(len(results['attention_maps'])) if top_idx < len(results['attention_maps'][t_idx])]
            if weight_evolution: fig.add_trace(go.Scatter(x=time_points[:len(weight_evolution)], y=weight_evolution, mode='lines+markers', line=dict(color='purple', width=3), showlegend=False), row=3, col=3)
        fig.update_layout(height=1000, title_text=f"ST-DGPA Analysis at t={time} ns (E={energy_query:.1f} mJ, τ={duration_query:.1f} ns)", showlegend=True, legend=dict(x=1.05, y=1))
        return fig

    @staticmethod
    def create_temporal_analysis(results, time_points):
        if not results or 'heat_transfer_indicators' not in results: return None
        fig = make_subplots(rows=2, cols=2, subplot_titles=["Heat Transfer Phase Evolution", "Fourier Number Evolution", "Temporal Confidence", "Thermal Penetration Depth"], vertical_spacing=0.15, horizontal_spacing=0.15)
        phases = [ind.get('phase', 'Unknown') for ind in results['heat_transfer_indicators'] if ind]
        phase_values = [{'Early Heating': 0, 'Heating': 1, 'Early Cooling': 2, 'Diffusion Cooling': 3}.get(p, 0) for p in phases]
        fig.add_trace(go.Scatter(x=time_points[:len(phase_values)], y=phase_values, mode='lines+markers', line=dict(color='red', width=3)), row=1, col=1)
        for pname, pval in {'Early Heating': 0, 'Heating': 1, 'Early Cooling': 2, 'Diffusion Cooling': 3}.items(): fig.add_hline(y=pval, line_dash="dot", line_color="gray", annotation_text=pname, row=1, col=1)
        fourier_numbers = [ind.get('fourier_number', 0) for ind in results['heat_transfer_indicators'] if ind]
        if fourier_numbers: fig.add_trace(go.Scatter(x=time_points[:len(fourier_numbers)], y=fourier_numbers, mode='lines+markers', line=dict(color='blue', width=3)), row=1, col=2)
        if 'temporal_confidences' in results: fig.add_trace(go.Scatter(x=time_points[:len(results['temporal_confidences'])], y=results['temporal_confidences'], mode='lines+markers', line=dict(color='green', width=3), fill='tozeroy', fillcolor='rgba(0, 255, 0, 0.2)'), row=2, col=1)
        penetration_depths = [ind.get('thermal_penetration_um', 0) for ind in results['heat_transfer_indicators'] if ind]
        if penetration_depths: fig.add_trace(go.Scatter(x=time_points[:len(penetration_depths)], y=penetration_depths, mode='lines+markers', line=dict(color='orange', width=3)), row=2, col=2)
        fig.update_layout(height=700, title_text="Temporal Analysis of Heat Transfer Characteristics", showlegend=False)
        return fig

# =============================================
# SANKEY VISUALIZATION ENGINE
# =============================================
def create_stdgpa_sankey(results, energy_query, duration_query, selected_time, source_metadata, attention_weights, physics_attention, ett_gating, temporal_sim=None, top_k=12):
    if len(attention_weights) == 0: return go.Figure()
    top_indices = np.argsort(attention_weights)[-top_k:][::-1]
    top_weights = attention_weights[top_indices]; top_physics = physics_attention[top_indices]; top_gating = ett_gating[top_indices]
    top_temporal = temporal_sim[top_indices] if temporal_sim is not None else np.ones_like(top_weights) * 0.5
    labels = ["Target Query"]
    for i, idx in enumerate(top_indices):
        meta = source_metadata[idx]; labels.append(f"S{idx}\nE={meta['energy']:.1f}mJ\nτ={meta['duration']:.1f}ns\nt={meta['time']:.1f}ns")
    component_start = len(labels); labels.extend(["Physics Attention", "(E, τ, t) Gating", "Temporal Similarity", "ST-DGPA Final Weight"])
    source_idx, target_idx, values, link_colors = [], [], [], []
    for i, src_idx in enumerate(top_indices):
        s_node = i + 1
        source_idx.append(s_node); target_idx.append(component_start); values.append(top_physics[i] * 100); link_colors.append("rgba(46, 204, 113, 0.85)")
        source_idx.append(s_node); target_idx.append(component_start + 1); values.append(top_gating[i] * 100); link_colors.append("rgba(231, 76, 60, 0.85)")
        source_idx.append(s_node); target_idx.append(component_start + 2); values.append(top_temporal[i] * 100); link_colors.append("rgba(52, 152, 219, 0.85)")
        source_idx.append(s_node); target_idx.append(component_start + 3); values.append(top_weights[i] * 100); link_colors.append("rgba(155, 89, 182, 0.9)")
    for c in range(4):
        comp_node = component_start + c; agg_value = sum(v for s, t, v in zip(source_idx, target_idx, values) if t == comp_node)
        if agg_value > 0: source_idx.append(comp_node); target_idx.append(0); values.append(agg_value * 0.6); link_colors.append("rgba(149, 165, 166, 0.7)")
    node_colors = ["#FF6B6B"] + ["rgba(155, 89, 182, 0.9)"] * len(top_indices) + ["#2ecc71", "#e74c3c", "#3498db", "#9b59b6"]
    fig = go.Figure(data=[go.Sankey(node=dict(pad=20, thickness=30, line=dict(color="black", width=1), label=labels, color=node_colors, hovertemplate='<b>%{label}</b><br>Value: %{value:.2f}<extra></extra>'), link=dict(source=source_idx, target=target_idx, value=values, color=link_colors, hovertemplate='<b>%{source.label}</b> → <b>%{target.label}</b><br>Flow: %{value:.2f}%<extra></extra>'))])
    fig.update_layout(title=dict(text=f"<b>ST-DGPA Weight Decomposition</b><br>Query: E={energy_query:.1f} mJ, τ={duration_query:.1f} ns, t={selected_time:.1f} ns", font=dict(size=22, family="Arial", color="#2c3e50"), x=0.5, y=0.95), font=dict(family="Arial", size=14), width=1350, height=850, plot_bgcolor='rgba(248, 249, 250, 0.95)', paper_bgcolor='white', margin=dict(t=120, l=50, r=50, b=50))
    return fig

# =============================================
# MAIN INTEGRATED APPLICATION
# =============================================
def main():
    st.set_page_config(page_title="Enhanced FEA Laser Simulation Platform with ST-DGPA", layout="wide", initial_sidebar_state="expanded", page_icon="🔬")
    st.markdown("""
    <style>
    .main-header { font-size: 3rem; background: linear-gradient(90deg, #1E88E5, #4A00E0); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; margin-bottom: 1.5rem; font-weight: 800; }
    .sub-header { font-size: 1.8rem; color: #2c3e50; margin-top: 1.5rem; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 3px solid #3498db; font-weight: 600; }
    .info-box { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 10px; margin: 1.5rem 0; }
    .warning-box { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 1.5rem; border-radius: 10px; margin: 1.5rem 0; }
    .success-box { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 1.5rem; border-radius: 10px; margin: 1.5rem 0; }
    .stdgpa-box { background: linear-gradient(135deg, #f093fb 0%, #00f2fe 100%); color: white; padding: 1.5rem; border-radius: 10px; margin: 1.5rem 0; }
    .heat-transfer-box { background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%); color: #333; padding: 1.5rem; border-radius: 10px; margin: 1.5rem 0; }
    .metric-card { background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 1rem; }
    .cache-status { background: #e8f5e8; border-left: 4px solid #4CAF50; padding: 0.75rem; margin: 0.5rem 0; border-radius: 4px; font-size: 0.9rem; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">🔬 Enhanced FEA Laser Simulation Platform with ST-DGPA</h1>', unsafe_allow_html=True)

    # ✅ SAFE SESSION STATE INITIALIZATION
    if 'data_loader' not in st.session_state: st.session_state.data_loader = UnifiedFEADataLoader()
    st.session_state.extrapolator = SpatioTemporalGatedPhysicsAttentionExtrapolator(sigma_param=0.3, spatial_weight=0.5, n_heads=4, temperature=1.0, sigma_g=0.20, s_E=10.0, s_tau=5.0, s_t=20.0, temporal_weight=0.3)
    st.session_state.visualizer = EnhancedVisualizer()
    st.session_state.data_loaded = False; st.session_state.current_mode = "Data Viewer"
    if 'selected_colormap' not in st.session_state: st.session_state.selected_colormap = "Viridis"
    st.session_state.interpolation_results = None; st.session_state.interpolation_params = None
    st.session_state.interpolation_3d_cache = {}; st.session_state.interpolation_field_history = OrderedDict()
    st.session_state.current_3d_field = None; st.session_state.current_3d_timestep = 0; st.session_state.last_prediction_id = None

    with st.sidebar:
        st.markdown("### ⚙️ Navigation")
        app_mode = st.radio("Select Mode", ["Data Viewer", "Interpolation/Extrapolation", "Comparative Analysis", "ST-DGPA Analysis", "Heat Transfer Analysis"], index=["Data Viewer", "Interpolation/Extrapolation", "Comparative Analysis", "ST-DGPA Analysis", "Heat Transfer Analysis"].index(st.session_state.current_mode if 'current_mode' in st.session_state else "Data Viewer"), key="nav_mode")
        st.session_state.current_mode = app_mode
        st.markdown("---")
        st.markdown("### 📊 Data Settings")
        col1, col2 = st.columns(2)
        with col1: load_full_data = st.checkbox("Load Full Mesh", value=True, help="Load complete mesh data for 3D visualization")
        with col2: st.session_state.selected_colormap = st.selectbox("Colormap", EnhancedVisualizer.EXTENDED_COLORMAPS, index=0)
        if st.session_state.current_mode == "Interpolation/Extrapolation" and not load_full_data: st.warning("⚠️ Full mesh loading is required for 3D interpolation visualization. Please enable and reload.")
        if st.button("🔄 Load All Simulations", type="primary", use_container_width=True):
            with st.spinner("Loading simulation data..."):
                CacheManager.clear_3d_cache(); st.session_state.last_prediction_id = None
                simulations, summaries = st.session_state.data_loader.load_all_simulations(load_full_mesh=load_full_data)
                st.session_state.simulations = simulations; st.session_state.summaries = summaries
                if simulations and summaries: st.session_state.extrapolator.load_summaries(summaries); st.session_state.data_loaded = True
                st.session_state.available_fields = set()
                for summary in summaries: st.session_state.available_fields.update(summary['field_stats'].keys())
        if st.session_state.data_loaded and 'interpolation_3d_cache' in st.session_state:
            st.markdown("---"); st.markdown("### 🗄️ Cache Management")
            with st.expander("Cache Statistics", expanded=False):
                cache_size = len(st.session_state.interpolation_3d_cache); history_size = len(st.session_state.interpolation_field_history)
                st.metric("Cached Fields", cache_size); st.metric("Field History", history_size)
                if cache_size > 0:
                    st.write("**Cached Fields:**")
                    for cache_key, cache_data in list(st.session_state.interpolation_3d_cache.items())[:5]: st.caption(f"• {cache_data.get('field_name', 'Unknown')} (t={cache_data.get('timestep_idx', 0)})")
                    if cache_size > 5: st.caption(f"... and {cache_size - 5} more")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Clear Cache", use_container_width=True): CacheManager.clear_3d_cache(); st.rerun()
                    with col2:
                        if st.button("Refresh Stats", use_container_width=True): st.rerun()
        if st.session_state.data_loaded:
            st.markdown("---"); st.markdown("### 📈 Loaded Data")
            with st.expander("Data Overview", expanded=True):
                st.metric("Simulations", len(st.session_state.simulations)); st.metric("Available Fields", len(st.session_state.available_fields))
                if st.session_state.summaries: st.metric("Energy Range", f"{min([s['energy'] for s in st.session_state.summaries]):.1f} - {max([s['energy'] for s in st.session_state.summaries]):.1f} mJ")

    if app_mode == "Data Viewer": render_data_viewer()
    elif app_mode == "Interpolation/Extrapolation": render_interpolation_extrapolation()
    elif app_mode == "Comparative Analysis": render_comparative_analysis()
    elif app_mode == "ST-DGPA Analysis": render_stdgpa_analysis()
    elif app_mode == "Heat Transfer Analysis": render_heat_transfer_analysis()

def render_data_viewer():
    st.markdown('<h2 class="sub-header">📁 Data Viewer</h2>', unsafe_allow_html=True)
    if not st.session_state.data_loaded: st.markdown('<div class="warning-box"><h3>⚠️ No Data Loaded</h3></div>', unsafe_allow_html=True); return
    simulations = st.session_state.simulations
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1: sim_name = st.selectbox("Select Simulation", sorted(simulations.keys()), key="viewer_sim_select")
    sim = simulations[sim_name]
    with col2: st.metric("Energy", f"{sim['energy_mJ']:.2f} mJ")
    with col3: st.metric("Duration", f"{sim['duration_ns']:.2f} ns")
    if not sim.get('has_mesh', False): st.warning("Reload with 'Load Full Mesh' enabled."); return
    col1, col2, col3, col4 = st.columns(4)
    with col1: field = st.selectbox("Select Field", sorted(sim['field_info'].keys()), key="viewer_field_select")
    with col2: timestep = st.slider("Timestep", 0, sim['n_timesteps'] - 1, 0, key="viewer_timestep_slider")
    with col3: colormap = st.selectbox("Colormap", EnhancedVisualizer.EXTENDED_COLORMAPS, index=EnhancedVisualizer.EXTENDED_COLORMAPS.index(st.session_state.get('selected_colormap', 'Viridis')), key="viewer_colormap")
    with col4: opacity = st.slider("Opacity", 0.0, 1.0, 0.9, 0.05, key="viewer_opacity")
    if 'points' in sim and 'fields' in sim and field in sim['fields']:
        pts = sim['points']; kind, _ = sim['field_info'][field]; raw = sim['fields'][field][timestep]
        values = np.where(np.isnan(raw), 0, raw) if kind == "scalar" else np.where(np.isnan(np.linalg.norm(raw, axis=1)), 0, np.linalg.norm(raw, axis=1))
        label = field if kind == "scalar" else f"{field} (magnitude)"
        if sim.get('triangles') is not None and len(sim['triangles']) > 0:
            valid_triangles = [tri for tri in sim['triangles'] if all(idx < len(pts) for idx in tri)]
            if valid_triangles and len(valid_triangles) > 0:
                tri_array = np.array(valid_triangles)
                if tri_array.ndim == 2 and tri_array.shape[1] >= 3:
                    mesh_data = go.Mesh3d(x=pts[:,0], y=pts[:,1], z=pts[:,2], i=tri_array[:,0], j=tri_array[:,1], k=tri_array[:,2], intensity=values, colorscale=st.session_state.get('selected_colormap', 'Viridis'), opacity=opacity, hovertemplate='<b>Value:</b> %{intensity:.3f}<extra></extra>')
                else:
                    mesh_data = go.Scatter3d(x=pts[:,0], y=pts[:,1], z=pts[:,2], mode='markers', marker=dict(size=4, color=values, colorscale=st.session_state.get('selected_colormap', 'Viridis'), opacity=opacity, showscale=True))
            else:
                mesh_data = go.Scatter3d(x=pts[:,0], y=pts[:,1], z=pts[:,2], mode='markers', marker=dict(size=4, color=values, colorscale=st.session_state.get('selected_colormap', 'Viridis'), opacity=opacity, showscale=True))
        else:
            mesh_data = go.Scatter3d(x=pts[:,0], y=pts[:,1], z=pts[:,2], mode='markers', marker=dict(size=4, color=values, colorscale=st.session_state.get('selected_colormap', 'Viridis'), opacity=opacity, showscale=True))
        fig = go.Figure(data=mesh_data)
        fig.update_layout(title=dict(text=f"{label} at Timestep {timestep + 1}<br><sub>{sim_name}</sub>", font=dict(size=20)), scene=dict(aspectmode="data", camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))), height=700)
        st.plotly_chart(fig, use_container_width=True)

def render_interpolation_extrapolation():
    st.markdown('<h2 class="sub-header">🔮 Interpolation/Extrapolation Engine with ST-DGPA</h2>', unsafe_allow_html=True)
    if not st.session_state.data_loaded: st.markdown('<div class="warning-box"><h3>⚠️ No Data Loaded</h3></div>', unsafe_allow_html=True); return
    if st.session_state.simulations and not next(iter(st.session_state.simulations.values())).get('has_mesh', False): st.markdown('<div class="warning-box"><h3>⚠️ Full Mesh Data Required</h3></div>', unsafe_allow_html=True)
    st.markdown('<div class="stdgpa-box"><h3>🧠 Spatio-Temporal Gated Physics Attention (ST-DGPA)</h3></div>', unsafe_allow_html=True)
    with st.expander("📋 Loaded Simulations Summary", expanded=True):
        if st.session_state.summaries: st.dataframe(pd.DataFrame([{'Simulation': s['name'], 'Energy (mJ)': s['energy'], 'Duration (ns)': s['duration'], 'Timesteps': len(s['timesteps'])} for s in st.session_state.summaries]), use_container_width=True, height=300)
    col1, col2, col3, col4 = st.columns(4)
    with col1: energy_query = st.number_input("Energy (mJ)", 0.1, 20.0, 3.5, 0.1, key="interp_energy")
    with col2: duration_query = st.number_input("Duration (ns)", 0.1, 20.0, 4.2, 0.1, key="interp_duration")
    with col3: max_time = st.number_input("Max Prediction Time (ns)", 1, 200, 50, 1, key="interp_maxtime")
    with col4: time_resolution = st.selectbox("Time Resolution", ["1 ns", "2 ns", "5 ns", "10 ns"], index=1, key="interp_resolution")
    time_points = np.arange(1, max_time + 1, {"1 ns": 1, "2 ns": 2, "5 ns": 5, "10 ns": 10}[time_resolution])
    with st.expander("⚙️ ST-DGPA Attention Parameters", expanded=False):
        st.session_state.extrapolator.sigma_g = st.slider("Gating Width (σ_g)", 0.05, 1.0, 0.20, 0.05, key="interp_sigma_g")
        st.session_state.extrapolator.s_E = st.slider("Energy Scale (s_E) [mJ]", 0.1, 50.0, 10.0, 0.5, key="interp_s_E")
        st.session_state.extrapolator.s_tau = st.slider("Duration Scale (s_τ) [ns]", 0.1, 20.0, 5.0, 0.5, key="interp_s_tau")
        st.session_state.extrapolator.s_t = st.slider("Time Scale (s_t) [ns]", 1.0, 50.0, 20.0, 1.0, key="interp_s_t")
        st.session_state.extrapolator.temporal_weight = st.slider("Temporal Similarity Weight", 0.0, 1.0, 0.3, 0.05, key="interp_temporal_weight")
    if st.button("🚀 Run ST-DGPA Prediction", type="primary", use_container_width=True):
        with st.spinner("Running ST-DGPA prediction..."):
            CacheManager.clear_3d_cache()
            results = st.session_state.extrapolator.predict_time_series(energy_query, duration_query, time_points)
            if results and 'field_predictions' in results and results['field_predictions']:
                st.session_state.interpolation_results = results; st.session_state.interpolation_params = {'energy_query': energy_query, 'duration_query': duration_query, 'time_points': time_points, 'sigma_g': st.session_state.extrapolator.sigma_g, 's_E': st.session_state.extrapolator.s_E, 's_tau': st.session_state.extrapolator.s_tau, 's_t': st.session_state.extrapolator.s_t, 'temporal_weight': st.session_state.extrapolator.temporal_weight}
                st.session_state.last_prediction_id = hashlib.md5(f"{energy_query}_{duration_query}".encode()).hexdigest()[:8]
                st.success("✅ Prediction Successful")
    elif st.session_state.interpolation_results is not None:
        st.info("📊 Previous Results Available"); results = st.session_state.interpolation_results; time_points = st.session_state.interpolation_params['time_points']
    if st.session_state.interpolation_results:
        render_prediction_results(st.session_state.interpolation_results, time_points, energy_query, duration_query)
        render_stdgpa_attention_visualization(st.session_state.interpolation_results, energy_query, duration_query, time_points)

def render_stdgpa_attention_visualization(results, energy_query, duration_query, time_points):
    if not results.get('physics_attention_maps') or len(results['physics_attention_maps'][0]) == 0: return
    st.markdown('<h4 class="sub-header">🧠 ST-DGPA Attention Analysis</h4>', unsafe_allow_html=True)
    selected_timestep_idx = st.slider("Select timestep for ST-DGPA analysis", 0, len(time_points) - 1, len(time_points) // 2, key="stdgpa_timestep")
    final_weights = results['attention_maps'][selected_timestep_idx]; physics_attention = results['physics_attention_maps'][selected_timestep_idx]; ett_gating = results['ett_gating_maps'][selected_timestep_idx]; selected_time = time_points[selected_timestep_idx]
    temporal_sim = None
    if st.session_state.extrapolator.temporal_weight > 0 and hasattr(st.session_state.extrapolator, 'source_metadata'):
        query_meta = {'time': selected_time, 'duration': duration_query, 'fourier_number': st.session_state.extrapolator._compute_fourier_number(selected_time)}
        temporal_sim = st.session_state.extrapolator._compute_temporal_similarity(query_meta, st.session_state.extrapolator.source_metadata)
    fig = st.session_state.visualizer.create_stdgpa_analysis(results, energy_query, duration_query, time_points)
    if fig: st.plotly_chart(fig, use_container_width=True)
    sankey_fig = create_stdgpa_sankey(results, energy_query, duration_query, selected_time, st.session_state.extrapolator.source_metadata, final_weights, physics_attention, ett_gating, temporal_sim)
    if sankey_fig.data: st.plotly_chart(sankey_fig, use_container_width=True)

def render_prediction_results(results, time_points, energy_query, duration_query):
    available_fields = list(results['field_predictions'].keys())
    if not available_fields: st.warning("No field predictions available."); return
    n_fields = min(len(available_fields), 4)
    fig = make_subplots(rows=n_fields, cols=1, subplot_titles=[f"Predicted {field}" for field in available_fields[:n_fields]], vertical_spacing=0.1, shared_xaxes=True)
    for idx, field in enumerate(available_fields[:n_fields]):
        row = idx + 1
        if results['field_predictions'][field]['mean']: fig.add_trace(go.Scatter(x=time_points, y=results['field_predictions'][field]['mean'], mode='lines+markers', name=f'{field} (mean)', line=dict(width=3, color='blue')), row=row, col=1)
    fig.update_layout(height=300 * n_fields, title_text=f"Field Predictions (E={energy_query:.1f} mJ, τ={duration_query:.1f} ns)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

def render_comparative_analysis(): pass
def render_stdgpa_analysis(): pass
def render_heat_transfer_analysis(): pass

if __name__ == "__main__": main()
