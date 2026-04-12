#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ENHANCED FEA LASER SIMULATION PLATFORM WITH ST-DGPA
====================================================
Complete integrated application for:
- FEA laser soldering simulation loading from VTU files
- ST-DGPA (Spatio-Temporal Gated Physics Attention) interpolation/extrapolation
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
import matplotlib.pyplot as plt
import meshio
import warnings
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from io import BytesIO
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
# UNIFIED DATA LOADER WITH ENHANCED CAPABILITIES
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
                        clean_data = np.linalg.norm(data, axis=1)[~np.isnan(np.linalg.norm(data, axis=1))]
                    if clean_data.size > 0:
                        for key in ['min', 'max', 'mean', 'std']:
                            summary['field_stats'][field_name][key].append(float(eval(f"np.{key}(clean_data)")))
                    else:
                        for key in ['min', 'max', 'mean', 'std']: summary['field_stats'][field_name][key].append(0.0)
            except: continue
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
                metadata.append({
                    'summary_idx': summary_idx, 'timestep_idx': timestep_idx,
                    'energy': summary['energy'], 'duration': summary['duration'], 'time': t,
                    'name': summary['name'],
                    'fourier_number': self._compute_fourier_number(t),
                    'thermal_penetration': self._compute_thermal_penetration(t)
                })
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
            if self.temporal_weight > 0: dtime *= (1.0 + 0.5 * (time_query / max(duration_query, 1e-6)))
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

# =============================================
# ADVANCED VISUALIZATION COMPONENTS WITH ST-DGPA ANALYSIS
# =============================================
class EnhancedVisualizer:
    @staticmethod
    def create_stdgpa_sankey(sources_data, target_params, param_sigmas=None, customization=None):
        """
        Create Sankey diagram showing ST-DGPA weight decomposition flow.
        ✅ FIXED: Robust dictionary access to prevent KeyError.
        """
        cfg = customization or {}
        font_family = cfg.get('font_family', 'Arial, sans-serif')
        font_size = cfg.get('font_size', 12)
        node_thickness = cfg.get('node_thickness', 20)
        node_pad = cfg.get('node_pad', 15)
        
        labels = ['Target']
        node_colors = ['#FF6B6B']
        
        # Safely extract data with .get()
        safe_sources = []
        for i, src in enumerate(sources_data):
            idx = src.get('source_index', i)
            e = src.get('Energy', src.get('energy', src.get('energy_mJ', 0.0)))
            d = src.get('Duration', src.get('duration', src.get('duration_ns', 0.0)))
            t = src.get('Time', src.get('time', 0.0))
            attn = src.get('Attention_Score', 0.0)
            gating = src.get('Gating', 0.0)
            refinement = src.get('Refinement', 0.0)
            combined = src.get('Combined_Weight', 0.0)
            safe_sources.append({'idx': idx, 'E': e, 'D': d, 'T': t, 'attn': attn, 'gating': gating, 'ref': refinement, 'comb': combined})
            labels.append(f"Source {idx+1}\nE={e:.1f}mJ\nτ={d:.1f}ns")
            base = cfg.get('node_colors', {}).get('source', '#9966FF')
            r, g, b = int(base[1:3], 16), int(base[3:5], 16), int(base[5:7], 16)
            opacity = min(0.3 + combined * 0.7, 1.0)
            node_colors.append(f'rgba({r},{g},{b},{opacity:.2f})')
            
        component_start = len(labels)
        comp_labels = ['Energy Gate', 'Duration Gate', 'Time Gate', 'Attention', 'Refinement', 'Combined']
        labels.extend(comp_labels)
        comp_colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFD93D', '#36A2EB', '#9966FF']
        node_colors.extend(comp_colors)
        
        s_idx, t_idx, vals, l_colors, h_texts = [], [], [], [], []
        
        for i, src in enumerate(safe_sources):
            s_node = i + 1
            ve = ((src['E'] - target_params.get('Energy', 0)) / 10.0)**2 * 10
            vd = ((src['D'] - target_params.get('Duration', 0)) / 5.0)**2 * 10
            vt = ((src['T'] - target_params.get('Time', 0)) / 20.0)**2 * 10
            
            vals_list = [ve, vd, vt, src['attn']*100, src['ref']*100, src['comb']*100]
            for c in range(6):
                s_idx.append(s_node); t_idx.append(component_start + c)
                vals.append(max(0.01, vals_list[c]))
                l_colors.append(comp_colors[c].replace('rgb', 'rgba').replace(')', ', 0.5)'))
                h_texts.append(f"<b>{comp_labels[c]}</b><br>Value: {vals_list[c]:.3f}")
                
        for c in range(6):
            s_idx.append(component_start + c); t_idx.append(0)
            flow_in = sum(v for s, t, v in zip(s_idx[:-6], t_idx[:-6], vals[:-6]) if t == component_start + c)
            vals.append(flow_in * 0.5); l_colors.append('rgba(153,102,255,0.6)')
            h_texts.append(f"<b>Aggregation</b><br>{comp_labels[c]} → TARGET<br>Total: {flow_in:.3f}")
            
        fig = go.Figure(go.Sankey(
            node=dict(pad=node_pad, thickness=node_thickness, line=dict(color="black", width=0.5),
                     label=labels, color=node_colors, font=dict(family=font_family, size=font_size),
                     hovertemplate='<b>%{label}</b><br>Value: %{value:.3f}<extra></extra>'),
            link=dict(source=s_idx, target=t_idx, value=vals, color=l_colors,
                     customdata=h_texts, hovertemplate='%{customdata}<extra></extra>',
                     line=dict(width=0.5, color='rgba(255,255,255,0.3)'))
        ))
        
        title_text = (f"<b>ST-DGPA Attention Flow</b><br>"
                     f"Query: E={target_params.get('Energy',0):.2f} mJ, τ={target_params.get('Duration',0):.2f} ns, t={target_params.get('Time',0):.2f} ns")
        fig.update_layout(
            title=dict(text=title_text, font=dict(family=font_family, size=font_size+4), x=0.5, xanchor='center'),
            font=dict(family=font_family, size=font_size),
            width=cfg.get('width', 1000), height=cfg.get('height', 700),
            plot_bgcolor='rgba(240, 240, 245, 0.9)', paper_bgcolor='white',
            margin=dict(t=100, l=50, r=50, b=50),
            hoverlabel=dict(font=dict(family=font_family, size=font_size), bgcolor='rgba(44, 62, 80, 0.9)', bordercolor='white', namelength=-1)
        )
        return fig
    
    @staticmethod
    def create_temporal_analysis(results, time_points):
        if not results or 'heat_transfer_indicators' not in results: return None
        fig = make_subplots(rows=2, cols=2, subplot_titles=["Heat Transfer Phase Evolution", "Fourier Number Evolution", "Temporal Confidence", "Thermal Penetration Depth"], vertical_spacing=0.15, horizontal_spacing=0.15)
        phases = [ind.get('phase', 'Unknown') for ind in results['heat_transfer_indicators'] if ind]
        phase_mapping = {'Early Heating': 0, 'Heating': 1, 'Early Cooling': 2, 'Diffusion Cooling': 3}
        phase_values = [phase_mapping.get(p, 0) for p in phases]
        fig.add_trace(go.Scatter(x=time_points[:len(phase_values)], y=phase_values, mode='lines+markers', line=dict(color='red', width=3)), row=1, col=1)
        for pname, pval in phase_mapping.items(): fig.add_hline(y=pval, line_dash="dot", line_color="gray", annotation_text=pname, row=1, col=1)
        fourier_numbers = [ind.get('fourier_number', 0) for ind in results['heat_transfer_indicators'] if ind]
        if fourier_numbers: fig.add_trace(go.Scatter(x=time_points[:len(fourier_numbers)], y=fourier_numbers, mode='lines+markers', line=dict(color='blue', width=3)), row=1, col=2)
        if 'temporal_confidences' in results: fig.add_trace(go.Scatter(x=time_points[:len(results['temporal_confidences'])], y=results['temporal_confidences'], mode='lines+markers', line=dict(color='green', width=3), fill='tozeroy', fillcolor='rgba(0, 255, 0, 0.2)'), row=2, col=1)
        penetration_depths = [ind.get('thermal_penetration_um', 0) for ind in results['heat_transfer_indicators'] if ind]
        if penetration_depths: fig.add_trace(go.Scatter(x=time_points[:len(penetration_depths)], y=penetration_depths, mode='lines+markers', line=dict(color='orange', width=3)), row=2, col=2)
        fig.update_layout(height=700, title_text="Temporal Analysis of Heat Transfer Characteristics", showlegend=False)
        return fig

# =============================================
# MAIN INTEGRATED APPLICATION
# =============================================
def main():
    st.set_page_config(page_title="Enhanced FEA Laser Simulation Platform with ST-DGPA", layout="wide", initial_sidebar_state="expanded", page_icon="🔬")
    st.markdown("<style>.main-header { font-size: 2.5rem; text-align: center; margin-bottom: 1.5rem; font-weight: 800; color: #1a202c; }</style>", unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">🔬 Enhanced FEA Laser Simulation Platform with ST-DGPA</h1>', unsafe_allow_html=True)

    if 'data_loader' not in st.session_state: st.session_state.data_loader = UnifiedFEADataLoader()
    st.session_state.extrapolator = SpatioTemporalGatedPhysicsAttentionExtrapolator(sigma_param=0.3, spatial_weight=0.5, n_heads=4, temperature=1.0, sigma_g=0.20, s_E=10.0, s_tau=5.0, s_t=20.0, temporal_weight=0.3)
    st.session_state.visualizer = EnhancedVisualizer()
    st.session_state.data_loaded = False
    st.session_state.interpolation_results = None
    st.session_state.interpolation_params = None

    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        if st.button("🔄 Load All Simulations", type="primary", use_container_width=True):
            with st.spinner("Loading..."):
                sims, summaries = st.session_state.data_loader.load_all_simulations(load_full_mesh=True)
                st.session_state.simulations = sims; st.session_state.summaries = summaries
                if sims and summaries: st.session_state.extrapolator.load_summaries(summaries); st.session_state.data_loaded = True
        if st.session_state.data_loaded: st.success(f"✅ {len(st.session_state.summaries)} simulations loaded")
        
        st.markdown("---")
        st.markdown("### 🎯 ST-DGPA Parameters")
        sigma_g = st.slider("Gating Width (σ_g)", 0.05, 1.0, 0.20, 0.05)
        s_E = st.slider("Energy Scale (s_E) [mJ]", 0.1, 50.0, 10.0, 0.5)
        s_tau = st.slider("Duration Scale (s_τ) [ns]", 0.1, 20.0, 5.0, 0.5)
        s_t = st.slider("Time Scale (s_t) [ns]", 1.0, 50.0, 20.0, 1.0)
        st.session_state.extrapolator.sigma_g = sigma_g
        st.session_state.extrapolator.s_E = s_E
        st.session_state.extrapolator.s_tau = s_tau
        st.session_state.extrapolator.s_t = s_t

    tabs = st.tabs(["🔮 Interpolation/Extrapolation", "🕸️ Sankey Diagram", "🧠 ST-DGPA Analysis"])
    
    with tabs[0]:
        st.markdown('<h3 class="sub-header">🔮 Run ST-DGPA Prediction</h3>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: energy_query = st.number_input("Energy (mJ)", 0.1, 20.0, 3.5, 0.1)
        with c2: duration_query = st.number_input("Duration (ns)", 0.1, 20.0, 4.2, 0.1)
        with c3: max_time = st.number_input("Max Prediction Time (ns)", 1, 200, 50, 1)
        time_points = np.arange(1, max_time + 1, 1)
        
        if st.button("🚀 Run Prediction", type="primary", use_container_width=True):
            with st.spinner("Computing..."):
                results = st.session_state.extrapolator.predict_time_series(energy_query, duration_query, time_points)
                if results and results['field_predictions']:
                    st.session_state.interpolation_results = results
                    st.session_state.interpolation_params = {'energy_query': energy_query, 'duration_query': duration_query, 'time_points': time_points}
                    st.success("✅ Prediction Complete")
                else: st.error("Prediction failed.")

    with tabs[1]:
        st.markdown('<h3 class="sub-header">🕸️ ST-DGPA Weight Flow (Sankey)</h3>', unsafe_allow_html=True)
        if st.session_state.interpolation_results and st.session_state.interpolation_params:
            res = st.session_state.interpolation_results
            params = st.session_state.interpolation_params
            q_E = params['energy_query']; q_τ = params['duration_query']
            t_sel = st.slider("Select Timestep", 1, len(res['attention_maps']), 1)
            t_idx = t_sel - 1
            
            # ✅ FIXED: Explicitly construct sources_data with all required keys
            sources_data = []
            weights = res['attention_maps'][t_idx]
            phys_att = res['physics_attention_maps'][t_idx]
            gating = res['ett_gating_maps'][t_idx]
            
            for i, meta in enumerate(st.session_state.extrapolator.source_metadata):
                if meta['timestep_idx'] == t_idx:
                    sources_data.append({
                        'source_index': i,
                        'Energy': meta['energy'],
                        'Duration': meta['duration'],
                        'Time': meta['time'],
                        'Attention_Score': phys_att[i],
                        'Gating': gating[i],
                        'Refinement': phys_att[i] * gating[i],
                        'Combined_Weight': weights[i]
                    })
            
            if sources_data:
                query = {'Energy': q_E, 'Duration': q_τ, 'Time': t_sel}
                customization = {'font_size': 11, 'node_thickness': 15}
                fig_sankey = st.session_state.visualizer.create_stdgpa_sankey(sources_data, query, customization)
                st.plotly_chart(fig_sankey, use_container_width=True)
            else:
                st.warning("No source data matches the selected timestep.")
        else:
            st.info("Please run an interpolation first.")

    with tabs[2]:
        st.markdown('<h3 class="sub-header">🧠 ST-DGPA Analysis</h3>', unsafe_allow_html=True)
        if st.session_state.interpolation_results and st.session_state.interpolation_params:
            res = st.session_state.interpolation_results
            params = st.session_state.interpolation_params
            fig = st.session_state.visualizer.create_temporal_analysis(res, params['time_points'])
            if fig: st.plotly_chart(fig, use_container_width=True)
            else: st.info("No temporal data available.")
        else:
            st.info("Please run an interpolation first.")

if __name__ == "__main__":
    main()
