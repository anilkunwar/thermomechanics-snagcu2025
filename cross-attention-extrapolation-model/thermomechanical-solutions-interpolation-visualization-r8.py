#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ENHANCED LASER SOLDERING ST-DGPA PLATFORM WITH ROBUST POLAR RADAR
==================================================================
Complete integrated application for:
- FEA laser soldering simulation loading from VTU files
- ST-DGPA (Spatio-Temporal Gated Physics Attention) interpolation/extrapolation
- ENHANCED Polar Radar Charts with robust data handling, autoscaling, and jitter for overlapping points
- 3D mesh visualization with caching
- Export functionality
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
import meshio
import warnings
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
import base64
import hashlib          # FIXED: added missing import
import json             # FIXED: added missing import
import time             # FIXED: added missing import
from collections import OrderedDict  # FIXED: added missing import
from io import BytesIO   # optional but safe
import traceback         # optional
import tempfile          # optional
from scipy.interpolate import griddata, RBFInterpolator  # optional
from sklearn.preprocessing import StandardScaler

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
    @staticmethod
    def generate_cache_key(field_name: str, timestep_idx: int, energy: float, 
                          duration: float, time: float, sigma_param: float,
                          spatial_weight: float, n_heads: int, temperature: float,
                          sigma_g: float, s_E: float, s_tau: float, s_t: float,
                          temporal_weight: float, top_k: Optional[int] = None,
                          subsample_factor: Optional[int] = None) -> str:
        params_str = f"{field_name}_{timestep_idx}_{energy:.2f}_{duration:.2f}_{time:.2f}"
        params_str += f"_{sigma_param:.2f}_{spatial_weight:.2f}_{n_heads}_{temperature:.2f}"
        params_str += f"_{sigma_g:.2f}_{s_E:.2f}_{s_tau:.2f}_{s_t:.2f}_{temporal_weight:.2f}"
        if top_k: params_str += f"_top{top_k}"
        if subsample_factor: params_str += f"_sub{subsample_factor}"
        return hashlib.md5(params_str.encode()).hexdigest()[:16]
    
    @staticmethod
    def clear_3d_cache():
        if 'interpolation_3d_cache' in st.session_state:
            st.session_state.interpolation_3d_cache = {}
        if 'interpolation_field_history' in st.session_state:
            st.session_state.interpolation_field_history = OrderedDict()
        st.success("✅ Cache cleared")
    
    @staticmethod
    def get_cached_interpolation(field_name: str, timestep_idx: int, params: Dict) -> Optional[Dict]:
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
        if 'interpolation_field_history' not in st.session_state:
            st.session_state.interpolation_field_history = OrderedDict()
        history_key = f"{field_name}_{timestep_idx}"
        st.session_state.interpolation_field_history[history_key] = cache_key
        if len(st.session_state.interpolation_3d_cache) > 20:
            oldest_key = min(st.session_state.interpolation_3d_cache.keys(),
                             key=lambda k: st.session_state.interpolation_3d_cache[k]['timestamp'])
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
        self.load_errors = []
    
    def parse_folder_name(self, folder: str) -> Tuple[Optional[float], Optional[float]]:
        match = re.match(r"q([\dp\.]+)mJ-delta([\dp\.]+)ns", folder)
        if not match:
            return None, None
        e, d = match.groups()
        return float(e.replace("p", ".")), float(d.replace("p", "."))
    
    @st.cache_data(show_spinner="Loading simulation data...")
    def load_all_simulations(_self, load_full_mesh: bool = True) -> Tuple[Dict, List]:
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
                            field_type = "vector" if arr.shape[1] <= 3 else "tensor"
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
                        except Exception as e:
                            st.warning(f"Error loading timestep {t} in {name}: {e}")
                    sim_data.update({
                        'points': points,
                        'fields': fields,
                        'triangles': triangles,
                        'has_mesh': True
                    })
                
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
                        summary['field_stats'][field_name]['q25'].append(float(np.percentile(clean_data, 25)))
                        summary['field_stats'][field_name]['q50'].append(float(np.percentile(clean_data, 50)))
                        summary['field_stats'][field_name]['q75'].append(float(np.percentile(clean_data, 75)))
                    else:
                        for key in ['min', 'max', 'mean', 'std', 'q25', 'q50', 'q75']:
                            summary['field_stats'][field_name][key].append(0.0)
            except Exception as e:
                st.warning(f"Error processing {vtu_file}: {e}")
                continue
        return summary

# =============================================
# 3. ST-DGPA EXTRAPOLATOR
# =============================================
class SpatioTemporalGatedPhysicsAttentionExtrapolator:
    def __init__(self, sigma_param: float = 0.3, spatial_weight: float = 0.5,
                 n_heads: int = 4, temperature: float = 1.0,
                 sigma_g: float = 0.20, s_E: float = 10.0, s_tau: float = 5.0,
                 s_t: float = 20.0, temporal_weight: float = 0.3):
        self.sigma_param = sigma_param
        self.spatial_weight = spatial_weight
        self.n_heads = n_heads
        self.temperature = temperature
        self.sigma_g = sigma_g
        self.s_E = s_E
        self.s_tau = s_tau
        self.s_t = s_t
        self.temporal_weight = temporal_weight
        self.thermal_diffusivity = 1e-5
        self.laser_spot_radius = 50e-6
        self.characteristic_length = 100e-6
        self.source_db = []
        self.embedding_scaler = StandardScaler()
        self.value_scaler = StandardScaler()
        self.source_embeddings = []
        self.source_values = []
        self.source_metadata = []
        self.fitted = False
    
    def load_summaries(self, summaries: List[Dict]):
        self.source_db = summaries
        if not summaries:
            return
        all_embeddings, all_values, metadata = [], [], []
        for summary_idx, summary in enumerate(summaries):
            for timestep_idx, t in enumerate(summary['timesteps']):
                emb = self._compute_enhanced_physics_embedding(
                    summary['energy'], summary['duration'], t
                )
                all_embeddings.append(emb)
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
            self.embedding_scaler.fit(all_embeddings)
            self.source_embeddings = self.embedding_scaler.transform(all_embeddings)
            self.value_scaler.fit(all_values)
            self.source_values = all_values
            self.source_metadata = metadata
            self.fitted = True
            st.info(f"✅ Prepared {len(all_embeddings)} embeddings with {all_embeddings.shape[1]} features")
    
    def _compute_fourier_number(self, time_ns: float) -> float:
        time_s = time_ns * 1e-9
        return self.thermal_diffusivity * time_s / (self.characteristic_length ** 2)
    
    def _compute_thermal_penetration(self, time_ns: float) -> float:
        time_s = time_ns * 1e-9
        return np.sqrt(self.thermal_diffusivity * time_s) * 1e6
    
    def _compute_enhanced_physics_embedding(self, energy: float, duration: float, time: float) -> np.ndarray:
        logE = np.log1p(energy)
        power = energy / max(duration, 1e-6)
        time_ratio = time / max(duration, 1e-3)
        fourier_number = self._compute_fourier_number(time)
        thermal_penetration = self._compute_thermal_penetration(time)
        heating_phase = 1.0 if time < duration else 0.0
        cooling_phase = 1.0 if time >= duration else 0.0
        return np.array([
            logE, duration, time, power, time_ratio,
            fourier_number, thermal_penetration,
            heating_phase, cooling_phase,
            np.log1p(power), np.log1p(time), np.sqrt(time)
        ], dtype=np.float32)
    
    def _compute_ett_gating(self, energy_query: float, duration_query: float,
                           time_query: float, source_metadata: Optional[List] = None) -> np.ndarray:
        if source_metadata is None:
            source_metadata = self.source_metadata
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
    
    def _compute_temporal_similarity(self, query_meta: Dict, source_metas: List[Dict]) -> np.ndarray:
        similarities = []
        for meta in source_metas:
            time_diff = abs(query_meta['time'] - meta['time'])
            if query_meta['time'] < query_meta['duration'] * 1.5:
                tolerance = max(query_meta['duration'] * 0.1, 1.0)
            else:
                tolerance = max(query_meta['duration'] * 0.3, 3.0)
            fourier_similarity = np.exp(
                -abs(query_meta.get('fourier_number', 0) - meta.get('fourier_number', 0)) / 0.1
            )
            time_similarity = np.exp(-time_diff / tolerance)
            similarities.append(
                (1 - self.temporal_weight) * time_similarity +
                self.temporal_weight * fourier_similarity
            )
        return np.array(similarities)
    
    def _compute_spatial_similarity(self, query_meta: Dict, source_metas: List[Dict]) -> np.ndarray:
        similarities = []
        for meta in source_metas:
            e_diff = abs(query_meta['energy'] - meta['energy']) / 50.0
            d_diff = abs(query_meta['duration'] - meta['duration']) / 20.0
            total_diff = np.sqrt(e_diff**2 + d_diff**2)
            similarities.append(np.exp(-total_diff / self.sigma_param))
        return np.array(similarities)
    
    def _multi_head_attention_with_gating(self, query_embedding: np.ndarray,
                                         query_meta: Dict) -> Tuple:
        if not self.fitted or len(self.source_embeddings) == 0:
            return None, None, None, None
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
        if self.temperature != 1.0:
            avg_weights = avg_weights ** (1.0 / self.temperature)
        max_weight = np.max(avg_weights)
        exp_weights = np.exp(avg_weights - max_weight)
        physics_attention = exp_weights / (np.sum(exp_weights) + 1e-12)
        ett_gating = self._compute_ett_gating(
            query_meta['energy'], query_meta['duration'], query_meta['time']
        )
        combined_weights = physics_attention * ett_gating
        combined_sum = np.sum(combined_weights)
        final_weights = combined_weights / combined_sum if combined_sum > 1e-12 else physics_attention
        prediction = np.sum(final_weights[:, np.newaxis] * self.source_values, axis=0) \
            if len(self.source_values) > 0 else np.zeros(1)
        return prediction, final_weights, physics_attention, ett_gating
    
    def predict_field_statistics(self, energy_query: float, duration_query: float,
                                time_query: float) -> Optional[Dict]:
        if not self.fitted:
            return None
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
        prediction, final_weights, physics_attention, ett_gating = \
            self._multi_head_attention_with_gating(query_embedding, query_meta)
        if prediction is None:
            return None
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
    
    def _compute_temporal_confidence(self, time_query: float, duration_query: float) -> float:
        if time_query < duration_query * 0.5:
            return 0.6
        elif time_query < duration_query * 1.5:
            return 0.8
        else:
            return 0.9
    
    def _compute_heat_transfer_indicators(self, energy: float, duration: float,
                                         time: float) -> Dict:
        fourier_number = self._compute_fourier_number(time)
        thermal_penetration = self._compute_thermal_penetration(time)
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

# =============================================
# 4. ENHANCED POLAR RADAR VISUALIZER (FIXED)
# =============================================
class PolarRadarVisualizer:
    def __init__(self):
        self.source_symbol = 'circle'
        self.target_symbol = 'star-diamond'
    
    @staticmethod
    def safe_normalize(values: np.ndarray, reference_max: Optional[float] = None) -> np.ndarray:
        """Normalize values to [0,1] with robust edge-case handling."""
        clean = np.asarray(values, dtype=float)
        clean = np.nan_to_num(clean, nan=0.0, posinf=0.0, neginf=0.0)
        
        if reference_max is not None and reference_max > 0:
            return np.clip(clean / reference_max, 0, 1)
        
        c_min, c_max = np.min(clean), np.max(clean)
        if c_max - c_min < 1e-9:  # All values identical
            return np.ones_like(clean) * 0.5
        return (clean - c_min) / (c_max - c_min)
    
    def create_enhanced_polar_radar_chart(self, df: pd.DataFrame, field_type: str,
                                         query_params: Optional[Dict] = None,
                                         timestep: int = 1,
                                         show_legend: bool = True,
                                         width: int = 800,
                                         height: int = 700,
                                         # Title styling
                                         title_font_size: int = 18,
                                         title_font_family: str = "Arial, sans-serif",
                                         title_padding_top: int = 40,
                                         # Label styling
                                         label_font_size: int = 12,
                                         label_font_family: str = "Arial, sans-serif",
                                         tick_font_size: int = 10,
                                         # Radar styling
                                         radar_line_width: float = 2.5,
                                         radar_line_opacity: float = 0.8,
                                         radar_fill_opacity: float = 0.3,
                                         radial_axis_range: Optional[Tuple[float, float]] = None,
                                         angular_axis_rotation: int = 90,
                                         angular_axis_direction: str = "clockwise",
                                         # Target query styling
                                         target_query_color: str = "#FF0000",
                                         target_query_line_width: float = 4.0,
                                         target_query_marker_size: int = 10,
                                         # Grid & background
                                         grid_color: str = "lightgray",
                                         grid_line_width: float = 1.0,
                                         background_color: str = "white",
                                         show_radial_grid: bool = True,
                                         show_angular_grid: bool = True,
                                         # Layout margins
                                         margin_left: int = 60,
                                         margin_right: int = 60,
                                         margin_top: int = 80,
                                         margin_bottom: int = 60,
                                         legend_position: str = "top-right",
                                         # Color mapping for target query
                                         use_colorbar_scale: bool = True,
                                         colorbar_title: str = "Peak Value",
                                         colorbar_tick_font_size: int = 9,
                                         # Target highlighting
                                         highlight_target: bool = True,
                                         target_label: str = "Target Query",
                                         # Data normalization
                                         normalize_by_max: bool = True,
                                         max_reference_value: Optional[float] = None,
                                         # Energy axis control (autoscaling)
                                         custom_energy_max: Optional[float] = None,
                                         custom_energy_min: Optional[float] = None,
                                         # Target peak value for color consistency
                                         target_peak_value: Optional[float] = None
                                         ) -> go.Figure:
        """
        Create enhanced polar radar chart with robust data handling and autoscaling.
        """
        # Early validation
        if df.empty:
            # Use st.error only if this function is called from within a Streamlit context
            try:
                st.error("❌ No source simulation data available for radar chart")
            except:
                pass
            fig = go.Figure()
            fig.update_layout(title="No Data", width=width, height=height)
            return fig
        
        # Coerce to numeric and handle NaN/Inf
        energies = pd.to_numeric(df['Energy'], errors='coerce').values
        durations = pd.to_numeric(df['Duration'], errors='coerce').values
        peak_values = pd.to_numeric(df['Peak_Value'], errors='coerce').values
        sim_names = df.get('Name', [f"Sim {i}" for i in range(len(df))]).tolist()
        
        # Filter out invalid rows (but allow zero values)
        valid_mask = (
            np.isfinite(energies) & 
            np.isfinite(durations) & 
            np.isfinite(peak_values) &
            (durations >= 0)  # Duration can't be negative
        )
        
        if not np.any(valid_mask):
            try:
                st.error("❌ All data rows contain invalid values (NaN/Inf/negative durations)")
            except:
                pass
            fig = go.Figure()
            fig.update_layout(title="Invalid Data", width=width, height=height)
            return fig
        
        energies = energies[valid_mask]
        durations = durations[valid_mask]
        peak_values = peak_values[valid_mask]
        sim_names = np.array(sim_names)[valid_mask].tolist()
        
        # --- IMPROVED ENERGY TO ANGLE MAPPING (AUTOSCALING) ---
        e_min_data = np.nanmin(energies)
        e_max_data = np.nanmax(energies)
        
        # Decide max: use custom only if explicitly set > 110% of data max
        if custom_energy_max is not None and custom_energy_max > e_max_data * 1.1:
            e_max = custom_energy_max
        else:
            e_max = e_max_data * 1.05   # small padding
        
        # Min: use custom only if set and less than data min
        if custom_energy_min is not None and custom_energy_min < e_min_data:
            e_min = custom_energy_min
        else:
            e_min = e_min_data
        
        e_range = e_max - e_min
        if e_range < 1e-6:
            e_range = 1.0
            e_min = e_min - 0.5  # Center the single value
        
        # Compute angles
        angles_rad = 2 * np.pi * (energies - e_min) / e_range
        angles_deg = np.degrees(angles_rad)
        
        # Add jitter to separate overlapping points (identical energies)
        unique_angles = np.unique(np.round(angles_deg, 1))
        if len(unique_angles) < len(angles_deg):
            np.random.seed(42)  # Reproducible jitter
            jitter = np.random.uniform(-2, 2, size=len(angles_deg))
            angles_deg = angles_deg + jitter
            try:
                st.info(f"ℹ️ Added ±2° jitter to separate {len(angles_deg) - len(unique_angles)} overlapping points")
            except:
                pass
        
        # Generate nice tick values
        tick_energies = np.linspace(e_min, e_max, 6)
        tick_angles = [2 * np.pi * (e - e_min) / e_range for e in tick_energies]
        tick_angles_deg = np.degrees(tick_angles)
        
        # --- SAFE NORMALIZATION (FIXED: separate normalization from radial range) ---
        # Normalize peak values for marker colors
        if normalize_by_max and max_reference_value is not None and max_reference_value > 0:
            norm_peak = self.safe_normalize(peak_values, max_reference_value)
        else:
            norm_peak = self.safe_normalize(peak_values, None)
        
        # Radial axis range (for duration)
        if radial_axis_range is not None:
            r_range = radial_axis_range
        else:
            max_dur = np.max(durations)
            r_range = (0, max_dur * 1.1 if max_dur > 0 else 1.2)
        
        # Determine colorscale
        is_temp = 'temp' in field_type.lower()
        c_scale = 'Inferno' if is_temp else 'Plasma'
        title_field = "Peak Temperature (K)" if is_temp else "Peak von Mises Stress (MPa)"
        
        # Create figure
        fig = go.Figure()
        
        # --- SOURCE SIMULATIONS SCATTER (with robust marker size) ---
        marker_sizes = 10 + 20 * norm_peak  # guaranteed in [10, 30]
        
        fig.add_trace(go.Scatterpolar(
            r=durations,
            theta=angles_deg,
            mode='markers',
            marker=dict(
                size=marker_sizes,
                color=norm_peak,
                colorscale=c_scale,
                colorbar=dict(title=title_field, thickness=20, x=1.02),
                line=dict(width=2, color='white'),
                symbol=self.source_symbol
            ),
            text=[f"<b>{name}</b><br>Energy: {e:.1f} mJ<br>Duration: {d:.1f} ns<br>Peak: {p:.3f}"
                  for name, e, d, p in zip(sim_names, energies, durations, peak_values)],
            hoverinfo='text',
            name='Source Simulations',
            opacity=0.9
        ))
        
        # --- TARGET QUERY POINT (if provided) ---
        if query_params and highlight_target:
            q_e = query_params.get('Energy')
            q_d = query_params.get('Duration')
            if q_e is not None and q_d is not None and np.isfinite(q_e) and np.isfinite(q_d):
                q_angle_rad = 2 * np.pi * (q_e - e_min) / e_range
                q_angle_deg = np.degrees(q_angle_rad)
                
                # Determine target marker color using same colorscale
                if use_colorbar_scale and target_peak_value is not None:
                    if normalize_by_max and max_reference_value is not None:
                        norm_target = target_peak_value / max_reference_value
                    else:
                        norm_target = target_peak_value / (np.max(peak_values) + 1e-9)
                    norm_target = np.clip(norm_target, 0, 1)
                    target_color = px.colors.sample_colorscale(c_scale, [norm_target])[0]
                else:
                    target_color = target_query_color
                
                fig.add_trace(go.Scatterpolar(
                    r=[q_d],
                    theta=[q_angle_deg],
                    mode='markers+lines',
                    marker=dict(
                        size=target_query_marker_size,
                        color=target_color,
                        symbol=self.target_symbol,
                        line=dict(width=2, color='white')
                    ),
                    line=dict(color=target_color, width=target_query_line_width, dash='dot'),
                    name=target_label,
                    hovertemplate=f"<b>Target Query</b><br>Energy: {q_e:.2f} mJ<br>Duration: {q_d:.2f} ns<br>Peak: {target_peak_value:.3f}<extra></extra>"
                ))
        
        # --- POLAR LAYOUT ---
        polar_layout = dict(
            radialaxis=dict(
                visible=True,
                title="Pulse Duration (ns)",
                range=r_range,
                gridcolor=grid_color if show_radial_grid else 'rgba(0,0,0,0)',
                gridwidth=grid_line_width,
                tickfont=dict(size=tick_font_size, family=label_font_family),
                title_font=dict(size=label_font_size, family=label_font_family)
            ),
            angularaxis=dict(
                visible=True,
                direction=angular_axis_direction,
                rotation=angular_axis_rotation,
                gridcolor=grid_color if show_angular_grid else 'rgba(0,0,0,0)',
                gridwidth=grid_line_width,
                tickfont=dict(size=tick_font_size, family=label_font_family),
                tickmode='array',
                tickvals=tick_angles_deg,
                ticktext=[f"{e:.2f}" for e in tick_energies],
                thetaunit="degrees",
                showline=True,
                linecolor="black",
                linewidth=1
            ),
            bgcolor=background_color,
            gridshape='circular'
        )
        
        # --- FINAL LAYOUT ---
        fig.update_layout(
            polar=polar_layout,
            title=dict(
                text=f"Polar Radar: {title_field} at t={timestep} ns<br>"
                     f"<span style='font-size:{title_font_size-4}px;'>Energy range: {e_min:.2f} – {e_max:.2f} mJ • Radial: Pulse Duration (ns)</span>",
                font=dict(size=title_font_size, family=title_font_family),
                x=0.5,
                xanchor='center',
                y=0.98,
                yanchor='top',
                pad=dict(t=title_padding_top, b=10)
            ),
            width=width,
            height=height,
            showlegend=show_legend,
            margin=dict(l=margin_left, r=margin_right, t=margin_top, b=margin_bottom),
            legend=dict(
                yanchor="top" if "top" in legend_position else "bottom",
                y=0.99 if "top" in legend_position else 0.01,
                xanchor="right" if "right" in legend_position else "left",
                x=0.99 if "right" in legend_position else 0.01,
                font=dict(size=label_font_size-1, family=label_font_family),
                bgcolor='rgba(255,255,255,0.95)',
                bordercolor=grid_color,
                borderwidth=1
            ),
            hovermode='closest'
        )
        return fig

# =============================================
# 5. ENHANCED VISUALIZER
# =============================================
class EnhancedVisualizer:
    @staticmethod
    def create_stdgpa_analysis(results: Dict, energy_query: float, duration_query: float,
                            time_points: np.ndarray) -> Optional[go.Figure]:
        if not results or 'attention_maps' not in results or len(results['attention_maps']) == 0:
            return None
        timestep_idx = len(time_points) // 2
        time = time_points[timestep_idx]
        fig = make_subplots(rows=3, cols=3, subplot_titles=["ST-DGPA Final Weights", "Physics Attention Only", "(E, τ, t) Gating Only", "ST-DGPA vs Physics Attention", "Temporal Coherence Analysis", "Heat Transfer Phase", "Parameter Space 3D", "Attention Network", "Weight Evolution"])
        final_weights = results['attention_maps'][timestep_idx]
        physics_attention = results['physics_attention_maps'][timestep_idx]
        ett_gating = results['ett_gating_maps'][timestep_idx]
        fig.add_trace(go.Bar(x=list(range(len(final_weights))), y=final_weights, marker_color='blue', showlegend=False), row=1, col=1)
        fig.add_trace(go.Bar(x=list(range(len(physics_attention))), y=physics_attention, marker_color='green', showlegend=False), row=1, col=2)
        fig.add_trace(go.Bar(x=list(range(len(ett_gating))), y=ett_gating, marker_color='red', showlegend=False), row=1, col=3)
        fig.add_trace(go.Scatter(x=list(range(len(final_weights))), y=final_weights, mode='lines+markers', line=dict(color='blue', width=3)), row=2, col=1)
        fig.add_trace(go.Scatter(x=list(range(len(physics_attention))), y=physics_attention, mode='lines+markers', line=dict(color='green', width=2, dash='dash')), row=2, col=1)
        if st.session_state.get('summaries') and hasattr(st.session_state.extrapolator, 'source_metadata'):
            times, weights = [], []
            for i, weight in enumerate(final_weights):
                if weight > 0.01 and i < len(st.session_state.extrapolator.source_metadata):
                    times.append(st.session_state.extrapolator.source_metadata[i]['time'])
                    weights.append(weight)
            if times and weights:
                fig.add_trace(go.Scatter(x=times, y=weights, mode='markers', marker=dict(size=np.array(weights)*50, color=weights, colorscale='Viridis', showscale=False)), row=2, col=2)
                fig.add_vline(x=time, line_dash="dash", line_color="red", row=2, col=2)
        fig.update_layout(height=1000, title_text=f"ST-DGPA Analysis at t={time} ns (E={energy_query:.1f} mJ, τ={duration_query:.1f} ns)", showlegend=True)
        return fig

    @staticmethod
    def create_confidence_plot(results: Dict, time_points: np.ndarray) -> go.Figure:
        fig = go.Figure()
        if 'confidence_scores' in results and results['confidence_scores']:
            fig.add_trace(go.Scatter(x=time_points, y=results['confidence_scores'], mode='lines+markers', name='Prediction Confidence', line=dict(color='orange', width=3), fill='tozeroy', fillcolor='rgba(255,165,0,0.2)'))
        if 'temporal_confidences' in results and results['temporal_confidences']:
            fig.add_trace(go.Scatter(x=time_points, y=results['temporal_confidences'], mode='lines+markers', name='Temporal Confidence', line=dict(color='green', width=3, dash='dash')))
        fig.update_layout(title="Prediction Confidence Over Time", xaxis_title="Time (ns)", yaxis_title="Confidence", height=400, yaxis_range=[0,1], hovermode='x unified')
        return fig

# =============================================
# 6. EXPORT UTILITIES
# =============================================
class ExportManager:
    @staticmethod
    def export_to_json(data: Dict, filename: Optional[str] = None) -> Tuple[str, str]:
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"stdgpa_export_{timestamp}.json"
        def convert_numpy(obj):
            if isinstance(obj, np.integer): return int(obj)
            elif isinstance(obj, np.floating): return float(obj)
            elif isinstance(obj, np.ndarray): return obj.tolist()
            elif isinstance(obj, dict): return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list): return [convert_numpy(v) for v in obj]
            return obj
        json_str = json.dumps(convert_numpy(data), indent=2)
        return json_str, filename

    @staticmethod
    def export_to_csv(df: pd.DataFrame, filename: Optional[str] = None) -> Tuple[str, str]:
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"stdgpa_export_{timestamp}.csv"
        csv_str = df.to_csv(index=False)
        return csv_str, filename

    @staticmethod
    def export_plotly_figure(fig: go.Figure, format: str = 'png', filename: Optional[str] = None) -> Optional[bytes]:
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
# 7. MAIN APPLICATION
# =============================================
def main():
    st.set_page_config(page_title="Laser Soldering ST-DGPA Platform", layout="wide", initial_sidebar_state="expanded", page_icon="🔬")
    st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; text-align: center; margin-bottom: 1.5rem; font-weight: 800; background: linear-gradient(90deg, #1E88E5, #4A00E0); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .sub-header { font-size: 1.5rem; color: #2c3e50; margin-top: 1.5rem; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 2px solid #3498db; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">🔬 Laser Soldering ST-DGPA Analysis Platform</h1>', unsafe_allow_html=True)

    # Initialize session state
    if 'data_loader' not in st.session_state: st.session_state.data_loader = UnifiedFEADataLoader()
    if 'extrapolator' not in st.session_state: st.session_state.extrapolator = SpatioTemporalGatedPhysicsAttentionExtrapolator()
    if 'polar_viz' not in st.session_state: st.session_state.polar_viz = PolarRadarVisualizer()
    if 'enhanced_viz' not in st.session_state: st.session_state.enhanced_viz = EnhancedVisualizer()
    if 'export_manager' not in st.session_state: st.session_state.export_manager = ExportManager()
    if 'interpolation_results' not in st.session_state: st.session_state.interpolation_results = None
    if 'interpolation_params' not in st.session_state: st.session_state.interpolation_params = None
    if 'polar_query' not in st.session_state: st.session_state.polar_query = None
    if 'data_loaded' not in st.session_state: st.session_state.data_loaded = False

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        load_full = st.checkbox("Load Full Mesh", value=True, help="Load complete mesh data for 3D visualization")
        if st.button("🔄 Load All Simulations", type="primary", use_container_width=True):
            with st.spinner("Loading simulation data..."):
                sims, summaries = st.session_state.data_loader.load_all_simulations(load_full_mesh=load_full)
                st.session_state.simulations = sims
                st.session_state.summaries = summaries
                if sims and summaries:
                    st.session_state.extrapolator.load_summaries(summaries)
                    st.session_state.data_loaded = True
                    st.session_state.available_fields = set()
                    for s in summaries:
                        st.session_state.available_fields.update(s['field_stats'].keys())
        if st.session_state.get('data_loaded') and st.session_state.get('summaries'):
            st.success(f"✅ {len(st.session_state.summaries)} simulations loaded")
            st.markdown("---")
            st.header("🎯 ST-DGPA Parameters")
            col1, col2 = st.columns(2)
            with col1:
                sigma_g = st.slider("σ_g (Gating)", 0.05, 1.0, 0.20, 0.05)
                s_E = st.slider("s_E (Energy)", 0.1, 50.0, 10.0, 0.5)
            with col2:
                s_tau = st.slider("s_τ (Duration)", 0.1, 20.0, 5.0, 0.5)
                s_t = st.slider("s_t (Time)", 1.0, 50.0, 20.0, 1.0)
            temporal_weight = st.slider("Temporal Weight", 0.0, 1.0, 0.3, 0.05)
            st.session_state.extrapolator.sigma_g = sigma_g
            st.session_state.extrapolator.s_E = s_E
            st.session_state.extrapolator.s_tau = s_tau
            st.session_state.extrapolator.s_t = s_t
            st.session_state.extrapolator.temporal_weight = temporal_weight
            st.markdown("---")
            st.header("🗄️ Cache Management")
            if st.button("Clear Cache", use_container_width=True):
                CacheManager.clear_3d_cache()
            if 'interpolation_3d_cache' in st.session_state:
                cache_size = len(st.session_state.interpolation_3d_cache)
                st.caption(f"Cache size: {cache_size}/20 entries")

    # Main Tabs
    tabs = st.tabs(["📊 Data Overview", "🔮 Interpolation", "🎯 Polar Radar", "🧠 ST-DGPA Analysis", "💾 Export"])
    if not st.session_state.get('data_loaded') or not st.session_state.get('summaries'):
        st.info("👈 Please load simulations using the sidebar to begin analysis.")
        with st.expander("📁 Expected Directory Structure"):
            st.code("""
            fea_solutions/
            ├── q0p5mJ-delta4p2ns/    # Energy: 0.5 mJ, Duration: 4.2 ns
            │   ├── a_t0001.vtu       # Timestep 1
            │   ├── a_t0002.vtu       # Timestep 2
            │   └── ...
            └── q1p0mJ-delta2p0ns/
            """)
        return

    # --- TAB 1: Data Overview ---
    with tabs[0]:
        st.subheader("Loaded Simulations Summary")
        df_summary = pd.DataFrame([{'Name': s['name'], 'Energy (mJ)': s['energy'], 'Duration (ns)': s['duration'], 'Timesteps': len(s['timesteps']), 'Fields': ', '.join(sorted(s['field_stats'].keys())[:3]) + ('...' if len(s['field_stats']) > 3 else '')} for s in st.session_state.summaries])
        st.dataframe(df_summary.style.format({'Energy (mJ)': '{:.2f}', 'Duration (ns)': '{:.2f}'}).background_gradient(subset=['Energy (mJ)', 'Duration (ns)'], cmap='Blues'), use_container_width=True, height=300)
        if st.session_state.simulations and next(iter(st.session_state.simulations.values())).get('has_mesh'):
            st.markdown("### 🎨 3D Field Viewer")
            col1, col2, col3 = st.columns(3)
            with col1: sim_name = st.selectbox("Simulation", sorted(st.session_state.simulations.keys()), key="viewer_sim")
            with col2: sim = st.session_state.simulations[sim_name]; field = st.selectbox("Field", sorted(sim['field_info'].keys()), key="viewer_field")
            with col3: timestep = st.slider("Timestep", 0, sim['n_timesteps']-1, 0, key="viewer_timestep")
            if sim['points'] is not None and field in sim['fields']:
                values = sim['fields'][field][timestep].copy()
                if values.ndim >= 2:
                    values = np.linalg.norm(values, axis=tuple(range(1, values.ndim)))
                values = np.nan_to_num(values, nan=0.0)
                colormap = 'Inferno' if 'temp' in field.lower() else ('Plasma' if 'stress' in field.lower() else 'Viridis')
                if sim['triangles'] is not None and len(sim['triangles']) > 0:
                    fig = go.Figure(go.Mesh3d(x=sim['points'][:,0], y=sim['points'][:,1], z=sim['points'][:,2], i=sim['triangles'][:,0], j=sim['triangles'][:,1], k=sim['triangles'][:,2], intensity=values, colorscale=colormap, intensitymode='vertex', colorbar=dict(title=field)))
                else:
                    fig = go.Figure(go.Scatter3d(x=sim['points'][:,0], y=sim['points'][:,1], z=sim['points'][:,2], mode='markers', marker=dict(size=3, color=values, colorscale=colormap, colorbar=dict(title=field), showscale=True)))
                fig.update_layout(scene=dict(aspectmode="data"), title=f"{field} at Timestep {timestep+1}", height=600)
                st.plotly_chart(fig, use_container_width=True)

    # --- TAB 2: Interpolation ---
    with tabs[1]:
        st.subheader("Run ST-DGPA Interpolation")
        c1, c2, c3 = st.columns(3)
        energies = [s['energy'] for s in st.session_state.summaries]
        min_e, max_e = min(energies), max(energies)
        durations = [s['duration'] for s in st.session_state.summaries]
        min_d, max_d = min(durations), max(durations)
        with c1: q_E = st.number_input("Energy (mJ)", float(min_e*0.5), float(max_e*2.0), float((min_e+max_e)/2), 0.1, key="interp_energy")
        with c2: q_τ = st.number_input("Duration (ns)", float(min_d*0.5), float(max_d*2.0), float((min_d+max_d)/2), 0.1, key="interp_duration")
        with c3: max_t = st.number_input("Max Time (ns)", 1, 200, 50, 1, key="interp_maxtime")
        time_resolution = st.selectbox("Time Resolution", ["1 ns", "2 ns", "5 ns", "10 ns"], index=1)
        time_step = {"1 ns": 1, "2 ns": 2, "5 ns": 5, "10 ns": 10}[time_resolution]
        time_points = np.arange(1, max_t + 1, time_step)
        if st.button("🚀 Run ST-DGPA Prediction", type="primary", use_container_width=True):
            with st.spinner("Computing Spatio-Temporal Gated Physics Attention..."):
                start_time = time.time()
                results = st.session_state.extrapolator.predict_time_series(q_E, q_τ, time_points)
                elapsed = time.time() - start_time
                if results and results['field_predictions']:
                    st.session_state.interpolation_results = results
                    st.session_state.interpolation_params = {'energy_query': q_E, 'duration_query': q_τ, 'time_points': time_points}
                    st.session_state.polar_query = {'Energy': q_E, 'Duration': q_τ}
                    st.success(f"✅ Prediction Complete ({elapsed:.2f}s)")
                else:
                    st.error("❌ Prediction failed. Check parameters and ensure sufficient training data.")
        if st.session_state.interpolation_results:
            results = st.session_state.interpolation_results
            params = st.session_state.interpolation_params
            st.subheader("Prediction Results")
            fields = list(results['field_predictions'].keys())
            if fields:
                selected_fields = st.multiselect("Select Fields to Plot", fields, default=fields[:3])
                if selected_fields:
                    fig_preds = go.Figure()
                    for f in selected_fields:
                        if results['field_predictions'][f]['mean']:
                            fig_preds.add_trace(go.Scatter(x=params['time_points'], y=results['field_predictions'][f]['mean'], mode='lines', name=f, line=dict(width=2)))
                            if results['field_predictions'][f]['std']:
                                mean = np.array(results['field_predictions'][f]['mean'])
                                std = np.array(results['field_predictions'][f]['std'])
                                fig_preds.add_trace(go.Scatter(x=np.concatenate([params['time_points'], params['time_points'][::-1]]), y=np.concatenate([mean+std, (mean-std)[::-1]]), fill='toself', fillcolor=f'rgba(0,100,255,0.1)', line=dict(color='rgba(255,255,255,0)'), showlegend=False, name=f'{f} ± std'))
                    fig_preds.update_layout(title="Predicted Mean Values with Confidence Bands", xaxis_title="Time (ns)", yaxis_title="Value", height=400, hovermode='x unified')
                    st.plotly_chart(fig_preds, use_container_width=True)
                if results['confidence_scores']:
                    fig_conf = st.session_state.enhanced_viz.create_confidence_plot(results, params['time_points'])
                    st.plotly_chart(fig_conf, use_container_width=True)

    # --- TAB 3: Polar Radar (with robust data handling) ---
    with tabs[2]:
        st.subheader("🎯 Polar Radar Visualization")
        all_fields = set()
        for s in st.session_state.summaries: all_fields.update(s['field_stats'].keys())
        col1, col2 = st.columns(2)
        with col1: field_type = st.selectbox("Field Type", sorted(all_fields), key="polar_field")
        with col2:
            # Determine the maximum available timestep across all simulations
            max_possible_t = max((len(s.get('timesteps', [])) for s in st.session_state.summaries), default=1)
            t_step = st.number_input("Timestep Index", 1, max_possible_t, 1, key="polar_timestep")

        # --- ROBUST DATA COLLECTION (permissive extraction) ---
        rows = []
        warnings_found = []

        for s in st.session_state.summaries:
            field_stats = s['field_stats'].get(field_type, {})
            peak_list = field_stats.get('max', [0])   # default [0] if field missing

            idx = t_step - 1
            if idx >= len(peak_list):
                warnings_found.append(f"Timestep {t_step} not available in {s['name']} (only {len(peak_list)} steps)")
                continue

            peak_val = peak_list[idx]
            if not np.isfinite(peak_val):
                peak_val = 0.0

            rows.append({
                'Name': s['name'],
                'Energy': s['energy'],
                'Duration': s['duration'],
                'Peak_Value': float(peak_val)
            })

        df_polar = pd.DataFrame(rows)

        if df_polar.empty:
            st.error(f"❌ No valid data for **{field_type}** at timestep **{t_step}**")
            if warnings_found:
                with st.expander("🔍 Why is data missing?"):
                    for w in warnings_found[:10]:
                        st.caption(w)
        else:
            st.success(f"✅ Loaded {len(df_polar)} simulations for radar")
            st.caption(f"DataFrame shape: {df_polar.shape} | Energy range: {df_polar['Energy'].min():.2f} – {df_polar['Energy'].max():.2f} mJ")

            e_max_data = df_polar['Energy'].max()
            
            # Styling controls
            with st.expander("🎨 Radar Chart Styling Options", expanded=True):
                colA, colB, colC, colD = st.columns(4)
                with colA:
                    st.markdown("**Title Styling**")
                    title_font_size = st.slider("Title Font Size", 12, 30, 18, key="radar_title_font")
                    title_font_family = st.selectbox("Title Font Family", ["Arial, sans-serif", "Courier New, monospace", "Times New Roman, serif"], index=0, key="radar_title_family")
                    title_padding = st.slider("Title Padding (top)", 20, 100, 40, key="radar_title_pad")
                with colB:
                    st.markdown("**Label Styling**")
                    label_font_size = st.slider("Label Font Size", 8, 20, 12, key="radar_label_font")
                    label_font_family = st.selectbox("Label Font Family", ["Arial, sans-serif", "Courier New, monospace", "Times New Roman, serif"], index=0, key="radar_label_family")
                    tick_font_size = st.slider("Tick Font Size", 6, 16, 10, key="radar_tick_font")
                with colC:
                    st.markdown("**Radar Line Styling**")
                    radar_line_width = st.slider("Radar Line Width", 1.0, 5.0, 2.5, 0.5, key="radar_line_width")
                    radar_line_opacity = st.slider("Radar Line Opacity", 0.1, 1.0, 0.8, 0.1, key="radar_line_opacity")
                    radar_fill_opacity = st.slider("Radar Fill Opacity", 0.0, 0.8, 0.3, 0.1, key="radar_fill_opacity")
                with colD:
                    st.markdown("**Target Query Styling**")
                    target_query_color = st.color_picker("Target Query Color", "#FF0000", key="radar_target_color")
                    target_query_line_width = st.slider("Target Line Width", 2.0, 6.0, 4.0, 0.5, key="radar_target_width")
                    target_query_marker_size = st.slider("Target Marker Size", 5, 20, 10, key="radar_target_marker")
            
            with st.expander("📐 Grid & Layout Options", expanded=False):
                colE, colF, colG, colH = st.columns(4)
                with colE:
                    st.markdown("**Grid Styling**")
                    grid_color = st.color_picker("Grid Color", "#d3d3d3", key="radar_grid_color")
                    grid_line_width = st.slider("Grid Line Width", 0.5, 3.0, 1.0, 0.5, key="radar_grid_width")
                    show_radial_grid = st.checkbox("Show Radial Grid", value=True, key="radar_show_radial")
                    show_angular_grid = st.checkbox("Show Angular Grid", value=True, key="radar_show_angular")
                with colF:
                    st.markdown("**Axis Configuration**")
                    radial_axis_min = st.number_input("Radial Axis Min", value=0.0, key="radar_radial_min")
                    radial_axis_max = st.number_input("Radial Axis Max", value=1.2, key="radar_radial_max")
                    angular_rotation = st.slider("Angular Axis Rotation", 0, 360, 90, key="radar_angular_rot")
                    angular_direction = st.selectbox("Angular Direction", ["clockwise", "counterclockwise"], index=0, key="radar_angular_dir")
                with colG:
                    st.markdown("**Background & Margins**")
                    background_color = st.color_picker("Background Color", "#ffffff", key="radar_bg_color")
                    margin_left = st.slider("Margin Left", 40, 120, 60, key="radar_margin_l")
                    margin_right = st.slider("Margin Right", 40, 120, 60, key="radar_margin_r")
                    margin_top = st.slider("Margin Top", 60, 140, 80, key="radar_margin_t")
                    margin_bottom = st.slider("Margin Bottom", 40, 120, 60, key="radar_margin_b")
                with colH:
                    st.markdown("**Legend & Color**")
                    legend_position = st.selectbox("Legend Position", ["top-right", "top-left", "bottom-right", "bottom-left", "center"], index=0, key="radar_legend_pos")
                    use_colorbar_scale = st.checkbox("Use Colorbar Scale for Target", value=True, key="radar_use_colorbar")
                    colorbar_title = st.text_input("Colorbar Title", "Peak Value", key="radar_cbar_title")
                    colorbar_tick_font = st.slider("Colorbar Tick Font", 6, 14, 9, key="radar_cbar_font")
            
            with st.expander("📊 Data Processing Options", expanded=False):
                colI, colJ = st.columns(2)
                with colI:
                    normalize_by_max = st.checkbox("Normalize by Maximum Value", value=True, key="radar_normalize")
                    max_reference_value = st.number_input("Max Reference Value", min_value=0.0, value=float(e_max_data * 1.2), key="radar_max_ref", help="Reference maximum for normalization (auto-set to data max +20%)")
                with colJ:
                    highlight_target = st.checkbox("Highlight Target Query", value=True, key="radar_highlight_target")
                    target_label = st.text_input("Target Label", "Target Query", key="radar_target_label")
                    custom_energy_max = st.number_input("Custom Max Energy (mJ)", min_value=0.0, value=float(e_max_data * 1.2), key="radar_custom_energy_max", help="Override angular axis maximum (0 = auto)")
                    custom_energy_min = st.number_input("Custom Min Energy (mJ)", min_value=0.0, value=0.0, key="radar_custom_energy_min")
            
            # Get target query params and predicted peak value
            query_params = None
            target_peak_value = None
            if st.session_state.get('polar_query') and st.session_state.get('interpolation_results'):
                query_params = st.session_state.polar_query
                results = st.session_state.interpolation_results
                if field_type in results['field_predictions']:
                    t_idx = t_step - 1
                    if t_idx < len(results['field_predictions'][field_type]['max']):
                        target_peak_value = results['field_predictions'][field_type]['max'][t_idx]
            
            # Create enhanced polar radar chart
            fig_polar = st.session_state.polar_viz.create_enhanced_polar_radar_chart(
                df_polar, field_type, query_params, timestep=t_step,
                show_legend=True, width=900, height=750,
                title_font_size=title_font_size, title_font_family=title_font_family, title_padding_top=title_padding,
                label_font_size=label_font_size, label_font_family=label_font_family, tick_font_size=tick_font_size,
                radar_line_width=radar_line_width, radar_line_opacity=radar_line_opacity, radar_fill_opacity=radar_fill_opacity,
                radial_axis_range=(radial_axis_min, radial_axis_max) if radial_axis_max > radial_axis_min else None,
                angular_axis_rotation=angular_rotation, angular_axis_direction=angular_direction,
                target_query_color=target_query_color, target_query_line_width=target_query_line_width, target_query_marker_size=target_query_marker_size,
                grid_color=grid_color, grid_line_width=grid_line_width, background_color=background_color,
                show_radial_grid=show_radial_grid, show_angular_grid=show_angular_grid,
                margin_left=margin_left, margin_right=margin_right, margin_top=margin_top, margin_bottom=margin_bottom,
                legend_position=legend_position,
                use_colorbar_scale=use_colorbar_scale, colorbar_title=colorbar_title, colorbar_tick_font_size=colorbar_tick_font,
                highlight_target=highlight_target, target_label=target_label,
                normalize_by_max=normalize_by_max, max_reference_value=max_reference_value if normalize_by_max else None,
                custom_energy_max=custom_energy_max if custom_energy_max > 0 else None,
                custom_energy_min=custom_energy_min if custom_energy_min > 0 else None,
                target_peak_value=target_peak_value
            )
            st.plotly_chart(fig_polar, use_container_width=True)

            # Multi-timestep comparison
            if st.checkbox("Show Multiple Timesteps", key="polar_multi"):
                st.markdown("##### Multi-Timestep Comparison")
                t_indices = st.multiselect("Select Timesteps", [1,2,3,4,5], default=[1,3,5])
                if t_indices:
                    cols = st.columns(len(t_indices))
                    for col, t_idx in zip(cols, t_indices):
                        if t_idx <= max_possible_t:
                            rows_t = []
                            for s in st.session_state.summaries:
                                if t_idx <= len(s['timesteps']):
                                    peak_t = s['field_stats'].get(field_type, {}).get('max', [0])
                                    if t_idx-1 < len(peak_t):
                                        rows_t.append({'Name': s['name'], 'Energy': s['energy'], 'Duration': s['duration'], 'Peak_Value': peak_t[t_idx-1]})
                            target_val = None
                            if st.session_state.get('interpolation_results') and field_type in st.session_state.interpolation_results['field_predictions']:
                                if t_idx-1 < len(st.session_state.interpolation_results['field_predictions'][field_type]['max']):
                                    target_val = st.session_state.interpolation_results['field_predictions'][field_type]['max'][t_idx-1]
                            fig_t = st.session_state.polar_viz.create_enhanced_polar_radar_chart(
                                pd.DataFrame(rows_t), field_type, query_params, timestep=t_idx,
                                width=350, height=350, show_legend=(t_idx == t_indices[0]),
                                title_font_size=12, label_font_size=9, tick_font_size=7,
                                radar_line_width=1.5, radar_fill_opacity=0.2,
                                margin_left=40, margin_right=40, margin_top=50, margin_bottom=40,
                                target_peak_value=target_val, use_colorbar_scale=False, highlight_target=highlight_target,
                                normalize_by_max=normalize_by_max, max_reference_value=max_reference_value if normalize_by_max else None,
                                custom_energy_max=custom_energy_max if custom_energy_max > 0 else None,
                                custom_energy_min=custom_energy_min if custom_energy_min > 0 else None
                            )
                            with col:
                                st.plotly_chart(fig_t, use_container_width=True)
                                st.caption(f"Timestep {t_idx}")

    # --- TAB 4: ST-DGPA Analysis ---
    with tabs[3]:
        st.subheader("🧠 ST-DGPA Attention & Physics Analysis")
        if st.session_state.interpolation_results and st.session_state.get('interpolation_params'):
            res = st.session_state.interpolation_results
            params = st.session_state.interpolation_params
            fig_stdgpa = st.session_state.enhanced_viz.create_stdgpa_analysis(res, params['energy_query'], params['duration_query'], params['time_points'])
            if fig_stdgpa:
                st.plotly_chart(fig_stdgpa, use_container_width=True)
                with st.expander("💾 Export Analysis"):
                    if st.button("Export as PNG"):
                        img_bytes = st.session_state.export_manager.export_plotly_figure(fig_stdgpa, format='png')
                        if img_bytes:
                            st.download_button(label="⬇️ Download PNG", data=img_bytes, file_name=f"stdgpa_analysis_{params['energy_query']:.1f}mJ.png", mime="image/png")
            else:
                st.info("ℹ️ No attention data available for analysis.")
        else:
            st.info("ℹ️ Please run an interpolation first (Tab 2) to see ST-DGPA analysis.")

    # --- TAB 5: Export ---
    with tabs[4]:
        st.subheader("💾 Export Results")
        if st.session_state.interpolation_results and st.session_state.get('interpolation_params'):
            results = st.session_state.interpolation_results
            params = st.session_state.interpolation_params
            st.markdown("### Export Prediction Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                export_data = {'metadata': {'generated_at': datetime.now().isoformat(), 'query_params': params, 'stdgpa_params': {'sigma_g': st.session_state.extrapolator.sigma_g, 's_E': st.session_state.extrapolator.s_E, 's_tau': st.session_state.extrapolator.s_tau, 's_t': st.session_state.extrapolator.s_t, 'temporal_weight': st.session_state.extrapolator.temporal_weight}}, 'results': results}
                json_str, json_name = st.session_state.export_manager.export_to_json(export_data)
                st.download_button(label="📥 Download as JSON", data=json_str, file_name=json_name, mime="application/json", use_container_width=True)
            with col2:
                if results['field_predictions']:
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
                        st.download_button(label="📥 Download as CSV", data=csv_str, file_name=csv_name, mime="text/csv", use_container_width=True)
            with col3:
                if results['attention_maps']:
                    attention_df = pd.DataFrame(results['attention_maps'])
                    att_csv, att_name = st.session_state.export_manager.export_to_csv(attention_df, filename="attention_weights.csv")
                    st.download_button(label="📥 Download Attention Weights", data=att_csv, file_name=att_name, mime="text/csv", use_container_width=True)
        else:
            st.info("ℹ️ Please run an interpolation first (Tab 2) to enable export.")

if __name__ == "__main__":
    main()
