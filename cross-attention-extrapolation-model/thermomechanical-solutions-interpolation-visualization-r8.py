#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ENHANCED LASER SOLDERING ST-DGPA PLATFORM WITH REPAIRED POLAR RADAR
==================================================================
Complete integrated application for:
- FEA laser soldering simulation loading from VTU files
- ST-DGPA (Spatio-Temporal Gated Physics Attention) interpolation/extrapolation
- Polar Radar Charts with robust data handling, autoscaling, jitter, and full customization
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
import hashlib
import json
import time
from collections import OrderedDict
from io import BytesIO
import traceback
import tempfile
from scipy.interpolate import griddata, RBFInterpolator
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
# 4. POLAR RADAR VISUALIZER (REPAIRED & EXPANDED)
# =============================================
class PolarRadarVisualizer:
    def __init__(self):
        self.color_scale_temp = 'Inferno'
        self.color_scale_stress = 'Plasma'
        self.target_symbol = 'star-diamond'
        self.source_symbol = 'circle'

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

    def create_polar_radar_chart(self, df: pd.DataFrame, field_type: str = 'temperature',
                                query_params: Optional[Dict] = None,
                                timestep: int = 1,
                                width: int = 800, height: int = 700,
                                show_legend: bool = True,
                                marker_size_range: Tuple[int, int] = (10, 30),
                                # ADDED customization parameters (original)
                                radial_grid_width: float = 1.0,
                                angular_grid_width: float = 1.0,
                                radial_tick_font_size: int = 12,
                                angular_tick_font_size: int = 12,
                                title_font_size: int = 18,
                                margin_pad: Dict[str, int] = None,
                                energy_min: float = None,
                                energy_max: float = None,
                                target_peak_value: float = None,
                                # NEW expansion parameters
                                add_jitter: bool = True,
                                jitter_amount: float = 2.0,
                                normalize_colors: bool = True,
                                color_reference_max: Optional[float] = None,
                                radial_axis_max: Optional[float] = None,
                                highlight_target: bool = True) -> go.Figure:
        """
        Create enhanced polar radar chart with robust data handling and autoscaling.
        """
        if margin_pad is None:
            margin_pad = {'l': 60, 'r': 60, 't': 80, 'b': 60}

        if df.empty:
            fig = go.Figure()
            fig.update_layout(title="No data available", width=width, height=height)
            return fig

        # Extract and validate data with explicit numeric coercion
        energies = pd.to_numeric(df['Energy'], errors='coerce').values
        durations = pd.to_numeric(df['Duration'], errors='coerce').values
        peak_values = pd.to_numeric(df['Peak_Value'], errors='coerce').values
        sim_names = df.get('Name', [f"Sim {i}" for i in range(len(df))]).tolist()

        # Filter out invalid rows (NaN, Inf, negative durations)
        valid_mask = (
            np.isfinite(energies) &
            np.isfinite(durations) &
            np.isfinite(peak_values) &
            (durations >= 0)
        )

        if not np.any(valid_mask):
            fig = go.Figure()
            fig.update_layout(title="No valid numeric data", width=width, height=height)
            return fig

        energies = energies[valid_mask]
        durations = durations[valid_mask]
        peak_values = peak_values[valid_mask]
        sim_names = np.array(sim_names)[valid_mask].tolist()

        # --- IMPROVED ENERGY TO ANGLE MAPPING (AUTOSCALING) ---
        e_min_data = np.nanmin(energies)
        e_max_data = np.nanmax(energies)

        # Use custom values if provided, otherwise auto-scale
        if energy_min is not None:
            e_min = energy_min
        else:
            e_min = e_min_data

        if energy_max is not None and energy_max > e_max_data * 1.1:
            e_max = energy_max
        else:
            e_max = e_max_data * 1.05  # Add 5% padding

        e_range = e_max - e_min
        if e_range < 1e-6:  # Handle single-value case
            e_range = 1.0
            e_min = e_min - 0.5

        # Compute angles in degrees
        angles_rad = 2 * np.pi * (energies - e_min) / e_range
        angles_deg = np.degrees(angles_rad)

        # Add jitter to separate overlapping points (identical energies)
        if add_jitter and len(angles_deg) > 1:
            unique_angles = np.unique(np.round(angles_deg, 1))
            if len(unique_angles) < len(angles_deg):
                np.random.seed(42)  # Reproducible jitter
                jitter = np.random.uniform(-jitter_amount, jitter_amount, size=len(angles_deg))
                angles_deg = angles_deg + jitter

        # Generate nice tick values for angular axis
        tick_energies = np.linspace(e_min, e_max, 6)
        tick_angles_deg = np.degrees([2 * np.pi * (e - e_min) / e_range for e in tick_energies])

        # --- SAFE NORMALIZATION ---
        if normalize_colors and color_reference_max is not None:
            norm_peak = self.safe_normalize(peak_values, color_reference_max)
        else:
            norm_peak = self.safe_normalize(peak_values, None)

        # Radial axis range (for duration)
        if radial_axis_max is not None:
            r_range = (0, radial_axis_max)
        else:
            max_dur = np.max(durations) if len(durations) > 0 else 1
            r_range = (0, max_dur * 1.1 if max_dur > 0 else 1.2)

        # Determine colorscale based on field type
        is_temp = 'temp' in field_type.lower()
        c_scale = self.color_scale_temp if is_temp else self.color_scale_stress
        title_field = "Peak Temperature (K)" if is_temp else "Peak von Mises Stress (MPa)"

        # Create Figure
        fig = go.Figure()

        # --- SOURCE SIMULATIONS SCATTER ---
        marker_sizes = marker_size_range[0] + (marker_size_range[1] - marker_size_range[0]) * norm_peak

        fig.add_trace(go.Scatterpolar(
            r=durations,
            theta=angles_deg,
            mode='markers',
            marker=dict(
                size=marker_sizes,
                color=norm_peak if normalize_colors else peak_values,
                colorscale=c_scale,
                colorbar=dict(title=title_field, thickness=20, x=1.02),
                line=dict(width=2, color='white'),
                symbol=self.source_symbol,
                showscale=True
            ),
            text=[f"<b>{name}</b><br>Timestep: {timestep}<br>Energy: {e:.2f} mJ<br>Duration: {d:.2f} ns<br>Peak: {p:.3f}"
                  for name, e, d, p in zip(sim_names, energies, durations, peak_values)],
            hoverinfo='text',
            name='Source Simulations',
            opacity=0.85
        ))

        # --- TARGET QUERY POINT (REPAIRED COLOR HANDLING) ---
        if query_params and highlight_target:
            q_e = query_params.get('Energy')
            q_d = query_params.get('Duration')
            if q_e is not None and q_d is not None and np.isfinite(q_e) and np.isfinite(q_d):
                q_angle_deg = np.degrees(2 * np.pi * (q_e - e_min) / e_range)

                # REPAIR: Properly determine target color from colorscale using target_peak_value
                if target_peak_value is not None and np.isfinite(target_peak_value):
                    if normalize_colors and color_reference_max is not None:
                        norm_target = target_peak_value / color_reference_max
                    else:
                        norm_target = target_peak_value / (np.max(peak_values) + 1e-9)
                    norm_target = np.clip(norm_target, 0, 1)
                    # Sample from same colorscale for consistency
                    target_color = px.colors.sample_colorscale(c_scale, [norm_target])[0]
                else:
                    # Fallback to distinct color if no peak value available
                    target_color = '#FF0000'  # Red

                fig.add_trace(go.Scatterpolar(
                    r=[q_d],
                    theta=[q_angle_deg],
                    mode='markers',
                    marker=dict(
                        size=25,
                        color=target_color,
                        symbol=self.target_symbol,
                        line=dict(width=3, color='black')
                    ),
                    name='Target Query',
                    hovertemplate=f"<b>Target Query</b><br>Energy: {q_e:.2f} mJ<br>Duration: {q_d:.2f} ns<br>Peak: {target_peak_value:.3f if target_peak_value else 'N/A'}<extra></extra>"
                ))

        # --- POLAR LAYOUT WITH FULL CUSTOMIZATION ---
        polar_layout = dict(
            radialaxis=dict(
                visible=True,
                title="Pulse Duration (ns)",
                range=r_range,
                gridcolor="lightgray",
                gridwidth=radial_grid_width,
                tickfont=dict(size=radial_tick_font_size),
                title_font=dict(size=radial_tick_font_size + 2)
            ),
            angularaxis=dict(
                visible=True,
                direction="clockwise",
                rotation=90,
                gridcolor="lightgray",
                gridwidth=angular_grid_width,
                tickfont=dict(size=angular_tick_font_size),
                tickmode='array',
                tickvals=tick_angles_deg,
                ticktext=[f"{e:.1f}" for e in tick_energies],
                thetaunit="degrees"
            ),
            bgcolor="white"
        )

        fig.update_layout(
            title=dict(
                text=f"Polar Radar: {title_field} at t={timestep} ns<br>"
                     f"<span style='font-size:{title_font_size-4}px; color:gray;'>"
                     f"Energy range: {e_min:.2f} – {e_max:.2f} mJ • Radial: Pulse Duration (ns)</span>",
                font=dict(size=title_font_size),
                x=0.5,
                xanchor='center'
            ),
            polar=polar_layout,
            width=width,
            height=height,
            showlegend=show_legend,
            margin=margin_pad,
            hovermode='closest'
        )

        return fig

    def create_multi_timestep_radar(self, summaries: List[Dict], field_type: str,
                                   query_params: Optional[Dict],
                                   timesteps: List[int],
                                   width: int = 350, height: int = 350,
                                   **kwargs) -> List[Tuple[go.Figure, int]]:
        """Create multiple radar charts for different timesteps."""
        figures = []
        for t in timesteps:
            rows = []
            for s in summaries:
                if t <= len(s.get('timesteps', [])):
                    peak_list = s['field_stats'].get(field_type, {}).get('max', [0])
                    if t-1 < len(peak_list):
                        rows.append({
                            'Name': s['name'],
                            'Energy': s['energy'],
                            'Duration': s['duration'],
                            'Peak_Value': peak_list[t-1]
                        })
            if rows:
                df_t = pd.DataFrame(rows)
                fig = self.create_polar_radar_chart(
                    df_t, field_type, query_params, timestep=t,
                    width=width, height=height, show_legend=(t == timesteps[0]),
                    marker_size_range=(8, 20),
                    **kwargs
                )
                figures.append((fig, t))
        return figures

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

    # --- TAB 3: Polar Radar (REPAIRED & EXPANDED) ---
    with tabs[2]:
        st.subheader("🎯 Polar Radar Visualization")
        all_fields = set()
        for s in st.session_state.summaries: all_fields.update(s['field_stats'].keys())
        col1, col2 = st.columns(2)
        with col1: field_type = st.selectbox("Field Type", sorted(all_fields), key="polar_field")
        with col2:
            max_possible_t = max((len(s.get('timesteps', [])) for s in st.session_state.summaries), default=1)
            t_step = st.number_input("Timestep Index", 1, max_possible_t, 1, key="polar_timestep")

        # --- ROBUST DATA COLLECTION (permissive extraction with warnings) ---
        rows = []
        warnings_found = []
        for s in st.session_state.summaries:
            field_stats = s['field_stats'].get(field_type, {})
            peak_list = field_stats.get('max', [0])
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
            st.caption(f"Data range: Energy {df_polar['Energy'].min():.2f}–{df_polar['Energy'].max():.2f} mJ | "
                      f"Duration {df_polar['Duration'].min():.2f}–{df_polar['Duration'].max():.2f} ns")

            # Styling expanders (full customization)
            with st.expander("🎨 Radar Chart Styling Options", expanded=True):
                colA, colB, colC, colD = st.columns(4)
                with colA:
                    st.markdown("**Title & Text**")
                    title_font_size = st.slider("Title Font Size", 12, 30, 18, key="radar_title_font")
                    title_font_family = st.selectbox("Title Font", ["Arial, sans-serif", "Courier New, monospace", "Times New Roman, serif"], index=0)
                    title_padding = st.slider("Title Padding", 20, 100, 40)
                with colB:
                    st.markdown("**Ticks & Labels**")
                    radial_tick_font = st.slider("Radial Tick Size", 8, 20, 12)
                    angular_tick_font = st.slider("Angular Tick Size", 8, 20, 12)
                    label_font_size = st.slider("Label Size", 8, 20, 12)
                with colC:
                    st.markdown("**Grid Lines**")
                    radial_grid_width = st.slider("Radial Grid Width", 0.5, 3.0, 1.0, 0.1)
                    angular_grid_width = st.slider("Angular Grid Width", 0.5, 3.0, 1.0, 0.1)
                    grid_color = st.color_picker("Grid Color", "#d3d3d3")
                with colD:
                    st.markdown("**Target Marker**")
                    highlight_target = st.checkbox("Show Target", value=True)
                    target_marker_size = st.slider("Target Size", 10, 40, 25)
                    use_colorbar_for_target = st.checkbox("Color Target by Value", value=True)

            with st.expander("📐 Data Processing & Scaling", expanded=False):
                colE, colF = st.columns(2)
                with colE:
                    st.markdown("**Energy Axis (Angular)**")
                    e_max_data = df_polar['Energy'].max()
                    use_custom_energy_max = st.checkbox("Set Custom Energy Max", value=True)
                    if use_custom_energy_max:
                        custom_energy_max = st.number_input("Max Energy (mJ)", 0.0, float(e_max_data*2), float(e_max_data*1.2))
                    else:
                        custom_energy_max = None
                    use_custom_energy_min = st.checkbox("Set Custom Energy Min", value=False)
                    if use_custom_energy_min:
                        custom_energy_min = st.number_input("Min Energy (mJ)", 0.0, float(e_max_data), 0.0)
                    else:
                        custom_energy_min = None
                    add_jitter = st.checkbox("Add Jitter for Overlapping Points", value=True)
                    jitter_amount = st.slider("Jitter (±degrees)", 0.5, 5.0, 2.0)
                with colF:
                    st.markdown("**Value Scaling**")
                    normalize_colors = st.checkbox("Normalize Colors", value=True)
                    if normalize_colors:
                        max_ref = st.number_input("Color Scale Max", value=float(df_polar['Peak_Value'].max() * 1.2))
                    else:
                        max_ref = None
                    radial_auto = st.checkbox("Auto Radial Range", value=True)
                    if not radial_auto:
                        radial_max = st.number_input("Max Duration (ns)", value=float(df_polar['Duration'].max() * 1.1))
                    else:
                        radial_max = None

            # Get target query parameters
            query_params = None
            target_peak = None
            if st.session_state.get('polar_query') and st.session_state.get('interpolation_results'):
                query_params = st.session_state.polar_query
                results = st.session_state.interpolation_results
                if field_type in results['field_predictions']:
                    t_idx = t_step - 1
                    if t_idx < len(results['field_predictions'][field_type]['max']):
                        target_peak = results['field_predictions'][field_type]['max'][t_idx]
                        st.info(f"🎯 Target predicted peak: {target_peak:.3f}")

            # Create radar chart with all parameters
            fig_polar = st.session_state.polar_viz.create_polar_radar_chart(
                df_polar, field_type=field_type, query_params=query_params, timestep=t_step,
                show_legend=True, width=900, height=750,
                radial_grid_width=radial_grid_width, angular_grid_width=angular_grid_width,
                radial_tick_font_size=radial_tick_font, angular_tick_font_size=angular_tick_font,
                title_font_size=title_font_size,
                margin_pad={'l': 60, 'r': 60, 't': 80, 'b': 60},
                energy_min=custom_energy_min, energy_max=custom_energy_max,
                target_peak_value=target_peak if use_colorbar_for_target else None,
                add_jitter=add_jitter, jitter_amount=jitter_amount,
                normalize_colors=normalize_colors, color_reference_max=max_ref,
                radial_axis_max=radial_max, highlight_target=highlight_target
            )
            st.plotly_chart(fig_polar, use_container_width=True)

            # Multi-timestep comparison
            if st.checkbox("📊 Show Multi-Timestep Comparison", key="polar_multi"):
                st.markdown("##### Compare Multiple Timesteps")
                available_ts = list(range(1, min(max_possible_t+1, 6)))
                selected_ts = st.multiselect("Select Timesteps", available_ts, default=[1, min(3, max_possible_t), min(5, max_possible_t)])
                if selected_ts:
                    cols = st.columns(len(selected_ts))
                    for col, ts in zip(cols, selected_ts):
                        rows_ts = []
                        for s in st.session_state.summaries:
                            if ts <= len(s['timesteps']):
                                peak_ts = s['field_stats'].get(field_type, {}).get('max', [0])
                                if ts-1 < len(peak_ts):
                                    rows_ts.append({
                                        'Name': s['name'],
                                        'Energy': s['energy'],
                                        'Duration': s['duration'],
                                        'Peak_Value': peak_ts[ts-1]
                                    })
                        target_val_ts = None
                        if st.session_state.get('interpolation_results'):
                            res = st.session_state.interpolation_results
                            if field_type in res['field_predictions'] and ts-1 < len(res['field_predictions'][field_type]['max']):
                                target_val_ts = res['field_predictions'][field_type]['max'][ts-1]
                        if rows_ts:
                            df_ts = pd.DataFrame(rows_ts)
                            fig_ts = st.session_state.polar_viz.create_polar_radar_chart(
                                df_ts, field_type, query_params, timestep=ts,
                                width=350, height=350, show_legend=(ts == selected_ts[0]),
                                marker_size_range=(8, 20),
                                title_font_size=title_font_size-4,
                                radial_tick_font_size=radial_tick_font-2, angular_tick_font_size=angular_tick_font-2,
                                energy_min=custom_energy_min, energy_max=custom_energy_max,
                                target_peak_value=target_val_ts if use_colorbar_for_target else None,
                                add_jitter=add_jitter, jitter_amount=jitter_amount*0.8,
                                normalize_colors=normalize_colors, color_reference_max=max_ref,
                                highlight_target=highlight_target
                            )
                            with col:
                                st.plotly_chart(fig_ts, use_container_width=True)
                                st.caption(f"Timestep {ts}")

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
