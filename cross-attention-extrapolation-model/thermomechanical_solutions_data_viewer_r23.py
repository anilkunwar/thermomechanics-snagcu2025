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
import json

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
            params.get('s_t', 20.0),
            params.get('temporal_weight', 0.3),
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
                            sim_data['field_info'][key] = ("scalar", 1)
                            fields[key] = np.full((len(vtu_files), n_pts), np.nan, dtype=np.float32)
                        else:
                            sim_data['field_info'][key] = ("vector", arr.shape[1])
                            fields[key] = np.full((len(vtu_files), n_pts, arr.shape[1]), np.nan, dtype=np.float32)
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
                            percentiles = np.percentile(clean_data, [10, 25, 50, 75, 90])
                            summary['field_stats'][field_name]['percentiles'].append(percentiles)
                        else:
                            for key in ['min', 'max', 'mean', 'std', 'q25', 'q50', 'q75']:
                                summary['field_stats'][field_name][key].append(0.0)
                            summary['field_stats'][field_name]['percentiles'].append(np.zeros(5))
                    else:
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

    def load_summaries(self, summaries):
        """Load summary statistics and prepare for attention mechanism"""
        self.source_db = summaries
        if not summaries:
            return
        all_embeddings = []
        all_values = []
        metadata = []
        for summary_idx, summary in enumerate(summaries):
            for timestep_idx, t in enumerate(summary['timesteps']):
                emb = self._compute_enhanced_physics_embedding(summary['energy'], summary['duration'], t)
                all_embeddings.append(emb)
                field_vals = []
                for field in sorted(summary['field_stats'].keys()):
                    stats = summary['field_stats'][field]
                    if timestep_idx < len(stats['mean']):
                        field_vals.extend([stats['mean'][timestep_idx], stats['max'][timestep_idx], stats['std'][timestep_idx]])
                    else:
                        field_vals.extend([0.0, 0.0, 0.0])
                all_values.append(field_vals)
                metadata.append({
                    'summary_idx': summary_idx, 'timestep_idx': timestep_idx,
                    'energy': summary['energy'], 'duration': summary['duration'], 'time': t,
                    'name': summary['name'], 'fourier_number': self._compute_fourier_number(t),
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

    def _compute_fourier_number(self, time_ns):
        time_s = time_ns * 1e-9
        return self.thermal_diffusivity * time_s / (self.characteristic_length ** 2)

    def _compute_thermal_penetration(self, time_ns):
        time_s = time_ns * 1e-9
        return np.sqrt(self.thermal_diffusivity * time_s) * 1e6

    def _compute_enhanced_physics_embedding(self, energy, duration, time):
        logE = np.log1p(energy)
        power = energy / max(duration, 1e-6)
        energy_density = energy / (duration * duration + 1e-6)
        time_ratio = time / max(duration, 1e-3)
        heating_rate = power / max(time, 1e-6)
        cooling_rate = 1.0 / (time + 1e-6)
        thermal_diffusion = np.sqrt(time * 0.1) / max(duration, 1e-3)
        thermal_penetration = np.sqrt(time) / 10.0
        strain_rate = energy_density / (time + 1e-6)
        stress_rate = power / (time + 1e-6)
        fourier_number = self._compute_fourier_number(time)
        thermal_penetration_depth = self._compute_thermal_penetration(time)
        diffusion_time_scale = time / (duration + 1e-6)
        heating_phase = 1.0 if time < duration else 0.0
        cooling_phase = 1.0 if time >= duration else 0.0
        early_time = 1.0 if time < duration * 0.5 else 0.0
        late_time = 1.0 if time > duration * 2.0 else 0.0
        return np.array([logE, duration, time, power, energy_density, time_ratio, heating_rate, cooling_rate,
                         thermal_diffusion, thermal_penetration, strain_rate, stress_rate, fourier_number,
                         thermal_penetration_depth, diffusion_time_scale, heating_phase, cooling_phase,
                         early_time, late_time, np.log1p(power), np.log1p(time), np.sqrt(time),
                         time / (duration + 1e-6)], dtype=np.float32)

    def _compute_ett_gating(self, energy_query, duration_query, time_query, source_metadata=None):
        if source_metadata is None:
            source_metadata = self.source_metadata
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
        if gating_sum > 0:
            return gating / gating_sum
        return np.ones_like(gating) / len(gating)

    def _compute_temporal_similarity(self, query_meta, source_metas):
        similarities = []
        for meta in source_metas:
            time_diff = abs(query_meta['time'] - meta['time'])
            if query_meta['time'] < query_meta['duration'] * 1.5:
                temporal_tolerance = max(query_meta['duration'] * 0.1, 1.0)
            else:
                temporal_tolerance = max(query_meta['duration'] * 0.3, 3.0)
            fourier_similarity = 1.0
            if 'fourier_number' in meta and 'fourier_number' in query_meta:
                fourier_diff = abs(query_meta['fourier_number'] - meta['fourier_number'])
                fourier_similarity = np.exp(-fourier_diff / 0.1)
            time_similarity = np.exp(-time_diff / temporal_tolerance)
            similarities.append((1 - self.temporal_weight) * time_similarity + self.temporal_weight * fourier_similarity)
        return np.array(similarities)

    def _compute_spatial_similarity(self, query_meta, source_metas):
        similarities = []
        for meta in source_metas:
            e_diff = abs(query_meta['energy'] - meta['energy']) / 50.0
            d_diff = abs(query_meta['duration'] - meta['duration']) / 20.0
            total_diff = np.sqrt(e_diff**2 + d_diff**2)
            similarities.append(np.exp(-total_diff / self.sigma_param))
        return np.array(similarities)

    def _multi_head_attention_with_gating(self, query_embedding, query_meta):
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
                scores = (1 - self.spatial_weight) * scores + self.spatial_weight * self._compute_spatial_similarity(query_meta, self.source_metadata)
            if self.temporal_weight > 0:
                scores = (1 - self.temporal_weight) * scores + self.temporal_weight * self._compute_temporal_similarity(query_meta, self.source_metadata)
            head_weights[head] = scores
        avg_weights = np.mean(head_weights, axis=0)
        if self.temperature != 1.0:
            avg_weights = avg_weights ** (1.0 / self.temperature)
        max_weight = np.max(avg_weights)
        exp_weights = np.exp(avg_weights - max_weight)
        physics_attention = exp_weights / (np.sum(exp_weights) + 1e-12)
        ett_gating = self._compute_ett_gating(query_meta['energy'], query_meta['duration'], query_meta['time'])
        combined_weights = physics_attention * ett_gating
        combined_sum = np.sum(combined_weights)
        if combined_sum > 1e-12:
            final_weights = combined_weights / combined_sum
        else:
            final_weights = physics_attention
        if len(self.source_values) > 0:
            prediction = np.sum(final_weights[:, np.newaxis] * self.source_values, axis=0)
        else:
            prediction = np.zeros(1)
        return prediction, final_weights, physics_attention, ett_gating

    def predict_field_statistics(self, energy_query, duration_query, time_query):
        if not self.fitted:
            return None
        query_embedding = self._compute_enhanced_physics_embedding(energy_query, duration_query, time_query)
        query_meta = {'energy': energy_query, 'duration': duration_query, 'time': time_query,
                      'fourier_number': self._compute_fourier_number(time_query),
                      'thermal_penetration': self._compute_thermal_penetration(time_query)}
        prediction, final_weights, physics_attention, ett_gating = self._multi_head_attention_with_gating(query_embedding, query_meta)
        if prediction is None:
            return None
        result = {'prediction': prediction, 'attention_weights': final_weights,
                  'physics_attention': physics_attention, 'ett_gating': ett_gating,
                  'confidence': float(np.max(final_weights)) if len(final_weights) > 0 else 0.0,
                  'temporal_confidence': self._compute_temporal_confidence(time_query, duration_query),
                  'heat_transfer_indicators': self._compute_heat_transfer_indicators(energy_query, duration_query, time_query),
                  'field_predictions': {}}
        if self.source_db:
            field_order = sorted(self.source_db[0]['field_stats'].keys())
            n_stats_per_field = 3
            for i, field in enumerate(field_order):
                start_idx = i * n_stats_per_field
                if start_idx + 2 < len(prediction):
                    result['field_predictions'][field] = {'mean': float(prediction[start_idx]),
                                                          'max': float(prediction[start_idx + 1]),
                                                          'std': float(prediction[start_idx + 2])}
        return result

    def _compute_temporal_confidence(self, time_query, duration_query):
        if time_query < duration_query * 0.5:
            return 0.6
        elif time_query < duration_query * 1.5:
            return 0.8
        return 0.9

    def _compute_heat_transfer_indicators(self, energy, duration, time):
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
        return {'phase': phase, 'regime': regime, 'fourier_number': fourier_number,
                'thermal_penetration_um': thermal_penetration,
                'normalized_time': time / max(duration, 1e-6), 'energy_density': energy / duration}

    def predict_time_series(self, energy_query, duration_query, time_points):
        results = {'time_points': time_points, 'field_predictions': {}, 'attention_maps': [],
                   'physics_attention_maps': [], 'ett_gating_maps': [], 'confidence_scores': [],
                   'temporal_confidences': [], 'heat_transfer_indicators': []}
        if self.source_db:
            for summary in self.source_db:
                for field in summary['field_stats'].keys():
                    if field not in results['field_predictions']:
                        results['field_predictions'][field] = {'mean': [], 'max': [], 'std': []}
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
        if not self.fitted or len(attention_weights) == 0:
            return None
        first_sim = next(iter(simulations.values()))
        if 'fields' not in first_sim or field_name not in first_sim['fields']:
            return None
        field_shape = first_sim['fields'][field_name].shape[1:]
        interpolated_field = np.zeros(first_sim['fields'][field_name].shape[1] if len(field_shape) == 0 else field_shape, dtype=np.float32)
        total_weight = 0.0
        n_sources_used = 0
        for idx, weight in enumerate(attention_weights):
            if weight < 1e-6:
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
            return interpolated_field / total_weight
        return None

    def _assess_temporal_coherence(self, source_metadata, attention_weights):
        if len(source_metadata) == 0 or len(attention_weights) == 0:
            return 0.0
        times = np.array([meta['time'] for meta in source_metadata])
        weighted_times = times * attention_weights
        mean_time = np.sum(weighted_times) / np.sum(attention_weights)
        time_diff = times - mean_time
        weighted_variance = np.sum(attention_weights * time_diff**2) / np.sum(attention_weights)
        temporal_spread = np.sqrt(weighted_variance)
        avg_duration = np.mean([meta['duration'] for meta in source_metadata])
        normalized_spread = temporal_spread / max(avg_duration, 1e-6)
        return float(np.exp(-normalized_spread))

    def export_interpolated_vtu(self, field_name, interpolated_values, simulations, output_path):
        if interpolated_values is None or len(simulations) == 0:
            return False
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
# ADVANCED VISUALIZATION COMPONENTS WITH ST-DGPA ANALYSIS
# =============================================
class EnhancedVisualizer:
    EXTENDED_COLORMAPS = [
        'Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'Rainbow', 'Jet', 'Hot', 'Cool', 'Portland',
        'Bluered', 'Electric', 'Thermal', 'Balance', 'Brwnyl', 'Darkmint', 'Emrld', 'Mint', 'Oranges',
        'Purp', 'Purples', 'Sunset', 'Sunsetdark', 'Teal', 'Tealgrn', 'Twilight', 'Burg', 'Burgyl'
    ]

    @staticmethod
    def create_stdgpa_analysis(results, energy_query, duration_query, time_points):
        if not results or 'attention_maps' not in results or len(results['attention_maps']) == 0:
            return None
        timestep_idx = len(time_points) // 2
        time = time_points[timestep_idx]
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=["ST-DGPA Final Weights", "Physics Attention Only", "(E, τ, t) Gating Only",
                            "ST-DGPA vs Physics Attention", "Temporal Coherence Analysis", "Heat Transfer Phase",
                            "Parameter Space 3D", "Attention Network", "Weight Evolution"],
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
        
        if st.session_state.get('summaries'):
            times, weights = [], []
            if hasattr(st.session_state.extrapolator, 'source_metadata'):
                for i, weight in enumerate(final_weights):
                    if weight > 0.01:
                        meta = st.session_state.extrapolator.source_metadata[i]
                        times.append(meta['time'])
                        weights.append(weight)
            if times and weights:
                fig.add_trace(go.Scatter(x=times, y=weights, mode='markers', marker=dict(size=np.array(weights)*50, color=weights, colorscale='Viridis', showscale=False), name='Weight vs Time', showlegend=False), row=2, col=2)
                fig.add_vline(x=time, line_dash="dash", line_color="red", row=2, col=2)

        if 'heat_transfer_indicators' in results and results['heat_transfer_indicators']:
            indicators = results['heat_transfer_indicators'][timestep_idx]
            if indicators:
                phase = indicators.get('phase', 'Unknown')
                if phase == 'Early Heating' or phase == 'Heating': values = [0.9, 0.3, 0.2, 0.1]
                elif phase == 'Early Cooling': values = [0.4, 0.8, 0.3, 0.1]
                elif phase == 'Diffusion Cooling': values = [0.2, 0.5, 0.9, 0.2]
                else: values = [0.7, 0.5, 0.3, 0.2]
                fig.add_trace(go.Scatterpolar(r=list(values)+[values[0]], theta=['Heating','Cooling','Diffusion','Adiabatic']+['Heating'], fill='toself', name='Heat Transfer', line=dict(color='orange', width=2), fillcolor='rgba(255, 165, 0, 0.5)', showlegend=False), row=2, col=3)

        if st.session_state.get('summaries'):
            energies, durations, times_3d, weights_3d = [], [], [], []
            for summary in st.session_state.summaries[:10]:
                for t in summary['timesteps'][:5]:
                    energies.append(summary['energy']); durations.append(summary['duration']); times_3d.append(t)
                    weights_3d.append(np.mean(final_weights) if len(final_weights)>0 else 0.1)
            fig.add_trace(go.Scatter3d(x=energies, y=durations, z=times_3d, mode='markers', marker=dict(size=np.array(weights_3d)*20, color=weights_3d, colorscale='Viridis', opacity=0.7, colorbar=dict(title="Weight", x=1.05)), name='Sources', showlegend=False), row=3, col=1)
            fig.add_trace(go.Scatter3d(x=[energy_query], y=[duration_query], z=[time], mode='markers', marker=dict(size=15, color='red', symbol='diamond'), name='Query', showlegend=False), row=3, col=1)
        
        if len(final_weights) > 5:
            top_indices = np.argsort(final_weights)[-5:]
            top_weights = final_weights[top_indices]
            fig.add_trace(go.Scatter(x=[0]+list(range(1,6)), y=[0]*6, mode='markers+text', text=['Query']+[f'Source {i+1}' for i in top_indices], textposition="top center", marker=dict(size=[30]+list(top_weights*50), color=['red']+['blue']*5), name='Network', showlegend=False), row=3, col=2)
            for i in range(1, 6):
                fig.add_trace(go.Scatter(x=[0, i], y=[0, 0], mode='lines', line=dict(width=top_weights[i-1]*10, color='gray'), showlegend=False), row=3, col=2)

        if len(results['attention_maps']) > 1 and len(final_weights) > 0:
            top_idx = np.argmax(final_weights)
            weight_evolution = [results['attention_maps'][t_idx][top_idx] for t_idx in range(len(results['attention_maps'])) if top_idx < len(results['attention_maps'][t_idx])]
            if weight_evolution:
                fig.add_trace(go.Scatter(x=time_points[:len(weight_evolution)], y=weight_evolution, mode='lines+markers', line=dict(color='purple', width=3), name='Evolution', showlegend=False), row=3, col=3)

        fig.update_layout(height=1000, title_text=f"ST-DGPA Analysis at t={time} ns (E={energy_query:.1f} mJ, τ={duration_query:.1f} ns)", showlegend=True, legend=dict(x=1.05, y=1))
        return fig

    @staticmethod
    def create_temporal_analysis(results, time_points):
        if not results or 'heat_transfer_indicators' not in results:
            return None
        fig = make_subplots(rows=2, cols=2, subplot_titles=["Heat Transfer Phase Evolution", "Fourier Number Evolution", "Temporal Confidence", "Thermal Penetration Depth"], vertical_spacing=0.15, horizontal_spacing=0.15)
        phases = [ind.get('phase', 'Unknown') for ind in results['heat_transfer_indicators'] if ind]
        phase_mapping = {'Early Heating': 0, 'Heating': 1, 'Early Cooling': 2, 'Diffusion Cooling': 3}
        fig.add_trace(go.Scatter(x=time_points[:len(phases)], y=[phase_mapping.get(p,0) for p in phases], mode='lines+markers', line=dict(color='red', width=3), name='Phase', showlegend=False), row=1, col=1)
        for pn, pv in phase_mapping.items():
            fig.add_hline(y=pv, line_dash="dot", line_color="gray", annotation_text=pn, row=1, col=1)
        
        fn = [ind.get('fourier_number', 0) for ind in results['heat_transfer_indicators'] if ind]
        if fn: fig.add_trace(go.Scatter(x=time_points[:len(fn)], y=fn, mode='lines+markers', line=dict(color='blue', width=3), showlegend=False), row=1, col=2)
        if 'temporal_confidences' in results: fig.add_trace(go.Scatter(x=time_points[:len(results['temporal_confidences'])], y=results['temporal_confidences'], mode='lines+markers', line=dict(color='green', width=3), fill='tozeroy', showlegend=False), row=2, col=1)
        pd_list = [ind.get('thermal_penetration_um', 0) for ind in results['heat_transfer_indicators'] if ind]
        if pd_list: fig.add_trace(go.Scatter(x=time_points[:len(pd_list)], y=pd_list, mode='lines+markers', line=dict(color='orange', width=3), showlegend=False), row=2, col=2)
        
        fig.update_layout(height=700, title_text="Temporal Analysis of Heat Transfer Characteristics", showlegend=False)
        return fig

    @staticmethod
    def create_sunburst_chart(summaries, selected_field='temperature', highlight_sim=None):
        labels, parents, values, colors = [], [], [], []
        labels.append("All Simulations"); parents.append(""); values.append(len(summaries)); colors.append("#1f77b4")
        energy_groups = {}
        for summary in summaries:
            energy_key = f"{summary['energy']:.1f} mJ"
            if energy_key not in energy_groups:
                energy_groups[energy_key] = []
            energy_groups[energy_key].append(summary)
        for energy_key, energy_sims in energy_groups.items():
            labels.append(f"Energy: {energy_key}"); parents.append("All Simulations"); values.append(len(energy_sims))
            colors.append("#ff7f0e" if highlight_sim and any(s['name'] == highlight_sim for s in energy_sims) else "#2ca02c")
            for summary in energy_sims:
                sim_label = f"{summary['name']}"
                labels.append(sim_label); parents.append(f"Energy: {energy_key}"); values.append(1)
                colors.append("#d62728" if highlight_sim and summary['name'] == highlight_sim else "#9467bd")
                if selected_field in summary['field_stats'] and summary['field_stats'][selected_field]['max']:
                    avg_max = np.mean(summary['field_stats'][selected_field]['max'])
                    labels.append(f"{selected_field}: {avg_max:.1f}"); parents.append(sim_label)
                    values.append(avg_max if avg_max > 0 else 1e-6); colors.append("#8c564b")
        values = [max(v, 1e-6) for v in values]
        fig = go.Figure(go.Sunburst(labels=labels, parents=parents, values=values, branchvalues="total",
                                    marker=dict(colors=colors, colorscale='Viridis', line=dict(width=2, color='white')),
                                    hovertemplate='<b>%{label}</b><br>Value: %{value:.2f}<br>Parent: %{parent}<extra></extra>',
                                    textinfo="label+value", textfont=dict(size=12)))
        title = f"Simulation Hierarchy - {selected_field}"
        if highlight_sim: title += f" (Target: {highlight_sim})"
        fig.update_layout(title=title, height=700, margin=dict(t=50, b=20, l=20, r=20))
        return fig

    @staticmethod
    def create_radar_chart(summaries, simulation_names, target_sim=None):
        all_fields = set()
        for summary in summaries: all_fields.update(summary['field_stats'].keys())
        if not all_fields: return go.Figure()
        selected_fields = list(all_fields)[:6]
        fig = go.Figure()
        for sim_name in simulation_names:
            summary = next((s for s in summaries if s['name'] == sim_name), None)
            if not summary: continue
            r_values, theta_values = [], []
            for field in selected_fields:
                if field in summary['field_stats'] and summary['field_stats'][field]['mean']:
                    avg_value = np.mean(summary['field_stats'][field]['mean'])
                    r_values.append(avg_value if avg_value > 0 else 1e-6)
                else: r_values.append(1e-6)
                theta_values.append(f"{field[:15]}...")
            line_width = 4 if target_sim and sim_name == target_sim else 2
            fill_opacity = 0.6 if target_sim and sim_name == target_sim else 0.3
            color = 'red' if target_sim and sim_name == target_sim else None
            fig.add_trace(go.Scatterpolar(r=r_values, theta=theta_values, fill='toself', name=sim_name,
                                          line=dict(width=line_width, color=color), fillcolor=f'rgba(255,0,0,{fill_opacity})' if color else None, opacity=0.8))
        if fig.data:
            max_r = max(max(t.r) for t in fig.data)
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, max_r*1.2], tickfont=dict(size=10), gridcolor='lightgray', linecolor='gray'), angularaxis=dict(tickfont=dict(size=11), rotation=90, direction="clockwise"), bgcolor='white', gridshape='circular'), showlegend=True, title="Radar Chart: Simulation Comparison", height=600, legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.05))
        return fig

    @staticmethod
    def create_attention_heatmap_3d(attention_weights, source_metadata):
        if len(attention_weights) == 0: return go.Figure()
        energies = np.array([m['energy'] for m in source_metadata])
        durations = np.array([m['duration'] for m in source_metadata])
        times = np.array([m['time'] for m in source_metadata])
        fig = go.Figure(data=go.Scatter3d(x=energies, y=durations, z=times, mode='markers',
                                          marker=dict(size=10, color=attention_weights, colorscale='Viridis', opacity=0.8, showscale=True),
                                          text=[f"E: {e:.1f} mJ<br>τ: {d:.1f} ns<br>t: {t:.1f} ns<br>Weight: {w:.4f}" for e,d,t,w in zip(energies, durations, times, attention_weights)],
                                          hovertemplate='%{text}<extra></extra>'))
        fig.update_layout(title="3D Attention Weight Distribution", scene=dict(xaxis_title="Energy", yaxis_title="Duration", zaxis_title="Time"), height=600, margin=dict(l=0, r=0, t=40, b=0))
        return fig

    @staticmethod
    def create_attention_network(attention_weights, source_metadata, top_k=10):
        if len(attention_weights) == 0: return go.Figure()
        sim_attention = {}
        for weight, meta in zip(attention_weights, source_metadata):
            sim_attention.setdefault(meta['name'], []).append(weight)
        avg_attention = {k: np.mean(v) for k, v in sim_attention.items()}
        sorted_sims = sorted(avg_attention.items(), key=lambda x: x[1], reverse=True)[:top_k]
        if not sorted_sims: return go.Figure()
        G = nx.Graph(); G.add_node("QUERY", size=50, color='red', label="Query")
        for i, (sim_name, weight) in enumerate(sorted_sims):
            node_id = f"SIM_{i}"
            sim_meta = next((m for m in source_metadata if m['name'] == sim_name), None)
            G.add_node(node_id, size=30*weight/max(avg_attention.values()), color='blue', label=sim_name,
                       energy=sim_meta['energy'] if sim_meta else 0, duration=sim_meta['duration'] if sim_meta else 0, time=sim_meta['time'] if sim_meta else 0, weight=weight)
            G.add_edge("QUERY", node_id, weight=weight, width=3*weight)
        pos = nx.spring_layout(G, seed=42, k=2)
        ex, ey = [], []
        for edge in G.edges():
            ex.extend([pos[edge[0]][0], pos[edge[1]][0], None]); ey.extend([pos[edge[0]][1], pos[edge[1]][1], None])
        edge_trace = go.Scatter(x=ex, y=ey, line=dict(width=2, color='gray'), hoverinfo='none', mode='lines')
        nx_, ny_, nt_, ns_, nc_ = [], [], [], [], []
        for node in G.nodes():
            nx_.append(pos[node][0]); ny_.append(pos[node][1])
            if node == "QUERY": nt_.append("QUERY"); ns_.append(30); nc_.append('red')
            else: d = G.nodes[node]; nt_.append(f"{d['label']}<br>E:{d['energy']:.1f} τ:{d['duration']:.1f} t:{d['time']:.1f} w:{d['weight']:.3f}"); ns_.append(d['size']+10); nc_.append('blue')
        node_trace = go.Scatter(x=nx_, y=ny_, mode='markers+text', text=['Query' if n=='QUERY' else f"Sim{i}" for i,n in enumerate(G.nodes()) if n!='QUERY'], textposition="middle center", hoverinfo='text', hovertext=nt_, marker=dict(size=ns_, color=nc_, line=dict(width=2, color='white')))
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(title=f"Attention Network (Top {len(sorted_sims)})", showlegend=False, height=500, plot_bgcolor='white', xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
        return fig

    @staticmethod
    def create_field_evolution_comparison(summaries, simulation_names, selected_field, target_sim=None):
        fig = go.Figure()
        for sim_name in simulation_names:
            summary = next((s for s in summaries if s['name'] == sim_name), None)
            if summary and selected_field in summary['field_stats']:
                stats = summary['field_stats'][selected_field]
                lw, ld = (4, 'solid') if target_sim and sim_name == target_sim else (2, 'dash')
                fig.add_trace(go.Scatter(x=summary['timesteps'], y=stats['mean'], mode='lines+markers', name=f"{sim_name} (mean)", line=dict(width=lw, dash=ld), opacity=0.8))
                if stats['std']:
                    yu = np.array(stats['mean'])+np.array(stats['std']); yl = np.array(stats['mean'])-np.array(stats['std'])
                    fig.add_trace(go.Scatter(x=summary['timesteps']+summary['timesteps'][::-1], y=np.concatenate([yu, yl[::-1]]), fill='toself', fillcolor='rgba(128,128,128,0.05)', line=dict(color='rgba(255,255,255,0)'), showlegend=False))
        if fig.data: fig.update_layout(title=f"{selected_field} Evolution", xaxis_title="Timestep", yaxis_title="Value", hovermode="x unified", height=500, legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02))
        return fig

# =============================================
# MAIN INTEGRATED APPLICATION
# =============================================
def main():
    st.set_page_config(page_title="Enhanced FEA Laser Simulation Platform with ST-DGPA", layout="wide", initial_sidebar_state="expanded", page_icon="🔬")
    st.markdown("""<style>
    .main-header{font-size:3rem;background:linear-gradient(90deg,#1E88E5,#4A00E0);-webkit-background-clip:text;-webkit-text-fill-color:transparent;text-align:center;margin-bottom:1.5rem;font-weight:800}
    .sub-header{font-size:1.8rem;color:#2c3e50;margin-top:1.5rem;margin-bottom:1rem;padding-bottom:0.5rem;border-bottom:3px solid #3498db;font-weight:600}
    .info-box{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;padding:1.5rem;border-radius:10px;margin:1.5rem 0;box-shadow:0 10px 20px rgba(0,0,0,0.1)}
    .warning-box{background:linear-gradient(135deg,#f093fb 0%,#f5576c 100%);color:white;padding:1.5rem;border-radius:10px;margin:1.5rem 0}
    .success-box{background:linear-gradient(135deg,#4facfe 0%,#00f2fe 100%);color:white;padding:1.5rem;border-radius:10px;margin:1.5rem 0}
    .stdgpa-box{background:linear-gradient(135deg,#f093fb 0%,#00f2fe 100%);color:white;padding:1.5rem;border-radius:10px;margin:1.5rem 0}
    .heat-transfer-box{background:linear-gradient(135deg,#ff9a9e 0%,#fad0c4 100%);color:#333;padding:1.5rem;border-radius:10px;margin:1.5rem 0}
    .metric-card{background:white;padding:1rem;border-radius:10px;box-shadow:0 4px 6px rgba(0,0,0,0.1);margin-bottom:1rem}
    .cache-status{background:#e8f5e8;border-left:4px solid #4CAF50;padding:0.75rem;margin:0.5rem 0;border-radius:4px;font-size:0.9rem}
    </style>""", unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">🔬 Enhanced FEA Laser Simulation Platform with ST-DGPA</h1>', unsafe_allow_html=True)

    if 'data_loader' not in st.session_state: st.session_state.data_loader = UnifiedFEADataLoader()
    if 'extrapolator' not in st.session_state: st.session_state.extrapolator = SpatioTemporalGatedPhysicsAttentionExtrapolator()
    if 'visualizer' not in st.session_state: st.session_state.visualizer = EnhancedVisualizer()
    if 'data_loaded' not in st.session_state: st.session_state.data_loaded = False
    if 'current_mode' not in st.session_state: st.session_state.current_mode = "Data Viewer"
    if 'simulations' not in st.session_state: st.session_state.simulations = {}
    if 'summaries' not in st.session_state: st.session_state.summaries = []
    if 'interpolation_results' not in st.session_state: st.session_state.interpolation_results = None
    if 'interpolation_params' not in st.session_state: st.session_state.interpolation_params = None
    if 'interpolation_3d_cache' not in st.session_state: st.session_state.interpolation_3d_cache = {}
    if 'interpolation_field_history' not in st.session_state: st.session_state.interpolation_field_history = OrderedDict()
    if 'current_3d_field' not in st.session_state: st.session_state.current_3d_field = None
    if 'current_3d_timestep' not in st.session_state: st.session_state.current_3d_timestep = 0
    if 'last_prediction_id' not in st.session_state: st.session_state.last_prediction_id = None
    if 'selected_colormap' not in st.session_state: st.session_state.selected_colormap = "Viridis"

    with st.sidebar:
        st.markdown("### ⚙️ Navigation")
        app_mode = st.radio("Select Mode", ["Data Viewer", "Interpolation/Extrapolation", "Comparative Analysis", "ST-DGPA Analysis", "Heat Transfer Analysis"],
                            index=["Data Viewer", "Interpolation/Extrapolation", "Comparative Analysis", "ST-DGPA Analysis", "Heat Transfer Analysis"].index(st.session_state.current_mode), key="nav_mode")
        st.session_state.current_mode = app_mode
        st.markdown("---")
        st.markdown("### 📊 Data Settings")
        col1, col2 = st.columns(2)
        with col1: load_full_data = st.checkbox("Load Full Mesh", value=True)
        with col2: st.session_state.selected_colormap = st.selectbox("Colormap", EnhancedVisualizer.EXTENDED_COLORMAPS, index=0)
        if st.session_state.current_mode == "Interpolation/Extrapolation" and not load_full_
            st.warning("⚠️ Full mesh loading is required for 3D interpolation visualization.")
        if st.button("🔄 Load All Simulations", type="primary", use_container_width=True):
            with st.spinner("Loading simulation data..."):
                CacheManager.clear_3d_cache(); st.session_state.last_prediction_id = None
                simulations, summaries = st.session_state.data_loader.load_all_simulations(load_full_mesh=load_full_data)
                st.session_state.simulations, st.session_state.summaries = simulations, summaries
                if simulations and summaries: st.session_state.extrapolator.load_summaries(summaries)
                st.session_state.data_loaded = True
                st.session_state.available_fields = set()
                for s in summaries: st.session_state.available_fields.update(s['field_stats'].keys())
        if st.session_state.data_loaded and 'interpolation_3d_cache' in st.session_state:
            st.markdown("---")
            st.markdown("### 🗄️ Cache Management")
            with st.expander("Cache Statistics", expanded=False):
                cache_size = len(st.session_state.interpolation_3d_cache)
                st.metric("Cached Fields", cache_size)
                if cache_size > 0: st.write("**Cached Fields:**")
                for cache_key, cache_data in list(st.session_state.interpolation_3d_cache.items())[:5]:
                    st.caption(f"• {cache_data.get('field_name')} (t={cache_data.get('timestep_idx')})")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Clear Cache", use_container_width=True): CacheManager.clear_3d_cache(); st.rerun()
        if st.session_state.data_loaded:
            st.markdown("---")
            st.markdown("### 📈 Loaded Data")
            st.metric("Simulations", len(st.session_state.simulations))
            st.metric("Available Fields", len(st.session_state.available_fields))

    if app_mode == "Data Viewer": render_data_viewer()
    elif app_mode == "Interpolation/Extrapolation": render_interpolation_extrapolation()
    elif app_mode == "Comparative Analysis": render_comparative_analysis()
    elif app_mode == "ST-DGPA Analysis": render_stdgpa_analysis()
    elif app_mode == "Heat Transfer Analysis": render_heat_transfer_analysis()

def render_data_viewer():
    st.markdown('<h2 class="sub-header">📁 Data Viewer</h2>', unsafe_allow_html=True)
    if not st.session_state.data_loaded: st.warning("Please load simulations first."); return
    simulations = st.session_state.simulations
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1: sim_name = st.selectbox("Select Simulation", sorted(simulations.keys()), key="viewer_sim_select")
    sim = simulations[sim_name]
    with col2: st.metric("Energy", f"{sim['energy_mJ']:.2f} mJ")
    with col3: st.metric("Duration", f"{sim['duration_ns']:.2f} ns")
    if not sim.get('has_mesh', False): st.warning("Reload with 'Load Full Mesh' enabled."); return
    if not sim.get('field_info'): st.error("No field data available."); return
    col1, col2, col3, col4 = st.columns(4)
    with col1: field = st.selectbox("Select Field", sorted(sim['field_info'].keys()), key="viewer_field_select")
    with col2: timestep = st.slider("Timestep", 0, sim['n_timesteps']-1, 0, key="viewer_timestep_slider")
    with col3: colormap = st.selectbox("Colormap", EnhancedVisualizer.EXTENDED_COLORMAPS, index=0, key=f"colormap_{field}")
    with col4: opacity = st.slider("Opacity", 0.0, 1.0, 0.9, 0.05, key="viewer_opacity")
    col1, col2, col3, col4 = st.columns(4)
    with col1: aspect_mode = st.selectbox("Aspect", ["data", "cube", "auto"], key="viewer_aspect")
    with col2: camera_preset = st.selectbox("Camera", ["Isometric", "Front", "Side", "Top", "Bottom"], index=0, key="viewer_camera")
    with col3: lighting_preset = st.selectbox("Lighting", ["Default", "Shiny", "Matte", "High Contrast"], index=0, key="viewer_lighting")
    with col4: bg_mode = st.selectbox("Theme", ["Light", "Dark"], index=0, key="viewer_theme")
    kind, _ = sim['field_info'][field]; raw = sim['fields'][field][timestep]
    values = np.where(np.isnan(raw), 0, raw) if kind=="scalar" else np.where(np.isnan(np.linalg.norm(raw, axis=1)), 0, np.linalg.norm(raw, axis=1))
    data_min, data_max = float(np.min(values)), float(np.max(values))
    st.markdown('<h5 class="sub-header">🌈 Color Scale Limits</h5>', unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    with col_a: auto_scale = st.checkbox("Auto Scale", value=True, key=f"auto_scale_{field}")
    with col_b: cmin = st.number_input("Min", value=data_min, format="%.3f", disabled=auto_scale, key=f"cmin_{field}")
    with col_c: cmax = st.number_input("Max", value=data_max, format="%.3f", disabled=auto_scale, key=f"cmax_{field}")
    if auto_scale: cmin, cmax = None, None
    lighting_map = {"Default": dict(ambient=0.8, diffuse=0.8, specular=0.5, roughness=0.5), "Shiny": dict(ambient=0.6, diffuse=0.9, specular=0.8, roughness=0.2), "Matte": dict(ambient=0.9, diffuse=0.6, specular=0.1, roughness=0.9), "High Contrast": dict(ambient=0.4, diffuse=0.9, specular=0.7, roughness=0.4)}
    camera_map = {"Isometric": dict(eye=dict(x=1.5, y=1.5, z=1.5)), "Front": dict(eye=dict(x=0, y=2, z=0.1)), "Side": dict(eye=dict(x=2, y=0, z=0.1)), "Top": dict(eye=dict(x=0, y=0, z=2)), "Bottom": dict(eye=dict(x=0, y=0, z=-2))}
    lighting = lighting_map[lighting_preset]; camera = camera_map[camera_preset]
    if bg_mode == "Dark": plot_bgcolor, paper_bgcolor, grid_color, font_color = "rgb(17,17,17)", "rgb(17,17,17)", "rgb(40,40,40)", "white"
    else: plot_bgcolor, paper_bgcolor, grid_color, font_color = "white", "white", "lightgray", "black"
    pts = sim['points']; tri = sim.get('triangles')
    if tri is not None and len(tri) > 0:
        valid = tri[np.all(tri < len(pts), axis=1)]
        trace = go.Mesh3d(x=pts[:,0], y=pts[:,1], z=pts[:,2], i=valid[:,0], j=valid[:,1], k=valid[:,2], intensity=values, colorscale=colormap, cmin=cmin, cmax=cmax, opacity=opacity, lighting=lighting, hovertemplate=f'<b>{field}:</b> %{{intensity:.3f}}<extra></extra>') if len(valid)>0 else go.Scatter3d(x=pts[:,0], y=pts[:,1], z=pts[:,2], mode='markers', marker=dict(size=4, color=values, colorscale=colormap, cmin=cmin, cmax=cmax, opacity=opacity))
    else:
        trace = go.Scatter3d(x=pts[:,0], y=pts[:,1], z=pts[:,2], mode='markers', marker=dict(size=4, color=values, colorscale=colormap, cmin=cmin, cmax=cmax, opacity=opacity))
    fig = go.Figure(data=trace).update_layout(title=dict(text=f"{field} at Timestep {timestep+1}", font=dict(size=18, color=font_color)), scene=dict(aspectmode=aspect_mode, camera=camera, xaxis=dict(showbackground=True, backgroundcolor=plot_bgcolor, gridcolor=grid_color, color=font_color, title="X"), yaxis=dict(showbackground=True, backgroundcolor=plot_bgcolor, gridcolor=grid_color, color=font_color, title="Y"), zaxis=dict(showbackground=True, backgroundcolor=plot_bgcolor, gridcolor=grid_color, color=font_color, title="Z")), plot_bgcolor=plot_bgcolor, paper_bgcolor=paper_bgcolor, height=700, margin=dict(l=0, r=0, t=50, b=0))
    for trace in fig.data:
        if hasattr(trace, 'colorbar') and trace.colorbar: trace.colorbar.title.font.color = font_color; trace.colorbar.tickfont.color = font_color
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<h3 class="sub-header">📊 Field Statistics</h3>', unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("Min", f"{np.min(values):.3f}")
    with col2: st.metric("Max", f"{np.max(values):.3f}")
    with col3: st.metric("Mean", f"{np.mean(values):.3f}")
    with col4: st.metric("Std", f"{np.std(values):.3f}")
    with col5: st.metric("Range", f"{np.max(values)-np.min(values):.3f}")
    summary = next((s for s in st.session_state.summaries if s['name']==sim_name), None)
    if summary and field in summary['field_stats']:
        stats = summary['field_stats'][field]; fig_time = go.Figure()
        if stats['mean']: fig_time.add_trace(go.Scatter(x=summary['timesteps'], y=stats['mean'], mode='lines', name='Mean', line=dict(color='blue', width=3)))
        if stats['std']: fig_time.add_trace(go.Scatter(x=summary['timesteps']+summary['timesteps'][::-1], y=np.concatenate([np.array(stats['mean'])+np.array(stats['std']), (np.array(stats['mean'])-np.array(stats['std']))[::-1]]), fill='toself', fillcolor='rgba(0,100,255,0.2)', line=dict(color='rgba(255,255,255,0)'), showlegend=False))
        if stats['max']: fig_time.add_trace(go.Scatter(x=summary['timesteps'], y=stats['max'], mode='lines', name='Max', line=dict(color='red', width=2, dash='dash')))
        fig_time.update_layout(title=f"{field} Statistics Over Time", xaxis_title="Timestep", yaxis_title=f"{field} Value", hovermode="x unified", height=400)
        st.plotly_chart(fig_time, use_container_width=True)

def render_interpolation_extrapolation():
    st.markdown('<h2 class="sub-header">🔮 Interpolation/Extrapolation Engine with ST-DGPA</h2>', unsafe_allow_html=True)
    if not st.session_state.data_loaded: st.warning("Please load simulations first."); return
    if st.session_state.simulations and not next(iter(st.session_state.simulations.values())).get('has_mesh', False): st.warning("⚠️ Full mesh required for 3D visualization.")
    st.markdown("""<div class="stdgpa-box"><h3>🧠 Spatio-Temporal Gated Physics Attention (ST-DGPA)</h3><p>Uses <strong>ST-DGPA</strong> to interpolate/extrapolate with (E, τ, t) proximity gating and heat transfer characterization.</p></div>""", unsafe_allow_html=True)
    with st.expander("📋 Loaded Simulations Summary", expanded=True):
        if st.session_state.summaries:
            df = pd.DataFrame([{'Simulation': s['name'], 'Energy (mJ)': s['energy'], 'Duration (ns)': s['duration'], 'Timesteps': len(s['timesteps']), 'Fields': ', '.join(sorted(s['field_stats'].keys())[:3])+'...' if len(s['field_stats'])>3 else ', '.join(sorted(s['field_stats'].keys()))} for s in st.session_state.summaries])
            st.dataframe(df, use_container_width=True, height=300)
    st.markdown('<h3 class="sub-header">🎯 Query Parameters</h3>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        energies = [s['energy'] for s in st.session_state.summaries]
        energy_query = st.number_input("Energy (mJ)", min_value=min(energies)*0.5, max_value=max(energies)*2.0, value=np.mean(energies), step=0.1, key="interp_energy")
    with col2:
        durations = [s['duration'] for s in st.session_state.summaries]
        duration_query = st.number_input("Pulse Duration (ns)", min_value=min(durations)*0.5, max_value=max(durations)*2.0, value=np.mean(durations), step=0.1, key="interp_duration")
    with col3: max_time = st.number_input("Max Time (ns)", min_value=1, max_value=200, value=50, key="interp_maxtime")
    with col4: time_res = st.selectbox("Resolution", ["1 ns", "2 ns", "5 ns", "10 ns"], index=1, key="interp_resolution")
    ts = {"1 ns": 1, "2 ns": 2, "5 ns": 5, "10 ns": 10}
    time_points = np.arange(1, max_time+1, ts[time_res])
    with st.expander("⚙️ ST-DGPA Attention Parameters", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1: sigma_param = st.slider("σ", 0.1, 1.0, 0.3, 0.05, key="interp_sigma")
        with col2: spatial_weight = st.slider("Spatial Weight", 0.0, 1.0, 0.5, 0.05, key="interp_spatial")
        with col3: n_heads = st.slider("Heads", 1, 8, 4, 1, key="interp_heads")
        with col4: temperature = st.slider("Temp", 0.1, 2.0, 1.0, 0.1, key="interp_temp")
        col5, col6, col7, col8 = st.columns(4)
        with col5: sigma_g = st.slider("σ_g", 0.05, 1.0, 0.20, 0.05, key="interp_sigma_g")
        with col6: s_E = st.slider("s_E", 0.1, 50.0, 10.0, 0.5, key="interp_s_E")
        with col7: s_tau = st.slider("s_τ", 0.1, 20.0, 5.0, 0.5, key="interp_s_tau")
        with col8: s_t = st.slider("s_t", 1.0, 50.0, 20.0, 1.0, key="interp_s_t")
        temporal_weight = st.slider("Temporal Weight", 0.0, 1.0, 0.3, 0.05, key="interp_temporal_weight")
    st.session_state.extrapolator.sigma_param = sigma_param; st.session_state.extrapolator.spatial_weight = spatial_weight
    st.session_state.extrapolator.n_heads = n_heads; st.session_state.extrapolator.temperature = temperature
    st.session_state.extrapolator.sigma_g = sigma_g; st.session_state.extrapolator.s_E = s_E
    st.session_state.extrapolator.s_tau = s_tau; st.session_state.extrapolator.s_t = s_t; st.session_state.extrapolator.temporal_weight = temporal_weight
    with st.expander("🔥 Heat Transfer Physics Parameters", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1: td = st.number_input("Diffusivity (m²/s)", 1e-7, 1e-4, 1e-5, format="%.1e", key="thermal_diff"); st.session_state.extrapolator.thermal_diffusivity = td
        with col2: sr = st.number_input("Spot Radius (μm)", 1.0, 200.0, 50.0, key="spot_rad"); st.session_state.extrapolator.laser_spot_radius = sr * 1e-6
        with col3: cl = st.number_input("Char. Length (μm)", 10.0, 500.0, 100.0, key="char_len"); st.session_state.extrapolator.characteristic_length = cl * 1e-6
    with st.expander("🖼️ 3D Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1: enable_3d = st.checkbox("Enable 3D", True); optimize = st.checkbox("Optimize", True)
        with col2:
            top_k = st.slider("Top-K", 3, 20, 10) if optimize else None
            subsample = st.slider("Subsample", 1, 10, 1) if optimize else None
    if st.button("🚀 Run ST-DGPA Prediction", type="primary", use_container_width=True):
        with st.spinner("Running ST-DGPA prediction..."):
            CacheManager.clear_3d_cache(); st.session_state.last_prediction_id = None
            res = st.session_state.extrapolator.predict_time_series(energy_query, duration_query, time_points)
            if res and res['field_predictions']:
                st.session_state.interpolation_results = res
                st.session_state.interpolation_params = {'energy_query': energy_query, 'duration_query': duration_query, 'time_points': time_points, 'sigma_param': sigma_param, 'spatial_weight': spatial_weight, 'n_heads': n_heads, 'temperature': temperature, 'sigma_g': sigma_g, 's_E': s_E, 's_tau': s_tau, 's_t': s_t, 'temporal_weight': temporal_weight, 'top_k': top_k, 'subsample_factor': subsample}
                st.markdown("""<div class="success-box"><h3>✅ ST-DGPA Prediction Successful</h3><p>Heat transfer characterized & cache initialized.</p></div>""", unsafe_allow_html=True)
                tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📈 Predictions", "🧠 ST-DGPA Analysis", "⏱️ Temporal", "🌐 3D Analysis", "📊 Details", "🖼️ 3D Rendering", "⚙️ Params"])
                with tab1: render_prediction_results(res, time_points, energy_query, duration_query)
                with tab2: render_stdgpa_attention_visualization(res, energy_query, duration_query, time_points)
                with tab3: render_temporal_analysis(res, time_points, energy_query, duration_query)
                with tab4: render_3d_analysis(res, time_points, energy_query, duration_query)
                with tab5: render_detailed_results(res, time_points, energy_query, duration_query)
                with tab6: render_3d_interpolation(res, time_points, energy_query, duration_query, enable_3d, optimize, top_k)
                with tab7: render_parameter_analysis(res, energy_query, duration_query, time_points)
            else: st.error("Prediction failed.")
    elif st.session_state.interpolation_results is not None:
        params = st.session_state.interpolation_params or {}; energy_query = params.get('energy_query', 0); duration_query = params.get('duration_query', 0); time_points = params.get('time_points', []); res = st.session_state.interpolation_results
        st.markdown(f"""<div class="info-box"><h3>📊 Previous ST-DGPA Results</h3><p>ID: {st.session_state.last_prediction_id or 'N/A'}</p></div>""", unsafe_allow_html=True)
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📈 Predictions", "🧠 ST-DGPA Analysis", "⏱️ Temporal", "🌐 3D Analysis", "📊 Details", "🖼️ 3D Rendering", "⚙️ Params"])
        with tab1: render_prediction_results(res, time_points, energy_query, duration_query)
        with tab2: render_stdgpa_attention_visualization(res, energy_query, duration_query, time_points)
        with tab3: render_temporal_analysis(res, time_points, energy_query, duration_query)
        with tab4: render_3d_analysis(res, time_points, energy_query, duration_query)
        with tab5: render_detailed_results(res, time_points, energy_query, duration_query)
        with tab6: render_3d_interpolation(res, time_points, energy_query, duration_query, True, params.get('top_k') is not None, params.get('top_k'))
        with tab7: render_parameter_analysis(res, energy_query, duration_query, time_points)

@st.cache_data(show_spinner=False)
def compute_interpolated_field_cached(_extrapolator, field_name, attention_weights, source_metadata, simulations, params):
    return _extrapolator.interpolate_full_field(field_name, attention_weights, source_metadata, simulations)

def render_stdgpa_attention_visualization(results, energy_query, duration_query, time_points):
    if not results.get('physics_attention_maps') or len(results['physics_attention_maps'][0]) == 0: st.info("No ST-DGPA attention data available."); return
    st.markdown('<h4 class="sub-header">🧠 ST-DGPA Attention Analysis</h4>', unsafe_allow_html=True)
    st.markdown('<h5 class="sub-header">🎨 Visualization Controls</h5>', unsafe_allow_html=True)
    viz_col1, viz_col2, viz_col3, viz_col4 = st.columns(4)
    with viz_col1: viz_backend = st.selectbox("Backend", ["Plotly (Interactive)", "Matplotlib (Static)", "Both"], index=0, key="viz_backend_stdgpa")
    with viz_col2:
        cmap_options = ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'Turbo', 'Rainbow', 'Jet', 'Hot', 'Cool', 'Spring', 'Summer', 'Autumn', 'Winter', 'Gray', 'Bone', 'Copper', 'Pink', 'RdYlGn', 'Spectral', 'Twilight']
        selected_colormap = st.selectbox("Colormap", cmap_options, index=cmap_options.index('Viridis'), key="attention_colormap")
    with viz_col3: font_size = st.slider("Font Size", 8, 18, 12, 1, key="attention_font_size")
    with viz_col4: fig_width = st.slider("Width (px)", 400, 1400, 1250, 50, key="attention_fig_width")
    selected_timestep_idx = st.slider("Select Timestep", 0, len(time_points)-1, len(time_points)//2, key="stdgpa_timestep")
    final_weights = results['attention_maps'][selected_timestep_idx]
    physics_attention = results['physics_attention_maps'][selected_timestep_idx]
    ett_gating = results['ett_gating_maps'][selected_timestep_idx]
    selected_time = time_points[selected_timestep_idx]
    st.caption(f"🎯 Query: E = {energy_query:.1f} mJ, τ = {duration_query:.1f} ns, t = {selected_time:.1f} ns")
    plotly_cmap_map = {'Viridis':'Viridis','Plasma':'Plasma','Inferno':'Inferno','Magma':'Magma','Cividis':'Cividis','Turbo':'Turbo','Rainbow':'Rainbow','Jet':'Jet','Hot':'Hot','Cool':'Portland','Spring':'Aggrnyl','Summer':'Sunset','Autumn':'Orange','Winter':'Bluered','Gray':'Greys','Bone':'Earth','Copper':'Copper','Pink':'Pinkyl','RdYlGn':'RdYlGn','Spectral':'Spectral','Twilight':'Twilight'}
    plotly_cmap = plotly_cmap_map.get(selected_colormap, 'Viridis')
    if viz_backend in ["Plotly (Interactive)", "Both"]:
        st.markdown(f"###### 🔹 Plotly Interactive View ({selected_colormap})")
        from plotly.subplots import make_subplots; import plotly.graph_objects as go
        fig = make_subplots(rows=3, cols=3, subplot_titles=["ST-DGPA Final Weights", "Physics Attention Only", "(E, τ, t) Gating Only", "ST-DGPA vs Physics Attention", "Temporal Coherence", "Heat Transfer Phase", "Parameter Space 3D", "Attention Network", "Weight Evolution"], vertical_spacing=0.18, horizontal_spacing=0.15, specs=[[{'type':'xy'},{'type':'xy'},{'type':'xy'}], [{'type':'xy'},{'type':'xy'},{'type':'polar'}], [{'type':'scene'},{'type':'xy'},{'type':'xy'}]])
        def update_stdgpa_axes(fig, row, col, x_title, y_title):
            fig.update_xaxes(title_text=x_title, row=row, col=col, title_font=dict(size=font_size, family="Arial", color="#2c3e50"), tickfont=dict(size=font_size-1, color="#2c3e50"), automargin=True)
            fig.update_yaxes(title_text=y_title, row=row, col=col, title_font=dict(size=font_size, family="Arial", color="#2c3e50"), tickfont=dict(size=font_size-1, color="#2c3e50"), automargin=True)
        fig.add_trace(go.Bar(x=list(range(len(final_weights))), y=final_weights, marker=dict(color='rgba(52,152,219,0.85)'), showlegend=False), row=1, col=1); update_stdgpa_axes(fig,1,1,"Source Index","Weight")
        fig.add_trace(go.Bar(x=list(range(len(physics_attention))), y=physics_attention, marker=dict(color='rgba(46,204,113,0.85)'), showlegend=False), row=1, col=2); update_stdgpa_axes(fig,1,2,"Source Index","Weight")
        fig.add_trace(go.Bar(x=list(range(len(ett_gating))), y=ett_gating, marker=dict(color='rgba(231,76,60,0.85)'), showlegend=False), row=1, col=3); update_stdgpa_axes(fig,1,3,"Source Index","Weight")
        fig.add_trace(go.Scatter(x=list(range(len(final_weights))), y=final_weights, mode='lines+markers', line=dict(color='#3498db',width=3), hovertemplate='ST-DGPA<br>Source: %{x}<br>Weight: %{y:.4f}<extra></extra>'), row=2, col=1)
        fig.add_trace(go.Scatter(x=list(range(len(physics_attention))), y=physics_attention, mode='lines+markers', line=dict(color='#2ecc71',width=2,dash='dash'), hovertemplate='Physics<br>Source: %{x}<br>Weight: %{y:.4f}<extra></extra>'), row=2, col=1); update_stdgpa_axes(fig,2,1,"Source Index","Weight")
        if st.session_state.get('summaries') and hasattr(st.session_state.extrapolator, 'source_metadata'):
            times, weights = [], []
            for i, w in enumerate(final_weights):
                if w > 0.01 and i < len(st.session_state.extrapolator.source_metadata): times.append(st.session_state.extrapolator.source_metadata[i]['time']); weights.append(w)
            if times and weights:
                fig.add_trace(go.Scatter(x=times, y=weights, mode='markers', marker=dict(size=np.array(weights)*60, color=weights, colorscale=plotly_cmap, showscale=False)), row=2, col=2); fig.add_vline(x=selected_time, line_dash="dash", line_color="#e74c3c", row=2, col=2); update_stdgpa_axes(fig,2,2,"Time (ns)","Weight")
        if 'heat_transfer_indicators' in results and results['heat_transfer_indicators']:
            indicators = results['heat_transfer_indicators'][selected_timestep_idx]
            if indicators:
                phase = indicators.get('phase','Unknown'); val_map = {'Early Heating':[0.9,0.3,0.2,0.1], 'Heating':[0.9,0.3,0.2,0.1], 'Early Cooling':[0.4,0.8,0.3,0.1], 'Diffusion Cooling':[0.2,0.5,0.9,0.2]}; values = val_map.get(phase, [0.7,0.5,0.3,0.2])
                fig.add_trace(go.Scatterpolar(r=list(values)+[values[0]], theta=['Heating','Cooling','Diffusion','Adiabatic']+['Heating'], fill='toself', fillcolor='rgba(243,156,18,0.35)', line=dict(color='#f39c12',width=3), showlegend=False), row=2, col=3)
                fig.update_polars(radialaxis=dict(visible=True, range=[0,1.1], tickfont=dict(size=font_size-1, color='#2c3e50'), gridcolor='lightgray'), angularaxis=dict(direction="clockwise", tickfont=dict(size=font_size, color='#2c3e50')), bgcolor='white', row=2, col=3)
        if st.session_state.get('summaries'):
            energies, durations, times_3d, weights_3d = [], [], [], []
            for summary in st.session_state.summaries[:10]:
                for t in summary['timesteps'][:5]: energies.append(summary['energy']); durations.append(summary['duration']); times_3d.append(t); weights_3d.append(np.mean(final_weights) if len(final_weights)>0 else 0.1)
            fig.add_trace(go.Scatter3d(x=energies, y=durations, z=times_3d, mode='markers', marker=dict(size=np.array(weights_3d)*25, color=weights_3d, colorscale=plotly_cmap, colorbar=dict(title="Weight", title_font=dict(size=font_size-1, family="Arial"), tickfont=dict(size=font_size-2, color='#2c3e50'), thickness=20, len=0.6, xpad=15), showscale=True, line=dict(width=0.5, color='white')), showlegend=False), row=3, col=1)
            fig.add_trace(go.Scatter3d(x=[energy_query], y=[duration_query], z=[selected_time], mode='markers', marker=dict(size=14, color='red', symbol='diamond', line=dict(width=2, color='black')), showlegend=False), row=3, col=1)
            fig.update_scenes(xaxis_title="Energy (mJ)", yaxis_title="Duration (ns)", zaxis_title="Time (ns)", camera=dict(eye=dict(x=1.5,y=1.5,z=1.5)), row=3, col=1)
        if len(final_weights) > 5:
            top_indices = np.argsort(final_weights)[-5:]; top_weights = final_weights[top_indices]
            fig.add_trace(go.Scatter(x=[0]+list(range(1,6)), y=[0]*6, mode='markers+text', text=['Query']+[f'S{i+1}' for i in top_indices], textposition="top center", marker=dict(size=[26]+list(top_weights*45), color=['#e74c3c']+['#3498db']*5, line=dict(width=2, color='white')), showlegend=False), row=3, col=2)
            for i in range(1, 6): fig.add_trace(go.Scatter(x=[0,i], y=[0,0], mode='lines', line=dict(width=top_weights[i-1]*12, color='#95a5a6'), showlegend=False), row=3, col=2)
            fig.update_xaxes(title_text="Node", row=3, col=2, showticklabels=False, showgrid=False); fig.update_yaxes(title_text="", row=3, col=2, showticklabels=False, showgrid=False)
        if len(results['attention_maps']) > 1:
            top_idx = np.argmax(final_weights); weight_evolution = [results['attention_maps'][t_idx][top_idx] for t_idx in range(len(results['attention_maps'])) if top_idx < len(results['attention_maps'][t_idx])]
            if weight_evolution: fig.add_trace(go.Scatter(x=time_points[:len(weight_evolution)], y=weight_evolution, mode='lines+markers', line=dict(color='#8e44ad',width=3), showlegend=False), row=3, col=3); update_stdgpa_axes(fig,3,3,"Time (ns)","Weight")
        fig.update_layout(height=1200, width=fig_width, title=dict(text=f"ST-DGPA Analysis at t={selected_time} ns (E={energy_query:.1f} mJ, τ={duration_query:.1f} ns)", font=dict(size=font_size+2, family="Arial", color='#2c3e50'), x=0.5, xanchor='center', y=0.98, pad=dict(t=20,b=10)), plot_bgcolor='white', paper_bgcolor='white', font=dict(family="Arial, sans-serif", size=font_size, color="#2c3e50"), margin=dict(l=90,r=80,t=140,b=80), showlegend=True, legend=dict(yanchor="top", y=1.12, xanchor="center", x=0.5, orientation="h", bgcolor='rgba(255,255,255,0.95)', bordercolor='#2c3e50', borderwidth=1, font=dict(size=font_size-1)), barmode='group', bargap=0.15, hoverlabel=dict(bgcolor='white', font_size=font_size-1, font_family="Arial", bordercolor='#2c3e50'))
        fig.update_annotations(font=dict(size=font_size, family="Arial", color="#2c3e50"))
        st.plotly_chart(fig, use_container_width=False)
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            if st.button("📥 Export HTML", key="export_plotly_html_attention"):
                html_str = fig.to_html(include_plotlyjs='cdn', full_html=True)
                st.download_button(label="⬇️ HTML", data=html_str, file_name=f"stdgpa_attention_E{energy_query:.1f}_tau{duration_query:.1f}_t{selected_time:.1f}.html", mime="text/html")
        with col_exp2:
            if st.button("📥 Export PNG", key="export_plotly_png_attention"):
                try:
                    img_bytes = fig.to_image(format="png", width=fig_width, height=1200, scale=2)
                    st.download_button(label="⬇️ PNG", data=img_bytes, file_name=f"stdgpa_attention_E{energy_query:.1f}_tau{duration_query:.1f}_t{selected_time:.1f}.png", mime="image/png")
                except Exception as e: st.error(f"Requires kaleido: {e}")
    if viz_backend in ["Matplotlib (Static)", "Both"]:
        st.markdown(f"###### 🔹 Matplotlib Static View ({selected_colormap})")
        import matplotlib.pyplot as plt; from io import BytesIO
        mpl_cmap = selected_colormap.lower()
        try: plt.get_cmap(mpl_cmap)
        except ValueError: mpl_cmap = 'viridis'
        fig_mpl, axes = plt.subplots(3, 3, figsize=(fig_width/100, 11.5), dpi=100, facecolor='white')
        fig_mpl.suptitle(f"ST-DGPA Analysis at t={selected_time} ns", fontsize=font_size+2, fontweight='bold', y=0.998, color='#2c3e50')
        ax = axes.flatten()
        ax[0].bar(range(len(final_weights)), final_weights, color='#3498db', alpha=0.85); ax[0].set_title("ST-DGPA Weights"); ax[1].bar(range(len(physics_attention)), physics_attention, color='#2ecc71', alpha=0.85); ax[1].set_title("Physics Attention"); ax[2].bar(range(len(ett_gating)), ett_gating, color='#e74c3c', alpha=0.85); ax[2].set_title("(E, τ, t) Gating")
        ax[3].plot(range(len(final_weights)), final_weights, 'o-', color='#3498db', linewidth=3, label='ST-DGPA'); ax[3].plot(range(len(physics_attention)), physics_attention, 's--', color='#2ecc71', linewidth=2, label='Physics'); ax[3].set_title("Comparison"); ax[3].legend(fontsize=font_size-1)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96]); st.pyplot(fig_mpl, bbox_inches='tight', dpi=100); plt.close(fig_mpl)

def render_temporal_analysis(results, time_points, energy_query, duration_query):
    if not results or 'heat_transfer_indicators' not in results: st.info("No temporal data."); return
    st.markdown('<h4 class="sub-header">⏱️ Temporal Analysis</h4>', unsafe_allow_html=True)
    fig = st.session_state.visualizer.create_temporal_analysis(results, time_points)
    if fig: st.plotly_chart(fig, use_container_width=True)
    if results['heat_transfer_indicators']:
        phases = [i.get('phase','Unknown') for i in results['heat_transfer_indicators'] if i]
        phase_counts = {p: phases.count(p) for p in ['Early Heating','Heating','Early Cooling','Diffusion Cooling']}
        col1, col2, col3, col4 = st.columns(4)
        phases_list = list(phase_counts.keys()); colors = ['#FF6B6B','#4ECDC4','#45B7D1','#96CEB4']
        for idx, phase in enumerate(phases_list):
            count = phase_counts.get(phase,0); pct = (count/len(phases)*100) if phases else 0
            with [col1,col2,col3,col4][idx%4]: st.metric(phase, f"{pct:.1f}%", delta=f"{count} timesteps" if count>0 else None)
    if 'temporal_confidences' in results:
        avg, mn, mx = np.mean(results['temporal_confidences']), np.min(results['temporal_confidences']), np.max(results['temporal_confidences'])
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Avg Temporal Confidence", f"{avg:.3f}")
        with col2: st.metric("Min", f"{mn:.3f}")
        with col3: st.metric("Max", f"{mx:.3f}")
        if avg < 0.5: st.warning("⚠️ Low Confidence")
        elif avg < 0.7: st.info("ℹ️ Moderate Confidence")
        else: st.success("✅ High Confidence")

def render_prediction_results(results, time_points, energy_query, duration_query):
    available_fields = list(results['field_predictions'].keys())
    if not available_fields: st.warning("No predictions."); return
    n_fields = min(len(available_fields), 4)
    fig = make_subplots(rows=n_fields, cols=1, subplot_titles=[f"Predicted {f}" for f in available_fields[:n_fields]], vertical_spacing=0.1, shared_xaxes=True)
    for idx, field in enumerate(available_fields[:n_fields]):
        row = idx + 1
        if results['field_predictions'][field]['mean']: fig.add_trace(go.Scatter(x=time_points, y=results['field_predictions'][field]['mean'], mode='lines+markers', name=f'{field} (mean)', line=dict(width=3, color='blue')), row=row, col=1)
        if results['field_predictions'][field]['mean'] and results['field_predictions'][field]['std']:
            mu = np.array(results['field_predictions'][field]['mean']); std = np.array(results['field_predictions'][field]['std'])
            fig.add_trace(go.Scatter(x=np.concatenate([time_points, time_points[::-1]]), y=np.concatenate([mu+std, (mu-std)[::-1]]), fill='toself', fillcolor='rgba(0,0,255,0.2)', line=dict(color='rgba(255,255,255,0)'), showlegend=False), row=row, col=1)
    fig.update_layout(height=300*n_fields, title=f"Field Predictions (E={energy_query:.1f} mJ, τ={duration_query:.1f} ns)", showlegend=True, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    if results['confidence_scores']:
        fig_conf = go.Figure()
        fig_conf.add_trace(go.Scatter(x=time_points, y=results['confidence_scores'], mode='lines+markers', line=dict(color='orange',width=3), fill='tozeroy', fillcolor='rgba(255,165,0,0.2)', name='Confidence'))
        if 'temporal_confidences' in results: fig_conf.add_trace(go.Scatter(x=time_points, y=results['temporal_confidences'], mode='lines+markers', line=dict(color='green',width=3,dash='dash'), name='Temporal'))
        fig_conf.update_layout(title="Prediction Confidence", xaxis_title="Time", yaxis_title="Confidence", height=400, yaxis_range=[0,1])
        st.plotly_chart(fig_conf, use_container_width=True)
        avg, mn, mx = np.mean(results['confidence_scores']), np.min(results['confidence_scores']), np.max(results['confidence_scores'])
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Avg Confidence", f"{avg:.3f}")
        with col2: st.metric("Min", f"{mn:.3f}")
        with col3: st.metric("Max", f"{mx:.3f}")
        if avg < 0.3: st.warning("⚠️ Low Confidence")
        elif avg < 0.6: st.info("ℹ️ Moderate")
        else: st.success("✅ High")

def render_3d_analysis(results, time_points, energy_query, duration_query):
    st.markdown('<h4 class="sub-header">🌐 3D Parameter Space</h4>', unsafe_allow_html=True)
    if st.session_state.summaries:
        train_e = [s['energy'] for s in st.session_state.summaries]; train_d = [s['duration'] for s in st.session_state.summaries]
        train_t = [np.max(s['field_stats']['temperature']['max']) if 'temperature' in s['field_stats'] and s['field_stats']['temperature']['max'] else 0 for s in st.session_state.summaries]
        train_s = [np.max(s['field_stats']['principal stress']['max']) if 'principal stress' in s['field_stats'] and s['field_stats']['principal stress']['max'] else 0 for s in st.session_state.summaries]
        query_t = np.max(results['field_predictions'].get('temperature', {}).get('max', [0])); query_s = np.max(results['field_predictions'].get('principal stress', {}).get('max', [0]))
        col1, col2 = st.columns(2)
        with col1:
            fig_t = go.Figure()
            fig_t.add_trace(go.Scatter3d(x=train_e, y=train_d, z=train_t, mode='markers', marker=dict(size=8, color=train_t, colorscale='Viridis', opacity=0.7, colorbar=dict(title="Max Temp")), name='Training'))
            fig_t.add_trace(go.Scatter3d(x=[energy_query], y=[duration_query], z=[query_t], mode='markers', marker=dict(size=12, color='red', symbol='diamond'), name='Query'))
            fig_t.update_layout(title="Max Temperature Space", scene=dict(xaxis_title="Energy", yaxis_title="Duration", zaxis_title="Max Temp"), height=500)
            st.plotly_chart(fig_t, use_container_width=True)
        with col2:
            fig_s = go.Figure()
            fig_s.add_trace(go.Scatter3d(x=train_e, y=train_d, z=train_s, mode='markers', marker=dict(size=8, color=train_s, colorscale='Plasma', opacity=0.7, colorbar=dict(title="Max Stress")), name='Training'))
            fig_s.add_trace(go.Scatter3d(x=[energy_query], y=[duration_query], z=[query_s], mode='markers', marker=dict(size=12, color='red', symbol='diamond'), name='Query'))
            fig_s.update_layout(title="Max Stress Space", scene=dict(xaxis_title="Energy", yaxis_title="Duration", zaxis_title="Max Stress"), height=500)
            st.plotly_chart(fig_s, use_container_width=True)

def render_detailed_results(results, time_points, energy_query, duration_query):
    st.markdown('<h4 class="sub-header">📊 Detailed Results</h4>', unsafe_allow_html=True)
    data_rows = []
    for idx, t in enumerate(time_points):
        row = {'Time (ns)': t}
        for field in results['field_predictions']:
            if idx < len(results['field_predictions'][field]['mean']): row[f'{field}_mean'] = results['field_predictions'][field]['mean'][idx]; row[f'{field}_max'] = results['field_predictions'][field]['max'][idx]; row[f'{field}_std'] = results['field_predictions'][field]['std'][idx]
        if idx < len(results['confidence_scores']): row['confidence'] = results['confidence_scores'][idx]
        if idx < len(results.get('temporal_confidences', [])): row['temporal_confidence'] = results['temporal_confidences'][idx]
        if idx < len(results.get('heat_transfer_indicators', [])):
            ind = results['heat_transfer_indicators'][idx]
            if ind: row['phase'] = ind.get('phase','Unknown'); row['fourier'] = ind.get('fourier_number',0); row['penetration'] = ind.get('thermal_penetration_um',0)
        data_rows.append(row)
    if data_rows:
        df = pd.DataFrame(data_rows); fmt = {col: "{:.3f}" if 'mean' in col or 'max' in col or 'std' in col or 'confidence' in col else ("{:.4f}" if 'fourier' in col else "{:.1f}") for col in df.columns}
        styled = df.style.format(fmt)
        def hl(v): return 'background-color:#ffcccc' if isinstance(v,(int,float)) and v<0.3 else ('background-color:#fff4cc' if isinstance(v,(int,float)) and v<0.6 else ('background-color:#ccffcc' if isinstance(v,(int,float)) and v>=0.6 else ''))
        for c in df.columns:
            if 'confidence' in c: styled = styled.map(hl, subset=[c])
        st.dataframe(styled, use_container_width=True, height=400)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 CSV", csv, file_name=f"stdgpa_E{energy_query:.1f}_tau{duration_query:.1f}.csv", mime="text/csv")

def render_parameter_analysis(results, energy_query, duration_query, time_points):
    st.markdown('<h4 class="sub-header">⚙️ Parameter Sensitivity</h4>', unsafe_allow_html=True)
    params = st.session_state.interpolation_params or {}
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("σ_g", f"{params.get('sigma_g', 0.20):.2f}")
    with col2: st.metric("s_E", f"{params.get('s_E', 10.0):.1f}")
    with col3: st.metric("s_τ", f"{params.get('s_tau', 5.0):.1f}")
    with col4: st.metric("s_t", f"{params.get('s_t', 20.0):.1f}")
    with st.expander("📖 Guide", expanded=True):
        st.markdown("### ST-DGPA Parameter Effects\n**σ_g:** Sharp (0.05-0.2) vs Broad (0.3-1.0)\n**s_E/s_τ/s_t:** Scaling factors for E, τ, t\n**temporal_weight:** Balances physics vs temporal similarity")
    test_col1, test_col2, test_col3, test_col4 = st.columns(4)
    with test_col1: test_sg = st.slider("Test σ_g", 0.05, 1.0, params.get('sigma_g', 0.20), 0.05, key="test_sigma_g")
    with test_col2: test_sE = st.slider("Test s_E", 0.1, 50.0, params.get('s_E', 10.0), 0.5, key="test_s_E")
    with test_col3: test_sT = st.slider("Test s_τ", 0.1, 20.0, params.get('s_tau', 5.0), 0.5, key="test_s_tau")
    with test_col4: test_st = st.slider("Test s_t", 1.0, 50.0, params.get('s_t', 20.0), 1.0, key="test_s_t")
    test_tw = st.slider("Test Temporal Weight", 0.0, 1.0, params.get('temporal_weight', 0.3), 0.05, key="test_tw")
    if st.button("🔄 Test Parameters", key="test_params"):
        orig = {'sigma_g': st.session_state.extrapolator.sigma_g, 's_E': st.session_state.extrapolator.s_E, 's_tau': st.session_state.extrapolator.s_tau, 's_t': st.session_state.extrapolator.s_t, 'temporal_weight': st.session_state.extrapolator.temporal_weight}
        st.session_state.extrapolator.sigma_g = test_sg; st.session_state.extrapolator.s_E = test_sE; st.session_state.extrapolator.s_tau = test_sT; st.session_state.extrapolator.s_t = test_st; st.session_state.extrapolator.temporal_weight = test_tw
        mid_t = time_points[len(time_points)//2]; res = st.session_state.extrapolator.predict_field_statistics(energy_query, duration_query, mid_t)
        st.session_state.extrapolator.sigma_g = orig['sigma_g']; st.session_state.extrapolator.s_E = orig['s_E']; st.session_state.extrapolator.s_tau = orig['s_tau']; st.session_state.extrapolator.s_t = orig['s_t']; st.session_state.extrapolator.temporal_weight = orig['temporal_weight']
        if res: st.info(f"Test at t={mid_t} ns: Conf={res['confidence']:.3f}, Temp={res['temporal_confidence']:.3f}, MaxW={np.max(res['attention_weights']):.4f}")

def render_3d_interpolation(results, time_points, energy_query, duration_query, enable_3d=True, optimize_performance=False, top_k=10):
    st.markdown('<h4 class="sub-header">🖼️ 3D Field Interpolation</h4>', unsafe_allow_html=True)
    if not st.session_state.get('simulations') or not next(iter(st.session_state.simulations.values())).get('has_mesh', False):
        st.error("Reload with 'Load Full Mesh'."); return
    if 'interpolation_3d_cache' in st.session_state and st.session_state.interpolation_3d_cache:
        st.markdown(f"""<div class="cache-status"><strong>Cache:</strong> {len(st.session_state.interpolation_3d_cache)} fields | <strong>ID:</strong> {st.session_state.last_prediction_id or 'N/A'}</div>""", unsafe_allow_html=True)
    params = st.session_state.interpolation_params or {}; available = list(results['field_predictions'].keys())
    if not available: st.warning("No fields."); return
    col1, col2, col3 = st.columns(3)
    with col1:
        if 'current_3d_field' not in st.session_state or st.session_state.current_3d_field not in available: st.session_state.current_3d_field = available[0]
        sel_field = st.selectbox("Field", available, index=available.index(st.session_state.current_3d_field), key="interp_3d_field"); st.session_state.current_3d_field = sel_field
    with col2:
        if 'current_3d_timestep' not in st.session_state: st.session_state.current_3d_timestep = len(time_points)//2
        ts_idx = st.slider("Timestep", 0, len(time_points)-1, st.session_state.current_3d_timestep, key="interp_3d_timestep"); st.session_state.current_3d_timestep = ts_idx
    with col3: op_3d = st.slider("Opacity", 0.0, 1.0, 0.9, 0.05, key="interp_3d_opacity")
    sel_time = time_points[ts_idx]; att_w = results['attention_maps'][ts_idx]
    conf_score = results['confidence_scores'][ts_idx]; temp_conf = results['temporal_confidences'][ts_idx] if 'temporal_confidences' in results else 0.0
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a: asp = st.selectbox("Aspect", ["data","cube","auto"], key="interp_3d_aspect")
    with col_b: cam = st.selectbox("Camera", ["Isometric","Front","Side","Top","Bottom"], key="interp_3d_camera")
    with col_c: light = st.selectbox("Lighting", ["Default","Shiny","Matte","High Contrast"], key="interp_3d_lighting")
    with col_d: theme = st.selectbox("Theme", ["Light","Dark"], key="interp_3d_theme")
    cached = CacheManager.get_cached_interpolation(sel_field, ts_idx, params) is not None
    if cached: interp_val = CacheManager.get_cached_interpolation(sel_field, ts_idx, params)['interpolated_values']; st.success("Loaded from cache")
    else:
        if optimize_performance and len(att_w) > top_k:
            top_idx = np.argsort(att_w)[-top_k:]; filt = np.zeros_like(att_w); filt[top_idx] = att_w[top_idx]; att_w = filt/np.sum(filt)
        with st.spinner("Computing..."): interp_val = st.session_state.extrapolator.interpolate_full_field(sel_field, att_w, st.session_state.extrapolator.source_metadata, st.session_state.simulations)
        if interp_val is None: st.error("Failed."); return
        CacheManager.set_cached_interpolation(sel_field, ts_idx, params, interp_val); st.info("Computed & cached")
    values = np.nan_to_num(interp_val) if interp_val.ndim==1 else np.nan_to_num(np.linalg.norm(interp_val, axis=1))
    label = sel_field if interp_val.ndim==1 else f"{sel_field} (mag)"
    data_min, data_max = float(np.min(values)), float(np.max(values))
    col_e, col_f, col_g = st.columns(3)
    with col_e: auto = st.checkbox("Auto Scale", True, key=f"auto_3d_{sel_field}_t{ts_idx}")
    with col_f: cmin = st.number_input("Min", value=data_min, format="%.3f", disabled=auto, key=f"cmin_3d_{sel_field}_t{ts_idx}")
    with col_g: cmax = st.number_input("Max", value=data_max, format="%.3f", disabled=auto, key=f"cmax_3d_{sel_field}_t{ts_idx}")
    if auto: cmin, cmax = None, None
    light_map = {"Default": dict(ambient=0.8,diffuse=0.8,specular=0.5,roughness=0.5), "Shiny": dict(ambient=0.6,diffuse=0.9,specular=0.8,roughness=0.2), "Matte": dict(ambient=0.9,diffuse=0.6,specular=0.1,roughness=0.9), "High Contrast": dict(ambient=0.4,diffuse=0.9,specular=0.7,roughness=0.4)}
    cam_map = {"Isometric": dict(eye=dict(x=1.5,y=1.5,z=1.5)), "Front": dict(eye=dict(x=0,y=2,z=0.1)), "Side": dict(eye=dict(x=2,y=0,z=0.1)), "Top": dict(eye=dict(x=0,y=0,z=2)), "Bottom": dict(eye=dict(x=0,y=0,z=-2))}
    if theme=="Dark": pbg, pbkg, gc, fc = "rgb(17,17,17)", "rgb(17,17,17)", "rgb(40,40,40)", "white"
    else: pbg, pbkg, gc, fc = "white", "white", "lightgray", "black"
    pts = next(iter(st.session_state.simulations.values()))['points']; tri = next(iter(st.session_state.simulations.values())).get('triangles')
    if tri is not None and len(tri)>0:
        valid = tri[np.all(tri<len(pts),axis=1)]
        trace = go.Mesh3d(x=pts[:,0], y=pts[:,1], z=pts[:,2], i=valid[:,0], j=valid[:,1], k=valid[:,2], intensity=values, colorscale=st.session_state.selected_colormap, cmin=cmin, cmax=cmax, opacity=op_3d, lighting=light_map[light], hovertemplate=f'<b>{label}:</b> %{{intensity:.3f}}<extra></extra>') if len(valid)>0 else go.Scatter3d(x=pts[:,0], y=pts[:,1], z=pts[:,2], mode='markers', marker=dict(size=4, color=values, colorscale=st.session_state.selected_colormap, cmin=cmin, cmax=cmax, opacity=op_3d))
    else: trace = go.Scatter3d(x=pts[:,0], y=pts[:,1], z=pts[:,2], mode='markers', marker=dict(size=4, color=values, colorscale=st.session_state.selected_colormap, cmin=cmin, cmax=cmax, opacity=op_3d))
    fig = go.Figure(trace).update_layout(title=f"Interpolated {label} at t={sel_time} ns", scene=dict(aspectmode=asp, camera=cam_map[cam], xaxis=dict(showbackground=True, backgroundcolor=pbg, gridcolor=gc, color=fc, title="X"), yaxis=dict(showbackground=True, backgroundcolor=pbg, gridcolor=gc, color=fc, title="Y"), zaxis=dict(showbackground=True, backgroundcolor=pbg, gridcolor=gc, color=fc, title="Z")), plot_bgcolor=pbg, paper_bgcolor=pbkg, height=700, margin=dict(l=0,r=0,t=50,b=0))
    for tr in fig.
        if hasattr(tr, 'colorbar') and tr.colorbar: tr.colorbar.title.font.color = fc; tr.colorbar.tickfont.color = fc
    st.plotly_chart(fig, use_container_width=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("Min", f"{np.min(values):.3f}"); with col2: st.metric("Max", f"{np.max(values):.3f}")
    with col3: st.metric("Mean", f"{np.mean(values):.3f}"); with col4: st.metric("Std", f"{np.std(values):.3f}")
    with col5: st.metric("Range", f"{np.max(values)-np.min(values):.3f}")

def render_comparative_analysis():
    st.markdown('<h2 class="sub-header">📊 Comparative Analysis</h2>', unsafe_allow_html=True)
    if not st.session_state.data_loaded: st.warning("Load simulations first."); return
    sims, sums = st.session_state.simulations, st.session_state.summaries
    col1, col2 = st.columns([3, 1])
    with col1: target = st.selectbox("Target Sim", sorted(sims.keys()), key="target_sim_select")
    with col2: n_comp = st.number_input("Comparisons", 1, 10, 5, key="n_comparisons")
    others = [s for s in sorted(sims.keys()) if s!=target]; comps = st.multiselect("Comparisons", others, default=others[:min(n_comp-1, len(others))])
    viz_sims = [target] + comps; if not viz_sims: st.info("Select at least one."); return
    fields = set()
    for s in viz_sims: fields.update(sims[s]['field_info'].keys())
    sel_field = st.selectbox("Analysis Field", sorted(fields), key="comparison_field")
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Sunburst", "🎯 Radar", "⏱️ Evolution", "🌐 3D"])
    with tab1:
        sf = st.session_state.visualizer.create_sunburst_chart(sums, sel_field, highlight_sim=target)
        if sf. st.plotly_chart(sf, use_container_width=True)
    with tab2:
        rf = st.session_state.visualizer.create_radar_chart(sums, viz_sims, target_sim=target)
        if rf. st.plotly_chart(rf, use_container_width=True)
    with tab3:
        ef = st.session_state.visualizer.create_field_evolution_comparison(sums, viz_sims, sel_field, target_sim=target)
        if ef. st.plotly_chart(ef, use_container_width=True)
    with tab4:
        st.markdown("##### 🌐 3D Parameter Space")
        energies, durations, max_vals, names, is_tgt = [], [], [], [], []
        for s in sums:
            if s['name'] in viz_sims and sel_field in s['field_stats']:
                energies.append(s['energy']); durations.append(s['duration']); names.append(s['name']); is_tgt.append(s['name']==target)
                stats = s['field_stats'][sel_field]; max_vals.append(np.max(stats['max']) if stats['max'] else 0)
        if energies:
            fig3 = go.Figure()
            for i, t in enumerate(is_tgt):
                if not t: fig3.add_trace(go.Scatter3d(x=[energies[i]], y=[durations[i]], z=[max_vals[i]], mode='markers', marker=dict(size=8, color=max_vals[i], colorscale='Viridis', opacity=0.7, colorbar=dict(title=f"Max {sel_field}")), text=names[i], hovertemplate='%{text}<extra></extra>', showlegend=i==0))
                else: fig3.add_trace(go.Scatter3d(x=[energies[i]], y=[durations[i]], z=[max_vals[i]], mode='markers', marker=dict(size=15, color='red', symbol='diamond'), text=names[i], hovertemplate='<b>%{text}</b><extra></extra>', showlegend=i==0))
            fig3.update_layout(title=f"Parameter Space - {sel_field}", scene=dict(xaxis_title="Energy", yaxis_title="Duration", zaxis_title=f"Max {sel_field}"), height=600)
            st.plotly_chart(fig3, use_container_width=True)

def render_stdgpa_analysis():
    st.markdown('<h2 class="sub-header">🔬 ST-DGPA Analysis</h2>', unsafe_allow_html=True)
    if not st.session_state.data_loaded: st.warning("Load simulations first."); return
    st.markdown("""<div class="stdgpa-box"><h3>📚 Theory</h3>ST-DGPA combines multi-head physics attention with explicit (E, τ, t) gating and heat transfer characterization.</div>""", unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">🎨 Visualization</h3>', unsafe_allow_html=True)
    col_v1, col_v2, col_v3, col_v4 = st.columns(4)
    with col_v1: vb = st.selectbox("Backend", ["Plotly","Matplotlib","Both"], key="viz_backend")
    with col_v2: sc = st.selectbox("Colormap", ['Viridis','Plasma','Inferno','Magma','Cividis'], key="kernel_colormap")
    with col_v3: fs = st.slider("Font Size", 8, 18, 12, 1, key="kernel_font")
    with col_v4: fw = st.slider("Width", 400, 1200, 800, 50, key="kernel_width")
    col1, col2, col3, col4 = st.columns(4)
    with col1: esg = st.slider("σ_g", 0.05, 1.0, 0.20, 0.05, key="explore_sg")
    with col2: esE = st.slider("s_E", 0.1, 50.0, 10.0, 0.5, key="explore_sE")
    with col3: esT = st.slider("s_τ", 0.1, 20.0, 5.0, 0.5, key="explore_sT")
    with col4: est = st.slider("s_t", 1.0, 50.0, 20.0, 1.0, key="explore_st")
    etw = st.slider("Temporal Weight", 0.0, 1.0, 0.3, 0.05, key="explore_tw")
    if vb in ["Plotly", "Both"]:
        st.markdown("###### 🔹 Plotly Interactive")
        energies = [s['energy'] for s in st.session_state.summaries]; durations = [s['duration'] for s in st.session_state.summaries]
        e_min, e_max = min(energies), max(energies); d_min, d_max = min(durations), max(durations)
        e_grid = np.linspace(e_min, e_max, 100); d_grid = np.linspace(d_min, d_max, 100)
        E, D = np.meshgrid(e_grid, d_grid); interp = st.session_state.get('interpolation_params', {})
        qe = interp.get('energy_query', (e_min+e_max)/2); qd = interp.get('duration_query', (d_min+d_max)/2)
        tp = interp.get('time_points', [10.0]); qt = tp[len(tp)//2]
        phi = ((E-qe)/esE)**2 + ((D-qd)/esT)**2; gating = np.exp(-phi/(2*esg**2))
        pcm = {'Viridis':'Viridis','Plasma':'Plasma','Inferno':'Inferno','Magma':'Magma','Cividis':'Cividis'}
        fig_k = go.Figure(go.Heatmap(z=gating, x=e_grid, y=d_grid, colorscale=pcm.get(sc,'Viridis'), colorbar=dict(title="Gating", thickness=30, len=0.75)))
        fig_k.add_trace(go.Scatter(x=energies, y=durations, mode='markers', marker=dict(size=10, color='crimson'), name='Training'))
        fig_k.add_trace(go.Scatter(x=[qe], y=[qd], mode='markers', marker=dict(size=18, color='gold', symbol='star'), name='Query'))
        fig_k.update_layout(title=f"(E, τ) Kernel at t={qt:.1f} ns", xaxis_title="Energy", yaxis_title="Duration", height=550, width=fw)
        st.plotly_chart(fig_k, use_container_width=False)
    if vb in ["Matplotlib", "Both"]:
        import matplotlib.pyplot as plt
        fig_m, ax = plt.subplots(figsize=(fw/100, 5.5), facecolor='white')
        im = ax.imshow(gating, extent=[e_min, e_max, d_min, d_max], aspect='auto', origin='lower', cmap=sc.lower(), interpolation='bilinear')
        ax.scatter(energies, durations, c='crimson', s=60, zorder=5, label='Training'); ax.scatter([qe], [qd], c='gold', s=200, marker='*', zorder=6, label='Query')
        plt.colorbar(im, ax=ax, label='Gating'); ax.set_title("Kernel"); ax.legend(); plt.tight_layout(); st.pyplot(fig_m); plt.close()

def render_heat_transfer_analysis():
    st.markdown('<h2 class="sub-header">🔥 Heat Transfer Analysis</h2>', unsafe_allow_html=True)
    if not st.session_state.data_loaded: st.warning("Load first."); return
    st.markdown("""<div class="heat-transfer-box"><h3>🌡️ Physics Concepts</h3>Fourier Number, Thermal Penetration, Temporal Phases.</div>""", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1: mat = st.selectbox("Material", ["Steel","Aluminum","Copper","Titanium","Custom"], key="ht_mat")
    props = {"Steel":{"α":1.2e-5}, "Aluminum":{"α":8.4e-5}, "Copper":{"α":1.1e-4}, "Titanium":{"α":8.9e-6}, "Custom":{"α":1e-5}}
    with col2: td = st.number_input("Diffusivity", 1e-7, 1e-3, props[mat]["α"], format="%.2e", key="ht_td"); st.session_state.extrapolator.thermal_diffusivity = td
    with col3: lr = st.number_input("Radius (μm)", 1.0, 500.0, 50.0, key="ht_lr"); st.session_state.extrapolator.laser_spot_radius = lr*1e-6; st.session_state.extrapolator.characteristic_length = lr*2e-6
    col1, col2 = st.columns(2)
    with col1: ct = st.number_input("Calc Time (ns)", 1, 500, 50, key="ht_ct")
    with col2: cd = st.number_input("Duration (ns)", 1, 100, 10, key="ht_cd")
    fn = st.session_state.extrapolator._compute_fourier_number(ct); pd = st.session_state.extrapolator._compute_thermal_penetration(ct)
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Fo", f"{fn:.4f}")
    with col2: st.metric("Pen (μm)", f"{pd:.1f}")
    with col3: st.metric("Phase", "Heating" if ct < cd else "Cooling")
    with col4: st.metric("Norm Time", f"{ct/cd:.2f}")
    time_rng = np.linspace(1, 200, 100)
    fn_evo = [st.session_state.extrapolator._compute_fourier_number(t) for t in time_rng]
    pd_evo = [st.session_state.extrapolator._compute_thermal_penetration(t) for t in time_rng]
    fig_e = make_subplots(rows=2, cols=1, subplot_titles=["Fourier Evolution", "Penetration Evolution"])
    fig_e.add_trace(go.Scatter(x=time_rng, y=fn_evo, line=dict(color='blue')), row=1, col=1)
    fig_e.add_trace(go.Scatter(x=time_rng, y=pd_evo, line=dict(color='red')), row=2, col=1)
    fig_e.update_layout(height=600, title="Heat Transfer Evolution"); st.plotly_chart(fig_e, use_container_width=True)

if __name__ == "__main__":
    main()
