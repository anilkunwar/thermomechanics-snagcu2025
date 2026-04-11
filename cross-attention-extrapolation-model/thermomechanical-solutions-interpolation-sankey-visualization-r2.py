#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ST-DGPA Laser Soldering Interpolation & Sankey Visualizer
===========================================================
Complete integrated application for:
- Loading FEA laser soldering simulations
- ST-DGPA (Spatio-Temporal Gated Physics Attention) interpolation
- Interactive Sankey diagram with full customization
- Mathematical formula explanations on hover
- User-editable colors, fonts, labels, and simulation counts
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict, Optional
import os
import glob
import re
import meshio
import tempfile
import hashlib
import json
from datetime import datetime
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler

# =============================================
# GLOBAL CONFIGURATION & PATHS
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEA_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "fea_solutions")
VISUALIZATION_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "visualization_outputs")
TEMP_ANIMATION_DIR = os.path.join(SCRIPT_DIR, "temp_animations")

os.makedirs(FEA_SOLUTIONS_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_ANIMATION_DIR, exist_ok=True)

# =============================================
# 1. CACHE MANAGEMENT UTILITIES
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
        if 'interpolation_3d_cache' in st.session_state:
            st.session_state.interpolation_3d_cache = {}
        if 'interpolation_field_history' in st.session_state:
            st.session_state.interpolation_field_history = OrderedDict()
    
    @staticmethod
    def get_cached_interpolation(field_name, timestep_idx, params):
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
            params.get('temporal_weight', 0.3), params.get('top_k'),
            params.get('subsample_factor')
        )
        return st.session_state.interpolation_3d_cache.get(cache_key)
    
    @staticmethod
    def set_cached_interpolation(field_name, timestep_idx, params, interpolated_values):
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
            params.get('temporal_weight', 0.3), params.get('top_k'),
            params.get('subsample_factor')
        )
        st.session_state.interpolation_3d_cache[cache_key] = {
            'interpolated_values': interpolated_values,
            'timestamp': datetime.now().timestamp(),
            'field_name': field_name, 'timestep_idx': timestep_idx,
            'params': {k: v for k, v in params.items() if k not in ['simulations', 'summaries']}
        }
        if 'interpolation_field_history' not in st.session_state:
            st.session_state.interpolation_field_history = OrderedDict()
        st.session_state.interpolation_field_history[f"{field_name}_{timestep_idx}"] = cache_key
        if len(st.session_state.interpolation_3d_cache) > 20:
            oldest = min(st.session_state.interpolation_3d_cache, 
                        key=lambda k: st.session_state.interpolation_3d_cache[k]['timestamp'])
            del st.session_state.interpolation_3d_cache[oldest]
        if len(st.session_state.interpolation_field_history) > 10:
            st.session_state.interpolation_field_history.popitem(last=False)

# =============================================
# 2. UNIFIED FEA DATA LOADER
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
        for idx, folder in enumerate(folders):
            name = os.path.basename(folder)
            energy, duration = _self.parse_folder_name(name)
            if energy is None: continue
            
            vtu_files = sorted(glob.glob(os.path.join(folder, "a_t????.vtu")))
            if not vtu_files: continue
            
            try:
                mesh0 = meshio.read(vtu_files[0])
                if not mesh0.point_data: continue
                
                sim_data = {
                    'name': name, 'energy_mJ': energy, 'duration_ns': duration,
                    'n_timesteps': len(vtu_files), 'vtu_files': vtu_files, 'field_info': {}, 'has_mesh': False
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
                                if key in mesh.point_data: fields[key][t] = mesh.point_data[key].astype(np.float32)
                        except Exception: pass
                    
                    sim_data.update({'points': points, 'fields': fields, 'triangles': triangles, 'has_mesh': True})
                
                summary = _self.extract_summary_statistics(vtu_files, energy, duration, name)
                summaries.append(summary)
                simulations[name] = sim_data
            except Exception: continue
            progress_bar.progress((idx + 1) / len(folders))
        
        progress_bar.empty()
        if simulations:
            st.success(f"✅ Loaded {len(simulations)} simulations with {len(_self.available_fields)} unique fields")
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
                        clean = data[~np.isnan(data)]
                    else:
                        clean = np.linalg.norm(data, axis=1)
                        clean = clean[~np.isnan(clean)]
                    
                    if clean.size > 0:
                        for stat in ['min', 'max', 'mean', 'std']:
                            summary['field_stats'][field_name][stat].append(float(eval(f"np.{stat}(clean)")))
                    else:
                        for stat in ['min', 'max', 'mean', 'std']:
                            summary['field_stats'][field_name][stat].append(0.0)
            except Exception: continue
        return summary

# =============================================
# 3. ST-DGPA INTERPOLATOR
# =============================================
class SpatioTemporalGatedPhysicsAttentionExtrapolator:
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
        self.characteristic_length = 100e-6
        self.source_embeddings = []
        self.source_values = []
        self.source_metadata = []
        self.embedding_scaler = StandardScaler()
        self.value_scaler = StandardScaler()
        self.fitted = False
    
    def load_summaries(self, summaries):
        self.source_db = summaries
        if not summaries: return
        
        all_embeddings, all_values, metadata = [], [], []
        for idx, summary in enumerate(summaries):
            for t_idx, t in enumerate(summary['timesteps']):
                emb = self._compute_enhanced_physics_embedding(summary['energy'], summary['duration'], t)
                all_embeddings.append(emb)
                field_vals = []
                for field in sorted(summary['field_stats'].keys()):
                    stats = summary['field_stats'][field]
                    if t_idx < len(stats.get('mean', [])):
                        field_vals.extend([stats['mean'][t_idx], stats['max'][t_idx], stats['std'][t_idx]])
                    else: field_vals.extend([0.0, 0.0, 0.0])
                all_values.append(field_vals)
                metadata.append({
                    'summary_idx': idx, 'timestep_idx': t_idx,
                    'energy': summary['energy'], 'duration': summary['duration'], 'time': t,
                    'name': summary['name'],
                    'fourier_number': self._compute_fourier_number(t),
                    'thermal_penetration': self._compute_thermal_penetration(t)
                })
        
        if all_embeddings:
            all_embeddings, all_values = np.array(all_embeddings), np.array(all_values)
            self.embedding_scaler.fit(all_embeddings)
            self.source_embeddings = self.embedding_scaler.transform(all_embeddings)
            self.value_scaler.fit(all_values)
            self.source_values = all_values
            self.source_metadata = metadata
            self.fitted = True
            st.info(f"✅ Prepared {len(all_embeddings)} embeddings with {all_embeddings.shape[1]} features")
    
    def _compute_fourier_number(self, time_ns): return self.thermal_diffusivity * (time_ns * 1e-9) / (self.characteristic_length ** 2)
    def _compute_thermal_penetration(self, time_ns): return np.sqrt(self.thermal_diffusivity * (time_ns * 1e-9)) * 1e6
    
    def _compute_enhanced_physics_embedding(self, energy, duration, time):
        logE = np.log1p(energy)
        power = energy / max(duration, 1e-6)
        time_ratio = time / max(duration, 1e-3)
        fo = self._compute_fourier_number(time)
        pen = self._compute_thermal_penetration(time)
        return np.array([logE, duration, time, power, time_ratio, fo, pen,
                        1.0 if time < duration else 0.0, 1.0 if time >= duration else 0.0,
                        np.log1p(power), np.log1p(time), np.sqrt(time)], dtype=np.float32)
    
    def _compute_ett_gating(self, energy_query, duration_query, time_query):
        phi_sq = []
        for meta in self.source_metadata:
            de = (energy_query - meta['energy']) / self.s_E
            dt = (duration_query - meta['duration']) / self.s_tau
            dtime = (time_query - meta['time']) / self.s_t
            if self.temporal_weight > 0:
                time_scale = 1.0 + 0.5 * (time_query / max(duration_query, 1e-6))
                dtime *= time_scale
            phi_sq.append(de**2 + dt**2 + dtime**2)
        phi_sq = np.array(phi_sq)
        gating = np.exp(-phi_sq / (2 * self.sigma_g**2))
        gs = gating.sum()
        return gating / gs if gs > 0 else np.ones_like(gating) / len(gating)
    
    def _compute_temporal_similarity(self, query_meta, source_metas):
        sims = []
        for meta in source_metas:
            t_diff = abs(query_meta['time'] - meta['time'])
            tol = max(query_meta['duration'] * 0.1, 1.0) if query_meta['time'] < query_meta['duration'] * 1.5 else max(query_meta['duration'] * 0.3, 3.0)
            fo_sim = np.exp(-abs(query_meta.get('fourier_number', 0) - meta.get('fourier_number', 0)) / 0.1) if 'fourier_number' in meta else 1.0
            t_sim = np.exp(-t_diff / tol)
            sims.append((1 - self.temporal_weight) * t_sim + self.temporal_weight * fo_sim)
        return np.array(sims)
    
    def _compute_spatial_similarity(self, query_meta, source_metas):
        sims = []
        for meta in source_metas:
            e_diff = abs(query_meta['energy'] - meta['energy']) / 50.0
            d_diff = abs(query_meta['duration'] - meta['duration']) / 20.0
            sims.append(np.exp(-np.sqrt(e_diff**2 + d_diff**2) / self.sigma_param))
        return np.array(sims)
    
    def predict_field_statistics(self, energy_query, duration_query, time_query):
        if not self.fitted: return None
        q_emb = self.embedding_scaler.transform([self._compute_enhanced_physics_embedding(energy_query, duration_query, time_query)])[0]
        q_meta = {'energy': energy_query, 'duration': duration_query, 'time': time_query,
                 'fourier_number': self._compute_fourier_number(time_query),
                 'thermal_penetration': self._compute_thermal_penetration(time_query)}
        
        # Physics attention
        n = len(self.source_embeddings)
        head_w = np.zeros((self.n_heads, n))
        for h in range(self.n_heads):
            np.random.seed(42 + h)
            P = np.random.randn(q_emb.shape[0], min(8, q_emb.shape[0]))
            qp, sp = q_emb @ P, self.source_embeddings @ P
            scores = np.exp(-np.linalg.norm(qp - sp, axis=1)**2 / (2 * self.sigma_param**2))
            if self.spatial_weight > 0:
                scores = (1 - self.spatial_weight) * scores + self.spatial_weight * self._compute_spatial_similarity(q_meta, self.source_metadata)
            if self.temporal_weight > 0:
                scores = (1 - self.temporal_weight) * scores + self.temporal_weight * self._compute_temporal_similarity(q_meta, self.source_metadata)
            head_w[h] = scores
        
        avg_w = np.mean(head_w, axis=0)
        if self.temperature != 1.0: avg_w = avg_w ** (1.0 / self.temperature)
        phys_att = np.exp(avg_w - np.max(avg_w))
        phys_att = phys_att / (phys_att.sum() + 1e-12)
        
        ett = self._compute_ett_gating(energy_query, duration_query, time_query)
        final = phys_att * ett
        fs = final.sum()
        final_w = final / fs if fs > 1e-12 else phys_att
        
        pred = final_w @ self.source_values if len(self.source_values) > 0 else np.zeros(1)
        
        # Confidence & indicators
        phase = "Early Heating" if time_query < duration_query * 0.3 else "Heating" if time_query < duration_query else "Early Cooling" if time_query < duration_query * 2 else "Diffusion Cooling"
        conf = 0.6 if time_query < duration_query * 0.5 else 0.8 if time_query < duration_query * 1.5 else 0.9
        
        return {
            'prediction': pred, 'attention_weights': final_w,
            'physics_attention': phys_att, 'ett_gating': ett,
            'confidence': float(np.max(final_w)), 'temporal_confidence': conf,
            'heat_transfer_indicators': {'phase': phase, 'fourier_number': self._compute_fourier_number(time_query),
                                       'thermal_penetration_um': self._compute_thermal_penetration(time_query)}
        }
    
    def interpolate_full_field(self, field_name, attention_weights, source_metadata, simulations):
        if not self.fitted or len(attention_weights) == 0: return None
        first = next(iter(simulations.values()))
        if 'fields' not in first or field_name not in first['fields']: return None
        
        shape = first['fields'][field_name].shape[1:]
        interp = np.zeros(shape if shape else (len(first['points']),), dtype=np.float32)
        tw = 0.0
        for idx, w in enumerate(attention_weights):
            if w < 1e-6: continue
            meta = source_metadata[idx]
            if meta['name'] in simulations and field_name in simulations[meta['name']]['fields']:
                interp += w * simulations[meta['name']]['fields'][field_name][meta['timestep_idx']]
                tw += w
        return interp / tw if tw > 0 else None

# =============================================
# 4. ENHANCED SANKEY VISUALIZER (FULLY CUSTOMIZABLE)
# =============================================
class SankeyVisualizer:
    def __init__(self):
        self.defaults = {
            'node_colors': {'target': '#FF6B6B', 'source': '#9966FF', 'components': ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFD93D', '#36A2EB', '#9966FF']},
            'font_family': 'Arial, sans-serif', 'font_size': 14,
            'node_thickness': 20, 'node_pad': 15,
            'width': 1000, 'height': 700
        }
    
    def create_stdgpa_sankey(self, sources_data: List[Dict], query: Dict, 
                           customization: Optional[Dict] = None) -> go.Figure:
        """Create fully customizable Sankey diagram with hover math explanations"""
        cfg = {**self.defaults, **(customization or {})}
        
        # Component labels (math on hover if enabled)
        show_math = cfg.get('show_math', True)
        if show_math:
            comp_labels = [
                'Energy Gate\nφ² = ((E*-Eᵢ)/s_E)²', 'Duration Gate\nφ² = ((τ*-τᵢ)/s_τ)²',
                'Time Gate\nφ² = ((t*-tᵢ)/s_t)²', 'Attention\nαᵢ = softmax(Q·Kᵀ/√dₖ)',
                'Refinement\nwᵢ ∝ αᵢ × gatingᵢ', 'Combined\nwᵢ = αᵢ·gatingᵢ / Σ(αⱼ·gatingⱼ)'
            ]
        else:
            comp_labels = ['Energy Gate', 'Duration Gate', 'Time Gate', 'Attention', 'Refinement', 'Combined']
        
        labels = [cfg.get('target_label', 'TARGET')]
        node_colors = [cfg['node_colors']['target']]
        
        n = len(sources_data)
        for i, row in enumerate(sources_data):
            w = row.get('Combined_Weight', 0)
            op = min(0.3 + w * 0.7, 1.0)
            base = cfg['node_colors']['source']
            r, g, b = int(base[1:3], 16), int(base[3:5], 16), int(base[5:7], 16)
            node_colors.append(f'rgba({r},{g},{b},{op:.2f})')
        
        comp_colors = cfg['node_colors']['components']
        labels.extend(comp_labels)
        node_colors.extend(comp_colors)
        
        s_idx, t_idx, vals, l_colors, h_texts = [], [], [], [], []
        comp_start = len([cfg['target_label']] + [None]*n)
        
        for i in range(n):
            row = sources_data[i]
            src = i + 1
            # Scaled values for visualization
            ve = ((row['Energy'] - query['Energy']) / 10.0)**2 * 10
            vτ = ((row['Duration'] - query['Duration']) / 5.0)**2 * 10
            vt = ((row['Time'] - query['Time']) / 20.0)**2 * 10
            va = row.get('Attention_Score', 0) * 100
            vr = row.get('Refinement', 0) * 100
            vc = row.get('Combined_Weight', 0) * 100
            
            vals_list = [ve, vτ, vt, va, vr, vc]
            for c in range(6):
                s_idx.append(src); t_idx.append(comp_start + c)
                vals.append(max(0.01, vals_list[c]))
                l_colors.append(comp_colors[c].replace('rgb', 'rgba').replace(')', ', 0.5)'))
                
                # Hover math
                if c == 0:
                    h_texts.append(f"<b>Energy Gate</b><br>S{src} | E*={query['Energy']}, Eᵢ={row['Energy']}<br>φ² = ((E*-Eᵢ)/s_E)² = {ve/10:.4f}")
                elif c == 1:
                    h_texts.append(f"<b>Duration Gate</b><br>S{src} | τ*={query['Duration']}, τᵢ={row['Duration']}<br>φ² = ((τ*-τᵢ)/s_τ)² = {vτ/10:.4f}")
                elif c == 2:
                    h_texts.append(f"<b>Time Gate</b><br>S{src} | t*={query['Time']}, tᵢ={row['Time']}<br>φ² = ((t*-tᵢ)/s_t)² = {vt/10:.4f}")
                elif c == 3:
                    h_texts.append(f"<b>Attention</b><br>S{src} | αᵢ = softmax(QKᵀ/√dₖ)<br>Score: {row.get('Attention_Score', 0):.4f}")
                elif c == 4:
                    h_texts.append(f"<b>Refinement</b><br>S{src} | wᵢ ∝ αᵢ·gatingᵢ<br>Refinement: {row.get('Refinement', 0):.4f}")
                else:
                    h_texts.append(f"<b>Combined Weight</b><br>S{src} | wᵢ = (αᵢ·gatingᵢ)/Σ(...)<br>Weight: {row.get('Combined_Weight', 0):.4f} ({row.get('Combined_Weight', 0)*100:.1f}%)")
        
        # Components to Target
        for c in range(6):
            s_idx.append(comp_start + c); t_idx.append(0)
            flow_in = sum(v for s, t, v in zip(s_idx[:-6], t_idx[:-6], vals[:-6]) if t == comp_start + c)
            vals.append(flow_in * 0.5)
            l_colors.append('rgba(153,102,255,0.6)')
            h_texts.append(f"<b>Aggregation</b><br>{comp_labels[c]} → TARGET<br>Total in: {flow_in:.3f}")
        
        fig = go.Figure(go.Sankey(
            node=dict(pad=cfg['node_pad'], thickness=cfg['node_thickness'],
                     line=dict(color="black", width=0.5), label=labels, color=node_colors,
                     font=dict(family=cfg['font_family'], size=cfg['font_size'])),
            link=dict(source=s_idx, target=t_idx, value=vals, color=l_colors,
                     hovertext=h_texts, hovertemplate='%{hovertext}<extra></extra>'),
            hoverinfo='text'
        ))
        
        fig.update_layout(
            title=f"<b>ST-DGPA Attention Flow</b><br>E={query['Energy']}, τ={query['Duration']}, t={query['Time']}",
            font=dict(family=cfg['font_family'], size=cfg['font_size']),
            width=cfg['width'], height=cfg['height'],
            hoverlabel=dict(font=dict(family=cfg['font_family'], size=cfg['font_size']),
                           bgcolor='rgba(44,62,80,0.9)', namelength=-1)
        )
        return fig

# =============================================
# 5. MAIN STREAMLIT APPLICATION
# =============================================
def main():
    st.set_page_config(page_title="ST-DGPA Sankey Visualizer", layout="wide", initial_sidebar_state="expanded")
    
    st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; text-align: center; margin-bottom: 1.5rem; font-weight: 800; }
    .stdgpa-box { background: linear-gradient(135deg, #f093fb 0%, #00f2fe 100%); color: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🔬 ST-DGPA Interpolation & Sankey Analysis</h1>', unsafe_allow_html=True)
    
    # Init session state
    for key, val in {
        'data_loader': UnifiedFEADataLoader(), 'extrapolator': SpatioTemporalGatedPhysicsAttentionExtrapolator(),
        'sankey_viz': SankeyVisualizer(), 'df_sources': None
    }.items():
        if key not in st.session_state: st.session_state[key] = val
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ ST-DGPA Parameters")
        sigma_g = st.slider("σ_g", 0.05, 1.0, 0.20, 0.05, help="Gating sharpness")
        s_E = st.slider("s_E (mJ)", 0.1, 50.0, 10.0, 0.5)
        s_tau = st.slider("s_τ (ns)", 0.1, 20.0, 5.0, 0.5)
        s_t = st.slider("s_t (ns)", 1.0, 50.0, 20.0, 1.0)
        st.session_state.extrapolator.sigma_g = sigma_g
        st.session_state.extrapolator.s_E = s_E
        st.session_state.extrapolator.s_tau = s_tau
        st.session_state.extrapolator.s_t = s_t
        
        st.markdown("---")
        st.header("🎨 Visualization Settings")
        font_family = st.selectbox("Font", ["Arial, sans-serif", "Courier New, monospace", "Times New Roman, serif"], index=0)
        font_size = st.slider("Font Size", 8, 20, 14, 1)
        node_thick = st.slider("Node Thickness", 10, 40, 20, 2)
        fig_w = st.slider("Width", 600, 1400, 1000, 50)
        fig_h = st.slider("Height", 400, 1000, 700, 50)
        show_math = st.checkbox("Show Math on Hover", True)
        
        st.markdown("**Node Colors**")
        c1, c2 = st.columns(2)
        with c1: target_color = st.color_picker("Target", "#FF6B6B")
        with c2: src_color = st.color_picker("Source", "#9966FF")
        comp_color = st.color_picker("Components", "#4ECDC4")
        
        st.markdown("---")
        st.header("📊 Data")
        n_sims = st.slider("Demo Simulations", 2, 20, 4, 1)
        if st.button("🔄 Generate Demo", use_container_width=True):
            np.random.seed(42)
            st.session_state.df_sources = pd.DataFrame({
                'Energy': np.random.uniform(0.5, 8.0, n_sims),
                'Duration': np.random.uniform(2.0, 7.0, n_sims),
                'Time': np.random.uniform(1.0, 10.0, n_sims),
                'Max_Temp': np.random.uniform(500, 1500, n_sims)
            })
            st.success(f"Generated {n_sims} simulations!")
        
        up = st.file_uploader("Upload CSV", type=["csv"])
        if up:
            try:
                st.session_state.df_sources = pd.read_csv(up)
                st.success(f"Loaded {len(st.session_state.df_sources)} rows")
            except Exception as e: st.error(str(e))
    
    # Main
    if st.session_state.df_sources is not None:
        df = st.session_state.df_sources
        st.success(f"✅ Loaded {len(df)} simulations")
        
        st.subheader("🎯 Target Query")
        c1, c2, c3 = st.columns(3)
        with c1: qE = st.number_input("Energy (mJ)", 0.1, 20.0, 3.5, 0.1)
        with c2: qτ = st.number_input("Duration (ns)", 0.1, 20.0, 4.2, 0.1)
        with c3: qt = st.number_input("Time (ns)", 0.1, 20.0, 5.0, 0.1)
        
        query = {'Energy': qE, 'Duration': qτ, 'Time': qt}
        
        if st.button("🚀 Compute ST-DGPA", type="primary", use_container_width=True):
            with st.spinner("Computing..."):
                # ST-DGPA weights
                dE = (df['Energy'] - qE) / s_E
                dτ = (df['Duration'] - qτ) / s_tau
                dt = (df['Time'] - qt) / s_t
                phi_sq = dE**2 + dτ**2 + dt**2
                gating = np.exp(-phi_sq / (2*sigma_g**2))
                attn = 1 / (1 + np.sqrt(phi_sq))
                ref = gating * attn
                df['Gating'] = gating
                df['Attention'] = attn
                df['Refinement'] = ref
                df['Combined_Weight'] = ref / (ref.sum() + 1e-12)
                
                st.markdown("### 📊 Top Sources")
                top = df.nlargest(5, 'Combined_Weight')
                st.dataframe(top.style.format({
                    'Energy': '{:.2f}', 'Duration': '{:.2f}', 'Time': '{:.2f}',
                    'Combined_Weight': '{:.4f}'
                }).highlight_max(subset=['Combined_Weight'], color='#90EE90'))
                
                # Prediction
                pred_temp = (df['Combined_Weight'] * df['Max_Temp']).sum()
                st.metric("Predicted Max Temp", f"{pred_temp:.1f} K")
                
                # Sankey
                sources = df.to_dict('records')
                cust = {
                    'font_family': font_family, 'font_size': font_size,
                    'node_thickness': node_thick, 'width': fig_w, 'height': fig_h,
                    'show_math': show_math, 'target_label': 'TARGET',
                    'node_colors': {
                        'target': target_color, 'source': src_color,
                        'components': [comp_color, '#4ECDC4', '#95E1D3', '#FFD93D', '#36A2EB', '#9966FF']
                    }
                }
                fig = st.session_state.sankey_viz.create_stdgpa_sankey(sources, query, cust)
                st.markdown("### 🕸️ Attention Weight Flow")
                st.plotly_chart(fig, use_container_width=True)
                
                # Weight distribution
                fig2 = go.Figure(go.Bar(
                    x=[f"Sim {i+1}" for i in range(len(df))],
                    y=df['Combined_Weight'],
                    marker_color=src_color, text=[f"{w*100:.1f}%" for w in df['Combined_Weight']]
                ))
                fig2.update_layout(title="Weight Distribution", xaxis_title="Source", yaxis_title="Weight")
                st.plotly_chart(fig2, use_container_width=True)
                
                # Math formulas
                with st.expander("📐 ST-DGPA Formulas"):
                    st.markdown("""
                    **Core:** `w_i = [α_i × gating_i] / Σ_j[α_j × gating_j]`
                    
                    **Gating:** `gating_i = exp(-φ_i² / (2σ_g²))`
                    `φ_i = √[((E*-E_i)/s_E)² + ((τ*-τ_i)/s_τ)² + ((t*-t_i)/s_t)²]`
                    
                    **Attention:** `α_i = softmax(Q·Kᵀ / √dₖ)`
                    """)
    else:
        st.info("👈 Generate demo data or upload CSV in sidebar to begin.")

if __name__ == "__main__":
    main()
