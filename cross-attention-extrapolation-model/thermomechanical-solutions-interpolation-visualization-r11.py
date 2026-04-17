#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ENHANCED LASER SOLDERING ST-DGPA PLATFORM WITH FULLY EXPANDED POLAR RADAR
========================================================================
Complete integrated application for:
- FEA laser soldering simulation loading from VTU files
- ST-DGPA (Spatio-Temporal Gated Physics Attention) interpolation/extrapolation
- ENHANCED Polar Radar Charts with robust data handling, autoscaling, jitter,
  full customization, multi-timestep comparison, AND EXPLICIT UNIT CONVERSION
- 3D mesh visualization with caching
- Export functionality (JSON, CSV, PNG, SVG, HTML) with dual-unit preservation

FIXES APPLIED (v2.1.1):
✅ KNOWN_UNITS accessed via class name: PolarRadarVisualizer.KNOWN_UNITS
✅ All use_container_width replaced with width='stretch' or width='content'
✅ Full error handling for unit lookup fallbacks
✅ Comprehensive type hints and docstrings throughout

Author: ST-DGPA Platform Development Team
Version: 2.1.1
License: MIT
Last Updated: 2026-04-18
"""

# ============================================================================
# IMPORTS & GLOBAL CONFIGURATION
# ============================================================================

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
from typing import List, Dict, Any, Optional, Tuple, Union, Callable, Literal
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
from dataclasses import dataclass, field, asdict, fields

# Suppress non-critical warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='plotly')

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEA_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "fea_solutions")
VISUALIZATION_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "visualization_outputs")
TEMP_ANIMATION_DIR = os.path.join(SCRIPT_DIR, "temp_animations")
EXPORT_DIR = os.path.join(SCRIPT_DIR, "exports")

# Create directories if they don't exist
for d in [FEA_SOLUTIONS_DIR, VISUALIZATION_OUTPUT_DIR, TEMP_ANIMATION_DIR, EXPORT_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================================
# DATA STRUCTURES: EXPLICIT PHYSICAL UNIT METADATA
# ============================================================================

@dataclass
class PhysicalUnit:
    """
    Metadata class for physical quantities with unit conversion support.
    
    This enables reversible conversion between normalized [0,1] visualization
    values and physical units (K, MPa, μm, etc.) with proper handling of:
    - Scale factors (e.g., meters → micrometers: ×1e6)
    - Offset transforms (e.g., Celsius → Kelvin: +273.15)
    - Validity ranges for scientific sanity checking
    - Critical thresholds for process control decisions
    """
    name: str                    # Human-readable name: "Temperature"
    symbol: str                  # Symbol for labels: "T"
    unit: str                    # Unit string: "K"
    latex_unit: str = ""         # LaTeX rendering: "$\\mathrm{K}$"
    scale_factor: float = 1.0    # Conversion factor: meters→μm = 1e6
    offset: float = 0.0          # Affine offset: °C→K = +273.15
    valid_range: Optional[Tuple[float, float]] = None  # Physical bounds
    critical_thresholds: Optional[Dict[str, float]] = None  # Process limits
    
    def to_physical(self, normalized_value: float, ref_min: float, ref_max: float) -> float:
        """
        Convert normalized [0,1] value back to physical units.
        
        This is the CRITICAL conversion for making predictions actionable.
        
        Formula: physical = (normalized × (ref_max - ref_min) + ref_min) × scale_factor + offset
        """
        if not np.isfinite(normalized_value):
            return np.nan
        physical = ref_min + normalized_value * (ref_max - ref_min)
        return physical * self.scale_factor + self.offset
    
    def to_normalized(self, physical_value: float, ref_min: float, ref_max: float) -> float:
        """
        Convert physical value to normalized [0,1] for visualization.
        
        Inverse of to_physical(). Used for color mapping and marker sizing.
        """
        if not np.isfinite(physical_value):
            return np.nan
        # Reverse the affine transform first
        unshifted = (physical_value - self.offset) / self.scale_factor
        range_val = ref_max - ref_min
        if range_val < 1e-9:
            return 0.5  # Degenerate case: all values identical
        return np.clip((unshifted - ref_min) / range_val, 0.0, 1.0)
    
    def format_value(self, value: float, precision: int = 3) -> str:
        """Format physical value with unit symbol for display."""
        if not np.isfinite(value):
            return f"N/A {self.unit}"
        return f"{value:.{precision}f} {self.unit}"
    
    def format_hover(self, name: str, value: float, normalized: Optional[float] = None, 
                    precision: int = 3) -> str:
        """
        Create rich hover text with both physical and normalized values.
        
        Enables experts to see both the interpretable physical value AND
        the visualization-normalized value for debugging color mapping.
        """
        base = f"<b>{name}</b><br>{self.format_value(value, precision)}"
        if normalized is not None and np.isfinite(normalized):
            base += f"<br><i>Normalized: {normalized:.3f}</i>"
        return base
    
    def check_thresholds(self, physical_value: float) -> Dict[str, str]:
        """
        Check physical value against critical thresholds.
        
        Returns status dict for process control decisions.
        """
        if not self.critical_thresholds:
            return {}
        
        status = {}
        for threshold_name, threshold_value in self.critical_thresholds.items():
            if physical_value >= threshold_value:
                status[threshold_name] = 'EXCEEDED'
            else:
                margin_pct = (threshold_value - physical_value) / max(threshold_value, 1e-9) * 100
                status[threshold_name] = f'OK ({margin_pct:.1f}% margin)'
        return status
    
    def is_valid(self, physical_value: float) -> Tuple[bool, Optional[str]]:
        """Check if physical value is within scientifically valid range."""
        if not np.isfinite(physical_value):
            return False, "Value is not finite"
        if self.valid_range:
            v_min, v_max = self.valid_range
            if not (v_min <= physical_value <= v_max):
                return False, f"Value {physical_value:.2f} {self.unit} outside valid range [{v_min}, {v_max}]"
        return True, None


@dataclass
class RadarDataPoint:
    """
    Single data point with explicit dual-representation: physical AND normalized.
    
    This ensures that visualization encoding (normalized) never loses the
    scientific meaning (physical) of the data.
    """
    name: str
    energy_mj: float           # Physical: laser energy in millijoules
    duration_ns: float         # Physical: pulse duration in nanoseconds  
    peak_physical: float       # Physical: peak field value (e.g., 523.4 K)
    peak_normalized: float = 0.0  # Computed: normalized for visualization [0,1]
    uncertainty_physical: Optional[float] = None  # Optional: ± uncertainty in physical units
    metadata: Dict = field(default_factory=dict)  # Additional context: unit, refs, etc.
    
    def __post_init__(self):
        """Auto-compute normalized value if physical value and refs provided."""
        if (self.peak_normalized == 0.0 and 
            'ref_min' in self.metadata and 
            'ref_max' in self.metadata and
            'unit' in self.metadata):
            unit = self.metadata['unit']
            if isinstance(unit, dict):
                unit = PhysicalUnit(**unit)
            self.peak_normalized = unit.to_normalized(
                self.peak_physical, 
                self.metadata['ref_min'], 
                self.metadata['ref_max']
            )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for export, preserving both representations."""
        return {
            'name': self.name,
            'energy_mj': self.energy_mj,
            'duration_ns': self.duration_ns,
            'peak_physical': self.peak_physical,
            'peak_normalized': self.peak_normalized,
            'uncertainty_physical': self.uncertainty_physical,
            'metadata': {
                k: (asdict(v) if isinstance(v, PhysicalUnit) else v) 
                for k, v in self.metadata.items()
            }
        }


@dataclass  
class RadarVisualizationConfig:
    """Comprehensive configuration for polar radar chart generation."""
    # Axis configuration
    energy_min: Optional[float] = None
    energy_max: Optional[float] = None
    duration_min: float = 0.0
    duration_max: Optional[float] = None
    
    # Color scaling
    normalize_colors: bool = True
    color_reference_min: Optional[float] = None
    color_reference_max: Optional[float] = None
    colorscale: str = "Inferno"
    
    # Visual styling
    width: int = 900
    height: int = 750
    marker_size_range: Tuple[int, int] = (10, 30)
    target_marker_size: int = 25
    show_grid: bool = True
    background_color: str = "white"
    
    # Annotation layers
    show_thresholds: bool = True
    show_confidence_bands: bool = True
    threshold_line_style: Dict = field(default_factory=lambda: {
        'color': 'red', 'width': 2, 'dash': 'dash'
    })
    
    # Interaction
    enable_hover: bool = True
    hover_mode: Literal['closest', 'x', 'y', 'x unified'] = 'closest'
    
    # Export
    include_physical_metadata: bool = True
    export_precision: int = 6
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for serialization."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


# ============================================================================
# 1. CACHE MANAGEMENT UTILITIES
# ============================================================================

class CacheManager:
    """Manages interpolation cache with LRU eviction and hash-based keys."""
    
    @staticmethod
    def generate_cache_key(field_name: str, timestep_idx: int, energy: float,
                          duration: float, time: float, sigma_param: float,
                          spatial_weight: float, n_heads: int, temperature: float,
                          sigma_g: float, s_E: float, s_tau: float, s_t: float,
                          temporal_weight: float, top_k: Optional[int] = None,
                          subsample_factor: Optional[int] = None) -> str:
        """Generate deterministic cache key from all relevant parameters."""
        params_str = f"{field_name}_{timestep_idx}_{energy:.2f}_{duration:.2f}_{time:.2f}"
        params_str += f"_{sigma_param:.2f}_{spatial_weight:.2f}_{n_heads}_{temperature:.2f}"
        params_str += f"_{sigma_g:.2f}_{s_E:.2f}_{s_tau:.2f}_{s_t:.2f}_{temporal_weight:.2f}"
        if top_k: params_str += f"_top{top_k}"
        if subsample_factor: params_str += f"_sub{subsample_factor}"
        return hashlib.md5(params_str.encode()).hexdigest()[:16]

    @staticmethod
    def clear_3d_cache():
        """Clear all cached interpolation results."""
        if 'interpolation_3d_cache' in st.session_state:
            st.session_state.interpolation_3d_cache = {}
        if 'interpolation_field_history' in st.session_state:
            st.session_state.interpolation_field_history = OrderedDict()
        st.success("✅ Cache cleared")

    @staticmethod
    def get_cached_interpolation(field_name: str, timestep_idx: int, params: Dict) -> Optional[Dict]:
        """Retrieve cached interpolation result if available."""
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
        """Store interpolation result in cache with metadata."""
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
        # Track access history for LRU eviction
        if 'interpolation_field_history' not in st.session_state:
            st.session_state.interpolation_field_history = OrderedDict()
        history_key = f"{field_name}_{timestep_idx}"
        st.session_state.interpolation_field_history[history_key] = cache_key
        # Evict oldest if cache exceeds limit
        if len(st.session_state.interpolation_3d_cache) > 20:
            oldest_key = min(st.session_state.interpolation_3d_cache.keys(),
                             key=lambda k: st.session_state.interpolation_3d_cache[k]['timestamp'])
            del st.session_state.interpolation_3d_cache[oldest_key]
        if len(st.session_state.interpolation_field_history) > 10:
            st.session_state.interpolation_field_history.popitem(last=False)


# ============================================================================
# 2. UNIFIED DATA LOADER
# ============================================================================

class UnifiedFEADataLoader:
    """Loads and parses FEA simulation data from VTU files with summary extraction."""
    
    def __init__(self):
        self.simulations: Dict[str, Dict] = {}
        self.summaries: List[Dict] = []
        self.available_fields: set = set()
        self.load_errors: List[str] = []

    def parse_folder_name(self, folder: str) -> Tuple[Optional[float], Optional[float]]:
        """Parse simulation parameters from folder name pattern: q{energy}mJ-delta{duration}ns"""
        match = re.match(r"q([\dp\.]+)mJ-delta([\dp\.]+)ns", folder)
        if not match:
            return None, None
        e, d = match.groups()
        # Handle decimal point encoding: "p" → "."
        return float(e.replace("p", ".")), float(d.replace("p", "."))

    @st.cache_data(show_spinner="Loading simulation data...")
    def load_all_simulations(_self, load_full_mesh: bool = True) -> Tuple[Dict, List]:
        """Load all VTU-based simulations from configured directory."""
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
                    
                    # Extract triangle connectivity for mesh rendering
                    triangles = None
                    for cell_block in mesh0.cells:
                        if cell_block.type == "triangle":
                            triangles = cell_block.data.astype(np.int32)
                            break
                    
                    # Pre-allocate field arrays for all timesteps
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
                        
                        # Shape: (timesteps, points) for scalar, (timesteps, points, components) for vector
                        shape = (len(vtu_files), n_pts) if field_type == "scalar" else (len(vtu_files), n_pts, comp)
                        fields[key] = np.full(shape, np.nan, dtype=np.float32)
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

                # Extract summary statistics for ST-DGPA training
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
        """Extract min/max/mean/std/quartiles for each field at each timestep."""
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
                    
                    # Handle scalar vs vector fields
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
                        # Fill with zeros if no valid data
                        for key in ['min', 'max', 'mean', 'std', 'q25', 'q50', 'q75']:
                            summary['field_stats'][field_name][key].append(0.0)
            except Exception as e:
                st.warning(f"Error processing {vtu_file}: {e}")
                continue
        return summary


# ============================================================================
# 3. ST-DGPA EXTRAPOLATOR
# ============================================================================

class SpatioTemporalGatedPhysicsAttentionExtrapolator:
    """
    Spatio-Temporal Gated Physics Attention (ST-DGPA) model for interpolation.
    
    Combines:
    - Physics-informed embeddings (Fourier number, thermal penetration)
    - Multi-head attention over source simulations
    - Energy/Duration/Time (ETT) gating for parameter-space proximity
    - Temporal coherence weighting for time-series predictions
    """
    
    def __init__(self, sigma_param: float = 0.3, spatial_weight: float = 0.5,
                 n_heads: int = 4, temperature: float = 1.0,
                 sigma_g: float = 0.20, s_E: float = 10.0, s_tau: float = 5.0,
                 s_t: float = 20.0, temporal_weight: float = 0.3):
        # Attention hyperparameters
        self.sigma_param = sigma_param  # Spatial similarity width
        self.spatial_weight = spatial_weight  # Blend factor for spatial similarity
        self.n_heads = n_heads  # Multi-head attention heads
        self.temperature = temperature  # Softmax temperature
        
        # ETT gating hyperparameters
        self.sigma_g = sigma_g  # Gating kernel width
        self.s_E = s_E  # Energy scale for gating
        self.s_tau = s_tau  # Duration scale for gating
        self.s_t = s_t  # Time scale for gating
        self.temporal_weight = temporal_weight  # Blend factor for temporal similarity
        
        # Physics constants (laser soldering domain)
        self.thermal_diffusivity = 1e-5  # m²/s (typical for metals)
        self.laser_spot_radius = 50e-6  # 50 μm
        self.characteristic_length = 100e-6  # 100 μm
        
        # Model state
        self.source_db: List[Dict] = []
        self.embedding_scaler = StandardScaler()
        self.value_scaler = StandardScaler()
        self.source_embeddings: np.ndarray = np.array([])
        self.source_values: np.ndarray = np.array([])
        self.source_metadata: List[Dict] = []
        self.fitted: bool = False

    def load_summaries(self, summaries: List[Dict]):
        """Prepare source database for ST-DGPA attention mechanism."""
        self.source_db = summaries
        if not summaries:
            return
        
        all_embeddings, all_values, metadata = [], [], []
        
        for summary_idx, summary in enumerate(summaries):
            for timestep_idx, t in enumerate(summary['timesteps']):
                # Compute physics-informed embedding for this (energy, duration, time) triplet
                emb = self._compute_enhanced_physics_embedding(
                    summary['energy'], summary['duration'], t
                )
                all_embeddings.append(emb)
                
                # Extract field statistics: [mean, max, std] for each field
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
                
                # Store metadata for interpretability
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
            
            # Fit scalers on source data
            self.embedding_scaler.fit(all_embeddings)
            self.source_embeddings = self.embedding_scaler.transform(all_embeddings)
            self.value_scaler.fit(all_values)
            self.source_values = all_values
            self.source_metadata = metadata
            self.fitted = True
            
            st.info(f"✅ Prepared {len(all_embeddings)} embeddings with {all_embeddings.shape[1]} features")

    def _compute_fourier_number(self, time_ns: float) -> float:
        """Compute dimensionless Fourier number for heat conduction."""
        time_s = time_ns * 1e-9
        return self.thermal_diffusivity * time_s / (self.characteristic_length ** 2)

    def _compute_thermal_penetration(self, time_ns: float) -> float:
        """Compute thermal penetration depth in micrometers."""
        time_s = time_ns * 1e-9
        return np.sqrt(self.thermal_diffusivity * time_s) * 1e6

    def _compute_enhanced_physics_embedding(self, energy: float, duration: float, time: float) -> np.ndarray:
        """
        Compute physics-informed embedding vector for ST-DGPA attention.
        
        Features include:
        - Log-transformed energy (handles wide dynamic range)
        - Power density (energy/duration)
        - Time ratio (time/duration) for phase identification
        - Fourier number and thermal penetration (heat conduction physics)
        - Heating/cooling phase indicators
        """
        logE = np.log1p(energy)  # log(1+E) handles E=0
        power = energy / max(duration, 1e-6)  # Avoid division by zero
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
        """
        Compute Energy/Duration/Time (ETT) gating weights.
        
        Gaussian kernel in normalized parameter space, with optional
        temporal scaling for improved time-series coherence.
        """
        if source_metadata is None:
            source_metadata = self.source_metadata
        
        phi_squared = []
        for meta in source_metadata:
            # Normalized parameter differences
            de = (energy_query - meta['energy']) / self.s_E
            dt = (duration_query - meta['duration']) / self.s_tau
            dtime = (time_query - meta['time']) / self.s_t
            
            # Optional temporal scaling for better extrapolation
            if self.temporal_weight > 0:
                time_scaling_factor = 1.0 + 0.5 * (time_query / max(duration_query, 1e-6))
                dtime *= time_scaling_factor
            
            phi_squared.append(de**2 + dt**2 + dtime**2)
        
        phi_squared = np.array(phi_squared)
        gating = np.exp(-phi_squared / (2 * self.sigma_g**2))
        
        # Normalize to sum to 1
        gating_sum = np.sum(gating)
        return gating / gating_sum if gating_sum > 0 else np.ones_like(gating) / len(gating)

    def _compute_temporal_similarity(self, query_meta: Dict, source_metas: List[Dict]) -> np.ndarray:
        """Compute temporal similarity combining absolute time and Fourier number."""
        similarities = []
        for meta in source_metas:
            time_diff = abs(query_meta['time'] - meta['time'])
            
            # Adaptive tolerance based on process phase
            if query_meta['time'] < query_meta['duration'] * 1.5:
                tolerance = max(query_meta['duration'] * 0.1, 1.0)
            else:
                tolerance = max(query_meta['duration'] * 0.3, 3.0)
            
            # Fourier number similarity (dimensionless time)
            fourier_similarity = np.exp(
                -abs(query_meta.get('fourier_number', 0) - meta.get('fourier_number', 0)) / 0.1
            )
            # Absolute time similarity
            time_similarity = np.exp(-time_diff / tolerance)
            
            # Blend based on temporal_weight
            similarities.append(
                (1 - self.temporal_weight) * time_similarity +
                self.temporal_weight * fourier_similarity
            )
        return np.array(similarities)

    def _compute_spatial_similarity(self, query_meta: Dict, source_metas: List[Dict]) -> np.ndarray:
        """Compute spatial similarity in (energy, duration) parameter space."""
        similarities = []
        for meta in source_metas:
            # Normalized differences with domain-specific scales
            e_diff = abs(query_meta['energy'] - meta['energy']) / 50.0
            d_diff = abs(query_meta['duration'] - meta['duration']) / 20.0
            total_diff = np.sqrt(e_diff**2 + d_diff**2)
            similarities.append(np.exp(-total_diff / self.sigma_param))
        return np.array(similarities)

    def _multi_head_attention_with_gating(self, query_embedding: np.ndarray,
                                         query_meta: Dict) -> Tuple:
        """
        Core ST-DGPA attention mechanism with multi-head + ETT gating.
        
        Returns:
            Tuple of (prediction, final_weights, physics_attention, ett_gating)
        """
        if not self.fitted or len(self.source_embeddings) == 0:
            return None, None, None, None
        
        # Normalize query embedding
        query_norm = self.embedding_scaler.transform([query_embedding])[0]
        n_sources = len(self.source_embeddings)
        
        # Multi-head attention
        head_weights = np.zeros((self.n_heads, n_sources))
        for head in range(self.n_heads):
            np.random.seed(42 + head)  # Reproducible per-head projections
            proj_dim = min(8, query_norm.shape[0])
            proj_matrix = np.random.randn(query_norm.shape[0], proj_dim)
            
            # Project to head-specific subspace
            query_proj = query_norm @ proj_matrix
            source_proj = self.source_embeddings @ proj_matrix
            
            # Compute attention scores via Gaussian kernel
            distances = np.linalg.norm(query_proj - source_proj, axis=1)
            scores = np.exp(-distances**2 / (2 * self.sigma_param**2))
            
            # Blend with spatial similarity if enabled
            if self.spatial_weight > 0:
                spatial_sim = self._compute_spatial_similarity(query_meta, self.source_metadata)
                scores = (1 - self.spatial_weight) * scores + self.spatial_weight * spatial_sim
            
            # Blend with temporal similarity if enabled
            if self.temporal_weight > 0:
                temporal_sim = self._compute_temporal_similarity(query_meta, self.source_metadata)
                scores = (1 - self.temporal_weight) * scores + self.temporal_weight * temporal_sim
            
            head_weights[head] = scores
        
        # Average across heads and apply temperature scaling
        avg_weights = np.mean(head_weights, axis=0)
        if self.temperature != 1.0:
            avg_weights = avg_weights ** (1.0 / self.temperature)
        
        # Stable softmax
        max_weight = np.max(avg_weights)
        exp_weights = np.exp(avg_weights - max_weight)
        physics_attention = exp_weights / (np.sum(exp_weights) + 1e-12)
        
        # Compute ETT gating
        ett_gating = self._compute_ett_gating(
            query_meta['energy'], query_meta['duration'], query_meta['time']
        )
        
        # Combine attention and gating
        combined_weights = physics_attention * ett_gating
        combined_sum = np.sum(combined_weights)
        final_weights = combined_weights / combined_sum if combined_sum > 1e-12 else physics_attention
        
        # Weighted prediction
        prediction = np.sum(final_weights[:, np.newaxis] * self.source_values, axis=0) \
            if len(self.source_values) > 0 else np.zeros(1)
        
        return prediction, final_weights, physics_attention, ett_gating

    def predict_field_statistics(self, energy_query: float, duration_query: float,
                                time_query: float) -> Optional[Dict]:
        """Predict field statistics (mean/max/std) for a single timestep."""
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
        
        # Unpack field predictions from flat prediction vector
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
        """Heuristic confidence based on temporal proximity to training data."""
        if time_query < duration_query * 0.5:
            return 0.6  # Early heating: moderate confidence
        elif time_query < duration_query * 1.5:
            return 0.8  # Peak heating/early cooling: high confidence
        else:
            return 0.9  # Late cooling: very high confidence (diffusion dominates)

    def _compute_heat_transfer_indicators(self, energy: float, duration: float,
                                         time: float) -> Dict:
        """Compute dimensionless heat transfer indicators for interpretability."""
        fourier_number = self._compute_fourier_number(time)
        thermal_penetration = self._compute_thermal_penetration(time)
        
        # Identify process phase
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
        """Predict field statistics across multiple timesteps."""
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
        
        # Initialize field prediction containers
        if self.source_db:
            common_fields = set(f for s in self.source_db for f in s['field_stats'].keys())
            for field in common_fields:
                results['field_predictions'][field] = {'mean': [], 'max': [], 'std': []}
        
        # Predict at each timestep
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
                # Fill with NaNs if prediction failed
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


# ============================================================================
# 4. POLAR RADAR VISUALIZER (FIXED: KNOWN_UNITS CLASS ATTRIBUTE)
# ============================================================================

class PolarRadarVisualizer:
    """
    Enhanced polar radar chart visualizer with EXPLICIT normalized↔physical unit handling.
    
    KEY DESIGN PRINCIPLES:
    1. Normalization is VISUAL ONLY - physical values preserved throughout pipeline
    2. All conversions are reversible and metadata-preserving
    3. Thresholds and annotations always use physical units for process control
    4. Exported data includes both representations for downstream reproducibility
    5. Interactive hover shows BOTH normalized (for debugging) and physical (for interpretation)
    
    FIX: KNOWN_UNITS is a CLASS attribute, accessed via PolarRadarVisualizer.KNOWN_UNITS
    """
    
    # CLASS ATTRIBUTE: Predefined physical units for common laser soldering fields
    # Accessed via PolarRadarVisualizer.KNOWN_UNITS (NOT via instance)
    KNOWN_UNITS: Dict[str, PhysicalUnit] = {
        'temperature': PhysicalUnit(
            name="Temperature", symbol="T", unit="K", latex_unit="$\\mathrm{K}$",
            valid_range=(293.0, 1500.0),  # Room temp to vaporization
            critical_thresholds={"solder_melt": 450.0, "substrate_damage": 800.0}
        ),
        'stress_von_mises': PhysicalUnit(
            name="Von Mises Stress", symbol="σ_vM", unit="MPa", latex_unit="$\\mathrm{MPa}$",
            valid_range=(0.0, 500.0),
            critical_thresholds={"yield_strength": 200.0, "ultimate_strength": 350.0}
        ),
        'strain': PhysicalUnit(
            name="Equivalent Strain", symbol="ε", unit="", latex_unit="$\\varepsilon$",
            valid_range=(0.0, 0.5)
        ),
        'displacement': PhysicalUnit(
            name="Displacement", symbol="u", unit="μm", latex_unit="$\\mu\\mathrm{m}$",
            scale_factor=1e6  # Store in meters, display in micrometers
        ),
        'heat_flux': PhysicalUnit(
            name="Heat Flux", symbol="q", unit="W/m²", latex_unit="$\\mathrm{W}/\\mathrm{m}^2$",
            valid_range=(0.0, 1e9)
        )
    }
    
    def __init__(self, config: Optional[RadarVisualizationConfig] = None):
        """Initialize visualizer with default or custom configuration."""
        self.config = config or RadarVisualizationConfig()
        self.color_scale_temp = 'Inferno'
        self.color_scale_stress = 'Plasma'
        self.color_scale_default = 'Viridis'
        self.target_symbol = 'star-diamond'
        self.source_symbol = 'circle'
        self.available_color_scales = [
            'Inferno', 'Plasma', 'Viridis', 'Magma', 'Cividis',
            'RdYlGn', 'RdYlBu', 'Spectral', 'Turbo', 'Rainbow'
        ]
        self._last_conversion_log: List[Dict] = []  # For debugging/auditing
    
    @staticmethod
    def safe_normalize(values: np.ndarray, reference_max: Optional[float] = None, 
                      reference_min: Optional[float] = None,
                      handle_invalid: Literal['zero', 'nan', 'mask'] = 'zero') -> Tuple[np.ndarray, float, float]:
        """
        Normalize values to [0,1] with robust edge-case handling.
        
        CRITICAL: Returns (normalized, actual_ref_min, actual_ref_max) to enable
        reversible conversion back to physical units.
        
        Args:
            values: Input array of numeric values
            reference_max/min: Optional fixed bounds for normalization
            handle_invalid: Strategy for NaN/Inf values
            
        Returns:
            Tuple of (normalized_array, actual_ref_min, actual_ref_max)
        """
        clean = np.asarray(values, dtype=float)
        
        # Handle invalid values per strategy
        if handle_invalid == 'zero':
            clean = np.nan_to_num(clean, nan=0.0, posinf=0.0, neginf=0.0)
        elif handle_invalid == 'mask':
            mask = np.isfinite(clean)
            if not np.any(mask):
                return np.zeros_like(clean), 0.0, 1.0
        # 'nan' strategy leaves NaNs for plotly to handle
        
        # Determine reference range
        if reference_max is None or reference_min is None:
            valid_vals = clean[np.isfinite(clean)] if handle_invalid == 'mask' else clean
            if len(valid_vals) == 0:
                return np.zeros_like(clean), 0.0, 1.0
            auto_min, auto_max = np.min(valid_vals), np.max(valid_vals)
            if reference_min is None:
                reference_min = auto_min
            if reference_max is None:
                reference_max = auto_max
        
        # Compute normalization
        range_val = reference_max - reference_min
        if range_val < 1e-9:
            # Degenerate case: all values identical
            normalized = np.full_like(clean, 0.5, dtype=float)
            actual_min = actual_max = reference_min
        else:
            normalized = (clean - reference_min) / range_val
            normalized = np.clip(normalized, 0.0, 1.0)
            actual_min, actual_max = reference_min, reference_max
        
        return normalized, actual_min, actual_max
    
    @staticmethod
    def denormalize_values(normalized_values: np.ndarray, ref_min: float, ref_max: float,
                          scale_factor: float = 1.0, offset: float = 0.0) -> np.ndarray:
        """
        Convert normalized [0,1] values back to physical units.
        
        This is the CRITICAL conversion for scientific interpretation.
        
        Formula: physical = (normalized × (ref_max - ref_min) + ref_min) × scale_factor + offset
        """
        physical = ref_min + normalized_values * (ref_max - ref_min)
        return physical * scale_factor + offset
    
    @staticmethod
    def compute_energy_to_angle(energies: np.ndarray, e_min: float, e_max: float, 
                               add_jitter: bool = False, jitter_amount: float = 2.0,
                               seed: Optional[int] = 42, offset_deg: float = 0.0) -> np.ndarray:
        """
        Convert energy values to polar angles in degrees.
        
        Formula: θ = 360° × (E - E_min) / (E_max - E_min) + offset
        """
        e_range = e_max - e_min
        if e_range < 1e-6:
            e_range = 1.0
            e_min = e_min - 0.5
        
        # Base angle computation
        angles_deg = 360.0 * (energies - e_min) / e_range
        angles_deg = (angles_deg + offset_deg) % 360.0
        
        # Add jitter to separate overlapping points
        if add_jitter and len(angles_deg) > 1:
            rounded = np.round(angles_deg, 1)
            unique, counts = np.unique(rounded, return_counts=True)
            
            if np.any(counts > 1):
                if seed is not None:
                    np.random.seed(seed)
                jitter = np.random.uniform(-jitter_amount, jitter_amount, size=len(angles_deg))
                angles_deg = (angles_deg + jitter) % 360.0
        
        return angles_deg
    
    @staticmethod
    def generate_polar_ticks(e_min: float, e_max: float, n_ticks: int = 6) -> Tuple[np.ndarray, np.ndarray]:
        """Generate tick values for the angular (energy) axis."""
        tick_energies = np.linspace(e_min, e_max, n_ticks)
        e_range = e_max - e_min
        if e_range < 1e-6:
            e_range = 1.0
        tick_angles_rad = 2 * np.pi * (tick_energies - e_min) / e_range
        tick_angles_deg = np.degrees(tick_angles_rad) % 360
        return tick_energies, tick_angles_deg
    
    def determine_colorscale(self, field_type: str) -> str:
        """Select appropriate colorscale based on field type."""
        field_lower = field_type.lower()
        if any(kw in field_lower for kw in ['temp', 'heat', 'thermal']):
            return self.color_scale_temp
        elif any(kw in field_lower for kw in ['stress', 'strain', 'von', 'mises']):
            return self.color_scale_stress
        else:
            return self.color_scale_default
    
    def determine_field_title(self, field_type: str) -> str:
        """Generate human-readable title for the field."""
        # FIX: Access KNOWN_UNITS via class name, not instance
        unit = PolarRadarVisualizer.KNOWN_UNITS.get(field_type)
        if unit:
            return f"Peak {unit.name} ({unit.unit})"
        
        field_lower = field_type.lower()
        if 'temp' in field_lower:
            return "Peak Temperature (K)"
        elif 'stress' in field_lower or 'von' in field_lower:
            return "Peak von Mises Stress (MPa)"
        elif 'strain' in field_lower:
            return "Peak Strain"
        elif 'displacement' in field_lower:
            return "Peak Displacement (μm)"
        else:
            return f"Peak {field_type.replace('_', ' ').title()}"
    
    def prepare_radar_data(self, df: pd.DataFrame, field_type: str, 
                          timestep: int = 1) -> Tuple[pd.DataFrame, Dict]:
        """
        Prepare DataFrame for radar visualization with explicit unit handling.
        
        Returns:
            Tuple of (prepared_df, scaling_metadata)
            scaling_metadata contains refs needed for denormalization
        """
        # FIX: Access KNOWN_UNITS via class name
        unit = PolarRadarVisualizer.KNOWN_UNITS.get(field_type, PhysicalUnit(
            name=field_type.replace('_', ' ').title(),
            symbol=field_type[0].upper() if field_type else "V",
            unit=""
        ))
        
        # Extract and validate physical values
        physical_values = pd.to_numeric(df['Peak_Value'], errors='coerce').values
        energies = pd.to_numeric(df['Energy'], errors='coerce').values
        durations = pd.to_numeric(df['Duration'], errors='coerce').values
        
        # Create validity mask
        valid_mask = (
            np.isfinite(physical_values) & 
            np.isfinite(energies) & 
            np.isfinite(durations) &
            (durations >= 0)
        )
        
        if not np.any(valid_mask):
            return pd.DataFrame(), {}
        
        # Filter to valid data
        df_valid = df[valid_mask].copy()
        physical_values = physical_values[valid_mask]
        
        # Determine color scaling references
        if self.config.normalize_colors:
            ref_min = self.config.color_reference_min
            ref_max = self.config.color_reference_max
            
            if ref_min is None or ref_max is None:
                p_min, p_max = np.min(physical_values), np.max(physical_values)
                padding = (p_max - p_min) * 0.05
                if ref_min is None:
                    ref_min = p_min - padding
                if ref_max is None:
                    ref_max = p_max + padding
        else:
            ref_min = ref_max = None
        
        # Compute normalized values for visual encoding
        normalized_values, actual_min, actual_max = self.safe_normalize(
            physical_values, ref_min, ref_max
        )
        
        # Log conversion for auditability
        self._last_conversion_log.append({
            'timestamp': datetime.now().isoformat(),
            'field': field_type,
            'timestep': timestep,
            'n_points': len(physical_values),
            'physical_range': (float(np.min(physical_values)), float(np.max(physical_values))),
            'normalization_refs': (float(actual_min), float(actual_max)),
            'unit': asdict(unit)
        })
        
        # Attach normalized values and metadata to DataFrame
        df_valid['Peak_Normalized'] = normalized_values
        df_valid['Peak_Physical'] = physical_values
        df_valid['Unit_Symbol'] = unit.symbol
        df_valid['Unit_Name'] = unit.unit
        
        scaling_metadata = {
            'field_type': field_type,
            'unit': unit,
            'normalization_ref_min': actual_min,
            'normalization_ref_max': actual_max,
            'scale_factor': unit.scale_factor,
            'offset': unit.offset,
            'timestep': timestep
        }
        
        return df_valid, scaling_metadata
    
    def create_polar_radar_chart(
        self,
        df: pd.DataFrame,
        field_type: str = 'temperature',
        query_params: Optional[Dict] = None,
        timestep: int = 1,
        width: int = 800,
        height: int = 700,
        show_legend: bool = True,
        marker_size_range: Tuple[int, int] = (10, 30),
        # === Original customization parameters ===
        radial_grid_width: float = 1.0,
        angular_grid_width: float = 1.0,
        radial_tick_font_size: int = 12,
        angular_tick_font_size: int = 12,
        title_font_size: int = 18,
        margin_pad: Optional[Dict[str, int]] = None,
        energy_min: Optional[float] = None,
        energy_max: Optional[float] = None,
        target_peak_value: Optional[float] = None,  # Physical value for target marker
        # === New expansion parameters ===
        add_jitter: bool = True,
        jitter_amount: float = 2.0,
        normalize_colors: bool = True,
        color_reference_max: Optional[float] = None,
        color_reference_min: Optional[float] = None,
        radial_axis_max: Optional[float] = None,
        radial_axis_min: Optional[float] = None,
        highlight_target: bool = True,
        # === Additional advanced parameters ===
        jitter_seed: Optional[int] = 42,
        angle_offset_deg: float = 0.0,
        show_grid: bool = True,
        show_radial_labels: bool = True,
        show_angular_labels: bool = True,
        background_color: str = 'white',
        polar_background_color: str = 'white',
        source_marker_opacity: float = 0.85,
        target_marker_size: int = 25,
        target_marker_line_width: int = 3,
        hover_template_source: Optional[str] = None,
        hover_template_target: Optional[str] = None,
        custom_colorscale: Optional[str] = None,
        colorbar_title: Optional[str] = None,
        colorbar_position_x: float = 1.02,
        colorbar_thickness: int = 20,
        enable_hover: bool = True,
        hover_mode: str = 'closest',
        # === NEW: Unit conversion parameters ===
        show_thresholds: bool = True,
        show_confidence_bands: bool = True,
        target_uncertainty: Optional[float] = None  # Physical uncertainty for target
    ) -> go.Figure:
        """
        Create enhanced polar radar chart with explicit normalized↔physical unit handling.
        
        KEY FEATURES:
        - Visual encoding uses normalized values for consistent color perception
        - All labels, tooltips, and exports use physical units for interpretation
        - Threshold annotations drawn in physical coordinates, converted to plot positions
        - Target query marker shows predicted physical value with uncertainty
        - Export metadata includes conversion refs for reproducibility
        """
        # Set default margin padding
        if margin_pad is None:
            margin_pad = {'l': 60, 'r': 80, 't': 80, 'b': 60}
        
        # Handle empty DataFrame
        if df is None or df.empty:
            fig = go.Figure()
            fig.update_layout(
                title="No data available for radar visualization",
                width=width,
                height=height,
                margin=margin_pad
            )
            return fig
        
        # === DATA PREPARATION WITH UNIT HANDLING ===
        df_prep, scaling_meta = self.prepare_radar_data(df, field_type, timestep)
        
        if df_prep.empty:
            fig = go.Figure()
            fig.update_layout(
                title=f"⚠️ No valid data for {field_type} at timestep {timestep}",
                width=width,
                height=height,
                margin=margin_pad
            )
            return fig
        
        unit = scaling_meta['unit']
        
        # Extract prepared columns
        energies = df_prep['Energy'].values
        durations = df_prep['Duration'].values
        physical_values = df_prep['Peak_Physical'].values
        normalized_values = df_prep['Peak_Normalized'].values
        sim_names = df_prep['Name'].tolist()
        
        # === ENERGY AXIS (ANGULAR) SCALING ===
        e_min_data = float(np.nanmin(energies))
        e_max_data = float(np.nanmax(energies))
        
        if energy_min is not None and np.isfinite(energy_min):
            e_min = energy_min
        else:
            e_min = e_min_data
        
        if energy_max is not None and np.isfinite(energy_max) and energy_max > e_max_data * 1.05:
            e_max = energy_max
        else:
            e_max = e_max_data * 1.05
        
        # Compute angles with optional jitter
        angles_deg = self.compute_energy_to_angle(
            energies, e_min, e_max,
            add_jitter=add_jitter,
            jitter_amount=jitter_amount,
            seed=jitter_seed,
            offset_deg=angle_offset_deg
        )
        
        # Generate axis ticks
        tick_energies, tick_angles_deg = self.generate_polar_ticks(e_min, e_max, n_ticks=6)
        tick_angles_deg = (tick_angles_deg + angle_offset_deg) % 360
        
        # === VALUE NORMALIZATION FOR COLORING ===
        if normalize_colors:
            norm_peak, norm_ref_min, norm_ref_max = self.safe_normalize(
                physical_values, 
                reference_max=color_reference_max,
                reference_min=color_reference_min
            )
        else:
            norm_peak = physical_values.copy()
            norm_ref_min, norm_ref_max = None, None
        
        # === RADIAL AXIS (DURATION) SCALING ===
        if radial_axis_min is not None and np.isfinite(radial_axis_min):
            r_min = radial_axis_min
        else:
            r_min = 0
        
        if radial_axis_max is not None and np.isfinite(radial_axis_max):
            r_max = radial_axis_max
        else:
            max_dur = float(np.max(durations)) if len(durations) > 0 else 1.0
            r_max = max_dur * 1.1 if max_dur > 0 else 1.2
        
        # === COLORSCALE AND LABELS ===
        colorscale = custom_colorscale if custom_colorscale else self.determine_colorscale(field_type)
        field_title = colorbar_title if colorbar_title else self.determine_field_title(field_type)
        
        # === CREATE PLOTLY FIGURE ===
        fig = go.Figure()
        
        # Compute marker sizes based on normalized values
        size_min, size_max = marker_size_range
        marker_sizes = size_min + (size_max - size_min) * normalized_values
        
        # === SOURCE SIMULATIONS TRACE ===
        # Build rich hover text with BOTH normalized and physical values
        if hover_template_source is None:
            hover_texts = []
            for i in range(len(df_prep)):
                norm_val = normalized_values[i]
                phys_val = physical_values[i]
                
                hover = unit.format_hover(
                    name=sim_names[i],
                    value=phys_val,
                    normalized=norm_val if enable_hover else None,
                    precision=3
                )
                hover += f"<br>Energy: {energies[i]:.2f} mJ"
                hover += f"<br>Duration: {durations[i]:.2f} ns"
                hover += f"<br>Timestep: {timestep}"
                hover_texts.append(hover)
        else:
            hover_texts = None
        
        # Determine color values: normalized for consistent colorscale, or physical for auto-scaling
        color_values = normalized_values if normalize_colors else physical_values
        
        fig.add_trace(go.Scatterpolar(
            r=durations,
            theta=angles_deg,
            mode='markers',
            marker=dict(
                size=marker_sizes,
                color=color_values,
                colorscale=colorscale,
                colorbar=dict(
                    title=field_title,
                    thickness=colorbar_thickness,
                    x=colorbar_position_x,
                    len=0.5,
                    y=0.5,
                    yanchor='middle',
                    tickformat=f".{6}f"
                ) if show_legend else None,
                line=dict(width=2, color='white'),
                symbol=self.source_symbol,
                showscale=show_legend
            ),
            text=hover_texts,
            hoverinfo='text' if enable_hover else 'skip',
            hovertemplate=hover_template_source,
            name='Source Simulations',
            opacity=source_marker_opacity
        ))
        
        # === THRESHOLD ANNOTATIONS (Physical Units → Plot Coordinates) ===
        if show_thresholds and unit.critical_thresholds:
            for threshold_name, threshold_value in unit.critical_thresholds.items():
                # Convert physical threshold to normalized for colorbar positioning
                if normalize_colors and norm_ref_min is not None and norm_ref_max is not None:
                    norm_threshold = unit.to_normalized(
                        threshold_value, norm_ref_min, norm_ref_max
                    )
                    # Add annotation near colorbar showing threshold
                    fig.add_annotation(
                        x=colorbar_position_x + 0.05,
                        y=norm_threshold,
                        xref='paper',
                        yref='colorbar',
                        text=f"{threshold_name}: {unit.format_value(threshold_value)}",
                        showarrow=True,
                        arrowhead=2,
                        ax=20,
                        ay=0,
                        font=dict(size=9, color='red'),
                        align='left',
                        bordercolor='red',
                        borderwidth=1,
                        borderpad=4,
                        bgcolor='white',
                        opacity=0.9
                    )
        
        # === TARGET QUERY MARKER (Prediction) ===
        if query_params and highlight_target:
            q_e = query_params.get('Energy')
            q_d = query_params.get('Duration')
            
            if (q_e is not None and q_d is not None and 
                np.isfinite(q_e) and np.isfinite(q_d)):
                
                # Compute target angle using same scaling as sources
                q_angle_deg = self.compute_energy_to_angle(
                    np.array([q_e]), e_min, e_max,
                    add_jitter=False,
                    offset_deg=angle_offset_deg
                )[0]
                
                # Determine target marker color based on physical value
                if target_peak_value is not None and np.isfinite(target_peak_value):
                    if normalize_colors and norm_ref_min is not None and norm_ref_max is not None:
                        norm_target = unit.to_normalized(
                            target_peak_value, norm_ref_min, norm_ref_max
                        )
                        norm_target = np.clip(norm_target, 0, 1)
                        target_color = px.colors.sample_colorscale(colorscale, [norm_target])[0]
                    else:
                        # Fallback: scale relative to source data range
                        p_min = np.min(physical_values)
                        p_max = np.max(physical_values)
                        p_range = p_max - p_min
                        if p_range > 1e-9:
                            norm_target = (target_peak_value - p_min) / p_range
                        else:
                            norm_target = 0.5
                        norm_target = np.clip(norm_target, 0, 1)
                        target_color = px.colors.sample_colorscale(colorscale, [norm_target])[0]
                else:
                    target_color = '#FF0000'  # Fallback: bright red
                
                # Build target hover with uncertainty if available
                if hover_template_target is None:
                    peak_display = unit.format_value(target_peak_value) if target_peak_value is not None else "N/A"
                    hover_template_target = (
                        f"<b>🎯 Target Query</b><br>"
                        f"Energy: {q_e:.2f} mJ<br>"
                        f"Duration: {q_d:.2f} ns<br>"
                        f"Predicted Peak: {peak_display}"
                    )
                    if target_uncertainty is not None and np.isfinite(target_uncertainty):
                        hover_template_target += f"<br>Uncertainty: ±{unit.format_value(target_uncertainty)}"
                    hover_template_target += "<extra></extra>"
                
                fig.add_trace(go.Scatterpolar(
                    r=[q_d],
                    theta=[q_angle_deg],
                    mode='markers',
                    marker=dict(
                        size=target_marker_size,
                        color=target_color,
                        symbol=self.target_symbol,
                        line=dict(width=target_marker_line_width, color='black'),
                        showscale=False
                    ),
                    name='Target Prediction',
                    hovertemplate=hover_template_target if enable_hover else None,
                    hoverinfo='text' if enable_hover else 'skip'
                ))
        
        # === CONFIDENCE BANDS (Optional) ===
        if show_confidence_bands and 'Peak_Uncertainty' in df_prep.columns:
            uncertainties = df_prep['Peak_Uncertainty'].values
            for i in range(len(df_prep)):
                if np.isfinite(uncertainties[i]):
                    fig.add_trace(go.Scatterpolar(
                        r=[durations[i] - uncertainties[i], durations[i] + uncertainties[i]],
                        theta=[angles_deg[i], angles_deg[i]],
                        mode='lines',
                        line=dict(color='gray', width=1, dash='dot'),
                        opacity=0.3,
                        showlegend=False,
                        hoverinfo='skip'
                    ))
        
        # === POLAR LAYOUT CONFIGURATION ===
        polar_config = dict(
            bgcolor=polar_background_color,
            radialaxis=dict(
                visible=show_radial_labels,
                title="Pulse Duration (ns)",
                range=[r_min, r_max],
                gridcolor="lightgray" if show_grid else "rgba(0,0,0,0)",
                gridwidth=radial_grid_width if show_grid else 0,
                tickfont=dict(size=radial_tick_font_size),
                title_font=dict(size=radial_tick_font_size + 2),
                showticklabels=show_radial_labels
            ),
            angularaxis=dict(
                visible=show_angular_labels,
                direction="clockwise",
                rotation=90 - angle_offset_deg,
                gridcolor="lightgray" if show_grid else "rgba(0,0,0,0)",
                gridwidth=angular_grid_width if show_grid else 0,
                tickfont=dict(size=angular_tick_font_size),
                tickmode='array',
                tickvals=tick_angles_deg,
                ticktext=[f"{e:.1f}" for e in tick_energies],
                thetaunit="degrees",
                showticklabels=show_angular_labels
            )
        )
        
        # === MAIN LAYOUT ===
        fig.update_layout(
            title=dict(
                text=(
                    f"Polar Radar: {field_title} at t={timestep} ns<br>"
                    f"<span style='font-size:{max(12, title_font_size-4)}px; color:gray;'>"
                    f"Energy: {e_min:.2f} – {e_max:.2f} mJ • Duration: {r_min:.1f} – {r_max:.1f} ns • "
                    f"Range: {unit.format_value(scaling_meta['normalization_ref_min'])}–"
                    f"{unit.format_value(scaling_meta['normalization_ref_max'])}"
                    f"</span>"
                ),
                font=dict(size=title_font_size, family="Arial, sans-serif"),
                x=0.5,
                xanchor='center',
                pad=dict(t=10)
            ),
            polar=polar_config,
            width=width,
            height=height,
            showlegend=show_legend,
            margin=margin_pad,
            hovermode=hover_mode if enable_hover else False,
            plot_bgcolor=background_color,
            paper_bgcolor=background_color,
            font=dict(family="Arial, sans-serif", size=12)
        )
        
        # Store metadata for export/reconstruction
        fig._scaling_metadata = scaling_meta  # type: ignore
        fig._conversion_log = self._last_conversion_log[-1] if self._last_conversion_log else None  # type: ignore
        
        return fig
    
    def create_multi_timestep_radar(
        self,
        summaries: List[Dict],
        field_type: str,
        query_params: Optional[Dict],
        timesteps: List[int],
        width: int = 350,
        height: int = 350,
        shared_colorscale: bool = True,
        shared_radial_range: bool = True,
        **kwargs
    ) -> List[Tuple[go.Figure, int, Dict]]:
        """Create multiple synchronized radar charts for timestep comparison."""
        figures = []
        
        # Pre-compute global ranges if sharing is enabled
        global_peak_min, global_peak_max = None, None
        global_dur_max = None
        
        if shared_colorscale or shared_radial_range:
            all_peaks = []
            all_durations = []
            for s in summaries:
                field_stats = s.get('field_stats', {}).get(field_type, {})
                peak_list = field_stats.get('max', [])
                dur = s.get('duration', 0)
                for p in peak_list:
                    if np.isfinite(p):
                        all_peaks.append(p)
                if np.isfinite(dur):
                    all_durations.append(dur)
            
            if shared_colorscale and all_peaks:
                global_peak_min = float(np.min(all_peaks))
                global_peak_max = float(np.max(all_peaks))
            
            if shared_radial_range and all_durations:
                global_dur_max = float(np.max(all_durations)) * 1.1
        
        for t in timesteps:
            # Collect data for this timestep
            rows = []
            for s in summaries:
                if t <= len(s.get('timesteps', [])):
                    field_stats = s.get('field_stats', {}).get(field_type, {})
                    peak_list = field_stats.get('max', [0])
                    if t - 1 < len(peak_list):
                        peak_val = peak_list[t - 1]
                        if not np.isfinite(peak_val):
                            peak_val = 0.0
                        rows.append({
                            'Name': s['name'],
                            'Energy': s['energy'],
                            'Duration': s['duration'],
                            'Peak_Value': float(peak_val)
                        })
            
            if not rows:
                continue
            
            df_t = pd.DataFrame(rows)
            
            # Prepare timestep-specific parameters
            ts_kwargs = kwargs.copy()
            
            if shared_colorscale and global_peak_max is not None:
                ts_kwargs['color_reference_max'] = global_peak_max
                ts_kwargs['color_reference_min'] = global_peak_min
            
            if shared_radial_range and global_dur_max is not None:
                ts_kwargs['radial_axis_max'] = global_dur_max
            
            # Get target value for this timestep if available
            target_val = None
            if query_params and 'interpolation_results' in kwargs.get('metadata', {}):
                results = kwargs['metadata']['interpolation_results']
                if field_type in results.get('field_predictions', {}):
                    preds = results['field_predictions'][field_type]
                    if 'max' in preds and t - 1 < len(preds['max']):
                        target_val = preds['max'][t - 1]
                ts_kwargs['target_peak_value'] = target_val
            
            # Create chart
            fig = self.create_polar_radar_chart(
                df_t,
                field_type=field_type,
                query_params=query_params,
                timestep=t,
                width=width,
                height=height,
                show_legend=(t == timesteps[0]),
                marker_size_range=(8, 20),
                title_font_size=kwargs.get('title_font_size', 14) - 2,
                radial_tick_font_size=kwargs.get('radial_tick_font_size', 10) - 1,
                angular_tick_font_size=kwargs.get('angular_tick_font_size', 10) - 1,
                **ts_kwargs
            )
            
            # Metadata for downstream use
            metadata = {
                'timestep': t,
                'n_points': len(df_t),
                'energy_range': (df_t['Energy'].min(), df_t['Energy'].max()),
                'duration_range': (df_t['Duration'].min(), df_t['Duration'].max()),
                'peak_range': (df_t['Peak_Value'].min(), df_t['Peak_Value'].max()),
                'target_value': target_val
            }
            
            figures.append((fig, t, metadata))
        
        return figures
    
    def export_chart_data(self, fig: go.Figure, df_original: pd.DataFrame,
                         format: Literal['json', 'csv', 'parquet'] = 'json') -> Union[str, bytes]:
        """
        Export chart data with explicit normalized↔physical unit mapping.
        
        Output includes:
        - Original physical values
        - Normalized values used for visualization
        - Conversion metadata for reproducibility
        - Unit definitions for interpretation
        """
        scaling_meta = getattr(fig, '_scaling_metadata', {})
        conversion_log = getattr(fig, '_conversion_log', {})
        
        # Build export dataframe with dual representations
        export_df = df_original.copy()
        if 'Peak_Value' in export_df.columns and 'Peak_Normalized' not in export_df.columns:
            # Re-compute normalized values if not already present
            unit = scaling_meta.get('unit', PhysicalUnit("value", "V", ""))
            normalized, ref_min, ref_max = self.safe_normalize(
                export_df['Peak_Value'].values,
                scaling_meta.get('normalization_ref_min'),
                scaling_meta.get('normalization_ref_max')
            )
            export_df['Peak_Normalized'] = normalized
            export_df['Normalization_Ref_Min'] = ref_min
            export_df['Normalization_Ref_Max'] = ref_max
        
        # Add unit metadata columns
        if scaling_meta.get('unit'):
            unit = scaling_meta['unit']
            export_df['Unit_Name'] = unit.name
            export_df['Unit_Symbol'] = unit.symbol
            export_df['Unit_Label'] = unit.unit
        
        if format == 'json':
            export_data = {
                'metadata': {
                    'export_timestamp': datetime.now().isoformat(),
                    'field_type': scaling_meta.get('field_type'),
                    'timestep': scaling_meta.get('timestep'),
                    'unit': asdict(scaling_meta.get('unit', PhysicalUnit("", "", ""))),
                    'normalization_refs': {
                        'min': scaling_meta.get('normalization_ref_min'),
                        'max': scaling_meta.get('normalization_ref_max'),
                        'scale_factor': scaling_meta.get('scale_factor', 1.0),
                        'offset': scaling_meta.get('offset', 0.0)
                    },
                    'conversion_log': conversion_log
                },
                'data': export_df.to_dict(orient='records')
            }
            return json.dumps(export_data, indent=2, default=str)
        
        elif format == 'csv':
            meta_prefix = {
                'meta_field': scaling_meta.get('field_type'),
                'meta_timestep': scaling_meta.get('timestep'),
                'meta_unit_name': scaling_meta.get('unit', PhysicalUnit("", "", "")).name,
                'meta_unit_symbol': scaling_meta.get('unit', PhysicalUnit("", "", "")).symbol,
                'meta_unit_label': scaling_meta.get('unit', PhysicalUnit("", "", "")).unit,
                'meta_norm_ref_min': scaling_meta.get('normalization_ref_min'),
                'meta_norm_ref_max': scaling_meta.get('normalization_ref_max')
            }
            for key, value in meta_prefix.items():
                export_df[key] = value
            return export_df.to_csv(index=False)
        
        elif format == 'parquet':
            import io
            buffer = io.BytesIO()
            export_df.to_parquet(buffer, index=False)
            return buffer.getvalue()
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def export_figure_data(self, fig: go.Figure, format: str = 'json') -> Union[str, bytes]:
        """Export figure configuration for external use."""
        if format == 'json':
            return fig.to_json()
        elif format == 'plotly':
            return fig.to_plotly_json()
        elif format == 'html':
            return fig.to_html(include_plotlyjs='cdn')
        else:
            raise ValueError(f"Unsupported export format: {format}")


# ============================================================================
# 5. ENHANCED VISUALIZER
# ============================================================================

class EnhancedVisualizer:
    """Creates advanced analysis visualizations for ST-DGPA results."""
    
    @staticmethod
    def create_stdgpa_analysis(results: Dict, energy_query: float, duration_query: float,
                            time_points: np.ndarray) -> Optional[go.Figure]:
        """Create comprehensive ST-DGPA attention analysis dashboard."""
        if not results or 'attention_maps' not in results or len(results['attention_maps']) == 0:
            return None
        
        timestep_idx = len(time_points) // 2
        time = time_points[timestep_idx]
        
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                "ST-DGPA Final Weights", "Physics Attention Only", "(E, τ, t) Gating Only",
                "ST-DGPA vs Physics Attention", "Temporal Coherence Analysis", "Heat Transfer Phase",
                "Parameter Space 3D", "Attention Network", "Weight Evolution"
            ]
        )
        
        final_weights = results['attention_maps'][timestep_idx]
        physics_attention = results['physics_attention_maps'][timestep_idx]
        ett_gating = results['ett_gating_maps'][timestep_idx]
        
        # Row 1: Component breakdown
        fig.add_trace(go.Bar(x=list(range(len(final_weights))), y=final_weights, 
                           marker_color='royalblue', showlegend=False), row=1, col=1)
        fig.add_trace(go.Bar(x=list(range(len(physics_attention))), y=physics_attention, 
                           marker_color='forestgreen', showlegend=False), row=1, col=2)
        fig.add_trace(go.Bar(x=list(range(len(ett_gating))), y=ett_gating, 
                           marker_color='crimson', showlegend=False), row=1, col=3)
        
        # Row 2: Comparison and temporal analysis
        fig.add_trace(go.Scatter(x=list(range(len(final_weights))), y=final_weights, 
                               mode='lines+markers', line=dict(color='royalblue', width=3),
                               name='ST-DGPA'), row=2, col=1)
        fig.add_trace(go.Scatter(x=list(range(len(physics_attention))), y=physics_attention, 
                               mode='lines+markers', line=dict(color='forestgreen', width=2, dash='dash'),
                               name='Physics Only'), row=2, col=1)
        
        if st.session_state.get('summaries') and hasattr(st.session_state.extrapolator, 'source_metadata'):
            times, weights = [], []
            for i, weight in enumerate(final_weights):
                if weight > 0.01 and i < len(st.session_state.extrapolator.source_metadata):
                    times.append(st.session_state.extrapolator.source_metadata[i]['time'])
                    weights.append(weight)
            if times and weights:
                fig.add_trace(go.Scatter(x=times, y=weights, mode='markers', 
                                       marker=dict(size=np.array(weights)*50, color=weights, 
                                                  colorscale='Viridis', showscale=False)), 
                            row=2, col=2)
                fig.add_vline(x=time, line_dash="dash", line_color="red", row=2, col=2)
        
        # Row 3: Heat transfer indicators
        indicators = results['heat_transfer_indicators'][timestep_idx] if timestep_idx < len(results['heat_transfer_indicators']) else {}
        if indicators:
            fig.add_annotation(text=f"Phase: {indicators.get('phase', 'N/A')}<br>Regime: {indicators.get('regime', 'N/A')}<br>Fo: {indicators.get('fourier_number', 0):.3f}", 
                            x=0.5, y=0.5, row=3, col=1, showarrow=False, 
                            font=dict(size=11), align='center', bordercolor='gray', 
                            borderwidth=1, borderpad=10, bgcolor='white')
        
        fig.update_layout(
            height=1000, 
            title_text=f"ST-DGPA Analysis at t={time} ns (E={energy_query:.1f} mJ, τ={duration_query:.1f} ns)", 
            showlegend=True,
            hovermode='closest'
        )
        return fig

    @staticmethod
    def create_confidence_plot(results: Dict, time_points: np.ndarray) -> go.Figure:
        """Create prediction confidence visualization over time."""
        fig = go.Figure()
        
        if 'confidence_scores' in results and results['confidence_scores']:
            fig.add_trace(go.Scatter(
                x=time_points, y=results['confidence_scores'], 
                mode='lines+markers', name='Prediction Confidence', 
                line=dict(color='royalblue', width=3), 
                fill='tozeroy', fillcolor='rgba(65,105,225,0.15)'
            ))
        
        if 'temporal_confidences' in results and results['temporal_confidences']:
            fig.add_trace(go.Scatter(
                x=time_points, y=results['temporal_confidences'], 
                mode='lines+markers', name='Temporal Confidence', 
                line=dict(color='forestgreen', width=3, dash='dash')
            ))
        
        fig.update_layout(
            title="Prediction Confidence Over Time", 
            xaxis_title="Time (ns)", 
            yaxis_title="Confidence", 
            height=400, 
            yaxis_range=[0, 1], 
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        return fig


# ============================================================================
# 6. EXPORT UTILITIES
# ============================================================================

class ExportManager:
    """Manages export of results in multiple formats with unit preservation."""
    
    @staticmethod
    def export_to_json(data: Dict, filename: Optional[str] = None) -> Tuple[str, str]:
        """Export data to JSON with NumPy type conversion."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"stdgpa_export_{timestamp}.json"
        
        def convert_numpy(obj):
            if isinstance(obj, np.integer): return int(obj)
            elif isinstance(obj, np.floating): return float(obj)
            elif isinstance(obj, np.ndarray): return obj.tolist()
            elif isinstance(obj, dict): return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list): return [convert_numpy(v) for v in obj]
            elif isinstance(obj, PhysicalUnit): return asdict(obj)
            return obj
        
        json_str = json.dumps(convert_numpy(data), indent=2)
        return json_str, filename

    @staticmethod
    def export_to_csv(df: pd.DataFrame, filename: Optional[str] = None) -> Tuple[str, str]:
        """Export DataFrame to CSV."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"stdgpa_export_{timestamp}.csv"
        
        if df is None or df.empty:
            return "", filename
        
        csv_str = df.to_csv(index=False)
        return csv_str, filename

    @staticmethod
    def export_plotly_figure(fig: go.Figure, format: str = 'png', 
                            filename: Optional[str] = None) -> Optional[bytes]:
        """Export Plotly figure to image or HTML."""
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
            st.info("💡 Tip: Install 'kaleido' for image export: pip install -U kaleido")
            return None


# ============================================================================
# 7. UTILITY FUNCTIONS: PREDICTION → PHYSICAL CONVERSION
# ============================================================================

def convert_prediction_to_physical(
    normalized_prediction: float,
    field_type: str,
    ref_min: float,
    ref_max: float,
    uncertainty_normalized: Optional[float] = None,
    viz: Optional[PolarRadarVisualizer] = None
) -> Dict[str, Any]:
    """
    Convert ST-DGPA normalized prediction back to physical units.
    
    THIS IS THE CRITICAL FUNCTION for making predictions actionable in engineering contexts.
    
    Args:
        normalized_prediction: Model output in [0,1]
        field_type: Field name (e.g., 'temperature', 'stress_von_mises')
        ref_min/ref_max: Normalization references used during training
        uncertainty_normalized: Optional normalized uncertainty
        viz: Optional PolarRadarVisualizer instance for unit lookup
        
    Returns:
        Dict with physical value, uncertainty, validity status, and threshold assessments
    """
    if viz is None:
        viz = PolarRadarVisualizer()
    
    # FIX: Access KNOWN_UNITS via class name
    unit = PolarRadarVisualizer.KNOWN_UNITS.get(field_type, PhysicalUnit(
        name=field_type.replace('_', ' ').title(),
        symbol=field_type[0].upper() if field_type else "V",
        unit=""
    ))
    
    # Convert normalized to physical
    physical_value = unit.to_physical(normalized_prediction, ref_min, ref_max)
    
    result = {
        'physical_value': physical_value,
        'unit': unit.unit,
        'formatted': unit.format_value(physical_value),
        'is_valid': True,
        'warnings': []
    }
    
    # Check against physical validity range
    valid, warning = unit.is_valid(physical_value)
    if not valid and warning:
        result['is_valid'] = False
        result['warnings'].append(warning)
    
    # Convert uncertainty if provided
    if uncertainty_normalized is not None and np.isfinite(uncertainty_normalized):
        # Uncertainty propagates linearly through the affine transform
        unc_physical = uncertainty_normalized * (ref_max - ref_min) * unit.scale_factor
        result['uncertainty_physical'] = unc_physical
        result['formatted_with_unc'] = f"{unit.format_value(physical_value)} ± {unit.format_value(unc_physical)}"
    
    # Check against critical thresholds for process control
    if unit.critical_thresholds:
        result['threshold_status'] = unit.check_thresholds(physical_value)
    
    # Add conversion metadata for reproducibility
    result['conversion_metadata'] = {
        'normalized_input': normalized_prediction,
        'ref_min': ref_min,
        'ref_max': ref_max,
        'scale_factor': unit.scale_factor,
        'offset': unit.offset,
        'unit_definition': asdict(unit)
    }
    
    return result


# ============================================================================
# 8. MAIN APPLICATION (FIXED: use_container_width → width)
# ============================================================================

def main():
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="Laser Soldering ST-DGPA Platform", 
        layout="wide", 
        initial_sidebar_state="expanded", 
        page_icon="🔬"
    )
    
    # Custom CSS for enhanced styling
    st.markdown("""
    <style>
    .main-header { 
        font-size: 2.5rem; 
        text-align: center; 
        margin-bottom: 1.5rem; 
        font-weight: 800; 
        background: linear-gradient(90deg, #1E88E5, #4A00E0); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
    }
    .sub-header { 
        font-size: 1.5rem; 
        color: #2c3e50; 
        margin-top: 1.5rem; 
        margin-bottom: 1rem; 
        padding-bottom: 0.5rem; 
        border-bottom: 2px solid #3498db; 
        font-weight: 600; 
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🔬 Laser Soldering ST-DGPA Analysis Platform</h1>', unsafe_allow_html=True)
    st.caption("v2.1.1 • Fixed: KNOWN_UNITS class attribute access + Streamlit width parameter")

    # Initialize session state
    if 'data_loader' not in st.session_state: 
        st.session_state.data_loader = UnifiedFEADataLoader()
    if 'extrapolator' not in st.session_state: 
        st.session_state.extrapolator = SpatioTemporalGatedPhysicsAttentionExtrapolator()
    if 'polar_viz' not in st.session_state: 
        st.session_state.polar_viz = PolarRadarVisualizer()
    if 'enhanced_viz' not in st.session_state: 
        st.session_state.enhanced_viz = EnhancedVisualizer()
    if 'export_manager' not in st.session_state: 
        st.session_state.export_manager = ExportManager()
    if 'interpolation_results' not in st.session_state: 
        st.session_state.interpolation_results = None
    if 'interpolation_params' not in st.session_state: 
        st.session_state.interpolation_params = None
    if 'polar_query' not in st.session_state: 
        st.session_state.polar_query = None
    if 'data_loaded' not in st.session_state: 
        st.session_state.data_loaded = False

    # ========================================================================
    # SIDEBAR: Configuration & Controls
    # ========================================================================
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        load_full = st.checkbox("Load Full Mesh", value=True, 
                               help="Load complete mesh data for 3D visualization (slower)")
        
        if st.button("🔄 Load All Simulations", type="primary", use_container_width=True):
            with st.spinner("Loading simulation data from VTU files..."):
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
                sigma_g = st.slider("σ_g (Gating)", 0.05, 1.0, 0.20, 0.05,
                                   help="Width of ETT gating kernel in parameter space")
                s_E = st.slider("s_E (Energy)", 0.1, 50.0, 10.0, 0.5,
                               help="Energy scale for gating normalization")
            with col2:
                s_tau = st.slider("s_τ (Duration)", 0.1, 20.0, 5.0, 0.5,
                                 help="Duration scale for gating normalization")
                s_t = st.slider("s_t (Time)", 1.0, 50.0, 20.0, 1.0,
                               help="Time scale for gating normalization")
            
            temporal_weight = st.slider("Temporal Weight", 0.0, 1.0, 0.3, 0.05,
                                       help="Blend factor for temporal vs Fourier similarity")
            
            # Update extrapolator parameters
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

    # ========================================================================
    # MAIN TABS
    # ========================================================================
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
                ├── a_t0001.vtu
                └── ...
            """)
        return

    # ------------------------------------------------------------------------
    # TAB 1: Data Overview
    # ------------------------------------------------------------------------
    with tabs[0]:
        st.subheader("Loaded Simulations Summary")
        
        df_summary = pd.DataFrame([
            {
                'Name': s['name'], 
                'Energy (mJ)': s['energy'], 
                'Duration (ns)': s['duration'], 
                'Timesteps': len(s['timesteps']), 
                'Fields': ', '.join(sorted(s['field_stats'].keys())[:3]) + ('...' if len(s['field_stats']) > 3 else '')
            } 
            for s in st.session_state.summaries
        ])
        
        # FIX: use_container_width → width='stretch'
        st.dataframe(
            df_summary.style.format({'Energy (mJ)': '{:.2f}', 'Duration (ns)': '{:.2f}'})
            .background_gradient(subset=['Energy (mJ)', 'Duration (ns)'], cmap='Blues'), 
            use_container_width=False, width='stretch', height=300
        )
        
        if st.session_state.simulations and next(iter(st.session_state.simulations.values())).get('has_mesh'):
            st.markdown("### 🎨 3D Field Viewer")
            col1, col2, col3 = st.columns(3)
            with col1: 
                sim_name = st.selectbox("Simulation", sorted(st.session_state.simulations.keys()), key="viewer_sim")
            with col2: 
                sim = st.session_state.simulations[sim_name]
                field = st.selectbox("Field", sorted(sim['field_info'].keys()), key="viewer_field")
            with col3: 
                timestep = st.slider("Timestep", 0, sim['n_timesteps']-1, 0, key="viewer_timestep")
            
            if sim['points'] is not None and field in sim['fields']:
                values = sim['fields'][field][timestep].copy()
                if values.ndim >= 2:
                    values = np.linalg.norm(values, axis=tuple(range(1, values.ndim)))
                values = np.nan_to_num(values, nan=0.0)
                
                colormap = 'Inferno' if 'temp' in field.lower() else ('Plasma' if 'stress' in field.lower() else 'Viridis')
                
                if sim['triangles'] is not None and len(sim['triangles']) > 0:
                    fig = go.Figure(go.Mesh3d(
                        x=sim['points'][:,0], y=sim['points'][:,1], z=sim['points'][:,2], 
                        i=sim['triangles'][:,0], j=sim['triangles'][:,1], k=sim['triangles'][:,2], 
                        intensity=values, colorscale=colormap, intensitymode='vertex', 
                        colorbar=dict(title=field)
                    ))
                else:
                    fig = go.Figure(go.Scatter3d(
                        x=sim['points'][:,0], y=sim['points'][:,1], z=sim['points'][:,2], 
                        mode='markers', marker=dict(size=3, color=values, colorscale=colormap, 
                                                   colorbar=dict(title=field), showscale=True)
                    ))
                
                fig.update_layout(scene=dict(aspectmode="data"), 
                                title=f"{field} at Timestep {timestep+1}", 
                                height=600)
                # FIX: use_container_width → width='stretch'
                st.plotly_chart(fig, use_container_width=False, width='stretch')

    # ------------------------------------------------------------------------
    # TAB 2: Interpolation
    # ------------------------------------------------------------------------
    with tabs[1]:
        st.subheader("Run ST-DGPA Interpolation")
        
        c1, c2, c3 = st.columns(3)
        energies = [s['energy'] for s in st.session_state.summaries]
        min_e, max_e = min(energies), max(energies)
        durations = [s['duration'] for s in st.session_state.summaries]
        min_d, max_d = min(durations), max(durations)
        
        with c1: 
            q_E = st.number_input("Energy (mJ)", float(min_e*0.5), float(max_e*2.0), 
                                 float((min_e+max_e)/2), 0.1, key="interp_energy")
        with c2: 
            q_τ = st.number_input("Duration (ns)", float(min_d*0.5), float(max_d*2.0), 
                                 float((min_d+max_d)/2), 0.1, key="interp_duration")
        with c3: 
            max_t = st.number_input("Max Time (ns)", 1, 200, 50, 1, key="interp_maxtime")
        
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
                    st.session_state.interpolation_params = {
                        'energy_query': q_E, 
                        'duration_query': q_τ, 
                        'time_points': time_points
                    }
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
                            fig_preds.add_trace(go.Scatter(
                                x=params['time_points'], 
                                y=results['field_predictions'][f]['mean'], 
                                mode='lines', name=f, line=dict(width=2)
                            ))
                            if results['field_predictions'][f]['std']:
                                mean = np.array(results['field_predictions'][f]['mean'])
                                std = np.array(results['field_predictions'][f]['std'])
                                fig_preds.add_trace(go.Scatter(
                                    x=np.concatenate([params['time_points'], params['time_points'][::-1]]), 
                                    y=np.concatenate([mean+std, (mean-std)[::-1]]), 
                                    fill='toself', fillcolor=f'rgba(0,100,255,0.1)', 
                                    line=dict(color='rgba(255,255,255,0)'), showlegend=False, 
                                    name=f'{f} ± std'
                                ))
                    fig_preds.update_layout(
                        title="Predicted Mean Values with Confidence Bands", 
                        xaxis_title="Time (ns)", yaxis_title="Value", 
                        height=400, hovermode='x unified'
                    )
                    # FIX: use_container_width → width='stretch'
                    st.plotly_chart(fig_preds, use_container_width=False, width='stretch')
                
                if results['confidence_scores']:
                    fig_conf = st.session_state.enhanced_viz.create_confidence_plot(results, params['time_points'])
                    # FIX: use_container_width → width='stretch'
                    st.plotly_chart(fig_conf, use_container_width=False, width='stretch')

    # ------------------------------------------------------------------------
    # TAB 3: Polar Radar (FULLY EXPANDED WITH UNIT CONVERSION - FIXED)
    # ------------------------------------------------------------------------
    with tabs[2]:
        st.subheader("🎯 Polar Radar Visualization")
        st.markdown("""
        *Visualize simulation results in polar coordinates:*
        - **Angular axis (θ)**: Laser Energy (mJ) — clockwise from top
        - **Radial axis (r)**: Pulse Duration (ns) — outward from center  
        - **Marker color**: Peak field value (temperature/stress) — colorscale shows PHYSICAL units
        - **Marker size**: Scaled by peak value for visual emphasis
        - **Target marker**: Predicted query point (★) with PHYSICAL value display
        - **Hover**: Shows BOTH normalized (for debugging) and physical (for interpretation) values
        """)
        
        # === FIELD AND TIMESTEP SELECTION ===
        all_fields = set()
        for s in st.session_state.summaries:
            all_fields.update(s.get('field_stats', {}).keys())
        
        if not all_fields:
            st.warning("⚠️ No fields available. Please load simulation data first.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                field_type = st.selectbox(
                    "Field Type", 
                    sorted(all_fields), 
                    key="polar_field_select",
                    help="Select the physical field to visualize"
                )
            with col2:
                max_timestep = max(
                    (len(s.get('timesteps', [1])) for s in st.session_state.summaries), 
                    default=1
                )
                t_step = st.number_input(
                    "Timestep Index", 
                    min_value=1, 
                    max_value=max_timestep, 
                    value=1, 
                    step=1,
                    key="polar_timestep_select",
                    help="Select which simulation timestep to visualize"
                )
            
            # === ROBUST DATA COLLECTION WITH WARNINGS ===
            rows = []
            warnings_found = []
            missing_field_count = 0
            
            for s in st.session_state.summaries:
                field_stats = s.get('field_stats', {}).get(field_type, {})
                peak_list = field_stats.get('max', [])
                
                idx = t_step - 1
                if idx >= len(peak_list):
                    warnings_found.append(
                        f"• {s['name']}: Timestep {t_step} not available "
                        f"(only {len(peak_list)} timesteps recorded)"
                    )
                    continue
                
                peak_val = peak_list[idx]
                if not np.isfinite(peak_val):
                    peak_val = 0.0
                    warnings_found.append(
                        f"• {s['name']}: Peak value at t={t_step} was non-finite, set to 0"
                    )
                
                if field_type not in s.get('field_stats', {}):
                    missing_field_count += 1
                
                rows.append({
                    'Name': s['name'],
                    'Energy': float(s['energy']),
                    'Duration': float(s['duration']),
                    'Peak_Value': float(peak_val)
                })
            
            df_polar = pd.DataFrame(rows) if rows else pd.DataFrame(columns=['Name', 'Energy', 'Duration', 'Peak_Value'])
            
            # === DATA AVAILABILITY FEEDBACK ===
            if df_polar.empty:
                st.error(f"❌ No valid data for **`{field_type}`** at timestep **{t_step}**")
                if warnings_found or missing_field_count > 0:
                    with st.expander("🔍 Diagnostic Information", expanded=True):
                        if missing_field_count > 0:
                            st.warning(f"⚠️ `{field_type}` field not found in {missing_field_count} simulations")
                        if warnings_found:
                            st.markdown("**Data collection warnings:**")
                            for w in warnings_found[:15]:
                                st.caption(w)
                            if len(warnings_found) > 15:
                                st.caption(f"... and {len(warnings_found) - 15} more warnings")
            else:
                st.success(f"✅ Loaded **{len(df_polar)}** simulations for radar visualization")
                st.caption(
                    f"📊 Data ranges: "
                    f"Energy {df_polar['Energy'].min():.2f}–{df_polar['Energy'].max():.2f} mJ | "
                    f"Duration {df_polar['Duration'].min():.2f}–{df_polar['Duration'].max():.2f} ns | "
                    f"Peak {df_polar['Peak_Value'].min():.3f}–{df_polar['Peak_Value'].max():.3f}"
                )
                
                # === STYLING EXPANDER ===
                with st.expander("🎨 Chart Styling Options", expanded=True):
                    colA, colB, colC, colD = st.columns(4)
                    
                    with colA:
                        st.markdown("**📝 Title & Text**")
                        title_font_size = st.slider("Title Font Size", 12, 30, 18, key="radar_title_font_size")
                        margin_top = st.slider("Top Margin (px)", 40, 120, 80, key="radar_margin_t")
                    
                    with colB:
                        st.markdown("**🔢 Axis Ticks**")
                        radial_tick_font = st.slider("Radial Tick Size", 8, 20, 12, key="radar_radial_tick_font")
                        angular_tick_font = st.slider("Angular Tick Size", 8, 20, 12, key="radar_angular_tick_font")
                    
                    with colC:
                        st.markdown("**⊞ Grid Lines**")
                        radial_grid_width = st.slider("Radial Grid Width", 0.0, 3.0, 1.0, 0.1, key="radar_radial_grid_w")
                        angular_grid_width = st.slider("Angular Grid Width", 0.0, 3.0, 1.0, 0.1, key="radar_angular_grid_w")
                        grid_color = st.color_picker("Grid Color", "#d3d3d3", key="radar_grid_color")
                        show_grid = st.checkbox("Show Grid", value=True, key="radar_show_grid")
                    
                    with colD:
                        st.markdown("**🎯 Target Marker**")
                        highlight_target = st.checkbox("Show Target Query", value=True, key="radar_show_target")
                        target_size = st.slider("Target Size", 10, 40, 25, key="radar_target_size")
                        target_line_width = st.slider("Target Border Width", 1, 5, 3, key="radar_target_line_w")
                        color_target_by_value = st.checkbox("Color Target by Value", value=True, key="radar_color_target")
                
                # === DATA PROCESSING EXPANDER ===
                with st.expander("📐 Data Processing & Scaling", expanded=False):
                    colE, colF = st.columns(2)
                    
                    with colE:
                        st.markdown("**🌀 Energy Axis (Angular)**")
                        e_max_data = float(df_polar['Energy'].max())
                        e_min_data = float(df_polar['Energy'].min())
                        
                        use_custom_e_max = st.checkbox("Custom Energy Max", value=False, key="radar_custom_e_max")
                        custom_energy_max = st.number_input(
                            "Max Energy (mJ)", min_value=e_min_data, max_value=e_max_data * 3.0, 
                            value=float(e_max_data * 1.2), step=0.1, key="radar_e_max_input",
                            disabled=not use_custom_e_max
                        ) if use_custom_e_max else None
                        
                        use_custom_e_min = st.checkbox("Custom Energy Min", value=False, key="radar_custom_e_min")
                        custom_energy_min = st.number_input(
                            "Min Energy (mJ)", min_value=0.0, max_value=e_max_data, 
                            value=float(max(0, e_min_data * 0.8)), step=0.1, key="radar_e_min_input",
                            disabled=not use_custom_e_min
                        ) if use_custom_e_min else None
                        
                        add_jitter = st.checkbox("Add Jitter for Overlaps", value=True, key="radar_add_jitter")
                        jitter_amount = st.slider(
                            "Jitter Amount (±degrees)", 0.1, 10.0, 2.0, 0.1, key="radar_jitter_amt"
                        ) if add_jitter else 0.0
                        jitter_seed = st.number_input(
                            "Jitter Seed", min_value=0, max_value=9999, value=42, key="radar_jitter_seed"
                        ) if add_jitter else None
                    
                    with colF:
                        st.markdown("**🎨 Value Scaling**")
                        normalize_colors = st.checkbox("Normalize Colors to [0,1]", value=True, key="radar_norm_colors")
                        
                        if normalize_colors:
                            p_max_data = float(df_polar['Peak_Value'].max())
                            p_min_data = float(df_polar['Peak_Value'].min())
                            
                            use_custom_c_max = st.checkbox("Custom Color Max", value=False, key="radar_custom_c_max")
                            color_ref_max = st.number_input(
                                "Color Scale Max", min_value=p_min_data, max_value=p_max_data * 5.0, 
                                value=float(p_max_data * 1.2), step=0.01, key="radar_c_max_input",
                                disabled=not use_custom_c_max
                            ) if use_custom_c_max else None
                            
                            use_custom_c_min = st.checkbox("Custom Color Min", value=False, key="radar_custom_c_min")
                            color_ref_min = st.number_input(
                                "Color Scale Min", min_value=0.0, max_value=p_max_data, 
                                value=float(max(0, p_min_data * 0.8)), step=0.01, key="radar_c_min_input",
                                disabled=not use_custom_c_min
                            ) if use_custom_c_min else None
                        else:
                            color_ref_max = None
                            color_ref_min = None
                        
                        radial_auto = st.checkbox("Auto Radial Range", value=True, key="radar_radial_auto")
                        radial_axis_max = st.number_input(
                            "Max Duration (ns)", min_value=0.1, 
                            value=float(df_polar['Duration'].max() * 1.1), step=0.1, 
                            key="radar_r_max_input", disabled=radial_auto
                        ) if not radial_auto else None
                
                # === ADVANCED OPTIONS EXPANDER ===
                with st.expander("⚙️ Advanced Options", expanded=False):
                    colG, colH = st.columns(2)
                    
                    with colG:
                        st.markdown("**🔄 Plot Orientation**")
                        angle_offset = st.slider("Rotation Offset (degrees)", -180, 180, 0, 5, key="radar_angle_offset")
                        st.markdown("**🖼️ Background**")
                        bg_color = st.color_picker("Figure Background", "#ffffff", key="radar_bg")
                        polar_bg = st.color_picker("Polar Area Background", "#ffffff", key="radar_polar_bg")
                    
                    with colH:
                        st.markdown("**👆 Interactions**")
                        enable_hover = st.checkbox("Enable Hover", value=True, key="radar_enable_hover")
                        hover_mode = st.selectbox(
                            "Hover Mode", ["closest", "x", "y", "x unified", "y unified"],
                            index=0, key="radar_hover_mode"
                        ) if enable_hover else "closest"
                
                # === UNIT CONVERSION OPTIONS ===
                with st.expander("🔬 Unit Conversion & Thresholds", expanded=True):
                    colI, colJ = st.columns(2)
                    
                    with colI:
                        show_thresholds = st.checkbox(
                            "Show Critical Thresholds", value=True, key="radar_show_thresholds",
                            help="Display process control thresholds (e.g., solder melt temperature)"
                        )
                        show_confidence_bands = st.checkbox(
                            "Show Uncertainty Bands", value=True, key="radar_show_uncertainty",
                            help="Display ± uncertainty ranges around data points"
                        )
                    
                    with colJ:
                        # FIX: Access KNOWN_UNITS via class name, not instance
                        viz = st.session_state.polar_viz
                        unit = PolarRadarVisualizer.KNOWN_UNITS.get(field_type)
                        if unit:
                            st.info(f"**Unit**: {unit.name} ({unit.unit})")
                            if unit.valid_range:
                                st.caption(f"Valid range: {unit.format_value(unit.valid_range[0])} – {unit.format_value(unit.valid_range[1])}")
                            if unit.critical_thresholds:
                                st.caption(f"Thresholds: {', '.join([f'{k}: {unit.format_value(v)}' for k,v in unit.critical_thresholds.items()])}")
                        else:
                            st.caption("ℹ️ Using default unit settings")
                
                # === TARGET QUERY INTEGRATION ===
                query_params = None
                target_peak = None
                target_unc = None
                
                if st.session_state.get('polar_query') and st.session_state.get('interpolation_results'):
                    query_params = st.session_state.polar_query
                    results = st.session_state.interpolation_results
                    if field_type in results.get('field_predictions', {}):
                        preds = results['field_predictions'][field_type]
                        t_idx = t_step - 1
                        if 'max' in preds and t_idx < len(preds['max']):
                            target_peak = float(preds['max'][t_idx])
                            if 'std' in preds and t_idx < len(preds['std']):
                                target_unc = float(preds['std'][t_idx])
                            st.info(f"🎯 Target predicted peak for `{field_type}`: **{target_peak:.4f}**")
                
                # === CREATE AND DISPLAY CHART ===
                viz = st.session_state.polar_viz
                
                fig_polar = viz.create_polar_radar_chart(
                    df=df_polar,
                    field_type=field_type,
                    query_params=query_params if highlight_target else None,
                    timestep=t_step,
                    width=900,
                    height=750,
                    show_legend=True,
                    marker_size_range=(10, 30),
                    # Styling
                    radial_grid_width=radial_grid_width if show_grid else 0,
                    angular_grid_width=angular_grid_width if show_grid else 0,
                    radial_tick_font_size=radial_tick_font,
                    angular_tick_font_size=angular_tick_font,
                    title_font_size=title_font_size,
                    margin_pad={'l': 60, 'r': 80, 't': margin_top, 'b': 60},
                    # Scaling
                    energy_min=custom_energy_min,
                    energy_max=custom_energy_max,
                    target_peak_value=target_peak if (highlight_target and color_target_by_value) else None,
                    # Processing
                    add_jitter=add_jitter,
                    jitter_amount=jitter_amount,
                    jitter_seed=jitter_seed,
                    normalize_colors=normalize_colors,
                    color_reference_max=color_ref_max,
                    color_reference_min=color_ref_min,
                    radial_axis_max=radial_axis_max,
                    # Advanced
                    angle_offset_deg=angle_offset,
                    background_color=bg_color,
                    polar_background_color=polar_bg,
                    enable_hover=enable_hover,
                    hover_mode=hover_mode,
                    target_marker_size=target_size,
                    target_marker_line_width=target_line_width,
                    highlight_target=highlight_target,
                    show_grid=show_grid,
                    show_radial_labels=True,
                    show_angular_labels=True,
                    # NEW: Unit conversion
                    show_thresholds=show_thresholds,
                    show_confidence_bands=show_confidence_bands,
                    target_uncertainty=target_unc
                )
                
                # FIX: use_container_width → width='stretch'
                st.plotly_chart(fig_polar, use_container_width=False, width='stretch')
                
                # === MULTI-TIMESTEP COMPARISON ===
                if st.checkbox("📊 Enable Multi-Timestep Comparison", key="polar_enable_multi"):
                    st.markdown("##### Compare Evolution Across Timesteps")
                    
                    available_ts = list(range(1, min(max_timestep + 1, 7)))
                    selected_ts = st.multiselect(
                        "Select Timesteps", available_ts,
                        default=[1, min(3, max_timestep), min(5, max_timestep)] if max_timestep >= 3 else available_ts[:min(3, len(available_ts))],
                        key="polar_multi_ts_select"
                    )
                    
                    if selected_ts:
                        n_cols = min(len(selected_ts), 3)
                        rows = (len(selected_ts) + n_cols - 1) // n_cols
                        
                        for row in range(rows):
                            cols = st.columns(n_cols)
                            for col_idx in range(n_cols):
                                ts_idx = row * n_cols + col_idx
                                if ts_idx >= len(selected_ts):
                                    break
                                
                                ts = selected_ts[ts_idx]
                                
                                # Build data for this timestep
                                rows_ts = []
                                for s in st.session_state.summaries:
                                    if ts <= len(s.get('timesteps', [])):
                                        field_stats = s.get('field_stats', {}).get(field_type, {})
                                        peak_list = field_stats.get('max', [0])
                                        if ts - 1 < len(peak_list):
                                            pv = peak_list[ts - 1]
                                            rows_ts.append({
                                                'Name': s['name'],
                                                'Energy': s['energy'],
                                                'Duration': s['duration'],
                                                'Peak_Value': float(pv) if np.isfinite(pv) else 0.0
                                            })
                                
                                if not rows_ts:
                                    with cols[col_idx]:
                                        st.caption(f"Timestep {ts}: No data")
                                    continue
                                
                                df_ts = pd.DataFrame(rows_ts)
                                
                                # Get target value for this timestep
                                target_val_ts = None
                                target_unc_ts = None
                                if query_params and st.session_state.interpolation_results:
                                    res = st.session_state.interpolation_results
                                    if field_type in res.get('field_predictions', {}):
                                        preds = res['field_predictions'][field_type]
                                        if 'max' in preds and ts - 1 < len(preds['max']):
                                            target_val_ts = float(preds['max'][ts - 1])
                                        if 'std' in preds and ts - 1 < len(preds['std']):
                                            target_unc_ts = float(preds['std'][ts - 1])
                                
                                # Create subplot
                                fig_ts = viz.create_polar_radar_chart(
                                    df_ts, field_type, 
                                    query_params=query_params if highlight_target else None,
                                    timestep=ts,
                                    width=320,
                                    height=320,
                                    show_legend=(ts == selected_ts[0]),
                                    marker_size_range=(8, 20),
                                    title_font_size=14,
                                    radial_tick_font_size=10,
                                    angular_tick_font_size=10,
                                    energy_min=custom_energy_min,
                                    energy_max=custom_energy_max,
                                    target_peak_value=target_val_ts if (highlight_target and color_target_by_value) else None,
                                    add_jitter=add_jitter,
                                    jitter_amount=jitter_amount * 0.7,
                                    jitter_seed=jitter_seed,
                                    normalize_colors=normalize_colors,
                                    color_reference_max=color_ref_max,
                                    color_reference_min=color_ref_min,
                                    radial_axis_max=radial_axis_max,
                                    angle_offset_deg=angle_offset,
                                    background_color=bg_color,
                                    polar_background_color=polar_bg,
                                    enable_hover=enable_hover,
                                    hover_mode=hover_mode,
                                    target_marker_size=target_size,
                                    target_marker_line_width=target_line_width,
                                    highlight_target=highlight_target,
                                    show_grid=show_grid,
                                    show_thresholds=show_thresholds,
                                    show_confidence_bands=show_confidence_bands,
                                    target_uncertainty=target_unc_ts
                                )
                                
                                with cols[col_idx]:
                                    # FIX: use_container_width → width='stretch'
                                    st.plotly_chart(fig_ts, use_container_width=False, width='stretch')
                                    st.caption(f"**t = {ts}** • {len(df_ts)} sims")
                
                # === EXPORT BUTTONS ===
                with st.expander("💾 Export Chart", expanded=False):
                    colX, colY = st.columns(2)
                    export_format = st.selectbox(
                        "Export Format", ["PNG", "SVG", "HTML", "JSON"], index=0, key="radar_export_fmt"
                    )
                    
                    with colX:
                        if st.button(f"Export as {export_format}", key="radar_export_btn"):
                            try:
                                if export_format == "PNG":
                                    img_bytes = fig_polar.to_image(format="png", width=1200, height=900, scale=2)
                                    st.download_button(
                                        label="⬇️ Download PNG", data=img_bytes,
                                        file_name=f"polar_radar_{field_type}_t{t_step}.png", mime="image/png"
                                    )
                                elif export_format == "SVG":
                                    svg_bytes = fig_polar.to_image(format="svg", width=1200, height=900)
                                    st.download_button(
                                        label="⬇️ Download SVG", data=svg_bytes,
                                        file_name=f"polar_radar_{field_type}_t{t_step}.svg", mime="image/svg+xml"
                                    )
                                elif export_format == "HTML":
                                    html_str = fig_polar.to_html(include_plotlyjs='cdn')
                                    st.download_button(
                                        label="⬇️ Download HTML", data=html_str,
                                        file_name=f"polar_radar_{field_type}_t{t_step}.html", mime="text/html"
                                    )
                                elif export_format == "JSON":
                                    # Use enhanced export with dual-unit preservation
                                    viz = st.session_state.polar_viz
                                    json_str = viz.export_chart_data(fig_polar, df_polar, format='json')
                                    st.download_button(
                                        label="⬇️ Download JSON (with units)", data=json_str,
                                        file_name=f"polar_radar_{field_type}_t{t_step}_with_units.json", 
                                        mime="application/json"
                                    )
                                st.success("✅ Export ready!")
                            except Exception as e:
                                st.error(f"❌ Export failed: {e}")
                                st.info("💡 Tip: Install `kaleido` for image export: `pip install -U kaleido`")
                    
                    with colY:
                        if st.button("Copy Plotly Config", key="radar_copy_config"):
                            config_json = fig_polar.to_plotly_json()
                            st.code(json.dumps(config_json, indent=2)[:500] + "...", language="json")
                            st.caption("↑ First 500 chars of Plotly figure config")

    # ------------------------------------------------------------------------
    # TAB 4: ST-DGPA Analysis
    # ------------------------------------------------------------------------
    with tabs[3]:
        st.subheader("🧠 ST-DGPA Attention & Physics Analysis")
        
        if st.session_state.interpolation_results and st.session_state.get('interpolation_params'):
            res = st.session_state.interpolation_results
            params = st.session_state.interpolation_params
            
            fig_stdgpa = st.session_state.enhanced_viz.create_stdgpa_analysis(
                res, params['energy_query'], params['duration_query'], params['time_points']
            )
            
            if fig_stdgpa:
                # FIX: use_container_width → width='stretch'
                st.plotly_chart(fig_stdgpa, use_container_width=False, width='stretch')
                
                with st.expander("💾 Export Analysis"):
                    if st.button("Export as PNG"):
                        img_bytes = st.session_state.export_manager.export_plotly_figure(fig_stdgpa, format='png')
                        if img_bytes:
                            st.download_button(
                                label="⬇️ Download PNG", data=img_bytes, 
                                file_name=f"stdgpa_analysis_{params['energy_query']:.1f}mJ.png", 
                                mime="image/png"
                            )
            else:
                st.info("ℹ️ No attention data available for analysis.")
        else:
            st.info("ℹ️ Please run an interpolation first (Tab 2) to see ST-DGPA analysis.")

    # ------------------------------------------------------------------------
    # TAB 5: Export
    # ------------------------------------------------------------------------
    with tabs[4]:
        st.subheader("💾 Export Results")
        
        if st.session_state.interpolation_results and st.session_state.get('interpolation_params'):
            results = st.session_state.interpolation_results
            params = st.session_state.interpolation_params
            
            st.markdown("### Export Prediction Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                export_data = {
                    'metadata': {
                        'generated_at': datetime.now().isoformat(), 
                        'query_params': params, 
                        'stdgpa_params': {
                            'sigma_g': st.session_state.extrapolator.sigma_g, 
                            's_E': st.session_state.extrapolator.s_E, 
                            's_tau': st.session_state.extrapolator.s_tau, 
                            's_t': st.session_state.extrapolator.s_t, 
                            'temporal_weight': st.session_state.extrapolator.temporal_weight
                        }
                    }, 
                    'results': results
                }
                json_str, json_name = st.session_state.export_manager.export_to_json(export_data)
                st.download_button(
                    label="📥 Download as JSON", data=json_str, 
                    file_name=json_name, mime="application/json", use_container_width=True
                )
            
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
                        if csv_str:
                            st.download_button(
                                label="📥 Download as CSV", data=csv_str, 
                                file_name=csv_name, mime="text/csv", use_container_width=True
                            )
                        else:
                            st.warning("⚠️ No valid CSV data to export.")
                    else:
                        st.info("ℹ️ No prediction data available for CSV export.")
            
            with col3:
                if results['attention_maps']:
                    attention_df = pd.DataFrame(results['attention_maps'])
                    att_csv, att_name = st.session_state.export_manager.export_to_csv(attention_df, filename="attention_weights.csv")
                    if att_csv:
                        st.download_button(
                            label="📥 Download Attention Weights", data=att_csv, 
                            file_name=att_name, mime="text/csv", use_container_width=True
                        )
                    else:
                        st.warning("⚠️ No attention weights to export.")
                else:
                    st.info("ℹ️ No attention maps available.")
            
            # === NEW: Physical Unit Conversion Demo ===
            st.markdown("---")
            st.subheader("🔬 Prediction → Physical Unit Converter")
            st.caption("Convert normalized model outputs to actionable engineering values")
            
            demo_field = st.selectbox("Select Field for Demo", list(PolarRadarVisualizer.KNOWN_UNITS.keys()), key="demo_field")
            demo_norm = st.slider("Normalized Prediction [0,1]", 0.0, 1.0, 0.68, 0.01, key="demo_norm")
            demo_unc = st.slider("Normalized Uncertainty", 0.0, 0.2, 0.04, 0.005, key="demo_unc")
            
            # Get refs from current interpolation or use defaults
            ref_min, ref_max = 400.0, 700.0  # Default for temperature
            if st.session_state.interpolation_results and demo_field in st.session_state.interpolation_results.get('field_predictions', {}):
                preds = st.session_state.interpolation_results['field_predictions'][demo_field]
                if preds['mean']:
                    ref_min = min(preds['mean']) * 0.9
                    ref_max = max(preds['mean']) * 1.1
            
            if st.button("🔄 Convert to Physical Units", key="demo_convert"):
                result = convert_prediction_to_physical(
                    normalized_prediction=demo_norm,
                    field_type=demo_field,
                    ref_min=ref_min,
                    ref_max=ref_max,
                    uncertainty_normalized=demo_unc
                )
                
                # Display results in metric cards
                colA, colB, colC = st.columns(3)
                with colA:
                    st.metric("Physical Value", result['formatted'], 
                             delta=None if result['is_valid'] else "⚠️ Invalid")
                with colB:
                    if 'formatted_with_unc' in result:
                        st.metric("With Uncertainty", result['formatted_with_unc'])
                    else:
                        st.metric("Uncertainty", "N/A")
                with colC:
                    status = "✅ Valid" if result['is_valid'] else "❌ Invalid"
                    st.metric("Validity", status)
                
                if result.get('warnings'):
                    for w in result['warnings']:
                        st.warning(f"⚠️ {w}")
                
                if result.get('threshold_status'):
                    st.markdown("**Threshold Status:**")
                    for thresh, status in result['threshold_status'].items():
                        if 'EXCEEDED' in status:
                            st.error(f"🔴 {thresh}: {status}")
                        else:
                            st.success(f"🟢 {thresh}: {status}")
                
                with st.expander("📋 Full Conversion Details"):
                    st.json(result)
        else:
            st.info("ℹ️ Please run an interpolation first (Tab 2) to enable export.")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
