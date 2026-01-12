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
from scipy.interpolate import griddata, RBFInterpolator, PchipInterpolator
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
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.integrate import solve_ivp
import cvxpy as cp

warnings.filterwarnings('ignore')

# =============================================
# PHYSICS CONSTANTS AND MATERIAL PROPERTIES
# =============================================
class PhysicsConstants:
    """Physics constants for solder material (Sn-3.0Ag-0.5Cu)"""
    # Material properties
    THERMAL_DIFFUSIVITY = 1.0e-5  # m²/s for SAC305
    THERMAL_CONDUCTIVITY = 64.0  # W/m·K
    DENSITY = 7390.0  # kg/m³
    SPECIFIC_HEAT = 226.0  # J/kg·K
    MELTING_TEMPERATURE = 217.0 + 273.15  # K (490.15 K)
    
    # Thermo-mechanical properties
    COEFFICIENT_THERMAL_EXPANSION = 22.0e-6  # 1/K
    YOUNGS_MODULUS = 51.0e9  # Pa
    POISSON_RATIO = 0.3
    YIELD_STRENGTH = 40.0e6  # Pa
    
    # Physical constants
    STEFAN_BOLTZMANN = 5.670374419e-8  # W/m²·K⁴
    AMBIENT_TEMPERATURE = 298.15  # K (25°C)
    
    # Characteristic lengths
    LASER_SPOT_RADIUS = 50e-6  # m (50 μm)
    BALL_RADIUS = 100e-6  # m (100 μm)
    
    @staticmethod
    def calculate_diffusion_time(characteristic_length: float) -> float:
        """Calculate characteristic diffusion time τ = L²/α"""
        return characteristic_length**2 / PhysicsConstants.THERMAL_DIFFUSIVITY
    
    @staticmethod
    def calculate_thermal_relaxation_time():
        """Calculate thermal relaxation time for laser heating"""
        return PhysicsConstants.LASER_SPOT_RADIUS**2 / (4 * PhysicsConstants.THERMAL_DIFFUSIVITY)

# =============================================
# ENHANCED PHYSICS-BASED EXTRAPOLATOR
# =============================================
class EnhancedPhysicsInformedAttentionExtrapolator:
    """Advanced extrapolator with physics-aware embeddings and multi-head attention
    Extended with physics-based constraints for temporal extrapolation to ms scale"""
    
    def __init__(self, sigma_param=0.3, spatial_weight=0.5, n_heads=4, temperature=1.0,
                 physics_enforcement=True, decay_model='exponential', enforce_monotonicity=True):
        self.sigma_param = sigma_param
        self.spatial_weight = spatial_weight
        self.n_heads = n_heads
        self.temperature = temperature
        self.physics_enforcement = physics_enforcement
        self.decay_model = decay_model  # 'exponential', 'diffusive', 'hybrid'
        self.enforce_monotonicity = enforce_monotonicity
        
        self.source_db = []
        self.embedding_scaler = StandardScaler()
        self.value_scaler = StandardScaler()
        self.source_embeddings = []
        self.source_values = []
        self.source_metadata = []
        self.fitted = False
        
        # Physics parameters
        self.physics_constants = PhysicsConstants()
        self.tau_diff = self.physics_constants.calculate_diffusion_time(
            self.physics_constants.BALL_RADIUS
        )  # Characteristic diffusion time (~ms)
        
        # POD components for hybrid extrapolation
        self.pod_modes = None
        self.pod_coeffs = None
        self.pod_eigenvalues = None
        
    def load_summaries(self, summaries, enable_pod=True):
        """Load summary statistics and prepare for attention mechanism
        Optionally compute POD modes for hybrid extrapolation"""
        self.source_db = summaries
        
        if not summaries:
            return
        
        # Prepare embeddings and values
        all_embeddings = []
        all_values = []
        metadata = []
        
        # Stack all field data for POD if enabled
        field_data_stack = []
        
        for summary_idx, summary in enumerate(summaries):
            for timestep_idx, t in enumerate(summary['timesteps']):
                # Compute physics-aware embedding
                emb = self._compute_physics_embedding(
                    summary['energy'], 
                    summary['duration'], 
                    t
                )
                all_embeddings.append(emb)
                
                # Extract field values (flattened)
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
                
                # Store metadata for spatial correlations
                metadata.append({
                    'summary_idx': summary_idx,
                    'timestep_idx': timestep_idx,
                    'energy': summary['energy'],
                    'duration': summary['duration'],
                    'time': t,
                    'name': summary['name']
                })
                
                # Collect field data for POD
                if enable_pod:
                    field_vector = []
                    for field in sorted(summary['field_stats'].keys()):
                        if timestep_idx < len(summary['field_stats'][field]['mean']):
                            field_vector.append(summary['field_stats'][field]['mean'][timestep_idx])
                            field_vector.append(summary['field_stats'][field]['max'][timestep_idx])
                    if field_vector:
                        field_data_stack.append(field_vector)
        
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
            
            # Compute POD modes if enabled
            if enable_pod and len(field_data_stack) > 0:
                self._compute_pod_modes(np.array(field_data_stack).T)
            
            st.info(f"✅ Prepared {len(all_embeddings)} embeddings with {all_embeddings.shape[1]} features")
    
    def _compute_pod_modes(self, data_matrix):
        """Compute Proper Orthogonal Decomposition modes for hybrid extrapolation"""
        # Center the data
        data_mean = np.mean(data_matrix, axis=1, keepdims=True)
        data_centered = data_matrix - data_mean
        
        # Compute SVD
        U, S, Vt = np.linalg.svd(data_centered, full_matrices=False)
        
        # Store POD components
        self.pod_modes = U  # Spatial modes
        self.pod_coeffs = np.diag(S) @ Vt  # Temporal coefficients
        self.pod_eigenvalues = S**2 / (data_matrix.shape[1] - 1)  # Energy content
        self.pod_mean = data_mean
        
        # Compute decay rates from eigenvalues (assuming diffusion-dominated)
        alpha = self.physics_constants.THERMAL_DIFFUSIVITY
        char_length = self.physics_constants.BALL_RADIUS
        self.pod_decay_rates = self.pod_eigenvalues / (char_length**2 / alpha)
    
    def _compute_physics_embedding(self, energy, duration, time):
        """Compute comprehensive physics-aware embedding with dimensionless numbers"""
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
        
        # Dimensionless numbers
        fourier_number = time * self.physics_constants.THERMAL_DIFFUSIVITY / (
            self.physics_constants.LASER_SPOT_RADIUS**2
        )
        
        # Post-pulse indicator
        is_post_pulse = 1.0 if time > duration else 0.0
        
        # Combined features
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
            np.log1p(power),
            np.log1p(time),
            fourier_number,
            is_post_pulse
        ], dtype=np.float32)
    
    def _apply_physics_constraints(self, predictions, time_points, duration_query, 
                                   energy_query, field_name):
        """Apply physics-based constraints to predictions"""
        if not self.physics_enforcement:
            return predictions
        
        delta_pulse = duration_query
        T_ambient = self.physics_constants.AMBIENT_TEMPERATURE
        tau_diff = self.tau_diff
        
        # Convert predictions to numpy array for manipulation
        mean_vals = np.array(predictions['mean'])
        max_vals = np.array(predictions['max'])
        
        # Find post-pulse indices
        post_pulse_idx = np.where(time_points > delta_pulse)[0]
        
        if len(post_pulse_idx) == 0:
            return predictions
        
        # Get peak values at pulse end
        pulse_end_idx = np.argmin(np.abs(time_points - delta_pulse))
        T_peak_mean = mean_vals[pulse_end_idx]
        T_peak_max = max_vals[pulse_end_idx]
        
        # Apply different decay models based on field type
        if 'temperature' in field_name.lower():
            for idx in post_pulse_idx:
                t_rel = time_points[idx] - delta_pulse
                
                if self.decay_model == 'exponential':
                    # Exponential decay: T(t) = T_amb + (T_peak - T_amb) * exp(-t/τ)
                    decay_factor = np.exp(-t_rel / tau_diff)
                elif self.decay_model == 'diffusive':
                    # Diffusive decay: 1/sqrt(t) behavior for point source
                    decay_factor = 1.0 / np.sqrt(1.0 + t_rel / (tau_diff / 10))
                elif self.decay_model == 'hybrid':
                    # Hybrid: exponential for short times, power law for long
                    if t_rel < tau_diff:
                        decay_factor = np.exp(-t_rel / tau_diff)
                    else:
                        decay_factor = np.sqrt(tau_diff / t_rel)
                else:
                    decay_factor = np.exp(-t_rel / tau_diff)
                
                # Apply decay
                mean_vals[idx] = T_ambient + (T_peak_mean - T_ambient) * decay_factor
                max_vals[idx] = T_ambient + (T_peak_max - T_ambient) * decay_factor
        
        # Enforce monotonicity for post-pulse temperature decay
        if self.enforce_monotonicity and 'temperature' in field_name.lower():
            for i in range(1, len(post_pulse_idx)):
                current_idx = post_pulse_idx[i]
                prev_idx = post_pulse_idx[i-1]
                
                # Temperature should not increase after pulse
                mean_vals[current_idx] = min(mean_vals[current_idx], mean_vals[prev_idx])
                max_vals[current_idx] = min(max_vals[current_idx], max_vals[prev_idx])
        
        # Apply bounds
        mean_vals = np.clip(mean_vals, T_ambient, self.physics_constants.MELTING_TEMPERATURE)
        max_vals = np.clip(max_vals, T_ambient, self.physics_constants.MELTING_TEMPERATURE)
        
        predictions['mean'] = mean_vals.tolist()
        predictions['max'] = max_vals.tolist()
        
        return predictions
    
    def _compute_stress_from_temperature(self, temperature_series, time_points):
        """Compute thermo-mechanical stress from temperature evolution"""
        alpha = self.physics_constants.COEFFICIENT_THERMAL_EXPANSION
        E = self.physics_constants.YOUNGS_MODULUS
        nu = self.physics_constants.POISSON_RATIO
        sigma_yield = self.physics_constants.YIELD_STRENGTH
        
        # Convert to numpy arrays
        T = np.array(temperature_series)
        t = np.array(time_points)
        
        # Reference temperature (ambient)
        T_ref = self.physics_constants.AMBIENT_TEMPERATURE
        
        # Thermal strain
        epsilon_thermal = alpha * (T - T_ref)
        
        # Thermal stress (assuming constrained expansion)
        sigma_thermal = E * epsilon_thermal / (1 - nu)
        
        # Apply relaxation for long times (simplified viscoelastic model)
        relaxation_time = self.tau_diff * 0.1  # Stress relaxs faster than heat diffusion
        for i in range(1, len(sigma_thermal)):
            sigma_thermal[i] = sigma_thermal[i-1] + (sigma_thermal[i] - sigma_thermal[i-1]) * \
                              np.exp(-(t[i] - t[i-1]) / relaxation_time)
        
        # Clip to yield strength
        sigma_thermal = np.clip(sigma_thermal, -sigma_yield, sigma_yield)
        
        return sigma_thermal.tolist()
    
    def _multi_head_attention(self, query_embedding, query_meta, 
                             apply_physics_penalty=True, lambda_phys=0.1):
        """Multi-head attention mechanism with physics-based regularization"""
        if not self.fitted or len(self.source_embeddings) == 0:
            return None, None
        
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
            
            # Apply physics-based penalty
            if apply_physics_penalty and self.physics_enforcement:
                physics_penalty = self._compute_physics_penalty(query_meta, self.source_metadata)
                scores = scores * (1 - lambda_phys * physics_penalty)
            
            head_weights[head] = scores
        
        # Combine head weights
        avg_weights = np.mean(head_weights, axis=0)
        
        # Apply temperature scaling
        if self.temperature != 1.0:
            avg_weights = avg_weights ** (1.0 / self.temperature)
        
        # Softmax normalization
        max_weight = np.max(avg_weights)
        exp_weights = np.exp(avg_weights - max_weight)
        attention_weights = exp_weights / (np.sum(exp_weights) + 1e-12)
        
        # Weighted prediction
        if len(self.source_values) > 0:
            prediction = np.sum(attention_weights[:, np.newaxis] * self.source_values, axis=0)
        else:
            prediction = np.zeros(1)
        
        return prediction, attention_weights
    
    def _compute_physics_penalty(self, query_meta, source_metadata):
        """Compute physics-based penalty for attention weights"""
        penalties = np.zeros(len(source_metadata))
        
        for idx, source_meta in enumerate(source_metadata):
            penalty = 0.0
            
            # Penalize sources with different pulse state
            if query_meta['time'] > query_meta['duration'] and source_meta['time'] <= source_meta['duration']:
                penalty += 0.5  # Source is during pulse, query is after
            
            # Penalize large time scale mismatches
            time_ratio = query_meta['time'] / max(source_meta['time'], 1e-6)
            if time_ratio > 100 or time_ratio < 0.01:
                penalty += 0.3
            
            # Penalize energy mismatches for post-pulse cooling
            if query_meta['time'] > query_meta['duration']:
                energy_ratio = query_meta['energy'] / max(source_meta['energy'], 1e-6)
                if energy_ratio > 2 or energy_ratio < 0.5:
                    penalty += 0.2
            
            penalties[idx] = penalty
        
        return np.clip(penalties, 0, 1)
    
    def predict_time_series(self, energy_query, duration_query, time_points_ms):
        """Predict over a series of time points with physics constraints
        
        Args:
            energy_query (float): Laser energy in mJ
            duration_query (float): Pulse duration in ns
            time_points_ms (array): Time points in ms (can range from ns to ms)
        """
        # Convert time points to consistent units (ns for compatibility with training)
        # Training data is in ns, but we can extrapolate to ms
        time_points_ns = np.array(time_points_ms) * 1e6  # Convert ms to ns
        
        results = {
            'time_points_ms': time_points_ms,
            'time_points_ns': time_points_ns.tolist(),
            'field_predictions': {},
            'attention_maps': [],
            'confidence_scores': [],
            'physics_constraints_applied': self.physics_enforcement
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
        
        # Phase 1: Predict using attention mechanism
        for t_ns in time_points_ns:
            pred = self._predict_single_timepoint(energy_query, duration_query, t_ns)
            
            if pred and 'field_predictions' in pred:
                for field in pred['field_predictions']:
                    if field in results['field_predictions']:
                        stats = pred['field_predictions'][field]
                        results['field_predictions'][field]['mean'].append(stats['mean'])
                        results['field_predictions'][field]['max'].append(stats['max'])
                        results['field_predictions'][field]['std'].append(stats['std'])
                
                results['attention_maps'].append(pred['attention_weights'])
                results['confidence_scores'].append(pred['confidence'])
            else:
                # Fill with NaN if prediction failed
                for field in results['field_predictions']:
                    results['field_predictions'][field]['mean'].append(np.nan)
                    results['field_predictions'][field]['max'].append(np.nan)
                    results['field_predictions'][field]['std'].append(np.nan)
                results['attention_maps'].append(np.array([]))
                results['confidence_scores'].append(0.0)
        
        # Phase 2: Apply physics-based constraints
        if self.physics_enforcement:
            results = self._apply_global_physics_constraints(
                results, time_points_ns, duration_query, energy_query
            )
        
        # Phase 3: Compute derived quantities (stress, strain) if temperature available
        if 'temperature' in results['field_predictions']:
            results = self._compute_thermomechanical_quantities(results, time_points_ns)
        
        return results
    
    def _predict_single_timepoint(self, energy_query, duration_query, time_ns):
        """Predict field statistics for a single time point"""
        if not self.fitted:
            return None
        
        # Compute query embedding and metadata
        query_embedding = self._compute_physics_embedding(energy_query, duration_query, time_ns)
        query_meta = {
            'energy': energy_query,
            'duration': duration_query,
            'time': time_ns,
            'is_post_pulse': 1.0 if time_ns > duration_query else 0.0
        }
        
        # Apply attention mechanism with physics penalty for long times
        apply_physics_penalty = time_ns > duration_query * 10  # Apply penalty for times >> pulse duration
        prediction, attention_weights = self._multi_head_attention(
            query_embedding, query_meta, 
            apply_physics_penalty=apply_physics_penalty,
            lambda_phys=0.2
        )
        
        if prediction is None:
            return None
        
        # Reconstruct field predictions
        result = {
            'prediction': prediction,
            'attention_weights': attention_weights,
            'confidence': float(np.max(attention_weights)) if len(attention_weights) > 0 else 0.0,
            'field_predictions': {}
        }
        
        # Map predictions back to fields
        if self.source_db:
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
    
    def _apply_global_physics_constraints(self, results, time_points_ns, 
                                         duration_query, energy_query):
        """Apply global physics constraints to all field predictions"""
        for field_name, predictions in results['field_predictions'].items():
            # Apply field-specific constraints
            predictions = self._apply_physics_constraints(
                predictions, time_points_ns, duration_query, energy_query, field_name
            )
            
            # Apply smoothness constraint for long-time extrapolation
            if len(predictions['mean']) > 3:
                predictions = self._apply_smoothness_constraint(predictions, time_points_ns)
            
            results['field_predictions'][field_name] = predictions
        
        return results
    
    def _apply_smoothness_constraint(self, predictions, time_points):
        """Apply smoothness constraint using monotonic interpolation"""
        mean_vals = np.array(predictions['mean'])
        max_vals = np.array(predictions['max'])
        
        # Use PCHIP interpolation for monotonic smoothing
        if len(mean_vals) > 3:
            # Convert to log-time for better interpolation across scales
            log_time = np.log10(time_points + 1e-6)
            
            # Create PCHIP interpolator
            interp_mean = PchipInterpolator(log_time, mean_vals)
            interp_max = PchipInterpolator(log_time, max_vals)
            
            # Resample on finer grid and interpolate back
            fine_log_time = np.linspace(log_time[0], log_time[-1], len(log_time) * 2)
            smooth_mean = interp_mean(fine_log_time)
            smooth_max = interp_max(fine_log_time)
            
            # Interpolate back to original time points
            predictions['mean'] = np.interp(log_time, fine_log_time, smooth_mean).tolist()
            predictions['max'] = np.interp(log_time, fine_log_time, smooth_max).tolist()
        
        return predictions
    
    def _compute_thermomechanical_quantities(self, results, time_points_ns):
        """Compute thermo-mechanical quantities from temperature predictions"""
        if 'temperature' not in results['field_predictions']:
            return results
        
        # Extract temperature evolution
        T_mean = results['field_predictions']['temperature']['mean']
        T_max = results['field_predictions']['temperature']['max']
        
        # Compute thermal stress
        stress_mean = self._compute_stress_from_temperature(T_mean, time_points_ns)
        stress_max = self._compute_stress_from_temperature(T_max, time_points_ns)
        
        # Compute thermal strain
        alpha = self.physics_constants.COEFFICIENT_THERMAL_EXPANSION
        T_ref = self.physics_constants.AMBIENT_TEMPERATURE
        
        strain_mean = [alpha * (T - T_ref) for T in T_mean]
        strain_max = [alpha * (T - T_ref) for T in T_max]
        
        # Add to results
        results['field_predictions']['thermal_stress'] = {
            'mean': stress_mean,
            'max': stress_max,
            'std': [0.0] * len(stress_mean)  # Placeholder
        }
        
        results['field_predictions']['thermal_strain'] = {
            'mean': strain_mean,
            'max': strain_max,
            'std': [0.0] * len(strain_mean)  # Placeholder
        }
        
        return results
    
    def interpolate_full_field(self, field_name, attention_weights, source_metadata, 
                              simulations, time_point_ns, duration_query):
        """Compute interpolated full field using attention weights with physics constraints"""
        if not self.fitted or len(attention_weights) == 0:
            return None
        
        # Get interpolated field using base method
        interpolated_field = super().interpolate_full_field(
            field_name, attention_weights, source_metadata, simulations
        )
        
        if interpolated_field is None:
            return None
        
        # Apply physics-based post-processing if after pulse
        if self.physics_enforcement and time_point_ns > duration_query:
            interpolated_field = self._apply_field_level_physics(
                interpolated_field, field_name, time_point_ns, duration_query
            )
        
        return interpolated_field
    
    def _apply_field_level_physics(self, field_values, field_name, time_ns, duration_ns):
        """Apply physics constraints at field level"""
        if 'temperature' in field_name.lower():
            # Apply diffusive smoothing for post-pulse temperature
            T_ambient = self.physics_constants.AMBIENT_TEMPERATURE
            
            # Estimate characteristic length from field variation
            if hasattr(self, 'last_interpolation_metadata') and 'spatial_gradient' in self.last_interpolation_metadata:
                grad = self.last_interpolation_metadata['spatial_gradient']
                decay_length = 1.0 / (np.mean(np.abs(grad)) + 1e-6)
            else:
                decay_length = self.physics_constants.LASER_SPOT_RADIUS
            
            # Apply spatial decay for far-field
            if field_values.ndim == 1:  # Scalar field
                # Simple Gaussian smoothing based on diffusion
                t_diff = (time_ns - duration_ns) * 1e-9  # Convert to seconds
                if t_diff > 0:
                    # Estimate diffusion length
                    diffusion_length = np.sqrt(4 * self.physics_constants.THERMAL_DIFFUSIVITY * t_diff)
                    
                    # Apply mild smoothing (simulated by convolution with small kernel)
                    # This is a simplified approach - in practice, would solve diffusion equation
                    kernel_size = min(5, int(len(field_values) * 0.1))
                    if kernel_size > 1:
                        kernel = np.exp(-np.linspace(-2, 2, kernel_size)**2)
                        kernel = kernel / kernel.sum()
                        field_values = np.convolve(field_values, kernel, mode='same')
            
            # Ensure ambient temperature far from heat source
            field_values = np.clip(field_values, T_ambient, None)
        
        return field_values

# =============================================
# MULTI-SCALE TEMPORAL EXTRAPOLATION ENGINE
# =============================================
class MultiScaleTemporalExtrapolator:
    """Handles temporal extrapolation across multiple scales (ns to ms)"""
    
    def __init__(self, physics_constants=None):
        self.physics_constants = physics_constants or PhysicsConstants()
        self.time_scales = {
            'pulse': (1e-9, 10e-9),      # 1-10 ns
            'short_term': (10e-9, 100e-9), # 10-100 ns
            'medium_term': (100e-9, 1e-6), # 100 ns - 1 μs
            'long_term': (1e-6, 1e-3),    # 1 μs - 1 ms
            'very_long_term': (1e-3, 20e-3) # 1-20 ms
        }
        
    def generate_time_points(self, duration_ns, max_time_ms=20.0, resolution='adaptive'):
        """Generate time points for multi-scale extrapolation"""
        pulse_duration = duration_ns * 1e-9  # Convert to seconds
        
        if resolution == 'adaptive':
            # Adaptive time stepping: fine during pulse, coarse after
            time_points = []
            
            # During pulse: fine resolution
            n_pulse_points = max(10, int(duration_ns / 0.1))
            time_points.extend(np.linspace(0, duration_ns, n_pulse_points))
            
            # Short-term after pulse: logarithmic spacing
            short_term_end = duration_ns * 10  # 10x pulse duration
            n_short = 20
            time_points.extend(np.logspace(np.log10(duration_ns + 0.1), 
                                          np.log10(short_term_end), n_short))
            
            # Long-term: coarser spacing
            long_term_end = max_time_ms * 1e6  # Convert ms to ns
            n_long = 30
            time_points.extend(np.logspace(np.log10(short_term_end * 1.1), 
                                          np.log10(long_term_end), n_long))
            
            # Sort and remove duplicates
            time_points = sorted(set(np.round(time_points, 2)))
        
        else:
            # Uniform logarithmic spacing
            time_points = np.logspace(np.log10(0.1), 
                                     np.log10(max_time_ms * 1e6), 
                                     100)
        
        # Convert to ms for output
        time_points_ms = np.array(time_points) * 1e-6
        
        return time_points, time_points_ms
    
    def apply_diffusion_model(self, temperature_field, time_points_s, initial_time_s):
        """Apply analytical diffusion model for post-pulse extrapolation"""
        alpha = self.physics_constants.THERMAL_DIFFUSIVITY
        L = self.physics_constants.LASER_SPOT_RADIUS
        
        # Simplified Green's function solution for point source
        # T(r,t) = Q/(4παρc t)^(3/2) * exp(-r²/(4αt))
        
        # This is a placeholder for actual implementation
        # In practice, would need to know spatial distribution of initial temperature
        
        return temperature_field  # Return modified field
    
    def compute_thermal_history(self, initial_profile, time_points_s, material_properties):
        """Compute thermal history using reduced-order model"""
        # Implement ROM for fast thermal simulation
        # Could use modal decomposition or neural network
        
        # Placeholder for actual implementation
        return initial_profile

# =============================================
# ENHANCED STREAMLIT APPLICATION WITH PHYSICS CONSTRAINTS
# =============================================
def render_physics_enhanced_extrapolation():
    """Render the enhanced extrapolation interface with physics constraints"""
    st.markdown('<h2 class="sub-header">🔬 Physics-Enhanced Temporal Extrapolation</h2>', 
               unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load simulation data first.")
        return
    
    st.markdown("""
    <div class="info-box">
    <h3>🌡️ Multi-Scale Temporal Extrapolation</h3>
    <p>This module extends predictions from ns to ms scale (6 orders of magnitude) 
    by enforcing physics-based constraints:</p>
    <ul>
    <li><strong>Post-pulse thermal decay</strong> via diffusion equation solutions</li>
    <li><strong>Thermo-mechanical coupling</strong> for stress/strain evolution</li>
    <li><strong>Monotonicity constraints</strong> for temperature after pulse</li>
    <li><strong>Bound constraints</strong> (ambient ↔ melting temperature)</li>
    </ul>
    <p><strong>Repetition rate:</strong> 50 Hz (20 ms between pulses)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Physics parameters
    with st.expander("⚛️ Physics Parameters", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            enable_physics = st.checkbox("Enable Physics Constraints", value=True,
                                       help="Apply physics-based constraints to extrapolation")
            enforce_monotonic = st.checkbox("Enforce Monotonic Cooling", value=True,
                                          help="Temperature must decrease after pulse")
        
        with col2:
            decay_model = st.selectbox(
                "Decay Model",
                ["exponential", "diffusive", "hybrid"],
                index=0,
                help="Model for post-pulse temperature decay"
            )
            
            material_type = st.selectbox(
                "Material",
                ["SAC305 (default)", "Copper", "Gold", "Custom"],
                index=0
            )
        
        with col3:
            ambient_temp = st.number_input(
                "Ambient Temperature (°C)",
                value=25.0,
                min_value=0.0,
                max_value=100.0,
                help="Reference ambient temperature"
            )
            
            max_time_ms = st.number_input(
                "Maximum Time (ms)",
                value=20.0,
                min_value=0.1,
                max_value=100.0,
                step=0.1,
                help="Extrapolation up to this time (50 Hz = 20 ms period)"
            )
    
    # Update extrapolator with physics settings
    if enable_physics:
        st.session_state.extrapolator.physics_enforcement = True
        st.session_state.extrapolator.decay_model = decay_model
        st.session_state.extrapolator.enforce_monotonicity = enforce_monotonic
        st.session_state.extrapolator.physics_constants.AMBIENT_TEMPERATURE = ambient_temp + 273.15
    
    # Query parameters
    st.markdown('<h3 class="sub-header">🎯 Simulation Parameters</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.session_state.summaries:
            energies = [s['energy'] for s in st.session_state.summaries]
            min_energy, max_energy = min(energies), max(energies)
        else:
            min_energy, max_energy = 0.1, 50.0
        
        energy_query = st.number_input(
            "Laser Energy (mJ)",
            min_value=float(min_energy * 0.5),
            max_value=float(max_energy * 2.0),
            value=float((min_energy + max_energy) / 2),
            step=0.1
        )
    
    with col2:
        if st.session_state.summaries:
            durations = [s['duration'] for s in st.session_state.summaries]
            min_duration, max_duration = min(durations), max(durations)
        else:
            min_duration, max_duration = 0.5, 20.0
        
        duration_query = st.number_input(
            "Pulse Duration (ns)",
            min_value=float(min_duration * 0.5),
            max_value=float(max_duration * 2.0),
            value=float((min_duration + max_duration) / 2),
            step=0.1
        )
    
    with col3:
        time_resolution = st.selectbox(
            "Time Scaling",
            ["Adaptive (recommended)", "Logarithmic", "Uniform"],
            index=0,
            help="Time point distribution for multi-scale extrapolation"
        )
    
    # Generate time points
    time_generator = MultiScaleTemporalExtrapolator()
    time_points_ns, time_points_ms = time_generator.generate_time_points(
        duration_query, max_time_ms, 
        resolution='adaptive' if time_resolution == "Adaptive (recommended)" else 'log'
    )
    
    # Show time scale information
    st.info(f"**Time range:** {time_points_ms[0]:.2e} ms to {time_points_ms[-1]:.2f} ms "
            f"({len(time_points_ms)} points, {time_points_ms[-1]/time_points_ms[0]:.1e} range)")
    
    # Run physics-enhanced extrapolation
    if st.button("🚀 Run Physics-Enhanced Extrapolation", type="primary", use_container_width=True):
        with st.spinner("Running multi-scale physics-enhanced extrapolation..."):
            # Clear cache
            if 'interpolation_3d_cache' in st.session_state:
                st.session_state.interpolation_3d_cache = {}
            
            # Run extrapolation
            results = st.session_state.extrapolator.predict_time_series(
                energy_query, duration_query, time_points_ms
            )
            
            if results and 'field_predictions' in results:
                st.session_state.interpolation_results = results
                st.session_state.interpolation_params = {
                    'energy_query': energy_query,
                    'duration_query': duration_query,
                    'time_points_ms': time_points_ms,
                    'time_points_ns': time_points_ns,
                    'physics_enabled': enable_physics,
                    'decay_model': decay_model
                }
                
                st.success("✅ Physics-enhanced extrapolation completed!")
                
                # Display results in tabs
                tab1, tab2, tab3, tab4 = st.tabs(["📈 Multi-Scale Plots", "🌡️ Temperature Evolution", 
                                                 "⚙️ Stress/Strain", "📊 Physics Diagnostics"])
                
                with tab1:
                    render_multi_scale_plots(results, time_points_ms, duration_query, energy_query)
                
                with tab2:
                    render_temperature_evolution(results, time_points_ms, duration_query)
                
                with tab3:
                    render_stress_strain_analysis(results, time_points_ms)
                
                with tab4:
                    render_physics_diagnostics(results, time_points_ms, duration_query)

def render_multi_scale_plots(results, time_points_ms, duration_query, energy_query):
    """Render multi-scale plots with physics constraints"""
    # Convert pulse duration to ms
    pulse_duration_ms = duration_query * 1e-6
    
    # Create figure with multiple y-axes
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Temperature Evolution", "Thermal Stress",
                       "Cooling Rate", "Cumulative Thermal Load"),
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    # Plot 1: Temperature evolution
    if 'temperature' in results['field_predictions']:
        temp_data = results['field_predictions']['temperature']
        
        fig.add_trace(
            go.Scatter(
                x=time_points_ms,
                y=temp_data['mean'],
                mode='lines',
                name='Mean Temp',
                line=dict(color='red', width=3),
                fill='tozeroy',
                fillcolor='rgba(255,0,0,0.1)'
            ),
            row=1, col=1
        )
        
        # Add pulse duration indicator
        fig.add_vline(x=pulse_duration_ms, line_dash="dash", line_color="blue", 
                     annotation_text="Pulse End", row=1, col=1)
    
    # Plot 2: Thermal stress
    if 'thermal_stress' in results['field_predictions']:
        stress_data = results['field_predictions']['thermal_stress']
        
        fig.add_trace(
            go.Scatter(
                x=time_points_ms,
                y=np.array(stress_data['mean']) / 1e6,  # Convert to MPa
                mode='lines',
                name='Thermal Stress',
                line=dict(color='orange', width=3),
                fill='tozeroy',
                fillcolor='rgba(255,165,0,0.1)'
            ),
            row=1, col=2
        )
        
        # Add yield strength line
        yield_strength = PhysicsConstants.YIELD_STRENGTH / 1e6
        fig.add_hline(y=yield_strength, line_dash="dot", line_color="red",
                     annotation_text="Yield Strength", row=1, col=2)
    
    # Plot 3: Cooling rate
    if 'temperature' in results['field_predictions']:
        temp_mean = np.array(results['field_predictions']['temperature']['mean'])
        
        # Compute cooling rate (dT/dt)
        dt = np.diff(time_points_ms) * 1e-3  # Convert to seconds
        dT = np.diff(temp_mean)
        cooling_rate = -dT / dt  # Negative for cooling
        
        # Smooth cooling rate
        if len(cooling_rate) > 10:
            window = max(3, len(cooling_rate) // 20)
            cooling_rate_smooth = np.convolve(cooling_rate, np.ones(window)/window, mode='same')
        else:
            cooling_rate_smooth = cooling_rate
        
        fig.add_trace(
            go.Scatter(
                x=time_points_ms[1:],
                y=cooling_rate_smooth,
                mode='lines',
                name='Cooling Rate',
                line=dict(color='green', width=3)
            ),
            row=2, col=1
        )
    
    # Plot 4: Cumulative thermal load
    if 'temperature' in results['field_predictions']:
        temp_mean = np.array(results['field_predictions']['temperature']['mean'])
        T_ref = PhysicsConstants.AMBIENT_TEMPERATURE
        
        # Compute thermal load integral
        thermal_load = np.cumsum(np.maximum(temp_mean - T_ref, 0)) * np.mean(np.diff(time_points_ms))
        
        fig.add_trace(
            go.Scatter(
                x=time_points_ms,
                y=thermal_load,
                mode='lines',
                name='Thermal Load',
                line=dict(color='purple', width=3),
                fill='tozeroy',
                fillcolor='rgba(128,0,128,0.1)'
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text=f"Multi-Scale Physics Analysis (E={energy_query:.1f} mJ, τ={duration_query:.1f} ns)",
        showlegend=True,
        hovermode="x unified"
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Time (ms)", row=2, col=1)
    fig.update_xaxes(title_text="Time (ms)", row=2, col=2)
    fig.update_xaxes(title_text="Time (ms)", row=1, col=1)
    fig.update_xaxes(title_text="Time (ms)", row=1, col=2)
    
    fig.update_yaxes(title_text="Temperature (K)", row=1, col=1)
    fig.update_yaxes(title_text="Stress (MPa)", row=1, col=2)
    fig.update_yaxes(title_text="Cooling Rate (K/s)", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative Load (K·ms)", row=2, col=2)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Time scale analysis
    with st.expander("⏱️ Time Scale Analysis"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Pulse heating time
            st.metric("Pulse Duration", f"{duration_query:.2f} ns")
        
        with col2:
            # Thermal diffusion time
            tau_diff_ms = PhysicsConstants().calculate_diffusion_time(
                PhysicsConstants.BALL_RADIUS
            ) * 1000  # Convert to ms
            st.metric("Diffusion Time", f"{tau_diff_ms:.2f} ms")
        
        with col3:
            # Repetition period
            rep_rate_hz = 50
            rep_period_ms = 1000 / rep_rate_hz
            st.metric("Repetition Period", f"{rep_period_ms:.1f} ms")

def render_temperature_evolution(results, time_points_ms, duration_query):
    """Render detailed temperature evolution analysis"""
    if 'temperature' not in results['field_predictions']:
        st.warning("No temperature data available.")
        return
    
    temp_data = results['field_predictions']['temperature']
    pulse_duration_ms = duration_query * 1e-6
    
    # Create multi-panel plot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Temperature vs Time", "Log-Log Plot", 
                       "Cooling Phases", "Temperature Gradient"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": True}, {"secondary_y": False}]]
    )
    
    # Panel 1: Linear plot
    fig.add_trace(
        go.Scatter(
            x=time_points_ms,
            y=temp_data['mean'],
            mode='lines+markers',
            name='Mean Temperature',
            line=dict(color='red', width=3),
            marker=dict(size=4)
        ),
        row=1, col=1
    )
    
    # Add confidence band
    if temp_data['std']:
        mean_vals = np.array(temp_data['mean'])
        std_vals = np.array(temp_data['std'])
        y_upper = mean_vals + std_vals
        y_lower = mean_vals - std_vals
        
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([time_points_ms, time_points_ms[::-1]]),
                y=np.concatenate([y_upper, y_lower[::-1]]),
                fill='toself',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                name='±1σ'
            ),
            row=1, col=1
        )
    
    # Panel 2: Log-log plot
    mask = time_points_ms > pulse_duration_ms  # Only post-pulse
    if np.any(mask):
        fig.add_trace(
            go.Scatter(
                x=time_points_ms[mask],
                y=np.array(temp_data['mean'])[mask],
                mode='lines+markers',
                name='Post-pulse',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ),
            row=1, col=2
        )
        
        # Add theoretical diffusion slope (t^(-3/2) for point source)
        t_fit = np.logspace(np.log10(pulse_duration_ms*1.1), 
                           np.log10(time_points_ms[-1]), 50)
        T0 = temp_data['mean'][np.argmin(np.abs(time_points_ms - pulse_duration_ms))]
        T_fit = T0 * (t_fit / pulse_duration_ms)**(-1.5)
        
        fig.add_trace(
            go.Scatter(
                x=t_fit,
                y=T_fit,
                mode='lines',
                name='t^(-3/2) slope',
                line=dict(color='green', width=2, dash='dash')
            ),
            row=1, col=2
        )
    
    # Panel 3: Cooling phases
    if len(temp_data['mean']) > 2:
        temp_vals = np.array(temp_data['mean'])
        temp_normalized = (temp_vals - PhysicsConstants.AMBIENT_TEMPERATURE) / \
                         (np.max(temp_vals) - PhysicsConstants.AMBIENT_TEMPERATURE)
        
        fig.add_trace(
            go.Scatter(
                x=time_points_ms,
                y=temp_normalized,
                mode='lines',
                name='Normalized Temp',
                line=dict(color='purple', width=3)
            ),
            row=2, col=1
        )
        
        # Add exponential fit for comparison
        post_pulse_idx = np.where(time_points_ms > pulse_duration_ms)[0]
        if len(post_pulse_idx) > 3:
            t_post = time_points_ms[post_pulse_idx] - pulse_duration_ms
            T_post = temp_normalized[post_pulse_idx]
            
            # Fit exponential
            try:
                log_T = np.log(T_post + 1e-6)
                coeffs = np.polyfit(t_post, log_T, 1)
                tau_fit = -1/coeffs[0]
                
                T_fit_exp = np.exp(coeffs[1]) * np.exp(coeffs[0] * t_post)
                
                fig.add_trace(
                    go.Scatter(
                        x=t_post + pulse_duration_ms,
                        y=T_fit_exp,
                        mode='lines',
                        name=f'Exp fit (τ={tau_fit:.2f} ms)',
                        line=dict(color='orange', width=2, dash='dash')
                    ),
                    row=2, col=1
                )
            except:
                pass
    
    # Panel 4: Temperature gradient
    if len(temp_data['mean']) > 2:
        dT_dt = np.gradient(temp_data['mean'], time_points_ms)
        
        fig.add_trace(
            go.Scatter(
                x=time_points_ms,
                y=dT_dt,
                mode='lines',
                name='dT/dt',
                line=dict(color='brown', width=2)
            ),
            row=2, col=2
        )
        
        # Add heating/cooling regions
        heating_region = dT_dt > 0
        if np.any(heating_region):
            fig.add_trace(
                go.Scatter(
                    x=time_points_ms[heating_region],
                    y=dT_dt[heating_region],
                    mode='markers',
                    name='Heating',
                    marker=dict(color='red', size=6)
                ),
                row=2, col=2
            )
    
    # Update layout
    fig.update_layout(
        height=700,
        title="Detailed Temperature Evolution Analysis",
        showlegend=True
    )
    
    # Update axes
    fig.update_xaxes(title_text="Time (ms)", row=1, col=1)
    fig.update_xaxes(type="log", title_text="Time (ms)", row=1, col=2)
    fig.update_xaxes(title_text="Time (ms)", row=2, col=1)
    fig.update_xaxes(title_text="Time (ms)", row=2, col=2)
    
    fig.update_yaxes(title_text="Temperature (K)", row=1, col=1)
    fig.update_yaxes(type="log", title_text="Temperature (K)", row=1, col=2)
    fig.update_yaxes(title_text="Normalized Temp", row=2, col=1)
    fig.update_yaxes(title_text="dT/dt (K/ms)", row=2, col=2)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Temperature statistics
    with st.expander("📊 Temperature Statistics"):
        temp_vals = np.array(temp_data['mean'])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            peak_idx = np.argmax(temp_vals)
            st.metric("Peak Temperature", f"{temp_vals[peak_idx]:.1f} K")
            st.caption(f"at {time_points_ms[peak_idx]:.3f} ms")
        
        with col2:
            final_temp = temp_vals[-1]
            st.metric("Final Temperature", f"{final_temp:.1f} K")
            st.caption(f"at {time_points_ms[-1]:.2f} ms")
        
        with col3:
            cooling_start = pulse_duration_ms
            idx_start = np.argmin(np.abs(time_points_ms - cooling_start))
            T_start = temp_vals[idx_start]
            T_end = temp_vals[-1]
            cooling_percent = 100 * (T_start - T_end) / (T_start - PhysicsConstants.AMBIENT_TEMPERATURE)
            st.metric("Cooling %", f"{cooling_percent:.1f}%")
        
        with col4:
            max_cooling_rate = np.min(np.gradient(temp_vals, time_points_ms))
            st.metric("Max Cooling Rate", f"{-max_cooling_rate:.1f} K/ms")

def render_stress_strain_analysis(results, time_points_ms):
    """Render stress and strain evolution analysis"""
    if 'thermal_stress' not in results['field_predictions']:
        st.warning("No stress/strain data computed. Ensure temperature data is available.")
        return
    
    stress_data = results['field_predictions']['thermal_stress']
    strain_data = results['field_predictions'].get('thermal_strain', {})
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Thermal Stress Evolution", "Stress vs Temperature",
                       "Strain Evolution", "Stress Relaxation"),
        vertical_spacing=0.15
    )
    
    # Panel 1: Stress evolution
    stress_mean = np.array(stress_data['mean']) / 1e6  # MPa
    stress_max = np.array(stress_data['max']) / 1e6 if stress_data['max'] else stress_mean
    
    fig.add_trace(
        go.Scatter(
            x=time_points_ms,
            y=stress_mean,
            mode='lines',
            name='Mean Stress',
            line=dict(color='orange', width=3)
        ),
        row=1, col=1
    )
    
    # Add yield strength line
    yield_strength = PhysicsConstants.YIELD_STRENGTH / 1e6
    fig.add_hline(y=yield_strength, line_dash="dash", line_color="red",
                 annotation_text=f"Yield: {yield_strength:.0f} MPa", 
                 row=1, col=1)
    
    # Panel 2: Stress vs Temperature
    if 'temperature' in results['field_predictions']:
        temp_vals = np.array(results['field_predictions']['temperature']['mean'])
        
        fig.add_trace(
            go.Scatter(
                x=temp_vals,
                y=stress_mean,
                mode='markers',
                marker=dict(
                    size=8,
                    color=time_points_ms,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Time (ms)")
                ),
                name='Stress vs Temp',
                hovertemplate='Temp: %{x:.1f} K<br>Stress: %{y:.1f} MPa<br>Time: %{marker.color:.3f} ms'
            ),
            row=1, col=2
        )
        
        # Add theoretical linear relationship
        alpha = PhysicsConstants.COEFFICIENT_THERMAL_EXPANSION
        E = PhysicsConstants.YOUNGS_MODULUS
        nu = PhysicsConstants.POISSON_RATIO
        
        T_min = np.min(temp_vals)
        T_max = np.max(temp_vals)
        T_range = np.linspace(T_min, T_max, 100)
        stress_theoretical = alpha * E * (T_range - PhysicsConstants.AMBIENT_TEMPERATURE) / (1 - nu) / 1e6
        
        fig.add_trace(
            go.Scatter(
                x=T_range,
                y=stress_theoretical,
                mode='lines',
                name='Theoretical (linear)',
                line=dict(color='red', width=2, dash='dash')
            ),
            row=1, col=2
        )
    
    # Panel 3: Strain evolution
    if strain_data and 'mean' in strain_data:
        strain_vals = np.array(strain_data['mean'])
        
        fig.add_trace(
            go.Scatter(
                x=time_points_ms,
                y=strain_vals * 1e6,  # Convert to microstrain
                mode='lines',
                name='Thermal Strain',
                line=dict(color='green', width=3)
            ),
            row=2, col=1
        )
        
        # Add accumulated strain
        accumulated_strain = np.cumsum(np.abs(np.diff(strain_vals, prepend=strain_vals[0])))
        fig.add_trace(
            go.Scatter(
                x=time_points_ms,
                y=accumulated_strain * 1e6,
                mode='lines',
                name='Accumulated Strain',
                line=dict(color='blue', width=2, dash='dot')
            ),
            row=2, col=1
        )
    
    # Panel 4: Stress relaxation
    if len(stress_mean) > 2:
        # Compute relaxation: stress normalized by peak
        peak_stress = np.max(stress_mean)
        stress_normalized = stress_mean / peak_stress
        
        fig.add_trace(
            go.Scatter(
                x=time_points_ms,
                y=stress_normalized,
                mode='lines',
                name='Normalized Stress',
                line=dict(color='purple', width=3)
            ),
            row=2, col=2
        )
        
        # Fit relaxation time
        try:
            # Find post-peak region
            peak_idx = np.argmax(stress_mean)
            post_peak_idx = np.where(time_points_ms > time_points_ms[peak_idx])[0]
            
            if len(post_peak_idx) > 3:
                t_relax = time_points_ms[post_peak_idx] - time_points_ms[peak_idx]
                stress_relax = stress_normalized[post_peak_idx]
                
                # Fit exponential
                log_stress = np.log(stress_relax + 1e-6)
                coeffs = np.polyfit(t_relax, log_stress, 1)
                tau_relax = -1/coeffs[0]
                
                stress_fit = np.exp(coeffs[1]) * np.exp(coeffs[0] * t_relax)
                
                fig.add_trace(
                    go.Scatter(
                        x=t_relax + time_points_ms[peak_idx],
                        y=stress_fit,
                        mode='lines',
                        name=f'Relaxation fit (τ={tau_relax:.2f} ms)',
                        line=dict(color='orange', width=2, dash='dash')
                    ),
                    row=2, col=2
                )
        except:
            pass
    
    # Update layout
    fig.update_layout(
        height=700,
        title="Thermo-Mechanical Analysis",
        showlegend=True,
        hovermode="closest"
    )
    
    # Update axes
    fig.update_xaxes(title_text="Time (ms)", row=1, col=1)
    fig.update_xaxes(title_text="Temperature (K)", row=1, col=2)
    fig.update_xaxes(title_text="Time (ms)", row=2, col=1)
    fig.update_xaxes(title_text="Time (ms)", row=2, col=2)
    
    fig.update_yaxes(title_text="Stress (MPa)", row=1, col=1)
    fig.update_yaxes(title_text="Stress (MPa)", row=1, col=2)
    fig.update_yaxes(title_text="Strain (με)", row=2, col=1)
    fig.update_yaxes(title_text="Normalized Stress", row=2, col=2)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Stress analysis
    with st.expander("⚙️ Stress Analysis"):
        peak_stress = np.max(stress_mean)
        peak_strain = np.max(strain_data['mean']) if strain_data and 'mean' in strain_data else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Peak Stress", f"{peak_stress:.1f} MPa")
            stress_ratio = peak_stress / yield_strength
            st.progress(min(stress_ratio, 1.0), 
                       text=f"{stress_ratio*100:.1f}% of yield")
        
        with col2:
            if peak_stress > yield_strength:
                st.error("⚠️ Yield Exceeded")
                st.caption("Plastic deformation likely")
            else:
                st.success("✅ Below Yield")
                st.caption("Elastic regime")
        
        with col3:
            st.metric("Peak Strain", f"{peak_strain*1e6:.1f} με")
        
        with col4:
            residual_stress = stress_mean[-1]
            st.metric("Residual Stress", f"{residual_stress:.1f} MPa")
            if abs(residual_stress) > yield_strength * 0.1:
                st.warning("Significant residual stress")

def render_physics_diagnostics(results, time_points_ms, duration_query):
    """Render physics diagnostics and validation plots"""
    st.markdown('<h4 class="sub-header">🔍 Physics Diagnostics</h4>', unsafe_allow_html=True)
    
    # Create diagnostics dashboard
    col1, col2 = st.columns(2)
    
    with col1:
        # Energy conservation check
        st.markdown("##### ⚡ Energy Conservation")
        
        if 'temperature' in results['field_predictions']:
            temp_vals = np.array(results['field_predictions']['temperature']['mean'])
            
            # Simplified energy calculation
            rho = PhysicsConstants.DENSITY
            cp = PhysicsConstants.SPECIFIC_HEAT
            V = (4/3) * np.pi * PhysicsConstants.BALL_RADIUS**3  # Volume of sphere
            
            # Temperature rise above ambient
            delta_T = np.maximum(temp_vals - PhysicsConstants.AMBIENT_TEMPERATURE, 0)
            
            # Instantaneous thermal energy
            thermal_energy = rho * cp * V * delta_T
            
            # Input laser energy (simplified)
            laser_energy_joules = st.session_state.interpolation_params.get('energy_query', 0) * 1e-3
            
            # Efficiency factor (how much laser energy becomes thermal energy)
            efficiency = thermal_energy / laser_energy_joules
            
            fig_energy = go.Figure()
            fig_energy.add_trace(go.Scatter(
                x=time_points_ms,
                y=efficiency,
                mode='lines',
                name='Thermal Efficiency',
                line=dict(color='blue', width=3)
            ))
            
            fig_energy.add_hline(y=1.0, line_dash="dash", line_color="red",
                               annotation_text="100% Efficiency")
            
            fig_energy.update_layout(
                title="Energy Conservation Check",
                xaxis_title="Time (ms)",
                yaxis_title="Efficiency (-)",
                height=300
            )
            
            st.plotly_chart(fig_energy, use_container_width=True)
            
            # Efficiency statistics
            max_efficiency = np.max(efficiency)
            final_efficiency = efficiency[-1]
            
            st.metric("Max Efficiency", f"{max_efficiency*100:.1f}%")
            st.metric("Final Efficiency", f"{final_efficiency*100:.1f}%")
    
    with col2:
        # Diffusion length scale
        st.markdown("##### 🌡️ Thermal Penetration")
        
        alpha = PhysicsConstants.THERMAL_DIFFUSIVITY
        t_seconds = time_points_ms * 1e-3
        
        # Diffusion length: δ = √(4αt)
        diffusion_length = np.sqrt(4 * alpha * t_seconds) * 1e6  # Convert to μm
        
        fig_diff = go.Figure()
        fig_diff.add_trace(go.Scatter(
            x=time_points_ms,
            y=diffusion_length,
            mode='lines',
            name='Diffusion Length',
            line=dict(color='green', width=3)
        ))
        
        # Add characteristic lengths
        spot_radius_um = PhysicsConstants.LASER_SPOT_RADIUS * 1e6
        ball_radius_um = PhysicsConstants.BALL_RADIUS * 1e6
        
        fig_diff.add_hline(y=spot_radius_um, line_dash="dash", line_color="orange",
                          annotation_text=f"Spot: {spot_radius_um:.0f} μm")
        fig_diff.add_hline(y=ball_radius_um, line_dash="dash", line_color="red",
                          annotation_text=f"Ball: {ball_radius_um:.0f} μm")
        
        fig_diff.update_layout(
            title="Thermal Penetration Depth",
            xaxis_title="Time (ms)",
            yaxis_title="Depth (μm)",
            height=300
        )
        
        st.plotly_chart(fig_diff, use_container_width=True)
        
        # Time when diffusion reaches ball radius
        t_reach_ball = ball_radius_um**2 / (4 * alpha * 1e12) * 1e3  # Convert to ms
        st.metric("Time to reach ball edge", f"{t_reach_ball:.2f} ms")
    
    # Physics validation metrics
    st.markdown("##### 📐 Dimensionless Numbers Analysis")
    
    # Compute Fourier numbers
    Fo = PhysicsConstants.THERMAL_DIFFUSIVITY * time_points_ms * 1e-3 / \
         (PhysicsConstants.LASER_SPOT_RADIUS**2)
    
    # Compute Biot numbers (simplified)
    h = 100  # Convection coefficient (W/m²K) - typical for air
    Bi = h * PhysicsConstants.LASER_SPOT_RADIUS / PhysicsConstants.THERMAL_CONDUCTIVITY
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Max Fourier Number", f"{np.max(Fo):.3f}")
        st.caption("Fo >> 1: diffusion dominant")
    
    with col2:
        st.metric("Biot Number", f"{Bi:.3f}")
        st.caption("Bi < 0.1: internal resistance dominant")
    
    with col3:
        # Thermal time constant
        tau_thermal = PhysicsConstants.LASER_SPOT_RADIUS**2 / \
                     (np.pi**2 * PhysicsConstants.THERMAL_DIFFUSIVITY)
        st.metric("Thermal Time Constant", f"{tau_thermal*1e3:.2f} ms")
    
    with col4:
        # Stress number
        alpha = PhysicsConstants.COEFFICIENT_THERMAL_EXPANSION
        E = PhysicsConstants.YOUNGS_MODULUS
        sigma_max = np.max(results['field_predictions']['thermal_stress']['mean']) \
                    if 'thermal_stress' in results['field_predictions'] else 0
        sigma_yield = PhysicsConstants.YIELD_STRENGTH
        stress_ratio = sigma_max / sigma_yield
        st.metric("Max Stress/Yield", f"{stress_ratio:.2f}")
    
    # Validation against analytical solutions
    with st.expander("🔬 Validation Against Analytical Solutions"):
        st.markdown("""
        **Comparison with 1D Heat Conduction Solution:**
        
        For instantaneous point source in infinite medium:
        \[
        T(r,t) = \frac{Q}{(\rho c_p)(4\pi \alpha t)^{3/2}} \exp\left(-\frac{r^2}{4\alpha t}\right)
        \]
        
        For continuous source:
        \[
        T(r,t) = \frac{Q}{4\pi k r} \text{erfc}\left(\frac{r}{\sqrt{4\alpha t}}\right)
        \]
        
        **Key observations for extrapolation validation:**
        1. Post-pulse decay follows t^{-3/2} for point source
        2. Temperature gradient smooths over time
        3. Far-field temperature rise is limited by diffusion
        4. Maximum temperature occurs at pulse end
        """)
        
        # Compare with analytical solution
        if 'temperature' in results['field_predictions']:
            temp_vals = np.array(results['field_predictions']['temperature']['mean'])
            pulse_duration_ms = duration_query * 1e-6
            
            # Simple analytical model for comparison
            T0 = temp_vals[np.argmin(np.abs(time_points_ms - pulse_duration_ms))]
            post_pulse_idx = np.where(time_points_ms > pulse_duration_ms)[0]
            
            if len(post_pulse_idx) > 3:
                t_post = time_points_ms[post_pulse_idx]
                t_rel = t_post - pulse_duration_ms
                
                # Point source decay model
                T_analytical = T0 * (pulse_duration_ms / t_rel)**1.5
                
                fig_compare = go.Figure()
                fig_compare.add_trace(go.Scatter(
                    x=t_post,
                    y=temp_vals[post_pulse_idx],
                    mode='lines',
                    name='Predicted',
                    line=dict(color='blue', width=3)
                ))
                fig_compare.add_trace(go.Scatter(
                    x=t_post,
                    y=T_analytical,
                    mode='lines',
                    name='Analytical (point source)',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                fig_compare.update_layout(
                    title="Comparison with Analytical Solution",
                    xaxis_title="Time (ms)",
                    yaxis_title="Temperature (K)",
                    xaxis_type="log",
                    yaxis_type="log",
                    height=400
                )
                
                st.plotly_chart(fig_compare, use_container_width=True)
                
                # Compute error
                valid_idx = t_rel > 0
                if np.any(valid_idx):
                    predicted = temp_vals[post_pulse_idx][valid_idx]
                    analytical = T_analytical[valid_idx]
                    rel_error = np.abs(predicted - analytical) / analytical
                    
                    st.metric("Mean Relative Error", f"{np.mean(rel_error)*100:.1f}%")
                    st.metric("Max Relative Error", f"{np.max(rel_error)*100:.1f}%")

# =============================================
# MAIN APPLICATION UPDATE
# =============================================
def main():
    """Main application with physics-enhanced extrapolation"""
    # [Previous setup code remains the same...]
    
    # Add new mode for physics-enhanced extrapolation
    if app_mode == "Physics-Enhanced Extrapolation":
        render_physics_enhanced_extrapolation()
    
    # [Rest of the application remains the same...]

# =============================================
# ADDITIONAL UTILITY FUNCTIONS
# =============================================
def analyze_extrapolation_accuracy(predictions_short, predictions_long):
    """Analyze accuracy of extrapolation from short to long times"""
    # This function compares predictions on overlapping time scales
    # to validate extrapolation accuracy
    
    metrics = {}
    
    # Find overlapping time points
    common_fields = set(predictions_short['field_predictions'].keys()) & \
                    set(predictions_long['field_predictions'].keys())
    
    for field in common_fields:
        # Interpolate to common time points
        # Compute RMSE, MAE, etc.
        pass
    
    return metrics

def generate_extrapolation_report(results, params, physics_constants):
    """Generate comprehensive report of extrapolation results"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'parameters': params,
        'physics_constants': {
            'thermal_diffusivity': physics_constants.THERMAL_DIFFUSIVITY,
            'characteristic_diffusion_time_ms': physics_constants.calculate_diffusion_time(
                physics_constants.BALL_RADIUS
            ) * 1000,
            'melting_temperature_K': physics_constants.MELTING_TEMPERATURE,
            'yield_strength_MPa': physics_constants.YIELD_STRENGTH / 1e6
        },
        'key_results': {},
        'warnings': [],
        'recommendations': []
    }
    
    # Extract key results
    if 'temperature' in results['field_predictions']:
        temp_data = results['field_predictions']['temperature']
        temp_vals = np.array(temp_data['mean'])
        
        report['key_results']['peak_temperature_K'] = float(np.max(temp_vals))
        report['key_results']['peak_time_ms'] = float(
            results['time_points_ms'][np.argmax(temp_vals)]
        )
        report['key_results']['final_temperature_K'] = float(temp_vals[-1])
        
        # Check for melting
        if report['key_results']['peak_temperature_K'] > physics_constants.MELTING_TEMPERATURE:
            report['warnings'].append("Temperature exceeds melting point")
    
    if 'thermal_stress' in results['field_predictions']:
        stress_data = results['field_predictions']['thermal_stress']
        stress_vals = np.array(stress_data['mean']) / 1e6  # MPa
        
        report['key_results']['peak_stress_MPa'] = float(np.max(stress_vals))
        report['key_results']['residual_stress_MPa'] = float(stress_vals[-1])
        
        # Check for yielding
        if report['key_results']['peak_stress_MPa'] > physics_constants.YIELD_STRENGTH / 1e6:
            report['warnings'].append("Stress exceeds yield strength")
            report['recommendations'].append("Consider reducing laser energy or increasing pulse duration")
    
    # Add extrapolation confidence
    if 'confidence_scores' in results:
        report['key_results']['mean_confidence'] = float(np.mean(results['confidence_scores']))
        report['key_results']['min_confidence'] = float(np.min(results['confidence_scores']))
    
    return report

# =============================================
# RUN APPLICATION
# =============================================
if __name__ == "__main__":
    main()
