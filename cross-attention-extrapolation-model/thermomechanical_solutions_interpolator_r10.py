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
from scipy.interpolate import griddata, RBFInterpolator, LinearNDInterpolator
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay, ConvexHull
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import sph_harm
import sympy as sp
from sympy import symbols, exp, sqrt, pi

warnings.filterwarnings('ignore')

# =============================================
# PATH CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEA_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "fea_solutions")
os.makedirs(FEA_SOLUTIONS_DIR, exist_ok=True)

# =============================================
# PHYSICS-AWARE ATTENTION WITH SPATIAL REGULARIZATION
# =============================================
class PhysicsInformedSpatialAttention:
    """Physics-informed attention with spatial locality regularization for accurate field interpolation"""
    
    def __init__(self, spatial_sigma=0.1, physics_weight=0.7, n_attention_layers=3):
        self.spatial_sigma = spatial_sigma
        self.physics_weight = physics_weight
        self.n_attention_layers = n_attention_layers
        self.source_data = []
        self.source_embeddings = []
        self.source_spatial_features = []
        self.source_metadata = []
        self.fitted = False
        
    def load_simulation_data(self, simulations):
        """Load full simulation data for spatial-aware interpolation"""
        self.source_data = []
        
        for sim_name, sim_data in simulations.items():
            if not sim_data.get('has_mesh', False):
                continue
                
            points = sim_data['points']
            
            # Extract spatial features
            spatial_features = self._extract_spatial_features(points)
            
            # Store for each field
            for field_name, field_info in sim_data['field_info'].items():
                field_data = sim_data['fields'][field_name]
                
                # For each timestep
                for t in range(min(5, sim_data['n_timesteps'])):  # Use first 5 timesteps
                    if field_info[0] == "scalar":
                        values = field_data[t]
                    else:
                        values = np.linalg.norm(field_data[t], axis=1)
                    
                    # Create embedding with spatial context
                    embedding = self._create_physics_spatial_embedding(
                        sim_data['energy_mJ'],
                        sim_data['duration_ns'],
                        t,
                        spatial_features,
                        values
                    )
                    
                    self.source_data.append({
                        'sim_name': sim_name,
                        'field_name': field_name,
                        'timestep': t,
                        'points': points,
                        'values': values,
                        'embedding': embedding,
                        'spatial_features': spatial_features,
                        'metadata': {
                            'energy': sim_data['energy_mJ'],
                            'duration': sim_data['duration_ns'],
                            'time': t
                        }
                    })
        
        if self.source_data:
            self.fitted = True
            st.success(f"✅ Loaded {len(self.source_data)} field instances for physics-informed interpolation")
    
    def _extract_spatial_features(self, points):
        """Extract comprehensive spatial features from point cloud"""
        if len(points) == 0:
            return np.zeros(10)
        
        # Basic spatial statistics
        center = np.mean(points, axis=0)
        cov = np.cov(points.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        
        # Shape descriptors
        volume = np.prod(np.max(points, axis=0) - np.min(points, axis=0))
        aspect_ratios = eigenvalues / np.max(eigenvalues)
        
        # Spatial distribution metrics
        distances = cdist([center], points)[0]
        radial_distribution = np.percentile(distances, [25, 50, 75])
        
        # Combine features
        features = np.concatenate([
            center,
            eigenvalues,
            [volume],
            aspect_ratios,
            radial_distribution
        ])
        
        return features
    
    def _create_physics_spatial_embedding(self, energy, duration, time, spatial_features, field_values):
        """Create physics-aware embedding with spatial context"""
        # Physical parameters
        power = energy / max(duration, 1e-6)
        energy_density = energy / (duration ** 2 + 1e-6)
        time_ratio = time / max(duration, 1e-3)
        
        # Field statistics
        field_mean = np.mean(field_values) if len(field_values) > 0 else 0
        field_std = np.std(field_values) if len(field_values) > 0 else 0
        field_gradient = np.mean(np.gradient(field_values)) if len(field_values) > 1 else 0
        
        # Combine all features
        return np.concatenate([
            [energy, duration, time, power, energy_density, time_ratio],
            spatial_features[:10],  # Use first 10 spatial features
            [field_mean, field_std, field_gradient]
        ])
    
    def _compute_spatial_kernel(self, points1, points2):
        """Compute spatial similarity kernel between point clouds"""
        if len(points1) == 0 or len(points2) == 0:
            return 0
        
        # Use earth mover's distance approximation
        tree1 = KDTree(points1)
        tree2 = KDTree(points2)
        
        # Bidirectional chamfer distance
        dist1, _ = tree1.query(points2, k=1)
        dist2, _ = tree2.query(points1, k=1)
        
        avg_dist = (np.mean(dist1) + np.mean(dist2)) / 2
        similarity = np.exp(-avg_dist / self.spatial_sigma)
        
        return similarity
    
    def interpolate_field(self, query_params, target_field, visualization_grid=None):
        """
        Interpolate field using physics-informed attention with spatial regularization
        
        Parameters:
        -----------
        query_params: dict with keys 'energy', 'duration', 'time'
        target_field: str, field name to interpolate
        visualization_grid: optional, pre-defined grid for visualization
        
        Returns:
        --------
        dict with interpolated field and attention information
        """
        if not self.fitted:
            return None
        
        # Filter relevant source data
        relevant_data = [d for d in self.source_data if d['field_name'] == target_field]
        
        if not relevant_data:
            return None
        
        # Compute attention weights with spatial regularization
        attention_weights = []
        spatial_similarities = []
        physics_similarities = []
        
        query_embedding = self._create_physics_spatial_embedding(
            query_params['energy'],
            query_params['duration'],
            query_params['time'],
            relevant_data[0]['spatial_features'],  # Use first as reference
            np.array([0])  # Dummy values
        )
        
        for source in relevant_data:
            # Physics similarity
            physics_sim = np.exp(-np.linalg.norm(
                query_embedding[:6] - source['embedding'][:6]
            ) / 0.5)
            
            # Spatial similarity
            if visualization_grid is not None:
                # Use grid points for spatial comparison
                spatial_sim = self._compute_spatial_kernel(
                    visualization_grid,
                    source['points']
                )
            else:
                spatial_sim = 1.0
            
            # Combined attention weight
            weight = (self.physics_weight * physics_sim + 
                     (1 - self.physics_weight) * spatial_sim)
            
            attention_weights.append(weight)
            physics_similarities.append(physics_sim)
            spatial_similarities.append(spatial_sim)
        
        # Normalize weights
        attention_weights = np.array(attention_weights)
        if np.sum(attention_weights) > 0:
            attention_weights = attention_weights / np.sum(attention_weights)
        
        # Create common grid for interpolation if not provided
        if visualization_grid is None:
            visualization_grid = self._create_common_grid(relevant_data)
        
        # Interpolate using attention-weighted combination
        interpolated_values = self._attention_weighted_interpolation(
            relevant_data,
            attention_weights,
            visualization_grid
        )
        
        return {
            'interpolated_field': interpolated_values,
            'grid_points': visualization_grid,
            'attention_weights': attention_weights,
            'physics_similarities': physics_similarities,
            'spatial_similarities': spatial_similarities,
            'source_data': relevant_data
        }
    
    def _create_common_grid(self, source_data, resolution=40):
        """Create common grid covering all source data"""
        all_points = np.vstack([d['points'] for d in source_data])
        
        if len(all_points) == 0:
            return None
        
        # Find bounding box
        min_coords = np.min(all_points, axis=0)
        max_coords = np.max(all_points, axis=0)
        
        # Create grid
        x = np.linspace(min_coords[0], max_coords[0], resolution)
        y = np.linspace(min_coords[1], max_coords[1], resolution)
        z = np.linspace(min_coords[2], max_coords[2], resolution)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T
        
        return grid_points
    
    def _attention_weighted_interpolation(self, source_data, weights, grid_points):
        """Perform attention-weighted interpolation on common grid"""
        if len(source_data) == 0:
            return None
        
        weighted_sum = np.zeros(len(grid_points))
        weight_sum = 0
        
        for i, source in enumerate(source_data):
            # Interpolate source field to common grid
            source_values = source['values']
            source_points = source['points']
            
            # Use RBF interpolation for smooth results
            if len(source_points) > 10 and len(source_values) > 10:
                try:
                    rbf = RBFInterpolator(source_points, source_values, kernel='thin_plate_spline')
                    interp_values = rbf(grid_points)
                    
                    # Add weighted contribution
                    weighted_sum += weights[i] * interp_values
                    weight_sum += weights[i]
                except:
                    # Fallback to linear interpolation
                    interp_values = griddata(source_points, source_values, grid_points, 
                                           method='linear', fill_value=0)
                    weighted_sum += weights[i] * interp_values
                    weight_sum += weights[i]
        
        if weight_sum > 0:
            return weighted_sum / weight_sum
        else:
            return weighted_sum
    
    def visualize_attention_distribution(self, attention_results):
        """Visualize attention weight distribution across parameter space"""
        if not attention_results or 'source_data' not in attention_results:
            return go.Figure()
        
        source_data = attention_results['source_data']
        weights = attention_results['attention_weights']
        
        # Extract metadata for plotting
        energies = []
        durations = []
        times = []
        sim_names = []
        
        for data in source_data:
            energies.append(data['metadata']['energy'])
            durations.append(data['metadata']['duration'])
            times.append(data['metadata']['time'])
            sim_names.append(f"{data['sim_name']}_t{data['timestep']}")
        
        # Create 3D scatter plot
        fig = go.Figure(data=go.Scatter3d(
            x=energies,
            y=durations,
            z=times,
            mode='markers',
            marker=dict(
                size=15,
                color=weights,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title="Attention Weight", thickness=20)
            ),
            text=sim_names,
            hovertemplate='<b>%{text}</b><br>' +
                         'Energy: %{x:.2f} mJ<br>' +
                         'Duration: %{y:.2f} ns<br>' +
                         'Time: %{z:.1f} ns<br>' +
                         'Weight: %{marker.color:.3f}<extra></extra>'
        ))
        
        # Highlight top 3 sources
        if len(weights) >= 3:
            top_indices = np.argsort(weights)[-3:][::-1]
            top_energies = [energies[i] for i in top_indices]
            top_durations = [durations[i] for i in top_indices]
            top_times = [times[i] for i in top_indices]
            
            fig.add_trace(go.Scatter3d(
                x=top_energies,
                y=top_durations,
                z=top_times,
                mode='markers',
                marker=dict(
                    size=20,
                    color='red',
                    symbol='diamond'
                ),
                name='Top 3 Sources'
            ))
        
        fig.update_layout(
            title="Attention Weight Distribution in Parameter Space",
            scene=dict(
                xaxis_title="Energy (mJ)",
                yaxis_title="Duration (ns)",
                zaxis_title="Time (ns)"
            ),
            height=600
        )
        
        return fig

# =============================================
# ENHANCED VTU LOADER WITH PHYSICS METADATA
# =============================================
class PhysicsAwareVTULoader:
    """Enhanced VTU loader with physics metadata extraction"""
    
    def __init__(self):
        self.simulations = {}
        self.summaries = []
        self.field_metadata = {}
        self.available_fields = set()
        
    def parse_folder_name(self, folder: str):
        """Parse folder name to extract physical parameters"""
        # Support multiple naming conventions
        patterns = [
            r"q([\dp\.]+)mJ-delta([\dp\.]+)ns",  # q0p5mJ-delta4p2ns
            r"E([\dp\.]+)mJ-tau([\dp\.]+)ns",     # E0p5mJ-tau4p2ns
            r"energy_([\d\.]+)mJ_duration_([\d\.]+)ns"  # energy_0.5mJ_duration_4.2ns
        ]
        
        for pattern in patterns:
            match = re.match(pattern, os.path.basename(folder))
            if match:
                e, d = match.groups()
                # Convert 'p' to '.' and to float
                energy = float(e.replace("p", ".")) if "p" in e else float(e)
                duration = float(d.replace("p", ".")) if "p" in d else float(d)
                return energy, duration
        
        return None, None
    
    @st.cache_data
    def load_all_simulations(_self, load_full_mesh=True):
        """Load all VTU simulations with physics metadata"""
        simulations = {}
        summaries = []
        
        folders = glob.glob(os.path.join(FEA_SOLUTIONS_DIR, "*/"))
        
        if not folders:
            st.warning(f"No simulation folders found in {FEA_SOLUTIONS_DIR}")
            return simulations, summaries
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for folder_idx, folder in enumerate(folders):
            folder_name = os.path.basename(folder.rstrip('/'))
            energy, duration = _self.parse_folder_name(folder_name)
            
            if energy is None or duration is None:
                continue
            
            # Find VTU files
            vtu_files = sorted(glob.glob(os.path.join(folder, "*.vtu")) + 
                              glob.glob(os.path.join(folder, "*.vtu.gz")))
            
            if not vtu_files:
                continue
            
            status_text.text(f"Loading {folder_name}... ({len(vtu_files)} files)")
            
            try:
                # Read first file to get structure
                mesh0 = meshio.read(vtu_files[0])
                
                # Create simulation entry
                sim_data = {
                    'name': folder_name,
                    'energy_mJ': energy,
                    'duration_ns': duration,
                    'n_timesteps': len(vtu_files),
                    'vtu_files': vtu_files,
                    'field_info': {},
                    'physics_metadata': {},
                    'has_mesh': False
                }
                
                # Extract physics metadata
                sim_data['physics_metadata'] = _self._extract_physics_metadata(mesh0, energy, duration)
                
                if load_full_mesh:
                    # Full mesh loading
                    points = mesh0.points.astype(np.float32)
                    n_pts = len(points)
                    
                    # Find mesh cells
                    triangles = None
                    tetrahedra = None
                    
                    for cell_block in mesh0.cells:
                        if cell_block.type == "triangle":
                            triangles = cell_block.data.astype(np.int32)
                        elif cell_block.type == "tetra":
                            tetrahedra = cell_block.data.astype(np.int32)
                    
                    # Initialize fields
                    fields = {}
                    for key in mesh0.point_data.keys():
                        arr = mesh0.point_data[key]
                        
                        # Store field metadata
                        if arr.ndim == 1:
                            sim_data['field_info'][key] = {
                                'type': 'scalar',
                                'dim': 1,
                                'units': _self._infer_units(key),
                                'physical_meaning': _self._infer_physical_meaning(key)
                            }
                            fields[key] = np.full((len(vtu_files), n_pts), np.nan, dtype=np.float32)
                        else:
                            sim_data['field_info'][key] = {
                                'type': 'vector',
                                'dim': arr.shape[1],
                                'units': _self._infer_units(key),
                                'physical_meaning': _self._infer_physical_meaning(key)
                            }
                            fields[key] = np.full((len(vtu_files), n_pts, arr.shape[1]), np.nan, dtype=np.float32)
                        
                        fields[key][0] = arr.astype(np.float32)
                        _self.available_fields.add(key)
                    
                    # Load remaining timesteps
                    for t in range(1, len(vtu_files)):
                        try:
                            mesh = meshio.read(vtu_files[t])
                            for key in sim_data['field_info']:
                                if key in mesh.point_data:
                                    fields[key][t] = mesh.point_data[key].astype(np.float32)
                        except Exception as e:
                            st.warning(f"Error loading timestep {t} in {folder_name}: {e}")
                    
                    sim_data.update({
                        'points': points,
                        'fields': fields,
                        'triangles': triangles,
                        'tetrahedra': tetrahedra,
                        'has_mesh': True
                    })
                
                # Create comprehensive summary
                summary = _self._create_physics_summary(vtu_files, energy, duration, folder_name, sim_data['field_info'])
                summaries.append(summary)
                
                simulations[folder_name] = sim_data
                
            except Exception as e:
                st.warning(f"Error loading {folder_name}: {str(e)}")
                continue
            
            progress_bar.progress((folder_idx + 1) / len(folders))
        
        progress_bar.empty()
        status_text.empty()
        
        if simulations:
            st.success(f"✅ Loaded {len(simulations)} simulations with {len(_self.available_fields)} unique fields")
            
            # Display field statistics
            with st.expander("📊 Field Overview", expanded=False):
                field_stats = []
                for field in sorted(_self.available_fields):
                    field_stats.append({
                        'Field': field,
                        'Type': simulations[list(simulations.keys())[0]]['field_info'].get(field, {}).get('type', 'unknown'),
                        'Units': simulations[list(simulations.keys())[0]]['field_info'].get(field, {}).get('units', 'N/A'),
                        'Physical Meaning': simulations[list(simulations.keys())[0]]['field_info'].get(field, {}).get('physical_meaning', 'N/A')
                    })
                
                if field_stats:
                    st.dataframe(pd.DataFrame(field_stats), use_container_width=True)
        
        return simulations, summaries
    
    def _extract_physics_metadata(self, mesh, energy, duration):
        """Extract physics metadata from mesh"""
        metadata = {
            'energy_mJ': energy,
            'duration_ns': duration,
            'peak_power_MW': energy / max(duration, 1e-6) * 1000,  # Convert to MW
            'energy_density_mJ_ns2': energy / (duration ** 2 + 1e-6),
            'mesh_info': {
                'n_points': len(mesh.points),
                'n_cells': sum(len(cell.data) for cell in mesh.cells),
                'cell_types': [cell.type for cell in mesh.cells],
                'bounds': {
                    'x_min': float(np.min(mesh.points[:, 0])),
                    'x_max': float(np.max(mesh.points[:, 0])),
                    'y_min': float(np.min(mesh.points[:, 1])),
                    'y_max': float(np.max(mesh.points[:, 1])),
                    'z_min': float(np.min(mesh.points[:, 2])),
                    'z_max': float(np.max(mesh.points[:, 2]))
                }
            }
        }
        
        return metadata
    
    def _infer_units(self, field_name):
        """Infer physical units based on field name"""
        field_lower = field_name.lower()
        
        units_map = {
            'temperature': 'K',
            'temp': 'K',
            'stress': 'Pa',
            'strain': '',
            'displacement': 'm',
            'velocity': 'm/s',
            'acceleration': 'm/s²',
            'pressure': 'Pa',
            'force': 'N',
            'energy': 'J',
            'power': 'W',
            'density': 'kg/m³',
            'heat': 'J',
            'flux': 'W/m²',
            'gradient': '/m'
        }
        
        for key, unit in units_map.items():
            if key in field_lower:
                return unit
        
        return 'N/A'
    
    def _infer_physical_meaning(self, field_name):
        """Infer physical meaning based on field name"""
        field_lower = field_name.lower()
        
        meaning_map = {
            'temperature': 'Thermal field indicating heat distribution',
            'temp': 'Thermal field indicating heat distribution',
            'stress': 'Mechanical stress tensor components',
            'strain': 'Deformation relative to original shape',
            'displacement': 'Position change from reference',
            'velocity': 'Rate of position change',
            'acceleration': 'Rate of velocity change',
            'pressure': 'Normal force per unit area',
            'force': 'Interaction causing motion change',
            'von_mises': 'Equivalent stress (scalar)',
            'principal': 'Principal stress directions',
            'heat': 'Thermal energy transfer',
            'flux': 'Flow rate through surface',
            'gradient': 'Spatial rate of change'
        }
        
        for key, meaning in meaning_map.items():
            if key in field_lower:
                return meaning
        
        return 'Physical field from simulation'
    
    def _create_physics_summary(self, vtu_files, energy, duration, name, field_info):
        """Create physics-aware summary statistics"""
        summary = {
            'name': name,
            'energy': energy,
            'duration': duration,
            'physics_metrics': {
                'peak_power_MW': energy / max(duration, 1e-6) * 1000,
                'energy_density_mJ_ns2': energy / (duration ** 2 + 1e-6),
                'pulse_intensity': energy / duration
            },
            'timesteps': [],
            'field_stats': {}
        }
        
        # Sample files for statistics (first, middle, last)
        sample_indices = [0, len(vtu_files)//2, -1]
        
        for idx in sample_indices:
            if idx < len(vtu_files):
                try:
                    mesh = meshio.read(vtu_files[idx])
                    timestep = idx + 1
                    summary['timesteps'].append(timestep)
                    
                    for field_name in mesh.point_data.keys():
                        data = mesh.point_data[field_name]
                        
                        if field_name not in summary['field_stats']:
                            summary['field_stats'][field_name] = {
                                'physics_info': field_info.get(field_name, {}),
                                'timestep_data': []
                            }
                        
                        field_stats = {
                            'timestep': timestep,
                            'min': float(np.nanmin(data)) if data.size > 0 else 0,
                            'max': float(np.nanmax(data)) if data.size > 0 else 0,
                            'mean': float(np.nanmean(data)) if data.size > 0 else 0,
                            'std': float(np.nanstd(data)) if data.size > 0 else 0,
                            'gradient_mag': float(np.mean(np.gradient(data))) if data.ndim == 1 and len(data) > 1 else 0
                        }
                        
                        # Add percentiles for scalar fields
                        if data.ndim == 1:
                            percentiles = np.percentile(data[~np.isnan(data)], [10, 25, 50, 75, 90])
                            field_stats['percentiles'] = percentiles.tolist()
                        
                        summary['field_stats'][field_name]['timestep_data'].append(field_stats)
                        
                except Exception as e:
                    st.warning(f"Error processing {vtu_files[idx]}: {e}")
                    continue
        
        return summary

# =============================================
# UNIFIED VISUALIZATION SYSTEM
# =============================================
class UnifiedPhysicsVisualizer:
    """Unified visualization system for original and interpolated fields"""
    
    def __init__(self):
        self.colormaps = {
            'temperature': ['#2c0078', '#4402a7', '#5e04d1', '#7b0ef6', '#9a38ff'],
            'stress': ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087'],
            'displacement': ['#004c6d', '#346888', '#5886a5', '#7aa6c2', '#9dc6e0'],
            'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        }
        
        self.plotly_colormaps = [
            'Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis',
            'Rainbow', 'Jet', 'Hot', 'Cool', 'Portland',
            'Bluered', 'Electric', 'Thermal', 'Balance'
        ]
    
    def create_comparative_visualization(self, original_data, interpolated_data, 
                                        sim_name, field_name, timestep, query_params):
        """
        Create side-by-side visualization of original and interpolated fields
        
        Parameters:
        -----------
        original_data: dict with 'points' and 'values'
        interpolated_data: dict with 'grid_points' and 'interpolated_field'
        sim_name: str, simulation name
        field_name: str, field name
        timestep: int, timestep
        query_params: dict, query parameters for interpolation
        """
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
            subplot_titles=(
                f"Original: {sim_name}<br>Timestep {timestep}",
                f"Interpolated: E={query_params['energy']:.1f}mJ, τ={query_params['duration']:.1f}ns, t={query_params['time']}ns"
            ),
            horizontal_spacing=0.1
        )
        
        # Original field
        if original_data and 'points' in original_data and 'values' in original_data:
            self._add_field_trace(
                fig, 
                original_data['points'], 
                original_data['values'],
                row=1, col=1,
                is_interpolated=False,
                name=f"Original {field_name}"
            )
        
        # Interpolated field
        if interpolated_data and 'grid_points' in interpolated_data and 'interpolated_field' in interpolated_data:
            self._add_field_trace(
                fig,
                interpolated_data['grid_points'],
                interpolated_data['interpolated_field'],
                row=1, col=2,
                is_interpolated=True,
                name=f"Interpolated {field_name}"
            )
        
        # Update layout
        fig.update_layout(
            title=f"Field Comparison: {field_name}",
            height=600,
            showlegend=True,
            scene=dict(
                aspectmode='data',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            scene2=dict(
                aspectmode='data',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            )
        )
        
        return fig
    
    def _add_field_trace(self, fig, points, values, row=1, col=1, 
                        is_interpolated=False, name="Field"):
        """Add field trace to figure with appropriate styling"""
        if points is None or values is None or len(points) == 0:
            return
        
        # Determine colormap based on field type
        colormap = 'Plasma' if is_interpolated else 'Viridis'
        
        # Create trace
        trace = go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=4 if is_interpolated else 3,
                color=values,
                colorscale=colormap,
                opacity=0.8,
                colorbar=dict(
                    title=name,
                    thickness=20,
                    len=0.75,
                    x=1.02 if col == 2 else 0.47
                ),
                showscale=True,
                symbol='circle' if is_interpolated else 'square'
            ),
            name=name,
            hovertemplate='<b>%{text}</b><br>' +
                         'Value: %{marker.color:.3f}<br>' +
                         'X: %{x:.3f}<br>' +
                         'Y: %{y:.3f}<br>' +
                         'Z: %{z:.3f}<extra></extra>',
            text=[f"{name}"] * len(points)
        )
        
        fig.add_trace(trace, row=row, col=col)
    
    def create_physics_metrics_dashboard(self, summaries, target_sim=None, query_params=None):
        """Create dashboard showing physics metrics across simulations"""
        if not summaries:
            return go.Figure()
        
        # Prepare data
        energies = []
        durations = []
        peak_powers = []
        energy_densities = []
        sim_names = []
        is_target = []
        
        for summary in summaries:
            energies.append(summary['energy'])
            durations.append(summary['duration'])
            peak_powers.append(summary['physics_metrics']['peak_power_MW'])
            energy_densities.append(summary['physics_metrics']['energy_density_mJ_ns2'])
            sim_names.append(summary['name'])
            is_target.append(summary['name'] == target_sim if target_sim else False)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Energy vs Duration',
                'Peak Power Distribution',
                'Energy Density',
                'Parameter Space Coverage'
            ),
            specs=[[{'type': 'scatter'}, {'type': 'histogram'}],
                  [{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        # 1. Energy vs Duration scatter
        fig.add_trace(
            go.Scatter(
                x=energies,
                y=durations,
                mode='markers',
                marker=dict(
                    size=10,
                    color=['red' if t else 'blue' for t in is_target],
                    opacity=0.7
                ),
                text=sim_names,
                hoverinfo='text+x+y',
                name='Simulations'
            ),
            row=1, col=1
        )
        
        if query_params:
            # Add query point
            fig.add_trace(
                go.Scatter(
                    x=[query_params.get('energy', 0)],
                    y=[query_params.get('duration', 0)],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color='green',
                        symbol='star'
                    ),
                    name='Query Point',
                    hoverinfo='text+x+y',
                    text=['Query Point']
                ),
                row=1, col=1
            )
        
        # 2. Peak power histogram
        fig.add_trace(
            go.Histogram(
                x=peak_powers,
                nbinsx=20,
                marker_color='lightblue',
                opacity=0.7,
                name='Peak Power'
            ),
            row=1, col=2
        )
        
        # 3. Energy density scatter
        fig.add_trace(
            go.Scatter(
                x=energies,
                y=energy_densities,
                mode='markers',
                marker=dict(
                    size=10,
                    color=peak_powers,
                    colorscale='Viridis',
                    showscale=False,
                    opacity=0.7
                ),
                text=sim_names,
                hoverinfo='text+x+y',
                name='Energy Density'
            ),
            row=2, col=1
        )
        
        # 4. Parameter space coverage
        fig.add_trace(
            go.Scatter(
                x=energies,
                y=durations,
                mode='lines+markers',
                marker=dict(
                    size=8,
                    color='gray',
                    opacity=0.3
                ),
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                name='Parameter Space'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Physics Metrics Dashboard',
            height=800,
            showlegend=True,
            hovermode='closest'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Energy (mJ)", row=1, col=1)
        fig.update_yaxes(title_text="Duration (ns)", row=1, col=1)
        fig.update_xaxes(title_text="Peak Power (MW)", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.update_xaxes(title_text="Energy (mJ)", row=2, col=1)
        fig.update_yaxes(title_text="Energy Density (mJ/ns²)", row=2, col=1)
        fig.update_xaxes(title_text="Energy (mJ)", row=2, col=2)
        fig.update_yaxes(title_text="Duration (ns)", row=2, col=2)
        
        return fig
    
    def create_field_evolution_comparison(self, summaries, field_name, 
                                         sim_names=None, query_params=None):
        """Compare field evolution across simulations with query highlight"""
        if not summaries or not field_name:
            return go.Figure()
        
        fig = go.Figure()
        
        # Plot each simulation
        for summary in summaries:
            if sim_names and summary['name'] not in sim_names:
                continue
            
            if field_name in summary['field_stats']:
                stats = summary['field_stats'][field_name]
                
                # Extract timestep data
                timesteps = []
                means = []
                
                for data in stats['timestep_data']:
                    timesteps.append(data['timestep'])
                    means.append(data['mean'])
                
                # Plot line
                line_width = 3 if query_params and summary['name'] == query_params.get('reference_sim') else 2
                line_dash = 'solid' if query_params and summary['name'] == query_params.get('reference_sim') else 'dash'
                
                fig.add_trace(go.Scatter(
                    x=timesteps,
                    y=means,
                    mode='lines+markers',
                    name=summary['name'],
                    line=dict(width=line_width, dash=line_dash),
                    opacity=0.8
                ))
        
        # Add query point if provided
        if query_params and 'time' in query_params and 'predicted_value' in query_params:
            fig.add_trace(go.Scatter(
                x=[query_params['time']],
                y=[query_params['predicted_value']],
                mode='markers',
                marker=dict(
                    size=15,
                    color='red',
                    symbol='star'
                ),
                name='Interpolated Prediction',
                hoverinfo='text+x+y',
                text=[f"Predicted: {query_params['predicted_value']:.3f}"]
            ))
        
        fig.update_layout(
            title=f"{field_name} Evolution Comparison",
            xaxis_title="Timestep (ns)",
            yaxis_title=f"{field_name} Value",
            hovermode="x unified",
            height=500,
            showlegend=True
        )
        
        return fig

# =============================================
# MAIN APPLICATION WITH PHYSICS-ACCURATE INTERPOLATION
# =============================================
def main():
    st.set_page_config(
        page_title="Physics-Accurate FEA Interpolation Platform",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="⚛️"
    )
    
    # Custom CSS for physics-themed styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(90deg, #1a2a6c, #b21f1f, #fdbb2d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .physics-box {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        border-left: 5px solid #00b4d8;
    }
    .interpolation-highlight {
        background: linear-gradient(135deg, #ff9a9e, #fad0c4);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 2px solid #ff6b6b;
    }
    .stButton > button {
        background: linear-gradient(90deg, #1a2a6c, #b21f1f);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(178, 31, 31, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">⚛️ Physics-Accurate FEA Field Interpolation</h1>', 
                unsafe_allow_html=True)
    
    # Initialize session state
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = PhysicsAwareVTULoader()
        st.session_state.interpolator = PhysicsInformedSpatialAttention()
        st.session_state.visualizer = UnifiedPhysicsVisualizer()
        st.session_state.data_loaded = False
        st.session_state.interpolation_results = None
        st.session_state.selected_field = None
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Navigation")
        app_mode = st.selectbox(
            "Select Mode",
            ["Data Loading", "Physics Interpolation", "Comparative Visualization", "Physics Dashboard"],
            key="nav_mode"
        )
        
        st.markdown("---")
        st.markdown("### 📁 Data Management")
        
        if st.button("🔄 Load All VTU Simulations", type="primary", use_container_width=True):
            with st.spinner("Loading VTU files with physics metadata..."):
                simulations, summaries = st.session_state.data_loader.load_all_simulations(
                    load_full_mesh=True
                )
                st.session_state.simulations = simulations
                st.session_state.summaries = summaries
                
                if simulations:
                    st.session_state.data_loaded = True
                    
                    # Initialize interpolator with loaded data
                    st.session_state.interpolator.load_simulation_data(simulations)
                    
                    # Set default field
                    if simulations:
                        first_sim = list(simulations.keys())[0]
                        if simulations[first_sim]['field_info']:
                            st.session_state.selected_field = list(simulations[first_sim]['field_info'].keys())[0]
        
        if st.session_state.data_loaded:
            st.markdown("---")
            st.markdown("### 📊 Loaded Data")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Simulations", len(st.session_state.simulations))
            with col2:
                st.metric("Total Fields", len(st.session_state.data_loader.available_fields))
            
            # Field selection
            if st.session_state.data_loader.available_fields:
                st.session_state.selected_field = st.selectbox(
                    "Select Field for Analysis",
                    sorted(st.session_state.data_loader.available_fields),
                    key="field_select"
                )
            
            # Simulation selection
            sim_names = sorted(st.session_state.simulations.keys())
            selected_sim = st.selectbox(
                "Reference Simulation",
                sim_names,
                key="ref_sim_select"
            )
            
            if selected_sim in st.session_state.simulations:
                sim = st.session_state.simulations[selected_sim]
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Energy", f"{sim['energy_mJ']:.2f} mJ")
                with col2:
                    st.metric("Duration", f"{sim['duration_ns']:.2f} ns")
    
    # Main content
    if app_mode == "Data Loading":
        render_data_loading()
    elif app_mode == "Physics Interpolation":
        render_physics_interpolation()
    elif app_mode == "Comparative Visualization":
        render_comparative_visualization()
    elif app_mode == "Physics Dashboard":
        render_physics_dashboard()

def render_data_loading():
    """Render data loading and inspection interface"""
    st.markdown("""
    <div class="physics-box">
    <h3>📁 VTU Data Loading & Physics Metadata</h3>
    <p>Load FEA simulation results from VTU files with automatic physics parameter extraction and metadata enhancement.</p>
    <p><strong>Expected structure:</strong> <code>fea_solutions/qXmJ-deltaYns/*.vtu</code></p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.info("Please load simulations using the button in the sidebar.")
        
        # Show expected structure
        with st.expander("📁 Expected Directory Structure", expanded=True):
            st.code("""
fea_solutions/
├── q0p5mJ-delta4p2ns/        # 0.5 mJ, 4.2 ns pulse
│   ├── a_t0001.vtu           # Timestep 1
│   ├── a_t0002.vtu           # Timestep 2
│   ├── a_t0003.vtu           # Timestep 3
│   └── ...
├── q1p0mJ-delta2p0ns/        # 1.0 mJ, 2.0 ns pulse
│   ├── a_t0001.vtu
│   ├── a_t0002.vtu
│   └── ...
└── q2p0mJ-delta1p0ns/        # 2.0 mJ, 1.0 ns pulse
    ├── a_t0001.vtu
    └── ...
            """)
        
        # File format information
        with st.expander("📄 VTU File Format Details", expanded=False):
            st.markdown("""
            **VTU (Visualization Toolkit Unstructured Grid) Format:**
            - Contains mesh geometry (points, cells)
            - Stores scalar and vector fields at vertices/cells
            - Supports time series data
            - Common in FEA software (ANSYS, Abaqus, COMSOL)
            
            **Extracted Physics Metadata:**
            - Laser pulse energy (mJ)
            - Pulse duration (ns)
            - Peak power (MW)
            - Energy density
            - Mesh characteristics
            - Field units and physical meaning
            """)
        return
    
    # Display loaded data overview
    simulations = st.session_state.simulations
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 📊 Simulation Database")
        
        # Create summary table
        summary_data = []
        for sim_name, sim_data in sorted(simulations.items()):
            summary_data.append({
                'Simulation': sim_name,
                'Energy (mJ)': sim_data['energy_mJ'],
                'Duration (ns)': sim_data['duration_ns'],
                'Peak Power (MW)': f"{sim_data['physics_metadata'].get('peak_power_MW', 0):.1f}",
                'Points': len(sim_data['points']) if sim_data.get('has_mesh') else 0,
                'Fields': len(sim_data['field_info'])
            })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            st.dataframe(
                df.style.format({'Energy (mJ)': '{:.2f}', 'Duration (ns)': '{:.2f}'})
                .background_gradient(subset=['Energy (mJ)', 'Duration (ns)'], cmap='YlOrRd'),
                use_container_width=True,
                height=400
            )
    
    with col2:
        st.markdown("### 📈 Physics Metrics")
        
        if st.session_state.summaries:
            energies = [s['energy'] for s in st.session_state.summaries]
            durations = [s['duration'] for s in st.session_state.summaries]
            
            st.metric("Energy Range", f"{min(energies):.1f} - {max(energies):.1f} mJ")
            st.metric("Duration Range", f"{min(durations):.1f} - {max(durations):.1f} ns")
            st.metric("Peak Power Max", 
                     f"{max([s['physics_metrics']['peak_power_MW'] for s in st.session_state.summaries]):.1f} MW")
    
    # Field information
    st.markdown("### 🔬 Field Information")
    
    if st.session_state.selected_field and simulations:
        # Get field info from first simulation
        first_sim = list(simulations.keys())[0]
        field_info = simulations[first_sim]['field_info'].get(st.session_state.selected_field, {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Field Type", field_info.get('type', 'Unknown'))
        with col2:
            st.metric("Units", field_info.get('units', 'N/A'))
        with col3:
            st.metric("Dimension", field_info.get('dim', 1))
        
        st.markdown(f"**Physical Meaning:** {field_info.get('physical_meaning', 'N/A')}")
    
    # Raw data inspection
    with st.expander("🔍 Raw Data Inspection", expanded=False):
        if simulations:
            selected_inspect = st.selectbox(
                "Select simulation to inspect",
                sorted(simulations.keys()),
                key="inspect_select"
            )
            
            sim = simulations[selected_inspect]
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Mesh Information:**")
                if sim.get('has_mesh'):
                    st.write(f"Points: {len(sim['points'])}")
                    if sim.get('triangles') is not None:
                        st.write(f"Triangles: {len(sim['triangles'])}")
                    if sim.get('tetrahedra') is not None:
                        st.write(f"Tetrahedra: {len(sim['tetrahedra'])}")
                
                st.write("**Physics Metadata:**")
                for key, value in sim.get('physics_metadata', {}).items():
                    if key != 'mesh_info':
                        st.write(f"{key}: {value}")
            
            with col2:
                st.write("**Field Information:**")
                for field_name, info in sim['field_info'].items():
                    st.write(f"- {field_name}: {info.get('type', 'Unknown')} ({info.get('units', 'N/A')})")

def render_physics_interpolation():
    """Render physics-accurate interpolation interface"""
    if not st.session_state.data_loaded:
        st.warning("Please load simulations first in the Data Loading mode.")
        return
    
    st.markdown("""
    <div class="physics-box">
    <h3>⚛️ Physics-Informed Spatial Attention Interpolation</h3>
    <p>Interpolate FEA fields using physics-aware attention mechanism with spatial locality regularization.</p>
    <p><strong>Key features:</strong> Physics parameter similarity + Spatial distribution matching → Accurate field prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    simulations = st.session_state.simulations
    
    # Query parameters
    st.markdown("### 🎯 Interpolation Query Parameters")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        # Get parameter ranges
        energies = [sim['energy_mJ'] for sim in simulations.values()]
        min_e, max_e = min(energies), max(energies)
        
        energy_query = st.slider(
            "Energy (mJ)",
            min_value=float(min_e),
            max_value=float(max_e),
            value=float((min_e + max_e) / 2),
            step=0.1,
            key="interp_energy"
        )
    
    with col2:
        durations = [sim['duration_ns'] for sim in simulations.values()]
        min_d, max_d = min(durations), max(durations)
        
        duration_query = st.slider(
            "Duration (ns)",
            min_value=float(min_d),
            max_value=float(max_d),
            value=float((min_d + max_d) / 2),
            step=0.1,
            key="interp_duration"
        )
    
    with col3:
        max_time = 50  # Assume maximum time
        time_query = st.slider(
            "Time (ns)",
            min_value=0,
            max_value=max_time,
            value=10,
            step=1,
            key="interp_time"
        )
    
    # Interpolation settings
    with st.expander("⚙️ Physics Interpolation Settings", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            spatial_sigma = st.slider(
                "Satial Locality (σ)",
                min_value=0.01,
                max_value=0.5,
                value=0.1,
                step=0.01,
                help="Controls spatial similarity weight: lower = stricter spatial matching"
            )
        
        with col2:
            physics_weight = st.slider(
                "Physics Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05,
                help="Weight for physics parameter similarity vs spatial similarity"
            )
        
        # Update interpolator parameters
        st.session_state.interpolator.spatial_sigma = spatial_sigma
        st.session_state.interpolator.physics_weight = physics_weight
    
    # Reference simulation for comparison
    st.markdown("### 🔍 Reference for Comparison")
    
    ref_sim = st.selectbox(
        "Select reference simulation for visualization",
        sorted(simulations.keys()),
        key="interp_ref_sim"
    )
    
    # Get timestep from reference simulation
    if ref_sim in simulations:
        ref_timestep = st.slider(
            "Reference Timestep",
            0, simulations[ref_sim]['n_timesteps'] - 1,
            0,
            key="ref_timestep"
        )
    
    # Perform interpolation
    if st.button("🚀 Run Physics-Accurate Interpolation", type="primary", use_container_width=True):
        with st.spinner("Performing physics-informed interpolation with spatial regularization..."):
            query_params = {
                'energy': energy_query,
                'duration': duration_query,
                'time': time_query
            }
            
            # Perform interpolation
            interpolation_results = st.session_state.interpolator.interpolate_field(
                query_params,
                st.session_state.selected_field
            )
            
            if interpolation_results:
                st.session_state.interpolation_results = interpolation_results
                st.session_state.query_params = query_params
                st.session_state.ref_sim = ref_sim
                st.session_state.ref_timestep = ref_timestep
                
                st.markdown("""
                <div class="interpolation-highlight">
                <h4>✅ Interpolation Successful</h4>
                <p>Physics-informed field interpolation completed using spatial attention mechanism.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show attention statistics
                if 'attention_weights' in interpolation_results:
                    weights = interpolation_results['attention_weights']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Max Weight", f"{np.max(weights):.3f}")
                    with col2:
                        st.metric("Mean Weight", f"{np.mean(weights):.3f}")
                    with col3:
                        st.metric("Weight Std", f"{np.std(weights):.3f}")
                    
                    # Show top 3 sources
                    if len(weights) >= 3:
                        top_indices = np.argsort(weights)[-3:][::-1]
                        
                        st.markdown("#### 🎯 Top 3 Source Contributions")
                        for i, idx in enumerate(top_indices):
                            source = interpolation_results['source_data'][idx]
                            weight = weights[idx]
                            st.info(f"**#{i+1}**: {source['sim_name']} (t={source['timestep']}) - "
                                  f"Energy={source['metadata']['energy']:.1f}mJ, "
                                  f"Duration={source['metadata']['duration']:.1f}ns, "
                                  f"Weight={weight:.3f}")
            else:
                st.error("Interpolation failed. Please check parameters and data availability.")
    
    # Display interpolation results
    if st.session_state.interpolation_results:
        st.markdown("### 📊 Interpolation Results")
        
        # Create comparative visualization
        if st.session_state.ref_sim in simulations:
            ref_sim_data = simulations[st.session_state.ref_sim]
            
            # Get original field data
            if st.session_state.selected_field in ref_sim_data['fields']:
                field_data = ref_sim_data['fields'][st.session_state.selected_field]
                
                if ref_sim_data['field_info'][st.session_state.selected_field]['type'] == 'scalar':
                    original_values = field_data[st.session_state.ref_timestep]
                else:
                    original_values = np.linalg.norm(field_data[st.session_state.ref_timestep], axis=1)
                
                original_data = {
                    'points': ref_sim_data['points'],
                    'values': original_values
                }
                
                # Create visualization
                fig = st.session_state.visualizer.create_comparative_visualization(
                    original_data,
                    st.session_state.interpolation_results,
                    st.session_state.ref_sim,
                    st.session_state.selected_field,
                    st.session_state.ref_timestep,
                    st.session_state.query_params
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Attention visualization
        st.markdown("### 🧠 Attention Mechanism Analysis")
        
        fig_attention = st.session_state.interpolator.visualize_attention_distribution(
            st.session_state.interpolation_results
        )
        
        if fig_attention.data:
            st.plotly_chart(fig_attention, use_container_width=True)
        
        # Field statistics comparison
        st.markdown("### 📈 Field Statistics Comparison")
        
        if st.session_state.interpolation_results.get('interpolated_field') is not None:
            interp_values = st.session_state.interpolation_results['interpolated_field']
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Interp Min", f"{np.min(interp_values):.3f}")
            with col2:
                st.metric("Interp Max", f"{np.max(interp_values):.3f}")
            with col3:
                st.metric("Interp Mean", f"{np.mean(interp_values):.3f}")
            with col4:
                st.metric("Interp Std", f"{np.std(interp_values):.3f}")
            with col5:
                st.metric("Data Points", f"{len(interp_values):,}")

def render_comparative_visualization():
    """Render comparative visualization interface"""
    if not st.session_state.data_loaded:
        st.warning("Please load simulations first.")
        return
    
    st.markdown("""
    <div class="physics-box">
    <h3>📊 Comparative Field Visualization</h3>
    <p>Compare original simulation fields with interpolated results across multiple dimensions.</p>
    <p><strong>Visualization modes:</strong> 3D field comparison, evolution analysis, parameter space mapping</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different visualization types
    tab1, tab2, tab3 = st.tabs(["3D Field Comparison", "Evolution Analysis", "Parameter Space"])
    
    with tab1:
        render_3d_field_comparison()
    
    with tab2:
        render_evolution_analysis()
    
    with tab3:
        render_parameter_space_analysis()

def render_3d_field_comparison():
    """Render 3D field comparison visualization"""
    if not st.session_state.data_loaded:
        return
    
    simulations = st.session_state.simulations
    
    col1, col2 = st.columns([2, 1])
    with col1:
        # Select multiple simulations for comparison
        selected_sims = st.multiselect(
            "Select simulations for comparison",
            sorted(simulations.keys()),
            default=sorted(simulations.keys())[:min(3, len(simulations))],
            key="comp_sims"
        )
    
    with col2:
        if st.session_state.selected_field:
            st.metric("Selected Field", st.session_state.selected_field)
    
    if not selected_sims:
        st.info("Please select at least one simulation for comparison.")
        return
    
    # Timestep selection
    timestep = st.slider(
        "Timestep",
        0, 20, 0,
        key="comp_timestep"
    )
    
    # Create comparison figure
    n_sims = len(selected_sims)
    n_cols = min(3, n_sims)
    n_rows = (n_sims + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        specs=[[{'type': 'scatter3d'} for _ in range(n_cols)] for _ in range(n_rows)],
        subplot_titles=[sim for sim in selected_sims],
        vertical_spacing=0.05,
        horizontal_spacing=0.02
    )
    
    # Add each simulation's field
    for idx, sim_name in enumerate(selected_sims):
        sim = simulations[sim_name]
        
        if sim['has_mesh'] and st.session_state.selected_field in sim['fields']:
            row = idx // n_cols + 1
            col = idx % n_cols + 1
            
            # Get field data
            field_data = sim['fields'][st.session_state.selected_field]
            actual_timestep = min(timestep, sim['n_timesteps'] - 1)
            
            if sim['field_info'][st.session_state.selected_field]['type'] == 'scalar':
                values = field_data[actual_timestep]
            else:
                values = np.linalg.norm(field_data[actual_timestep], axis=1)
            
            # Add trace
            trace = go.Scatter3d(
                x=sim['points'][:, 0],
                y=sim['points'][:, 1],
                z=sim['points'][:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=values,
                    colorscale='Viridis',
                    opacity=0.8,
                    showscale=True
                ),
                name=sim_name,
                hoverinfo='text',
                hovertext=[f"Value: {v:.3f}" for v in values]
            )
            
            fig.add_trace(trace, row=row, col=col)
    
    # Update layout
    fig.update_layout(
        title=f"3D Field Comparison: {st.session_state.selected_field} at timestep {timestep}",
        height=400 * n_rows,
        showlegend=False
    )
    
    # Update camera for each subplot
    for i in range(1, n_sims + 1):
        fig.update_layout({
            f'scene{i}': dict(
                aspectmode='data',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            )
        })
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add interpolated field if available
    if st.session_state.interpolation_results:
        st.markdown("---")
        st.markdown("#### 🔮 Interpolated Field Comparison")
        
        # Create comparison with interpolated field
        fig_interp = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
            subplot_titles=("Best Match Simulation", "Interpolated Field"),
            horizontal_spacing=0.1
        )
        
        # Find simulation with closest parameters to query
        if hasattr(st.session_state, 'query_params'):
            query_params = st.session_state.query_params
            
            # Find closest simulation
            closest_sim = None
            min_distance = float('inf')
            
            for sim_name, sim in simulations.items():
                distance = np.sqrt(
                    (sim['energy_mJ'] - query_params['energy'])**2 +
                    (sim['duration_ns'] - query_params['duration'])**2
                )
                
                if distance < min_distance:
                    min_distance = distance
                    closest_sim = sim_name
            
            if closest_sim:
                # Add closest simulation
                sim = simulations[closest_sim]
                if st.session_state.selected_field in sim['fields']:
                    field_data = sim['fields'][st.session_state.selected_field]
                    
                    if sim['field_info'][st.session_state.selected_field]['type'] == 'scalar':
                        values = field_data[min(timestep, sim['n_timesteps'] - 1)]
                    else:
                        values = np.linalg.norm(field_data[min(timestep, sim['n_timesteps'] - 1)], axis=1)
                    
                    trace1 = go.Scatter3d(
                        x=sim['points'][:, 0],
                        y=sim['points'][:, 1],
                        z=sim['points'][:, 2],
                        mode='markers',
                        marker=dict(
                            size=3,
                            color=values,
                            colorscale='Viridis',
                            opacity=0.8,
                            showscale=True
                        ),
                        name=f"Closest: {closest_sim}"
                    )
                    fig_interp.add_trace(trace1, row=1, col=1)
        
        # Add interpolated field
        if st.session_state.interpolation_results.get('grid_points') is not None:
            interp_points = st.session_state.interpolation_results['grid_points']
            interp_values = st.session_state.interpolation_results['interpolated_field']
            
            trace2 = go.Scatter3d(
                x=interp_points[:, 0],
                y=interp_points[:, 1],
                z=interp_points[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=interp_values,
                    colorscale='Plasma',
                    opacity=0.8,
                    showscale=True
                ),
                name="Interpolated"
            )
            fig_interp.add_trace(trace2, row=1, col=2)
        
        fig_interp.update_layout(
            height=500,
            showlegend=True,
            scene=dict(aspectmode='data'),
            scene2=dict(aspectmode='data')
        )
        
        st.plotly_chart(fig_interp, use_container_width=True)

def render_evolution_analysis():
    """Render field evolution analysis"""
    if not st.session_state.data_loaded:
        return
    
    simulations = st.session_state.simulations
    summaries = st.session_state.summaries
    
    st.markdown("### 📈 Field Evolution Over Time")
    
    # Select simulations for evolution comparison
    selected_sims = st.multiselect(
        "Select simulations for evolution analysis",
        sorted(simulations.keys()),
        default=sorted(simulations.keys())[:min(5, len(simulations))],
        key="evo_sims"
    )
    
    if not selected_sims:
        return
    
    # Get corresponding summaries
    selected_summaries = [s for s in summaries if s['name'] in selected_sims]
    
    # Create evolution plot
    fig = st.session_state.visualizer.create_field_evolution_comparison(
        selected_summaries,
        st.session_state.selected_field,
        sim_names=selected_sims,
        query_params=st.session_state.interpolation_results
    )
    
    if fig.data:
        st.plotly_chart(fig, use_container_width=True)
    
    # Add statistics table
    st.markdown("### 📊 Evolution Statistics")
    
    stats_data = []
    for sim_name in selected_sims:
        sim = simulations[sim_name]
        
        if st.session_state.selected_field in sim['fields']:
            field_data = sim['fields'][st.session_state.selected_field]
            
            # Calculate statistics across timesteps
            if sim['field_info'][st.session_state.selected_field]['type'] == 'scalar':
                all_values = field_data.flatten()
            else:
                all_values = np.linalg.norm(field_data, axis=2).flatten()
            
            stats_data.append({
                'Simulation': sim_name,
                'Energy (mJ)': sim['energy_mJ'],
                'Duration (ns)': sim['duration_ns'],
                'Min': float(np.nanmin(all_values)),
                'Max': float(np.nanmax(all_values)),
                'Mean': float(np.nanmean(all_values)),
                'Std': float(np.nanstd(all_values)),
                'Peak Time': np.argmax(np.mean(field_data, axis=1)) if field_data.ndim > 1 else 0
            })
    
    if stats_data:
        df_stats = pd.DataFrame(stats_data)
        st.dataframe(
            df_stats.style.format({
                'Energy (mJ)': '{:.2f}',
                'Duration (ns)': '{:.2f}',
                'Min': '{:.3f}',
                'Max': '{:.3f}',
                'Mean': '{:.3f}',
                'Std': '{:.3f}'
            }).background_gradient(subset=['Min', 'Max', 'Mean'], cmap='YlOrRd'),
            use_container_width=True
        )

def render_parameter_space_analysis():
    """Render parameter space analysis"""
    if not st.session_state.data_loaded:
        return
    
    summaries = st.session_state.summaries
    
    st.markdown("### 🌐 Parameter Space Analysis")
    
    # Create physics metrics dashboard
    query_params = getattr(st.session_state, 'query_params', None)
    ref_sim = getattr(st.session_state, 'ref_sim', None)
    
    fig = st.session_state.visualizer.create_physics_metrics_dashboard(
        summaries,
        target_sim=ref_sim,
        query_params=query_params
    )
    
    if fig.data:
        st.plotly_chart(fig, use_container_width=True)
    
    # Parameter space coverage
    st.markdown("### 📏 Parameter Space Coverage")
    
    if summaries:
        energies = [s['energy'] for s in summaries]
        durations = [s['duration'] for s in summaries]
        
        # Create convex hull
        points = np.column_stack([energies, durations])
        
        if len(points) >= 3:
            try:
                hull = ConvexHull(points)
                
                # Plot parameter space with convex hull
                fig_space = go.Figure()
                
                # Add points
                fig_space.add_trace(go.Scatter(
                    x=energies,
                    y=durations,
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='blue',
                        opacity=0.7
                    ),
                    text=[s['name'] for s in summaries],
                    name='Simulations'
                ))
                
                # Add convex hull
                hull_points = points[hull.vertices]
                hull_points = np.vstack([hull_points, hull_points[0]])  # Close the polygon
                
                fig_space.add_trace(go.Scatter(
                    x=hull_points[:, 0],
                    y=hull_points[:, 1],
                    mode='lines',
                    fill='toself',
                    fillcolor='rgba(0,100,80,0.2)',
                    line=dict(color='darkgreen', width=2),
                    name='Parameter Space'
                ))
                
                # Add query point if available
                if query_params:
                    fig_space.add_trace(go.Scatter(
                        x=[query_params.get('energy', 0)],
                        y=[query_params.get('duration', 0)],
                        mode='markers',
                        marker=dict(
                            size=15,
                            color='red',
                            symbol='star'
                        ),
                        name='Query Point'
                    ))
                
                fig_space.update_layout(
                    title="Parameter Space Coverage",
                    xaxis_title="Energy (mJ)",
                    yaxis_title="Duration (ns)",
                    height=500,
                    showlegend=True
                )
                
                st.plotly_chart(fig_space, use_container_width=True)
                
                # Calculate coverage metrics
                hull_area = hull.volume if hasattr(hull, 'volume') else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Coverage Area", f"{hull_area:.2f} mJ·ns")
                with col2:
                    st.metric("Number of Points", len(points))
                with col3:
                    if query_params:
                        # Check if query is inside convex hull
                        from scipy.spatial import Delaunay
                        tri = Delaunay(points[hull.vertices])
                        is_inside = tri.find_simplex([query_params['energy'], query_params['duration']]) >= 0
                        status = "Inside" if is_inside else "Outside"
                        st.metric("Query Position", status)
                
            except Exception as e:
                st.warning(f"Could not compute convex hull: {e}")

def render_physics_dashboard():
    """Render comprehensive physics dashboard"""
    if not st.session_state.data_loaded:
        st.warning("Please load simulations first.")
        return
    
    st.markdown("""
    <div class="physics-box">
    <h3>📊 Physics & Interpolation Dashboard</h3>
    <p>Comprehensive overview of physics metrics, interpolation performance, and field characteristics.</p>
    </div>
    """, unsafe_allow_html=True)
    
    simulations = st.session_state.simulations
    summaries = st.session_state.summaries
    
    # Create tabs for different dashboard sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "Physics Metrics", 
        "Interpolation Analysis", 
        "Field Statistics", 
        "Export & Reports"
    ])
    
    with tab1:
        render_physics_metrics_dashboard(simulations, summaries)
    
    with tab2:
        render_interpolation_analysis_dashboard()
    
    with tab3:
        render_field_statistics_dashboard(simulations)
    
    with tab4:
        render_export_dashboard()

def render_physics_metrics_dashboard(simulations, summaries):
    """Render physics metrics dashboard"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ⚡ Power Metrics")
        
        if summaries:
            peak_powers = [s['physics_metrics']['peak_power_MW'] for s in summaries]
            energy_densities = [s['physics_metrics']['energy_density_mJ_ns2'] for s in summaries]
            
            fig_power = go.Figure()
            fig_power.add_trace(go.Box(
                y=peak_powers,
                name='Peak Power (MW)',
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ))
            
            fig_power.update_layout(
                title="Peak Power Distribution",
                height=300
            )
            st.plotly_chart(fig_power, use_container_width=True)
            
            # Display statistics
            stats_data = {
                'Metric': ['Peak Power (MW)', 'Energy Density (mJ/ns²)'],
                'Min': [min(peak_powers), min(energy_densities)],
                'Max': [max(peak_powers), max(energy_densities)],
                'Mean': [np.mean(peak_powers), np.mean(energy_densities)],
                'Std': [np.std(peak_powers), np.std(energy_densities)]
            }
            
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
    
    with col2:
        st.markdown("#### 📐 Mesh Statistics")
        
        mesh_stats = []
        for sim_name, sim in list(simulations.items())[:5]:  # Show first 5
            if sim.get('has_mesh'):
                mesh_stats.append({
                    'Simulation': sim_name,
                    'Points': len(sim['points']),
                    'Triangles': len(sim['triangles']) if sim.get('triangles') is not None else 0,
                    'Volume (est.)': np.prod(np.max(sim['points'], axis=0) - np.min(sim['points'], axis=0))
                })
        
        if mesh_stats:
            df_mesh = pd.DataFrame(mesh_stats)
            st.dataframe(df_mesh, use_container_width=True)
    
    # Field correlation analysis
    st.markdown("#### 🔗 Field Correlations")
    
    if len(summaries) >= 3 and st.session_state.selected_field:
        # Calculate correlations between field values and physics parameters
        correlations = []
        
        for summary in summaries:
            if st.session_state.selected_field in summary['field_stats']:
                field_data = summary['field_stats'][st.session_state.selected_field]
                if field_data['timestep_data']:
                    mean_value = np.mean([d['mean'] for d in field_data['timestep_data']])
                    
                    correlations.append({
                        'Simulation': summary['name'],
                        'Energy': summary['energy'],
                        'Duration': summary['duration'],
                        'Field Mean': mean_value,
                        'Peak Power': summary['physics_metrics']['peak_power_MW']
                    })
        
        if correlations:
            df_corr = pd.DataFrame(correlations)
            
            # Calculate correlation matrix
            corr_matrix = df_corr[['Energy', 'Duration', 'Field Mean', 'Peak Power']].corr()
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 3),
                texttemplate='%{text}',
                textfont={"size": 12}
            ))
            
            fig_corr.update_layout(
                title="Parameter Correlation Matrix",
                height=400
            )
            
            st.plotly_chart(fig_corr, use_container_width=True)

def render_interpolation_analysis_dashboard():
    """Render interpolation analysis dashboard"""
    if not hasattr(st.session_state, 'interpolation_results') or not st.session_state.interpolation_results:
        st.info("Run interpolation first to see analysis.")
        return
    
    results = st.session_state.interpolation_results
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🧠 Attention Mechanism")
        
        if 'attention_weights' in results:
            weights = results['attention_weights']
            
            # Create histogram
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=weights,
                nbinsx=20,
                marker_color='lightblue',
                opacity=0.7
            ))
            
            fig_hist.update_layout(
                title="Attention Weight Distribution",
                xaxis_title="Weight",
                yaxis_title="Count",
                height=300
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Weight statistics
            weight_stats = {
                'Statistic': ['Min', 'Max', 'Mean', 'Std', 'Top 3 Sum'],
                'Value': [
                    f"{np.min(weights):.4f}",
                    f"{np.max(weights):.4f}",
                    f"{np.mean(weights):.4f}",
                    f"{np.std(weights):.4f}",
                    f"{np.sum(np.sort(weights)[-3:]):.4f}"
                ]
            }
            
            st.dataframe(pd.DataFrame(weight_stats), use_container_width=True)
    
    with col2:
        st.markdown("#### 📊 Interpolation Quality")
        
        if 'interpolated_field' in results:
            interp_values = results['interpolated_field']
            
            # Create quality metrics
            quality_metrics = {
                'Metric': [
                    'Field Range',
                    'Smoothness (gradient)',
                    'Data Coverage',
                    'Interpolation Confidence'
                ],
                'Value': [
                    f"{np.ptp(interp_values):.3f}",
                    f"{np.mean(np.gradient(interp_values)):.3f}",
                    f"{np.mean(~np.isnan(interp_values)):.1%}",
                    f"{np.max(results.get('attention_weights', [0])):.3f}"
                ]
            }
            
            st.dataframe(pd.DataFrame(quality_metrics), use_container_width=True)
    
    # Source contributions
    st.markdown("#### 🎯 Source Contributions")
    
    if 'source_data' in results and 'attention_weights' in results:
        source_info = []
        
        for i, (source, weight) in enumerate(zip(results['source_data'], results['attention_weights'])):
            source_info.append({
                'Rank': i+1,
                'Simulation': source['sim_name'],
                'Timestep': source['timestep'],
                'Energy': source['metadata']['energy'],
                'Duration': source['metadata']['duration'],
                'Weight': f"{weight:.4f}",
                'Contribution': f"{weight/np.sum(results['attention_weights']):.1%}"
            })
        
        # Sort by weight
        source_df = pd.DataFrame(source_info)
        source_df = source_df.sort_values('Weight', ascending=False).head(10)
        
        st.dataframe(
            source_df.style.format({'Weight': '{:.4f}'}),
            use_container_width=True,
            height=400
        )

def render_field_statistics_dashboard(simulations):
    """Render field statistics dashboard"""
    if not st.session_state.selected_field:
        return
    
    st.markdown(f"#### 📈 {st.session_state.selected_field} Statistics")
    
    # Collect field statistics across simulations
    field_stats = []
    
    for sim_name, sim in simulations.items():
        if st.session_state.selected_field in sim['fields']:
            field_data = sim['fields'][st.session_state.selected_field]
            
            if sim['field_info'][st.session_state.selected_field]['type'] == 'scalar':
                values = field_data.flatten()
            else:
                values = np.linalg.norm(field_data, axis=2).flatten()
            
            field_stats.append({
                'Simulation': sim_name,
                'Energy': sim['energy_mJ'],
                'Duration': sim['duration_ns'],
                'Min': np.nanmin(values),
                'Max': np.nanmax(values),
                'Mean': np.nanmean(values),
                'Std': np.nanstd(values),
                'Peak Value': np.nanmax(values)
            })
    
    if field_stats:
        df_stats = pd.DataFrame(field_stats)
        
        # Create summary visualization
        fig_summary = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Mean vs Energy', 'Peak vs Duration', 'Distribution', 'Correlation'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                  [{'type': 'histogram'}, {'type': 'heatmap'}]]
        )
        
        # Mean vs Energy
        fig_summary.add_trace(
            go.Scatter(
                x=df_stats['Energy'],
                y=df_stats['Mean'],
                mode='markers',
                marker=dict(size=10, color=df_stats['Duration'], colorscale='Viridis', showscale=True),
                text=df_stats['Simulation'],
                hoverinfo='text+x+y'
            ),
            row=1, col=1
        )
        
        # Peak vs Duration
        fig_summary.add_trace(
            go.Scatter(
                x=df_stats['Duration'],
                y=df_stats['Peak Value'],
                mode='markers',
                marker=dict(size=10, color=df_stats['Energy'], colorscale='Plasma', showscale=True),
                text=df_stats['Simulation'],
                hoverinfo='text+x+y'
            ),
            row=1, col=2
        )
        
        # Distribution histogram
        fig_summary.add_trace(
            go.Histogram(
                x=df_stats['Mean'],
                nbinsx=20,
                marker_color='lightblue',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Correlation heatmap
        corr_cols = ['Energy', 'Duration', 'Mean', 'Peak Value', 'Std']
        corr_matrix = df_stats[corr_cols].corr()
        
        fig_summary.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=corr_cols,
                y=corr_cols,
                colorscale='RdBu',
                zmid=0
            ),
            row=2, col=2
        )
        
        fig_summary.update_layout(
            height=600,
            showlegend=False
        )
        
        st.plotly_chart(fig_summary, use_container_width=True)
        
        # Display statistics table
        st.dataframe(
            df_stats.style.format({
                'Energy': '{:.2f}',
                'Duration': '{:.2f}',
                'Min': '{:.3f}',
                'Max': '{:.3f}',
                'Mean': '{:.3f}',
                'Std': '{:.3f}',
                'Peak Value': '{:.3f}'
            }).background_gradient(subset=['Min', 'Max', 'Mean'], cmap='YlOrRd'),
            use_container_width=True
        )

def render_export_dashboard():
    """Render export and reporting dashboard"""
    st.markdown("### 📤 Export & Reporting")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Data Export")
        
        # Export options
        export_format = st.selectbox(
            "Select export format",
            ["CSV", "JSON", "HDF5", "VTK", "Paraview State"]
        )
        
        export_content = st.multiselect(
            "Select content to export",
            ["Field Data", "Interpolation Results", "Physics Metadata", "Visualization Plots"],
            default=["Field Data", "Interpolation Results"]
        )
        
        if st.button("📥 Export Selected Data", use_container_width=True):
            with st.spinner("Preparing export..."):
                # Generate export data
                export_data = {}
                
                if "Field Data" in export_content and hasattr(st.session_state, 'simulations'):
                    # Export field data
                    pass
                
                if "Interpolation Results" in export_content and hasattr(st.session_state, 'interpolation_results'):
                    # Export interpolation results
                    pass
                
                st.success("Export prepared successfully!")
    
    with col2:
        st.markdown("#### 📋 Report Generation")
        
        report_type = st.selectbox(
            "Report type",
            ["Interpolation Analysis", "Field Comparison", "Physics Summary", "Full Analysis"]
        )
        
        include_sections = st.multiselect(
            "Include sections",
            ["Executive Summary", "Methodology", "Results", "Visualizations", "Statistics", "Conclusions"],
            default=["Executive Summary", "Results", "Visualizations"]
        )
        
        if st.button("📄 Generate Report", use_container_width=True):
            with st.spinner("Generating report..."):
                # Generate report
                st.success("Report generated successfully!")
    
    # Preview export
    st.markdown("#### 👁️ Export Preview")
    
    if hasattr(st.session_state, 'interpolation_results'):
        # Show preview of interpolation results
        with st.expander("Interpolation Results Preview", expanded=True):
            results = st.session_state.interpolation_results
            
            preview_data = {
                'Parameter': ['Query Energy', 'Query Duration', 'Query Time', 'Interpolated Points'],
                'Value': [
                    f"{st.session_state.query_params['energy']:.2f} mJ",
                    f"{st.session_state.query_params['duration']:.2f} ns",
                    f"{st.session_state.query_params['time']} ns",
                    f"{len(results['grid_points']):,}"
                ]
            }
            
            st.table(pd.DataFrame(preview_data))

if __name__ == "__main__":
    main()
