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
from scipy.interpolate import griddata, RBFInterpolator, LinearNDInterpolator, CloughTocher2DInterpolator
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay, ConvexHull, KDTree
from skimage import measure
import pyvista as pv
import tempfile
from stpyvista import stpyvista
import subprocess
import threading

warnings.filterwarnings('ignore')

# =============================================
# PATH CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEA_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "fea_solutions")
INTERPOLATED_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "interpolated_solutions")
os.makedirs(FEA_SOLUTIONS_DIR, exist_ok=True)
os.makedirs(INTERPOLATED_SOLUTIONS_DIR, exist_ok=True)

# =============================================
# START XVFB FOR HEADLESS RENDERING
# =============================================
def start_xvfb():
    """Start Xvfb for headless rendering if not already running"""
    try:
        # Check if Xvfb is already running
        result = subprocess.run(['pgrep', 'Xvfb'], capture_output=True)
        if result.returncode != 0:
            # Start Xvfb on display :99
            subprocess.Popen(['Xvfb', ':99', '-screen', '0', '1920x1080x24'])
            os.environ['DISPLAY'] = ':99'
            st.info("Started Xvfb for headless rendering")
    except Exception as e:
        st.warning(f"Could not start Xvfb: {e}")

# Start Xvfb if needed
if "IS_XVFB_RUNNING" not in st.session_state:
    start_xvfb()
    st.session_state.IS_XVFB_RUNNING = True

# =============================================
# INTERPOLATED SOLUTIONS MANAGER
# =============================================
class InterpolatedSolutionsManager:
    """Manager for handling interpolated solutions alongside original ones"""
    
    def __init__(self):
        self.interpolated_simulations = {}
        self.interpolated_summaries = []
        
    def create_interpolated_solution(self, sim_name, field, timestep, 
                                    interpolated_points, interpolated_values,
                                    method='rbf', energy=None, duration=None):
        """Create a new interpolated solution"""
        
        # Create a unique ID for this interpolated solution
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        interpolated_id = f"{sim_name}_{field}_t{timestep}_{method}_{timestamp}"
        
        # Get original simulation for reference
        original_sim = st.session_state.simulations.get(sim_name, {})
        
        # Create interpolated simulation data structure
        interpolated_sim = {
            'name': interpolated_id,
            'original_simulation': sim_name,
            'original_field': field,
            'original_timestep': timestep,
            'interpolation_method': method,
            'energy_mJ': energy or original_sim.get('energy_mJ', 0),
            'duration_ns': duration or original_sim.get('duration_ns', 0),
            'points': interpolated_points,
            'values': interpolated_values,
            'n_points': len(interpolated_points),
            'field_type': 'scalar' if interpolated_values.ndim == 1 else 'vector',
            'is_interpolated': True,
            'created_at': timestamp
        }
        
        # Store in session state
        self.interpolated_simulations[interpolated_id] = interpolated_sim
        
        # Create summary
        summary = self.create_interpolated_summary(interpolated_sim)
        self.interpolated_summaries.append(summary)
        
        # Save to disk
        self.save_interpolated_solution(interpolated_sim)
        
        return interpolated_sim
    
    def create_interpolated_summary(self, interpolated_sim):
        """Create summary statistics for interpolated solution"""
        values = interpolated_sim['values']
        
        if values.ndim == 1:
            clean_values = values[~np.isnan(values)]
        else:
            magnitude = np.linalg.norm(values, axis=1)
            clean_values = magnitude[~np.isnan(magnitude)]
        
        if len(clean_values) > 0:
            percentiles = np.percentile(clean_values, [10, 25, 50, 75, 90])
        else:
            percentiles = np.zeros(5)
        
        summary = {
            'name': interpolated_sim['name'],
            'energy': interpolated_sim['energy_mJ'],
            'duration': interpolated_sim['duration_ns'],
            'original_simulation': interpolated_sim['original_simulation'],
            'interpolation_method': interpolated_sim['interpolation_method'],
            'is_interpolated': True,
            'timesteps': [1],  # Single timestep for interpolated
            'field_stats': {
                'interpolated_field': {
                    'min': [float(np.min(clean_values))] if len(clean_values) > 0 else [0.0],
                    'max': [float(np.max(clean_values))] if len(clean_values) > 0 else [0.0],
                    'mean': [float(np.mean(clean_values))] if len(clean_values) > 0 else [0.0],
                    'std': [float(np.std(clean_values))] if len(clean_values) > 0 else [0.0],
                    'q25': [percentiles[1]] if len(percentiles) > 0 else [0.0],
                    'q50': [percentiles[2]] if len(percentiles) > 0 else [0.0],
                    'q75': [percentiles[3]] if len(percentiles) > 0 else [0.0],
                    'percentiles': [percentiles]
                }
            }
        }
        
        return summary
    
    def save_interpolated_solution(self, interpolated_sim):
        """Save interpolated solution to disk as numpy file"""
        save_path = os.path.join(INTERPOLATED_SOLUTIONS_DIR, f"{interpolated_sim['name']}.npz")
        
        np.savez_compressed(
            save_path,
            points=interpolated_sim['points'],
            values=interpolated_sim['values'],
            metadata={
                'name': interpolated_sim['name'],
                'original_simulation': interpolated_sim['original_simulation'],
                'original_field': interpolated_sim['original_field'],
                'original_timestep': interpolated_sim['original_timestep'],
                'interpolation_method': interpolated_sim['interpolation_method'],
                'energy_mJ': interpolated_sim['energy_mJ'],
                'duration_ns': interpolated_sim['duration_ns'],
                'field_type': interpolated_sim['field_type'],
                'is_interpolated': True,
                'created_at': interpolated_sim['created_at']
            }
        )
    
    def load_interpolated_solutions(self):
        """Load all saved interpolated solutions"""
        npz_files = glob.glob(os.path.join(INTERPOLATED_SOLUTIONS_DIR, "*.npz"))
        
        self.interpolated_simulations = {}
        self.interpolated_summaries = []
        
        for npz_file in npz_files:
            try:
                data = np.load(npz_file, allow_pickle=True)
                points = data['points']
                values = data['values']
                metadata = data['metadata'].item()
                
                sim_data = {
                    'name': metadata['name'],
                    'original_simulation': metadata['original_simulation'],
                    'original_field': metadata['original_field'],
                    'original_timestep': metadata['original_timestep'],
                    'interpolation_method': metadata['interpolation_method'],
                    'energy_mJ': metadata['energy_mJ'],
                    'duration_ns': metadata['duration_ns'],
                    'points': points,
                    'values': values,
                    'field_type': metadata['field_type'],
                    'is_interpolated': True,
                    'created_at': metadata.get('created_at', 'unknown')
                }
                
                self.interpolated_simulations[metadata['name']] = sim_data
                
                # Create summary
                summary = self.create_interpolated_summary(sim_data)
                self.interpolated_summaries.append(summary)
                
            except Exception as e:
                st.warning(f"Error loading interpolated solution {npz_file}: {e}")
        
        return len(self.interpolated_simulations)
    
    def get_combined_simulations(self):
        """Get combined dictionary of original and interpolated simulations"""
        combined = {}
        
        # Add original simulations
        if 'simulations' in st.session_state:
            combined.update(st.session_state.simulations)
        
        # Add interpolated simulations
        combined.update(self.interpolated_simulations)
        
        return combined
    
    def get_combined_summaries(self):
        """Get combined list of original and interpolated summaries"""
        combined = []
        
        # Add original summaries
        if 'summaries' in st.session_state:
            combined.extend(st.session_state.summaries)
        
        # Add interpolated summaries
        combined.extend(self.interpolated_summaries)
        
        return combined

# =============================================
# ENHANCED 3D VISUALIZER WITH INTERPOLATED SUPPORT
# =============================================
class Enhanced3DVisualizer:
    """Enhanced 3D visualization with multiple rendering modes"""
    
    def __init__(self):
        self.interpolator = Advanced3DInterpolator()
        self.visualization_modes = {
            'Point Cloud': 'points',
            'Wireframe Mesh': 'wireframe',
            'Surface Mesh': 'surface',
            'Isosurfaces': 'isosurface',
            'Volume Rendering': 'volume',
            'Sliced Volume': 'slice'
        }
    
    def create_3d_visualization(self, points, values, triangles=None, 
                               mode='surface', colormap='Viridis',
                               grid_resolution=40, n_isosurfaces=4,
                               opacity=0.8, show_edges=False):
        """Create comprehensive 3D visualization based on selected mode"""
        
        # Normalize values for consistent coloring
        if len(values) > 0:
            valid_values = values[~np.isnan(values)]
            if len(valid_values) > 0:
                vmin, vmax = np.min(valid_values), np.max(valid_values)
                if vmax - vmin > 1e-10:
                    normalized_values = (values - vmin) / (vmax - vmin)
                else:
                    normalized_values = np.zeros_like(values)
            else:
                normalized_values = np.zeros_like(values)
        else:
            normalized_values = np.zeros_like(values)
        
        fig = go.Figure()
        
        if mode == 'points':
            # Point cloud visualization
            fig.add_trace(go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    color=values,
                    colorscale=colormap,
                    opacity=opacity,
                    colorbar=dict(title="Value", thickness=20),
                    showscale=True
                ),
                name='Point Cloud',
                hovertemplate='<b>Value:</b> %{marker.color:.3f}<br>' +
                            '<b>X:</b> %{x:.3f}<br>' +
                            '<b>Y:</b> %{y:.3f}<br>' +
                            '<b>Z:</b> %{z:.3f}<extra></extra>'
            ))
        
        elif mode == 'wireframe' and triangles is not None:
            # Wireframe mesh
            if len(triangles) > 0:
                # Extract triangle vertices
                i, j, k = triangles[:, 0], triangles[:, 1], triangles[:, 2]
                
                # Create mesh
                fig.add_trace(go.Mesh3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    i=i, j=j, k=k,
                    intensity=values,
                    colorscale=colormap,
                    intensitymode='vertex',
                    opacity=opacity * 0.5,
                    showscale=True,
                    colorbar=dict(title="Value", thickness=20),
                    hoverinfo='none'
                ))
                
                # Add wireframe edges
                edges_x, edges_y, edges_z = [], [], []
                for tri in triangles:
                    for i in range(3):
                        edges_x.extend([points[tri[i], 0], points[tri[(i+1)%3], 0], None])
                        edges_y.extend([points[tri[i], 1], points[tri[(i+1)%3], 1], None])
                        edges_z.extend([points[tri[i], 2], points[tri[(i+1)%3], 2], None])
                
                fig.add_trace(go.Scatter3d(
                    x=edges_x,
                    y=edges_y,
                    z=edges_z,
                    mode='lines',
                    line=dict(color='black', width=1),
                    hoverinfo='none',
                    showlegend=False
                ))
        
        elif mode == 'surface' and triangles is not None:
            # Smooth surface visualization
            if len(triangles) > 0:
                # Create smoothed surface
                smoothed_points, smoothed_values, smoothed_triangles = \
                    self.interpolator.create_smooth_surface(points, values, triangles)
                
                if smoothed_triangles is not None:
                    i, j, k = smoothed_triangles[:, 0], smoothed_triangles[:, 1], smoothed_triangles[:, 2]
                    
                    fig.add_trace(go.Mesh3d(
                        x=smoothed_points[:, 0],
                        y=smoothed_points[:, 1],
                        z=smoothed_points[:, 2],
                        i=i, j=j, k=k,
                        intensity=smoothed_values,
                        colorscale=colormap,
                        intensitymode='vertex',
                        opacity=opacity,
                        showscale=True,
                        colorbar=dict(title="Value", thickness=20),
                        flatshading=False,
                        lighting=dict(
                            ambient=0.8,
                            diffuse=0.8,
                            specular=0.5,
                            roughness=0.5
                        ),
                        lightposition=dict(x=100, y=200, z=300),
                        hoverinfo='none'
                    ))
        
        elif mode == 'isosurface':
            # Isosurface visualization with interpolation
            X_grid, Y_grid, Z_grid, grid_points, _ = \
                self.interpolator.create_regular_grid(points, values, grid_resolution)
            
            if grid_points is not None:
                grid_values = self.interpolator.interpolate_to_grid(
                    points, values, grid_points, method='rbf'
                )
                
                if grid_values is not None:
                    # Reshape for isosurface
                    values_3d = grid_values.reshape(X_grid.shape)
                    
                    # Get percentiles for isosurface levels
                    percentiles = np.linspace(20, 80, n_isosurfaces)
                    levels = np.percentile(grid_values[~np.isnan(grid_values)], percentiles)
                    
                    # Create isosurfaces
                    for i, level in enumerate(levels):
                        # Create isosurface
                        fig.add_trace(go.Isosurface(
                            x=X_grid.flatten(),
                            y=Y_grid.flatten(),
                            z=Z_grid.flatten(),
                            value=grid_values,
                            isomin=level - 0.01,
                            isomax=level + 0.01,
                            colorscale=colormap,
                            surface_count=1,
                            opacity=opacity * (0.8 - i * 0.15),
                            showscale=i==0,
                            colorbar=dict(title="Value", thickness=20),
                            caps=dict(x_show=False, y_show=False, z_show=False),
                            name=f'Isosurface {i+1}: {level:.2f}'
                        ))
        
        elif mode == 'volume':
            # Volume rendering
            X_grid, Y_grid, Z_grid, grid_points, _ = \
                self.interpolator.create_regular_grid(points, values, grid_resolution)
            
            if grid_points is not None:
                grid_values = self.interpolator.interpolate_to_grid(
                    points, values, grid_points, method='linear'
                )
                
                if grid_values is not None:
                    fig.add_trace(go.Volume(
                        x=X_grid.flatten(),
                        y=Y_grid.flatten(),
                        z=Z_grid.flatten(),
                        value=grid_values,
                        isomin=np.nanmin(grid_values),
                        isomax=np.nanmax(grid_values),
                        opacity=opacity * 0.3,  # Lower opacity for volume
                        opacityscale='uniform',
                        surface_count=20,
                        colorscale=colormap,
                        showscale=True,
                        colorbar=dict(title="Value", thickness=20),
                        caps=dict(x_show=False, y_show=False, z_show=False),
                        name='Volume'
                    ))
        
        elif mode == 'slice':
            # Sliced volume visualization
            X_grid, Y_grid, Z_grid, grid_points, _ = \
                self.interpolator.create_regular_grid(points, values, grid_resolution)
            
            if grid_points is not None:
                grid_values = self.interpolator.interpolate_to_grid(
                    points, values, grid_points, method='linear'
                )
                
                if grid_values is not None:
                    values_3d = grid_values.reshape(X_grid.shape)
                    
                    # Create slices at different positions
                    slice_positions = [0.25, 0.5, 0.75]
                    
                    for i, pos in enumerate(slice_positions):
                        slice_idx = int(pos * (grid_resolution - 1))
                        
                        # X-slice
                        fig.add_trace(go.Surface(
                            x=np.ones_like(Y_grid[slice_idx, :, :]) * X_grid[slice_idx, 0, 0],
                            y=Y_grid[slice_idx, :, :],
                            z=Z_grid[slice_idx, :, :],
                            surfacecolor=values_3d[slice_idx, :, :],
                            colorscale=colormap,
                            showscale=i==0,
                            colorbar=dict(title="Value", thickness=20),
                            opacity=opacity * 0.7,
                            name=f'X-Slice at {X_grid[slice_idx, 0, 0]:.2f}'
                        ))
        
        # Add original points for reference in all modes
        if mode not in ['points', 'wireframe']:
            fig.add_trace(go.Scatter3d(
                x=points[::10, 0],  # Sample every 10th point for clarity
                y=points[::10, 1],
                z=points[::10, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color='rgba(0, 0, 0, 0.3)',
                    symbol='circle'
                ),
                name='Sample Points',
                hoverinfo='none',
                showlegend=False
            ))
        
        # Update layout
        fig.update_layout(
            scene=dict(
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    up=dict(x=0, y=0, z=1)
                ),
                xaxis=dict(
                    title="X",
                    gridcolor="lightgray",
                    showbackground=True,
                    backgroundcolor="white"
                ),
                yaxis=dict(
                    title="Y",
                    gridcolor="lightgray",
                    showbackground=True,
                    backgroundcolor="white"
                ),
                zaxis=dict(
                    title="Z",
                    gridcolor="lightgray",
                    showbackground=True,
                    backgroundcolor="white"
                )
            ),
            height=700,
            margin=dict(l=0, r=0, t=40, b=0),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    def create_pyvista_visualization(self, points, values, triangles=None, mode='surface', 
                                   colormap='viridis', grid_resolution=40, n_isosurfaces=4, 
                                   opacity=0.8, show_edges=False):
        """Create PyVista plotter for true 3D rendering"""
        
        try:
            # Create plotter with higher quality settings
            plotter = pv.Plotter(window_size=[800, 600], off_screen=True)
            
            if mode == 'points':
                # Point cloud
                cloud = pv.PolyData(points)
                cloud['values'] = values
                plotter.add_points(cloud, scalars='values', cmap=colormap, point_size=4, 
                                 render_points_as_spheres=True, opacity=opacity)
                
            elif mode in ['wireframe', 'surface'] and triangles is not None:
                # Create mesh and smooth it
                if len(triangles) > 0:
                    # Ensure triangles are in correct format for PyVista
                    try:
                        faces = np.hstack([[3, *tri] for tri in triangles])
                        mesh = pv.PolyData(points, faces)
                    except:
                        # Alternative format
                        mesh = pv.PolyData(points)
                        if len(triangles) > 0:
                            mesh.faces = triangles
                    
                    mesh['values'] = values
                    
                    if mode == 'wireframe':
                        plotter.add_mesh(mesh, style='wireframe', line_width=2, color='black', 
                                       scalars='values', cmap=colormap)
                    else:  # surface
                        # Smooth the surface
                        try:
                            smoothed = mesh.smooth(n_iter=100, relaxation_factor=0.1)
                            plotter.add_mesh(smoothed, scalars='values', cmap=colormap, 
                                           opacity=opacity, show_edges=show_edges, 
                                           edge_color='black')
                        except:
                            plotter.add_mesh(mesh, scalars='values', cmap=colormap, 
                                           opacity=opacity, show_edges=show_edges)
            
            elif mode == 'isosurface':
                # Interpolate to grid first
                X_grid, Y_grid, Z_grid, grid_points, _ = self.interpolator.create_regular_grid(
                    points, values, grid_resolution)
                if grid_points is not None:
                    grid_values = self.interpolator.interpolate_to_grid(points, values, grid_points, 
                                                                      method='rbf')
                    if grid_values is not None:
                        # Create structured grid
                        grid = pv.UniformGrid()
                        grid.dimensions = X_grid.shape[::-1]
                        grid.origin = (X_grid.min(), Y_grid.min(), Z_grid.min())
                        grid.spacing = (
                            (X_grid.max() - X_grid.min()) / (X_grid.shape[0] - 1),
                            (Y_grid.max() - Y_grid.min()) / (Y_grid.shape[1] - 1),
                            (Z_grid.max() - Z_grid.min()) / (X_grid.shape[2] - 1)
                        )
                        grid['field'] = grid_values.reshape(X_grid.shape[::-1])
                        
                        # Multiple isosurfaces
                        valid_values = grid_values[~np.isnan(grid_values)]
                        if len(valid_values) > 0:
                            for i in range(n_isosurfaces):
                                level = np.percentile(valid_values, 
                                                    20 + 60*i/(max(1, n_isosurfaces-1)))
                                contours = grid.contour([level])
                                plotter.add_mesh(contours, cmap=colormap, 
                                               opacity=0.8 - i*0.15, show_edges=True)
            
            elif mode == 'volume':
                # Volume rendering
                X_grid, Y_grid, Z_grid, grid_points, _ = self.interpolator.create_regular_grid(
                    points, values, grid_resolution)
                if grid_points is not None:
                    grid_values = self.interpolator.interpolate_to_grid(points, values, grid_points, 
                                                                      method='linear')
                    if grid_values is not None:
                        grid = pv.UniformGrid()
                        grid.dimensions = X_grid.shape[::-1]
                        grid.origin = (X_grid.min(), Y_grid.min(), Z_grid.min())
                        grid.spacing = (
                            (X_grid.max() - X_grid.min()) / (X_grid.shape[0] - 1),
                            (Y_grid.max() - Y_grid.min()) / (X_grid.shape[1] - 1),
                            (Z_grid.max() - Z_grid.min()) / (X_grid.shape[2] - 1)
                        )
                        grid['field'] = grid_values.reshape(X_grid.shape[::-1])
                        
                        # True volume rendering with transfer function
                        valid_values = grid_values[~np.isnan(grid_values)]
                        if len(valid_values) > 0:
                            plotter.add_volume(grid, scalars='field', cmap=colormap, 
                                             opacity='linear', opacity_unit=0.5, 
                                             shade=True, clim=[valid_values.min(), valid_values.max()])
            
            elif mode == 'slice':
                # Multiple slicing planes
                X_grid, Y_grid, Z_grid, grid_points, _ = self.interpolator.create_regular_grid(
                    points, values, grid_resolution)
                if grid_points is not None:
                    grid_values = self.interpolator.interpolate_to_grid(points, values, grid_points)
                    if grid_values is not None:
                        grid = pv.UniformGrid()
                        grid.dimensions = X_grid.shape[::-1]
                        grid.origin = (X_grid.min(), Y_grid.min(), Z_grid.min())
                        grid.spacing = (
                            (X_grid.max() - X_grid.min()) / (X_grid.shape[0] - 1),
                            (Y_grid.max() - Y_grid.min()) / (X_grid.shape[1] - 1),
                            (Z_grid.max() - Z_grid.min()) / (X_grid.shape[2] - 1)
                        )
                        grid['field'] = grid_values.reshape(X_grid.shape[::-1])
                        
                        # XY, XZ, YZ slices
                        plotter.add_mesh(grid.slice(normal='x'), scalars='field', cmap=colormap, 
                                       opacity=0.7, name='X-slice')
                        plotter.add_mesh(grid.slice(normal='y'), scalars='field', cmap=colormap, 
                                       opacity=0.7, name='Y-slice')
                        plotter.add_mesh(grid.slice(normal='z'), scalars='field', cmap=colormap, 
                                       opacity=0.7, name='Z-slice')
            
            # Add scalar bar and improve lighting
            plotter.add_scalar_bar(title='Field Value', title_font_size=16, n_labels=4)
            plotter.add_axes()
            
            # Professional camera setup
            plotter.view_isometric()
            plotter.background_color = 'white'
            
            return plotter
            
        except Exception as e:
            st.error(f"PyVista visualization error: {str(e)}")
            # Fallback to Plotly visualization
            return None

# =============================================
# ADVANCED 3D INTERPOLATION AND VISUALIZATION
# =============================================
class Advanced3DInterpolator:
    """Advanced 3D field interpolation for smooth visualization"""
    
    def __init__(self):
        self.interpolation_methods = {
            'RBF (Smooth)': 'rbf',
            'Linear': 'linear',
            'Cubic': 'cubic',
            'Nearest': 'nearest'
        }
        
    def create_regular_grid(self, points, values, grid_resolution=50, padding=0.1):
        """Create a regular 3D grid covering the point cloud"""
        if len(points) == 0:
            return None, None, None, None, None
        
        # Get bounding box
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        
        # Add padding
        ranges = max_coords - min_coords
        min_coords -= padding * ranges
        max_coords += padding * ranges
        
        # Create grid
        xi = np.linspace(min_coords[0], max_coords[0], grid_resolution)
        yi = np.linspace(min_coords[1], max_coords[1], grid_resolution)
        zi = np.linspace(min_coords[2], max_coords[2], grid_resolution)
        
        X_grid, Y_grid, Z_grid = np.meshgrid(xi, yi, zi, indexing='ij')
        grid_points = np.vstack([X_grid.ravel(), Y_grid.ravel(), Z_grid.ravel()]).T
        
        return X_grid, Y_grid, Z_grid, grid_points, (min_coords, max_coords)
    
    def interpolate_to_grid(self, points, values, grid_points, method='rbf', 
                           function='linear', epsilon=None):
        """Interpolate scattered data to regular grid"""
        if len(points) < 4:  # Need at least 4 points for interpolation
            return None
        
        # Remove NaN values
        valid_mask = ~np.isnan(values)
        if np.sum(valid_mask) < 4:
            return None
            
        points_valid = points[valid_mask]
        values_valid = values[valid_mask]
        
        if method == 'rbf':
            try:
                # Use RBF interpolation for smoother results
                if epsilon is None:
                    # Estimate epsilon based on point spacing
                    from scipy.spatial import cKDTree
                    tree = cKDTree(points_valid)
                    distances, _ = tree.query(points_valid, k=2)
                    epsilon = np.mean(distances[:, 1]) * 2
                
                rbf = RBFInterpolator(points_valid, values_valid, 
                                      kernel='multiquadric', epsilon=epsilon)
                grid_values = rbf(grid_points)
            except Exception as e:
                st.warning(f"RBF interpolation failed: {e}, falling back to linear")
                method = 'linear'
        
        if method in ['linear', 'cubic', 'nearest']:
            try:
                grid_values = griddata(points_valid, values_valid, grid_points, 
                                      method=method, fill_value=0)
            except Exception as e:
                st.warning(f"Griddata interpolation failed: {e}")
                return None
        
        return grid_values
    
    def extract_isosurfaces(self, X_grid, Y_grid, Z_grid, grid_values, 
                           n_isosurfaces=5, use_percentiles=True):
        """Extract isosurfaces from 3D scalar field"""
        if grid_values is None:
            return []
        
        # Reshape grid values
        values_3d = grid_values.reshape(X_grid.shape)
        
        # Determine isosurface levels
        if use_percentiles:
            percentiles = np.linspace(10, 90, n_isosurfaces)
            levels = np.percentile(grid_values[~np.isnan(grid_values)], percentiles)
        else:
            min_val = np.nanmin(grid_values)
            max_val = np.nanmax(grid_values)
            levels = np.linspace(min_val, max_val, n_isosurfaces + 2)[1:-1]
        
        isosurfaces = []
        for level in levels:
            try:
                # Extract isosurface using marching cubes
                vertices, faces, _, _ = measure.marching_cubes(
                    values_3d, level, spacing=(1, 1, 1), allow_degenerate=False
                )
                
                # Transform vertices back to original coordinates
                xi = np.linspace(0, X_grid.shape[0]-1, X_grid.shape[0])
                yi = np.linspace(0, X_grid.shape[1]-1, X_grid.shape[1])
                zi = np.linspace(0, X_grid.shape[2]-1, X_grid.shape[2])
                
                vertices[:, 0] = np.interp(vertices[:, 0], np.arange(len(xi)), X_grid[:, 0, 0])
                vertices[:, 1] = np.interp(vertices[:, 1], np.arange(len(yi)), Y_grid[0, :, 0])
                vertices[:, 2] = np.interp(vertices[:, 2], np.arange(len(zi)), Z_grid[0, 0, :])
                
                isosurfaces.append({
                    'vertices': vertices,
                    'faces': faces,
                    'level': level
                })
            except Exception as e:
                continue
        
        return isosurfaces
    
    def create_smooth_surface(self, points, values, triangles=None, method='delaunay'):
        """Create smooth surface from point cloud"""
        if triangles is None or len(points) < 4:
            return None, None, None
        
        try:
            # Create mesh
            mesh = pv.PolyData(points, triangles.reshape(-1, 4)[:, 1:4])
            mesh['values'] = values
            
            # Smooth the mesh
            smoothed = mesh.smooth(n_iter=100, relaxation_factor=0.1)
            
            # Extract smoothed vertices and values
            smoothed_points = smoothed.points
            smoothed_values = smoothed['values']
            
            # Get triangles
            faces = smoothed.faces.reshape(-1, 4)[:, 1:4]
            
            return smoothed_points, smoothed_values, faces
            
        except Exception as e:
            st.warning(f"Surface smoothing failed: {e}")
            return points, values, triangles

# =============================================
# UNIFIED DATA LOADER WITH INTERPOLATED SUPPORT
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
    
    @st.cache_data
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
                
                # Create simulation entry
                sim_data = {
                    'name': name,
                    'energy_mJ': energy,
                    'duration_ns': duration,
                    'n_timesteps': len(vtu_files),
                    'vtu_files': vtu_files,
                    'field_info': {},
                    'has_mesh': False,
                    'is_interpolated': False  # Mark as original
                }
                
                if load_full_mesh:
                    # Full mesh loading
                    points = mesh0.points.astype(np.float32)
                    n_pts = len(points)
                    
                    # Find triangles
                    triangles = None
                    for cell_block in mesh0.cells:
                        if cell_block.type == "triangle":
                            triangles = cell_block.data.astype(np.int32)
                            break
                    
                    # Initialize fields
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
                
                # Create summary statistics
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
            'field_stats': {},
            'is_interpolated': False  # Mark as original
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
                            # Store full percentiles for detailed analysis
                            percentiles = np.percentile(clean_data, [10, 25, 50, 75, 90])
                            summary['field_stats'][field_name]['percentiles'].append(percentiles)
                        else:
                            for key in ['min', 'max', 'mean', 'std', 'q25', 'q50', 'q75']:
                                summary['field_stats'][field_name][key].append(0.0)
                            summary['field_stats'][field_name]['percentiles'].append(np.zeros(5))
                    else:
                        # Vector field - compute magnitude statistics
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
# ENHANCED ATTENTION MECHANISM WITH PHYSICS-AWARE EMBEDDINGS
# =============================================
class EnhancedPhysicsInformedAttentionExtrapolator:
    """Advanced extrapolator with physics-aware embeddings and multi-head attention"""
    
    def __init__(self, sigma_param=0.3, spatial_weight=0.5, n_heads=4, temperature=1.0):
        self.sigma_param = sigma_param
        self.spatial_weight = spatial_weight
        self.n_heads = n_heads
        self.temperature = temperature
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
        
        # Prepare embeddings and values
        all_embeddings = []
        all_values = []
        metadata = []
        
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
                    # Use mean, max, std as representative features
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
            
            st.info(f"✅ Prepared {len(all_embeddings)} embeddings with {all_embeddings.shape[1]} features")
    
    def _compute_physics_embedding(self, energy, duration, time):
        """Compute comprehensive physics-aware embedding"""
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
            np.log1p(time)
        ], dtype=np.float32)
    
    def _compute_spatial_similarity(self, query_meta, source_metas):
        """Compute spatial similarity based on parameter proximity"""
        similarities = []
        for meta in source_metas:
            # Normalized differences
            e_diff = abs(query_meta['energy'] - meta['energy']) / 50.0
            d_diff = abs(query_meta['duration'] - meta['duration']) / 20.0
            t_diff = abs(query_meta['time'] - meta['time']) / 50.0
            
            # Combined similarity (inverse distance)
            total_diff = np.sqrt(e_diff**2 + d_diff**2 + t_diff**2)
            similarity = np.exp(-total_diff / self.sigma_param)
            similarities.append(similarity)
        
        return np.array(similarities)
    
    def _multi_head_attention(self, query_embedding, query_meta):
        """Multi-head attention mechanism with spatial regulation"""
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
        
        # Weighted prediction (using original values, not scaled)
        if len(self.source_values) > 0:
            prediction = np.sum(attention_weights[:, np.newaxis] * self.source_values, axis=0)
        else:
            prediction = np.zeros(1)
        
        return prediction, attention_weights
    
    def predict_field_statistics(self, energy_query, duration_query, time_query):
        """Predict field statistics for given parameters"""
        if not self.fitted:
            return None
        
        # Compute query embedding and metadata
        query_embedding = self._compute_physics_embedding(energy_query, duration_query, time_query)
        query_meta = {
            'energy': energy_query,
            'duration': duration_query,
            'time': time_query
        }
        
        # Apply attention mechanism
        prediction, attention_weights = self._multi_head_attention(query_embedding, query_meta)
        
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
            # Get field order from first summary
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
    
    def predict_time_series(self, energy_query, duration_query, time_points):
        """Predict over a series of time points"""
        results = {
            'time_points': time_points,
            'field_predictions': {},
            'attention_maps': [],
            'confidence_scores': []
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
                results['confidence_scores'].append(pred['confidence'])
            else:
                # Fill with NaN if prediction failed
                for field in results['field_predictions']:
                    results['field_predictions'][field]['mean'].append(np.nan)
                    results['field_predictions'][field]['max'].append(np.nan)
                    results['field_predictions'][field]['std'].append(np.nan)
                results['attention_maps'].append(np.array([]))
                results['confidence_scores'].append(0.0)
        
        return results

# =============================================
# ADVANCED VISUALIZATION COMPONENTS WITH EXTENDED COLORMAPS
# =============================================
class EnhancedVisualizer:
    """Comprehensive visualization with extended colormaps and advanced features"""
    
    # Extended colormaps for better visualization
    COLORSCALES = {
        'temperature': ['#2c0078', '#4402a7', '#5e04d1', '#7b0ef6', '#9a38ff', '#b966ff', '#d691ff', '#f2bcff'],
        'stress': ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600'],
        'displacement': ['#004c6d', '#346888', '#5886a5', '#7aa6c2', '#9dc6e0', '#c1e7ff'],
        'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    }
    
    # Extended colormap options for Plotly
    EXTENDED_COLORMAPS = [
        'Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis',
        'Rainbow', 'Jet', 'Hot', 'Cool', 'Portland',
        'Bluered', 'Electric', 'Thermal', 'Balance',
        'Brwnyl', 'Darkmint', 'Emrld', 'Mint', 'Oranges',
        'Purp', 'Purples', 'Sunset', 'Sunsetdark', 'Teal',
        'Tealgrn', 'Twilight', 'Burg', 'Burgyl'
    ]
    
    @staticmethod
    def create_sunburst_chart(summaries, selected_field='temperature', highlight_sim=None):
        """Create enhanced sunburst chart with highlighted target simulation"""
        labels = []
        parents = []
        values = []
        colors = []
        
        # Separate original and interpolated simulations
        original_summaries = [s for s in summaries if not s.get('is_interpolated', False)]
        interpolated_summaries = [s for s in summaries if s.get('is_interpolated', False)]
        
        # Root node
        labels.append("All Simulations")
        parents.append("")
        values.append(len(summaries))
        colors.append("#1f77b4")
        
        # Original simulations branch
        labels.append("Original Simulations")
        parents.append("All Simulations")
        values.append(len(original_summaries))
        colors.append("#2ca02c")
        
        # Interpolated simulations branch
        labels.append("Interpolated Simulations")
        parents.append("All Simulations")
        values.append(len(interpolated_summaries))
        colors.append("#9467bd")
        
        # Add original simulations
        for summary in original_summaries:
            energy_key = f"{summary['energy']:.1f} mJ"
            duration_key = f"τ: {summary['duration']:.1f} ns"
            sim_label = f"{summary['name']}"
            
            # Energy node
            energy_node = f"Energy: {energy_key} (Original)"
            if energy_node not in labels:
                labels.append(energy_node)
                parents.append("Original Simulations")
                values.append(1)
                colors.append("#ff7f0e")
            
            # Simulation node
            labels.append(sim_label)
            parents.append(energy_node)
            values.append(1)
            
            # Highlight target simulation
            if highlight_sim and summary['name'] == highlight_sim:
                colors.append("#d62728")  # Red for target
            else:
                colors.append("#17becf")  # Teal for other originals
        
        # Add interpolated simulations
        for summary in interpolated_summaries:
            sim_label = f"{summary['name']}"
            
            # Simulation node
            labels.append(sim_label)
            parents.append("Interpolated Simulations")
            values.append(1)
            
            # Highlight target simulation
            if highlight_sim and summary['name'] == highlight_sim:
                colors.append("#d62728")  # Red for target
            else:
                colors.append("#e377c2")  # Pink for interpolated
        
        # Ensure all values are positive
        values = [max(v, 1e-6) for v in values]
        
        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            marker=dict(
                colors=colors,
                colorscale='Viridis',
                line=dict(width=2, color='white')
            ),
            hovertemplate='<b>%{label}</b><br>Value: %{value:.2f}<br>Parent: %{parent}<extra></extra>',
            textinfo="label+value",
            textfont=dict(size=12)
        ))
        
        title = f"Simulation Hierarchy - {selected_field}"
        if highlight_sim:
            title += f" (Target: {highlight_sim})"
        
        fig.update_layout(
            title=title,
            height=700,
            margin=dict(t=50, b=20, l=20, r=20)
        )
        
        return fig
    
    @staticmethod
    def create_radar_chart(summaries, simulation_names, target_sim=None):
        """Create enhanced radar chart with highlighted target simulation"""
        # Determine available fields
        all_fields = set()
        for summary in summaries:
            all_fields.update(summary['field_stats'].keys())
        
        if not all_fields:
            return go.Figure()
        
        # Select top 6 fields for clarity
        selected_fields = list(all_fields)[:6]
        
        fig = go.Figure()
        
        for sim_name in simulation_names:
            # Find summary
            summary = next((s for s in summaries if s['name'] == sim_name), None)
            if not summary:
                continue
            
            r_values = []
            theta_values = []
            
            for field in selected_fields:
                if field in summary['field_stats']:
                    stats = summary['field_stats'][field]
                    # Use mean value across timesteps
                    if stats['mean']:
                        avg_value = np.mean(stats['mean'])
                        r_values.append(avg_value if avg_value > 0 else 1e-6)
                        theta_values.append(f"{field[:15]}...")
                    else:
                        r_values.append(1e-6)
                        theta_values.append(f"{field[:15]}...")
                else:
                    r_values.append(1e-6)
                    theta_values.append(f"{field[:15]}...")
            
            # Highlight target simulation
            line_width = 4 if target_sim and sim_name == target_sim else 2
            line_dash = 'solid' if target_sim and sim_name == target_sim else 'dash'
            
            # Different colors for interpolated vs original
            if summary.get('is_interpolated', False):
                color = 'rgba(255,0,255,0.3)'  # Magenta for interpolated
            else:
                color = f'rgba(255,0,0,0.3)' if target_sim and sim_name == target_sim else None
            
            fig.add_trace(go.Scatterpolar(
                r=r_values,
                theta=theta_values,
                fill='toself',
                name=f"{sim_name} {'(Interpolated)' if summary.get('is_interpolated', False) else ''}",
                line=dict(width=line_width, dash=line_dash),
                fillcolor=color,
                opacity=0.8
            ))
        
        if fig.data:
            # Find maximum value for scaling
            max_values = []
            for trace in fig.data:
                max_values.append(max(trace.r))
            
            if max_values:
                max_r = max(max_values)
                
                # Add circular grid lines
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, max_r * 1.2],
                            tickfont=dict(size=10),
                            gridcolor='lightgray',
                            linecolor='gray'
                        ),
                        angularaxis=dict(
                            tickfont=dict(size=11),
                            rotation=90,
                            direction="clockwise"
                        ),
                        bgcolor='white',
                        gridshape='circular'
                    ),
                    showlegend=True,
                    title="Radar Chart: Simulation Comparison",
                    height=600,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=1.05
                    )
                )
        
        return fig
    
    @staticmethod
    def create_attention_heatmap_3d(attention_weights, source_metadata):
        """Create 3D heatmap of attention weights"""
        if len(attention_weights) == 0:
            return go.Figure()
        
        # Extract metadata for 3D coordinates
        energies = []
        durations = []
        times = []
        
        for meta in source_metadata:
            energies.append(meta['energy'])
            durations.append(meta['duration'])
            times.append(meta['time'])
        
        energies = np.array(energies)
        durations = np.array(durations)
        times = np.array(times)
        
        # Create 3D scatter plot with attention weights as color
        fig = go.Figure(data=go.Scatter3d(
            x=energies,
            y=durations,
            z=times,
            mode='markers',
            marker=dict(
                size=10,
                color=attention_weights,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(
                    title="Attention Weight",
                    thickness=20,
                    len=0.5
                ),
                showscale=True
            ),
            text=[f"E: {e:.1f} mJ<br>τ: {d:.1f} ns<br>t: {t:.1f} ns<br>Weight: {w:.4f}" 
                  for e, d, t, w in zip(energies, durations, times, attention_weights)],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title="3D Attention Weight Distribution",
            scene=dict(
                xaxis_title="Energy (mJ)",
                yaxis_title="Duration (ns)",
                zaxis_title="Time (ns)",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            height=600,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig
    
    @staticmethod
    def create_attention_network(attention_weights, source_metadata, top_k=10):
        """Create network graph of attention relationships"""
        if len(attention_weights) == 0 or len(source_metadata) == 0:
            return go.Figure()
        
        # Aggregate attention by simulation
        sim_attention = {}
        for idx, (weight, meta) in enumerate(zip(attention_weights, source_metadata)):
            sim_key = meta['name']
            if sim_key not in sim_attention:
                sim_attention[sim_key] = []
            sim_attention[sim_key].append(weight)
        
        # Average attention per simulation
        avg_attention = {k: np.mean(v) for k, v in sim_attention.items()}
        
        # Get top-k simulations
        sorted_sims = sorted(avg_attention.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        if not sorted_sims:
            return go.Figure()
        
        # Create network graph
        G = nx.Graph()
        G.add_node("QUERY", size=50, color='red', label="Query")
        
        # Add simulation nodes
        for i, (sim_name, weight) in enumerate(sorted_sims):
            node_id = f"SIM_{i}"
            # Find metadata for this simulation
            sim_meta = next((m for m in source_metadata if m['name'] == sim_name), None)
            
            G.add_node(node_id,
                      size=30 * weight / max(avg_attention.values()),
                      color='blue',
                      label=sim_name,
                      energy=sim_meta['energy'] if sim_meta else 0,
                      duration=sim_meta['duration'] if sim_meta else 0,
                      weight=weight)
            
            # Add edge from query to simulation
            G.add_edge("QUERY", node_id, weight=weight, width=3 * weight)
        
        # Spring layout for node positions
        pos = nx.spring_layout(G, seed=42, k=2)
        
        # Create edge traces
        edge_x = []
        edge_y = []
        edge_text = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            weight = G[edge[0]][edge[1]]['weight']
            edge_text.append(f"Attention: {weight:.3f}")
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='gray'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node traces
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            if node == "QUERY":
                node_text.append("QUERY")
                node_size.append(30)
                node_color.append('red')
            else:
                sim_data = G.nodes[node]
                energy = sim_data.get('energy', 0)
                duration = sim_data.get('duration', 0)
                weight = sim_data.get('weight', 0)
                
                node_text.append(
                    f"Simulation: {sim_data['label']}<br>"
                    f"Energy: {energy:.1f} mJ<br>"
                    f"Duration: {duration:.1f} ns<br>"
                    f"Attention: {weight:.3f}"
                )
                node_size.append(sim_data['size'] + 10)
                node_color.append('blue')
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=[n if n == "QUERY" else f"Sim{i}" for i, n in enumerate(G.nodes()) if n != "QUERY"],
            textposition="middle center",
            hoverinfo='text',
            hovertext=node_text,
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='white')
            )
        )
        
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title=f"Attention Network (Top {len(sorted_sims)} Simulations)",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            height=500,
            plot_bgcolor='white',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
    @staticmethod
    def create_field_evolution_comparison(summaries, simulation_names, selected_field, target_sim=None):
        """Create enhanced field evolution comparison plot"""
        fig = go.Figure()
        
        for sim_name in simulation_names:
            summary = next((s for s in summaries if s['name'] == sim_name), None)
            
            if summary and selected_field in summary['field_stats']:
                stats = summary['field_stats'][selected_field]
                
                # Highlight target simulation
                line_width = 4 if target_sim and sim_name == target_sim else 2
                line_dash = 'solid' if target_sim and sim_name == target_sim else 'dash'
                
                # Different colors for interpolated
                if summary.get('is_interpolated', False):
                    line_color = 'rgba(255, 0, 255, 0.8)'  # Magenta for interpolated
                    fill_color = 'rgba(255, 0, 255, 0.1)'
                else:
                    line_color = 'rgba(0, 0, 255, 0.8)'  # Blue for original
                    fill_color = 'rgba(0, 0, 255, 0.1)'
                
                # Plot mean
                fig.add_trace(go.Scatter(
                    x=summary['timesteps'],
                    y=stats['mean'],
                    mode='lines+markers',
                    name=f"{sim_name} {'(Interpolated)' if summary.get('is_interpolated', False) else ''}",
                    line=dict(width=line_width, dash=line_dash, color=line_color),
                    opacity=0.8
                ))
                
                # Add confidence band (mean ± std)
                if stats['std']:
                    y_upper = np.array(stats['mean']) + np.array(stats['std'])
                    y_lower = np.array(stats['mean']) - np.array(stats['std'])
                    
                    fig.add_trace(go.Scatter(
                        x=summary['timesteps'] + summary['timesteps'][::-1],
                        y=np.concatenate([y_upper, y_lower[::-1]]),
                        fill='toself',
                        fillcolor=fill_color,
                        line=dict(color='rgba(255,255,255,0)'),
                        showlegend=False,
                        name=f"{sim_name} ± std"
                    ))
        
        if fig.data:
            fig.update_layout(
                title=f"{selected_field} Evolution Comparison",
                xaxis_title="Timestep (ns)",
                yaxis_title=f"{selected_field} Value",
                hovermode="x unified",
                height=500,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.02
                )
            )
        
        return fig

# =============================================
# ENHANCED MAIN APPLICATION WITH INTERPOLATED SOLUTIONS SUPPORT
# =============================================
def main():
    st.set_page_config(
        page_title="Enhanced FEA Laser Simulation Platform",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="🔬"
    )
    
    # Custom CSS with enhanced styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(90deg, #1E88E5, #4A00E0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 800;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3498db;
        font-weight: 600;
    }
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    .warning-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    .success-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f8f9fa;
        border-radius: 8px 8px 0 0;
        color: #495057;
        font-weight: 600;
        padding: 0 24px;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e9ecef;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5;
        color: white;
    }
    .stButton > button {
        background: linear-gradient(90deg, #1E88E5, #4A00E0);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .interpolated-badge {
        background: linear-gradient(90deg, #FF6B6B, #FF8E53);
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-left: 8px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🔬 Enhanced FEA Laser Simulation Platform</h1>', 
                unsafe_allow_html=True)
    
    # Initialize session state
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = UnifiedFEADataLoader()
        st.session_state.extrapolator = EnhancedPhysicsInformedAttentionExtrapolator()
        st.session_state.visualizer = EnhancedVisualizer()
        st.session_state.visualizer_3d = Enhanced3DVisualizer()
        st.session_state.interpolated_manager = InterpolatedSolutionsManager()
        st.session_state.data_loaded = False
        st.session_state.current_mode = "Data Viewer"
        st.session_state.selected_colormap = "Viridis"
    
    # Load interpolated solutions
    if 'interpolated_loaded' not in st.session_state:
        num_interpolated = st.session_state.interpolated_manager.load_interpolated_solutions()
        st.session_state.interpolated_loaded = True
        if num_interpolated > 0:
            st.sidebar.info(f"📊 Loaded {num_interpolated} interpolated solutions")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Navigation")
        app_mode = st.radio(
            "Select Mode",
            ["Data Viewer", "3D Visualization", "Interpolation/Extrapolation", "Comparative Analysis", "Create Interpolated"],
            index=["Data Viewer", "3D Visualization", "Interpolation/Extrapolation", "Comparative Analysis", "Create Interpolated"].index(
                st.session_state.current_mode if 'current_mode' in st.session_state else "Data Viewer"
            ),
            key="nav_mode"
        )
        
        st.session_state.current_mode = app_mode
        
        st.markdown("---")
        st.markdown("### 📊 Data Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            load_full_data = st.checkbox("Load Full Mesh", value=True, 
                                        help="Load complete mesh data for 3D visualization")
        with col2:
            st.session_state.selected_colormap = st.selectbox(
                "Colormap",
                EnhancedVisualizer.EXTENDED_COLORMAPS,
                index=0
            )
        
        if st.button("🔄 Load All Simulations", type="primary", use_container_width=True):
            with st.spinner("Loading simulation data..."):
                simulations, summaries = st.session_state.data_loader.load_all_simulations(
                    load_full_mesh=load_full_data
                )
                st.session_state.simulations = simulations
                st.session_state.summaries = summaries
                
                if simulations and summaries:
                    st.session_state.extrapolator.load_summaries(summaries)
                    st.session_state.data_loaded = True
                    
                    # Store available fields
                    st.session_state.available_fields = set()
                    for summary in summaries:
                        st.session_state.available_fields.update(summary['field_stats'].keys())
        
        if st.session_state.data_loaded or len(st.session_state.interpolated_manager.interpolated_simulations) > 0:
            st.markdown("---")
            st.markdown("### 📈 Loaded Data")
            
            with st.expander("Data Overview", expanded=True):
                total_simulations = len(st.session_state.get('simulations', {})) + \
                                   len(st.session_state.interpolated_manager.interpolated_simulations)
                st.metric("Total Simulations", total_simulations)
                st.metric("Original Simulations", len(st.session_state.get('simulations', {})))
                st.metric("Interpolated Solutions", len(st.session_state.interpolated_manager.interpolated_simulations))
                
                if st.session_state.summaries:
                    energies = [s['energy'] for s in st.session_state.summaries]
                    durations = [s['duration'] for s in st.session_state.summaries]
                    st.metric("Energy Range", f"{min(energies):.1f} - {max(energies):.1f} mJ")
                    st.metric("Duration Range", f"{min(durations):.1f} - {max(durations):.1f} ns")
    
    # Main content based on selected mode
    if app_mode == "Data Viewer":
        render_data_viewer()
    elif app_mode == "3D Visualization":
        render_3d_visualization()
    elif app_mode == "Interpolation/Extrapolation":
        render_interpolation_extrapolation()
    elif app_mode == "Comparative Analysis":
        render_comparative_analysis()
    elif app_mode == "Create Interpolated":
        render_create_interpolated()

# =============================================
# RENDER FUNCTIONS WITH INTERPOLATED SUPPORT
# =============================================
def render_data_viewer():
    """Render the enhanced data visualization interface with interpolated support"""
    st.markdown('<h2 class="sub-header">📁 Data Viewer</h2>', unsafe_allow_html=True)
    
    # Get combined simulations and summaries
    combined_simulations = st.session_state.interpolated_manager.get_combined_simulations()
    combined_summaries = st.session_state.interpolated_manager.get_combined_summaries()
    
    if not combined_simulations:
        st.markdown("""
        <div class="warning-box">
        <h3>⚠️ No Data Loaded</h3>
        <p>Please load simulations first using the "Load All Simulations" button in the sidebar.</p>
        <p>Ensure your data follows this structure:</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📁 Expected Directory Structure"):
            st.code("""
fea_solutions/
├── q0p5mJ-delta4p2ns/        # Energy: 0.5 mJ, Duration: 4.2 ns
│   ├── a_t0001.vtu           # Timestep 1
│   ├── a_t0002.vtu           # Timestep 2
│   ├── a_t0003.vtu           # Timestep 3
│   └── ...
├── q1p0mJ-delta2p0ns/        # Energy: 1.0 mJ, Duration: 2.0 ns
│   ├── a_t0001.vtu
│   ├── a_t0002.vtu
│   └── ...
└── q2p0mJ-delta1p0ns/        # Energy: 2.0 mJ, Duration: 1.0 ns
    ├── a_t0001.vtu
    └── ...
            """)
        return
    
    # Simulation selection with badges for interpolated
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        # Create display names with badges
        display_names = []
        for sim_name in sorted(combined_simulations.keys()):
            sim = combined_simulations[sim_name]
            if sim.get('is_interpolated', False):
                display_name = f"{sim_name} 🌟"
            else:
                display_name = sim_name
            display_names.append(display_name)
        
        selected_display = st.selectbox(
            "Select Simulation",
            display_names,
            key="viewer_sim_select",
            help="🌟 indicates interpolated solutions"
        )
        
        # Get actual simulation name
        sim_name = selected_display.split(' 🌟')[0] if ' 🌟' in selected_display else selected_display
    
    sim = combined_simulations[sim_name]
    
    with col2:
        st.metric("Energy", f"{sim['energy_mJ']:.2f} mJ")
    with col3:
        st.metric("Duration", f"{sim['duration_ns']:.2f} ns")
    
    # Show interpolated badge
    if sim.get('is_interpolated', False):
        st.markdown('<div class="interpolated-badge" style="display: inline-block;">INTERPOLATED</div>', 
                   unsafe_allow_html=True)
        st.info(f"Interpolated from: {sim['original_simulation']} | Method: {sim['interpolation_method']}")
    
    if sim.get('is_interpolated', False):
        # Handle interpolated simulation
        if 'points' in sim and 'values' in sim:
            pts = sim['points']
            values = sim['values']
            
            # Visualization mode selection
            visualization_mode = st.selectbox(
                "Visualization Mode",
                ["Basic Mesh", "Point Cloud"],
                key="viewer_viz_mode_interp"
            )
            
            colormap = st.selectbox(
                "Colormap",
                EnhancedVisualizer.EXTENDED_COLORMAPS,
                index=EnhancedVisualizer.EXTENDED_COLORMAPS.index(st.session_state.selected_colormap),
                key="viewer_colormap_interp"
            )
            
            # Create 3D visualization
            if visualization_mode == "Point Cloud":
                # Scatter plot for point cloud
                mesh_data = go.Scatter3d(
                    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=values,
                        colorscale=colormap,
                        opacity=0.8,
                        colorbar=dict(
                            title=dict(text="Interpolated Value", font=dict(size=14)),
                            thickness=20,
                            len=0.75
                        ),
                        showscale=True
                    ),
                    hovertemplate='<b>Value:</b> %{marker.color:.3f}<br>' +
                                 '<b>X:</b> %{x:.3f}<br>' +
                                 '<b>Y:</b> %{y:.3f}<br>' +
                                 '<b>Z:</b> %{z:.3f}<extra></extra>'
                )
                
                fig = go.Figure(data=mesh_data)
                fig.update_layout(
                    title=dict(
                        text=f"Interpolated Solution<br><sub>{sim_name}</sub>",
                        font=dict(size=20)
                    ),
                    scene=dict(
                        aspectmode="data",
                        camera=dict(
                            eye=dict(x=1.5, y=1.5, z=1.5),
                            up=dict(x=0, y=0, z=1)
                        ),
                        xaxis=dict(
                            title="X",
                            gridcolor="lightgray",
                            showbackground=True,
                            backgroundcolor="white"
                        ),
                        yaxis=dict(
                            title="Y",
                            gridcolor="lightgray",
                            showbackground=True,
                            backgroundcolor="white"
                        ),
                        zaxis=dict(
                            title="Z",
                            gridcolor="lightgray",
                            showbackground=True,
                            backgroundcolor="white"
                        )
                    ),
                    height=700,
                    margin=dict(l=0, r=0, t=80, b=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Field statistics
                st.markdown('<h3 class="sub-header">📊 Interpolated Field Statistics</h3>', unsafe_allow_html=True)
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Min", f"{np.min(values):.3f}")
                with col2:
                    st.metric("Max", f"{np.max(values):.3f}")
                with col3:
                    st.metric("Mean", f"{np.mean(values):.3f}")
                with col4:
                    st.metric("Std Dev", f"{np.std(values):.3f}")
                with col5:
                    st.metric("Range", f"{np.max(values) - np.min(values):.3f}")
        
        else:
            st.error("Interpolated simulation data is malformed.")
        return
    
    # Original simulation handling (existing code)
    if not sim.get('has_mesh', False):
        st.warning("This simulation was loaded without mesh data. Please reload with 'Load Full Mesh' enabled.")
        return
    
    if 'field_info' not in sim or not sim['field_info']:
        st.error("No field data available for this simulation.")
        return
    
    # Field and timestep selection
    col1, col2, col3 = st.columns(3)
    with col1:
        field = st.selectbox(
            "Select Field",
            sorted(sim['field_info'].keys()),
            key="viewer_field_select",
            help="Choose a field to visualize"
        )
    with col2:
        timestep = st.slider(
            "Timestep",
            0, sim['n_timesteps'] - 1, 0,
            key="viewer_timestep_slider",
            help="Select timestep to display"
        )
    with col3:
        colormap = st.selectbox(
            "Colormap",
            EnhancedVisualizer.EXTENDED_COLORMAPS,
            index=EnhancedVisualizer.EXTENDED_COLORMAPS.index(st.session_state.selected_colormap),
            key="viewer_colormap"
        )
    
    # Visualization mode selection
    visualization_mode = st.selectbox(
        "Visualization Mode",
        ["Basic Mesh", "Point Cloud"],
        key="viewer_viz_mode"
    )
    
    # Main 3D visualization
    if 'points' in sim and 'fields' in sim and field in sim['fields']:
        pts = sim['points']
        kind, _ = sim['field_info'][field]
        raw = sim['fields'][field][timestep]
        
        if kind == "scalar":
            values = np.where(np.isnan(raw), 0, raw)
            label = field
        else:
            magnitude = np.linalg.norm(raw, axis=1)
            values = np.where(np.isnan(magnitude), 0, magnitude)
            label = f"{field} (magnitude)"
        
        # Create 3D visualization
        if visualization_mode == "Basic Mesh" and sim.get('triangles') is not None and len(sim['triangles']) > 0:
            tri = sim['triangles']
            
            # Check triangle validity
            valid_triangles = []
            for triangle in tri:
                if all(idx < len(pts) for idx in triangle):
                    valid_triangles.append(triangle)
            
            if valid_triangles:
                valid_triangles = np.array(valid_triangles)
                mesh_data = go.Mesh3d(
                    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                    i=valid_triangles[:, 0], j=valid_triangles[:, 1], k=valid_triangles[:, 2],
                    intensity=values,
                    colorscale=colormap,
                    intensitymode='vertex',
                    colorbar=dict(
                        title=dict(text=label, font=dict(size=14)),
                        thickness=20,
                        len=0.75
                    ),
                    opacity=0.9,
                    lighting=dict(
                        ambient=0.8,
                        diffuse=0.8,
                        specular=0.5,
                        roughness=0.5
                    ),
                    lightposition=dict(x=100, y=200, z=300),
                    hovertemplate='<b>Value:</b> %{intensity:.3f}<br>' +
                                 '<b>X:</b> %{x:.3f}<br>' +
                                 '<b>Y:</b> %{y:.3f}<br>' +
                                 '<b>Z:</b> %{z:.3f}<extra></extra>'
                )
            else:
                # Fallback to scatter plot
                visualization_mode = "Point Cloud"
        
        if visualization_mode == "Point Cloud":
            # Scatter plot for point cloud
            mesh_data = go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    color=values,
                    colorscale=colormap,
                    opacity=0.8,
                    colorbar=dict(
                        title=dict(text=label, font=dict(size=14)),
                        thickness=20,
                        len=0.75
                    ),
                    showscale=True
                ),
                hovertemplate='<b>Value:</b> %{marker.color:.3f}<br>' +
                             '<b>X:</b> %{x:.3f}<br>' +
                             '<b>Y:</b> %{y:.3f}<br>' +
                             '<b>Z:</b> %{z:.3f}<extra></extra>'
            )
        
        fig = go.Figure(data=mesh_data)
        fig.update_layout(
            title=dict(
                text=f"{label} at Timestep {timestep + 1}<br><sub>{sim_name}</sub>",
                font=dict(size=20)
            ),
            scene=dict(
                aspectmode="data",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    up=dict(x=0, y=0, z=1)
                ),
                xaxis=dict(
                    title="X",
                    gridcolor="lightgray",
                    showbackground=True,
                    backgroundcolor="white"
                ),
                yaxis=dict(
                    title="Y",
                    gridcolor="lightgray",
                    showbackground=True,
                    backgroundcolor="white"
                ),
                zaxis=dict(
                    title="Z",
                    gridcolor="lightgray",
                    showbackground=True,
                    backgroundcolor="white"
                )
            ),
            height=700,
            margin=dict(l=0, r=0, t=80, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Field statistics
        st.markdown('<h3 class="sub-header">📊 Field Statistics</h3>', unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Min", f"{np.min(values):.3f}")
        with col2:
            st.metric("Max", f"{np.max(values):.3f}")
        with col3:
            st.metric("Mean", f"{np.mean(values):.3f}")
        with col4:
            st.metric("Std Dev", f"{np.std(values):.3f}")
        with col5:
            st.metric("Range", f"{np.max(values) - np.min(values):.3f}")
    
    # Field evolution over time
    st.markdown('<h3 class="sub-header">📈 Field Evolution Over Time</h3>', unsafe_allow_html=True)
    
    # Find corresponding summary
    summary = next((s for s in combined_summaries if s['name'] == sim_name), None)
    
    if summary and field in summary['field_stats']:
        stats = summary['field_stats'][field]
        
        fig_time = go.Figure()
        
        if stats['mean']:
            # Mean line
            fig_time.add_trace(go.Scatter(
                x=summary['timesteps'],
                y=stats['mean'],
                mode='lines',
                name='Mean',
                line=dict(color='blue', width=3)
            ))
            
            # Confidence band (mean ± std)
            if stats['std']:
                y_upper = np.array(stats['mean']) + np.array(stats['std'])
                y_lower = np.array(stats['mean']) - np.array(stats['std'])
                
                fig_time.add_trace(go.Scatter(
                    x=summary['timesteps'] + summary['timesteps'][::-1],
                    y=np.concatenate([y_upper, y_lower[::-1]]),
                    fill='toself',
                    fillcolor='rgba(0, 100, 255, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False,
                    name='± Std Dev'
                ))
            
            # Max line
            if stats['max']:
                fig_time.add_trace(go.Scatter(
                    x=summary['timesteps'],
                    y=stats['max'],
                    mode='lines',
                    name='Maximum',
                    line=dict(color='red', width=2, dash='dash')
                ))
        
        fig_time.update_layout(
            title=dict(
                text=f"{field} Statistics Over Time",
                font=dict(size=18)
            ),
            xaxis_title="Timestep (ns)",
            yaxis_title=f"{field} Value",
            hovermode="x unified",
            height=400,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        st.plotly_chart(fig_time, use_container_width=True)

def render_3d_visualization():
    """Render enhanced 3D visualization with interpolated support"""
    st.markdown('<h2 class="sub-header">🌐 Advanced 3D Visualization</h2>', unsafe_allow_html=True)
    
    # Get combined simulations
    combined_simulations = st.session_state.interpolated_manager.get_combined_simulations()
    combined_summaries = st.session_state.interpolated_manager.get_combined_summaries()
    
    if not combined_simulations:
        st.markdown("""
        <div class="warning-box">
        <h3>⚠️ No Data Loaded</h3>
        <p>Please load simulations first using the "Load All Simulations" button in the sidebar.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Simulation selection with badges for interpolated
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        # Create display names with badges
        display_names = []
        for sim_name in sorted(combined_simulations.keys()):
            sim = combined_simulations[sim_name]
            if sim.get('is_interpolated', False):
                display_name = f"{sim_name} 🌟"
            else:
                display_name = sim_name
            display_names.append(display_name)
        
        selected_display = st.selectbox(
            "Select Simulation",
            display_names,
            key="3d_sim_select"
        )
        
        # Get actual simulation name
        sim_name = selected_display.split(' 🌟')[0] if ' 🌟' in selected_display else selected_display
    
    sim = combined_simulations[sim_name]
    
    with col2:
        st.metric("Energy", f"{sim['energy_mJ']:.2f} mJ")
    with col3:
        st.metric("Duration", f"{sim['duration_ns']:.2f} ns")
    
    # Show interpolated badge
    if sim.get('is_interpolated', False):
        st.markdown('<div class="interpolated-badge" style="display: inline-block;">INTERPOLATED</div>', 
                   unsafe_allow_html=True)
        st.info(f"Interpolated from: {sim['original_simulation']} | Method: {sim['interpolation_method']}")
    
    # Check if simulation has mesh data
    if not sim.get('has_mesh', False) and not sim.get('is_interpolated', False):
        st.warning("This simulation was loaded without mesh data. Please reload with 'Load Full Mesh' enabled.")
        return
    
    # For interpolated simulations, we need to handle differently
    if sim.get('is_interpolated', False):
        if 'points' in sim and 'values' in sim:
            pts = sim['points']
            values = sim['values']
            
            # Visualization settings
            col1, col2, col3 = st.columns(3)
            with col1:
                visualization_mode = st.selectbox(
                    "Visualization Mode",
                    list(st.session_state.visualizer_3d.visualization_modes.keys()),
                    key="3d_viz_mode_interp"
                )
            with col2:
                colormap = st.selectbox(
                    "Colormap",
                    EnhancedVisualizer.EXTENDED_COLORMAPS,
                    index=EnhancedVisualizer.EXTENDED_COLORMAPS.index(st.session_state.selected_colormap),
                    key="3d_colormap_interp"
                )
            with col3:
                render_engine = st.selectbox(
                    "Render Engine",
                    ["Plotly", "PyVista"],
                    key="3d_render_engine_interp",
                    help="PyVista provides better 3D rendering but may be slower"
                )
            
            # Advanced settings
            with st.expander("⚙️ Advanced Visualization Settings", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    grid_resolution = st.slider(
                        "Grid Resolution",
                        min_value=20,
                        max_value=100,
                        value=40,
                        step=5,
                        key="3d_grid_res_interp"
                    )
                with col2:
                    opacity = st.slider(
                        "Opacity",
                        min_value=0.1,
                        max_value=1.0,
                        value=0.8,
                        step=0.1,
                        key="3d_opacity_interp"
                    )
                with col3:
                    n_isosurfaces = st.slider(
                        "Isosurfaces Count",
                        min_value=2,
                        max_value=8,
                        value=4,
                        step=1,
                        key="3d_isosurfaces_interp"
                    )
            
            # Create visualization based on selected engine
            if render_engine == "PyVista":
                try:
                    # Try PyVista first
                    plotter = st.session_state.visualizer_3d.create_pyvista_visualization(
                        points=pts,
                        values=values,
                        triangles=None,
                        mode=st.session_state.visualizer_3d.visualization_modes[visualization_mode],
                        colormap=colormap.lower(),
                        grid_resolution=grid_resolution,
                        n_isosurfaces=n_isosurfaces,
                        opacity=opacity
                    )
                    
                    if plotter is not None:
                        stpyvista(plotter, key=f"pyvista_interp_{sim_name}", height=600)
                        st.success("✅ PyVista 3D visualization rendered successfully!")
                    else:
                        st.warning("PyVista visualization failed, falling back to Plotly")
                        render_engine = "Plotly"
                
                except Exception as e:
                    st.error(f"PyVista visualization error: {str(e)}")
                    st.warning("Falling back to Plotly visualization")
                    render_engine = "Plotly"
            
            if render_engine == "Plotly":
                # Create Plotly visualization for interpolated data
                fig = st.session_state.visualizer_3d.create_3d_visualization(
                    points=pts,
                    values=values,
                    triangles=None,
                    mode=st.session_state.visualizer_3d.visualization_modes[visualization_mode],
                    colormap=colormap,
                    grid_resolution=grid_resolution,
                    n_isosurfaces=n_isosurfaces,
                    opacity=opacity
                )
                
                # Update title
                fig.update_layout(
                    title=dict(
                        text=f"Interpolated Solution - {visualization_mode}<br><sub>{sim_name}</sub>",
                        font=dict(size=20)
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Field statistics and analysis
            st.markdown('<h3 class="sub-header">📊 Interpolated Field Analysis</h3>', unsafe_allow_html=True)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Min", f"{np.min(values):.3f}")
            with col2:
                st.metric("Max", f"{np.max(values):.3f}")
            with col3:
                st.metric("Mean", f"{np.mean(values):.3f}")
            with col4:
                st.metric("Std Dev", f"{np.std(values):.3f}")
            with col5:
                st.metric("Data Points", f"{len(values):,}")
        
        else:
            st.error("Interpolated simulation data is malformed.")
        return
    
    # Original simulation handling
    if 'field_info' not in sim or not sim['field_info']:
        st.error("No field data available for this simulation.")
        return
    
    # Visualization settings for original simulations
    col1, col2, col3 = st.columns(3)
    with col1:
        field = st.selectbox(
            "Select Field",
            sorted(sim['field_info'].keys()),
            key="3d_field_select"
        )
    with col2:
        timestep = st.slider(
            "Timestep",
            0, sim['n_timesteps'] - 1, 0,
            key="3d_timestep_slider"
        )
    with col3:
        visualization_mode = st.selectbox(
            "Visualization Mode",
            list(st.session_state.visualizer_3d.visualization_modes.keys()),
            key="3d_viz_mode"
        )
    
    # Advanced settings
    with st.expander("⚙️ Advanced Visualization Settings", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            grid_resolution = st.slider(
                "Grid Resolution",
                min_value=20,
                max_value=100,
                value=40,
                step=5,
                key="3d_grid_res"
            )
        with col2:
            opacity = st.slider(
                "Opacity",
                min_value=0.1,
                max_value=1.0,
                value=0.8,
                step=0.1,
                key="3d_opacity"
            )
        with col3:
            n_isosurfaces = st.slider(
                "Isosurfaces Count",
                min_value=2,
                max_value=8,
                value=4,
                step=1,
                key="3d_isosurfaces"
            )
        
        col4, col5 = st.columns(2)
        with col4:
            interpolation_method = st.selectbox(
                "Interpolation Method",
                list(st.session_state.visualizer_3d.interpolator.interpolation_methods.keys()),
                key="3d_interp_method"
            )
        with col5:
            render_engine = st.selectbox(
                "Render Engine",
                ["Plotly", "PyVista"],
                key="3d_render_engine",
                help="PyVista provides better 3D rendering but may be slower"
            )
    
    # Get data for visualization
    if 'points' in sim and 'fields' in sim and field in sim['fields']:
        pts = sim['points']
        kind, _ = sim['field_info'][field]
        raw = sim['fields'][field][timestep]
        
        if kind == "scalar":
            values = np.where(np.isnan(raw), 0, raw)
            label = field
        else:
            magnitude = np.linalg.norm(raw, axis=1)
            values = np.where(np.isnan(magnitude), 0, magnitude)
            label = f"{field} (magnitude)"
        
        triangles = sim.get('triangles')
        
        # Create visualization based on selected engine
        if render_engine == "PyVista":
            try:
                # Try PyVista first
                plotter = st.session_state.visualizer_3d.create_pyvista_visualization(
                    points=pts,
                    values=values,
                    triangles=triangles,
                    mode=st.session_state.visualizer_3d.visualization_modes[visualization_mode],
                    colormap=st.session_state.selected_colormap.lower(),
                    grid_resolution=grid_resolution,
                    n_isosurfaces=n_isosurfaces,
                    opacity=opacity
                )
                
                if plotter is not None:
                    stpyvista(plotter, key=f"pyvista_{sim_name}_{field}_{timestep}", height=600)
                    st.success("✅ PyVista 3D visualization rendered successfully!")
                else:
                    st.warning("PyVista visualization failed, falling back to Plotly")
                    render_engine = "Plotly"
                
            except Exception as e:
                st.error(f"PyVista visualization error: {str(e)}")
                st.warning("Falling back to Plotly visualization")
                render_engine = "Plotly"
        
        if render_engine == "Plotly":
            # Create Plotly visualization
            fig = st.session_state.visualizer_3d.create_3d_visualization(
                points=pts,
                values=values,
                triangles=triangles,
                mode=st.session_state.visualizer_3d.visualization_modes[visualization_mode],
                colormap=st.session_state.selected_colormap,
                grid_resolution=grid_resolution,
                n_isosurfaces=n_isosurfaces,
                opacity=opacity
            )
            
            # Update title
            fig.update_layout(
                title=dict(
                    text=f"{label} - {visualization_mode}<br><sub>{sim_name} | Timestep: {timestep + 1}</sub>",
                    font=dict(size=20)
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)

def render_interpolation_extrapolation():
    """Render the enhanced interpolation/extrapolation interface"""
    st.markdown('<h2 class="sub-header">🔮 Interpolation/Extrapolation Engine</h2>', 
               unsafe_allow_html=True)
    
    # Get combined summaries
    combined_summaries = st.session_state.interpolated_manager.get_combined_summaries()
    
    if not combined_summaries:
        st.markdown("""
        <div class="warning-box">
        <h3>⚠️ No Data Loaded</h3>
        <p>Please load simulations first to enable interpolation/extrapolation capabilities.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown("""
    <div class="info-box">
    <h3>🧠 Physics-Informed Attention Mechanism</h3>
    <p>This engine uses a <strong>transformer-inspired multi-head attention mechanism</strong> with <strong>spatial locality regulation</strong> to interpolate and extrapolate simulation results. The model learns from existing FEA simulations and can predict outcomes for new parameter combinations with quantified confidence.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display loaded simulations summary with interpolated badge
    with st.expander("📋 Loaded Simulations Summary", expanded=True):
        if combined_summaries:
            df_summary = pd.DataFrame([{
                'Simulation': s['name'],
                'Type': '🌟 Interpolated' if s.get('is_interpolated', False) else 'Original',
                'Energy (mJ)': s['energy'],
                'Duration (ns)': s['duration'],
                'Timesteps': len(s['timesteps']),
                'Fields': ', '.join(sorted(s['field_stats'].keys())[:3]) + ('...' if len(s['field_stats']) > 3 else '')
            } for s in combined_summaries])
            
            st.dataframe(
                df_summary.style.format({
                    'Energy (mJ)': '{:.2f}',
                    'Duration (ns)': '{:.2f}'
                }).background_gradient(subset=['Energy (mJ)', 'Duration (ns)'], cmap='Blues'),
                use_container_width=True,
                height=300
            )
    
    # Load summaries into extrapolator (including interpolated ones)
    st.session_state.extrapolator.load_summaries(combined_summaries)
    
    # Query parameters
    st.markdown('<h3 class="sub-header">🎯 Query Parameters</h3>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        # Get parameter ranges from loaded data
        if combined_summaries:
            energies = [s['energy'] for s in combined_summaries]
            min_energy, max_energy = min(energies), max(energies)
        else:
            min_energy, max_energy = 0.1, 50.0
        
        energy_query = st.number_input(
            "Energy (mJ)",
            min_value=float(min_energy * 0.5),
            max_value=float(max_energy * 2.0),
            value=float((min_energy + max_energy) / 2),
            step=0.1,
            key="interp_energy",
            help=f"Training range: {min_energy:.1f} - {max_energy:.1f} mJ"
        )
    
    with col2:
        if combined_summaries:
            durations = [s['duration'] for s in combined_summaries]
            min_duration, max_duration = min(durations), max(durations)
        else:
            min_duration, max_duration = 0.5, 20.0
        
        duration_query = st.number_input(
            "Pulse Duration (ns)",
            min_value=float(min_duration * 0.5),
            max_value=float(max_duration * 2.0),
            value=float((min_duration + max_duration) / 2),
            step=0.1,
            key="interp_duration",
            help=f"Training range: {min_duration:.1f} - {max_duration:.1f} ns"
        )
    
    with col3:
        max_time = st.number_input(
            "Max Prediction Time (ns)",
            min_value=1,
            max_value=100,
            value=20,
            step=1,
            key="interp_maxtime"
        )
    
    with col4:
        time_resolution = st.selectbox(
            "Time Resolution",
            ["1 ns", "2 ns", "5 ns", "10 ns"],
            index=0,
            key="interp_resolution"
        )
    
    # Generate time points
    time_step_map = {"1 ns": 1, "2 ns": 2, "5 ns": 5, "10 ns": 10}
    time_step = time_step_map[time_resolution]
    time_points = np.arange(1, max_time + 1, time_step)
    
    # Model parameters
    with st.expander("⚙️ Attention Mechanism Parameters", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            sigma_param = st.slider(
                "Kernel Width (σ)",
                min_value=0.1,
                max_value=1.0,
                value=0.3,
                step=0.05,
                key="interp_sigma",
                help="Controls the attention focus width"
            )
        with col2:
            spatial_weight = st.slider(
                "Spatial Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                key="interp_spatial",
                help="Weight for spatial locality regulation"
            )
        with col3:
            n_heads = st.slider(
                "Attention Heads",
                min_value=1,
                max_value=8,
                value=4,
                step=1,
                key="interp_heads",
                help="Number of parallel attention heads"
            )
        with col4:
            temperature = st.slider(
                "Temperature",
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                step=0.1,
                key="interp_temp",
                help="Softmax temperature for attention weights"
            )
        
        # Update extrapolator parameters
        st.session_state.extrapolator.sigma_param = sigma_param
        st.session_state.extrapolator.spatial_weight = spatial_weight
        st.session_state.extrapolator.n_heads = n_heads
        st.session_state.extrapolator.temperature = temperature
    
    # Run prediction
    if st.button("🚀 Run Physics-Informed Prediction", type="primary", use_container_width=True):
        with st.spinner("Running multi-head attention prediction with spatial locality regulation..."):
            results = st.session_state.extrapolator.predict_time_series(
                energy_query, duration_query, time_points
            )
            
            if results and 'field_predictions' in results and results['field_predictions']:
                st.markdown("""
                <div class="success-box">
                <h3>✅ Prediction Successful</h3>
                <p>Physics-informed predictions generated using multi-head attention with spatial locality regulation.</p>
                <p>Predictions are based on {num_simulations} simulations ({num_original} original, {num_interpolated} interpolated).</p>
                </div>
                """.format(
                    num_simulations=len(combined_summaries),
                    num_original=len([s for s in combined_summaries if not s.get('is_interpolated', False)]),
                    num_interpolated=len([s for s in combined_summaries if s.get('is_interpolated', False)])
                ), unsafe_allow_html=True)
                
                # Visualization tabs
                tab1, tab2, tab3, tab4 = st.tabs(["📈 Predictions", "🧠 Attention", "🌐 3D Analysis", "📊 Details"])
                
                with tab1:
                    render_prediction_results(results, time_points, energy_query, duration_query)
                
                with tab2:
                    render_attention_visualization(results, energy_query, duration_query, time_points)
                
                with tab3:
                    render_3d_analysis(results, time_points, energy_query, duration_query)
                
                with tab4:
                    render_detailed_results(results, time_points, energy_query, duration_query)
            else:
                st.error("Prediction failed. Please check input parameters and ensure sufficient training data.")

def render_comparative_analysis():
    """Render enhanced comparative analysis interface with interpolated support"""
    st.markdown('<h2 class="sub-header">📊 Comparative Analysis</h2>', unsafe_allow_html=True)
    
    # Get combined simulations and summaries
    combined_simulations = st.session_state.interpolated_manager.get_combined_simulations()
    combined_summaries = st.session_state.interpolated_manager.get_combined_summaries()
    
    if not combined_simulations:
        st.markdown("""
        <div class="warning-box">
        <h3>⚠️ No Data Loaded</h3>
        <p>Please load simulations first to enable comparative analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Target simulation selection with badges
    st.markdown('<h3 class="sub-header">🎯 Select Target Simulation</h3>', unsafe_allow_html=True)
    
    available_simulations = sorted(combined_simulations.keys())
    
    col1, col2 = st.columns([3, 1])
    with col1:
        # Create display names with badges
        display_names = []
        for sim_name in available_simulations:
            sim = combined_simulations[sim_name]
            if sim.get('is_interpolated', False):
                display_name = f"{sim_name} 🌟"
            else:
                display_name = sim_name
            display_names.append(display_name)
        
        selected_display = st.selectbox(
            "Select target simulation for highlighting",
            display_names,
            key="target_sim_select",
            help="This simulation will be highlighted in all visualizations"
        )
        
        # Get actual simulation name
        target_simulation = selected_display.split(' 🌟')[0] if ' 🌟' in selected_display else selected_display
    
    with col2:
        n_comparisons = st.number_input(
            "Number of comparisons",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            key="n_comparisons"
        )
    
    # Select comparison simulations (excluding target)
    comparison_sims = [sim for sim in available_simulations if sim != target_simulation]
    selected_comparisons = st.multiselect(
        "Select simulations for comparison",
        comparison_sims,
        default=comparison_sims[:min(n_comparisons - 1, len(comparison_sims))],
        help="Select simulations to compare with the target"
    )
    
    # Include target in visualization list
    visualization_sims = [target_simulation] + selected_comparisons
    
    if not visualization_sims:
        st.info("Please select at least one simulation for comparison.")
        return
    
    # Field selection
    st.markdown('<h3 class="sub-header">📈 Select Field for Analysis</h3>', unsafe_allow_html=True)
    
    # Get available fields from selected simulations
    available_fields = set()
    for sim_name in visualization_sims:
        if sim_name in combined_simulations:
            sim = combined_simulations[sim_name]
            if sim.get('is_interpolated', False):
                available_fields.add('interpolated_field')
            else:
                if 'field_info' in sim:
                    available_fields.update(sim['field_info'].keys())
    
    if not available_fields:
        st.error("No field data available for selected simulations.")
        return
    
    selected_field = st.selectbox(
        "Select field for analysis",
        sorted(available_fields),
        key="comparison_field",
        help="Choose a field to compare across simulations"
    )
    
    # Visualization tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Sunburst", "🎯 Radar", "⏱️ Evolution", "🌐 3D Analysis"])
    
    with tab1:
        # Sunburst chart
        st.markdown("##### 📊 Hierarchical Sunburst Chart")
        sunburst_fig = st.session_state.visualizer.create_sunburst_chart(
            combined_summaries,
            selected_field,
            highlight_sim=target_simulation
        )
        if sunburst_fig.data:
            st.plotly_chart(sunburst_fig, use_container_width=True)
        else:
            st.info("Insufficient data for sunburst chart")
    
    with tab2:
        # Radar chart
        st.markdown("##### 🎯 Multi-Field Radar Comparison")
        radar_fig = st.session_state.visualizer.create_radar_chart(
            combined_summaries,
            visualization_sims,
            target_sim=target_simulation
        )
        if radar_fig.data:
            st.plotly_chart(radar_fig, use_container_width=True)
        else:
            st.info("Insufficient data for radar chart")
    
    with tab3:
        # Field evolution comparison
        st.markdown("##### ⏱️ Field Evolution Over Time")
        evolution_fig = st.session_state.visualizer.create_field_evolution_comparison(
            combined_summaries,
            visualization_sims,
            selected_field,
            target_sim=target_simulation
        )
        if evolution_fig.data:
            st.plotly_chart(evolution_fig, use_container_width=True)
        else:
            st.info(f"No {selected_field} data available for selected simulations")
    
    with tab4:
        # 3D parameter space analysis
        st.markdown("##### 🌐 3D Parameter Space Analysis")
        
        if combined_summaries:
            # Extract data for 3D plot
            energies = []
            durations = []
            max_vals = []
            sim_names = []
            is_target = []
            is_interpolated = []
            
            for summary in combined_summaries:
                if summary['name'] in visualization_sims and selected_field in summary['field_stats']:
                    energies.append(summary['energy'])
                    durations.append(summary['duration'])
                    sim_names.append(summary['name'])
                    is_target.append(summary['name'] == target_simulation)
                    is_interpolated.append(summary.get('is_interpolated', False))
                    
                    stats = summary['field_stats'][selected_field]
                    if stats['max']:
                        max_vals.append(np.max(stats['max']))
                    else:
                        max_vals.append(0)
            
            if energies and durations and max_vals:
                # Create 3D scatter plot
                fig_3d = go.Figure()
                
                # Separate points by type
                original_indices = [i for i, interp in enumerate(is_interpolated) if not interp]
                interpolated_indices = [i for i, interp in enumerate(is_interpolated) if interp]
                target_indices = [i for i, target in enumerate(is_target) if target]
                
                # Original points
                if original_indices:
                    original_x = [energies[i] for i in original_indices]
                    original_y = [durations[i] for i in original_indices]
                    original_z = [max_vals[i] for i in original_indices]
                    original_names = [sim_names[i] for i in original_indices]
                    
                    fig_3d.add_trace(go.Scatter3d(
                        x=original_x,
                        y=original_y,
                        z=original_z,
                        mode='markers',
                        marker=dict(
                            size=8,
                            color='blue',
                            opacity=0.7,
                            symbol='circle'
                        ),
                        name='Original Sims',
                        text=original_names,
                        hovertemplate='%{text}<br>Energy: %{x:.1f} mJ<br>Duration: %{y:.1f} ns<br>Value: %{z:.1f}<extra></extra>'
                    ))
                
                # Interpolated points
                if interpolated_indices:
                    interp_x = [energies[i] for i in interpolated_indices]
                    interp_y = [durations[i] for i in interpolated_indices]
                    interp_z = [max_vals[i] for i in interpolated_indices]
                    interp_names = [sim_names[i] for i in interpolated_indices]
                    
                    fig_3d.add_trace(go.Scatter3d(
                        x=interp_x,
                        y=interp_y,
                        z=interp_z,
                        mode='markers',
                        marker=dict(
                            size=10,
                            color='magenta',
                            opacity=0.9,
                            symbol='diamond'
                        ),
                        name='Interpolated Sims',
                        text=interp_names,
                        hovertemplate='%{text} (Interpolated)<br>Energy: %{x:.1f} mJ<br>Duration: %{y:.1f} ns<br>Value: %{z:.1f}<extra></extra>'
                    ))
                
                # Target point
                if target_indices:
                    for idx in target_indices:
                        fig_3d.add_trace(go.Scatter3d(
                            x=[energies[idx]],
                            y=[durations[idx]],
                            z=[max_vals[idx]],
                            mode='markers',
                            marker=dict(
                                size=15,
                                color='red',
                                symbol='star'
                            ),
                            name='Target Sim',
                            text=sim_names[idx],
                            hovertemplate='<b>%{text}</b><br>Energy: %{x:.1f} mJ<br>Duration: %{y:.1f} ns<br>Value: %{z:.1f}<extra></extra>'
                        ))
                
                fig_3d.update_layout(
                    title=f"Parameter Space - {selected_field}",
                    scene=dict(
                        xaxis_title="Energy (mJ)",
                        yaxis_title="Duration (ns)",
                        zaxis_title=f"Max {selected_field}"
                    ),
                    height=600
                )
                
                st.plotly_chart(fig_3d, use_container_width=True)

def render_create_interpolated():
    """Render interface for creating interpolated solutions"""
    st.markdown('<h2 class="sub-header">✨ Create Interpolated Solutions</h2>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.markdown("""
        <div class="warning-box">
        <h3>⚠️ No Data Loaded</h3>
        <p>Please load simulations first to create interpolated solutions.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    simulations = st.session_state.simulations
    
    st.markdown("""
    <div class="info-box">
    <h3>🔮 Create New Interpolated Solutions</h3>
    <p>This tool allows you to create interpolated solutions from existing simulation data. 
    Interpolated solutions can be used in all visualization modes alongside original data.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Source simulation selection
    col1, col2, col3 = st.columns(3)
    with col1:
        source_sim = st.selectbox(
            "Select Source Simulation",
            sorted(simulations.keys()),
            key="interp_source_sim",
            help="Choose the simulation to interpolate from"
        )
    
    sim = simulations[source_sim]
    
    with col2:
        st.metric("Energy", f"{sim['energy_mJ']:.2f} mJ")
    with col3:
        st.metric("Duration", f"{sim['duration_ns']:.2f} ns")
    
    if not sim.get('has_mesh', False):
        st.warning("This simulation was loaded without mesh data. Please reload with 'Load Full Mesh' enabled.")
        return
    
    if 'field_info' not in sim or not sim['field_info']:
        st.error("No field data available for this simulation.")
        return
    
    # Field and timestep selection
    col1, col2, col3 = st.columns(3)
    with col1:
        field = st.selectbox(
            "Select Field to Interpolate",
            sorted(sim['field_info'].keys()),
            key="interp_field_select"
        )
    with col2:
        timestep = st.slider(
            "Source Timestep",
            0, sim['n_timesteps'] - 1, 0,
            key="interp_timestep_slider"
        )
    with col3:
        interpolation_method = st.selectbox(
            "Interpolation Method",
            list(st.session_state.visualizer_3d.interpolator.interpolation_methods.keys()),
            key="create_interp_method"
        )
    
    # Interpolation parameters
    with st.expander("⚙️ Interpolation Parameters", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            grid_resolution = st.slider(
                "Output Grid Resolution",
                min_value=20,
                max_value=200,
                value=100,
                step=10,
                key="create_grid_res",
                help="Resolution of the interpolated grid"
            )
        with col2:
            padding = st.slider(
                "Grid Padding",
                min_value=0.0,
                max_value=0.5,
                value=0.1,
                step=0.05,
                key="create_padding",
                help="Padding around the point cloud"
            )
        with col3:
            epsilon = st.number_input(
                "RBF Epsilon",
                min_value=0.01,
                max_value=10.0,
                value=1.0,
                step=0.1,
                key="create_epsilon",
                help="RBF kernel parameter (for RBF interpolation only)"
            )
    
    # Create interpolated solution
    if st.button("🚀 Create Interpolated Solution", type="primary", use_container_width=True):
        with st.spinner("Creating interpolated solution..."):
            try:
                # Get source data
                pts = sim['points']
                kind, _ = sim['field_info'][field]
                raw = sim['fields'][field][timestep]
                
                if kind == "scalar":
                    values = np.where(np.isnan(raw), 0, raw)
                else:
                    magnitude = np.linalg.norm(raw, axis=1)
                    values = np.where(np.isnan(magnitude), 0, magnitude)
                
                # Create regular grid
                X_grid, Y_grid, Z_grid, grid_points, _ = \
                    st.session_state.visualizer_3d.interpolator.create_regular_grid(
                        pts, values, grid_resolution, padding
                    )
                
                if grid_points is not None:
                    # Interpolate to grid
                    method_key = st.session_state.visualizer_3d.interpolator.interpolation_methods[interpolation_method]
                    grid_values = st.session_state.visualizer_3d.interpolator.interpolate_to_grid(
                        pts, values, grid_points, method=method_key, epsilon=epsilon
                    )
                    
                    if grid_values is not None:
                        # Create interpolated solution
                        interpolated_sim = st.session_state.interpolated_manager.create_interpolated_solution(
                            sim_name=source_sim,
                            field=field,
                            timestep=timestep,
                            interpolated_points=grid_points,
                            interpolated_values=grid_values,
                            method=interpolation_method,
                            energy=sim['energy_mJ'],
                            duration=sim['duration_ns']
                        )
                        
                        st.success(f"✅ Interpolated solution created successfully!")
                        st.markdown(f"""
                        <div class="success-box">
                        <h3>✨ Interpolation Complete</h3>
                        <p><strong>Solution ID:</strong> {interpolated_sim['name']}</p>
                        <p><strong>Original:</strong> {source_sim} | Field: {field} | Timestep: {timestep}</p>
                        <p><strong>Method:</strong> {interpolation_method}</p>
                        <p><strong>Grid Size:</strong> {len(grid_points):,} points</p>
                        <p>The interpolated solution has been saved and is now available in all visualization modes.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show preview
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("##### Original Data")
                            st.metric("Points", f"{len(pts):,}")
                            st.metric("Min Value", f"{np.min(values):.3f}")
                            st.metric("Max Value", f"{np.max(values):.3f}")
                        
                        with col2:
                            st.markdown("##### Interpolated Data")
                            st.metric("Grid Points", f"{len(grid_points):,}")
                            st.metric("Min Value", f"{np.min(grid_values):.3f}")
                            st.metric("Max Value", f"{np.max(grid_values):.3f}")
                        
                        # Visual comparison
                        st.markdown("##### 📊 Visual Comparison")
                        
                        fig_compare = make_subplots(
                            rows=1, cols=2,
                            subplot_titles=("Original Point Cloud", "Interpolated Grid"),
                            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]]
                        )
                        
                        # Original
                        fig_compare.add_trace(
                            go.Scatter3d(
                                x=pts[::10, 0],  # Sample for clarity
                                y=pts[::10, 1],
                                z=pts[::10, 2],
                                mode='markers',
                                marker=dict(
                                    size=3,
                                    color=values[::10],
                                    colorscale='Viridis',
                                    opacity=0.7
                                ),
                                name='Original'
                            ),
                            row=1, col=1
                        )
                        
                        # Interpolated
                        fig_compare.add_trace(
                            go.Scatter3d(
                                x=grid_points[::10, 0],
                                y=grid_points[::10, 1],
                                z=grid_points[::10, 2],
                                mode='markers',
                                marker=dict(
                                    size=3,
                                    color=grid_values[::10],
                                    colorscale='Viridis',
                                    opacity=0.7
                                ),
                                name='Interpolated'
                            ),
                            row=1, col=2
                        )
                        
                        fig_compare.update_layout(
                            height=500,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig_compare, use_container_width=True)
                        
                        # Add option to view immediately
                        if st.button("👁️ View Interpolated Solution in 3D Visualizer"):
                            st.session_state.current_mode = "3D Visualization"
                            st.rerun()
                    
                    else:
                        st.error("Interpolation failed to produce valid grid values.")
                else:
                    st.error("Failed to create regular grid.")
                    
            except Exception as e:
                st.error(f"Error creating interpolated solution: {str(e)}")
                st.error(traceback.format_exc())

# =============================================
# HELPER FUNCTIONS FOR INTERPOLATION/EXTRAPOLATION
# =============================================
def render_prediction_results(results, time_points, energy_query, duration_query):
    """Render prediction results visualization"""
    # Determine which fields to plot
    available_fields = list(results['field_predictions'].keys())
    
    if not available_fields:
        st.warning("No field predictions available.")
        return
    
    # Create subplots
    n_fields = min(len(available_fields), 4)
    fig = make_subplots(
        rows=n_fields, cols=1,
        subplot_titles=[f"Predicted {field}" for field in available_fields[:n_fields]],
        vertical_spacing=0.1,
        shared_xaxes=True
    )
    
    for idx, field in enumerate(available_fields[:n_fields]):
        row = idx + 1
        
        # Plot mean prediction
        if results['field_predictions'][field]['mean']:
            fig.add_trace(
                go.Scatter(
                    x=time_points,
                    y=results['field_predictions'][field]['mean'],
                    mode='lines+markers',
                    name=f'{field} (mean)',
                    line=dict(width=3, color='blue'),
                    fillcolor='rgba(0, 0, 255, 0.1)'
                ),
                row=row, col=1
            )
        
        # Add confidence band (mean ± std)
        if (results['field_predictions'][field]['mean'] and 
            results['field_predictions'][field]['std']):
            
            mean_vals = results['field_predictions'][field]['mean']
            std_vals = results['field_predictions'][field]['std']
            
            y_upper = np.array(mean_vals) + np.array(std_vals)
            y_lower = np.array(mean_vals) - np.array(std_vals)
            
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([time_points, time_points[::-1]]),
                    y=np.concatenate([y_upper, y_lower[::-1]]),
                    fill='toself',
                    fillcolor='rgba(0, 0, 255, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False,
                    name=f'{field} ± std'
                ),
                row=row, col=1
            )
    
    fig.update_layout(
        height=300 * n_fields,
        title_text=f"Field Predictions (E={energy_query:.1f} mJ, τ={duration_query:.1f} ns)",
        showlegend=True,
        hovermode="x unified"
    )
    
    # Update y-axes
    for i in range(1, n_fields + 1):
        fig.update_yaxes(title_text="Value", row=i, col=1)
    
    fig.update_xaxes(title_text="Time (ns)", row=n_fields, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Confidence plot
    if results['confidence_scores']:
        fig_conf = go.Figure()
        fig_conf.add_trace(go.Scatter(
            x=time_points,
            y=results['confidence_scores'],
            mode='lines+markers',
            line=dict(color='orange', width=3),
            fill='tozeroy',
            fillcolor='rgba(255, 165, 0, 0.2)',
            name='Prediction Confidence'
        ))
        
        fig_conf.update_layout(
            title="Prediction Confidence Over Time",
            xaxis_title="Time (ns)",
            yaxis_title="Confidence",
            height=400,
            yaxis_range=[0, 1]
        )
        
        st.plotly_chart(fig_conf, use_container_width=True)
        
        # Confidence insights
        avg_conf = np.mean(results['confidence_scores'])
        min_conf = np.min(results['confidence_scores'])
        max_conf = np.max(results['confidence_scores'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Confidence", f"{avg_conf:.3f}")
        with col2:
            st.metric("Minimum Confidence", f"{min_conf:.3f}")
        with col3:
            st.metric("Maximum Confidence", f"{max_conf:.3f}")
        
        if avg_conf < 0.3:
            st.warning("⚠️ **Low Confidence**: Query parameters are far from training data. Extrapolation risk is high.")
        elif avg_conf < 0.6:
            st.info("ℹ️ **Moderate Confidence**: Query parameters are in extrapolation region.")
        else:
            st.success("✅ **High Confidence**: Query parameters are well-supported by training data.")

def render_attention_visualization(results, energy_query, duration_query, time_points):
    """Render attention mechanism visualizations"""
    if not results['attention_maps'] or len(results['attention_maps'][0]) == 0:
        st.info("No attention data available.")
        return
    
    st.markdown('<h4 class="sub-header">🧠 Attention Mechanism Visualization</h4>', unsafe_allow_html=True)
    
    # Select timestep for attention visualization
    selected_timestep_idx = st.slider(
        "Select timestep for attention visualization",
        0, len(time_points) - 1, 0,
        key="attention_timestep"
    )
    
    attention_weights = results['attention_maps'][selected_timestep_idx]
    selected_time = time_points[selected_timestep_idx]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 3D attention heatmap
        st.markdown("##### 3D Attention Distribution")
        heatmap_3d = st.session_state.visualizer.create_attention_heatmap_3d(
            attention_weights,
            st.session_state.extrapolator.source_metadata
        )
        if heatmap_3d.data:
            st.plotly_chart(heatmap_3d, use_container_width=True)
        else:
            st.info("Insufficient data for 3D heatmap")
    
    with col2:
        # Attention network
        st.markdown("##### Attention Network")
        network_fig = st.session_state.visualizer.create_attention_network(
            attention_weights,
            st.session_state.extrapolator.source_metadata,
            top_k=8
        )
        if network_fig.data:
            st.plotly_chart(network_fig, use_container_width=True)
        else:
            st.info("Insufficient data for network visualization")
    
    # Attention weight distribution
    st.markdown("##### Attention Weight Distribution")
    
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=attention_weights,
        nbinsx=30,
        marker_color='skyblue',
        opacity=0.7,
        name='Attention Weights'
    ))
    
    fig_dist.update_layout(
        title=f"Attention Weight Distribution at t={selected_time} ns",
        xaxis_title="Attention Weight",
        yaxis_title="Frequency",
        height=400
    )
    
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Top attention sources
    if len(attention_weights) > 0:
        # Get top attention sources
        top_indices = np.argsort(attention_weights)[-10:][::-1]
        
        st.markdown("##### Top 10 Attention Sources")
        
        top_sources_data = []
        for idx in top_indices:
            if idx < len(st.session_state.extrapolator.source_metadata):
                meta = st.session_state.extrapolator.source_metadata[idx]
                top_sources_data.append({
                    'Simulation': meta['name'],
                    'Energy (mJ)': meta['energy'],
                    'Duration (ns)': meta['duration'],
                    'Time (ns)': meta['time'],
                    'Attention Weight': attention_weights[idx]
                })
        
        if top_sources_data:
            df_top = pd.DataFrame(top_sources_data)
            st.dataframe(
                df_top.style.format({
                    'Energy (mJ)': '{:.2f}',
                    'Duration (ns)': '{:.2f}',
                    'Time (ns)': '{:.1f}',
                    'Attention Weight': '{:.4f}'
                }).background_gradient(subset=['Attention Weight'], cmap='YlOrRd'),
                use_container_width=True
            )

def render_3d_analysis(results, time_points, energy_query, duration_query):
    """Render 3D analysis visualizations"""
    st.markdown('<h4 class="sub-header">🌐 3D Parameter Space Analysis</h4>', unsafe_allow_html=True)
    
    # Get combined summaries
    combined_summaries = st.session_state.interpolated_manager.get_combined_summaries()
    
    # Create 3D parameter space visualization
    if combined_summaries:
        # Extract training data
        train_energies = []
        train_durations = []
        train_max_temps = []
        train_max_stresses = []
        train_types = []  # 0 for original, 1 for interpolated
        
        for summary in combined_summaries:
            train_energies.append(summary['energy'])
            train_durations.append(summary['duration'])
            train_types.append(1 if summary.get('is_interpolated', False) else 0)
            
            if 'temperature' in summary['field_stats']:
                train_max_temps.append(np.max(summary['field_stats']['temperature']['max']))
            else:
                train_max_temps.append(0)
            
            if 'principal stress' in summary['field_stats']:
                train_max_stresses.append(np.max(summary['field_stats']['principal stress']['max']))
            else:
                train_max_stresses.append(0)
        
        # Add query point
        query_max_temp = np.max(results['field_predictions'].get('temperature', {}).get('max', [0]))
        query_max_stress = np.max(results['field_predictions'].get('principal stress', {}).get('max', [0]))
        
        # Create 3D scatter plot
        col1, col2 = st.columns(2)
        
        with col1:
            # Temperature parameter space
            fig_temp = go.Figure()
            
            # Separate original and interpolated points
            original_indices = [i for i, t in enumerate(train_types) if t == 0]
            interpolated_indices = [i for i, t in enumerate(train_types) if t == 1]
            
            # Original training points
            if original_indices:
                fig_temp.add_trace(go.Scatter3d(
                    x=[train_energies[i] for i in original_indices],
                    y=[train_durations[i] for i in original_indices],
                    z=[train_max_temps[i] for i in original_indices],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color='blue',
                        opacity=0.7,
                        symbol='circle'
                    ),
                    name='Original Training Data',
                    hovertemplate='Original<br>Energy: %{x:.1f} mJ<br>Duration: %{y:.1f} ns<br>Max Temp: %{z:.1f}<extra></extra>'
                ))
            
            # Interpolated training points
            if interpolated_indices:
                fig_temp.add_trace(go.Scatter3d(
                    x=[train_energies[i] for i in interpolated_indices],
                    y=[train_durations[i] for i in interpolated_indices],
                    z=[train_max_temps[i] for i in interpolated_indices],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='magenta',
                        opacity=0.9,
                        symbol='diamond'
                    ),
                    name='Interpolated Training Data',
                    hovertemplate='Interpolated<br>Energy: %{x:.1f} mJ<br>Duration: %{y:.1f} ns<br>Max Temp: %{z:.1f}<extra></extra>'
                ))
            
            # Query point
            fig_temp.add_trace(go.Scatter3d(
                x=[energy_query],
                y=[duration_query],
                z=[query_max_temp],
                mode='markers',
                marker=dict(
                    size=12,
                    color='red',
                    symbol='star'
                ),
                name='Query Point',
                hovertemplate='Query<br>Energy: %{x:.1f} mJ<br>Duration: %{y:.1f} ns<br>Pred Temp: %{z:.1f}<extra></extra>'
            ))
            
            fig_temp.update_layout(
                title="Parameter Space - Maximum Temperature",
                scene=dict(
                    xaxis_title="Energy (mJ)",
                    yaxis_title="Duration (ns)",
                    zaxis_title="Max Temperature"
                ),
                height=500
            )
            
            st.plotly_chart(fig_temp, use_container_width=True)
        
        with col2:
            # Stress parameter space
            fig_stress = go.Figure()
            
            # Original training points
            if original_indices:
                fig_stress.add_trace(go.Scatter3d(
                    x=[train_energies[i] for i in original_indices],
                    y=[train_durations[i] for i in original_indices],
                    z=[train_max_stresses[i] for i in original_indices],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color='blue',
                        opacity=0.7,
                        symbol='circle'
                    ),
                    name='Original Training Data',
                    hovertemplate='Original<br>Energy: %{x:.1f} mJ<br>Duration: %{y:.1f} ns<br>Max Stress: %{z:.1f}<extra></extra>'
                ))
            
            # Interpolated training points
            if interpolated_indices:
                fig_stress.add_trace(go.Scatter3d(
                    x=[train_energies[i] for i in interpolated_indices],
                    y=[train_durations[i] for i in interpolated_indices],
                    z=[train_max_stresses[i] for i in interpolated_indices],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='magenta',
                        opacity=0.9,
                        symbol='diamond'
                    ),
                    name='Interpolated Training Data',
                    hovertemplate='Interpolated<br>Energy: %{x:.1f} mJ<br>Duration: %{y:.1f} ns<br>Max Stress: %{z:.1f}<extra></extra>'
                ))
            
            # Query point
            fig_stress.add_trace(go.Scatter3d(
                x=[energy_query],
                y=[duration_query],
                z=[query_max_stress],
                mode='markers',
                marker=dict(
                    size=12,
                    color='red',
                    symbol='star'
                ),
                name='Query Point',
                hovertemplate='Query<br>Energy: %{x:.1f} mJ<br>Duration: %{y:.1f} ns<br>Pred Stress: %{z:.1f}<extra></extra>'
            ))
            
            fig_stress.update_layout(
                title="Parameter Space - Maximum Stress",
                scene=dict(
                    xaxis_title="Energy (mJ)",
                    yaxis_title="Duration (ns)",
                    zaxis_title="Max Stress"
                ),
                height=500
            )
            
            st.plotly_chart(fig_stress, use_container_width=True)

def render_detailed_results(results, time_points, energy_query, duration_query):
    """Render detailed prediction results"""
    st.markdown('<h4 class="sub-header">📊 Detailed Prediction Results</h4>', unsafe_allow_html=True)
    
    # Create results table
    data_rows = []
    for idx, t in enumerate(time_points):
        row = {'Time (ns)': t}
        
        for field in results['field_predictions']:
            if field in results['field_predictions']:
                if idx < len(results['field_predictions'][field]['mean']):
                    row[f'{field}_mean'] = results['field_predictions'][field]['mean'][idx]
                    row[f'{field}_max'] = results['field_predictions'][field]['max'][idx]
                    row[f'{field}_std'] = results['field_predictions'][field]['std'][idx]
        
        if idx < len(results['confidence_scores']):
            row['confidence'] = results['confidence_scores'][idx]
        
        data_rows.append(row)
    
    if data_rows:
        df_results = pd.DataFrame(data_rows)
        
        # Format numeric columns
        format_dict = {}
        for col in df_results.columns:
            if col != 'Time (ns)':
                format_dict[col] = "{:.3f}"
        
        # Display with highlighting
        styled_df = df_results.style.format(format_dict)
        
        # Add conditional formatting for confidence
        def highlight_confidence(val):
            if isinstance(val, (int, float)):
                if val < 0.3:
                    return 'background-color: #ffcccc'
                elif val < 0.6:
                    return 'background-color: #fff4cc'
                else:
                    return 'background-color: #ccffcc'
            return ''
        
        if 'confidence' in df_results.columns:
            styled_df = styled_df.applymap(highlight_confidence, subset=['confidence'])
        
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Statistics summary
        st.markdown("##### 📈 Prediction Statistics")
        
        if 'confidence' in df_results.columns:
            conf_stats = df_results['confidence'].describe()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean Confidence", f"{conf_stats['mean']:.3f}")
            with col2:
                st.metric("Min Confidence", f"{conf_stats['min']:.3f}")
            with col3:
                st.metric("Max Confidence", f"{conf_stats['max']:.3f}")
            with col4:
                st.metric("Std Dev", f"{conf_stats['std']:.3f}")
        
        # Export options
        st.markdown("##### 💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV export
            csv = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"predictions_E{energy_query:.1f}mJ_tau{duration_query:.1f}ns.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # JSON export
            json_str = df_results.to_json(orient='records', indent=2)
            st.download_button(
                label="📥 Download as JSON",
                data=json_str.encode('utf-8'),
                file_name=f"predictions_E{energy_query:.1f}mJ_tau{duration_query:.1f}ns.json",
                mime="application/json",
                use_container_width=True
            )

if __name__ == "__main__":
    main()
