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
import scipy.ndimage as ndimage
from scipy.stats import gaussian_kde
import json
import base64
from PIL import Image
import io

warnings.filterwarnings('ignore')

# =============================================
# PATH CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEA_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "fea_solutions")
os.makedirs(FEA_SOLUTIONS_DIR, exist_ok=True)

# =============================================
# ENHANCED MESH DATA STRUCTURES
# =============================================
class EnhancedMeshData:
    """Container for enhanced mesh data with spatial indexing"""
    
    def __init__(self):
        self.points = None
        self.triangles = None
        self.fields = {}
        self.field_info = {}
        self.metadata = {}
        self.spatial_tree = None
        self.region_labels = None
        self.mesh_stats = {}
        
    def compute_spatial_features(self):
        """Compute spatial features for the mesh"""
        if self.points is None:
            return
        
        # Compute bounding box
        min_coords = np.min(self.points, axis=0)
        max_coords = np.max(self.points, axis=0)
        self.mesh_stats['bbox'] = {'min': min_coords.tolist(), 'max': max_coords.tolist()}
        
        # Compute centroid
        self.mesh_stats['centroid'] = np.mean(self.points, axis=0).tolist()
        
        # Compute distances from centroid
        if self.points.shape[0] > 0:
            centroid = self.mesh_stats['centroid']
            distances = np.linalg.norm(self.points - centroid, axis=1)
            self.mesh_stats['radial_stats'] = {
                'min': float(np.min(distances)),
                'max': float(np.max(distances)),
                'mean': float(np.mean(distances)),
                'std': float(np.std(distances))
            }
        
        # Compute surface area if triangles exist
        if self.triangles is not None and len(self.triangles) > 0:
            areas = self.compute_triangle_areas()
            self.mesh_stats['surface_area'] = float(np.sum(areas))
            self.mesh_stats['triangle_count'] = len(self.triangles)
            self.mesh_stats['avg_triangle_area'] = float(np.mean(areas))
    
    def compute_triangle_areas(self):
        """Compute areas of triangles"""
        if self.triangles is None or self.points is None:
            return np.array([])
        
        areas = []
        for tri in self.triangles:
            if tri[0] < len(self.points) and tri[1] < len(self.points) and tri[2] < len(self.points):
                v0 = self.points[tri[0]]
                v1 = self.points[tri[1]]
                v2 = self.points[tri[2]]
                
                # Compute area using cross product
                v0v1 = v1 - v0
                v0v2 = v2 - v0
                area = 0.5 * np.linalg.norm(np.cross(v0v1, v0v2))
                areas.append(area)
        
        return np.array(areas)
    
    def segment_regions(self, n_regions=5):
        """Segment mesh into spatial regions"""
        if self.points is None:
            return
        
        # Simple region segmentation based on spatial clustering
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=n_regions, random_state=42, n_init=10)
        self.region_labels = kmeans.fit_predict(self.points)
        
        # Compute region statistics
        self.region_stats = {}
        for region_id in range(n_regions):
            region_mask = self.region_labels == region_id
            if np.any(region_mask):
                region_points = self.points[region_mask]
                self.region_stats[region_id] = {
                    'size': int(np.sum(region_mask)),
                    'centroid': np.mean(region_points, axis=0).tolist(),
                    'bbox': {
                        'min': np.min(region_points, axis=0).tolist(),
                        'max': np.max(region_points, axis=0).tolist()
                    }
                }

# =============================================
# ENHANCED DATA LOADER WITH FIELD MAPPING AND VALIDATION
# =============================================
class EnhancedFEADataLoader:
    """Enhanced data loader with field mapping, validation, and full mesh capabilities"""
    
    def __init__(self):
        self.simulations = {}
        self.summaries = []
        self.available_fields = set()
        self.reference_mesh = None
        self.common_fields = set()
        self.field_mapping = {}
        self.field_categories = {
            'thermal': ['temperature', 'temp', 'thermal_energy', 'heat_flux', 'thermal'],
            'mechanical': ['principal stress', 'vonmises_stress', 'displacement', 'strain', 'pressure'],
            'material': ['density', 'young_modulus', 'poissons_ratio', 'material_properties'],
            'energy': ['energy', 'power', 'energy_flux']
        }
        self.load_field_mappings()
    
    def load_field_mappings(self):
        """Load or create field mapping configuration"""
        mapping_file = os.path.join(SCRIPT_DIR, "field_mappings.json")
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r') as f:
                self.field_mapping = json.load(f)
        else:
            # Default mappings
            self.field_mapping = {
                "temperature": ["temp", "thermal", "temp_C", "temperature_C", "temp (C)", "temperature (C)"],
                "principal stress": ["stress", "vonmises_stress", "max_stress", "stress_eqv", "equivalent_stress"],
                "displacement": ["displacement_magnitude", "disp", "disp_mag", "total_displacement"],
                "strain": ["strain_eqv", "equivalent_strain", "plastic_strain"]
            }
            self.save_field_mappings()
    
    def save_field_mappings(self):
        """Save field mappings to file"""
        mapping_file = os.path.join(SCRIPT_DIR, "field_mappings.json")
        with open(mapping_file, 'w') as f:
            json.dump(self.field_mapping, f, indent=2)
    
    def get_canonical_field_name(self, raw_field_name):
        """Map raw field name to canonical name"""
        # Check if it's already a canonical name
        if raw_field_name in self.field_mapping:
            return raw_field_name
        
        # Check against variants
        raw_lower = raw_field_name.lower()
        for canonical, variants in self.field_mapping.items():
            if raw_lower in [v.lower() for v in variants]:
                return canonical
        
        # Try fuzzy matching for similar field names
        for canonical in self.field_mapping.keys():
            if canonical.lower() in raw_lower or raw_lower in canonical.lower():
                return canonical
        
        # Fallback: normalize the name
        normalized = re.sub(r'[^a-zA-Z0-9]', '_', raw_field_name.lower())
        return normalized
    
    def normalize_field_names(self, simulation_data):
        """Normalize field names in a simulation using mapping"""
        normalized_fields = {}
        field_info = {}
        canonical_field_map = {}
        
        for raw_name, data in simulation_data['mesh_data'].fields.items():
            canonical_name = self.get_canonical_field_name(raw_name)
            normalized_fields[canonical_name] = data
            if raw_name in simulation_data['mesh_data'].field_info:
                field_info[canonical_name] = simulation_data['mesh_data'].field_info[raw_name]
            
            # Track mapping for reference
            canonical_field_map[raw_name] = canonical_name
            self.available_fields.add(canonical_name)
        
        simulation_data['mesh_data'].fields = normalized_fields
        simulation_data['mesh_data'].field_info = field_info
        simulation_data['canonical_field_map'] = canonical_field_map
        
        return simulation_data
    
    def parse_folder_name(self, folder: str):
        """q0p5mJ-delta4p2ns → (0.5, 4.2)"""
        match = re.match(r"q([\dp\.]+)mJ-delta([\dp\.]+)ns", folder)
        if not match:
            return None, None
        e, d = match.groups()
        return float(e.replace("p", ".")), float(d.replace("p", "."))
    
    @st.cache_data
    def load_all_simulations(_self, load_full_mesh=True):
        """Load all simulations with enhanced mesh capabilities"""
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
                
                # Create enhanced mesh data structure
                mesh_data = EnhancedMeshData()
                
                # Basic simulation info
                sim_data = {
                    'name': name,
                    'energy_mJ': energy,
                    'duration_ns': duration,
                    'n_timesteps': len(vtu_files),
                    'vtu_files': vtu_files,
                    'field_info': {},
                    'has_mesh': False,
                    'mesh_data': mesh_data,
                    'metadata': {
                        'folder': folder,
                        'file_count': len(vtu_files)
                    }
                }
                
                # Load points
                mesh_data.points = mesh0.points.astype(np.float32)
                
                # Find triangles
                triangles = None
                for cell_block in mesh0.cells:
                    if cell_block.type == "triangle":
                        triangles = cell_block.data.astype(np.int32)
                        break
                
                mesh_data.triangles = triangles
                
                # Initialize fields
                mesh_data.fields = {}
                for key in mesh0.point_data.keys():
                    arr = mesh0.point_data[key].astype(np.float32)
                    if arr.ndim == 1:
                        sim_data['field_info'][key] = ("scalar", 1)
                        mesh_data.fields[key] = np.full((len(vtu_files), len(mesh_data.points)), 
                                                       np.nan, dtype=np.float32)
                    else:
                        sim_data['field_info'][key] = ("vector", arr.shape[1])
                        mesh_data.fields[key] = np.full((len(vtu_files), len(mesh_data.points), arr.shape[1]), 
                                                       np.nan, dtype=np.float32)
                    mesh_data.fields[key][0] = arr
                    _self.available_fields.add(key)
                    mesh_data.field_info[key] = sim_data['field_info'][key]
                
                # Load remaining timesteps
                for t in range(1, len(vtu_files)):
                    try:
                        mesh = meshio.read(vtu_files[t])
                        for key in sim_data['field_info']:
                            if key in mesh.point_data:
                                mesh_data.fields[key][t] = mesh.point_data[key].astype(np.float32)
                    except Exception as e:
                        st.warning(f"Error loading timestep {t} in {name}: {e}")
                
                # Normalize field names
                sim_data = _self.normalize_field_names(sim_data)
                
                # Compute mesh statistics
                mesh_data.compute_spatial_features()
                mesh_data.segment_regions(n_regions=5)
                
                sim_data['has_mesh'] = True
                simulations[name] = sim_data
                
                # Create summary statistics
                summary = _self.extract_summary_statistics(mesh_data, energy, duration, name)
                summaries.append(summary)
                
                # Set as reference mesh if not already set
                if _self.reference_mesh is None:
                    _self.reference_mesh = mesh_data
                
            except Exception as e:
                st.warning(f"Error loading {name}: {str(e)}")
                continue
            
            progress_bar.progress((folder_idx + 1) / len(folders))
        
        progress_bar.empty()
        status_text.empty()
        
        # Determine common fields across all simulations
        if simulations:
            field_counts = {}
            for sim in simulations.values():
                for field in sim['mesh_data'].field_info.keys():
                    field_counts[field] = field_counts.get(field, 0) + 1
            
            _self.common_fields = {field for field, count in field_counts.items() 
                                 if count == len(simulations)}
            
            st.success(f"✅ Loaded {len(simulations)} simulations with {len(_self.available_fields)} unique fields")
            
            if not _self.common_fields:
                st.warning("⚠️ No common fields found across all simulations. This will limit field comparison capabilities.")
            else:
                st.info(f"📊 {len(_self.common_fields)} common fields across all simulations")
            
            # Generate consistency report
            consistency_report = _self.validate_simulation_consistency(simulations)
            st.session_state.consistency_report = consistency_report
            
            # Show report if there are inconsistencies
            if consistency_report['recommendations']:
                with st.expander("⚠️ Data Consistency Issues", expanded=False):
                    _self.render_data_validation_report(consistency_report)
        else:
            st.error("❌ No simulations loaded successfully")
        
        return simulations, summaries
    
    def extract_summary_statistics(self, mesh_data, energy, duration, name):
        """Extract comprehensive summary statistics from mesh data"""
        summary = {
            'name': name,
            'energy': energy,
            'duration': duration,
            'timesteps': list(range(1, len(mesh_data.fields[list(mesh_data.fields.keys())[0]]) + 1)),
            'field_stats': {},
            'mesh_stats': mesh_data.mesh_stats,
            'region_stats': mesh_data.region_stats if hasattr(mesh_data, 'region_stats') else {}
        }
        
        for field_name in mesh_data.fields.keys():
            field_data = mesh_data.fields[field_name]
            summary['field_stats'][field_name] = {
                'min': [], 'max': [], 'mean': [], 'std': [],
                'q25': [], 'q50': [], 'q75': [], 'skew': [], 'kurtosis': []
            }
            
            for t in range(field_data.shape[0]):
                if field_data[t].ndim == 1:
                    values = field_data[t]
                else:
                    # Vector field - compute magnitude
                    values = np.linalg.norm(field_data[t], axis=1)
                
                # Remove NaN values
                valid_values = values[~np.isnan(values)]
                
                if len(valid_values) > 0:
                    summary['field_stats'][field_name]['min'].append(float(np.min(valid_values)))
                    summary['field_stats'][field_name]['max'].append(float(np.max(valid_values)))
                    summary['field_stats'][field_name]['mean'].append(float(np.mean(valid_values)))
                    summary['field_stats'][field_name]['std'].append(float(np.std(valid_values)))
                    summary['field_stats'][field_name]['q25'].append(float(np.percentile(valid_values, 25)))
                    summary['field_stats'][field_name]['q50'].append(float(np.percentile(valid_values, 50)))
                    summary['field_stats'][field_name]['q75'].append(float(np.percentile(valid_values, 75)))
                    
                    # Higher-order statistics
                    if len(valid_values) > 3:
                        from scipy.stats import skew, kurtosis
                        summary['field_stats'][field_name]['skew'].append(float(skew(valid_values)))
                        summary['field_stats'][field_name]['kurtosis'].append(float(kurtosis(valid_values)))
                    else:
                        summary['field_stats'][field_name]['skew'].append(0.0)
                        summary['field_stats'][field_name]['kurtosis'].append(0.0)
                else:
                    for stat in ['min', 'max', 'mean', 'std', 'q25', 'q50', 'q75', 'skew', 'kurtosis']:
                        summary['field_stats'][field_name][stat].append(0.0)
        
        return summary
    
    def validate_simulation_consistency(self, simulations):
        """Validate consistency across simulations and generate report"""
        report = {
            'field_name_inconsistencies': [],
            'mesh_inconsistencies': [],
            'timestep_inconsistencies': [],
            'recommendations': []
        }
        
        # Field name analysis
        field_names = {}
        for sim_name, sim in simulations.items():
            for field in sim['mesh_data'].field_info.keys():
                canonical = self.get_canonical_field_name(field)
                if canonical not in field_names:
                    field_names[canonical] = {'count': 0, 'variants': set(), 'simulations': []}
                field_names[canonical]['count'] += 1
                field_names[canonical]['variants'].add(field)
                field_names[canonical]['simulations'].append(sim_name)
        
        # Check for inconsistent field names
        for canonical, data in field_names.items():
            if len(data['variants']) > 1:
                report['field_name_inconsistencies'].append({
                    'canonical': canonical,
                    'variants': list(data['variants']),
                    'count': data['count'],
                    'simulations': data['simulations']
                })
                report['recommendations'].append(
                    f"Field '{canonical}' has {len(data['variants'])} naming variants. "
                    f"Consider standardizing: {', '.join(data['variants'])}"
                )
        
        # Mesh compatibility analysis
        mesh_stats = {}
        for sim_name, sim in simulations.items():
            if sim.get('has_mesh') and hasattr(sim.get('mesh_data'), 'mesh_stats'):
                stats = sim['mesh_data'].mesh_stats
                mesh_stats[sim_name] = {
                    'point_count': len(sim['mesh_data'].points) if sim['mesh_data'].points is not None else 0,
                    'triangle_count': len(sim['mesh_data'].triangles) if sim['mesh_data'].triangles is not None else 0,
                    'bbox': stats.get('bbox', {})
                }
        
        if mesh_stats:
            point_counts = [stats['point_count'] for stats in mesh_stats.values()]
            if max(point_counts) > min(point_counts) * 1.1:  # More than 10% difference
                report['mesh_inconsistencies'].append(f"Mesh sizes vary significantly: {min(point_counts)} to {max(point_counts)} points")
                report['recommendations'].append("Consider resampling meshes to a common resolution for better interpolation.")
        
        # Timestep analysis
        timesteps = {sim_name: sim['n_timesteps'] for sim_name, sim in simulations.items()}
        if len(set(timesteps.values())) > 1:
            report['timestep_inconsistencies'].append(f"Timestep counts vary: {min(timesteps.values())} to {max(timesteps.values())}")
            report['recommendations'].append("Normalize timesteps using interpolation for consistent time series prediction.")
        
        return report
    
    def render_data_validation_report(self, report):
        """Render data validation report in the UI"""
        st.markdown('<h3 class="sub-header">🔍 Data Consistency Report</h3>', unsafe_allow_html=True)
        
        if not report or not any(report.values()):
            st.success("✅ All simulations are consistent and ready for analysis")
            return
        
        tabs = st.tabs(["Field Names", "Mesh Geometry", "Time Steps", "Recommendations"])
        
        with tabs[0]:
            if report['field_name_inconsistencies']:
                st.warning("⚠️ Field name inconsistencies detected")
                for issue in report['field_name_inconsistencies']:
                    with st.expander(f"Field: {issue['canonical']} ({issue['count']} simulations)"):
                        st.write(f"**Different names used:** {', '.join(issue['variants'])}")
                        st.write(f"**Simulations:** {', '.join(issue['simulations'])}")
                        # Suggest mapping
                        if st.button(f"Map to '{issue['canonical']}'", key=f"map_{issue['canonical']}"):
                            for variant in issue['variants']:
                                if variant != issue['canonical']:
                                    if issue['canonical'] not in self.field_mapping:
                                        self.field_mapping[issue['canonical']] = []
                                    if variant not in self.field_mapping[issue['canonical']]:
                                        self.field_mapping[issue['canonical']].append(variant)
                            self.save_field_mappings()
                            st.success(f"Mapped {len(issue['variants'])-1} variants to '{issue['canonical']}'")
                            st.experimental_rerun()
            else:
                st.success("✅ All fields use consistent naming")
        
        with tabs[1]:
            if report['mesh_inconsistencies']:
                st.warning("⚠️ Mesh geometry inconsistencies detected")
                for issue in report['mesh_inconsistencies']:
                    st.write(f"- {issue}")
                if st.button("🔧 Resample Meshes to Common Resolution"):
                    st.info("Mesh resampling functionality would be implemented here")
            else:
                st.success("✅ All meshes have consistent geometry")
        
        with tabs[2]:
            if report['timestep_inconsistencies']:
                st.warning("⚠️ Timestep inconsistencies detected")
                for issue in report['timestep_inconsistencies']:
                    st.write(f"- {issue}")
                if st.button("🔧 Normalize Timesteps"):
                    st.info("Timestep normalization functionality would be implemented here")
            else:
                st.success("✅ All simulations have consistent timesteps")
        
        with tabs[3]:
            if report['recommendations']:
                st.markdown("### 🛠️ Recommendations")
                for i, rec in enumerate(report['recommendations'], 1):
                    st.markdown(f"{i}. {rec}")
                
                # One-click fixes
                if any("mapping" in rec.lower() for rec in report['recommendations']):
                    if st.button("✨ Auto-fix Field Name Mappings", type="primary"):
                        # Apply smart mappings
                        for issue in report['field_name_inconsistencies']:
                            canonical = issue['canonical']
                            for variant in issue['variants']:
                                if variant != canonical:
                                    if canonical not in self.field_mapping:
                                        self.field_mapping[canonical] = []
                                    if variant not in self.field_mapping[canonical]:
                                        self.field_mapping[canonical].append(variant)
                        self.save_field_mappings()
                        st.success("✅ Field name mappings updated")
                        st.experimental_rerun()
            else:
                st.success("✅ No recommendations needed")

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
            else:
                # Fill with zeros if prediction failed
                for field in results['field_predictions']:
                    results['field_predictions'][field]['mean'].append(0.0)
                    results['field_predictions'][field]['max'].append(0.0)
                    results['field_predictions'][field]['std'].append(0.0)
                    results['attention_maps'].append(np.array([]))
                    results['confidence_scores'].append(0.0)
        
        return results

# =============================================
# GEOMETRICAL FIELD PREDICTOR
# =============================================
class GeometricalFieldPredictor:
    """Predict field values on mesh geometry using attention-based interpolation"""
    
    def __init__(self, extrapolator, data_loader):
        self.extrapolator = extrapolator
        self.data_loader = data_loader
        self.rbf_interpolators = {}
        
    def predict_field_on_mesh(self, energy_query, duration_query, time_query, reference_mesh, field_name):
        """Predict field values on a reference mesh geometry"""
        if not self.extrapolator.fitted:
            return None
        
        # Get prediction statistics for the field
        stats_prediction = self.extrapolator.predict_field_statistics(energy_query, duration_query, time_query)
        
        if not stats_prediction or 'field_predictions' not in stats_prediction:
            return None
        
        if field_name not in stats_prediction['field_predictions']:
            st.warning(f"⚠️ Field '{field_name}' not available in predictions. Trying to find similar field...")
            
            # Try to find similar field
            similar_fields = []
            field_lower = field_name.lower()
            for available_field in stats_prediction['field_predictions'].keys():
                if field_lower in available_field.lower() or available_field.lower() in field_lower:
                    similar_fields.append(available_field)
            
            if similar_fields:
                field_name = similar_fields[0]
                st.info(f"✅ Using similar field: '{field_name}' instead")
            else:
                return None
        
        attention_weights = stats_prediction['attention_weights']
        target_stats = stats_prediction['field_predictions'][field_name]
        
        # Initialize mesh values
        mesh_points = reference_mesh.points
        n_points = len(mesh_points)
        predicted_values = np.zeros(n_points)
        
        # Collect source field distributions weighted by attention
        source_distributions = []
        source_weights = []
        source_metadata = []
        
        for i, (weight, meta) in enumerate(zip(attention_weights, self.extrapolator.source_metadata)):
            if weight > 0.001:  # Only consider significant sources
                sim_name = meta['name']
                timestep_idx = meta['timestep_idx']
                
                # Find the simulation
                if sim_name in self.data_loader.simulations:
                    sim = self.data_loader.simulations[sim_name]
                    if hasattr(sim['mesh_data'], 'fields') and field_name in sim['mesh_data'].fields:
                        field_data = sim['mesh_data'].fields[field_name]
                        
                        if timestep_idx < field_data.shape[0]:
                            source_values = field_data[timestep_idx]
                            
                            # Handle vector fields
                            if source_values.ndim == 2:
                                source_values = np.linalg.norm(source_values, axis=1)
                            
                            # Ensure same number of points
                            if len(source_values) == n_points:
                                source_distributions.append(source_values)
                                source_weights.append(weight)
                                source_metadata.append({
                                    'sim_name': sim_name,
                                    'timestep': meta['time'],
                                    'energy': meta['energy'],
                                    'duration': meta['duration']
                                })
        
        if not source_distributions:
            # Fallback: create synthetic distribution based on statistics
            mean_val = target_stats['mean']
            std_val = target_stats['std']
            
            # Create spatial variation pattern
            centroid = np.mean(mesh_points, axis=0)
            distances = np.linalg.norm(mesh_points - centroid, axis=1)
            distances_norm = distances / np.max(distances)
            
            # Gaussian radial pattern
            spatial_pattern = np.exp(-distances_norm ** 2 / 0.3)
            
            # Scale to match target statistics
            spatial_pattern = (spatial_pattern - np.mean(spatial_pattern)) / np.std(spatial_pattern)
            predicted_values = spatial_pattern * std_val + mean_val
            
            return {
                'values': predicted_values,
                'method': 'synthetic',
                'confidence': 0.0
            }
        
        # Blend distributions based on attention weights
        source_weights = np.array(source_weights)
        source_weights = source_weights / np.sum(source_weights)
        
        # Weighted average of source distributions
        blended_distribution = np.zeros(n_points)
        for dist, weight in zip(source_distributions, source_weights):
            blended_distribution += dist * weight
        
        # Scale blended distribution to match predicted statistics
        current_mean = np.mean(blended_distribution)
        current_std = np.std(blended_distribution)
        
        target_mean = target_stats['mean']
        target_std = target_stats['std']
        
        if current_std > 1e-6:
            # Scale and shift to match target statistics
            scaled_distribution = (blended_distribution - current_mean) / current_std * target_std + target_mean
        else:
            # Constant distribution with target mean
            scaled_distribution = np.full(n_points, target_mean)
        
        # Add spatial coherence using RBF interpolation of residuals
        residuals = scaled_distribution - blended_distribution
        
        if len(source_distributions) > 3 and np.std(residuals) > 1e-6:
            # Sample points for RBF
            n_samples = min(50, len(mesh_points))
            sample_indices = np.random.choice(len(mesh_points), n_samples, replace=False)
            
            try:
                rbf = RBFInterpolator(
                    mesh_points[sample_indices],
                    residuals[sample_indices],
                    kernel='thin_plate_spline',
                    epsilon=0.1
                )
                
                # Interpolate residuals to all points
                interpolated_residuals = rbf(mesh_points)
                
                # Apply residuals with smoothing
                scaled_distribution += interpolated_residuals * 0.5
            except Exception as e:
                st.warning(f"RBF interpolation failed: {str(e)}. Using basic interpolation.")
        
        # Ensure physical constraints (e.g., non-negative for some fields)
        if field_name.lower() in ['temperature', 'stress', 'displacement', 'strain', 'heat_flux']:
            scaled_distribution = np.maximum(scaled_distribution, 0)
        
        # Compute confidence based on attention weight distribution
        confidence = float(np.mean(source_weights))
        
        return {
            'values': scaled_distribution,
            'method': 'attention_blend',
            'confidence': confidence,
            'n_sources': len(source_distributions),
            'source_stats': {
                'min': float(np.min(scaled_distribution)),
                'max': float(np.max(scaled_distribution)),
                'mean': float(np.mean(scaled_distribution)),
                'std': float(np.std(scaled_distribution))
            }
        }
    
    def predict_field_evolution(self, energy_query, duration_query, time_points, reference_mesh, field_name):
        """Predict field evolution over time on mesh"""
        predictions = []
        confidences = []
        
        for t in time_points:
            pred = self.predict_field_on_mesh(energy_query, duration_query, t, reference_mesh, field_name)
            if pred:
                predictions.append(pred['values'])
                confidences.append(pred['confidence'])
            else:
                predictions.append(None)
                confidences.append(0.0)
        
        return predictions, confidences
    
    def compute_spatial_correlations(self, field_values, reference_mesh):
        """Compute spatial correlation structure of predicted field"""
        if field_values is None or reference_mesh.points is None:
            return None
        
        points = reference_mesh.points
        values = field_values
        
        # Compute distance matrix (sampled for efficiency)
        n_samples = min(100, len(points))
        sample_indices = np.random.choice(len(points), n_samples, replace=False)
        sample_points = points[sample_indices]
        sample_values = values[sample_indices]
        
        # Compute pairwise distances and value differences
        distances = []
        value_diffs = []
        
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                dist = np.linalg.norm(sample_points[i] - sample_points[j])
                val_diff = abs(sample_values[i] - sample_values[j])
                
                distances.append(dist)
                value_diffs.append(val_diff)
        
        if len(distances) > 10:
            # Compute correlation
            from scipy.stats import pearsonr
            try:
                corr, _ = pearsonr(distances, value_diffs)
                return float(corr)
            except:
                return 0.0
        
        return 0.0

# =============================================
# ENHANCED VISUALIZER WITH GEOMETRICAL FEATURES
# =============================================
class EnhancedGeometricalVisualizer:
    """Visualization components with advanced geometrical features"""
    
    EXTENDED_COLORMAPS = [
        'Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'Rainbow',
        'Jet', 'Hot', 'Cool', 'Portland', 'Bluered', 'Electric',
        'Thermal', 'Balance', 'Brwnyl', 'Darkmint', 'Emrld', 'Mint',
        'Oranges', 'Purp', 'Purples', 'Sunset', 'Sunsetdark', 'Teal',
        'Tealgrn', 'Twilight', 'Burg', 'Burgyl', 'RdYlBu', 'RdYlGn'
    ]
    
    @staticmethod
    def create_mesh_field_visualization(mesh_data, field_name, field_values, 
                                       colormap="Viridis", title="", opacity=0.9,
                                       show_wireframe=True, show_points=False):
        """Create interactive 3D mesh visualization of field values"""
        if mesh_data is None or field_values is None:
            return go.Figure()
        
        pts = mesh_data.points
        triangles = mesh_data.triangles
        
        fig = go.Figure()
        
        # Determine if we have triangles for surface rendering
        has_triangles = triangles is not None and len(triangles) > 0
        
        if has_triangles and not show_points:
            # Surface mesh visualization
            fig.add_trace(go.Mesh3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
                intensity=field_values,
                colorscale=colormap,
                intensitymode='vertex',
                colorbar=dict(
                    title=dict(text=field_name, font=dict(size=14)),
                    thickness=20,
                    len=0.75
                ),
                opacity=opacity,
                lighting=dict(
                    ambient=0.8,
                    diffuse=0.8,
                    specular=0.5,
                    roughness=0.5
                ),
                hovertemplate=(
                    '<b>Location:</b> (%{x:.3f}, %{y:.3f}, %{z:.3f})<br>'
                    f'<b>{field_name}:</b> %{{intensity:.3f}}<br>'
                    '<extra></extra>'
                ),
                name="Field Distribution",
                showlegend=True
            ))
            
            # Add wireframe for better geometry perception
            if show_wireframe:
                edges = set()
                for tri in triangles[:min(1000, len(triangles))]:  # Limit for performance
                    for i in range(3):
                        edge = tuple(sorted((tri[i], tri[(i+1)%3])))
                        edges.add(edge)
                
                edge_x = []
                edge_y = []
                edge_z = []
                for edge in list(edges)[:500]:  # Limit edges for performance
                    if edge[0] < len(pts) and edge[1] < len(pts):
                        edge_x.extend([pts[edge[0], 0], pts[edge[1], 0], None])
                        edge_y.extend([pts[edge[0], 1], pts[edge[1], 1], None])
                        edge_z.extend([pts[edge[0], 2], pts[edge[1], 2], None])
                
                fig.add_trace(go.Scatter3d(
                    x=edge_x, y=edge_y, z=edge_z,
                    mode='lines',
                    line=dict(color='rgba(0, 0, 0, 0.3)', width=1),
                    opacity=0.2,
                    hoverinfo='none',
                    name="Mesh Wireframe",
                    showlegend=True
                ))
        else:
            # Point cloud visualization
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode='markers',
                marker=dict(
                    size=3 if show_points else 2,
                    color=field_values,
                    colorscale=colormap,
                    opacity=opacity,
                    colorbar=dict(
                        title=dict(text=field_name, font=dict(size=14)),
                        thickness=20,
                        len=0.75
                    ),
                    showscale=True,
                    line=dict(width=0)
                ),
                hovertemplate=(
                    '<b>Location:</b> (%{x:.3f}, %{y:.3f}, %{z:.3f})<br>'
                    f'<b>{field_name}:</b> %{{marker.color:.3f}}<br>'
                    '<extra></extra>'
                ),
                name="Field Points",
                showlegend=True
            ))
        
        # Add bounding box for context
        if pts.shape[0] > 0:
            min_coords = np.min(pts, axis=0)
            max_coords = np.max(pts, axis=0)
            
            # Create bounding box lines
            bbox_lines = []
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        for l in range(2):
                            if i != l and j == k:
                                x = [min_coords[0] + i*(max_coords[0]-min_coords[0]),
                                     min_coords[0] + l*(max_coords[0]-min_coords[0])]
                                y = [min_coords[1] + j*(max_coords[1]-min_coords[1]),
                                     min_coords[1] + j*(max_coords[1]-min_coords[1])]
                                z = [min_coords[2] + k*(max_coords[2]-min_coords[2]),
                                     min_coords[2] + k*(max_coords[2]-min_coords[2])]
                                bbox_lines.append((x, y, z))
            
            for line in bbox_lines:
                fig.add_trace(go.Scatter3d(
                    x=line[0], y=line[1], z=line[2],
                    mode='lines',
                    line=dict(color='rgba(100, 100, 100, 0.2)', width=1),
                    hoverinfo='none',
                    showlegend=False
                ))
        
        # Add coordinate axes
        if pts.shape[0] > 0:
            centroid = np.mean(pts, axis=0)
            axis_length = np.max(pts, axis=0) - np.min(pts, axis=0)
            max_length = np.max(axis_length)
            
            # X axis (red)
            fig.add_trace(go.Scatter3d(
                x=[centroid[0], centroid[0] + max_length*0.2],
                y=[centroid[1], centroid[1]],
                z=[centroid[2], centroid[2]],
                mode='lines+text',
                line=dict(color='red', width=4),
                text=['', 'X'],
                textposition="top center",
                hoverinfo='none',
                showlegend=False
            ))
            
            # Y axis (green)
            fig.add_trace(go.Scatter3d(
                x=[centroid[0], centroid[0]],
                y=[centroid[1], centroid[1] + max_length*0.2],
                z=[centroid[2], centroid[2]],
                mode='lines+text',
                line=dict(color='green', width=4),
                text=['', 'Y'],
                textposition="top center",
                hoverinfo='none',
                showlegend=False
            ))
            
            # Z axis (blue)
            fig.add_trace(go.Scatter3d(
                x=[centroid[0], centroid[0]],
                y=[centroid[1], centroid[1]],
                z=[centroid[2], centroid[2] + max_length*0.2],
                mode='lines+text',
                line=dict(color='blue', width=4),
                text=['', 'Z'],
                textposition="top center",
                hoverinfo='none',
                showlegend=False
            ))
        
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=20, family="Arial, sans-serif")
            ),
            scene=dict(
                aspectmode="data",
                xaxis=dict(
                    title="X",
                    gridcolor="lightgray",
                    showbackground=True,
                    backgroundcolor="white",
                    showspikes=False
                ),
                yaxis=dict(
                    title="Y",
                    gridcolor="lightgray",
                    showbackground=True,
                    backgroundcolor="white",
                    showspikes=False
                ),
                zaxis=dict(
                    title="Z",
                    gridcolor="lightgray",
                    showbackground=True,
                    backgroundcolor="white",
                    showspikes=False
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    up=dict(x=0, y=0, z=1)
                )
            ),
            height=700,
            margin=dict(l=0, r=0, t=60, b=0),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    @staticmethod
    def create_cross_section_view(mesh_data, field_values, slice_axis, slice_position, 
                                 field_name, colormap="Viridis"):
        """Create 2D cross-section visualization"""
        if mesh_data is None or field_values is None:
            return go.Figure()
        
        pts = mesh_data.points
        values = field_values
        
        # Determine slice axis
        if slice_axis == 'X':
            slice_coord = pts[:, 0]
            x_axis = pts[:, 1]
            y_axis = pts[:, 2]
            axis_label = f"X = {slice_position:.3f}"
            x_label = "Y"
            y_label = "Z"
        elif slice_axis == 'Y':
            slice_coord = pts[:, 1]
            x_axis = pts[:, 0]
            y_axis = pts[:, 2]
            axis_label = f"Y = {slice_position:.3f}"
            x_label = "X"
            y_label = "Z"
        else:  # 'Z'
            slice_coord = pts[:, 2]
            x_axis = pts[:, 0]
            y_axis = pts[:, 1]
            axis_label = f"Z = {slice_position:.3f}"
            x_label = "X"
            y_label = "Y"
        
        # Find points within slice tolerance
        tolerance = 0.01 * (np.max(slice_coord) - np.min(slice_coord))
        slice_mask = np.abs(slice_coord - slice_position) < tolerance
        
        if np.sum(slice_mask) < 10:
            # If not enough points, interpolate
            from scipy.interpolate import griddata
            
            # Create grid for interpolation
            x_min, x_max = np.min(x_axis), np.max(x_axis)
            y_min, y_max = np.min(y_axis), np.max(y_axis)
            
            grid_x, grid_y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
            
            # Interpolate values
            try:
                grid_z = griddata(
                    (x_axis, y_axis), 
                    values, 
                    (grid_x, grid_y), 
                    method='cubic',
                    fill_value=np.nanmean(values)
                )
                
                # Create heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=grid_z.T,
                    x=grid_x[:, 0],
                    y=grid_y[0, :],
                    colorscale=colormap,
                    colorbar=dict(title=field_name)
                ))
            except Exception as e:
                st.warning(f"Interpolation failed: {str(e)}. Using scatter plot instead.")
                
                # Fallback to scatter plot
                fig = go.Figure(data=go.Scatter(
                    x=x_axis,
                    y=y_axis,
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=values,
                        colorscale=colormap,
                        showscale=True,
                        colorbar=dict(title=field_name)
                    ),
                    hovertemplate=f'{field_name}: %{{marker.color:.3f}}<br>{x_label}: %{{x:.3f}}<br>{y_label}: %{{y:.3f}}<extra></extra>'
                ))
        else:
            # Scatter plot of slice points
            slice_x = x_axis[slice_mask]
            slice_y = y_axis[slice_mask]
            slice_values = values[slice_mask]
            
            fig = go.Figure(data=go.Scatter(
                x=slice_x,
                y=slice_y,
                mode='markers',
                marker=dict(
                    size=8,
                    color=slice_values,
                    colorscale=colormap,
                    showscale=True,
                    colorbar=dict(title=field_name)
                ),
                hovertemplate=f'{field_name}: %{{marker.color:.3f}}<br>{x_label}: %{{x:.3f}}<br>{y_label}: %{{y:.3f}}<extra></extra>'
            ))
        
        fig.update_layout(
            title=f"{field_name} Cross-Section at {axis_label}",
            xaxis_title=x_label,
            yaxis_title=y_label,
            height=500,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig
    
    @staticmethod
    def create_attention_visualization_on_mesh(mesh_data, attention_weights, 
                                              source_metadata, query_point=None):
        """Visualize attention weights distribution on mesh geometry"""
        if mesh_data is None or attention_weights is None:
            return go.Figure()
        
        pts = mesh_data.points
        
        # Aggregate attention by simulation
        sim_attention = {}
        for weight, meta in zip(attention_weights, source_metadata):
            sim_name = meta['name']
            sim_attention[sim_name] = sim_attention.get(sim_name, 0) + weight
        
        # Create attention influence map
        attention_map = np.zeros(len(pts))
        
        if len(sim_attention) > 0:
            # Find source simulations in loaded data
            for sim_name, weight in sim_attention.items():
                if sim_name in st.session_state.data_loader.simulations:
                    sim = st.session_state.data_loader.simulations[sim_name]
                    sim_points = sim['mesh_data'].points
                    
                    # Find nearest neighbor influence
                    if len(sim_points) > 0:
                        from scipy.spatial import cKDTree
                        tree = cKDTree(sim_points)
                        
                        # For each point in reference mesh, find distance to nearest source point
                        distances, indices = tree.query(pts, k=1)
                        
                        # Apply Gaussian influence based on distance
                        influence = weight * np.exp(-distances**2 / (2 * 0.1**2))
                        attention_map += influence
        
        # Normalize attention map
        if np.max(attention_map) > 0:
            attention_map = attention_map / np.max(attention_map)
        
        # Create visualization
        fig = go.Figure()
        
        if mesh_data.triangles is not None and len(mesh_data.triangles) > 0:
            fig.add_trace(go.Mesh3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                i=mesh_data.triangles[:, 0],
                j=mesh_data.triangles[:, 1],
                k=mesh_data.triangles[:, 2],
                intensity=attention_map,
                colorscale="RdYlBu_r",  # Reverse for red = high attention
                intensitymode='vertex',
                colorbar=dict(title="Attention Influence"),
                opacity=0.8,
                hovertemplate='Attention: %{intensity:.3f}<br>Location: (%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>'
            ))
        else:
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=attention_map,
                    colorscale="RdYlBu_r",
                    opacity=0.8,
                    colorbar=dict(title="Attention Influence")
                ),
                hovertemplate='Attention: %{marker.color:.3f}<br>Location: (%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>'
            ))
        
        # Add query point if provided
        if query_point is not None:
            fig.add_trace(go.Scatter3d(
                x=[query_point[0]], y=[query_point[1]], z=[query_point[2]],
                mode='markers',
                marker=dict(
                    size=10,
                    color='yellow',
                    symbol='diamond',
                    line=dict(color='black', width=2)
                ),
                name='Query Point',
                hovertemplate='Query Point<br>Location: (%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>'
            ))
        
        fig.update_layout(
            title="Attention Influence on Mesh Geometry",
            scene=dict(
                aspectmode="data",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=600,
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def create_field_animation(field_evolution, mesh_data, time_points, field_name, colormap="Viridis"):
        """Create animation of field evolution over time"""
        if not field_evolution or mesh_data is None:
            return go.Figure()
        
        frames = []
        
        for i, (values, t) in enumerate(zip(field_evolution, time_points)):
            if values is None:
                continue
            
            frame = go.Frame(
                data=[go.Mesh3d(
                    x=mesh_data.points[:, 0],
                    y=mesh_data.points[:, 1],
                    z=mesh_data.points[:, 2],
                    i=mesh_data.triangles[:, 0] if mesh_data.triangles is not None else [],
                    j=mesh_data.triangles[:, 1] if mesh_data.triangles is not None else [],
                    k=mesh_data.triangles[:, 2] if mesh_data.triangles is not None else [],
                    intensity=values,
                    colorscale=colormap,
                    intensitymode='vertex',
                    opacity=0.9
                )],
                name=f"t={t:.1f}",
                traces=[0]
            )
            frames.append(frame)
        
        if not frames:
            return go.Figure()
        
        # Create initial frame
        fig = go.Figure(
            data=frames[0].data,
            frames=frames
        )
        
        # Add play button and slider
        fig.update_layout(
            title=f"{field_name} Evolution Animation",
            updatemenus=[{
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "▶️ Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 200, "redraw": True}, "fromcurrent": True}]
                    },
                    {
                        "label": "⏸️ Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]
                    }
                ],
                "x": 0.1,
                "y": 0,
                "xanchor": "right",
                "yanchor": "top"
            }],
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "prefix": "Time: ",
                    "suffix": " ns",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 300, "easing": "cubic-in-out"},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [[f"t={t:.1f}"], {"frame": {"duration": 300, "redraw": True}, "mode": "immediate"}],
                        "label": f"{t:.1f}",
                        "method": "animate"
                    }
                    for t in time_points
                ]
            }],
            height=600
        )
        
        return fig

# =============================================
# MAIN APPLICATION WITH GEOMETRICAL VISUALIZATION
# =============================================
class FEAVisualizationPlatform:
    """Main application class integrating all components"""
    
    def __init__(self):
        self.data_loader = EnhancedFEADataLoader()
        self.visualizer = EnhancedGeometricalVisualizer()
        self.extrapolator = EnhancedPhysicsInformedAttentionExtrapolator()
        self.geom_predictor = None
        
        # Session state initialization
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'selected_colormap' not in st.session_state:
            st.session_state.selected_colormap = "Viridis"
        if 'current_mode' not in st.session_state:
            st.session_state.current_mode = "Data Viewer"
        if 'consistency_report' not in st.session_state:
            st.session_state.consistency_report = {}
    
    def run(self):
        """Main application entry point"""
        st.set_page_config(
            page_title="Enhanced FEA Laser Simulation Platform",
            layout="wide",
            initial_sidebar_state="expanded",
            page_icon="🔬"
        )
        
        # Apply custom CSS
        self.apply_custom_css()
        
        # Render header
        self.render_header()
        
        # Initialize session state
        self.initialize_session_state()
        
        # Render sidebar
        self.render_sidebar()
        
        # Render main content based on mode
        self.render_main_content()
    
    def apply_custom_css(self):
        """Apply custom CSS styling"""
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
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
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
        .geometry-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
            border-left: 4px solid #1E88E5;
        }
        .field-mapping-card {
            background: linear-gradient(135deg, #8e44ad 0%, #3498db 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
            border-left: 4px solid #e74c3c;
        }
        .field-category {
            background: linear-gradient(135deg, #2ecc71, #1abc9c);
            padding: 0.5rem;
            border-radius: 5px;
            margin: 0.5rem 0;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render_header(self):
        """Render application header"""
        st.markdown('<h1 class="main-header">🔬 Advanced FEA Laser Simulation Platform</h1>', 
                   unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Geometrical Mesh Visualization with Physics-Informed Attention</p>',
                   unsafe_allow_html=True)
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'data_loader' not in st.session_state:
            st.session_state.data_loader = self.data_loader
        if 'visualizer' not in st.session_state:
            st.session_state.visualizer = self.visualizer
        if 'extrapolator' not in st.session_state:
            st.session_state.extrapolator = self.extrapolator
    
    def render_sidebar(self):
        """Render application sidebar"""
        with st.sidebar:
            st.markdown("### ⚙️ Navigation")
            
            # Mode selection
            app_mode = st.radio(
                "Select Mode",
                ["Data Viewer", "Interpolation/Extrapolation", "Comparative Analysis", "Geometrical Visualization"],
                index=["Data Viewer", "Interpolation/Extrapolation", "Comparative Analysis", "Geometrical Visualization"].index(
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
                    EnhancedGeometricalVisualizer.EXTENDED_COLORMAPS,
                    index=0
                )
            
            if st.button("🔄 Load All Simulations", type="primary", use_container_width=True):
                with st.spinner("Loading simulation data..."):
                    simulations, summaries = self.data_loader.load_all_simulations(
                        load_full_mesh=load_full_data
                    )
                    
                    if simulations and summaries:
                        st.session_state.simulations = simulations
                        st.session_state.summaries = summaries
                        st.session_state.data_loaded = True
                        
                        # Initialize extrapolator with summaries
                        self.extrapolator.load_summaries(summaries)
                        st.session_state.extrapolator_ready = True
                        
                        # Create geometric predictor
                        self.geom_predictor = GeometricalFieldPredictor(
                            self.extrapolator,
                            self.data_loader
                        )
                        
                        st.success("✅ Data loaded successfully!")
                        
                        # Display data statistics
                        with st.expander("📊 Data Statistics", expanded=True):
                            st.metric("Simulations", len(simulations))
                            st.metric("Available Fields", len(self.data_loader.available_fields))
                            st.metric("Common Fields", len(self.data_loader.common_fields))
                            
                            if summaries:
                                energies = [s['energy'] for s in summaries]
                                durations = [s['duration'] for s in summaries]
                                st.metric("Energy Range", f"{min(energies):.1f} - {max(energies):.1f} mJ")
                                st.metric("Duration Range", f"{min(durations):.1f} - {max(durations):.1f} ns")
                    else:
                        st.error("❌ Failed to load data")
            
            if st.session_state.get('data_loaded', False):
                st.markdown("---")
                st.markdown("### 🛠️ Advanced Settings")
                
                with st.expander("⚙️ Field Mapping Configuration", expanded=False):
                    self.render_field_mapping_interface()
                
                with st.expander("🎨 Visualization Settings", expanded=False):
                    st.checkbox("Show Wireframe", value=True, key="show_wireframe")
                    st.checkbox("Show Points", value=False, key="show_points")
                    st.slider("Mesh Opacity", 0.1, 1.0, 0.9, 0.1, key="mesh_opacity")
    
    def render_field_mapping_interface(self):
        """Render field mapping configuration interface"""
        if not self.data_loader.field_mapping:
            st.info("No field mappings configured yet. System will auto-detect common patterns.")
        
        # Show current mappings
        with st.expander("Current Field Mappings", expanded=True):
            for canonical, variants in self.data_loader.field_mapping.items():
                st.markdown(f"**{canonical}** ← {', '.join(variants)}")
        
        # Add new mapping
        col1, col2, col3 = st.columns([3, 3, 1])
        with col1:
            canonical_name = st.text_input("Canonical Field Name", key="canonical_field")
        with col2:
            variant_name = st.text_input("Variant Name", key="variant_field")
        with col3:
            if st.button("➕ Add Mapping", key="add_mapping"):
                if canonical_name and variant_name:
                    if canonical_name not in self.data_loader.field_mapping:
                        self.data_loader.field_mapping[canonical_name] = []
                    if variant_name not in self.data_loader.field_mapping[canonical_name]:
                        self.data_loader.field_mapping[canonical_name].append(variant_name)
                        self.data_loader.save_field_mappings()
                        st.success(f"Added mapping: {variant_name} → {canonical_name}")
        
        # Manual field renaming - FIXED VERSION
        if st.session_state.data_loaded:
            with st.expander("Manual Field Renaming"):
                if 'simulations' in st.session_state and st.session_state.simulations:
                    sim_name = st.selectbox("Select Simulation", sorted(st.session_state.simulations.keys()),
                                           key="rename_sim")
                    if sim_name:
                        sim = st.session_state.simulations[sim_name]
                        # FIXED: Use hasattr() to check for attributes on EnhancedMeshData object
                        if hasattr(sim['mesh_data'], 'field_info') and sim['mesh_data'].field_info:
                            field_name = st.selectbox("Field to Rename", sorted(sim['mesh_data'].field_info.keys()),
                                                    key="rename_field")
                            new_name = st.text_input("New Name", value=field_name, key="new_field_name")
                            if st.button("Rename Field"):
                                if hasattr(sim['mesh_data'], 'fields') and field_name in sim['mesh_data'].fields and field_name in sim['mesh_data'].field_info:
                                    # FIXED: Access fields and field_info as attributes of EnhancedMeshData
                                    sim['mesh_data'].fields[new_name] = sim['mesh_data'].fields.pop(field_name)
                                    sim['mesh_data'].field_info[new_name] = sim['mesh_data'].field_info.pop(field_name)
                                    st.success(f"Renamed field from '{field_name}' to '{new_name}'")
                                    # Rebuild common fields
                                    self.rebuild_common_fields()
    
    def rebuild_common_fields(self):
        """Rebuild common fields list after manual changes"""
        if 'simulations' in st.session_state and st.session_state.simulations:
            field_counts = {}
            for sim in st.session_state.simulations.values():
                # FIXED: Access field_info as attribute of EnhancedMeshData
                for field in sim['mesh_data'].field_info.keys():
                    field_counts[field] = field_counts.get(field, 0) + 1
            self.data_loader.common_fields = {field for field, count in field_counts.items()
                                            if count == len(st.session_state.simulations)}
            st.success("✅ Common fields rebuilt successfully")
    
    def render_main_content(self):
        """Render main content based on selected mode"""
        app_mode = st.session_state.current_mode
        
        if app_mode == "Data Viewer":
            self.render_data_viewer()
        elif app_mode == "Interpolation/Extrapolation":
            self.render_interpolation_extrapolation()
        elif app_mode == "Comparative Analysis":
            self.render_comparative_analysis()
        elif app_mode == "Geometrical Visualization":
            self.render_geometrical_visualization()
    
    def render_field_category_selector(self, available_fields):
        """Render field category selector UI"""
        # Define field categories based on available fields
        categories = {
            'Thermal Fields': [],
            'Mechanical Fields': [],
            'Material Properties': [],
            'Energy Fields': [],
            'Other Fields': []
        }
        
        # Categorize fields
        for field in available_fields:
            field_lower = field.lower()
            if any(kw in field_lower for kw in ['temp', 'heat', 'thermal', 'energy', 'flux']):
                categories['Thermal Fields'].append(field)
            elif any(kw in field_lower for kw in ['stress', 'strain', 'displacement', 'force', 'pressure']):
                categories['Mechanical Fields'].append(field)
            elif any(kw in field_lower for kw in ['density', 'modulus', 'ratio', 'property', 'material']):
                categories['Material Properties'].append(field)
            elif any(kw in field_lower for kw in ['energy', 'power', 'work']):
                categories['Energy Fields'].append(field)
            else:
                categories['Other Fields'].append(field)
        
        # Create UI
        category = st.selectbox(
            "Field Category",
            list(categories.keys()),
            format_func=lambda x: f"{x} ({len(categories[x])})" if categories[x] else f"{x} (none)",
            key="field_category"
        )
        
        # Show fields in selected category
        fields_in_category = categories[category]
        if fields_in_category:
            selected_field = st.selectbox(
                "Select Field",
                fields_in_category,
                key=f"field_in_{category.replace(' ', '_')}"
            )
            return selected_field
        else:
            st.info(f"No fields available in category '{category}'")
            return None
    
    def render_interpolation_extrapolation(self):
        """Render interpolation/extrapolation interface with field mapping support"""
        st.markdown('<h2 class="sub-header">🔮 Interpolation/Extrapolation Engine</h2>', 
                   unsafe_allow_html=True)
        
        if not st.session_state.get('data_loaded', False):
            self.show_data_not_loaded_warning()
            return
        
        st.markdown("""
        <div class="info-box">
        <h3>🧠 Physics-Informed Attention Mechanism</h3>
        <p>This engine uses a <strong>transformer-inspired multi-head attention mechanism</strong> with <strong>spatial locality regulation</strong> to interpolate and extrapolate simulation results. The model learns from existing FEA simulations and can predict outcomes for new parameter combinations with quantified confidence.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display loaded simulations summary
        with st.expander("📋 Loaded Simulations Summary", expanded=True):
            if st.session_state.summaries:
                df_summary = pd.DataFrame([{
                    'Simulation': s['name'],
                    'Energy (mJ)': s['energy'],
                    'Duration (ns)': s['duration'],
                    'Timesteps': len(s['timesteps']),
                    'Fields': ', '.join(sorted(s['field_stats'].keys())[:3]) + ('...' if len(s['field_stats']) > 3 else '')
                } for s in st.session_state.summaries])
                
                st.dataframe(
                    df_summary.style.format({
                        'Energy (mJ)': '{:.2f}',
                        'Duration (ns)': '{:.2f}'
                    }).background_gradient(subset=['Energy (mJ)', 'Duration (ns)'], cmap='Blues'),
                    use_container_width=True,
                    height=300
                )
        
        # Query parameters
        st.markdown('<h3 class="sub-header">🎯 Query Parameters</h3>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            # Get parameter ranges from loaded data
            if st.session_state.summaries:
                energies = [s['energy'] for s in st.session_state.summaries]
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
        
        # Field selection with fallbacks
        st.markdown('<h3 class="sub-header">📊 Field Selection</h3>', unsafe_allow_html=True)
        
        # Initialize selected_field with default value
        selected_field = None
        
        # Flexible field selection with options
        use_common_fields = st.checkbox("Use common fields only", value=True,
                                       help="Uncheck to select fields from specific simulations")
        
        if use_common_fields:
            if self.data_loader.common_fields:
                selected_field = self.render_field_category_selector(self.data_loader.common_fields)
            else:
                st.warning("⚠️ No common fields found across all simulations. Try unchecking 'Use common fields only' to select fields from specific simulations.")
        else:
            # Per-simulation field selection
            col1, col2 = st.columns([3, 2])
            with col1:
                reference_sim = st.selectbox(
                    "Select Reference Simulation",
                    sorted(st.session_state.simulations.keys()),
                    key="ref_sim_select"
                )
            if reference_sim:
                sim = st.session_state.simulations[reference_sim]
                available_fields = sorted(sim['mesh_data'].field_info.keys())
                with col2:
                    selected_field = self.render_field_category_selector(available_fields)
        
        # Visualization options
        st.markdown('<h3 class="sub-header">🎨 Visualization Options</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if use_common_fields:
                viz_reference = st.selectbox(
                    "Reference Geometry",
                    sorted(st.session_state.simulations.keys()),
                    key="ref_geom_select"
                )
            else:
                viz_reference = reference_sim  # Use the same simulation as reference
        with col2:
            viz_type = st.selectbox(
                "Visualization Type",
                ["3D Mesh", "Cross-Section", "Animation"],
                key="viz_type_select"
            )
        
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
        self.extrapolator.sigma_param = sigma_param
        self.extrapolator.spatial_weight = spatial_weight
        self.extrapolator.n_heads = n_heads
        self.extrapolator.temperature = temperature
        
        # Run prediction
        if st.button("🚀 Generate Prediction", type="primary", use_container_width=True):
            if selected_field is None:
                st.error("❌ Unable to generate prediction: No field selected. Please ensure simulations have common fields or select a specific simulation.")
                return
            
            with st.spinner("Generating physics-informed prediction..."):
                # Get reference mesh
                reference_sim = st.session_state.simulations[viz_reference]
                reference_mesh = reference_sim['mesh_data']
                
                # Generate predictions using the geometric predictor
                if self.geom_predictor is None:
                    self.geom_predictor = GeometricalFieldPredictor(
                        self.extrapolator,
                        self.data_loader
                    )
                
                predictions, confidences = self.geom_predictor.predict_field_evolution(
                    energy_query, duration_query, time_points, reference_mesh, selected_field
                )
                
                results = {
                    'predictions': predictions,
                    'confidences': confidences,
                    'time_points': time_points
                }
                
                st.success(f"✅ Prediction generated for {selected_field} (E={energy_query:.1f}mJ, τ={duration_query:.1f}ns)")
                
                # Visualization based on type
                if viz_type == "3D Mesh":
                    self.render_prediction_3d_mesh(results, reference_mesh, selected_field, time_points, energy_query, duration_query)
                elif viz_type == "Cross-Section":
                    self.render_prediction_cross_section(results, reference_mesh, selected_field, time_points, energy_query, duration_query)
                elif viz_type == "Animation":
                    self.render_prediction_animation(results, reference_mesh, selected_field, time_points, energy_query, duration_query)
                
                # Prediction statistics
                self.render_prediction_statistics(results, selected_field)
    
    def render_prediction_3d_mesh(self, results, reference_mesh, field_name, time_points, energy_query, duration_query):
        """Render 3D mesh visualization of predictions"""
        selected_time_idx = st.slider(
            "Select time for mesh visualization",
            0, len(time_points) - 1, 0,
            key="mesh_time_slider"
        )
        selected_time = time_points[selected_time_idx]
        
        if results['predictions'][selected_time_idx] is not None:
            fig = self.visualizer.create_mesh_field_visualization(
                reference_mesh,
                field_name,
                results['predictions'][selected_time_idx],
                colormap=st.session_state.selected_colormap,
                title=f"Predicted {field_name} at t={selected_time}ns<br><sub>E={energy_query:.1f}mJ, τ={duration_query:.1f}ns (Confidence: {results['confidences'][selected_time_idx]:.2%})</sub>",
                opacity=st.session_state.get('mesh_opacity', 0.9),
                show_wireframe=st.session_state.get('show_wireframe', True),
                show_points=st.session_state.get('show_points', False)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_prediction_cross_section(self, results, reference_mesh, field_name, time_points, energy_query, duration_query):
        """Render cross-section visualization of predictions"""
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_time_idx = st.slider(
                "Select time",
                0, len(time_points)-1, 0,
                key="cs_time_slider"
            )
        with col2:
            slice_axis = st.selectbox("Slice Axis", ["X", "Y", "Z"], key="pred_slice_axis")
        with col3:
            if reference_mesh.points is not None:
                if slice_axis == 'X':
                    coord_range = (np.min(reference_mesh.points[:, 0]), np.max(reference_mesh.points[:, 0]))
                elif slice_axis == 'Y':
                    coord_range = (np.min(reference_mesh.points[:, 1]), np.max(reference_mesh.points[:, 1]))
                else:
                    coord_range = (np.min(reference_mesh.points[:, 2]), np.max(reference_mesh.points[:, 2]))
                
                slice_pos = st.slider(
                    f"{slice_axis} Position",
                    float(coord_range[0]),
                    float(coord_range[1]),
                    float((coord_range[0] + coord_range[1]) / 2),
                    key="pred_slice_pos"
                )
        
        selected_time = time_points[selected_time_idx]
        
        if results['predictions'][selected_time_idx] is not None:
            slice_fig = self.visualizer.create_cross_section_view(
                reference_mesh,
                results['predictions'][selected_time_idx],
                slice_axis,
                slice_pos,
                field_name,
                colormap=st.session_state.selected_colormap
            )
            
            slice_fig.update_layout(
                title=f"Predicted {field_name} Cross-Section at t={selected_time}ns<br><sub>E={energy_query:.1f}mJ, τ={duration_query:.1f}ns (Confidence: {results['confidences'][selected_time_idx]:.2%})</sub>"
            )
            
            st.plotly_chart(slice_fig, use_container_width=True)
    
    def render_prediction_animation(self, results, reference_mesh, field_name, time_points, energy_query, duration_query):
        """Render animation visualization of predictions"""
        anim_fig = self.visualizer.create_field_animation(
            [pred for pred in results['predictions'] if pred is not None],
            reference_mesh,
            time_points,
            field_name,
            colormap=st.session_state.selected_colormap
        )
        
        anim_fig.update_layout(
            title=f"Predicted {field_name} Evolution<br><sub>E={energy_query:.1f}mJ, τ={duration_query:.1f}ns</sub>"
        )
        
        st.plotly_chart(anim_fig, use_container_width=True)
    
    def render_prediction_statistics(self, results, field_name):
        """Render prediction statistics"""
        st.markdown("##### 📊 Prediction Statistics")
        
        # Compute statistics across time
        valid_predictions = [pred for pred in results['predictions'] if pred is not None]
        valid_confidences = [conf for conf in results['confidences'] if conf > 0]
        
        if valid_predictions:
            all_values = np.concatenate([pred.flatten() for pred in valid_predictions])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean Value", f"{np.mean(all_values):.3f}")
            with col2:
                st.metric("Maximum Value", f"{np.max(all_values):.3f}")
            with col3:
                st.metric("Std Deviation", f"{np.std(all_values):.3f}")
            with col4:
                st.metric("Average Confidence", f"{np.mean(valid_confidences):.2%}")
            
            # Time series plot
            if valid_confidences:
                fig = go.Figure()
                
                # Field mean values over time
                mean_values = [np.mean(pred) if pred is not None else np.nan for pred in results['predictions']]
                fig.add_trace(go.Scatter(
                    x=results['time_points'],
                    y=mean_values,
                    mode='lines+markers',
                    name=f'{field_name} Mean',
                    line=dict(color='blue', width=3)
                ))
                
                # Confidence values
                fig.add_trace(go.Scatter(
                    x=results['time_points'],
                    y=results['confidences'],
                    mode='lines+markers',
                    name='Confidence',
                    line=dict(color='orange', width=2, dash='dash'),
                    yaxis='y2'
                ))
                
                fig.update_layout(
                    title=f"{field_name} Evolution and Confidence",
                    xaxis_title="Time (ns)",
                    yaxis_title=f"{field_name} Value",
                    yaxis2=dict(
                        title="Confidence",
                        overlaying='y',
                        side='right',
                        range=[0, 1]
                    ),
                    height=400,
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def render_data_viewer(self):
        """Render data viewer with mesh visualization"""
        st.markdown('<h2 class="sub-header">📁 Data Viewer with Mesh Visualization</h2>', 
                   unsafe_allow_html=True)
        
        if not st.session_state.get('data_loaded', False):
            self.show_data_not_loaded_warning()
            return
        
        simulations = st.session_state.simulations
        
        # Simulation selection
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            sim_name = st.selectbox(
                "Select Simulation",
                sorted(simulations.keys()),
                key="viewer_sim_select"
            )
        
        sim = simulations[sim_name]
        
        with col2:
            st.metric("Energy", f"{sim['energy_mJ']:.2f} mJ")
        with col3:
            st.metric("Duration", f"{sim['duration_ns']:.2f} ns")
        
        if not sim.get('has_mesh', False):
            st.warning("This simulation was loaded without mesh data.")
            return
        
        # Field and timestep selection
        col1, col2, col3 = st.columns(3)
        with col1:
            field = st.selectbox(
                "Select Field",
                sorted(sim['mesh_data'].field_info.keys()),
                key="viewer_field_select"
            )
        with col2:
            timestep = st.slider(
                "Timestep",
                0, sim['n_timesteps'] - 1, 0,
                key="viewer_timestep_slider"
            )
        with col3:
            display_mode = st.selectbox(
                "Display Mode",
                ["Surface Mesh", "Point Cloud", "Wireframe"],
                key="display_mode"
            )
        
        # Get mesh data and field values
        mesh_data = sim['mesh_data']
        field_values = mesh_data.fields[field][timestep]
        
        # Handle vector fields
        if field_values.ndim == 2:
            field_values = np.linalg.norm(field_values, axis=1)
        
        # Create visualization
        show_wireframe = display_mode == "Wireframe" or st.session_state.get('show_wireframe', True)
        show_points = display_mode == "Point Cloud" or st.session_state.get('show_points', False)
        
        fig = self.visualizer.create_mesh_field_visualization(
            mesh_data,
            field,
            field_values,
            colormap=st.session_state.selected_colormap,
            title=f"{field} at Timestep {timestep + 1} - {sim_name}",
            opacity=st.session_state.get('mesh_opacity', 0.9),
            show_wireframe=show_wireframe,
            show_points=show_points
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Field statistics and cross-sections
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 📊 Field Statistics")
            
            stats = {
                "Minimum": float(np.min(field_values)),
                "Maximum": float(np.max(field_values)),
                "Mean": float(np.mean(field_values)),
                "Std Dev": float(np.std(field_values)),
                "Median": float(np.median(field_values)),
                "Range": float(np.max(field_values) - np.min(field_values))
            }
            
            for stat_name, stat_value in stats.items():
                st.metric(stat_name, f"{stat_value:.3f}")
        
        with col2:
            st.markdown("##### 🔍 Cross-Section View")
            
            slice_axis = st.selectbox("Slice Axis", ["X", "Y", "Z"], key="slice_axis")
            
            if mesh_data.points is not None:
                if slice_axis == 'X':
                    coord_range = (np.min(mesh_data.points[:, 0]), np.max(mesh_data.points[:, 0]))
                elif slice_axis == 'Y':
                    coord_range = (np.min(mesh_data.points[:, 1]), np.max(mesh_data.points[:, 1]))
                else:
                    coord_range = (np.min(mesh_data.points[:, 2]), np.max(mesh_data.points[:, 2]))
                
                slice_pos = st.slider(
                    f"{slice_axis} Position",
                    float(coord_range[0]),
                    float(coord_range[1]),
                    float((coord_range[0] + coord_range[1]) / 2),
                    key="slice_pos"
                )
                
                if st.button("Generate Cross-Section"):
                    slice_fig = self.visualizer.create_cross_section_view(
                        mesh_data,
                        field_values,
                        slice_axis,
                        slice_pos,
                        field,
                        colormap=st.session_state.selected_colormap
                    )
                    st.plotly_chart(slice_fig, use_container_width=True)
        
        # Mesh statistics
        with st.expander("📐 Mesh Statistics", expanded=False):
            if hasattr(mesh_data, 'mesh_stats'):
                mesh_stats = mesh_data.mesh_stats
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Geometrical Properties**")
                    if 'surface_area' in mesh_stats:
                        st.metric("Surface Area", f"{mesh_stats['surface_area']:.3f}")
                    if 'triangle_count' in mesh_stats:
                        st.metric("Triangle Count", f"{mesh_stats['triangle_count']:,}")
                    if 'avg_triangle_area' in mesh_stats:
                        st.metric("Avg Triangle Area", f"{mesh_stats['avg_triangle_area']:.6f}")
                
                with col2:
                    st.markdown("**Spatial Properties**")
                    if 'bbox' in mesh_stats:
                        bbox = mesh_stats['bbox']
                        st.metric("X Range", f"{bbox['min'][0]:.3f} - {bbox['max'][0]:.3f}")
                        st.metric("Y Range", f"{bbox['min'][1]:.3f} - {bbox['max'][1]:.3f}")
                        st.metric("Z Range", f"{bbox['min'][2]:.3f} - {bbox['max'][2]:.3f}")
    
    def render_comparative_analysis(self):
        """Render comparative analysis interface"""
        st.markdown('<h2 class="sub-header">📊 Comparative Analysis</h2>', 
                   unsafe_allow_html=True)
        
        if not st.session_state.get('data_loaded', False):
            self.show_data_not_loaded_warning()
            return
        
        simulations = st.session_state.simulations
        
        # Simulation selection
        selected_sims = st.multiselect(
            "Select simulations for comparison",
            sorted(simulations.keys()),
            default=list(simulations.keys())[:min(3, len(simulations))]
        )
        
        if not selected_sims:
            st.info("Please select at least one simulation for comparison.")
            return
        
        # Field selection with field mapping support
        all_fields = set()
        for sim_name in selected_sims:
            if sim_name in simulations:
                for field in simulations[sim_name]['mesh_data'].field_info.keys():
                    canonical_field = self.data_loader.get_canonical_field_name(field)
                    all_fields.add(canonical_field)
        
        if all_fields:
            selected_field = self.render_field_category_selector(sorted(all_fields))
        else:
            st.error("No fields found in selected simulations.")
            return
        
        # Comparison visualization
        tab1, tab2, tab3 = st.tabs(["📈 Field Evolution", "🎯 Radar Chart", "🏗️ Geometry Comparison"])
        
        with tab1:
            self.render_field_evolution_comparison(selected_sims, selected_field)
        
        with tab2:
            self.render_radar_comparison(selected_sims)
        
        with tab3:
            self.render_geometry_comparison(selected_sims, selected_field)
    
    def render_field_evolution_comparison(self, selected_sims, selected_field):
        """Render field evolution comparison"""
        fig = go.Figure()
        
        for sim_name in selected_sims:
            sim = st.session_state.simulations[sim_name]
            
            if selected_field in sim['mesh_data'].field_info:
                mesh_data = sim['mesh_data']
                field_data = mesh_data.fields[selected_field]
                
                # Compute mean value over space for each timestep
                spatial_means = []
                for t in range(field_data.shape[0]):
                    if field_data[t].ndim == 1:
                        values = field_data[t]
                    else:
                        values = np.linalg.norm(field_data[t], axis=1)
                    spatial_means.append(np.nanmean(values))
                
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(spatial_means) + 1)),
                    y=spatial_means,
                    mode='lines+markers',
                    name=sim_name,
                    line=dict(width=3),
                    hovertemplate=f'{sim_name}<br>Timestep: %{{x}}<br>Mean {selected_field}: %{{y:.3f}}<extra></extra>'
                ))
        
        fig.update_layout(
            title=f"{selected_field} Evolution Comparison",
            xaxis_title="Timestep",
            yaxis_title=f"Mean {selected_field}",
            height=500,
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_radar_comparison(self, selected_sims):
        """Render radar chart comparison"""
        # Extract field statistics for each simulation
        field_stats = {}
        
        for sim_name in selected_sims:
            sim = st.session_state.simulations[sim_name]
            summary = next((s for s in st.session_state.summaries if s['name'] == sim_name), None)
            
            if summary:
                stats = {}
                for field in summary['field_stats']:
                    if 'mean' in summary['field_stats'][field] and summary['field_stats'][field]['mean']:
                        stats[field] = np.mean(summary['field_stats'][field]['mean'])
                
                field_stats[sim_name] = stats
        
        # Create radar chart
        if field_stats:
            fig = go.Figure()
            
            for sim_name, stats in field_stats.items():
                fields = list(stats.keys())[:6]  # Limit to 6 fields for clarity
                values = [stats[field] for field in fields]
                
                # Normalize values
                if len(values) > 0 and max(values) > 0:
                    values = [v / max(values) for v in values]
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=fields,
                    fill='toself',
                    name=sim_name
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                showlegend=True,
                title="Field Statistics Comparison (Normalized)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_geometry_comparison(self, selected_sims, selected_field):
        """Render geometry comparison"""
        for sim_name in selected_sims:
            sim = st.session_state.simulations[sim_name]
            mesh_data = sim['mesh_data']
            
            if selected_field in mesh_data.fields:
                field_data = mesh_data.fields[selected_field]
                
                # Use first timestep
                if field_data[0].ndim == 1:
                    values = field_data[0]
                else:
                    values = np.linalg.norm(field_data[0], axis=1)
                
                fig = self.visualizer.create_mesh_field_visualization(
                    mesh_data,
                    selected_field,
                    values,
                    colormap=st.session_state.selected_colormap,
                    title=f"{sim_name} - {selected_field}",
                    opacity=0.8
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def render_geometrical_visualization(self):
        """Render advanced geometrical visualization interface"""
        st.markdown('<h2 class="sub-header">🏗️ Advanced Geometrical Visualization</h2>', 
                   unsafe_allow_html=True)
        
        if not st.session_state.get('data_loaded', False):
            self.show_data_not_loaded_warning()
            return
        
        st.markdown("""
        <div class="info-box">
        <h3>🎨 Geometrical Mesh Visualization</h3>
        <p>This module provides advanced 3D visualization of field distributions on mesh geometry, 
        including cross-sections, animations, and spatial analysis tools.</p>
        </div>
        """, unsafe_allow_html=True)
        
        simulations = st.session_state.simulations
        
        # Main visualization controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sim_name = st.selectbox(
                "Select Simulation",
                sorted(simulations.keys()),
                key="geom_sim_select"
            )
        
        with col2:
            sim = simulations[sim_name]
            field = st.selectbox(
                "Select Field",
                sorted(sim['mesh_data'].field_info.keys()),
                key="geom_field_select"
            )
        
        with col3:
            viz_mode = st.selectbox(
                "Visualization Mode",
                ["Interactive 3D", "Cross-Section Analysis", "Time Animation", "Spatial Statistics"],
                key="geom_viz_mode"
            )
        
        # Visualization parameters
        if viz_mode == "Interactive 3D":
            self.render_interactive_3d(sim, field)
        elif viz_mode == "Cross-Section Analysis":
            self.render_cross_section_analysis(sim, field)
        elif viz_mode == "Time Animation":
            self.render_time_animation(sim, field)
        elif viz_mode == "Spatial Statistics":
            self.render_spatial_statistics(sim, field)
    
    def render_interactive_3d(self, sim, field):
        """Render interactive 3D visualization"""
        mesh_data = sim['mesh_data']
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            timestep = st.slider(
                "Timestep",
                0, sim['n_timesteps'] - 1, 0,
                key="interactive_timestep"
            )
        
        with col2:
            display_options = st.multiselect(
                "Display Options",
                ["Wireframe", "Points", "Bounding Box", "Coordinate Axes"],
                default=["Wireframe", "Coordinate Axes"],
                key="display_options"
            )
        
        # Get field values
        field_data = mesh_data.fields[field][timestep]
        if field_data.ndim == 2:
            field_data = np.linalg.norm(field_data, axis=1)
        
        # Create visualization
        fig = self.visualizer.create_mesh_field_visualization(
            mesh_data,
            field,
            field_data,
            colormap=st.session_state.selected_colormap,
            title=f"{field} at Timestep {timestep + 1}",
            opacity=0.9,
            show_wireframe="Wireframe" in display_options,
            show_points="Points" in display_options
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional controls
        with st.expander("🎨 Visualization Controls", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                camera_x = st.slider("Camera X", -3.0, 3.0, 1.5, 0.1)
                camera_y = st.slider("Camera Y", -3.0, 3.0, 1.5, 0.1)
                camera_z = st.slider("Camera Z", -3.0, 3.0, 1.5, 0.1)
            
            with col2:
                light_ambient = st.slider("Ambient Light", 0.0, 1.0, 0.8, 0.1)
                light_diffuse = st.slider("Diffuse Light", 0.0, 1.0, 0.8, 0.1)
            
            with col3:
                if st.button("Apply Camera Settings"):
                    # Update camera settings
                    fig.update_layout(
                        scene_camera=dict(
                            eye=dict(x=camera_x, y=camera_y, z=camera_z)
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Field statistics
        self.render_field_statistics(field_data, field)
    
    def render_cross_section_analysis(self, sim, field):
        """Render cross-section analysis visualization"""
        mesh_data = sim['mesh_data']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            slice_axis = st.selectbox("Slice Axis", ["X", "Y", "Z"], key="cs_axis")
        
        with col2:
            if mesh_data.points is not None:
                if slice_axis == 'X':
                    coord_range = (np.min(mesh_data.points[:, 0]), np.max(mesh_data.points[:, 0]))
                elif slice_axis == 'Y':
                    coord_range = (np.min(mesh_data.points[:, 1]), np.max(mesh_data.points[:, 1]))
                else:
                    coord_range = (np.min(mesh_data.points[:, 2]), np.max(mesh_data.points[:, 2]))
                
                slice_pos = st.slider(
                    f"{slice_axis} Position",
                    float(coord_range[0]),
                    float(coord_range[1]),
                    float((coord_range[0] + coord_range[1]) / 2),
                    key="cs_pos"
                )
        
        with col3:
            timestep = st.slider(
                "Timestep",
                0, sim['n_timesteps'] - 1, 0,
                key="cs_timestep"
            )
        
        # Get field values
        field_data = mesh_data.fields[field][timestep]
        if field_data.ndim == 2:
            field_data = np.linalg.norm(field_data, axis=1)
        
        # Create cross-section visualization
        slice_fig = self.visualizer.create_cross_section_view(
            mesh_data,
            field_data,
            slice_axis,
            slice_pos,
            field,
            colormap=st.session_state.selected_colormap
        )
        
        st.plotly_chart(slice_fig, use_container_width=True)
        
        # Multiple cross-sections
        with st.expander("📊 Multiple Cross-Sections", expanded=False):
            n_slices = st.slider("Number of slices", 1, 10, 3, key="n_slices")
            
            slice_positions = np.linspace(coord_range[0], coord_range[1], n_slices + 2)[1:-1]
            
            cols = st.columns(min(n_slices, 4))
            
            for idx, pos in enumerate(slice_positions):
                if idx < len(cols):
                    with cols[idx]:
                        st.markdown(f"**{slice_axis} = {pos:.3f}**")
                        
                        # Create small cross-section
                        small_fig = self.visualizer.create_cross_section_view(
                            mesh_data,
                            field_data,
                            slice_axis,
                            float(pos),
                            field,
                            colormap=st.session_state.selected_colormap
                        )
                        
                        small_fig.update_layout(height=300, showlegend=False, margin=dict(t=20, b=20, l=20, r=20))
                        st.plotly_chart(small_fig, use_container_width=True)
    
    def render_time_animation(self, sim, field):
        """Render time animation visualization"""
        mesh_data = sim['mesh_data']
        
        col1, col2 = st.columns(2)
        
        with col1:
            start_time = st.slider("Start Time", 0, sim['n_timesteps'] - 1, 0, key="anim_start")
            end_time = st.slider("End Time", 0, sim['n_timesteps'] - 1, sim['n_timesteps'] - 1, key="anim_end")
        
        with col2:
            frame_rate = st.slider("Frame Rate (fps)", 1, 30, 10, key="anim_fps")
            play_direction = st.selectbox("Play Direction", ["Forward", "Reverse"], key="anim_dir")
        
        if start_time >= end_time:
            st.error("Start time must be less than end time.")
            return
        
        # Prepare animation data
        time_indices = range(start_time, end_time + 1)
        field_evolution = []
        
        for t in time_indices:
            field_data = mesh_data.fields[field][t]
            if field_data.ndim == 2:
                field_data = np.linalg.norm(field_data, axis=1)
            field_evolution.append(field_data)
        
        if play_direction == "Reverse":
            field_evolution = field_evolution[::-1]
            time_indices = time_indices[::-1]
        
        # Create animation
        anim_fig = self.visualizer.create_field_animation(
            field_evolution,
            mesh_data,
            list(time_indices),
            field,
            colormap=st.session_state.selected_colormap
        )
        
        st.plotly_chart(anim_fig, use_container_width=True)
        
        # Animation controls
        with st.expander("🎬 Animation Controls", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                loop_animation = st.checkbox("Loop Animation", value=True, key="anim_loop")
            
            with col2:
                if st.button("🔄 Restart Animation"):
                    st.rerun()
            
            with col3:
                # Export animation as GIF (conceptual)
                if st.button("💾 Export Animation"):
                    st.info("Animation export feature would generate and download a GIF file.")
        
        # Time series statistics
        st.markdown("##### 📈 Time Series Statistics")
        
        time_series_data = []
        for t in range(sim['n_timesteps']):
            field_data = mesh_data.fields[field][t]
            if field_data.ndim == 2:
                field_data = np.linalg.norm(field_data, axis=1)
            
            time_series_data.append({
                'timestep': t + 1,
                'mean': float(np.mean(field_data)),
                'max': float(np.max(field_data)),
                'min': float(np.min(field_data)),
                'std': float(np.std(field_data))
            })
        
        df_time_series = pd.DataFrame(time_series_data)
        
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=df_time_series['timestep'],
            y=df_time_series['mean'],
            mode='lines+markers',
            name='Mean',
            line=dict(width=2, color='blue')
        ))
        fig_ts.add_trace(go.Scatter(
            x=df_time_series['timestep'],
            y=df_time_series['max'],
            mode='lines',
            name='Maximum',
            line=dict(width=1, color='red', dash='dash')
        ))
        fig_ts.add_trace(go.Scatter(
            x=df_time_series['timestep'],
            y=df_time_series['min'],
            mode='lines',
            name='Minimum',
            line=dict(width=1, color='green', dash='dash')
        ))
        
        fig_ts.update_layout(
            title=f"{field} Time Series Statistics",
            xaxis_title="Timestep",
            yaxis_title=f"{field} Value",
            height=400,
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_ts, use_container_width=True)
    
    def render_spatial_statistics(self, sim, field):
        """Render spatial statistics visualization"""
        mesh_data = sim['mesh_data']
        
        col1, col2 = st.columns(2)
        
        with col1:
            timestep = st.slider(
                "Timestep for Analysis",
                0, sim['n_timesteps'] - 1, 0,
                key="spatial_timestep"
            )
        
        with col2:
            analysis_type = st.selectbox(
                "Analysis Type",
                ["Spatial Distribution", "Radial Analysis", "Region Analysis", "Gradient Analysis"],
                key="analysis_type"
            )
        
        # Get field values
        field_data = mesh_data.fields[field][timestep]
        if field_data.ndim == 2:
            field_data = np.linalg.norm(field_data, axis=1)
        
        if analysis_type == "Spatial Distribution":
            self.render_spatial_distribution(mesh_data, field_data, field)
        elif analysis_type == "Radial Analysis":
            self.render_radial_analysis(mesh_data, field_data, field)
        elif analysis_type == "Region Analysis":
            self.render_region_analysis(mesh_data, field_data, field)
        elif analysis_type == "Gradient Analysis":
            self.render_gradient_analysis(mesh_data, field_data, field)
    
    def render_spatial_distribution(self, mesh_data, field_values, field_name):
        """Render spatial distribution analysis"""
        points = mesh_data.points
        
        # Create 2D histogram
        fig_hist = go.Figure()
        
        # Project onto XY plane
        fig_hist.add_trace(go.Histogram2d(
            x=points[:, 0],
            y=points[:, 1],
            z=field_values,
            histfunc="avg",
            colorscale=st.session_state.selected_colormap,
            colorbar=dict(title=field_name)
        ))
        
        fig_hist.update_layout(
            title=f"Spatial Distribution of {field_name} (XY Projection)",
            xaxis_title="X",
            yaxis_title="Y",
            height=500
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Statistical analysis
        st.markdown("##### 📊 Spatial Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Spatial autocorrelation
            if len(field_values) > 100:
                from scipy.spatial.distance import pdist, squareform
                from scipy.stats import pearsonr
                
                # Sample for efficiency
                sample_indices = np.random.choice(len(field_values), min(100, len(field_values)), replace=False)
                sample_points = points[sample_indices]
                sample_values = field_values[sample_indices]
                
                # Compute distance matrix
                distances = pdist(sample_points)
                value_diffs = pdist(sample_values.reshape(-1, 1), metric=lambda u, v: abs(u[0] - v[0]))
                
                if len(distances) > 10:
                    try:
                        corr, _ = pearsonr(distances, value_diffs)
                        st.metric("Spatial Correlation", f"{corr:.3f}")
                    except:
                        st.metric("Spatial Correlation", "N/A")
        
        with col2:
            st.metric("Spatial Mean", f"{np.mean(field_values):.3f}")
        
        with col3:
            st.metric("Spatial Std Dev", f"{np.std(field_values):.3f}")
        
        with col4:
            st.metric("Spatial Range", f"{np.max(field_values) - np.min(field_values):.3f}")
    
    def render_radial_analysis(self, mesh_data, field_values, field_name):
        """Render radial analysis"""
        points = mesh_data.points
        
        # Compute distances from centroid
        centroid = np.mean(points, axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        
        # Create radial scatter plot
        fig_radial = go.Figure()
        
        fig_radial.add_trace(go.Scatter(
            x=distances,
            y=field_values,
            mode='markers',
            marker=dict(
                size=3,
                color=field_values,
                colorscale=st.session_state.selected_colormap,
                showscale=True,
                colorbar=dict(title=field_name)
            ),
            hovertemplate=f'Distance: %{{x:.3f}}<br>{field_name}: %{{y:.3f}}<extra></extra>'
        ))
        
        # Add trend line
        if len(distances) > 10:
            try:
                z = np.polyfit(distances, field_values, 2)
                p = np.poly1d(z)
                
                sorted_indices = np.argsort(distances)
                trend_x = distances[sorted_indices]
                trend_y = p(trend_x)
                
                fig_radial.add_trace(go.Scatter(
                    x=trend_x,
                    y=trend_y,
                    mode='lines',
                    line=dict(color='red', width=2),
                    name='Trend Line'
                ))
            except:
                pass
        
        fig_radial.update_layout(
            title=f"{field_name} vs Radial Distance from Centroid",
            xaxis_title="Radial Distance",
            yaxis_title=field_name,
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig_radial, use_container_width=True)
    
    def render_region_analysis(self, mesh_data, field_values, field_name):
        """Render region-based analysis"""
        if not hasattr(mesh_data, 'region_labels') or mesh_data.region_labels is None:
            mesh_data.segment_regions(n_regions=5)
        
        region_labels = mesh_data.region_labels
        
        # Compute statistics per region
        region_stats = []
        
        for region_id in np.unique(region_labels):
            region_mask = region_labels == region_id
            region_values = field_values[region_mask]
            
            if len(region_values) > 0:
                region_stats.append({
                    'Region': region_id,
                    'Count': int(np.sum(region_mask)),
                    'Mean': float(np.mean(region_values)),
                    'Std': float(np.std(region_values)),
                    'Min': float(np.min(region_values)),
                    'Max': float(np.max(region_values))
                })
        
        if region_stats:
            df_region_stats = pd.DataFrame(region_stats)
            
            # Display region statistics
            st.dataframe(df_region_stats.style.format({
                'Mean': '{:.3f}',
                'Std': '{:.3f}',
                'Min': '{:.3f}',
                'Max': '{:.3f}'
            }), use_container_width=True)
            
            # Visualize region statistics
            fig_regions = go.Figure()
            
            fig_regions.add_trace(go.Bar(
                x=df_region_stats['Region'],
                y=df_region_stats['Mean'],
                error_y=dict(
                    type='data',
                    array=df_region_stats['Std'],
                    visible=True
                ),
                name='Mean ± Std',
                marker_color='skyblue'
            ))
            
            fig_regions.update_layout(
                title=f"{field_name} Statistics by Region",
                xaxis_title="Region ID",
                yaxis_title=f"{field_name} Value",
                height=400
            )
            
            st.plotly_chart(fig_regions, use_container_width=True)
            
            # Color mesh by region
            region_colors = region_labels / np.max(region_labels)
            
            fig_region_viz = self.visualizer.create_mesh_field_visualization(
                mesh_data,
                "Region",
                region_colors,
                colormap="Viridis",
                title="Mesh Regions",
                opacity=0.8
            )
            
            st.plotly_chart(fig_region_viz, use_container_width=True)
    
    def render_gradient_analysis(self, mesh_data, field_values, field_name):
        """Render gradient analysis"""
        points = mesh_data.points
        
        if mesh_data.triangles is not None and len(mesh_data.triangles) > 0:
            # Compute gradients on mesh
            gradients = np.zeros_like(points)
            
            for tri in mesh_data.triangles[:min(1000, len(mesh_data.triangles))]:
                if (tri[0] < len(points) and tri[1] < len(points) and tri[2] < len(points) and
                    tri[0] < len(field_values) and tri[1] < len(field_values) and tri[2] < len(field_values)):
                    
                    # Get triangle vertices and values
                    v0, v1, v2 = points[tri]
                    f0, f1, f2 = field_values[tri]
                    
                    # Compute gradient using finite differences
                    # Simple approximation for demonstration
                    grad = np.zeros(3)
                    for i in range(3):
                        if v1[i] != v0[i]:
                            grad[i] += (f1 - f0) / (v1[i] - v0[i])
                        if v2[i] != v0[i]:
                            grad[i] += (f2 - f0) / (v2[i] - v0[i])
                    
                    gradients[tri] += grad / 3
            
            # Compute gradient magnitude
            grad_magnitude = np.linalg.norm(gradients, axis=1)
            
            # Visualize gradient magnitude
            fig_grad = self.visualizer.create_mesh_field_visualization(
                mesh_data,
                f"{field_name} Gradient Magnitude",
                grad_magnitude,
                colormap="Hot",
                title=f"Gradient Magnitude of {field_name}",
                opacity=0.9
            )
            
            st.plotly_chart(fig_grad, use_container_width=True)
            
            # Gradient statistics
            st.markdown("##### 📊 Gradient Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Mean Gradient", f"{np.mean(grad_magnitude):.3f}")
            
            with col2:
                st.metric("Max Gradient", f"{np.max(grad_magnitude):.3f}")
            
            with col3:
                # Compute gradient direction consistency
                if len(gradients) > 10:
                    # Normalize gradients
                    grad_norm = gradients / (np.linalg.norm(gradients, axis=1, keepdims=True) + 1e-8)
                    
                    # Compute mean direction
                    mean_direction = np.mean(grad_norm, axis=0)
                    direction_magnitude = np.linalg.norm(mean_direction)
                    
                    st.metric("Direction Consistency", f"{direction_magnitude:.3f}")
        else:
            st.info("Gradient analysis requires triangular mesh data.")
    
    def render_field_statistics(self, field_values, field_name):
        """Render field statistics panel"""
        st.markdown("##### 📊 Field Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Minimum", f"{np.min(field_values):.3f}")
        
        with col2:
            st.metric("Maximum", f"{np.max(field_values):.3f}")
        
        with col3:
            st.metric("Mean", f"{np.mean(field_values):.3f}")
        
        with col4:
            st.metric("Std Dev", f"{np.std(field_values):.3f}")
        
        # Additional statistics
        with st.expander("📈 Detailed Statistics", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Median", f"{np.median(field_values):.3f}")
            
            with col2:
                st.metric("Variance", f"{np.var(field_values):.3f}")
            
            with col3:
                from scipy.stats import skew, kurtosis
                try:
                    st.metric("Skewness", f"{skew(field_values):.3f}")
                except:
                    st.metric("Skewness", "N/A")
            
            with col4:
                try:
                    st.metric("Kurtosis", f"{kurtosis(field_values):.3f}")
                except:
                    st.metric("Kurtosis", "N/A")
            
            # Histogram
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=field_values,
                nbinsx=50,
                marker_color='skyblue',
                opacity=0.7,
                name='Distribution'
            ))
            
            fig_hist.update_layout(
                title=f"{field_name} Value Distribution",
                xaxis_title=field_name,
                yaxis_title="Frequency",
                height=300
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)
    
    def show_data_not_loaded_warning(self):
        """Show warning when data is not loaded"""
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

# =============================================
# MAIN APPLICATION ENTRY POINT
# =============================================
def main():
    """Main application entry point"""
    app = FEAVisualizationPlatform()
    app.run()

if __name__ == "__main__":
    main()
