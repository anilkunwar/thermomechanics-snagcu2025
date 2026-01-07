import streamlit as st
import os
import glob
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')
import json
from pathlib import Path
import sys

# Imports for enhanced renderer
import pyvista as pv
from stpyvista import stpyvista
from stpyvista.utils import start_xvfb

# Initialize Xvfb for cloud compatibility
if "IS_XVFB_RUNNING" not in st.session_state:
    start_xvfb()
    st.session_state.IS_XVFB_RUNNING = True

# =============================================
# DEBUG MODE & SETTINGS
# =============================================
DEBUG_MODE = True  # Set to True for debugging, False for production

# =============================================
# ALTERNATIVE VTU READER (FALLBACK IF PyVista NOT AVAILABLE)
# =============================================
class VTUReader:
    """Fallback VTU reader if PyVista is not available"""
    
    @staticmethod
    def try_import_pyvista():
        try:
            import pyvista as pv
            pv.set_jupyter_backend(None)
            return pv, None
        except ImportError as e:
            error_msg = """
            ### ⚠️ PyVista Installation Required
            pip install pyvista stpyvista
            """
            return None, error_msg
    
    @staticmethod
    def read_vtu_fallback(vtu_path):
        try:
            import meshio
            mesh = meshio.read(vtu_path)
            return mesh
        except ImportError:
            st.warning("Install meshio for fallback.")
            return None
    
    @staticmethod
    def create_mock_data():
        st.info("📊 Using mock data for demonstration")
        
        # Create a structured grid for better mesh viz
        x = np.linspace(-10, 10, 20)
        y = np.linspace(-10, 10, 20)
        z = np.linspace(-5, 5, 10)
        xx, yy, zz = np.meshgrid(x, y, z)
        points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
        r = np.sqrt(xx**2 + yy**2 + zz**2).ravel()
        
        base_mesh = pv.StructuredGrid(xx, yy, zz)
        
        n_timesteps = 20
        fields = {}
        field_info = {}
        
        # Scalar field
        temperature = 300 + 1000 * np.exp(-r / 50) * np.sin(np.linspace(0, 2*np.pi, n_timesteps))[:, None]
        fields['temperature'] = temperature
        field_info['temperature'] = ('scalar', 1)
        
        # Vector field
        displacement = np.zeros((n_timesteps, len(r), 3))
        for t in range(n_timesteps):
            displacement[t, :, 0] = 0.1 * np.sin(0.1 * t) * xx.ravel() / (r + 1)
            displacement[t, :, 1] = 0.1 * np.sin(0.1 * t) * yy.ravel() / (r + 1)
            displacement[t, :, 2] = 0.2 * np.sin(0.1 * t) * np.exp(-zz.ravel()**2 / 100)
        fields['displacement'] = displacement
        field_info['displacement'] = ('vector', 3)
        
        # Add to base_mesh for first timestep
        base_mesh.point_data['temperature'] = fields['temperature'][0]
        base_mesh.point_data['displacement'] = fields['displacement'][0]
        
        return {
            'mock_1': {
                'energy_mJ': 0.5,
                'duration_ns': 4.2,
                'points': points,
                'base_mesh': base_mesh,
                'fields': fields,
                'field_info': field_info,
                'units': {'temperature': 'K', 'displacement': 'm'},
                'n_timesteps': n_timesteps,
                'timesteps': np.linspace(0, 4.2, n_timesteps),
                'folder': 'q0p5mJ-delta4p2ns',
                'is_mock': True
            },
            'mock_2': {
                'energy_mJ': 1.0,
                'duration_ns': 2.0,
                'points': points * 0.8,
                'base_mesh': base_mesh.scale(0.8),
                'fields': {k: v * 0.5 for k, v in fields.items()},
                'field_info': field_info,
                'units': {'temperature': 'K', 'displacement': 'm'},
                'n_timesteps': n_timesteps,
                'timesteps': np.linspace(0, 2.0, n_timesteps),
                'folder': 'q1p0mJ-delta2p0ns',
                'is_mock': True
            }
        }

# =============================================
# PATH VALIDATION & DEBUG UTILITIES
# =============================================
# (Retained from original, no changes)

def validate_and_setup_paths():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    FEA_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "fea_solutions")
    debug_info = {
        "script_directory": SCRIPT_DIR,
        "fea_solutions_dir": FEA_SOLUTIONS_DIR,
        "fea_dir_exists": os.path.exists(FEA_SOLUTIONS_DIR),
        "current_working_dir": os.getcwd(),
        "python_version": sys.version,
        "os": os.name
    }
    if not debug_info["fea_dir_exists"]:
        st.warning(f"Directory not found: {FEA_SOLUTIONS_DIR}. Creating it.")
    os.makedirs(FEA_SOLUTIONS_DIR, exist_ok=True)
    return FEA_SOLUTIONS_DIR, debug_info

def scan_directory_structure(base_dir):
    structure = {
        "directories": [],
        "vtu_files": [],
        "other_files": [],
        "total_vtu_count": 0
    }
    try:
        for root, dirs, files in os.walk(base_dir):
            rel_root = os.path.relpath(root, base_dir)
            if rel_root == ".":
                rel_root = "/"
            for d in dirs:
                structure["directories"].append(os.path.join(rel_root, d))
            for f in files:
                rel_path = os.path.join(rel_root, f)
                if f.endswith('.vtu'):
                    structure["vtu_files"].append(rel_path)
                    structure["total_vtu_count"] += 1
                else:
                    structure["other_files"].append(rel_path)
    except Exception as e:
        structure["error"] = str(e)
    return structure

# =============================================
# FOLDER PARSER WITH MULTIPLE PATTERNS
# =============================================
# (Retained from original, no changes)

def parse_folder_name(folder: str):
    base_name = os.path.basename(folder)
    patterns = [
        r"q([\dp\.]+)mJ[-_]delta([\dp\.]+)ns",
        r"energy[_-]([\d\.]+)mJ[-_]duration[_-]([\d\.]+)ns",
        r"q([\d\.]+)mJ[-_]delta([\d\.]+)ns",
        r"sim[_-]([\d\.]+)mJ[_-]([\d\.]+)ns"
    ]
    for pattern in patterns:
        match = re.match(pattern, base_name, re.IGNORECASE)
        if match:
            e, d = match.groups()
            energy = float(e.replace("p", ".") if "p" in e else e)
            duration = float(d.replace("p", ".") if "p" in d else d)
            return energy, duration
    numbers = re.findall(r"[\d\.]+", base_name)
    if len(numbers) >= 2:
        try:
            return float(numbers[0]), float(numbers[1])
        except:
            pass
    if DEBUG_MODE:
        st.sidebar.warning(f"Could not parse folder: {base_name}")
    return None, None

# =============================================
# ROBUST DATA LOADER WITH DEBUGGING
# =============================================
@st.cache_data(show_spinner=False, ttl=3600)
def load_simulations_with_fallback(use_mock_data=False, debug_info=None):
    if use_mock_data:
        return VTUReader.create_mock_data(), "mock"
    
    try:
        pyvista, error_msg = VTUReader.try_import_pyvista()
        if pyvista is None:
            if debug_info:
                debug_info["pyvista_available"] = False
                debug_info["pyvista_error"] = error_msg
            return {}, "no_pyvista"
        
        if debug_info:
            debug_info["pyvista_available"] = True
        
        FEA_SOLUTIONS_DIR = debug_info["fea_solutions_dir"] if debug_info else os.path.join(os.path.dirname(__file__), "fea_solutions")
        
        folders = glob.glob(os.path.join(FEA_SOLUTIONS_DIR, "q*mJ-delta*ns"))
        all_folders = [f for f in glob.glob(os.path.join(FEA_SOLUTIONS_DIR, "*")) if os.path.isdir(f)]
        for folder in all_folders:
            if folder not in folders and glob.glob(os.path.join(folder, "*.vtu")):
                folders.append(folder)
        
        if debug_info:
            debug_info["found_folders"] = folders
            debug_info["folder_count"] = len(folders)
        
        if not folders:
            return {}, "no_folders"
        
        simulations = {}
        loaded_count = 0
        
        for folder in folders:
            folder_name = os.path.basename(folder)
            energy, duration = parse_folder_name(folder_name)
            vtu_files = sorted(glob.glob(os.path.join(folder, "*.vtu")))
            if not vtu_files:
                continue
            
            base_mesh = pyvista.read(vtu_files[0])
            if not base_mesh.point_data:
                continue
            
            points = base_mesh.points
            n_pts = base_mesh.n_points
            n_steps = len(vtu_files)
            
            fields = {}
            field_info = {}
            units = {}
            
            for key in base_mesh.point_data.keys():
                arr = np.asarray(base_mesh.point_data[key])
                kind = "scalar" if arr.ndim == 1 else "vector"
                dim = 1 if kind == "scalar" else arr.shape[1]
                unit = guess_unit(key)
                field_info[key] = (kind, dim)
                units[key] = unit
                shape = (n_steps, n_pts) if kind == "scalar" else (n_steps, n_pts, dim)
                fields[key] = np.full(shape, np.nan, dtype=np.float32)
            
            for t, vtu in enumerate(vtu_files[:50]):
                mesh = pyvista.read(vtu)
                for key in field_info:
                    if key in mesh.point_data:
                        fields[key][t] = np.asarray(mesh.point_data[key], dtype=np.float32)
            
            derived = calculate_derived_fields(fields, field_info)
            fields.update(derived)
            
            simulations[folder_name] = {
                'energy_mJ': energy or 0.0,
                'duration_ns': duration or 0.0,
                'points': points,
                'base_mesh': base_mesh,
                'fields': fields,
                'field_info': field_info,
                'units': units,
                'n_timesteps': n_steps,
                'timesteps': np.linspace(0, duration or 10.0, n_steps),
                'folder': folder_name,
                'is_mock': False
            }
            
            loaded_count += 1
        
        if debug_info:
            debug_info["loaded_simulations"] = loaded_count
        
        return simulations, "success" if loaded_count > 0 else "no_valid_data"
    
    except Exception as e:
        if debug_info:
            debug_info["load_error"] = str(e)
        return {}, f"error: {str(e)[:100]}"

def guess_unit(field_name: str) -> str:
    field_lower = field_name.lower()
    if 'temp' in field_lower:
        return 'K'
    # ... (retained other guesses)
    return 'a.u.'

def calculate_derived_fields(fields: dict, field_info: dict) -> dict:
    derived = {}
    for key, (kind, dim) in field_info.items():
        if kind == "vector":
            derived[f"{key}_magnitude"] = np.linalg.norm(fields[key], axis=-1)
            if dim == 3:
                for i, comp in enumerate('xyz'):
                    derived[f"{key}_{comp}"] = fields[key][..., i]
    return derived

# =============================================
# DEBUG PANEL
# =============================================
# (Retained, added has_base_mesh in json)

def show_debug_panel(debug_info, directory_structure, load_status, simulations):
    with st.expander("🐛 Debug Information", expanded=False):
        tab1, tab2, tab3, tab4 = st.tabs(["Paths", "Directory", "Simulations", "Python Info"])
        # ... (retained content)
        with tab3:
            # ... 
            st.json({
                # ... retained
                'has_base_mesh': sim.get('base_mesh') is not None,
                # ...
            })

# =============================================
# TROUBLESHOOTING GUIDE
# =============================================
def show_troubleshooting_guide():
    with st.expander("🔧 Troubleshooting Guide", expanded=True):
        st.markdown("""
        ### Common Issues
        1. **PyVista/stpyvista Errors**: Install in order: `pip install pyvista stpyvista`. For cloud, add packages.txt with Xvfb.
        2. **No Files**: Ensure 'fea_solutions' has folders with VTU files.
        3. **Rendering Blank**: Check Xvfb init; test locally.
        4. **Large Meshes**: Downsample in Plotter.
        """)

# =============================================
# MAIN APPLICATION
# =============================================
def main():
    st.set_page_config(page_title="Enhanced FEA Viewer", layout="wide", page_icon="🔍")
    
    st.title("🔍 Enhanced FEA Laser Simulation Viewer")
    st.markdown("Interactive 3D mesh visualization with PyVista and stpyvista.")
    
    FEA_SOLUTIONS_DIR, debug_info = validate_and_setup_paths()
    directory_structure = scan_directory_structure(FEA_SOLUTIONS_DIR)
    
    st.sidebar.header("⚙️ Configuration")
    data_source = st.sidebar.radio("Data Source", ["Auto-detect", "Use Mock Data"])
    use_mock = data_source == "Use Mock Data"
    
    with st.sidebar.expander("📁 Directory Summary"):
        st.text(f"VTU files: {directory_structure.get('total_vtu_count', 0)}")
        if directory_structure.get('total_vtu_count', 0) > 0:
            st.success("✓ VTU files found!")
        else:
            st.error("✗ No VTU files found")
    
    global DEBUG_MODE
    DEBUG_MODE = st.sidebar.checkbox("Enable Debug Mode", value=True)
    
    with st.spinner("Loading data..."):
        simulations, load_status = load_simulations_with_fallback(use_mock_data=use_mock, debug_info=debug_info)
    
    if not simulations:
        show_troubleshooting_guide()
        if DEBUG_MODE:
            show_debug_panel(debug_info, directory_structure, load_status, simulations)
        return
    
    st.success(f"✅ Loaded {len(simulations)} simulation(s)")
    
    if DEBUG_MODE:
        show_debug_panel(debug_info, directory_structure, load_status, simulations)
    
    st.sidebar.header("🎯 Visualization")
    sim_names = sorted(simulations.keys())
    sim_name = st.sidebar.selectbox("Select Simulation", sim_names, format_func=lambda x: f"{x} (E={simulations[x]['energy_mJ']}mJ, D={simulations[x]['duration_ns']}ns)")
    sim = simulations[sim_name]
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Simulation Info")
    st.sidebar.metric("Energy", f"{sim['energy_mJ']:.2f} mJ")
    st.sidebar.metric("Duration", f"{sim['duration_ns']:.2f} ns")
    st.sidebar.metric("Points", f"{sim['points'].shape[0]:,}")
    st.sidebar.metric("Timesteps", sim['n_timesteps'])
    if sim['is_mock']:
        st.sidebar.warning("⚠️ Mock data")
    
    field_options = list(sim['fields'])
    field = st.sidebar.selectbox("Field", field_options)
    
    timestep = st.sidebar.slider("Timestep", 0, sim['n_timesteps'] - 1, 0)
    
    points = sim['points']
    field_data = sim['fields'][field]
    kind, _ = sim['field_info'].get(field, ('scalar', 1))
    
    if kind == 'vector' or '_magnitude' in field:
        values = np.linalg.norm(field_data[timestep], axis=1) if len(field_data.shape) > 2 else field_data[timestep]
        field_label = f"{field} (Magnitude)" if kind == 'vector' else field
    else:
        values = field_data[timestep]
        field_label = field
    
    unit = sim['units'].get(field, 'a.u.')
    
    st.header(f"{field_label} [{unit}] at t={timestep + 1}")
    st.caption(f"Simulation: {sim_name}")
    
    # Enhanced 3D Visualization with stpyvista
    base_mesh = sim['base_mesh']
    if base_mesh is None:
        st.warning("No base mesh; falling back to Plotly.")
        # Fallback Plotly code (retained from original)
        fig = go.Figure(data=go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='markers',
            marker=dict(size=3, color=values, colorscale='Viridis', opacity=0.8)
        ))
        fig.update_layout(height=600, scene=dict(aspectmode='data'))
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Update mesh with current data
        mesh = base_mesh.copy()
        if kind == 'scalar' or '_magnitude' in field:
            mesh.point_data[field_label] = values
            active_scalar = field_label
            vectors = None
        else:
            mesh.point_data[field] = field_data[timestep]
            active_scalar = None
            vectors = field
        
        # Setup Plotter
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, scalars=active_scalar, cmap='viridis', show_edges=True, opacity=0.85)
        
        if vectors:
            # Add glyphs for vectors
            arrows = mesh.glyph(orient=vectors, scale=vectors, factor=0.05)
            plotter.add_mesh(arrows, color='red')
        
        # Optional isosurface for scalars
        if active_scalar and st.sidebar.checkbox("Show Isosurface", False):
            contours = mesh.contour()
            plotter.add_mesh(contours, color='white', line_width=2)
        
        plotter.view_isometric()
        plotter.background_color = 'white'
        plotter.add_scalar_bar(title=f"{field_label} [{unit}]")
        
        # Render with stpyvista
        stpyvista(plotter, key=f"pv_{field}_{timestep}", use_trame=True)  # Trame for better cloud perf
        
    # Statistics
    clean_values = values[~np.isnan(values)]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Min", f"{clean_values.min():.3e}")
    col2.metric("Max", f"{clean_values.max():.3e}")
    col3.metric("Mean", f"{clean_values.mean():.3e}")
    col4.metric("Std", f"{clean_values.std():.3e}")
    
    # Additional tabs (retained, with enhancements if needed)
    tab1, tab2, tab3 = st.tabs(["📈 Distribution", "📊 2D Projection", "📋 Data Table"])
    with tab1:
        fig_hist = plt.figure(figsize=(10, 4))
        plt.hist(clean_values, bins=50)
        st.pyplot(fig_hist)
    with tab2:
        fig_2d = plt.figure(figsize=(10, 6))
        plt.scatter(points[:, 0], points[:, 1], c=values, cmap='viridis')
        st.pyplot(fig_2d)
    with tab3:
        df = pd.DataFrame({'X': points[:, 0], 'Y': points[:, 1], 'Z': points[:, 2], field_label: values})
        st.dataframe(df.head(100))
        csv = df.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, f"{sim_name}_{field}_t{timestep}.csv")

if __name__ == "__main__":
    main()
