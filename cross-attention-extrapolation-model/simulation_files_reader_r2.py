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

# =============================================
# DEBUG MODE & SETTINGS
# =============================================
DEBUG_MODE = True  # Set to True for debugging, False for production

# =============================================
# ALTERNATIVE VTU READER (FALLBACK IF PyVista NOT AVAILABLE)
# =============================================
class VTUReader:
    """Fallback VTU reader using XML parsing if PyVista is not available"""
    
    @staticmethod
    def try_import_pyvista():
        """Try to import pyvista with helpful error messages"""
        try:
            import pyvista as pv
            pv.set_jupyter_backend(None)
            return pv, None
        except ImportError as e:
            error_msg = """
            ### ⚠️ PyVista Installation Required
            
            PyVista is required for reading VTU files. Please install it:
            
            ```bash
            pip install pyvista
            ```
            
            **For Conda users:**
            ```bash
            conda install -c conda-forge pyvista
            ```
            
            **If you're in a restricted environment**, you can try:
            ```bash
            pip install pyvista --user
            ```
            
            **Alternative approach:** Convert VTU files to CSV/numpy format first.
            """
            return None, error_msg
    
    @staticmethod
    def read_vtu_fallback(vtu_path):
        """Simple fallback using numpy if structure is known"""
        # This is a placeholder - in production, you'd implement XML parsing
        # or use meshio as an alternative
        try:
            # Try using meshio as alternative
            import meshio
            mesh = meshio.read(vtu_path)
            return mesh
        except ImportError:
            st.warning("Neither pyvista nor meshio available. Install at least one.")
            return None
    
    @staticmethod
    def create_mock_data():
        """Create mock data for testing/demo purposes"""
        st.info("📊 Using mock data for demonstration")
        
        # Create a simple mesh grid
        n_points = 1000
        x = np.random.randn(n_points) * 10
        y = np.random.randn(n_points) * 10
        z = np.random.randn(n_points) * 5
        points = np.column_stack([x, y, z])
        
        # Create mock fields
        n_timesteps = 20
        fields = {}
        field_info = {}
        
        # Temperature field (scalar)
        r = np.sqrt(x**2 + y**2 + z**2)
        temperature = 300 + 1000 * np.exp(-r**2 / 50) * np.sin(np.linspace(0, 2*np.pi, n_timesteps))[:, None]
        fields['temperature'] = temperature
        field_info['temperature'] = ('scalar', 1)
        
        # Displacement field (vector)
        displacement = np.zeros((n_timesteps, n_points, 3))
        for t in range(n_timesteps):
            displacement[t, :, 0] = 0.1 * np.sin(0.1 * t) * x / (r + 1)
            displacement[t, :, 1] = 0.1 * np.sin(0.1 * t) * y / (r + 1)
            displacement[t, :, 2] = 0.2 * np.sin(0.1 * t) * np.exp(-z**2 / 100)
        fields['displacement'] = displacement
        field_info['displacement'] = ('vector', 3)
        
        # Stress field (scalar)
        stress = 1e6 + 5e6 * np.exp(-r**2 / 100) * np.cos(np.linspace(0, np.pi, n_timesteps))[:, None]
        fields['stress'] = stress
        field_info['stress'] = ('scalar', 1)
        
        return {
            'mock_1': {
                'energy_mJ': 0.5,
                'duration_ns': 4.2,
                'points': points,
                'fields': fields,
                'field_info': field_info,
                'units': {'temperature': 'K', 'displacement': 'm', 'stress': 'Pa'},
                'n_timesteps': n_timesteps,
                'timesteps': np.linspace(0, 4.2, n_timesteps),
                'folder': 'q0p5mJ-delta4p2ns',
                'is_mock': True
            },
            'mock_2': {
                'energy_mJ': 1.0,
                'duration_ns': 2.0,
                'points': points * 0.8,
                'fields': {k: v * 0.5 for k, v in fields.items()},
                'field_info': field_info,
                'units': {'temperature': 'K', 'displacement': 'm', 'stress': 'Pa'},
                'n_timesteps': n_timesteps,
                'timesteps': np.linspace(0, 2.0, n_timesteps),
                'folder': 'q1p0mJ-delta2p0ns',
                'is_mock': True
            }
        }

# =============================================
# PATH VALIDATION & DEBUG UTILITIES
# =============================================
def validate_and_setup_paths():
    """Validate paths and provide helpful debug information"""
    # Get current script directory
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Define FEA solutions directory
    FEA_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "fea_solutions")
    
    # Create debug information
    debug_info = {
        "script_directory": SCRIPT_DIR,
        "fea_solutions_dir": FEA_SOLUTIONS_DIR,
        "fea_dir_exists": os.path.exists(FEA_SOLUTIONS_DIR),
        "current_working_dir": os.getcwd(),
        "python_version": sys.version,
        "os": os.name
    }
    
    # Check if directory exists
    if not debug_info["fea_dir_exists"]:
        st.warning(f"""
        ### 📁 Directory Not Found
        
        The FEA solutions directory was not found at:
        ```
        {FEA_SOLUTIONS_DIR}
        ```
        
        **Possible solutions:**
        1. Create the directory manually:
           ```bash
           mkdir "{FEA_SOLUTIONS_DIR}"
           ```
        
        2. Place your VTU files in folders with names like:
           - `q0p5mJ-delta4p2ns/`
           - `q1p0mJ-delta2p0ns/`
           - `energy_5mJ-duration_10ns/`
        
        3. Check if your files are in a different location and update the path.
        """)
    
    # Create directory if it doesn't exist
    os.makedirs(FEA_SOLUTIONS_DIR, exist_ok=True)
    
    return FEA_SOLUTIONS_DIR, debug_info

def scan_directory_structure(base_dir):
    """Scan directory and return structure for debugging"""
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
            
            # Check directories
            for d in dirs:
                structure["directories"].append(os.path.join(rel_root, d))
            
            # Check files
            for f in files:
                full_path = os.path.join(root, f)
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
def parse_folder_name(folder: str):
    """
    Parse folder names with multiple pattern support:
    - q0p5mJ-delta4p2ns → (0.5, 4.2)
    - energy_5mJ-duration_10ns → (5.0, 10.0)
    - q1.0mJ-delta2.0ns → (1.0, 2.0)
    - sim_5mJ_10ns → (5.0, 10.0)
    """
    base_name = os.path.basename(folder)
    
    # Pattern 1: q0p5mJ-delta4p2ns
    match = re.match(r"q([\dp\.]+)mJ[-_]delta([\dp\.]+)ns", base_name, re.IGNORECASE)
    if match:
        e, d = match.groups()
        energy = float(e.replace("p", ".") if "p" in e else e)
        duration = float(d.replace("p", ".") if "p" in d else d)
        return energy, duration
    
    # Pattern 2: energy_5mJ-duration_10ns
    match = re.match(r"energy[_-]([\d\.]+)mJ[-_]duration[_-]([\d\.]+)ns", base_name, re.IGNORECASE)
    if match:
        e, d = match.groups()
        return float(e), float(d)
    
    # Pattern 3: q1.0mJ-delta2.0ns (with dots)
    match = re.match(r"q([\d\.]+)mJ[-_]delta([\d\.]+)ns", base_name, re.IGNORECASE)
    if match:
        e, d = match.groups()
        return float(e), float(d)
    
    # Pattern 4: sim_5mJ_10ns
    match = re.match(r"sim[_-]([\d\.]+)mJ[_-]([\d\.]+)ns", base_name, re.IGNORECASE)
    if match:
        e, d = match.groups()
        return float(e), float(d)
    
    # Pattern 5: Try to extract any numbers
    numbers = re.findall(r"[\d\.]+", base_name)
    if len(numbers) >= 2:
        try:
            return float(numbers[0]), float(numbers[1])
        except:
            pass
    
    # Return None if no pattern matches
    if DEBUG_MODE:
        st.sidebar.warning(f"Could not parse folder name: {base_name}")
    
    return None, None

# =============================================
# ROBUST DATA LOADER WITH DEBUGGING
# =============================================
@st.cache_data(show_spinner=False, ttl=3600)
def load_simulations_with_fallback(use_mock_data=False, debug_info=None):
    """
    Load simulations with multiple fallback strategies
    """
    # Option 1: Use mock data for demo
    if use_mock_data:
        return VTUReader.create_mock_data(), "mock"
    
    # Option 2: Try to load real data
    try:
        # Try to import pyvista
        pyvista, error_msg = VTUReader.try_import_pyvista()
        if pyvista is None:
            if debug_info:
                debug_info["pyvista_available"] = False
                debug_info["pyvista_error"] = error_msg
            return {}, "no_pyvista"
        
        if debug_info:
            debug_info["pyvista_available"] = True
        
        # Get FEA solutions directory
        FEA_SOLUTIONS_DIR = debug_info["fea_solutions_dir"] if debug_info else os.path.join(os.path.dirname(__file__), "fea_solutions")
        
        # Scan for folders
        folders = []
        
        # Look for pattern-based folders
        pattern_folders = glob.glob(os.path.join(FEA_SOLUTIONS_DIR, "q*mJ-delta*ns"))
        folders.extend(pattern_folders)
        
        # Also look for any folder containing VTU files
        all_folders = [f for f in glob.glob(os.path.join(FEA_SOLUTIONS_DIR, "*")) 
                      if os.path.isdir(f)]
        
        # Add folders that have VTU files but weren't caught by pattern
        for folder in all_folders:
            if folder not in folders:
                vtu_files = glob.glob(os.path.join(folder, "*.vtu"))
                if vtu_files:
                    folders.append(folder)
        
        if debug_info:
            debug_info["found_folders"] = folders
            debug_info["folder_count"] = len(folders)
        
        if not folders:
            return {}, "no_folders"
        
        # Load simulations
        simulations = {}
        loaded_count = 0
        
        for folder in folders:
            try:
                folder_name = os.path.basename(folder)
                
                # Parse folder name
                energy, duration = parse_folder_name(folder_name)
                
                # Find VTU files
                vtu_files = sorted(glob.glob(os.path.join(folder, "*.vtu")))
                if not vtu_files:
                    if DEBUG_MODE:
                        st.sidebar.warning(f"No VTU files in {folder_name}")
                    continue
                
                # Read first file to get structure
                mesh0 = pyvista.read(vtu_files[0])
                
                if not mesh0.point_data:
                    if DEBUG_MODE:
                        st.sidebar.warning(f"No point data in {folder_name}")
                    continue
                
                points = mesh0.points
                n_pts = mesh0.n_points
                n_steps = len(vtu_files)
                
                fields = {}
                field_info = {}
                units = {}
                
                # Detect field types
                for key in mesh0.point_data.keys():
                    arr = np.asarray(mesh0.point_data[key])
                    if arr.ndim == 1:
                        field_type = "scalar"
                        dim = 1
                    else:
                        field_type = "vector"
                        dim = arr.shape[1]
                    
                    # Guess unit
                    unit = guess_unit(key)
                    
                    field_info[key] = (field_type, dim)
                    units[key] = unit
                    
                    if field_type == "scalar":
                        fields[key] = np.full((n_steps, n_pts), np.nan, dtype=np.float32)
                    else:
                        fields[key] = np.full((n_steps, n_pts, dim), np.nan, dtype=np.float32)
                
                # Load all timesteps
                for t, vtu in enumerate(vtu_files[:50]):  # Limit to 50 timesteps for performance
                    try:
                        mesh = pyvista.read(vtu)
                        for key, (kind, _) in field_info.items():
                            if key in mesh.point_data:
                                data = np.asarray(mesh.point_data[key], dtype=np.float32)
                                fields[key][t] = data
                    except Exception as e:
                        if DEBUG_MODE:
                            st.sidebar.warning(f"Error reading {vtu}: {str(e)[:100]}")
                        continue
                
                # Create derived fields
                derived = calculate_derived_fields(fields, field_info)
                fields.update(derived)
                
                # Store simulation data
                simulations[folder_name] = {
                    'energy_mJ': energy if energy is not None else 0.0,
                    'duration_ns': duration if duration is not None else 0.0,
                    'points': points,
                    'fields': fields,
                    'field_info': field_info,
                    'units': units,
                    'n_timesteps': n_steps,
                    'timesteps': np.linspace(0, duration if duration else 10.0, n_steps),
                    'folder': folder_name,
                    'is_mock': False
                }
                
                loaded_count += 1
                
                if DEBUG_MODE:
                    st.sidebar.success(f"✓ Loaded {folder_name}")
                
            except Exception as e:
                if DEBUG_MODE:
                    st.sidebar.error(f"Error loading {folder}: {str(e)[:100]}")
                continue
        
        if debug_info:
            debug_info["loaded_simulations"] = loaded_count
        
        return simulations, "success" if loaded_count > 0 else "no_valid_data"
    
    except Exception as e:
        if debug_info:
            debug_info["load_error"] = str(e)
        return {}, f"error: {str(e)[:100]}"

def guess_unit(field_name: str) -> str:
    """Guess unit based on field name"""
    field_lower = field_name.lower()
    if 'temp' in field_lower:
        return 'K'
    elif 'heat' in field_lower or 'flux' in field_lower:
        return 'W/m²'
    elif 'stress' in field_lower:
        return 'Pa'
    elif 'strain' in field_lower:
        return ''
    elif 'disp' in field_lower:
        return 'm'
    elif 'vel' in field_lower:
        return 'm/s'
    elif 'force' in field_lower:
        return 'N'
    elif 'pressure' in field_lower:
        return 'Pa'
    elif 'energy' in field_lower:
        return 'J'
    elif 'power' in field_lower:
        return 'W'
    else:
        return 'a.u.'

def calculate_derived_fields(fields: dict, field_info: dict) -> dict:
    """Calculate derived fields"""
    derived = {}
    
    for key, (kind, dim) in field_info.items():
        if kind == "vector":
            # Calculate magnitude
            derived[f"{key}_magnitude"] = np.linalg.norm(fields[key], axis=-1)
            
            # Calculate components for 3D vectors
            if dim == 3:
                derived[f"{key}_x"] = fields[key][..., 0]
                derived[f"{key}_y"] = fields[key][..., 1]
                derived[f"{key}_z"] = fields[key][..., 2]
    
    return derived

# =============================================
# DEBUG PANEL
# =============================================
def show_debug_panel(debug_info, directory_structure, load_status, simulations):
    """Display debug information in an expandable panel"""
    with st.expander("🐛 Debug Information", expanded=False):
        tab1, tab2, tab3, tab4 = st.tabs(["Paths", "Directory", "Simulations", "Python Info"])
        
        with tab1:
            st.subheader("Path Information")
            st.json(debug_info)
            
            st.subheader("Current Files")
            for key, value in debug_info.items():
                st.text(f"{key}: {value}")
        
        with tab2:
            st.subheader("Directory Structure")
            st.text(f"Total VTU files found: {directory_structure.get('total_vtu_count', 0)}")
            
            if directory_structure.get('directories'):
                st.subheader("Directories")
                for d in directory_structure['directories'][:20]:  # Limit display
                    st.text(f"  📁 {d}")
            
            if directory_structure.get('vtu_files'):
                st.subheader("VTU Files (first 20)")
                for v in directory_structure['vtu_files'][:20]:
                    st.text(f"  📄 {v}")
            
            if 'error' in directory_structure:
                st.error(f"Scan error: {directory_structure['error']}")
        
        with tab3:
            st.subheader("Simulation Load Status")
            st.code(f"Load status: {load_status}")
            st.code(f"Number of simulations loaded: {len(simulations)}")
            
            if simulations:
                st.subheader("Loaded Simulations")
                for name, sim in list(simulations.items())[:5]:  # Show first 5
                    with st.expander(f"Simulation: {name}"):
                        st.json({
                            'energy_mJ': sim.get('energy_mJ'),
                            'duration_ns': sim.get('duration_ns'),
                            'points_shape': sim.get('points', []).shape,
                            'fields': list(sim.get('fields', {}).keys()),
                            'timesteps': sim.get('n_timesteps', 0),
                            'is_mock': sim.get('is_mock', False)
                        })
        
        with tab4:
            st.subheader("Python Environment")
            try:
                import importlib.metadata
                packages = ['streamlit', 'numpy', 'pandas', 'plotly', 'matplotlib']
                for pkg in packages:
                    try:
                        version = importlib.metadata.version(pkg)
                        st.text(f"{pkg}: {version}")
                    except:
                        st.text(f"{pkg}: Not installed")
            except:
                st.text("Could not retrieve package versions")

# =============================================
# FILE UPLOADER (ALTERNATIVE TO FIXED DIRECTORY)
# =============================================
def handle_file_upload():
    """Handle file upload as alternative to fixed directory"""
    st.sidebar.header("📤 Alternative: Upload Files")
    
    uploaded_files = st.sidebar.file_uploader(
        "Upload VTU files (alternative to directory)",
        type=['vtu', 'vtk', 'csv', 'npy'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} files uploaded")
        # Here you would implement file processing
        # For now, we'll just show what was uploaded
        for file in uploaded_files:
            st.sidebar.text(f"  • {file.name} ({file.size/1024:.1f} KB)")
        
        return True
    
    return False

# =============================================
# TROUBLESHOOTING GUIDE
# =============================================
def show_troubleshooting_guide():
    """Display troubleshooting guide"""
    with st.expander("🔧 Troubleshooting Guide", expanded=True):
        st.markdown("""
        ### Common Issues and Solutions
        
        #### 1. **PyVista Not Installed**
        ```bash
        # Install PyVista
        pip install pyvista
        
        # Or with conda
        conda install -c conda-forge pyvista
        
        # If you have permission issues
        pip install --user pyvista
        ```
        
        #### 2. **No VTU Files Found**
        - Check that your files are in `fea_solutions/` directory
        - Ensure folder names follow patterns like:
          - `q0p5mJ-delta4p2ns/`
          - `energy_5mJ-duration_10ns/`
          - `q1.0mJ-delta2.0ns/`
        - Each folder should contain `.vtu` files
        
        #### 3. **Directory Structure Example**
        ```
        your_project/
        ├── app.py
        └── fea_solutions/
            ├── q0p5mJ-delta4p2ns/
            │   ├── a_t0001.vtu
            │   ├── a_t0002.vtu
            │   └── ...
            └── q1p0mJ-delta2p0ns/
                ├── a_t0001.vtu
                └── ...
        ```
        
        #### 4. **Quick Test with Mock Data**
        - Check the "Use Mock Data" option in the sidebar
        - This will generate sample data for testing the visualization
        
        #### 5. **Check File Permissions**
        - Ensure Python has read access to the files
        - Check file paths don't contain special characters
        
        #### 6. **Alternative Libraries**
        If PyVista doesn't work, try:
        ```bash
        pip install meshio
        ```
        Then modify the code to use meshio instead.
        """)

# =============================================
# MAIN APPLICATION
# =============================================
def main():
    st.set_page_config(
        page_title="FEA Viewer with Debug Tools",
        layout="wide",
        page_icon="🔍"
    )
    
    # Title and description
    st.title("🔍 FEA Laser Simulation Viewer")
    st.markdown("""
    Visualize and analyze FEA simulation results from VTU files.
    Includes debugging tools to help identify issues.
    """)
    
    # Initialize debug information
    FEA_SOLUTIONS_DIR, debug_info = validate_and_setup_paths()
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Data Source",
        ["Auto-detect", "Use Mock Data", "Manual Setup"],
        help="Choose where to get simulation data from"
    )
    
    use_mock = False
    if data_source == "Use Mock Data":
        use_mock = True
        st.sidebar.info("Using mock data for demonstration")
    
    # Scan directory structure
    directory_structure = scan_directory_structure(FEA_SOLUTIONS_DIR)
    
    # Show directory summary
    with st.sidebar.expander("📁 Directory Summary", expanded=False):
        st.text(f"VTU files: {directory_structure.get('total_vtu_count', 0)}")
        st.text(f"Folders: {len(directory_structure.get('directories', []))}")
        st.text(f"Other files: {len(directory_structure.get('other_files', []))}")
        
        if directory_structure.get('total_vtu_count', 0) > 0:
            st.success("✓ VTU files found!")
        else:
            st.error("✗ No VTU files found")
    
    # Debug mode toggle
    global DEBUG_MODE
    DEBUG_MODE = st.sidebar.checkbox("Enable Debug Mode", value=True)
    
    # Load simulations
    load_status = "initializing"
    simulations = {}
    
    with st.spinner("Loading simulation data..."):
        if data_source == "Manual Setup":
            st.info("Please configure manual setup in the code")
            simulations = {}
            load_status = "manual"
        else:
            simulations, load_status = load_simulations_with_fallback(
                use_mock_data=use_mock,
                debug_info=debug_info
            )
    
    # Show troubleshooting guide if no data
    if not simulations:
        show_troubleshooting_guide()
        
        # Show debug panel
        if DEBUG_MODE:
            show_debug_panel(debug_info, directory_structure, load_status, simulations)
        
        # Provide next steps
        st.error("""
        ### ❌ No Simulations Loaded
        
        **Possible reasons:**
        1. PyVista is not installed
        2. No VTU files found in the expected directory
        3. Folder names don't match expected patterns
        4. File permissions issues
        
        **Immediate solutions:**
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Retry Loading"):
                st.rerun()
        
        with col2:
            if st.button("📁 Show Debug Info"):
                show_debug_panel(debug_info, directory_structure, load_status, simulations)
        
        with col3:
            if st.button("🧪 Use Mock Data"):
                st.session_state.use_mock = True
                st.rerun()
        
        # Installation instructions
        st.markdown("---")
        st.subheader("📦 Installation Commands")
        
        install_tab1, install_tab2 = st.tabs(["pip", "conda"])
        
        with install_tab1:
            st.code("""
            # Install all required packages
            pip install streamlit numpy pandas plotly matplotlib pyvista
            
            # Or install individually
            pip install pyvista
            """)
        
        with install_tab2:
            st.code("""
            # Create a new conda environment
            conda create -n fea-viewer python=3.9
            conda activate fea-viewer
            
            # Install packages
            conda install -c conda-forge streamlit numpy pandas plotly matplotlib pyvista
            """)
        
        return
    
    # Show success message
    st.success(f"✅ Successfully loaded {len(simulations)} simulation(s)")
    
    # Show debug panel if enabled
    if DEBUG_MODE:
        show_debug_panel(debug_info, directory_structure, load_status, simulations)
    
    # Main application interface
    st.sidebar.header("🎯 Visualization")
    
    # Simulation selection
    sim_names = sorted(simulations.keys())
    sim_name = st.sidebar.selectbox(
        "Select Simulation",
        sim_names,
        format_func=lambda x: f"{x} (E={simulations[x].get('energy_mJ', '?')}mJ, D={simulations[x].get('duration_ns', '?')}ns)"
    )
    
    sim = simulations[sim_name]
    
    # Show simulation info
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Simulation Info")
    st.sidebar.metric("Energy", f"{sim.get('energy_mJ', 0):.2f} mJ")
    st.sidebar.metric("Duration", f"{sim.get('duration_ns', 0):.2f} ns")
    st.sidebar.metric("Points", f"{sim.get('points', np.array([])).shape[0]:,}")
    st.sidebar.metric("Timesteps", sim.get('n_timesteps', 0))
    
    if sim.get('is_mock', False):
        st.sidebar.warning("⚠️ Using mock data")
    
    # Field selection
    field_options = list(sim.get('fields', {}).keys())
    if not field_options:
        st.error("No fields found in simulation")
        return
    
    field = st.sidebar.selectbox("Field", field_options)
    
    # Timestep selection
    n_steps = sim.get('n_timesteps', 1)
    timestep = st.sidebar.slider("Timestep", 0, max(0, n_steps-1), 0)
    
    # Get data for visualization
    points = sim.get('points', np.array([]))
    field_data = sim.get('fields', {}).get(field, np.array([]))
    
    if field_data.size == 0:
        st.error(f"No data for field '{field}'")
        return
    
    # Handle scalar vs vector data
    if len(field_data.shape) == 3:  # Vector field
        values = np.linalg.norm(field_data[timestep], axis=1)
        field_label = f"{field} (Magnitude)"
    else:  # Scalar field
        values = field_data[timestep] if timestep < field_data.shape[0] else field_data[0]
        field_label = field
    
    unit = sim.get('units', {}).get(field, 'a.u.')
    
    # Main visualization
    st.header(f"{field_label} [{unit}]")
    st.caption(f"Timestep {timestep+1}/{n_steps} | {sim_name}")
    
    # Create 3D plot
    fig = go.Figure(data=go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=values,
            colorscale='Viridis',
            colorbar=dict(title=f"{field_label} [{unit}]"),
            opacity=0.8,
            showscale=True
        ),
        hovertemplate=(
            f"<b>{field_label}</b>: %{{color:.3e}} {unit}<br>"
            "X: %{x:.3e}<br>"
            "Y: %{y:.3e}<br>"
            "Z: %{z:.3e}<br>"
            "<extra></extra>"
        )
    ))
    
    fig.update_layout(
        height=600,
        scene=dict(
            aspectmode='data',
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(
            text=f"3D Visualization: {field_label}",
            x=0.5
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics panel
    col1, col2, col3, col4 = st.columns(4)
    
    clean_values = values[~np.isnan(values)]
    
    with col1:
        st.metric("Min", f"{clean_values.min():.3e}")
    with col2:
        st.metric("Max", f"{clean_values.max():.3e}")
    with col3:
        st.metric("Mean", f"{clean_values.mean():.3e}")
    with col4:
        st.metric("Std", f"{clean_values.std():.3e}")
    
    # Additional visualizations
    tab1, tab2, tab3 = st.tabs(["📈 Distribution", "📊 2D Projection", "📋 Data Table"])
    
    with tab1:
        # Histogram
        fig_hist, ax = plt.subplots(figsize=(10, 4))
        ax.hist(clean_values, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel(f"{field_label} [{unit}]")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Distribution of {field_label}")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig_hist)
    
    with tab2:
        # 2D projection
        fig_2d, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(
            points[:, 0],
            points[:, 1],
            c=values,
            s=10,
            cmap='viridis',
            alpha=0.7
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"2D Projection (XY plane)")
        plt.colorbar(scatter, ax=ax, label=f"{field_label} [{unit}]")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig_2d)
    
    with tab3:
        # Data table
        df = pd.DataFrame({
            'X': points[:, 0],
            'Y': points[:, 1],
            'Z': points[:, 2],
            field_label: values
        })
        st.dataframe(df.head(100), use_container_width=True)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"{sim_name}_{field}_t{timestep}.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.caption(f"Loaded {len(simulations)} simulations | Data source: {load_status}")

# =============================================
# ENTRY POINT
# =============================================
if __name__ == "__main__":
    main()
