import streamlit as st
import os
import glob
import re
import numpy as np
import pyvista as pv
from pathlib import Path

# =============================================
# PATH CONFIGURATION (same style as reference)
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEA_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "fea_solutions")

# Create directory if it doesn't exist
if not os.path.exists(FEA_SOLUTIONS_DIR):
    os.makedirs(FEA_SOLUTIONS_DIR, exist_ok=True)
    st.info(f"📁 Created fea_solutions directory at: {FEA_SOLUTIONS_DIR}")

# =============================================
# PATTERN UTILS
# =============================================
def parse_folder_name(folder: str):
    """
    Parse 'q0p5mJ-delta4p2ns' → energy=0.5 (mJ), duration=4.2 (ns)
    Handles 'p' as decimal separator.
    """
    match = re.match(r"q([\d\.p]+)mJ-delta([\d\.p]+)ns", folder)
    if not match:
        return None, None
    e_str, d_str = match.groups()
    try:
        energy = float(e_str.replace('p', '.'))
        duration = float(d_str.replace('p', '.'))
        return energy, duration
    except ValueError:
        return None, None

# =============================================
# LOAD ALL FEA SIMULATIONS
# =============================================
@st.cache_data
def load_all_simulations():
    """
    Returns a dictionary of simulation data with fields as NumPy arrays.
    Each simulation key: folder name like 'q0p5mJ-delta4p2ns'
    """
    simulations = {}
    pattern = os.path.join(FEA_SOLUTIONS_DIR, "q*mJ-delta*ns")
    folders = glob.glob(pattern)

    for folder_path in folders:
        folder_name = os.path.basename(folder_path)
        energy, duration = parse_folder_name(folder_name)
        if energy is None:
            continue

        # Get all .vtu files: a_t0001.vtu ... a_t0008.vtu
        vtu_files = sorted(glob.glob(os.path.join(folder_path, "a_t????.vtu")))
        if not vtu_files:
            continue

        try:
            first_mesh = pv.read(vtu_files[0])
        except Exception as e:
            st.warning(f"⚠️ Skipping {folder_name}: failed to read first VTU file – {e}")
            continue

        # Determine data location: point vs cell
        point_keys = list(first_mesh.point_data.keys())
        cell_keys = list(first_mesh.cell_data.keys())

        if not point_keys and not cell_keys:
            st.warning(f"⚠️ Skipping {folder_name}: no scalar data found in VTU files.")
            continue

        use_point_data = len(point_keys) >= len(cell_keys)
        data_keys = point_keys if use_point_data else cell_keys
        n_timesteps = len(vtu_files)
        n_entities = first_mesh.n_points if use_point_data else first_mesh.n_cells

        # Initialize field arrays
        fields = {key: np.empty((n_timesteps, n_entities), dtype=np.float32) for key in data_keys}

        # Load each timestep
        for t_idx, vtu in enumerate(vtu_files):
            try:
                mesh = pv.read(vtu)
                data_container = mesh.point_data if use_point_data else mesh.cell_data

                for key in data_keys:
                    if key not in data_container:
                        st.warning(f"⚠️ Field '{key}' missing in {vtu}. Filling with NaN.")
                        fields[key][t_idx, :] = np.nan
                        continue

                    raw_data = np.asarray(data_container[key], dtype=np.float32)
                    if raw_data.ndim > 1:
                        raw_data = raw_data.flatten()  # e.g., vector → scalar magnitude?
                    if raw_data.size != n_entities:
                        st.warning(f"⚠️ Size mismatch for '{key}' in {vtu}. Expected {n_entities}, got {raw_data.size}. Truncating/padding.")
                        raw_data = raw_data[:n_entities] if raw_data.size >= n_entities else np.pad(raw_data, (0, n_entities - raw_data.size), constant_values=np.nan)
                    fields[key][t_idx, :] = raw_data

            except Exception as e:
                st.error(f"❌ Error reading {vtu}: {e}")
                for key in data_keys:
                    fields[key][t_idx, :] = np.nan

        # Store
        simulations[folder_name] = {
            "energy_mJ": energy,
            "duration_ns": duration,
            "timesteps_ns": np.arange(1, n_timesteps + 1),  # 1ns, 2ns, ..., 8ns
            "fields": fields,
            "points": first_mesh.points if use_point_data else None,
            "cell_data": not use_point_data,
            "n_timesteps": n_timesteps,
            "n_entities": n_entities,
            "use_point_data": use_point_data
        }

    return simulations

# =============================================
# STREAMLIT APP
# =============================================
def main():
    st.set_page_config(page_title="FEA Laser Simulation Viewer", layout="wide")
    st.title("🔍 FEA Laser Simulation Viewer (.vtu)")
    st.caption(f"Scanning: `{FEA_SOLUTIONS_DIR}`")

    with st.spinner("Loading FEA simulations..."):
        all_sims = load_all_simulations()

    if not all_sims:
        st.warning(
            f"No valid simulations found in `{FEA_SOLUTIONS_DIR}`.\n"
            "Expected subfolders like `q0p5mJ-delta4p2ns` containing `a_t0001.vtu`, ..., `a_t0008.vtu`."
        )
        st.stop()

    # Sidebar selection
    st.sidebar.header("⚙️ Simulation")
    sim_names = sorted(all_sims.keys())
    selected_sim = st.sidebar.selectbox("Select simulation", sim_names)
    sim_data = all_sims[selected_sim]

    st.sidebar.write(f"**Energy**: {sim_data['energy_mJ']} mJ")
    st.sidebar.write(f"**Pulse Duration**: {sim_data['duration_ns']} ns")
    st.sidebar.write(f"**Timesteps**: {sim_data['n_timesteps']} (1 ns each)")

    # Field selection
    field_names = list(sim_data["fields"].keys())
    if not field_names:
        st.error("No scalar fields available in selected simulation.")
        st.stop()
    selected_field = st.selectbox("Select field to visualize", field_names)

    # Timestep slider
    timestep = st.slider(
        "Timestep (ns)",
        0,
        sim_data["n_timesteps"] - 1,
        0,
        format="%dns"
    )

    # Get data
    field_values = sim_data["fields"][selected_field][timestep]  # (N,)
    points = sim_data.get("points")

    # Reconstruct mesh for rendering
    if points is not None and len(points) == len(field_values):
        mesh = pv.PolyData(points)
        mesh.point_data[selected_field] = field_values
    else:
        # Fallback: use points from first mesh or dummy
        N = len(field_values)
        if points is not None and len(points) != N:
            st.warning("Point count mismatch – using dummy point cloud.")
        dummy_points = np.random.rand(N, 3) if N > 0 else np.zeros((1, 3))
        mesh = pv.PolyData(dummy_points)
        mesh.point_data[selected_field] = field_values if N > 0 else np.array([0.0])

    # Visualization
    st.subheader(f"{selected_field} at {timestep + 1} ns – {selected_sim}")

    plotter = pv.Plotter()
    plotter.add_mesh(
        mesh,
        scalars=selected_field,
        cmap="viridis",
        nan_color="gray",
        show_edges=False,
        clim=[np.nanmin(field_values), np.nanmax(field_values)]
    )
    plotter.add_scalar_bar(title=selected_field, vertical=True)
    plotter.camera_position = 'iso'

    try:
        st.pyvista(plotter.show(auto_close=False))
    except Exception:
        st.warning("3D rendering failed – ensure PyVista is supported in your environment.")

    # Statistics
    with st.expander("📊 Field Statistics"):
        clean_vals = field_values[~np.isnan(field_values)]
        if len(clean_vals) > 0:
            st.write({
                "Min": float(np.min(clean_vals)),
                "Max": float(np.max(clean_vals)),
                "Mean": float(np.mean(clean_vals)),
                "Std": float(np.std(clean_vals))
            })
        else:
            st.write("⚠️ All values are NaN.")

if __name__ == "__main__":
    main()
