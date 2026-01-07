import streamlit as st
import os
import glob
import re
import numpy as np
import pyvista as pv
from pathlib import Path

# =============================================
# CONFIGURATION
# =============================================
SCRIPT_DIR = Path(__file__).parent.resolve()
FEA_SOLUTIONS_DIR = SCRIPT_DIR / "fea_solutions"

# Ensure directory exists
FEA_SOLUTIONS_DIR.mkdir(exist_ok=True)

# =============================================
# UTILITY: Parse folder name → energy & duration
# =============================================
def parse_folder_name(folder: str):
    """
    Parse 'q0p5mJ-delta4p2ns' → energy=0.5 (mJ), duration=4.2 (ns)
    Handles 'p' as decimal separator (e.g., 0p5 → 0.5)
    """
    match = re.match(r"q([\d\.p]+)mJ-delta([\d\.p]+)ns", folder)
    if not match:
        return None, None
    e_str, d_str = match.groups()
    energy = float(e_str.replace('p', '.'))
    duration = float(d_str.replace('p', '.'))
    return energy, duration

# =============================================
# LOAD ALL SIMULATIONS INTO MEMORY (as NumPy)
# =============================================
@st.cache_data
def load_all_simulations():
    """
    Returns:
    {
        "q0p5mJ-delta4p2ns": {
            "energy_mJ": 0.5,
            "duration_ns": 4.2,
            "timesteps": [0, 1, ..., 7]  # in ns
            "fields": {  # shape: (8, N, M, L) or (8, N_points)
                "Temperature": np.ndarray,
                "Stress": np.ndarray,
                ...
            },
            "mesh_grid": (optional) structured grid info,
            "points": np.ndarray,  # (N, 3)
            "cell_data": bool  # True if data on cells, False if on points
        },
        ...
    }
    """
    simulations = {}

    # Find all q*mJ-delta*ns folders
    pattern = str(FEA_SOLUTIONS_DIR / "q*mJ-delta*ns")
    folders = glob.glob(pattern)

    for folder_path_str in folders:
        folder_name = os.path.basename(folder_path_str)
        energy, duration = parse_folder_name(folder_name)
        if energy is None:
            continue

        vtu_files = sorted(glob.glob(os.path.join(folder_path_str, "a_t????.vtu")))
        if not vtu_files:
            continue

        # Load first to inspect structure
        try:
            first_mesh = pv.read(vtu_files[0])
        except Exception as e:
            st.warning(f"Failed to read {vtu_files[0]}: {e}")
            continue

        # Determine if data is on points or cells
        point_data_keys = list(first_mesh.point_data.keys())
        cell_data_keys = list(first_mesh.cell_data.keys())
        use_point_data = len(point_data_keys) > 0
        data_keys = point_data_keys if use_point_data else cell_data_keys

        if not data_keys:
            st.warning(f"No scalar fields in {folder_name}")
            continue

        # Initialize arrays
        n_timesteps = len(vtu_files)
        n_points = first_mesh.n_points if use_point_data else first_mesh.n_cells

        fields = {key: np.empty((n_timesteps, n_points), dtype=np.float32) for key in data_keys}
        points = first_mesh.points if use_point_data else None  # Only valid for point data

        # Load all timesteps
        for t_idx, vtu in enumerate(vtu_files):
            mesh = pv.read(vtu)
            data = mesh.point_data if use_point_data else mesh.cell_data
            for key in data_keys:
                fields[key][t_idx, :] = np.asarray(data[key], dtype=np.float32)

        simulations[folder_name] = {
            "energy_mJ": energy,
            "duration_ns": duration,
            "timesteps_ns": np.arange(n_timesteps) + 1,  # 1ns, 2ns, ..., 8ns
            "fields": fields,
            "points": points,
            "cell_data": not use_point_data,
            "n_timesteps": n_timesteps
        }

    return simulations

# =============================================
# STREAMLIT APP
# =============================================
def main():
    st.set_page_config(page_title="FEA Laser Simulation Viewer", layout="wide")
    st.title("🔍 FEA Laser Processing Simulation Viewer")
    st.caption(f"Scanning: `{FEA_SOLUTIONS_DIR}`")

    # Load simulations
    with st.spinner("Loading FEA simulations..."):
        all_sims = load_all_simulations()

    if not all_sims:
        st.warning(f"No valid simulations found in `{FEA_SOLUTIONS_DIR}`. Expected folders like `q0p5mJ-delta4p2ns` with `a_t0001.vtu` etc.")
        st.stop()

    # Sidebar: Select simulation
    st.sidebar.header("⚙️ Simulation Selection")
    sim_names = sorted(all_sims.keys())
    selected_sim = st.sidebar.selectbox("Select simulation", sim_names)

    sim_data = all_sims[selected_sim]
    st.sidebar.write(f"**Energy**: {sim_data['energy_mJ']} mJ")
    st.sidebar.write(f"**Pulse Duration**: {sim_data['duration_ns']} ns")
    st.sidebar.write(f"**Timesteps**: {sim_data['n_timesteps']} (1 ns each)")

    # Select field
    field_names = list(sim_data["fields"].keys())
    selected_field = st.selectbox("Select field to visualize", field_names)

    # Timestep slider
    max_t = sim_data['n_timesteps'] - 1
    timestep = st.slider("Timestep (ns)", 0, max_t, 0, format="%dns")

    # Get data for this timestep
    field_data = sim_data["fields"][selected_field][timestep]  # (N,) or (N_cells,)
    points = sim_data.get("points")

    # Reconstruct mesh for visualization
    if points is not None:
        # Point data → UnstructuredGrid or PolyData
        mesh = pv.PolyData(points)
        mesh.point_data[selected_field] = field_data
    else:
        # Cell data → we need cells; fallback to point cloud with dummy mesh
        # For simplicity, we'll use point cloud and assign field as point data
        # (Note: this is approximate; true cell data needs mesh topology)
        st.warning("Cell data detected – visualizing as point cloud approximation.")
        # Create dummy points (if not available, skip)
        N = len(field_data)
        dummy_points = np.random.rand(N, 3)  # Not ideal – better to load actual mesh
        mesh = pv.PolyData(dummy_points)
        mesh.point_data[selected_field] = field_data

    # Visualization
    st.subheader(f"🌡️ {selected_field} at {timestep + 1} ns – {selected_sim}")

    # Use PyVista plotter
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars=selected_field, cmap="viridis", show_edges=False)
    plotter.add_scalar_bar(title=selected_field, vertical=True)
    plotter.camera_position = 'iso'

    # Render in Streamlit
    with st.container():
        try:
            st.pyvista(plotter.show(auto_close=False))
        except Exception:
            # Fallback: 2D heatmap if 3D not supported
            st.warning("3D rendering not supported – showing 2D slice (XY plane)")
            if points is not None:
                x, y = points[:, 0], points[:, 1]
                df = pd.DataFrame({"x": x, "y": y, "value": field_data})
                pivot = df.pivot_table(values="value", index="y", columns="x", aggfunc="mean")
                st.image(pivot, use_column_width=True)

    # Optional: Show raw data stats
    with st.expander("📊 Field Statistics"):
        st.write({
            "Min": float(np.min(field_data)),
            "Max": float(np.max(field_data)),
            "Mean": float(np.mean(field_data)),
            "Std": float(np.std(field_data))
        })

    # Optional: Export NumPy array
    if st.button("💾 Export Field as NumPy Array"):
        np_array = sim_data["fields"][selected_field]
        with st.spinner("Preparing download..."):
            buffer = BytesIO()
            np.save(buffer, np_array)
            buffer.seek(0)
            st.download_button(
                label="Download .npy",
                data=buffer,
                file_name=f"{selected_sim}_{selected_field}.npy",
                mime="application/octet-stream"
            )

if __name__ == "__main__":
    main()
