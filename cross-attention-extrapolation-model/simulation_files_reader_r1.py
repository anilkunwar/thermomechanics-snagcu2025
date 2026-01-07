import streamlit as st
import os
import glob
import re
import numpy as np
import pyvista as pv
from pathlib import Path

# =====================================================
# PYVISTA SAFE MODE (CRITICAL FOR STREAMLIT CLOUD)
# =====================================================
pv.OFF_SCREEN = True

# =============================================
# PATH CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEA_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "fea_solutions")
os.makedirs(FEA_SOLUTIONS_DIR, exist_ok=True)

# =============================================
# PATTERN UTILS
# =============================================
def parse_folder_name(folder: str):
    match = re.match(r"q([\d\.p]+)mJ-delta([\d\.p]+)ns", folder)
    if not match:
        return None, None
    e_str, d_str = match.groups()
    try:
        return float(e_str.replace("p", ".")), float(d_str.replace("p", "."))
    except ValueError:
        return None, None

# =============================================
# LOAD ALL FEA SIMULATIONS (FIXED)
# =============================================
@st.cache_data
def load_all_simulations():
    simulations = {}
    folders = glob.glob(os.path.join(FEA_SOLUTIONS_DIR, "q*mJ-delta*ns"))

    for folder_path in folders:
        folder_name = os.path.basename(folder_path)
        energy, duration = parse_folder_name(folder_name)
        if energy is None:
            continue

        vtu_files = sorted(glob.glob(os.path.join(folder_path, "a_t????.vtu")))
        if not vtu_files:
            continue

        try:
            first_mesh = pv.read(vtu_files[0])
        except Exception as e:
            st.warning(f"⚠️ Skipping {folder_name}: {e}")
            continue

        # Prefer point data
        data_container_name = "point"
        data_keys = list(first_mesh.point_data.keys())
        n_entities = first_mesh.n_points

        if not data_keys:
            data_container_name = "cell"
            data_keys = list(first_mesh.cell_data.keys())
            n_entities = first_mesh.n_cells

        if not data_keys:
            continue

        n_timesteps = len(vtu_files)

        # Inspect field dimensionality
        field_info = {}
        for key in data_keys:
            arr = np.asarray(
                first_mesh.point_data[key]
                if data_container_name == "point"
                else first_mesh.cell_data[key]
            )
            if arr.ndim == 1:
                field_info[key] = ("scalar", 1)
            else:
                field_info[key] = ("vector", arr.shape[1])

        # Allocate storage
        fields = {}
        for key, (kind, ncomp) in field_info.items():
            if kind == "scalar":
                fields[key] = np.full((n_timesteps, n_entities), np.nan, dtype=np.float32)
            else:
                fields[key] = np.full(
                    (n_timesteps, n_entities, ncomp), np.nan, dtype=np.float32
                )

        # Load timesteps
        for t_idx, vtu in enumerate(vtu_files):
            try:
                mesh = pv.read(vtu)
                container = mesh.point_data if data_container_name == "point" else mesh.cell_data

                for key, (kind, _) in field_info.items():
                    if key not in container:
                        continue
                    arr = np.asarray(container[key], dtype=np.float32)

                    if kind == "scalar":
                        fields[key][t_idx, :] = arr
                    else:
                        fields[key][t_idx, :, :] = arr

            except Exception as e:
                st.error(f"❌ Error reading {vtu}: {e}")

        simulations[folder_name] = {
            "energy_mJ": energy,
            "duration_ns": duration,
            "timesteps_ns": np.arange(1, n_timesteps + 1),
            "fields": fields,
            "field_info": field_info,
            "points": first_mesh.points if data_container_name == "point" else None,
            "n_timesteps": n_timesteps,
            "n_entities": n_entities,
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
        st.warning("No valid simulations found.")
        st.stop()

    st.sidebar.header("⚙️ Simulation")
    sim_name = st.sidebar.selectbox("Select simulation", sorted(all_sims))
    sim = all_sims[sim_name]

    st.sidebar.write(f"**Energy:** {sim['energy_mJ']} mJ")
    st.sidebar.write(f"**Pulse:** {sim['duration_ns']} ns")

    # Field selection
    field = st.selectbox("Select field", list(sim["fields"].keys()))
    kind, ncomp = sim["field_info"][field]

    timestep = st.slider("Timestep (ns)", 0, sim["n_timesteps"] - 1, 0)

    # Extract scalar for visualization
    if kind == "scalar":
        values = sim["fields"][field][timestep]
    else:
        values = np.linalg.norm(sim["fields"][field][timestep], axis=1)

    points = sim["points"]
    mesh = pv.PolyData(points)
    mesh.point_data[field] = values

    st.subheader(f"{field} at {timestep + 1} ns – {sim_name}")

    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(
        mesh,
        scalars=field,
        cmap="viridis",
        nan_color="gray",
        show_edges=False,
    )
    plotter.camera_position = "iso"
    plotter.add_scalar_bar(title=f"{field} ({'magnitude' if kind!='scalar' else 'scalar'})")

    st.pyvista(plotter)

    with st.expander("📊 Field Statistics"):
        clean = values[~np.isnan(values)]
        if clean.size:
            st.write(
                dict(
                    Min=float(clean.min()),
                    Max=float(clean.max()),
                    Mean=float(clean.mean()),
                    Std=float(clean.std()),
                )
            )
        else:
            st.write("All values are NaN.")

if __name__ == "__main__":
    main()
