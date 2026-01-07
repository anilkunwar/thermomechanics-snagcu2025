import streamlit as st
import os
import glob
import re
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import meshio  # Add to requirements.txt for Streamlit Cloud

# =============================================
# PATH CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEA_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "fea_solutions")
os.makedirs(FEA_SOLUTIONS_DIR, exist_ok=True)
# =============================================
# FOLDER NAME PARSER
# =============================================
def parse_folder_name(folder: str):
    """
    q0p5mJ-delta4p2ns → (0.5, 4.2)
    """
    match = re.match(r"q([\dp\.]+)mJ-delta([\dp\.]+)ns", folder)
    if not match:
        return None, None
    e, d = match.groups()
    return float(e.replace("p", ".")), float(d.replace("p", "."))
# =============================================
# LOAD FEA DATA (VTU → NUMPY, NO RENDERING)
# =============================================
@st.cache_data
def load_all_simulations():
    simulations = {}
    folders = glob.glob(os.path.join(FEA_SOLUTIONS_DIR, "q*mJ-delta*ns"))
    for folder in folders:
        name = os.path.basename(folder)
        energy, duration = parse_folder_name(name)
        if energy is None:
            continue
        vtu_files = sorted(glob.glob(os.path.join(folder, "a_t????.vtu")))
        if not vtu_files:
            continue
        mesh0 = meshio.read(vtu_files[0])
        if not mesh0.point_data:
            continue
        points = mesh0.points.astype(np.float32)
        n_pts = len(points)
        n_steps = len(vtu_files)
        fields = {}
        field_info = {}
        triangles = None
        for cell_block in mesh0.cells:
            if cell_block.type == "triangle":
                triangles = cell_block.data.astype(np.int32)
                break
        # Detect scalar vs vector
        for key in mesh0.point_data.keys():
            arr = mesh0.point_data[key].astype(np.float32)
            if arr.ndim == 1:
                field_info[key] = ("scalar", 1)
                fields[key] = np.full((n_steps, n_pts), np.nan, dtype=np.float32)
            else:
                field_info[key] = ("vector", arr.shape[1])
                fields[key] = np.full(
                    (n_steps, n_pts, arr.shape[1]), np.nan, dtype=np.float32
                )
            fields[key][0] = arr
        # Load all timesteps
        for t in range(1, n_steps):
            mesh = meshio.read(vtu_files[t])
            for key in field_info.keys():
                fields[key][t] = mesh.point_data[key].astype(np.float32)
        simulations[name] = dict(
            energy_mJ=energy,
            duration_ns=duration,
            points=points,
            fields=fields,
            field_info=field_info,
            n_timesteps=n_steps,
            triangles=triangles,
        )
    return simulations
# =============================================
# STREAMLIT APP
# =============================================
def main():
    st.set_page_config(page_title="FEA Viewer (Cloud-Safe)", layout="wide")
    st.title("🔍 FEA Laser Simulation Viewer")
    st.caption("✅ Plotly + Matplotlib | ❌ No PyVista rendering")
    simulations = load_all_simulations()
    if not simulations:
        st.error("No valid simulations found.")
        return
    # -----------------------------------------
    # SIDEBAR
    # -----------------------------------------
    st.sidebar.header("⚙️ Simulation")
    sim_name = st.sidebar.selectbox("Select simulation", sorted(simulations))
    sim = simulations[sim_name]
    st.sidebar.write(f"**Energy:** {sim['energy_mJ']} mJ")
    st.sidebar.write(f"**Pulse Duration:** {sim['duration_ns']} ns")
    # -----------------------------------------
    # FIELD & TIME
    # -----------------------------------------
    field = st.selectbox("Select field", list(sim["fields"].keys()))
    timestep = st.slider(
        "Timestep (ns)",
        0,
        sim["n_timesteps"] - 1,
        0,
    )
    pts = sim["points"]
    kind, _ = sim["field_info"][field]
    raw = sim["fields"][field][timestep]
    if kind == "scalar":
        values = raw
        label = field
    else:
        values = np.linalg.norm(raw, axis=1)
        label = f"{field} (magnitude)"
    # -----------------------------------------
    # PLOTLY 3D (PRIMARY)
    # -----------------------------------------
    st.subheader(f"{label} at {timestep + 1} ns – {sim_name}")
    if sim["triangles"] is not None:
        tri = sim["triangles"]
        data=go.Mesh3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            i=tri[:, 0],
            j=tri[:, 1],
            k=tri[:, 2],
            intensity=values,
            colorscale="Viridis",
            intensitymode='vertex',
            colorbar=dict(title=label),
            opacity=0.85,
            lighting=dict(ambient=0.8, diffuse=0.6, specular=0.5, roughness=0.5, fresnel=0.2),
            lightposition=dict(x=100, y=200, z=150),
            hovertemplate='Value: %{intensity:.2f}<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}',
        )
    else:
        data=go.Mesh3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            alphahull=3,
            intensity=values,
            colorscale="Viridis",
            intensitymode='vertex',
            colorbar=dict(title=label),
            opacity=0.85,
            lighting=dict(ambient=0.8, diffuse=0.6, specular=0.5, roughness=0.5, fresnel=0.2),
            lightposition=dict(x=100, y=200, z=150),
            hovertemplate='Value: %{intensity:.2f}<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}',
        )
    fig = go.Figure(data=data)
    fig.update_layout(
        height=700,
        scene=dict(aspectmode="data", camera_eye=dict(x=1.5, y=1.5, z=1.5)),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)
    # -----------------------------------------
    # MATPLOTLIB FALLBACK / EXPORT VIEW
    # -----------------------------------------
    with st.expander("🖼️ 2D Projection (Matplotlib)"):
        fig2, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=values,
            s=5,
            cmap="viridis",
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(label)
        plt.colorbar(sc, ax=ax)
        st.pyplot(fig2)
    # -----------------------------------------
    # STATISTICS
    # -----------------------------------------
    with st.expander("📊 Field Statistics"):
        clean = values[~np.isnan(values)]
        st.write(
            {
                "Min": float(clean.min()),
                "Max": float(clean.max()),
                "Mean": float(clean.mean()),
                "Std": float(clean.std()),
            }
        )
# =============================================
# ENTRY POINT
# =============================================
if __name__ == "__main__":
    main()
