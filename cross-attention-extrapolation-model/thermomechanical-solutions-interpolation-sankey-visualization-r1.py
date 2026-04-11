import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict

st.set_page_config(page_title="ST-DGPA Sankey Visualizer", layout="wide")

# ==========================================
# 1. CORE PHYSICS & INTERPOLATION LOGIC
# ==========================================

class LaserSolderingInterolator:
    def __init__(self, sigma_g=0.20, s_E=10.0, s_tau=5.0, s_t=20.0):
        self.sigma_g = sigma_g
        self.s_E = s_E
        self.s_tau = s_tau
        self.s_t = s_t
        
    def compute_stdgpa_weights(self, sources: pd.DataFrame, query: Dict) -> pd.DataFrame:
        """
        Computes Spatio-Temporal Gated Physics Attention weights.
        w_i \propto exp(-phi^2 / 2sigma_g^2)
        phi^2 = ((E-E*)/sE)^2 + ((tau-tau*)/sTau)^2 + ((t-t*)/sT)^2
        """
        # Calculate distances
        dE = (sources['Energy'] - query['Energy']) / self.s_E
        dTau = (sources['Duration'] - query['Duration']) / self.s_tau
        dT = (sources['Time'] - query['Time']) / self.s_t
        
        phi_squared = dE**2 + dTau**2 + dT**2
        
        # Gating Kernel
        sources['Gating_Weight'] = np.exp(-phi_squared / (2 * self.sigma_g**2))
        
        # Attention (Simplified as inverse distance similarity for this concise version)
        # In full implementation, this uses projected embeddings.
        sources['Attention_Score'] = 1 / (1 + np.sqrt(phi_squared))
        
        # Physics Refinement (Interaction term)
        sources['Refinement'] = sources['Gating_Weight'] * sources['Attention_Score']
        
        # Final Normalized Weights
        sources['Combined_Weight'] = sources['Refinement'] / sources['Refinement'].sum()
        
        return sources

# ==========================================
# 2. SANKEY VISUALIZATION ENGINE
# ==========================================

def create_stdgpa_sankey(df_sources: pd.DataFrame, query: Dict) -> go.Figure:
    """Creates a Sankey diagram showing flow from Sources -> Weight Components -> Target."""
    
    labels = ['TARGET']
    node_colors = ['#FF6B6B'] # Target color
    
    # Add Sources
    n_sources = len(df_sources)
    for i in range(n_sources):
        labels.append(f"Sim {i+1}\n(E:{df_sources.iloc[i]['Energy']}, t:{df_sources.iloc[i]['Duration']})")
        node_colors.append('rgba(153,102,255,0.8)')
        
    # Add Components
    components = ['Energy Gate', 'Duration Gate', 'Time Gate', 'Attention', 'Refinement', 'Combined']
    component_start = len(labels)
    comp_colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFD93D', '#36A2EB', '#9966FF']
    labels.extend(components)
    node_colors.extend(comp_colors)
    
    # Link Construction
    sources_idx, targets_idx, values, link_colors = [], [], [], []
    
    # 1. Sources -> Components (The decomposition)
    for i in range(n_sources):
        src_idx = i + 1
        row = df_sources.iloc[i]
        
        # We map components to indices starting at component_start
        # Energy, Duration, Time gates are derived from distance components
        # For visualization, we distribute the weight contribution
        
        # Energy Component
        sources_idx.append(src_idx)
        targets_idx.append(component_start + 0)
        # Contribution proportional to distance term in phi
        val_e = (row['Energy'] - query['Energy'])**2
        sources_idx.append(src_idx)
        targets_idx.append(component_start + 1)
        val_tau = (row['Duration'] - query['Duration'])**2
        sources_idx.append(src_idx)
        targets_idx.append(component_start + 2)
        val_t = (row['Time'] - query['Time'])**2
        sources_idx.append(src_idx)
        targets_idx.append(component_start + 3)
        val_attn = row['Attention_Score'] * 100 # Scale for viz
        sources_idx.append(src_idx)
        targets_idx.append(component_start + 4)
        val_ref = row['Refinement'] * 100
        sources_idx.append(src_idx)
        targets_idx.append(component_start + 5)
        val_comb = row['Combined_Weight'] * 100 # Scale to percentage
        
        # Add scaled values for flow
        values.extend([val_e*10, val_tau*10, val_t*10, val_attn, val_ref, val_comb])
        link_colors.extend(['rgba(255,107,107,0.5)']*6)

    # 2. Components -> Target (Aggregation)
    # We sum the flows entering each component and route to Target
    # This is a conceptual representation of how components feed the final prediction
    
    for comp_idx in range(len(components)):
        sources_idx.append(component_start + comp_idx)
        targets_idx.append(0) # Target index
        
        # Sum values flowing into this component from sources
        total_in = sum(v for s, t, v in zip(sources_idx, targets_idx, values) if s == component_start + comp_idx and t != 0)
        # In the previous logic, we are summing into the component, now we output from it
        # But wait, in the loop above we added source->component links. 
        # We need to calculate the total flow INTO the component to set the flow OUT to Target
        
        # Recalculate strictly:
        flow_in = 0
        for s, t, v in zip(sources_idx, targets_idx, values):
            if t == component_start + comp_idx:
                flow_in += v
        
        values.append(flow_in * 0.5) # Damping factor for visual balance
        link_colors.append('rgba(153,102,255,0.6)')

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=node_colors
        ),
        link=dict(
            source=sources_idx,
            target=targets_idx,
            value=values,
            color=link_colors
        ),
        hoverinfo='all'
    )])
    
    fig.update_layout(
        title_text=f"<b>ST-DGPA Attention Flow</b><br>Query: E={query['Energy']}, τ={query['Duration']}, t={query['Time']}",
        font_size=12,
        width=800,
        height=600
    )
    return fig

# ==========================================
# 3. STREAMLIT APPLICATION
# ==========================================

def main():
    st.title("🔬 ST-DGPA Interpolation & Sankey Analysis")
    st.markdown("Visualizing the flow of weights from source simulations to the target query.")
    
    # Sidebar for Data & Parameters
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        sigma_g = st.slider("Gating Width ($\sigma_g$)", 0.05, 1.0, 0.20, 0.05)
        s_E = st.slider("Energy Scale ($s_E$)", 0.1, 50.0, 10.0, 0.5)
        s_tau = st.slider("Duration Scale ($s_\\tau$)", 0.1, 20.0, 5.0, 0.5)
        s_t = st.slider("Time Scale ($s_t$)", 1.0, 50.0, 20.0, 1.0)
        
        st.markdown("---")
        
        # Generate Mock Data or Load CSV
        if st.button("Generate Demo Data"):
            data = {
                'Energy': np.random.uniform(0.5, 8.0, 20),
                'Duration': np.random.uniform(2.0, 7.0, 20),
                'Time': np.random.uniform(1.0, 10.0, 20),
                'Max_Temp': np.random.uniform(500, 1500, 20)
            }
            st.session_state.df_sources = pd.DataFrame(data)
            
        uploaded_file = st.file_uploader("Upload CSV (Energy, Duration, Time, Max_Temp)", type=["csv"])
        if uploaded_file is not None:
            st.session_state.df_sources = pd.read_csv(uploaded_file)
            
    # Main Area: Query and Visualization
    if 'df_sources' in st.session_state:
        df = st.session_state.df_sources
        st.success(f"Loaded {len(df)} simulations.")
        
        st.subheader("🎯 Target Query")
        col1, col2, col3 = st.columns(3)
        with col1:
            q_energy = st.number_input("Energy (mJ)", 0.1, 20.0, 3.5)
        with col2:
            q_tau = st.number_input("Duration (ns)", 0.1, 20.0, 4.2)
        with col3:
            q_time = st.number_input("Time (ns)", 0.1, 20.0, 5.0)
            
        query = {'Energy': q_energy, 'Duration': q_tau, 'Time': q_time}
        
        if st.button("🚀 Compute Interpolation"):
            # Run Interpolation
            interpolator = LaserSolderingInterolator(sigma_g=sigma_g, s_E=s_E, s_tau=s_tau, s_t=s_t)
            results = interpolator.compute_stdgpa_weights(df.copy(), query)
            
            # Display Results
            st.markdown("### 📊 Top Contributing Sources")
            top_sources = results.nlargest(5, 'Combined_Weight')[['Energy', 'Duration', 'Time', 'Combined_Weight', 'Max_Temp']]
            st.dataframe(top_sources.style.highlight_max(axis=0, subset=['Combined_Weight']), use_container_width=True)
            
            # Prediction (Weighted Average of Max_Temp as an example field)
            predicted_temp = (results['Combined_Weight'] * results['Max_Temp']).sum()
            st.metric("Predicted Max Temperature", f"{predicted_temp:.2f} K")
            
            # Sankey Diagram
            st.markdown("### 🕸️ Attention Weight Flow")
            fig_sankey = create_stdgpa_sankey(results, query)
            st.plotly_chart(fig_sankey, use_container_width=True)
            
    else:
        st.info("Please generate demo data or upload a CSV in the sidebar to begin.")

if __name__ == "__main__":
    main()
