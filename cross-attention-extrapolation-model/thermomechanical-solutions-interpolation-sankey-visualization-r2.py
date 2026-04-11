import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict, Optional
import plotly.colors

st.set_page_config(page_title="ST-DGPA Sankey Visualizer - Enhanced", layout="wide")

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
        w_i ∝ exp(-phi^2 / 2sigma_g^2)
        phi^2 = ((E-E*)/sE)^2 + ((tau-tau*)/sTau)^2 + ((t-t*)/sT)^2
        """
        # Calculate distances
        dE = (sources['Energy'] - query['Energy']) / self.s_E
        dTau = (sources['Duration'] - query['Duration']) / self.s_tau
        dT = (sources['Time'] - query['Time']) / self.s_t
        
        phi_squared = dE**2 + dTau**2 + dT**2
        
        # Gating Kernel: Gaussian kernel for parameter proximity
        sources['Gating_Weight'] = np.exp(-phi_squared / (2 * self.sigma_g**2))
        
        # Attention: Simplified as inverse distance similarity
        # In full implementation, this uses projected embeddings from transformer
        sources['Attention_Score'] = 1 / (1 + np.sqrt(phi_squared))
        
        # Physics Refinement: Multiplicative interaction term
        sources['Refinement'] = sources['Gating_Weight'] * sources['Attention_Score']
        
        # Final Normalized Weights: Softmax-like normalization
        sources['Combined_Weight'] = sources['Refinement'] / sources['Refinement'].sum()
        
        return sources

# ==========================================
# 2. ENHANCED SANKEY VISUALIZATION ENGINE
# ==========================================

def create_stdgpa_sankey(
    df_sources: pd.DataFrame, 
    query: Dict,
    # Customization parameters
    node_colors: Optional[List[str]] = None,
    link_colors: Optional[List[str]] = None,
    font_family: str = "Arial, sans-serif",
    font_size: int = 12,
    node_thickness: int = 20,
    node_pad: int = 15,
    show_math_explanations: bool = True,
    component_labels: Optional[List[str]] = None,
    target_label: str = "TARGET",
    width: int = 1000,
    height: int = 700
) -> go.Figure:
    """
    Creates an enhanced Sankey diagram with customization options and mathematical explanations.
    
    Parameters:
    -----------
    df_sources : pd.DataFrame
        DataFrame containing source simulations with computed weights
    query : Dict
        Query parameters (Energy, Duration, Time)
    node_colors : List[str], optional
        Custom colors for nodes [target, sources..., components...]
    link_colors : List[str], optional
        Custom colors for links
    font_family : str
        Font family for labels
    font_size : int
        Font size for labels
    node_thickness : int
        Thickness of Sankey nodes
    node_pad : int
        Padding between nodes
    show_math_explanations : bool
        Whether to show mathematical formula explanations on hover
    component_labels : List[str], optional
        Custom labels for weight components
    target_label : str
        Label for the target node
    width, height : int
        Figure dimensions
    """
    
    # Default component labels if not provided
    if component_labels is None:
        component_labels = [
            'Energy Gate\nφ² = ((E*-Eᵢ)/s_E)²',
            'Duration Gate\nφ² = ((τ*-τᵢ)/s_τ)²',
            'Time Gate\nφ² = ((t*-tᵢ)/s_t)²',
            'Attention\nαᵢ = softmax(Q·Kᵀ/√dₖ)',
            'Refinement\nwᵢ ∝ αᵢ × gatingᵢ',
            'Combined\nwᵢ = αᵢ·gatingᵢ / Σ(αⱼ·gatingⱼ)'
        ] if show_math_explanations else [
            'Energy Gate', 'Duration Gate', 'Time Gate', 
            'Attention', 'Refinement', 'Combined'
        ]
    
    # Default colors if not provided
    if node_colors is None:
        node_colors = ['#FF6B6B']  # Target color (red)
        # Source colors (purple with varying opacity based on weight)
        n_sources = len(df_sources)
        for i in range(n_sources):
            weight = df_sources.iloc[i]['Combined_Weight']
            opacity = min(0.3 + weight * 0.7, 1.0)  # Scale opacity by weight
            node_colors.append(f'rgba(153,102,255,{opacity:.2f})')
        # Component colors
        comp_colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFD93D', '#36A2EB', '#9966FF']
        node_colors.extend(comp_colors)
    
    if link_colors is None:
        # Default link colors: semi-transparent versions of component colors
        link_colors = [
            'rgba(255,107,107,0.5)',   # Energy Gate links
            'rgba(78,205,196,0.5)',    # Duration Gate links
            'rgba(149,225,211,0.5)',   # Time Gate links
            'rgba(255,217,61,0.5)',    # Attention links
            'rgba(54,162,235,0.5)',    # Refinement links
            'rgba(153,102,255,0.5)'    # Combined links
        ]
    
    # Build node labels
    labels = [target_label]
    
    # Add Sources with detailed hover info
    for i in range(n_sources):
        row = df_sources.iloc[i]
        source_label = (
            f"Sim {i+1}\n"
            f"E: {row['Energy']:.2f} mJ\n"
            f"τ: {row['Duration']:.2f} ns\n"
            f"t: {row['Time']:.2f} ns\n"
            f"w: {row['Combined_Weight']:.3f}"
        )
        labels.append(source_label)
    
    # Add Components
    component_start = len(labels)
    labels.extend(component_labels)
    
    # Link Construction
    sources_idx, targets_idx, values, link_colors_list, hover_texts = [], [], [], [], []
    
    # 1. Sources -> Components (The decomposition)
    for i in range(n_sources):
        src_idx = i + 1  # +1 because target is at index 0
        row = df_sources.iloc[i]
        
        # Calculate individual contributions for visualization
        # These are scaled for visual clarity, not actual weight values
        val_e = ((row['Energy'] - query['Energy']) / 10.0)**2 * 10  # Scale for viz
        val_tau = ((row['Duration'] - query['Duration']) / 5.0)**2 * 10
        val_t = ((row['Time'] - query['Time']) / 20.0)**2 * 10
        val_attn = row['Attention_Score'] * 100  # Scale for viz
        val_ref = row['Refinement'] * 100
        val_comb = row['Combined_Weight'] * 100  # Scale to percentage
        
        # Energy Gate link
        sources_idx.append(src_idx)
        targets_idx.append(component_start + 0)
        values.append(max(0.01, val_e))  # Ensure positive
        link_colors_list.append(link_colors[0])
        hover_texts.append(
            f"<b>Energy Gate Contribution</b><br>"
            f"Source: Sim {i+1}<br>"
            f"Formula: φ² = ((E* - Eᵢ) / s_E)²<br>"
            f"E* = {query['Energy']:.2f} mJ, Eᵢ = {row['Energy']:.2f} mJ<br>"
            f"s_E = 10.0 mJ (scaling factor)<br>"
            f"Contribution: {val_e:.3f}"
        )
        
        # Duration Gate link
        sources_idx.append(src_idx)
        targets_idx.append(component_start + 1)
        values.append(max(0.01, val_tau))
        link_colors_list.append(link_colors[1])
        hover_texts.append(
            f"<b>Duration Gate Contribution</b><br>"
            f"Source: Sim {i+1}<br>"
            f"Formula: φ² = ((τ* - τᵢ) / s_τ)²<br>"
            f"τ* = {query['Duration']:.2f} ns, τᵢ = {row['Duration']:.2f} ns<br>"
            f"s_τ = 5.0 ns (scaling factor)<br>"
            f"Contribution: {val_tau:.3f}"
        )
        
        # Time Gate link
        sources_idx.append(src_idx)
        targets_idx.append(component_start + 2)
        values.append(max(0.01, val_t))
        link_colors_list.append(link_colors[2])
        hover_texts.append(
            f"<b>Time Gate Contribution</b><br>"
            f"Source: Sim {i+1}<br>"
            f"Formula: φ² = ((t* - tᵢ) / s_t)²<br>"
            f"t* = {query['Time']:.2f} ns, tᵢ = {row['Time']:.2f} ns<br>"
            f"s_t = 20.0 ns (scaling factor)<br>"
            f"Contribution: {val_t:.3f}"
        )
        
        # Attention link
        sources_idx.append(src_idx)
        targets_idx.append(component_start + 3)
        values.append(max(0.01, val_attn))
        link_colors_list.append(link_colors[3])
        hover_texts.append(
            f"<b>Physics Attention Score</b><br>"
            f"Source: Sim {i+1}<br>"
            f"Formula: αᵢ = softmax(Q·Kᵀ / √dₖ)<br>"
            f"Q = query embedding, K = source key embeddings<br>"
            f"dₖ = embedding dimension, T = temperature<br>"
            f"Score: {row['Attention_Score']:.4f}"
        )
        
        # Refinement link
        sources_idx.append(src_idx)
        targets_idx.append(component_start + 4)
        values.append(max(0.01, val_ref))
        link_colors_list.append(link_colors[4])
        hover_texts.append(
            f"<b>Physics Refinement</b><br>"
            f"Source: Sim {i+1}<br>"
            f"Formula: wᵢ ∝ αᵢ × gatingᵢ<br>"
            f"gatingᵢ = exp(-φ² / (2σ_g²))<br>"
            f"σ_g = {sigma_g:.2f} (gating width)<br>"
            f"Refinement: {row['Refinement']:.4f}"
        )
        
        # Combined Weight link
        sources_idx.append(src_idx)
        targets_idx.append(component_start + 5)
        values.append(max(0.01, val_comb))
        link_colors_list.append(link_colors[5])
        hover_texts.append(
            f"<b>Final Combined Weight</b><br>"
            f"Source: Sim {i+1}<br>"
            f"Formula: wᵢ = (αᵢ × gatingᵢ) / Σⱼ(αⱼ × gatingⱼ)<br>"
            f"This is the normalized weight used for interpolation<br>"
            f"Weight: {row['Combined_Weight']:.4f} ({row['Combined_Weight']*100:.2f}%)"
        )

    # 2. Components -> Target (Aggregation)
    for comp_idx in range(len(component_labels)):
        sources_idx.append(component_start + comp_idx)
        targets_idx.append(0)  # Target index
        
        # Calculate total flow INTO this component from all sources
        flow_in = sum(
            v for s, t, v in zip(sources_idx[:-len(component_labels)], 
                               targets_idx[:-len(component_labels)], 
                               values[:-len(component_labels)])
            if t == component_start + comp_idx
        )
        
        # Scale for visual balance (damping factor)
        values.append(flow_in * 0.5)
        link_colors_list.append(f'rgba(153,102,255,0.6)')
        hover_texts.append(
            f"<b>Component Aggregation</b><br>"
            f"{component_labels[comp_idx]} → {target_label}<br>"
            f"Total contribution from all sources: {flow_in:.3f}<br>"
            f"Scaled for visualization: {flow_in * 0.5:.3f}"
        )

    # Create Sankey figure
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=node_pad,
            thickness=node_thickness,
            line=dict(color="black", width=0.5),
            label=labels,
            color=node_colors,
            hovertemplate='<b>%{label}</b><br>Value: %{value:.3f}<extra></extra>',
            font=dict(family=font_family, size=font_size)
        ),
        link=dict(
            source=sources_idx,
            target=targets_idx,
            value=values,
            color=link_colors_list,
            hovertemplate='%{hovertext}<extra></extra>',
            hovertext=hover_texts,
            line=dict(width=0.5, color='rgba(255,255,255,0.3)')
        ),
        hoverinfo='text'
    )])
    
    # Update layout with customizations
    title_text = (
        f"<b>ST-DGPA Attention Flow</b><br>"
        f"Query: E={query['Energy']:.2f} mJ, τ={query['Duration']:.2f} ns, t={query['Time']:.2f} ns<br>"
        f"<sub>σ_g={sigma_g:.2f}, s_E={s_E:.1f}, s_τ={s_tau:.1f}, s_t={s_t:.1f}</sub>"
    )
    
    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(family=font_family, size=font_size+4),
            x=0.5,
            xanchor='center'
        ),
        font=dict(family=font_family, size=font_size),
        width=width,
        height=height,
        plot_bgcolor='rgba(240, 240, 245, 0.9)',
        paper_bgcolor='white',
        margin=dict(t=100, l=50, r=50, b=50),
        hoverlabel=dict(
            font=dict(family=font_family, size=font_size),
            bgcolor='rgba(44, 62, 80, 0.9)',
            bordercolor='white',
            namelength=-1  # Show full text
        )
    )
    
    return fig

# ==========================================
# 3. ENHANCED STREAMLIT APPLICATION
# ==========================================

def main():
    st.title("🔬 ST-DGPA Interpolation & Sankey Visualizer")
    st.markdown("""
    **Interactive visualization of Spatio-Temporal Gated Physics Attention (ST-DGPA) weights.**
    
    This demo shows how source simulations contribute to a target query through:
    1. **Physics Attention**: Transformer-inspired similarity in embedding space
    2. **(E, τ, t) Gating**: Explicit Gaussian kernel over process parameters
    3. **Final Weights**: Multiplicative combination with normalization
    
    Hover over diagram elements to see mathematical formulas and explanations!
    """)
    
    # Sidebar for Configuration
    with st.sidebar:
        st.header("⚙️ ST-DGPA Parameters")
        
        # Gating parameters
        sigma_g = st.slider(
            "Gating Width (σ_g)", 
            min_value=0.05, 
            max_value=1.0, 
            value=0.20, 
            step=0.05,
            help="Controls sharpness of the (E, τ, t) gating kernel. Smaller = sharper gating."
        )
        s_E = st.slider(
            "Energy Scale (s_E) [mJ]", 
            min_value=0.1, 
            max_value=50.0, 
            value=10.0, 
            step=0.5,
            help="Scaling factor for energy differences in gating kernel."
        )
        s_tau = st.slider(
            "Duration Scale (s_τ) [ns]", 
            min_value=0.1, 
            max_value=20.0, 
            value=5.0, 
            step=0.5,
            help="Scaling factor for duration differences in gating kernel."
        )
        s_t = st.slider(
            "Time Scale (s_t) [ns]", 
            min_value=1.0, 
            max_value=50.0, 
            value=20.0, 
            step=1.0,
            help="Scaling factor for time differences in gating kernel."
        )
        
        st.markdown("---")
        st.header("🎨 Visualization Settings")
        
        # Font settings
        font_family = st.selectbox(
            "Font Family",
            ["Arial, sans-serif", "Courier New, monospace", "Times New Roman, serif", "Verdana, sans-serif"],
            index=0
        )
        font_size = st.slider("Font Size", min_value=8, max_value=20, value=12, step=1)
        
        # Node settings
        node_thickness = st.slider("Node Thickness", min_value=10, max_value=40, value=20, step=2)
        node_pad = st.slider("Node Padding", min_value=5, max_value=30, value=15, step=1)
        
        # Figure size
        fig_width = st.slider("Figure Width", min_value=600, max_value=1400, value=1000, step=50)
        fig_height = st.slider("Figure Height", min_value=400, max_value=1000, value=700, step=50)
        
        # Color customization
        st.markdown("**Node Colors**")
        col1, col2 = st.columns(2)
        with col1:
            target_color = st.color_picker("Target Node", "#FF6B6B")
        with col2:
            source_base_color = st.color_picker("Source Base Color", "#9966FF")
        
        st.markdown("**Component Colors**")
        comp_colors = st.color_picker("Component Base Color", "#4ECDC4")
        
        # Toggle mathematical explanations
        show_math = st.checkbox("Show Mathematical Formulas on Hover", value=True, 
                               help="Display ST-DGPA formulas when hovering over diagram elements")
        
        # Custom component labels
        st.markdown("**Component Labels**")
        use_custom_labels = st.checkbox("Use Custom Component Labels", value=False)
        if use_custom_labels:
            component_labels = []
            for i, default in enumerate(['Energy Gate', 'Duration Gate', 'Time Gate', 
                                        'Attention', 'Refinement', 'Combined']):
                label = st.text_input(f"Component {i+1}", value=default, key=f"comp_{i}")
                component_labels.append(label)
        else:
            component_labels = None
        
        st.markdown("---")
        st.header("📊 Data Settings")
        
        # Number of simulations
        n_sims = st.slider("Number of Simulations", min_value=2, max_value=20, value=4, step=1,
                          help="Number of source simulations to generate for demo")
        
        # Generate or load data
        if st.button("🔄 Generate Demo Data", use_container_width=True):
            # Generate realistic laser soldering data
            np.random.seed(42)  # For reproducibility
            data = {
                'Energy': np.random.uniform(0.5, 8.0, n_sims),
                'Duration': np.random.uniform(2.0, 7.0, n_sims),
                'Time': np.random.uniform(1.0, 10.0, n_sims),
                'Max_Temp': np.random.uniform(500, 1500, n_sims)
            }
            st.session_state.df_sources = pd.DataFrame(data)
            st.success(f"Generated {n_sims} demo simulations!")
        
        uploaded_file = st.file_uploader("📁 Upload CSV", type=["csv"], 
                                        help="CSV with columns: Energy, Duration, Time, Max_Temp")
        if uploaded_file is not None:
            try:
                st.session_state.df_sources = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(st.session_state.df_sources)} simulations from CSV!")
            except Exception as e:
                st.error(f"Error loading CSV: {e}")
        
        # Show current data
        if 'df_sources' in st.session_state:
            st.markdown("---")
            st.markdown(f"**Loaded Data:** {len(st.session_state.df_sources)} simulations")
            with st.expander("📋 View Data"):
                st.dataframe(st.session_state.df_sources, use_container_width=True)
    
    # Main Area: Query and Visualization
    if 'df_sources' in st.session_state:
        df = st.session_state.df_sources
        
        st.subheader("🎯 Target Query Parameters")
        col1, col2, col3 = st.columns(3)
        with col1:
            q_energy = st.number_input(
                "Energy (mJ)", 
                min_value=0.1, 
                max_value=20.0, 
                value=3.5,
                step=0.1,
                help="Target pulse energy for interpolation"
            )
        with col2:
            q_tau = st.number_input(
                "Duration (ns)", 
                min_value=0.1, 
                max_value=20.0, 
                value=4.2,
                step=0.1,
                help="Target pulse duration for interpolation"
            )
        with col3:
            q_time = st.number_input(
                "Time (ns)", 
                min_value=0.1, 
                max_value=20.0, 
                value=5.0,
                step=0.1,
                help="Target observation time for interpolation"
            )
        
        query = {'Energy': q_energy, 'Duration': q_tau, 'Time': q_time}
        
        # Prepare colors based on user selection
        n_sources = len(df)
        node_colors = [target_color]
        for i in range(n_sources):
            # Vary opacity based on a placeholder (will be updated after weight computation)
            node_colors.append(f'rgba({int(source_base_color[1:3], 16)}, '
                             f'{int(source_base_color[3:5], 16)}, '
                             f'{int(source_base_color[5:7], 16)}, 0.8)')
        
        # Component colors
        comp_base = [int(comp_colors[1:3], 16), int(comp_colors[3:5], 16), int(comp_colors[5:7], 16)]
        comp_node_colors = []
        for i in range(6):
            # Create gradient of component colors
            r = min(255, comp_base[0] + i*20)
            g = min(255, comp_base[1] + i*15)
            b = min(255, comp_base[2] + i*10)
            comp_node_colors.append(f'rgb({r},{g},{b})')
        node_colors.extend(comp_node_colors)
        
        # Link colors (semi-transparent versions)
        link_colors = []
        for color in comp_node_colors:
            rgb = color.replace('rgb', 'rgba').replace(')', ', 0.5)')
            link_colors.append(rgb)
        
        if st.button("🚀 Compute ST-DGPA Interpolation", type="primary", use_container_width=True):
            with st.spinner("Computing Spatio-Temporal Gated Physics Attention weights..."):
                # Run Interpolation
                interpolator = LaserSolderingInterolator(
                    sigma_g=sigma_g, s_E=s_E, s_tau=s_tau, s_t=s_t
                )
                results = interpolator.compute_stdgpa_weights(df.copy(), query)
                
                # Update source node colors based on actual weights
                for i in range(n_sources):
                    weight = results.iloc[i]['Combined_Weight']
                    opacity = min(0.3 + weight * 0.7, 1.0)
                    node_colors[i+1] = f'rgba({int(source_base_color[1:3], 16)}, ' \
                                     f'{int(source_base_color[3:5], 16)}, ' \
                                     f'{int(source_base_color[5:7], 16)}, {opacity:.2f})'
                
                # Display Results
                st.markdown("### 📊 Top Contributing Sources")
                top_sources = results.nlargest(5, 'Combined_Weight')[
                    ['Energy', 'Duration', 'Time', 'Combined_Weight', 'Max_Temp']
                ]
                st.dataframe(
                    top_sources.style.format({
                        'Energy': '{:.2f}',
                        'Duration': '{:.2f}',
                        'Time': '{:.2f}',
                        'Combined_Weight': '{:.4f}',
                        'Max_Temp': '{:.1f}'
                    }).highlight_max(axis=0, subset=['Combined_Weight'], color='#90EE90'),
                    use_container_width=True
                )
                
                # Prediction (Weighted Average of Max_Temp as example field)
                predicted_temp = (results['Combined_Weight'] * results['Max_Temp']).sum()
                st.metric(
                    "Predicted Max Temperature", 
                    f"{predicted_temp:.2f} K",
                    delta=f"{predicted_temp - df['Max_Temp'].mean():.1f} K vs mean"
                )
                
                # Show ST-DGPA formula explanation
                with st.expander("📐 ST-DGPA Mathematical Formulas", expanded=True):
                    st.markdown("""
                    **Core ST-DGPA Weight Formula:**
                    ```
                    w_i(θ*) = [α_i(θ*) × gating_i(θ*)] / Σ_j[α_j(θ*) × gating_j(θ*)]
                    ```
                    
                    **Where:**
                    - `α_i(θ*)` = Physics attention from transformer-like mechanism
                    - `gating_i(θ*)` = exp(-φ_i² / (2σ_g²)) with φ_i² = Σ[(p*-p_i)/s_p]²
                    - `s_p` = scaling factor for parameter p ∈ {E, τ, t}
                    - `σ_g` = gating kernel width (controls sharpness)
                    
                    **Parameter Proximity Kernel:**
                    ```
                    φ_i = √[((E*-E_i)/s_E)² + ((τ*-τ_i)/s_τ)² + ((t*-t_i)/s_t)²]
                    ```
                    
                    **Physics Attention:**
                    ```
                    α_i = softmax(Q·Kᵀ / √dₖ)  [multi-head transformer attention]
                    ```
                    """)
                
                # Sankey Diagram
                st.markdown("### 🕸️ Attention Weight Flow Diagram")
                st.markdown("""
                **How to read this diagram:**
                - **Left nodes**: Source simulations (colored by their final weight)
                - **Middle nodes**: Weight components (Energy Gate, Duration Gate, etc.)
                - **Right node**: Target prediction (aggregated result)
                - **Flow thickness**: Proportional to contribution magnitude
                - **Hover over any element**: See mathematical formulas and detailed explanations
                """)
                
                fig_sankey = create_stdgpa_sankey(
                    results, query,
                    node_colors=node_colors,
                    link_colors=link_colors,
                    font_family=font_family,
                    font_size=font_size,
                    node_thickness=node_thickness,
                    node_pad=node_pad,
                    show_math_explanations=show_math,
                    component_labels=component_labels,
                    target_label=target_label,
                    width=fig_width,
                    height=fig_height
                )
                st.plotly_chart(fig_sankey, use_container_width=True)
                
                # Additional analysis
                with st.expander("📈 Weight Distribution Analysis"):
                    # Plot weight distribution
                    fig_dist = go.Figure()
                    fig_dist.add_trace(go.Bar(
                        x=[f"Sim {i+1}" for i in range(len(results))],
                        y=results['Combined_Weight'],
                        marker_color=[node_colors[i+1] for i in range(len(results))],
                        text=[f"{w*100:.1f}%" for w in results['Combined_Weight']],
                        textposition='auto'
                    ))
                    fig_dist.update_layout(
                        title="Final Weight Distribution Across Sources",
                        xaxis_title="Source Simulation",
                        yaxis_title="Weight",
                        height=400
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
                    
                    # Show parameter distances
                    st.markdown("**Parameter Distances from Query:**")
                    dist_df = pd.DataFrame({
                        'Source': [f"Sim {i+1}" for i in range(len(results))],
                        'Energy Dist': np.abs(results['Energy'] - query['Energy']),
                        'Duration Dist': np.abs(results['Duration'] - query['Duration']),
                        'Time Dist': np.abs(results['Time'] - query['Time']),
                        'Combined Weight': results['Combined_Weight']
                    })
                    st.dataframe(
                        dist_df.style.format({
                            'Energy Dist': '{:.2f}',
                            'Duration Dist': '{:.2f}',
                            'Time Dist': '{:.2f}',
                            'Combined Weight': '{:.4f}'
                        }),
                        use_container_width=True
                    )
                
    else:
        st.info("👈 Please generate demo data or upload a CSV file in the sidebar to begin.")
        
        # Show example CSV format
        with st.expander("📋 Expected CSV Format"):
            st.markdown("""
            Your CSV file should have the following columns:
            ```csv
            Energy,Duration,Time,Max_Temp
            0.5,2.0,1.0,520.5
            1.2,3.5,2.5,680.3
            2.8,4.2,4.0,890.1
            ...
            ```
            
            **Column descriptions:**
            - `Energy`: Pulse energy in millijoules (mJ)
            - `Duration`: Pulse duration in nanoseconds (ns)
            - `Time`: Observation time in nanoseconds (ns)
            - `Max_Temp`: Maximum temperature in Kelvin (example field for prediction)
            """)

if __name__ == "__main__":
    main()
