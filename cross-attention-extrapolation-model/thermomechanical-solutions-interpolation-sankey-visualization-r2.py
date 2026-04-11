#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ST-DGPA Laser Soldering Interpolation & Sankey Visualizer
===========================================================
Enhanced version with:
- Robust error handling and debugging
- Mathematical hover explanations
- Full customization (colors, fonts, sizes)
- Session state management
- Graceful handling of missing columns
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict, Optional
import traceback

st.set_page_config(page_title="ST-DGPA Sankey Visualizer", layout="wide", page_icon="🔬")

# ==========================================
# 1. CORE PHYSICS & INTERPOLATION LOGIC
# ==========================================

class LaserSolderingInterpolator:
    """ST-DGPA interpolator for laser soldering simulations"""
    
    def __init__(self, sigma_g: float = 0.20, s_E: float = 10.0, 
                 s_tau: float = 5.0, s_t: float = 20.0, 
                 temporal_weight: float = 0.3):
        self.sigma_g = sigma_g  # Gating kernel width
        self.s_E = s_E          # Energy scaling (mJ)
        self.s_tau = s_tau      # Duration scaling (ns)
        self.s_t = s_t          # Time scaling (ns)
        self.temporal_weight = temporal_weight  # Weight for temporal similarity
        
    def compute_stdgpa_weights(self, sources: pd.DataFrame, query: Dict) -> pd.DataFrame:
        """
        Compute ST-DGPA weights: w_i ∝ α_i × gating_i
        where φ_i² = ((E*-E_i)/s_E)² + ((τ*-τ_i)/s_τ)² + ((t*-t_i)/s_t)²
        and gating_i = exp(-φ_i² / (2σ_g²))
        """
        try:
            # Calculate normalized distances
            dE = (sources['Energy'] - query['Energy']) / self.s_E
            dTau = (sources['Duration'] - query['Duration']) / self.s_tau
            dT = (sources['Time'] - query['Time']) / self.s_t
            
            # Physics-aware temporal scaling: tighter matching during heating
            if self.temporal_weight > 0 and query['Duration'] > 0:
                time_scale = 1.0 + 0.5 * (query['Time'] / query['Duration'])
                dT = dT * time_scale
            
            # Compute φ² for gating kernel
            phi_squared = dE**2 + dTau**2 + dT**2
            
            # Gating Kernel: Gaussian proximity in (E, τ, t) space
            sources['Gating_Weight'] = np.exp(-phi_squared / (2 * self.sigma_g**2))
            
            # Physics Attention: inverse distance similarity (simplified)
            sources['Attention_Score'] = 1.0 / (1.0 + np.sqrt(phi_squared))
            
            # Physics Refinement: multiplicative interaction
            sources['Refinement'] = sources['Gating_Weight'] * sources['Attention_Score']
            
            # Final Normalized Weights (softmax-like)
            total = sources['Refinement'].sum() + 1e-12
            sources['Combined_Weight'] = sources['Refinement'] / total
            
            return sources
            
        except Exception as e:
            st.error(f"Error computing ST-DGPA weights: {e}")
            st.exception(e)
            return sources

# ==========================================
# 2. ENHANCED SANKEY VISUALIZATION ENGINE
# ==========================================

def create_stdgpa_sankey(df_sources: pd.DataFrame, query: Dict, 
                        customization: Optional[Dict] = None) -> go.Figure:
    """
    Create Sankey diagram with hover math explanations and full customization.
    
    Parameters:
    -----------
    df_sources : pd.DataFrame
        Sources with computed ST-DGPA weights
    query : Dict
        Query parameters {Energy, Duration, Time}
    customization : Dict, optional
        Custom settings for colors, fonts, sizes
    """
    # Default customization
    defaults = {
        'font_family': 'Arial, sans-serif',
        'font_size': 12,
        'node_thickness': 20,
        'node_pad': 15,
        'width': 1000,
        'height': 700,
        'show_math': True,
        'target_label': 'TARGET',
        'node_colors': {
            'target': '#FF6B6B',
            'source': '#9966FF',
            'components': ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFD93D', '#36A2EB', '#9966FF']
        }
    }
    cfg = {**defaults, **(customization or {})}
    
    # Component labels (with math if enabled)
    if cfg['show_math']:
        comp_labels = [
            'Energy Gate\nφ² = ((E*-Eᵢ)/s_E)²',
            'Duration Gate\nφ² = ((τ*-τᵢ)/s_τ)²',
            'Time Gate\nφ² = ((t*-tᵢ)/s_t)²',
            'Attention\nαᵢ = 1/(1+√φ²)',
            'Refinement\nwᵢ ∝ αᵢ×gatingᵢ',
            'Combined\nwᵢ = (αᵢ·gatingᵢ)/Σ(...)'
        ]
    else:
        comp_labels = ['Energy Gate', 'Duration Gate', 'Time Gate', 
                      'Attention', 'Refinement', 'Combined']
    
    # Build node labels and colors
    labels = [cfg['target_label']]
    node_colors = [cfg['node_colors']['target']]
    
    n_sources = len(df_sources)
    for i in range(n_sources):
        row = df_sources.iloc[i]
        w = row.get('Combined_Weight', 0)
        # Scale opacity by weight for visual emphasis
        opacity = min(0.3 + w * 0.7, 1.0)
        base = cfg['node_colors']['source']
        r, g, b = int(base[1:3], 16), int(base[3:5], 16), int(base[5:7], 16)
        node_colors.append(f'rgba({r},{g},{b},{opacity:.2f})')
        labels.append(f"Sim {i+1}\nE:{row.get('Energy',0):.1f} τ:{row.get('Duration',0):.1f}")
    
    # Add component nodes
    comp_start = len(labels)
    labels.extend(comp_labels)
    node_colors.extend(cfg['node_colors']['components'])
    
    # Build links: Sources → Components → Target
    s_idx, t_idx, vals, l_colors, h_texts = [], [], [], [], []
    
    # Stage 1: Sources → Components (decomposition)
    for i in range(n_sources):
        src = i + 1  # +1 because target is index 0
        row = df_sources.iloc[i]
        
        # Scaled values for visualization (not actual weights)
        ve = ((row.get('Energy', query['Energy']) - query['Energy']) / 10.0)**2 * 10
        vτ = ((row.get('Duration', query['Duration']) - query['Duration']) / 5.0)**2 * 10
        vt = ((row.get('Time', query['Time']) - query['Time']) / 20.0)**2 * 10
        va = row.get('Attention_Score', 0) * 100
        vr = row.get('Refinement', 0) * 100
        vc = row.get('Combined_Weight', 0) * 100
        
        vals_list = [ve, vτ, vt, va, vr, vc]
        for c in range(6):
            s_idx.append(src)
            t_idx.append(comp_start + c)
            vals.append(max(0.01, vals_list[c]))
            l_colors.append(cfg['node_colors']['components'][c].replace('rgb', 'rgba').replace(')', ', 0.5)'))
            
            # Hover text with mathematical explanations
            if c == 0:
                h_texts.append(f"<b>Energy Gate</b><br>S{src} | φ² = ((E*-Eᵢ)/s_E)² = {ve/10:.4f}")
            elif c == 1:
                h_texts.append(f"<b>Duration Gate</b><br>S{src} | φ² = ((τ*-τᵢ)/s_τ)² = {vτ/10:.4f}")
            elif c == 2:
                h_texts.append(f"<b>Time Gate</b><br>S{src} | φ² = ((t*-tᵢ)/s_t)² = {vt/10:.4f}")
            elif c == 3:
                h_texts.append(f"<b>Attention</b><br>S{src} | αᵢ = 1/(1+√φ²) = {row.get('Attention_Score',0):.4f}")
            elif c == 4:
                h_texts.append(f"<b>Refinement</b><br>S{src} | wᵢ ∝ αᵢ·gatingᵢ = {row.get('Refinement',0):.4f}")
            else:
                h_texts.append(f"<b>Combined Weight</b><br>S{src} | wᵢ = (αᵢ·gatingᵢ)/Σ(...) = {row.get('Combined_Weight',0):.4f}")
    
    # Stage 2: Components → Target (aggregation)
    for c in range(6):
        s_idx.append(comp_start + c)
        t_idx.append(0)  # Target index
        # Sum flows INTO this component
        flow_in = sum(v for s, t, v in zip(s_idx[:-6], t_idx[:-6], vals[:-6]) if t == comp_start + c)
        vals.append(flow_in * 0.5)  # Damping for visual balance
        l_colors.append('rgba(153,102,255,0.6)')
        h_texts.append(f"<b>Aggregation</b><br>{comp_labels[c]} → TARGET<br>Total: {flow_in:.3f}")
    
    # Create Sankey figure - FIXED: removed invalid top-level hoverinfo
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=cfg['node_pad'],
            thickness=cfg['node_thickness'],
            line=dict(color="black", width=0.5),
            label=labels,
            color=node_colors,
            font=dict(family=cfg['font_family'], size=cfg['font_size']),
            hovertemplate='<b>%{label}</b><br>Value: %{value:.3f}<extra></extra>'  # FIXED: in node dict
        ),
        link=dict(
            source=s_idx,
            target=t_idx,
            value=vals,
            color=l_colors,
            hovertext=h_texts,
            hovertemplate='%{hovertext}<extra></extra>',  # FIXED: in link dict
            line=dict(width=0.5, color='rgba(255,255,255,0.3)')
        )
        # FIXED: removed hoverinfo='all' from top level
    ))
    
    # Update layout
    title_text = (
        f"<b>ST-DGPA Attention Flow</b><br>"
        f"Query: E={query['Energy']:.2f} mJ, τ={query['Duration']:.2f} ns, t={query['Time']:.2f} ns<br>"
        f"<sub>σ_g={cfg.get('sigma_g', 0.20):.2f}, s_E={cfg.get('s_E', 10.0):.1f}, s_τ={cfg.get('s_tau', 5.0):.1f}, s_t={cfg.get('s_t', 20.0):.1f}</sub>"
    )
    
    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(family=cfg['font_family'], size=cfg['font_size']+4),
            x=0.5,
            xanchor='center'
        ),
        font=dict(family=cfg['font_family'], size=cfg['font_size']),
        width=cfg['width'],
        height=cfg['height'],
        plot_bgcolor='rgba(240, 240, 245, 0.9)',
        paper_bgcolor='white',
        margin=dict(t=100, l=50, r=50, b=50),
        hoverlabel=dict(
            font=dict(family=cfg['font_family'], size=cfg['font_size']),
            bgcolor='rgba(44, 62, 80, 0.9)',
            bordercolor='white',
            namelength=-1  # Show full text
        )
    )
    
    return fig

# ==========================================
# 3. STREAMLIT APPLICATION (ENHANCED)
# ==========================================

def main():
    # Custom CSS for enhanced visual appeal
    st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; text-align: center; margin-bottom: 1.5rem; font-weight: 800; }
    .info-box { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; }
    .metric-card { background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 1rem; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🔬 ST-DGPA Interpolation & Sankey Analysis</h1>', unsafe_allow_html=True)
    st.markdown("Visualizing the flow of weights from source simulations to the target query using Spatio-Temporal Gated Physics Attention.")
    
    # Initialize session state
    if 'df_sources' not in st.session_state:
        st.session_state.df_sources = None
    if 'interpolator' not in st.session_state:
        st.session_state.interpolator = LaserSolderingInterpolator()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ ST-DGPA Parameters")
        
        # Physics parameters
        sigma_g = st.slider("Gating Width (σ_g)", 0.05, 1.0, 0.20, 0.05, 
                          help="Controls sharpness of the (E, τ, t) gating kernel")
        s_E = st.slider("Energy Scale (s_E) [mJ]", 0.1, 50.0, 10.0, 0.5,
                       help="Scaling factor for energy differences")
        s_tau = st.slider("Duration Scale (s_τ) [ns]", 0.1, 20.0, 5.0, 0.5,
                         help="Scaling factor for duration differences")
        s_t = st.slider("Time Scale (s_t) [ns]", 1.0, 50.0, 20.0, 1.0,
                       help="Scaling factor for time differences")
        temporal_weight = st.slider("Temporal Weight", 0.0, 1.0, 0.3, 0.05,
                                   help="Weight for temporal similarity in attention")
        
        # Update interpolator
        st.session_state.interpolator.sigma_g = sigma_g
        st.session_state.interpolator.s_E = s_E
        st.session_state.interpolator.s_tau = s_tau
        st.session_state.interpolator.s_t = s_t
        st.session_state.interpolator.temporal_weight = temporal_weight
        
        st.markdown("---")
        st.header("🎨 Visualization Settings")
        
        # Font settings
        font_family = st.selectbox("Font Family", 
                                  ["Arial, sans-serif", "Courier New, monospace", "Times New Roman, serif"],
                                  index=0)
        font_size = st.slider("Font Size", 8, 20, 12, 1)
        
        # Node settings
        node_thickness = st.slider("Node Thickness", 10, 40, 20, 2)
        
        # Figure size
        fig_width = st.slider("Figure Width", 600, 1400, 1000, 50)
        fig_height = st.slider("Figure Height", 400, 1000, 700, 50)
        
        # Color customization
        st.markdown("**Node Colors**")
        c1, c2 = st.columns(2)
        with c1:
            target_color = st.color_picker("Target Node", "#FF6B6B")
        with c2:
            source_color = st.color_picker("Source Base Color", "#9966FF")
        comp_color = st.color_picker("Component Base Color", "#4ECDC4")
        
        # Toggle mathematical explanations
        show_math = st.checkbox("Show Mathematical Formulas on Hover", value=True,
                               help="Display ST-DGPA formulas when hovering over diagram elements")
        
        st.markdown("---")
        st.header("📊 Data Settings")
        
        # Number of simulations for demo
        n_sims = st.slider("Demo Simulations", 2, 20, 4, 1,
                          help="Number of source simulations to generate")
        
        if st.button("🔄 Generate Demo Data", use_container_width=True):
            np.random.seed(42)  # Reproducible
            data = {
                'Energy': np.random.uniform(0.5, 8.0, n_sims),
                'Duration': np.random.uniform(2.0, 7.0, n_sims),
                'Time': np.random.uniform(1.0, 10.0, n_sims),
                'Max_Temp': np.random.uniform(500, 1500, n_sims)
            }
            st.session_state.df_sources = pd.DataFrame(data)
            st.success(f"✅ Generated {n_sims} demo simulations!")
        
        # CSV upload
        uploaded_file = st.file_uploader("📁 Upload CSV", type=["csv"],
                                        help="CSV with columns: Energy, Duration, Time, [Max_Temp]")
        if uploaded_file is not None:
            try:
                st.session_state.df_sources = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(st.session_state.df_sources)} rows from CSV")
            except Exception as e:
                st.error(f"❌ Error loading CSV: {e}")
                st.exception(e)
        
        # Show parameter summary
        if st.session_state.df_sources is not None:
            st.markdown("---")
            st.markdown("### 📋 Active Parameters")
            with st.expander("Parameter Summary", expanded=False):
                st.write(f"**σ_g**: {sigma_g:.2f} (gating sharpness)")
                st.write(f"**s_E**: {s_E:.1f} mJ (energy scale)")
                st.write(f"**s_τ**: {s_tau:.1f} ns (duration scale)")
                st.write(f"**s_t**: {s_t:.1f} ns (time scale)")
                st.write(f"**Temporal Weight**: {temporal_weight:.2f}")

    # Main area: Query and visualization
    if st.session_state.df_sources is not None:
        df = st.session_state.df_sources
        st.success(f"✅ Loaded {len(df)} simulations")
        
        st.subheader("🎯 Target Query Parameters")
        col1, col2, col3 = st.columns(3)
        with col1:
            q_energy = st.number_input("Energy (mJ)", 0.1, 20.0, 3.5, 0.1,
                                      help="Target pulse energy for interpolation")
        with col2:
            q_tau = st.number_input("Duration (ns)", 0.1, 20.0, 4.2, 0.1,
                                   help="Target pulse duration for interpolation")
        with col3:
            q_time = st.number_input("Time (ns)", 0.1, 20.0, 5.0, 0.1,
                                    help="Target observation time for interpolation")
        
        query = {'Energy': q_energy, 'Duration': q_tau, 'Time': q_time}
        
        # Build customization dict
        customization = {
            'font_family': font_family,
            'font_size': font_size,
            'node_thickness': node_thickness,
            'width': fig_width,
            'height': fig_height,
            'show_math': show_math,
            'node_colors': {
                'target': target_color,
                'source': source_color,
                'components': [comp_color, '#4ECDC4', '#95E1D3', '#FFD93D', '#36A2EB', '#9966FF']
            }
        }
        
        if st.button("🚀 Compute ST-DGPA Interpolation", type="primary", use_container_width=True):
            try:
                with st.spinner("Computing Spatio-Temporal Gated Physics Attention weights..."):
                    # Run interpolation
                    interpolator = st.session_state.interpolator
                    results = interpolator.compute_stdgpa_weights(df.copy(), query)
                    
                    # Display Top Sources
                    st.markdown("### 📊 Top Contributing Sources")
                    top_sources = results.nlargest(5, 'Combined_Weight')[
                        ['Energy', 'Duration', 'Time', 'Combined_Weight']
                    ]
                    # Add Max_Temp if it exists
                    if 'Max_Temp' in results.columns:
                        top_sources = top_sources.join(results[['Max_Temp']])
                    
                    st.dataframe(
                        top_sources.style.format({
                            'Energy': '{:.2f}',
                            'Duration': '{:.2f}',
                            'Time': '{:.2f}',
                            'Combined_Weight': '{:.4f}',
                            'Max_Temp': '{:.1f}' if 'Max_Temp' in top_sources.columns else None
                        }).highlight_max(axis=0, subset=['Combined_Weight'], color='#90EE90'),
                        use_container_width=True
                    )
                    
                    # Prediction (weighted average) - only if Max_Temp exists
                    if 'Max_Temp' in results.columns:
                        predicted_temp = (results['Combined_Weight'] * results['Max_Temp']).sum()
                        st.metric("Predicted Max Temperature", f"{predicted_temp:.2f} K")
                    else:
                        st.info("⚠️ `Max_Temp` column not found. Skipping temperature prediction.")
                    
                    # Sankey Diagram
                    st.markdown("### 🕸️ Attention Weight Flow")
                    st.markdown("""
                    **How to read this diagram:**
                    - **Left nodes**: Source simulations (colored by final weight)
                    - **Middle nodes**: Weight components (Energy Gate, Duration Gate, etc.)
                    - **Right node**: Target prediction (aggregated result)
                    - **Flow thickness**: Proportional to contribution magnitude
                    - **Hover over any element**: See mathematical formulas and detailed explanations
                    """)
                    
                    sources_data = results.to_dict('records')
                    fig_sankey = create_stdgpa_sankey(sources_data, query, customization)
                    st.plotly_chart(fig_sankey, use_container_width=True)
                    
                    # Weight distribution plot
                    st.markdown("### 📈 Weight Distribution")
                    fig_dist = go.Figure()
                    fig_dist.add_trace(go.Bar(
                        x=[f"Sim {i+1}" for i in range(len(results))],
                        y=results['Combined_Weight'],
                        marker_color=source_color,
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
                    
                    # ST-DGPA formula reference
                    with st.expander("📐 ST-DGPA Mathematical Formulas", expanded=True):
                        st.markdown("""
                        **Core ST-DGPA Weight Formula:**
                        ```
                        w_i(θ*) = [α_i(θ*) × gating_i(θ*)] / Σ_j[α_j(θ*) × gating_j(θ*)]
                        ```
                        
                        **Where:**
                        - `α_i(θ*)` = Physics attention (inverse distance similarity)
                        - `gating_i(θ*)` = exp(-φ_i² / (2σ_g²)) with φ_i² = Σ[(p*-p_i)/s_p]²
                        - `s_p` = scaling factor for parameter p ∈ {E, τ, t}
                        - `σ_g` = gating kernel width (controls sharpness)
                        
                        **Parameter Proximity Kernel:**
                        ```
                        φ_i = √[((E*-E_i)/s_E)² + ((τ*-τᵢ)/s_τ)² + ((t*-tᵢ)/s_t)²]
                        ```
                        
                        **Physics Attention:**
                        ```
                        αᵢ = 1 / (1 + √φ²)  [simplified inverse distance]
                        ```
                        """)
                        
            except Exception as e:
                st.error("❌ Prediction failed. Please check input parameters and data format.")
                st.exception(e)  # Show full traceback for debugging
                
    else:
        st.info("👈 Generate demo data or upload a CSV file in the sidebar to begin.")
        
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
            - `Max_Temp`: Maximum temperature in Kelvin (optional, for prediction example)
            """)

if __name__ == "__main__":
    main()
