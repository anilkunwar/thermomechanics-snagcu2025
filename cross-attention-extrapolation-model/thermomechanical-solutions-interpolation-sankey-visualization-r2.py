#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ST-DGPA Laser Soldering Interpolation & Sankey Visualizer
===========================================================
Complete integrated application for:
- ST-DGPA (Spatio-Temporal Gated Physics Attention) interpolation
- Interactive, fully customizable Sankey diagram with mathematical hover explanations
- Real-time weight computation, caching, and UI controls
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict, Optional
import hashlib
from datetime import datetime
from collections import OrderedDict

st.set_page_config(page_title="ST-DGPA Sankey Visualizer", layout="wide", initial_sidebar_state="expanded")

# =============================================
# 1. CACHE & SESSION MANAGEMENT
# =============================================
class CacheManager:
    @staticmethod
    def clear_cache():
        if 'cache' in st.session_state: st.session_state.cache.clear()
    
    @staticmethod
    def get(key): return st.session_state.get('cache', {}).get(key)
    
    @staticmethod
    def set(key, value):
        if 'cache' not in st.session_state: st.session_state.cache = OrderedDict()
        st.session_state.cache[key] = {'data': value, 'ts': datetime.now().timestamp()}
        if len(st.session_state.cache) > 20:
            st.session_state.cache.popitem(last=False)

# =============================================
# 2. ST-DGPA PHYSICS INTERPOLATOR
# =============================================
class STDGPAInterpolator:
    def __init__(self, sigma_g=0.20, s_E=10.0, s_tau=5.0, s_t=20.0, temporal_weight=0.3):
        self.sigma_g = sigma_g
        self.s_E = s_E
        self.s_tau = s_tau
        self.s_t = s_t
        self.temporal_weight = temporal_weight

    def compute_weights(self, df: pd.DataFrame, query: Dict) -> pd.DataFrame:
        """Compute ST-DGPA weights: w_i ∝ α_i × gating_i"""
        # Distance calculations
        dE = (df['Energy'] - query['Energy']) / self.s_E
        dTau = (df['Duration'] - query['Duration']) / self.s_tau
        dT = (df['Time'] - query['Time']) / self.s_t
        
        # Physics-aware temporal scaling
        if self.temporal_weight > 0:
            scale_factor = 1.0 + 0.5 * (query['Time'] / max(query['Duration'], 1e-6))
            dT = dT * scale_factor
            
        phi_sq = dE**2 + dTau**2 + dT**2
        
        # Gating Kernel
        gating = np.exp(-phi_sq / (2 * self.sigma_g**2))
        
        # Attention (inverse distance similarity proxy)
        attention = 1.0 / (1.0 + np.sqrt(phi_sq))
        
        # Refinement & Normalization
        refinement = gating * attention
        combined = refinement / (refinement.sum() + 1e-12)
        
        df_out = df.copy()
        df_out['Gating_Weight'] = gating
        df_out['Attention_Score'] = attention
        df_out['Refinement'] = refinement
        df_out['Combined_Weight'] = combined
        return df_out

# =============================================
# 3. ENHANCED SANKEY VISUALIZER (FIXED)
# =============================================
class SankeyVisualizer:
    def __init__(self):
        self.defaults = {
            'node_colors': {'target': '#FF6B6B', 'source': '#9966FF', 
                          'components': ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFD93D', '#36A2EB', '#9966FF']},
            'font_family': 'Arial, sans-serif', 'font_size': 14,
            'node_thickness': 20, 'node_pad': 15,
            'width': 1000, 'height': 700,
            'target_label': 'TARGET'
        }

    def create_stdgpa_sankey(self, sources_data: List[Dict], query: Dict, 
                           customization: Optional[Dict] = None) -> go.Figure:
        cfg = {**self.defaults, **(customization or {})}
        
        # Component labels (math on hover if enabled)
        show_math = cfg.get('show_math', True)
        comp_labels = [
            'Energy Gate\nφ² = ((E*-Eᵢ)/s_E)²', 'Duration Gate\nφ² = ((τ*-τᵢ)/s_τ)²',
            'Time Gate\nφ² = ((t*-tᵢ)/s_t)²', 'Attention\nαᵢ = softmax(Q·Kᵀ/√dₖ)',
            'Refinement\nwᵢ ∝ αᵢ × gatingᵢ', 'Combined\nwᵢ = αᵢ·gatingᵢ / Σ(αⱼ·gatingⱼ)'
        ] if show_math else ['Energy Gate', 'Duration Gate', 'Time Gate', 'Attention', 'Refinement', 'Combined']
        
        # Build nodes
        labels = [cfg['target_label']]
        node_colors = [cfg['node_colors']['target']]
        
        n = len(sources_data)
        for i, row in enumerate(sources_data):
            w = row.get('Combined_Weight', 0)
            op = min(0.3 + w * 0.7, 1.0)
            base = cfg['node_colors']['source']
            r, g, b = int(base[1:3], 16), int(base[3:5], 16), int(base[5:7], 16)
            node_colors.append(f'rgba({r},{g},{b},{op:.2f})')
            labels.append(f"Sim {i+1}\nE:{row.get('Energy',0):.1f} τ:{row.get('Duration',0):.1f}")
            
        comp_colors = cfg['node_colors']['components']
        labels.extend(comp_labels)
        node_colors.extend(comp_colors)
        
        # Build links
        s_idx, t_idx, vals, l_colors, h_texts = [], [], [], [], []
        comp_start = n + 1  # Target(0) + n Sources
        
        for i in range(n):
            src = i + 1
            row = sources_data[i]
            
            # Scaled values for visualization
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
                l_colors.append(comp_colors[c].replace('rgb', 'rgba').replace(')', ', 0.5)'))
                
                # Hover math
                if c == 0: h_texts.append(f"<b>Energy Gate</b><br>S{src} | φ² = ((E*-Eᵢ)/s_E)² = {ve/10:.4f}")
                elif c == 1: h_texts.append(f"<b>Duration Gate</b><br>S{src} | φ² = ((τ*-τᵢ)/s_τ)² = {vτ/10:.4f}")
                elif c == 2: h_texts.append(f"<b>Time Gate</b><br>S{src} | φ² = ((t*-tᵢ)/s_t)² = {vt/10:.4f}")
                elif c == 3: h_texts.append(f"<b>Attention</b><br>S{src} | αᵢ = softmax(QKᵀ/√dₖ)<br>Score: {row.get('Attention_Score',0):.4f}")
                elif c == 4: h_texts.append(f"<b>Refinement</b><br>S{src} | wᵢ ∝ αᵢ·gatingᵢ<br>Ref: {row.get('Refinement',0):.4f}")
                else: h_texts.append(f"<b>Combined Weight</b><br>S{src} | wᵢ = (αᵢ·gatingᵢ)/Σ(...)<br>Weight: {row.get('Combined_Weight',0):.4f}")
                
        # Components to Target
        for c in range(6):
            s_idx.append(comp_start + c)
            t_idx.append(0)
            flow_in = sum(v for s, t, v in zip(s_idx[:-6], t_idx[:-6], vals[:-6]) if t == comp_start + c)
            vals.append(flow_in * 0.5)
            l_colors.append('rgba(153,102,255,0.6)')
            h_texts.append(f"<b>Aggregation</b><br>{comp_labels[c]} → TARGET<br>Total: {flow_in:.3f}")
            
        # FIXED: Removed invalid top-level hoverinfo='text'
        fig = go.Figure(go.Sankey(
            node=dict(pad=cfg['node_pad'], thickness=cfg['node_thickness'],
                     line=dict(color="black", width=0.5), label=labels, color=node_colors,
                     font=dict(family=cfg['font_family'], size=cfg['font_size']),
                     hovertemplate='<b>%{label}</b><br>Value: %{value:.3f}<extra></extra>'),
            link=dict(source=s_idx, target=t_idx, value=vals, color=l_colors,
                     hovertext=h_texts, hovertemplate='%{hovertext}<extra></extra>',
                     line=dict(width=0.5, color='rgba(255,255,255,0.3)')),
        ))
        
        fig.update_layout(
            title=dict(text=f"<b>ST-DGPA Attention Flow</b><br>E={query['Energy']}, τ={query['Duration']}, t={query['Time']}",
                      font=dict(family=cfg['font_family'], size=cfg['font_size']+2), x=0.5),
            font=dict(family=cfg['font_family'], size=cfg['font_size']),
            width=cfg['width'], height=cfg['height'],
            hoverlabel=dict(font=dict(family=cfg['font_family'], size=cfg['font_size']),
                           bgcolor='rgba(44,62,80,0.9)', namelength=-1),
            margin=dict(t=100, l=50, r=50, b=50)
        )
        return fig

# =============================================
# 4. MAIN STREAMLIT APPLICATION
# =============================================
def main():
    st.title("🔬 ST-DGPA Interpolation & Sankey Visualizer")
    st.markdown("Interactive visualization of Spatio-Temporal Gated Physics Attention (ST-DGPA) weights.")
    
    # Session State Init
    if 'sankey_viz' not in st.session_state: st.session_state.sankey_viz = SankeyVisualizer()
    if 'interpolator' not in st.session_state: st.session_state.interpolator = STDGPAInterpolator()
    if 'df_sources' not in st.session_state: st.session_state.df_sources = None
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ ST-DGPA Parameters")
        sigma_g = st.slider("Gating Width (σ_g)", 0.05, 1.0, 0.20, 0.05)
        s_E = st.slider("Energy Scale (s_E)", 0.1, 50.0, 10.0, 0.5)
        s_tau = st.slider("Duration Scale (s_τ)", 0.1, 20.0, 5.0, 0.5)
        s_t = st.slider("Time Scale (s_t)", 1.0, 50.0, 20.0, 1.0)
        st.session_state.interpolator.sigma_g = sigma_g
        st.session_state.interpolator.s_E = s_E
        st.session_state.interpolator.s_tau = s_tau
        st.session_state.interpolator.s_t = s_t
        
        st.markdown("---")
        st.header("🎨 Visualization Settings")
        font_family = st.selectbox("Font", ["Arial, sans-serif", "Courier New, monospace", "Times New Roman, serif"], index=0)
        font_size = st.slider("Font Size", 8, 20, 14, 1)
        node_thick = st.slider("Node Thickness", 10, 40, 20, 2)
        fig_w = st.slider("Width", 600, 1400, 1000, 50)
        fig_h = st.slider("Height", 400, 1000, 700, 50)
        show_math = st.checkbox("Show Math on Hover", True)
        
        st.markdown("**Node Colors**")
        c1, c2 = st.columns(2)
        with c1: target_color = st.color_picker("Target", "#FF6B6B")
        with c2: src_color = st.color_picker("Source", "#9966FF")
        comp_color = st.color_picker("Components", "#4ECDC4")
        
        st.markdown("---")
        st.header("📊 Data")
        n_sims = st.slider("Demo Simulations", 2, 20, 4, 1)
        if st.button("🔄 Generate Demo Data", use_container_width=True):
            np.random.seed(42)
            st.session_state.df_sources = pd.DataFrame({
                'Energy': np.random.uniform(0.5, 8.0, n_sims),
                'Duration': np.random.uniform(2.0, 7.0, n_sims),
                'Time': np.random.uniform(1.0, 10.0, n_sims),
                'Max_Temp': np.random.uniform(500, 1500, n_sims)
            })
            st.success(f"Generated {n_sims} simulations!")
            
        up = st.file_uploader("Upload CSV", type=["csv"])
        if up:
            try:
                st.session_state.df_sources = pd.read_csv(up)
                st.success(f"Loaded {len(st.session_state.df_sources)} rows")
            except Exception as e: st.error(str(e))
            
        if st.session_state.df_sources is not None:
            st.info(f"✅ {len(st.session_state.df_sources)} simulations loaded")

    # Main Area
    if st.session_state.df_sources is not None:
        df = st.session_state.df_sources
        st.subheader("🎯 Target Query")
        c1, c2, c3 = st.columns(3)
        with c1: qE = st.number_input("Energy (mJ)", 0.1, 20.0, 3.5, 0.1)
        with c2: qτ = st.number_input("Duration (ns)", 0.1, 20.0, 4.2, 0.1)
        with c3: qt = st.number_input("Time (ns)", 0.1, 20.0, 5.0, 0.1)
        query = {'Energy': qE, 'Duration': qτ, 'Time': qt}
        
        cust = {
            'font_family': font_family, 'font_size': font_size, 'node_thickness': node_thick,
            'width': fig_w, 'height': fig_h, 'show_math': show_math,
            'node_colors': {'target': target_color, 'source': src_color, 
                          'components': [comp_color, '#4ECDC4', '#95E1D3', '#FFD93D', '#36A2EB', '#9966FF']}
        }
        
        if st.button("🚀 Compute ST-DGPA", type="primary", use_container_width=True):
            with st.spinner("Computing weights..."):
                # Compute weights
                df_w = st.session_state.interpolator.compute_weights(df.copy(), query)
                
                # Display Top Sources
                st.markdown("### 📊 Top Contributing Sources")
                top = df_w.nlargest(5, 'Combined_Weight')
                st.dataframe(top.style.format({'Energy':'{:.2f}', 'Duration':'{:.2f}', 'Time':'{:.2f}', 'Combined_Weight':'{:.4f}'})
                             .highlight_max(subset=['Combined_Weight'], color='#90EE90'), use_container_width=True)
                             
                # Prediction
                if 'Max_Temp' in df_w.columns:
                    pred_temp = (df_w['Combined_Weight'] * df_w['Max_Temp']).sum()
                    st.metric("Predicted Max Temp", f"{pred_temp:.1f} K")
                else:
                    st.info("⚠️ `Max_Temp` column missing. Skipping prediction.")
                    
                # Sankey
                st.markdown("### 🕸️ Attention Weight Flow")
                sources = df_w.to_dict('records')
                fig = st.session_state.sankey_viz.create_stdgpa_sankey(sources, query, cust)
                st.plotly_chart(fig, use_container_width=True)
                
                # Math Formulas
                with st.expander("📐 ST-DGPA Formulas"):
                    st.markdown("""
                    **Core:** `w_i = [α_i × gating_i] / Σ_j[α_j × gating_j]`
                    **Gating:** `gating_i = exp(-φ_i² / (2σ_g²))`
                    `φ_i = √[((E*-E_i)/s_E)² + ((τ*-τ_i)/s_τ)² + ((t*-t_i)/s_t)²]`
                    **Attention:** `α_i = softmax(Q·Kᵀ / √dₖ)`
                    """)
    else:
        st.info("👈 Generate demo data or upload CSV in the sidebar to begin.")
        with st.expander("📋 Expected CSV Format"):
            st.markdown("""
            ```csv
            Energy,Duration,Time,Max_Temp
            0.5,2.0,1.0,520.5
            1.2,3.5,2.5,680.3
            ```
            Columns: `Energy` (mJ), `Duration` (ns), `Time` (ns), `Max_Temp` (K, optional)
            """)

if __name__ == "__main__":
    main()
