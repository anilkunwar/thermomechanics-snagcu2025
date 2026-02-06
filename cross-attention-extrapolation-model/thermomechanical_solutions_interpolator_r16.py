# =============================================
# SPATIO-TEMPORAL DGPA WITH GATED TEMPORAL ATTENTION
# =============================================
class SpatioTemporalDGPAExtrapolator:
    """Enhanced DGPA with gated temporal attention for heat transfer characterization"""
    
    def __init__(self, sigma_param=0.3, spatial_weight=0.5, n_heads=4, temperature=1.0,
                 sigma_g=0.20, sigma_t=0.15,  # Separate gating widths for space and time
                 s_E=10.0, s_tau=5.0, s_t=20.0,  # Scaling factors for E, τ, t
                 temporal_gating_mode='adaptive',  # 'fixed', 'adaptive', 'physics_informed'
                 thermal_diffusivity=1e-5,  # α [m²/s] - material property
                 characteristic_length=50e-6):  # L [m] - laser spot radius
        
        # Base attention parameters
        self.sigma_param = sigma_param
        self.spatial_weight = spatial_weight
        self.n_heads = n_heads
        self.temperature = temperature
        
        # DGPA gating parameters
        self.sigma_g = sigma_g  # Spatial (E, τ) gating width
        self.sigma_t = sigma_t  # Temporal gating width
        
        # Scaling factors
        self.s_E = s_E          # Energy scaling [mJ]
        self.s_tau = s_tau      # Duration scaling [ns]
        self.s_t = s_t          # Time scaling [ns]
        
        # Heat transfer characterization
        self.temporal_gating_mode = temporal_gating_mode
        self.thermal_diffusivity = thermal_diffusivity  # α [m²/s]
        self.characteristic_length = characteristic_length  # L [m]
        
        # Derived thermal parameters
        self.thermal_time_constant = characteristic_length**2 / thermal_diffusivity  # L²/α [s]
        self.thermal_time_constant_ns = self.thermal_time_constant * 1e9  # Convert to ns
        
        # Data structures
        self.source_db = []
        self.embedding_scaler = StandardScaler()
        self.value_scaler = StandardScaler()
        self.source_embeddings = []
        self.source_values = []
        self.source_metadata = []
        self.fitted = False
        
        # Temporal attention cache
        self.temporal_similarity_cache = {}
        self.last_gating_analysis = None
    
    def load_summaries(self, summaries):
        """Load summary statistics and prepare for ST-DGPA attention"""
        self.source_db = summaries
        
        if not summaries:
            return
        
        # Prepare embeddings and values with enhanced temporal features
        all_embeddings = []
        all_values = []
        metadata = []
        
        for summary_idx, summary in enumerate(summaries):
            for timestep_idx, t in enumerate(summary['timesteps']):
                # Compute enhanced physics-aware embedding with temporal features
                emb = self._compute_spatiotemporal_embedding(
                    summary['energy'], 
                    summary['duration'], 
                    t
                )
                all_embeddings.append(emb)
                
                # Extract field values
                field_vals = []
                for field in sorted(summary['field_stats'].keys()):
                    stats = summary['field_stats'][field]
                    if timestep_idx < len(stats['mean']):
                        field_vals.extend([
                            stats['mean'][timestep_idx],
                            stats['max'][timestep_idx],
                            stats['std'][timestep_idx]
                        ])
                    else:
                        field_vals.extend([0.0, 0.0, 0.0])
                
                all_values.append(field_vals)
                
                # Store metadata with thermal characteristics
                metadata.append({
                    'summary_idx': summary_idx,
                    'timestep_idx': timestep_idx,
                    'energy': summary['energy'],
                    'duration': summary['duration'],
                    'time': t,
                    'name': summary['name'],
                    # Add thermal characteristics
                    'fourier_number': self._compute_fourier_number(t),
                    'thermal_penetration': self._compute_thermal_penetration(t),
                    'heating_phase': self._classify_heating_phase(t, summary['duration'])
                })
        
        if all_embeddings and all_values:
            all_embeddings = np.array(all_embeddings)
            all_values = np.array(all_values)
            
            # Scale embeddings
            self.embedding_scaler.fit(all_embeddings)
            self.source_embeddings = self.embedding_scaler.transform(all_embeddings)
            
            # Scale values
            self.value_scaler.fit(all_values)
            self.source_values = all_values
            
            self.source_metadata = metadata
            self.fitted = True
            
            st.info(f"✅ Prepared {len(all_embeddings)} embeddings with {all_embeddings.shape[1]} spatio-temporal features")
    
    def _compute_spatiotemporal_embedding(self, energy, duration, time):
        """Compute comprehensive spatio-temporal embedding for heat transfer"""
        # Basic physical parameters
        logE = np.log1p(energy)
        power = energy / max(duration, 1e-6)
        energy_density = energy / (duration**2 + 1e-6)
        
        # Dimensionless parameters
        time_ratio = time / max(duration, 1e-3)
        heating_rate = power / max(time, 1e-6)
        cooling_rate = 1.0 / (time + 1e-6)
        
        # Thermal diffusion features
        thermal_diffusion = np.sqrt(time * 0.1) / max(duration, 1e-3)
        thermal_penetration = self._compute_thermal_penetration(time)
        fourier_number = self._compute_fourier_number(time)
        
        # Strain rate proxies
        strain_rate = energy_density / (time + 1e-6)
        stress_rate = power / (time + 1e-6)
        
        # Temporal features for heat transfer
        log_time = np.log1p(time)
        sqrt_time = np.sqrt(time)
        time_over_tau = time / max(duration, 1e-3)
        exp_decay = np.exp(-time_over_tau) if time_over_tau > 0 else 1.0
        
        # Heating phase classification
        heating_phase = self._classify_heating_phase(time, duration)
        
        # Combined features
        return np.array([
            logE,
            duration,
            time,
            power,
            energy_density,
            time_ratio,
            heating_rate,
            cooling_rate,
            thermal_diffusion,
            thermal_penetration,
            strain_rate,
            stress_rate,
            log_time,
            sqrt_time,
            fourier_number,
            heating_phase,
            exp_decay,
            np.log1p(power),
            np.log1p(time)
        ], dtype=np.float32)
    
    def _compute_fourier_number(self, time):
        """Compute Fourier number (dimensionless time) for heat transfer"""
        # Fo = αt / L²
        time_s = time * 1e-9  # Convert ns to s
        return self.thermal_diffusivity * time_s / (self.characteristic_length**2)
    
    def _compute_thermal_penetration(self, time):
        """Compute thermal penetration depth"""
        # δ ~ √(αt)
        time_s = time * 1e-9  # Convert ns to s
        return np.sqrt(self.thermal_diffusivity * time_s)
    
    def _classify_heating_phase(self, time, duration):
        """Classify heating phase for temporal gating"""
        if time < 0.1 * duration:
            return 0.0  # Very early heating
        elif time < duration:
            return 1.0  # Active heating
        elif time < 2 * duration:
            return 2.0  # Early cooling
        else:
            return 3.0  # Late cooling/diffusion
    
    def _compute_spatiotemporal_gating(self, energy_query, duration_query, time_query):
        """Compute ST-DGPA gating kernel in (E, τ, t) space
        
        φ_i = sqrt( ((E* - E_i)/s_E)^2 + ((τ* - τ_i)/s_τ)^2 + ((t* - t_i)/s_t)^2 )
        gating_i = exp( - (φ_i^2) / (2 * sigma^2) )
        
        Different gating modes:
        1. 'fixed': Use sigma_g for spatial, sigma_t for temporal
        2. 'adaptive': Adjust sigma_t based on time query
        3. 'physics_informed': Use thermal diffusion scaling
        """
        if not self.source_metadata:
            return np.array([])
        
        phi_squared_spatial = []
        phi_squared_temporal = []
        
        for meta in self.source_metadata:
            # Spatial (E, τ) differences
            de = (energy_query - meta['energy']) / self.s_E
            dtau = (duration_query - meta['duration']) / self.s_tau
            
            # Temporal difference with physics-aware scaling
            dt = (time_query - meta['time'])
            
            # Apply physics-informed temporal scaling
            if self.temporal_gating_mode == 'physics_informed':
                # Scale temporal difference by thermal time constant
                s_t_effective = self.s_t * (1.0 + meta['fourier_number'])
            elif self.temporal_gating_mode == 'adaptive':
                # Wider gating for diffusion-dominated regimes
                if time_query > 2 * duration_query:  # Late time
                    s_t_effective = self.s_t * 2.0
                else:
                    s_t_effective = self.s_t
            else:  # 'fixed'
                s_t_effective = self.s_t
            
            dt_scaled = dt / s_t_effective
            
            phi_squared_spatial.append(de**2 + dtau**2)
            phi_squared_temporal.append(dt_scaled**2)
        
        phi_squared_spatial = np.array(phi_squared_spatial)
        phi_squared_temporal = np.array(phi_squared_temporal)
        
        # Apply separate gating for spatial and temporal dimensions
        if self.temporal_gating_mode == 'combined':
            # Combined 3D gating
            phi_squared_total = phi_squared_spatial + phi_squared_temporal
            gating = np.exp(-phi_squared_total / (2 * self.sigma_g**2))
        else:
            # Separate gating with different widths
            gating_spatial = np.exp(-phi_squared_spatial / (2 * self.sigma_g**2))
            gating_temporal = np.exp(-phi_squared_temporal / (2 * self.sigma_t**2))
            gating = gating_spatial * gating_temporal
        
        # Normalize gating weights
        gating_sum = np.sum(gating)
        if gating_sum > 0:
            gating = gating / gating_sum
        else:
            gating = np.ones_like(gating) / len(gating)
        
        # Store analysis for visualization
        self.last_gating_analysis = {
            'gating_spatial': gating_spatial if 'gating_spatial' in locals() else None,
            'gating_temporal': gating_temporal if 'gating_temporal' in locals() else None,
            'gating_combined': gating,
            'phi_spatial': np.sqrt(phi_squared_spatial),
            'phi_temporal': np.sqrt(phi_squared_temporal),
            's_t_effective': s_t_effective if 's_t_effective' in locals() else self.s_t
        }
        
        return gating
    
    def _compute_anisotropic_gating(self, energy_query, duration_query, time_query):
        """Anisotropic gating with dimension-specific widths"""
        if not self.source_metadata:
            return np.array([])
        
        # Different widths for each dimension
        sigma_E = self.sigma_g * 1.2  # Slightly wider for energy
        sigma_tau = self.sigma_g * 0.8  # Tighter for duration
        sigma_t = self.sigma_t  # Temporal width
        
        gating_components = []
        
        for meta in self.source_metadata:
            # Dimension-specific gating
            g_E = np.exp(-((energy_query - meta['energy'])/self.s_E)**2 / (2 * sigma_E**2))
            g_tau = np.exp(-((duration_query - meta['duration'])/self.s_tau)**2 / (2 * sigma_tau**2))
            g_t = np.exp(-((time_query - meta['time'])/self.s_t)**2 / (2 * sigma_t**2))
            
            # Combine with weights
            gating_components.append(g_E * g_tau * g_t)
        
        gating = np.array(gating_components)
        gating_sum = np.sum(gating)
        
        if gating_sum > 0:
            gating = gating / gating_sum
        else:
            gating = np.ones_like(gating) / len(gating)
        
        return gating
    
    def _multi_head_attention_with_temporal_gating(self, query_embedding, query_meta):
        """Multi-head attention with spatio-temporal DGPA"""
        if not self.fitted or len(self.source_embeddings) == 0:
            return None, None, None, None
        
        # Normalize query embedding
        query_norm = self.embedding_scaler.transform([query_embedding])[0]
        
        n_sources = len(self.source_embeddings)
        head_weights = np.zeros((self.n_heads, n_sources))
        
        # Multi-head attention
        for head in range(self.n_heads):
            np.random.seed(42 + head)
            proj_dim = min(8, query_norm.shape[0])
            proj_matrix = np.random.randn(query_norm.shape[0], proj_dim)
            
            # Project embeddings
            query_proj = query_norm @ proj_matrix
            source_proj = self.source_embeddings @ proj_matrix
            
            # Compute attention scores
            distances = np.linalg.norm(query_proj - source_proj, axis=1)
            scores = np.exp(-distances**2 / (2 * self.sigma_param**2))
            
            # Apply spatial regulation
            if self.spatial_weight > 0:
                spatial_sim = self._compute_spatial_similarity(query_meta, self.source_metadata)
                scores = (1 - self.spatial_weight) * scores + self.spatial_weight * spatial_sim
            
            head_weights[head] = scores
        
        # Combine head weights
        avg_weights = np.mean(head_weights, axis=0)
        
        # Apply temperature scaling
        if self.temperature != 1.0:
            avg_weights = avg_weights ** (1.0 / self.temperature)
        
        # Softmax for physics attention
        max_weight = np.max(avg_weights)
        exp_weights = np.exp(avg_weights - max_weight)
        physics_attention = exp_weights / (np.sum(exp_weights) + 1e-12)
        
        # Apply spatio-temporal DGPA gating
        st_gating = self._compute_spatiotemporal_gating(
            query_meta['energy'], 
            query_meta['duration'], 
            query_meta['time']
        )
        
        # ST-DGPA combination
        combined_weights = physics_attention * st_gating
        combined_sum = np.sum(combined_weights)
        
        if combined_sum > 1e-12:
            final_weights = combined_weights / combined_sum
        else:
            final_weights = physics_attention
        
        # Weighted prediction
        if len(self.source_values) > 0:
            prediction = np.sum(final_weights[:, np.newaxis] * self.source_values, axis=0)
        else:
            prediction = np.zeros(1)
        
        return prediction, final_weights, physics_attention, st_gating
    
    def predict_field_statistics(self, energy_query, duration_query, time_query):
        """Predict field statistics using ST-DGPA"""
        if not self.fitted:
            return None
        
        # Compute query embedding and metadata
        query_embedding = self._compute_spatiotemporal_embedding(
            energy_query, duration_query, time_query
        )
        query_meta = {
            'energy': energy_query,
            'duration': duration_query,
            'time': time_query,
            'fourier_number': self._compute_fourier_number(time_query),
            'thermal_penetration': self._compute_thermal_penetration(time_query),
            'heating_phase': self._classify_heating_phase(time_query, duration_query)
        }
        
        # Apply ST-DGPA attention mechanism
        prediction, final_weights, physics_attention, st_gating = \
            self._multi_head_attention_with_temporal_gating(query_embedding, query_meta)
        
        if prediction is None:
            return None
        
        # Reconstruct field predictions
        result = {
            'prediction': prediction,
            'attention_weights': final_weights,
            'physics_attention': physics_attention,
            'st_gating': st_gating,
            'confidence': float(np.max(final_weights)) if len(final_weights) > 0 else 0.0,
            'field_predictions': {},
            'temporal_analysis': self._analyze_temporal_aspects(
                query_meta, physics_attention, st_gating
            )
        }
        
        # Map predictions back to fields
        if self.source_db:
            field_order = sorted(self.source_db[0]['field_stats'].keys())
            n_stats_per_field = 3
            
            for i, field in enumerate(field_order):
                start_idx = i * n_stats_per_field
                if start_idx + 2 < len(prediction):
                    result['field_predictions'][field] = {
                        'mean': float(prediction[start_idx]),
                        'max': float(prediction[start_idx + 1]),
                        'std': float(prediction[start_idx + 2])
                    }
        
        return result
    
    def _analyze_temporal_aspects(self, query_meta, physics_attention, st_gating):
        """Analyze temporal aspects of the prediction"""
        if not self.source_metadata:
            return {}
        
        # Compute temporal statistics
        times = np.array([meta['time'] for meta in self.source_metadata])
        fourier_numbers = np.array([meta['fourier_number'] for meta in self.source_metadata])
        heating_phases = np.array([meta['heating_phase'] for meta in self.source_metadata])
        
        # Weighted temporal statistics
        time_mean = np.average(times, weights=st_gating)
        time_std = np.sqrt(np.average((times - time_mean)**2, weights=st_gating))
        
        fourier_mean = np.average(fourier_numbers, weights=st_gating)
        
        # Phase matching
        query_phase = query_meta['heating_phase']
        phase_match = np.sum(st_gating[heating_phases == query_phase])
        
        # Temporal spread metric
        temporal_spread = np.std(times[st_gating > np.percentile(st_gating, 50)])
        
        return {
            'time_mean': float(time_mean),
            'time_std': float(time_std),
            'fourier_mean': float(fourier_mean),
            'phase_match': float(phase_match),
            'temporal_spread': float(temporal_spread),
            'query_phase': int(query_phase),
            'effective_sources': int(np.sum(st_gating > 0.01))
        }
    
    def predict_time_series(self, energy_query, duration_query, time_points):
        """Predict over time series with temporal gating analysis"""
        results = {
            'time_points': time_points,
            'field_predictions': {},
            'attention_maps': [],
            'physics_attention_maps': [],
            'st_gating_maps': [],
            'confidence_scores': [],
            'temporal_analyses': []
        }
        
        # Initialize field predictions
        if self.source_db:
            common_fields = set()
            for summary in self.source_db:
                common_fields.update(summary['field_stats'].keys())
            
            for field in common_fields:
                results['field_predictions'][field] = {
                    'mean': [], 'max': [], 'std': []
                }
        
        for t in time_points:
            pred = self.predict_field_statistics(energy_query, duration_query, t)
            
            if pred and 'field_predictions' in pred:
                for field in pred['field_predictions']:
                    if field in results['field_predictions']:
                        stats = pred['field_predictions'][field]
                        results['field_predictions'][field]['mean'].append(stats['mean'])
                        results['field_predictions'][field]['max'].append(stats['max'])
                        results['field_predictions'][field]['std'].append(stats['std'])
                
                results['attention_maps'].append(pred['attention_weights'])
                results['physics_attention_maps'].append(pred['physics_attention'])
                results['st_gating_maps'].append(pred['st_gating'])
                results['confidence_scores'].append(pred['confidence'])
                results['temporal_analyses'].append(pred.get('temporal_analysis', {}))
            else:
                # Fill with NaN if prediction failed
                for field in results['field_predictions']:
                    results['field_predictions'][field]['mean'].append(np.nan)
                    results['field_predictions'][field]['max'].append(np.nan)
                    results['field_predictions'][field]['std'].append(np.nan)
                results['attention_maps'].append(np.array([]))
                results['physics_attention_maps'].append(np.array([]))
                results['st_gating_maps'].append(np.array([]))
                results['confidence_scores'].append(0.0)
                results['temporal_analyses'].append({})
        
        return results

# =============================================
# ENHANCED VISUALIZATION FOR TEMPORAL GATING
# =============================================
class TemporalGatingVisualizer(EnhancedVisualizer):
    """Extended visualizer for temporal gating analysis"""
    
    @staticmethod
    def create_temporal_gating_analysis(results, energy_query, duration_query, time_points):
        """Create comprehensive temporal gating analysis visualizations"""
        if not results or 'st_gating_maps' not in results or len(results['st_gating_maps']) == 0:
            return None
        
        # Select timestep for detailed analysis
        timestep_idx = len(time_points) // 2
        time = time_points[timestep_idx]
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                "ST-DGPA Weights vs Time", "Temporal Gating Contribution", 
                "Source Time Distribution", "Fourier Number Analysis",
                "Heating Phase Matching", "Temporal Spread Analysis"
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Get data for selected timestep
        final_weights = results['attention_maps'][timestep_idx]
        st_gating = results['st_gating_maps'][timestep_idx]
        temporal_analysis = results['temporal_analyses'][timestep_idx]
        
        # 1. ST-DGPA weights vs time
        if len(final_weights) > 0:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(final_weights))),
                    y=final_weights,
                    mode='lines+markers',
                    name='ST-DGPA Weights',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
        
        # 2. Temporal gating contribution
        if len(st_gating) > 0:
            fig.add_trace(
                go.Bar(
                    x=list(range(len(st_gating))),
                    y=st_gating,
                    name='Temporal Gating',
                    marker_color='orange',
                    opacity=0.7
                ),
                row=1, col=2
            )
        
        # 3. Source time distribution (if metadata available)
        if st.session_state.get('extrapolator') and st.session_state.extrapolator.source_metadata:
            source_times = [meta['time'] for meta in st.session_state.extrapolator.source_metadata]
            
            # Weight times by gating
            weighted_times = []
            for i, (t, g) in enumerate(zip(source_times, st_gating)):
                if g > 0.01:  # Only show sources with significant gating
                    weighted_times.extend([t] * int(g * 100))
            
            if weighted_times:
                fig.add_trace(
                    go.Histogram(
                        x=weighted_times,
                        nbinsx=20,
                        name='Weighted Time Distribution',
                        marker_color='green',
                        opacity=0.6
                    ),
                    row=1, col=3
                )
        
        # 4. Fourier number analysis
        if 'fourier_mean' in temporal_analysis:
            # Show Fourier number evolution
            fourier_numbers = []
            for analysis in results['temporal_analyses']:
                if 'fourier_mean' in analysis:
                    fourier_numbers.append(analysis['fourier_mean'])
            
            if fourier_numbers:
                fig.add_trace(
                    go.Scatter(
                        x=time_points[:len(fourier_numbers)],
                        y=fourier_numbers,
                        mode='lines+markers',
                        name='Fourier Number',
                        line=dict(color='red', width=2)
                    ),
                    row=2, col=1
                )
        
        # 5. Heating phase matching
        if 'phase_match' in temporal_analysis:
            phase_matches = []
            for analysis in results['temporal_analyses']:
                if 'phase_match' in analysis:
                    phase_matches.append(analysis['phase_match'])
            
            if phase_matches:
                fig.add_trace(
                    go.Scatter(
                        x=time_points[:len(phase_matches)],
                        y=phase_matches,
                        mode='lines+markers',
                        name='Phase Match',
                        line=dict(color='purple', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(128, 0, 128, 0.2)'
                    ),
                    row=2, col=2
                )
        
        # 6. Temporal spread analysis
        if 'temporal_spread' in temporal_analysis:
            temporal_spreads = []
            for analysis in results['temporal_analyses']:
                if 'temporal_spread' in analysis:
                    temporal_spreads.append(analysis['temporal_spread'])
            
            if temporal_spreads:
                fig.add_trace(
                    go.Scatter(
                        x=time_points[:len(temporal_spreads)],
                        y=temporal_spreads,
                        mode='lines+markers',
                        name='Temporal Spread',
                        line=dict(color='brown', width=2)
                    ),
                    row=2, col=3
                )
        
        fig.update_layout(
            height=800,
            title_text=f"Temporal Gating Analysis at t={time} ns",
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Source Index", row=1, col=1)
        fig.update_yaxes(title_text="Weight", row=1, col=1)
        fig.update_xaxes(title_text="Source Index", row=1, col=2)
        fig.update_yaxes(title_text="Gating Weight", row=1, col=2)
        fig.update_xaxes(title_text="Time [ns]", row=1, col=3)
        fig.update_yaxes(title_text="Count", row=1, col=3)
        fig.update_xaxes(title_text="Time [ns]", row=2, col=1)
        fig.update_yaxes(title_text="Fourier Number", row=2, col=1)
        fig.update_xaxes(title_text="Time [ns]", row=2, col=2)
        fig.update_yaxes(title_text="Phase Match", row=2, col=2)
        fig.update_xaxes(title_text="Time [ns]", row=2, col=3)
        fig.update_yaxes(title_text="Temporal Spread [ns]", row=2, col=3)
        
        return fig
    
    @staticmethod
    def create_3d_spatiotemporal_visualization(results, energy_query, duration_query, time_points):
        """Create 3D visualization of spatio-temporal attention"""
        if not results or 'st_gating_maps' not in results or not st.session_state.get('extrapolator'):
            return go.Figure()
        
        # Get source metadata
        metadata = st.session_state.extrapolator.source_metadata
        if not metadata:
            return go.Figure()
        
        # Select a timestep
        timestep_idx = len(time_points) // 2
        time = time_points[timestep_idx]
        gating_weights = results['st_gating_maps'][timestep_idx]
        
        # Extract coordinates
        energies = []
        durations = []
        times = []
        fourier_numbers = []
        
        for meta in metadata:
            energies.append(meta['energy'])
            durations.append(meta['duration'])
            times.append(meta['time'])
            fourier_numbers.append(meta.get('fourier_number', 0))
        
        # Create 3D scatter plot
        fig = go.Figure(data=go.Scatter3d(
            x=energies,
            y=durations,
            z=times,
            mode='markers',
            marker=dict(
                size=10,
                color=gating_weights,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title="Temporal Gating Weight"),
                line=dict(width=2, color='white')
            ),
            text=[
                f"E: {e:.1f} mJ<br>τ: {d:.1f} ns<br>t: {t:.1f} ns<br>Fo: {fo:.3f}<br>Weight: {w:.4f}"
                for e, d, t, fo, w in zip(energies, durations, times, fourier_numbers, gating_weights)
            ],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # Add query point
        fig.add_trace(go.Scatter3d(
            x=[energy_query],
            y=[duration_query],
            z=[time],
            mode='markers',
            marker=dict(
                size=15,
                color='red',
                symbol='diamond',
                line=dict(width=3, color='white')
            ),
            name='Query Point',
            hovertemplate=f'Query<br>E: {energy_query:.1f} mJ<br>τ: {duration_query:.1f} ns<br>t: {time:.1f} ns<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Spatio-Temporal Gating at t={time} ns",
            scene=dict(
                xaxis_title="Energy [mJ]",
                yaxis_title="Duration [ns]",
                zaxis_title="Time [ns]",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            height=600,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig

# =============================================
# UPDATE MAIN APPLICATION WITH TEMPORAL GATING
# =============================================
def update_main_application():
    """Update the main application to include temporal gating options"""
    
    # In the main() function, update the extrapolator initialization:
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = UnifiedFEADataLoader()
        
        # Use SpatioTemporalDGPAExtrapolator instead of basic DGPA
        st.session_state.extrapolator = SpatioTemporalDGPAExtrapolator(
            sigma_param=0.3, spatial_weight=0.5, n_heads=4, temperature=1.0,
            sigma_g=0.20, sigma_t=0.15,
            s_E=10.0, s_tau=5.0, s_t=20.0,
            temporal_gating_mode='physics_informed',
            thermal_diffusivity=1e-5,  # α for steel ~1e-5 m²/s
            characteristic_length=50e-6  # 50 μm spot radius
        )
        
        st.session_state.visualizer = TemporalGatingVisualizer()
        st.session_state.data_loaded = False
        st.session_state.current_mode = "Data Viewer"
        st.session_state.selected_colormap = "Viridis"
        st.session_state.interpolation_results = None
        st.session_state.interpolation_params = None
        st.session_state.interpolation_3d_cache = {}
        st.session_state.interpolation_field_history = OrderedDict()
        st.session_state.current_3d_field = None
        st.session_state.current_3d_timestep = 0
        st.session_state.last_prediction_id = None
    
    # Add temporal gating controls to the sidebar
    def add_temporal_gating_controls():
        with st.sidebar.expander("⏱️ Temporal Gating Settings", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                sigma_t = st.slider(
                    "Temporal Gating Width (σ_t)",
                    min_value=0.05,
                    max_value=1.0,
                    value=0.15,
                    step=0.05,
                    help="Sharpness of temporal gating kernel"
                )
                
                s_t = st.slider(
                    "Time Scaling (s_t) [ns]",
                    min_value=1.0,
                    max_value=100.0,
                    value=20.0,
                    step=1.0,
                    help="Scaling factor for time differences"
                )
            
            with col2:
                temporal_mode = st.selectbox(
                    "Temporal Gating Mode",
                    ['fixed', 'adaptive', 'physics_informed', 'combined'],
                    index=2,
                    help="Mode for temporal gating calculation"
                )
                
                thermal_diffusivity = st.number_input(
                    "Thermal Diffusivity α [m²/s]",
                    min_value=1e-7,
                    max_value=1e-3,
                    value=1e-5,
                    format="%.2e",
                    help="Material thermal diffusivity"
                )
            
            characteristic_length = st.slider(
                "Characteristic Length L [μm]",
                min_value=1.0,
                max_value=200.0,
                value=50.0,
                step=5.0,
                help="Laser spot radius or thermal length scale"
            )
            
            # Update extrapolator parameters
            if st.button("Apply Temporal Settings", use_container_width=True):
                st.session_state.extrapolator.sigma_t = sigma_t
                st.session_state.extrapolator.s_t = s_t
                st.session_state.extrapolator.temporal_gating_mode = temporal_mode
                st.session_state.extrapolator.thermal_diffusivity = thermal_diffusivity
                st.session_state.extrapolator.characteristic_length = characteristic_length * 1e-6
                st.success("✅ Temporal gating settings updated")
    
    # Update the render_interpolation_extrapolation function
    def update_interpolation_interface():
        """Add temporal gating controls to interpolation interface"""
        
        # In the model parameters expander, add temporal gating controls:
        with st.expander("⏱️ Spatio-Temporal DGPA Parameters", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                sigma_param = st.slider(
                    "Kernel Width (σ)",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.3,
                    step=0.05,
                    key="interp_sigma"
                )
            with col2:
                spatial_weight = st.slider(
                    "Spatial Weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    key="interp_spatial"
                )
            with col3:
                n_heads = st.slider(
                    "Attention Heads",
                    min_value=1,
                    max_value=8,
                    value=4,
                    step=1,
                    key="interp_heads"
                )
            with col4:
                temperature = st.slider(
                    "Temperature",
                    min_value=0.1,
                    max_value=2.0,
                    value=1.0,
                    step=0.1,
                    key="interp_temp"
                )
            
            st.markdown("### 🎯 Spatial Gating Parameters")
            col5, col6, col7 = st.columns(3)
            with col5:
                sigma_g = st.slider(
                    "Spatial Gating Width (σ_g)",
                    min_value=0.05,
                    max_value=1.0,
                    value=0.20,
                    step=0.05,
                    key="interp_sigma_g"
                )
            with col6:
                s_E = st.slider(
                    "Energy Scale (s_E) [mJ]",
                    min_value=0.1,
                    max_value=50.0,
                    value=10.0,
                    step=0.5,
                    key="interp_s_E"
                )
            with col7:
                s_tau = st.slider(
                    "Duration Scale (s_τ) [ns]",
                    min_value=0.1,
                    max_value=20.0,
                    value=5.0,
                    step=0.5,
                    key="interp_s_tau"
                )
            
            st.markdown("### ⏱️ Temporal Gating Parameters")
            col8, col9, col10 = st.columns(3)
            with col8:
                sigma_t = st.slider(
                    "Temporal Gating Width (σ_t)",
                    min_value=0.05,
                    max_value=1.0,
                    value=0.15,
                    step=0.05,
                    key="interp_sigma_t"
                )
            with col9:
                s_t = st.slider(
                    "Time Scale (s_t) [ns]",
                    min_value=1.0,
                    max_value=100.0,
                    value=20.0,
                    step=1.0,
                    key="interp_s_t"
                )
            with col10:
                temporal_mode = st.selectbox(
                    "Temporal Mode",
                    ['fixed', 'adaptive', 'physics_informed', 'combined'],
                    index=2,
                    key="interp_temp_mode"
                )
            
            # Update extrapolator parameters
            st.session_state.extrapolator.sigma_param = sigma_param
            st.session_state.extrapolator.spatial_weight = spatial_weight
            st.session_state.extrapolator.n_heads = n_heads
            st.session_state.extrapolator.temperature = temperature
            st.session_state.extrapolator.sigma_g = sigma_g
            st.session_state.extrapolator.sigma_t = sigma_t
            st.session_state.extrapolator.s_E = s_E
            st.session_state.extrapolator.s_tau = s_tau
            st.session_state.extrapolator.s_t = s_t
            st.session_state.extrapolator.temporal_gating_mode = temporal_mode
    
    # Add temporal analysis tab to interpolation results
    def add_temporal_analysis_tab(results, time_points, energy_query, duration_query):
        """Add temporal analysis tab to interpolation results"""
        
        with st.expander("⏱️ Temporal Gating Analysis", expanded=True):
            # Create temporal analysis visualization
            temporal_fig = st.session_state.visualizer.create_temporal_gating_analysis(
                results, energy_query, duration_query, time_points
            )
            
            if temporal_fig:
                st.plotly_chart(temporal_fig, use_container_width=True)
            
            # Temporal statistics
            if results.get('temporal_analyses'):
                st.markdown("##### 📊 Temporal Statistics")
                
                # Select a timestep
                selected_time_idx = st.slider(
                    "Select timestep for temporal analysis",
                    0, len(time_points) - 1, len(time_points) // 2,
                    key="temp_analysis_timestep"
                )
                
                selected_analysis = results['temporal_analyses'][selected_time_idx]
                
                if selected_analysis:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(
                            "Phase Match",
                            f"{selected_analysis.get('phase_match', 0):.3f}",
                            help="Fraction of weights in same heating phase"
                        )
                    with col2:
                        st.metric(
                            "Temporal Spread",
                            f"{selected_analysis.get('temporal_spread', 0):.1f} ns",
                            help="Standard deviation of weighted source times"
                        )
                    with col3:
                        st.metric(
                            "Fourier Number",
                            f"{selected_analysis.get('fourier_mean', 0):.4f}",
                            help="Dimensionless time (αt/L²)"
                        )
                    with col4:
                        st.metric(
                            "Effective Sources",
                            f"{selected_analysis.get('effective_sources', 0)}",
                            help="Number of sources with significant gating"
                        )
                    
                    # Heating phase information
                    phase_names = {
                        0: "Very Early Heating",
                        1: "Active Heating",
                        2: "Early Cooling",
                        3: "Late Cooling/Diffusion"
                    }
                    
                    query_phase = selected_analysis.get('query_phase', 0)
                    st.info(f"**Heating Phase:** {phase_names.get(query_phase, 'Unknown')}")
            
            # 3D spatio-temporal visualization
            st.markdown("##### 🌐 3D Spatio-Temporal Visualization")
            
            spatiotemporal_fig = st.session_state.visualizer.create_3d_spatiotemporal_visualization(
                results, energy_query, duration_query, time_points
            )
            
            if spatiotemporal_fig.data:
                st.plotly_chart(spatiotemporal_fig, use_container_width=True)

# =============================================
# HEAT TRANSFER SPECIFIC UTILITIES
# =============================================
class HeatTransferAnalyzer:
    """Heat transfer specific analysis utilities for temporal gating"""
    
    @staticmethod
    def compute_thermal_metrics(time_points, duration_query, thermal_diffusivity=1e-5, spot_radius=50e-6):
        """Compute thermal metrics for heat transfer characterization"""
        
        metrics = {
            'time_points': time_points,
            'fourier_numbers': [],
            'thermal_penetration': [],
            'diffusion_lengths': [],
            'heating_phases': []
        }
        
        for t in time_points:
            # Fourier number: Fo = αt / L²
            t_s = t * 1e-9  # Convert ns to s
            fo = thermal_diffusivity * t_s / (spot_radius**2)
            metrics['fourier_numbers'].append(fo)
            
            # Thermal penetration depth: δ ~ √(αt)
            penetration = np.sqrt(thermal_diffusivity * t_s)
            metrics['thermal_penetration'].append(penetration)
            
            # Diffusion length scale
            diffusion_length = np.sqrt(4 * thermal_diffusivity * t_s)
            metrics['diffusion_lengths'].append(diffusion_length)
            
            # Heating phase classification
            if t < 0.1 * duration_query:
                phase = 0  # Very early heating
            elif t < duration_query:
                phase = 1  # Active heating
            elif t < 2 * duration_query:
                phase = 2  # Early cooling
            else:
                phase = 3  # Late cooling/diffusion
            metrics['heating_phases'].append(phase)
        
        return metrics
    
    @staticmethod
    def create_thermal_analysis_plot(thermal_metrics, time_points):
        """Create thermal analysis plot for heat transfer characterization"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Fourier Number Evolution",
                "Thermal Penetration Depth",
                "Diffusion Length Scale",
                "Heating Phase Timeline"
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Fourier number
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=thermal_metrics['fourier_numbers'],
                mode='lines+markers',
                name='Fo = αt/L²',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Thermal penetration depth
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=thermal_metrics['thermal_penetration'],
                mode='lines+markers',
                name='δ ~ √(αt)',
                line=dict(color='red', width=2)
            ),
            row=1, col=2
        )
        
        # Diffusion length scale
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=thermal_metrics['diffusion_lengths'],
                mode='lines+markers',
                name='L_diff ~ √(4αt)',
                line=dict(color='green', width=2)
            ),
            row=2, col=1
        )
        
        # Heating phase timeline
        phase_colors = {
            0: 'lightblue',
            1: 'orange',
            2: 'red',
            3: 'purple'
        }
        
        phase_names = {
            0: 'Early Heating',
            1: 'Active Heating',
            2: 'Early Cooling',
            3: 'Late Cooling'
        }
        
        for phase in range(4):
            phase_mask = np.array(thermal_metrics['heating_phases']) == phase
            if np.any(phase_mask):
                fig.add_trace(
                    go.Scatter(
                        x=time_points[phase_mask],
                        y=[phase] * np.sum(phase_mask),
                        mode='markers',
                        name=phase_names[phase],
                        marker=dict(
                            size=10,
                            color=phase_colors[phase],
                            opacity=0.7
                        )
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            height=600,
            title_text="Heat Transfer Thermal Metrics",
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time [ns]", row=1, col=1)
        fig.update_yaxes(title_text="Fourier Number", row=1, col=1)
        fig.update_xaxes(title_text="Time [ns]", row=1, col=2)
        fig.update_yaxes(title_text="Penetration [m]", row=1, col=2)
        fig.update_xaxes(title_text="Time [ns]", row=2, col=1)
        fig.update_yaxes(title_text="Diffusion Length [m]", row=2, col=1)
        fig.update_xaxes(title_text="Time [ns]", row=2, col=2)
        fig.update_yaxes(title_text="Heating Phase", row=2, col=2)
        
        # Set y-ticks for phase plot
        fig.update_yaxes(
            tickvals=[0, 1, 2, 3],
            ticktext=['Early Heat', 'Active Heat', 'Early Cool', 'Late Cool'],
            row=2, col=2
        )
        
        return fig

# =============================================
# INTEGRATE INTO EXISTING APPLICATION
# =============================================
def integrate_temporal_gating():
    """
    Integration guide for adding temporal gating to existing application:
    
    1. Replace DistanceGatedPhysicsAttentionExtrapolator with SpatioTemporalDGPAExtrapolator
    2. Update CacheManager to include temporal parameters in cache keys
    3. Add temporal gating controls to sidebar and interpolation interface
    4. Extend visualization tabs to include temporal analysis
    5. Add heat transfer analysis utilities
    """
    
    # Update CacheManager to include temporal parameters
    class EnhancedCacheManager(CacheManager):
        @staticmethod
        def generate_cache_key(field_name, timestep_idx, energy, duration, time,
                              sigma_param, spatial_weight, n_heads, temperature,
                              sigma_g, sigma_t, s_E, s_tau, s_t, temporal_mode,
                              top_k=None, subsample_factor=None):
            """Generate cache key with temporal gating parameters"""
            params_str = f"{field_name}_{timestep_idx}_{energy:.2f}_{duration:.2f}_{time:.2f}"
            params_str += f"_{sigma_param:.2f}_{spatial_weight:.2f}_{n_heads}_{temperature:.2f}"
            params_str += f"_{sigma_g:.2f}_{sigma_t:.2f}_{s_E:.2f}_{s_tau:.2f}_{s_t:.2f}"
            params_str += f"_{temporal_mode}"
            if top_k:
                params_str += f"_top{top_k}"
            if subsample_factor:
                params_str += f"_sub{subsample_factor}"
            
            return hashlib.md5(params_str.encode()).hexdigest()[:16]
    
    # Update main application flow
    st.markdown("""
    <style>
    .temporal-box {
        background: linear-gradient(135deg, #00b4db 0%, #0083b0 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Add temporal gating information box
    st.markdown("""
    <div class="temporal-box">
    <h3>⏱️ Spatio-Temporal DGPA with Gated Temporal Attention</h3>
    <p>This enhanced version includes <strong>temporal gating</strong> for improved heat transfer characterization:</p>
    <ul>
    <li><strong>Physics-aware temporal scaling</strong>: Incorporates Fourier number and thermal penetration depth</li>
    <li><strong>Heating phase matching</strong>: Ensures temporal similarity in heat transfer stages</li>
    <li><strong>Anisotropic gating</strong>: Separate widths for spatial (E, τ) and temporal (t) dimensions</li>
    <li><strong>Heat transfer metrics</strong>: Fourier number, thermal penetration, diffusion length analysis</li>
    </ul>
    <p><strong>Core Equation:</strong> φ_i = √[((E*-E_i)/s_E)² + ((τ*-τ_i)/s_τ)² + ((t*-t_i)/s_t)²]</p>
    </div>
    """, unsafe_allow_html=True)

# =============================================
# EXAMPLE USAGE AND DEMONSTRATION
# =============================================
def demonstrate_temporal_gating():
    """Demonstrate temporal gating functionality"""
    
    st.markdown("## 🎯 Temporal Gating Demonstration")
    
    # Create example scenario
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Heat Transfer Scenario")
        st.write("""
        **Laser Processing Parameters:**
        - Material: Steel (α = 1e-5 m²/s)
        - Spot Radius: 50 μm
        - Pulse Duration: 10 ns
        - Energy: 5 mJ
        
        **Temporal Characteristics:**
        - Thermal Time Constant: L²/α = 0.25 ms
        - Fourier Number Range: 0.001 to 0.1
        - Penetration Depth: ~0.1 to 1 μm
        """)
    
    with col2:
        st.markdown("### Temporal Gating Benefits")
        st.write("""
        **Improved Interpolation:**
        1. **Phase Matching**: Avoids blending heating and cooling phases
        2. **Diffusion Awareness**: Respects √(αt) scaling
        3. **Time Localization**: Prioritizes temporally similar sources
        4. **Uncertainty Control**: Wider gating for diffusion-dominated regimes
        """)
    
    # Create example plots
    if st.button("Show Temporal Analysis Example"):
        # Generate example thermal metrics
        time_points = np.linspace(1, 100, 50)
        thermal_metrics = HeatTransferAnalyzer.compute_thermal_metrics(
            time_points, duration_query=10, thermal_diffusivity=1e-5, spot_radius=50e-6
        )
        
        # Create thermal analysis plot
        thermal_fig = HeatTransferAnalyzer.create_thermal_analysis_plot(
            thermal_metrics, time_points
        )
        
        st.plotly_chart(thermal_fig, use_container_width=True)
        
        # Show gating kernel evolution
        st.markdown("### Temporal Gating Kernel Evolution")
        
        # Simulate gating kernel at different times
        times_example = [1, 10, 50, 100]
        fig_gating = go.Figure()
        
        for t in times_example:
            # Simulate gating weights
            source_times = np.linspace(1, 100, 20)
            sigma_t = 0.15
            s_t = 20.0
            
            gating_weights = np.exp(-((source_times - t) / s_t)**2 / (2 * sigma_t**2))
            gating_weights = gating_weights / np.sum(gating_weights)
            
            fig_gating.add_trace(go.Scatter(
                x=source_times,
                y=gating_weights,
                mode='lines+markers',
                name=f't={t} ns',
                line=dict(width=2)
            ))
        
        fig_gating.update_layout(
            title="Temporal Gating Kernel at Different Times",
            xaxis_title="Source Time [ns]",
            yaxis_title="Gating Weight",
            height=400
        )
        
        st.plotly_chart(fig_gating, use_container_width=True)

# Run the enhanced application
if __name__ == "__main__":
    # Add temporal gating controls to sidebar
    add_temporal_gating_controls()
    
    # Update main application with temporal features
    update_main_application()
    
    # Run demonstration if in DGPA Analysis mode
    if st.session_state.get('current_mode') == "DGPA Analysis":
        demonstrate_temporal_gating()
    
    # Continue with existing main() function
    main()
