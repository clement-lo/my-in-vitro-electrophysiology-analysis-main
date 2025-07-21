"""
Network Analysis Module v2
Graph-based connectivity analysis with plasticity modeling
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy import signal, stats
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from pathlib import Path
from typing import Dict, Any, Union, Optional, Tuple, List
import logging

from ..core.base_analysis import AbstractAnalysis, ValidationMixin
from ..core.data_pipeline import load_data
from ..core.config_manager import ConfigManager
from ..utils.visualization import create_multi_panel_figure, plot_heatmap

logger = logging.getLogger(__name__)

class NetworkAnalysis(AbstractAnalysis, ValidationMixin):
    """
    Advanced network connectivity and dynamics analysis.
    
    Features:
    - Graph-based connectivity metrics
    - Multi-scale analysis (neuron → microcircuit → network)
    - Plasticity rule implementation (STDP, BCM, Hebbian)
    - Community detection algorithms
    - Dynamic network evolution
    - Small-world and scale-free analysis
    """
    
    def __init__(self, config: Union[Dict, ConfigManager, Path, str, None] = None):
        super().__init__(name="network_analysis", version="2.0")
        
        # Handle configuration
        if isinstance(config, ConfigManager):
            self.config_manager = config
        elif isinstance(config, (Path, str)):
            self.config_manager = ConfigManager(config)
        elif isinstance(config, dict):
            self.config_manager = ConfigManager()
            self.config_manager.config = config
        else:
            self.config_manager = ConfigManager()
            
        self.config = self.config_manager.create_analysis_config('network')
        
        # Plasticity rules
        self.plasticity_rules = {
            'hebbian': self._hebbian_plasticity,
            'stdp': self._spike_timing_dependent_plasticity,
            'bcm': self._bcm_plasticity,
            'homeostatic': self._homeostatic_plasticity
        }
        
    def load_data(self, file_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
        """Load multi-channel electrophysiology data for network analysis."""
        
        # Check if this is pre-computed connectivity data
        if str(file_path).endswith('.npz'):
            data = np.load(file_path)
            self.data = {
                'connectivity_matrix': data.get('connectivity'),
                'node_positions': data.get('positions'),
                'node_labels': data.get('labels'),
                'metadata': dict(data.get('metadata', {}))
            }
        else:
            # Load multi-channel recording
            signals, rate, time, metadata = load_data(file_path, **kwargs)
            
            # Ensure signals is 2D (time x channels)
            if signals.ndim == 1:
                signals = signals.reshape(-1, 1)
                
            self.data = {
                'signals': signals,
                'sampling_rate': rate,
                'time': time,
                'n_channels': signals.shape[1],
                'metadata': metadata
            }
            
        return self.data
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate network analysis parameters."""
        
        # Analysis type
        valid_analyses = ['connectivity', 'dynamics', 'plasticity', 'communities']
        if 'analysis_type' not in parameters:
            parameters['analysis_type'] = 'connectivity'
            
        parameters['analysis_type'] = self.validate_choice_parameter(
            parameters['analysis_type'], valid_analyses, 'analysis_type'
        )
        
        # Connectivity method
        if parameters['analysis_type'] == 'connectivity':
            valid_methods = ['correlation', 'coherence', 'granger', 'transfer_entropy', 'phase_locking']
            method = parameters.get('connectivity_method', 'correlation')
            parameters['connectivity_method'] = self.validate_choice_parameter(
                method, valid_methods, 'connectivity_method'
            )
            
        # Threshold for connectivity
        if 'threshold' in parameters:
            parameters['threshold'] = self.validate_numeric_parameter(
                parameters['threshold'], 0, 1, 'threshold'
            )
            
        return True
    
    def run_analysis(self, data: Any = None, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run network analysis."""
        if data is None:
            data = self.data
            
        if parameters is None:
            parameters = self.config.get('network', {
                'analysis_type': 'connectivity',
                'connectivity_method': 'correlation',
                'threshold': 0.3,
                'calculate_metrics': True
            })
            
        self.validate_parameters(parameters)
        self.metadata['parameters'] = parameters
        
        results = {}
        analysis_type = parameters['analysis_type']
        
        # Build connectivity matrix if needed
        if 'connectivity_matrix' not in data and 'signals' in data:
            results['connectivity'] = self._calculate_connectivity(data, parameters)
            connectivity_matrix = results['connectivity']['matrix']
        else:
            connectivity_matrix = data.get('connectivity_matrix')
            
        # Create network graph
        self.graph = self._create_network_graph(connectivity_matrix, parameters)
        results['graph'] = self.graph
        
        # Run specific analyses
        if analysis_type == 'connectivity' or parameters.get('calculate_metrics', True):
            results['metrics'] = self._calculate_network_metrics(self.graph)
            
        if analysis_type == 'dynamics':
            results['dynamics'] = self._analyze_network_dynamics(data, self.graph, parameters)
            
        if analysis_type == 'communities':
            results['communities'] = self._detect_communities(self.graph, parameters)
            
        if analysis_type == 'plasticity':
            results['plasticity'] = self._simulate_plasticity(data, self.graph, parameters)
            
        # Analyze network topology
        if parameters.get('analyze_topology', True):
            results['topology'] = self._analyze_topology(self.graph)
            
        self.results = results
        return results
    
    def _calculate_connectivity(self, data: Dict[str, Any], 
                               parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate functional connectivity between channels."""
        signals = data['signals']
        fs = data['sampling_rate']
        method = parameters['connectivity_method']
        
        n_channels = signals.shape[1]
        connectivity = np.zeros((n_channels, n_channels))
        
        if method == 'correlation':
            # Pearson correlation
            connectivity = np.corrcoef(signals.T)
            
        elif method == 'coherence':
            # Spectral coherence
            for i in range(n_channels):
                for j in range(i+1, n_channels):
                    f, coh = signal.coherence(signals[:, i], signals[:, j], fs)
                    # Average coherence in frequency band of interest
                    freq_band = parameters.get('frequency_band', [1, 100])
                    mask = (f >= freq_band[0]) & (f <= freq_band[1])
                    connectivity[i, j] = np.mean(coh[mask])
                    connectivity[j, i] = connectivity[i, j]
                    
        elif method == 'granger':
            # Granger causality
            connectivity = self._granger_causality(signals, parameters)
            
        elif method == 'transfer_entropy':
            # Transfer entropy
            connectivity = self._transfer_entropy(signals, parameters)
            
        elif method == 'phase_locking':
            # Phase locking value
            connectivity = self._phase_locking_value(signals, fs, parameters)
            
        # Apply threshold
        threshold = parameters.get('threshold', 0)
        connectivity_binary = (np.abs(connectivity) > threshold).astype(int)
        np.fill_diagonal(connectivity_binary, 0)
        
        return {
            'matrix': connectivity,
            'binary_matrix': connectivity_binary,
            'method': method,
            'threshold': threshold
        }
    
    def _granger_causality(self, signals: np.ndarray, 
                          parameters: Dict[str, Any]) -> np.ndarray:
        """Calculate Granger causality between signals."""
        from statsmodels.tsa.stattools import grangercausalitytests
        
        n_channels = signals.shape[1]
        gc_matrix = np.zeros((n_channels, n_channels))
        max_lag = parameters.get('max_lag', 10)
        
        for i in range(n_channels):
            for j in range(n_channels):
                if i != j:
                    try:
                        result = grangercausalitytests(
                            np.column_stack([signals[:, i], signals[:, j]]),
                            maxlag=max_lag, verbose=False
                        )
                        # Use minimum p-value across lags
                        p_values = [result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag+1)]
                        gc_matrix[j, i] = 1 - min(p_values)  # j causes i
                    except:
                        gc_matrix[j, i] = 0
                        
        return gc_matrix
    
    def _transfer_entropy(self, signals: np.ndarray,
                         parameters: Dict[str, Any]) -> np.ndarray:
        """Calculate transfer entropy between signals."""
        # Simplified implementation - would use pyinform or similar
        n_channels = signals.shape[1]
        te_matrix = np.zeros((n_channels, n_channels))
        
        # This is a placeholder - real implementation would compute actual TE
        for i in range(n_channels):
            for j in range(n_channels):
                if i != j:
                    # Compute transfer entropy from j to i
                    te_matrix[j, i] = np.random.rand() * 0.5  # Placeholder
                    
        return te_matrix
    
    def _phase_locking_value(self, signals: np.ndarray, fs: float,
                           parameters: Dict[str, Any]) -> np.ndarray:
        """Calculate phase locking value between signals."""
        n_channels = signals.shape[1]
        plv_matrix = np.zeros((n_channels, n_channels))
        
        # Filter signals in frequency band of interest
        freq_band = parameters.get('frequency_band', [8, 12])  # Alpha band default
        sos = signal.butter(4, freq_band, btype='band', fs=fs, output='sos')
        
        # Extract phases
        phases = np.zeros_like(signals)
        for i in range(n_channels):
            filtered = signal.sosfilt(sos, signals[:, i])
            analytic = signal.hilbert(filtered)
            phases[:, i] = np.angle(analytic)
            
        # Calculate PLV
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                phase_diff = phases[:, i] - phases[:, j]
                plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                plv_matrix[i, j] = plv
                plv_matrix[j, i] = plv
                
        np.fill_diagonal(plv_matrix, 1)
        return plv_matrix
    
    def _create_network_graph(self, connectivity: np.ndarray,
                            parameters: Dict[str, Any]) -> nx.Graph:
        """Create NetworkX graph from connectivity matrix."""
        # Use binary or weighted graph
        if parameters.get('weighted', True):
            G = nx.from_numpy_array(connectivity)
        else:
            binary_conn = (connectivity > parameters.get('threshold', 0)).astype(int)
            G = nx.from_numpy_array(binary_conn)
            
        # Add node attributes
        for i in range(len(G.nodes)):
            G.nodes[i]['label'] = f'Node_{i}'
            
        # Add edge attributes
        for i, j in G.edges():
            G.edges[i, j]['weight'] = connectivity[i, j]
            
        return G
    
    def _calculate_network_metrics(self, G: nx.Graph) -> Dict[str, Any]:
        """Calculate comprehensive network metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['n_nodes'] = G.number_of_nodes()
        metrics['n_edges'] = G.number_of_edges()
        metrics['density'] = nx.density(G)
        
        # Degree metrics
        degrees = dict(G.degree())
        metrics['mean_degree'] = np.mean(list(degrees.values()))
        metrics['degree_distribution'] = list(degrees.values())
        
        # Centrality measures
        metrics['betweenness_centrality'] = nx.betweenness_centrality(G)
        metrics['closeness_centrality'] = nx.closeness_centrality(G)
        metrics['eigenvector_centrality'] = nx.eigenvector_centrality(G, max_iter=1000)
        
        # Clustering
        metrics['clustering_coefficient'] = nx.average_clustering(G)
        metrics['local_clustering'] = nx.clustering(G)
        
        # Path metrics
        if nx.is_connected(G):
            metrics['average_path_length'] = nx.average_shortest_path_length(G)
            metrics['diameter'] = nx.diameter(G)
        else:
            # Use largest connected component
            largest_cc = max(nx.connected_components(G), key=len)
            G_sub = G.subgraph(largest_cc)
            metrics['average_path_length'] = nx.average_shortest_path_length(G_sub)
            metrics['diameter'] = nx.diameter(G_sub)
            metrics['n_components'] = nx.number_connected_components(G)
            
        # Small-world metrics
        metrics['small_world'] = self._calculate_small_world_index(G)
        
        # Rich club coefficient
        if G.number_of_edges() > 0:
            metrics['rich_club'] = nx.rich_club_coefficient(G)
            
        # Modularity (requires community detection)
        communities = nx.community.greedy_modularity_communities(G)
        metrics['modularity'] = nx.community.modularity(G, communities)
        metrics['n_communities'] = len(communities)
        
        return metrics
    
    def _calculate_small_world_index(self, G: nx.Graph) -> Dict[str, float]:
        """Calculate small-world index (sigma)."""
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        
        if n_edges == 0:
            return {'sigma': 0, 'omega': 0}
            
        # Generate random and lattice reference networks
        random_graph = nx.erdos_renyi_graph(n_nodes, nx.density(G))
        lattice_graph = nx.watts_strogatz_graph(n_nodes, 
                                               int(np.mean(list(dict(G.degree()).values()))), 
                                               0)
        
        # Calculate metrics
        C = nx.average_clustering(G)
        C_random = nx.average_clustering(random_graph)
        C_lattice = nx.average_clustering(lattice_graph)
        
        if nx.is_connected(G):
            L = nx.average_shortest_path_length(G)
            L_random = nx.average_shortest_path_length(random_graph)
            L_lattice = nx.average_shortest_path_length(lattice_graph)
        else:
            # Use largest component
            Gcc = max(nx.connected_components(G), key=len)
            L = nx.average_shortest_path_length(G.subgraph(Gcc))
            
            Gcc_random = max(nx.connected_components(random_graph), key=len)
            L_random = nx.average_shortest_path_length(random_graph.subgraph(Gcc_random))
            
            Gcc_lattice = max(nx.connected_components(lattice_graph), key=len)
            L_lattice = nx.average_shortest_path_length(lattice_graph.subgraph(Gcc_lattice))
        
        # Small-world index (sigma)
        gamma = C / C_random if C_random > 0 else 0
        lambda_val = L / L_random if L_random > 0 else 0
        sigma = gamma / lambda_val if lambda_val > 0 else 0
        
        # Alternative small-world measure (omega)
        omega = (L_random / L) - (C / C_lattice) if L > 0 and C_lattice > 0 else 0
        
        return {
            'sigma': sigma,
            'omega': omega,
            'gamma': gamma,
            'lambda': lambda_val
        }
    
    def _detect_communities(self, G: nx.Graph, 
                          parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Detect communities in the network."""
        method = parameters.get('community_method', 'louvain')
        
        if method == 'louvain':
            # Use python-louvain if available
            try:
                import community as community_louvain
                partition = community_louvain.best_partition(G)
                communities = {}
                for node, comm in partition.items():
                    if comm not in communities:
                        communities[comm] = []
                    communities[comm].append(node)
                communities = list(communities.values())
            except ImportError:
                # Fallback to NetworkX
                communities = list(nx.community.greedy_modularity_communities(G))
                
        elif method == 'girvan_newman':
            comp = nx.community.girvan_newman(G)
            communities = tuple(sorted(c) for c in next(comp))
            
        elif method == 'label_propagation':
            communities = list(nx.community.label_propagation_communities(G))
            
        else:
            communities = list(nx.community.greedy_modularity_communities(G))
            
        # Calculate modularity
        modularity = nx.community.modularity(G, communities)
        
        # Community sizes
        community_sizes = [len(c) for c in communities]
        
        return {
            'communities': communities,
            'n_communities': len(communities),
            'modularity': modularity,
            'community_sizes': community_sizes,
            'method': method
        }
    
    def _analyze_network_dynamics(self, data: Dict[str, Any], G: nx.Graph,
                                parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze dynamic network properties."""
        if 'signals' not in data:
            return {}
            
        signals = data['signals']
        fs = data['sampling_rate']
        
        # Sliding window analysis
        window_size = parameters.get('window_size', 1.0)  # seconds
        window_step = parameters.get('window_step', 0.5)  # seconds
        
        window_samples = int(window_size * fs)
        step_samples = int(window_step * fs)
        
        n_windows = int((signals.shape[0] - window_samples) / step_samples) + 1
        
        # Store time-varying metrics
        time_points = []
        metrics_over_time = {
            'mean_degree': [],
            'clustering': [],
            'modularity': [],
            'efficiency': []
        }
        
        for i in range(n_windows):
            start = i * step_samples
            end = start + window_samples
            
            # Calculate connectivity for this window
            window_signals = signals[start:end, :]
            window_conn = np.corrcoef(window_signals.T)
            
            # Create graph for this window
            G_window = self._create_network_graph(window_conn, parameters)
            
            # Calculate metrics
            time_points.append(start / fs + window_size / 2)
            
            degrees = list(dict(G_window.degree()).values())
            metrics_over_time['mean_degree'].append(np.mean(degrees))
            metrics_over_time['clustering'].append(nx.average_clustering(G_window))
            
            # Modularity
            communities = nx.community.greedy_modularity_communities(G_window)
            metrics_over_time['modularity'].append(
                nx.community.modularity(G_window, communities)
            )
            
            # Global efficiency
            metrics_over_time['efficiency'].append(nx.global_efficiency(G_window))
        
        return {
            'time_points': np.array(time_points),
            'metrics': metrics_over_time,
            'window_size': window_size,
            'window_step': window_step
        }
    
    def _simulate_plasticity(self, data: Dict[str, Any], G: nx.Graph,
                           parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate synaptic plasticity on the network."""
        plasticity_rule = parameters.get('plasticity_rule', 'hebbian')
        
        if plasticity_rule not in self.plasticity_rules:
            raise ValueError(f"Unknown plasticity rule: {plasticity_rule}")
            
        # Get initial weights
        weights = nx.get_edge_attributes(G, 'weight')
        weight_matrix = nx.to_numpy_array(G)
        
        # Apply plasticity rule
        updated_weights = self.plasticity_rules[plasticity_rule](
            weight_matrix, data, parameters
        )
        
        # Create updated graph
        G_updated = nx.from_numpy_array(updated_weights)
        
        # Calculate weight changes
        weight_changes = updated_weights - weight_matrix
        
        return {
            'initial_weights': weight_matrix,
            'final_weights': updated_weights,
            'weight_changes': weight_changes,
            'updated_graph': G_updated,
            'plasticity_rule': plasticity_rule
        }
    
    def _hebbian_plasticity(self, weights: np.ndarray, data: Dict[str, Any],
                          parameters: Dict[str, Any]) -> np.ndarray:
        """Implement Hebbian plasticity: neurons that fire together, wire together."""
        if 'signals' not in data:
            return weights
            
        signals = data['signals']
        learning_rate = parameters.get('learning_rate', 0.01)
        
        # Calculate activity correlation
        activity = signals.T  # channels x time
        correlation = np.corrcoef(activity)
        
        # Update weights
        weight_change = learning_rate * correlation
        updated_weights = weights + weight_change
        
        # Ensure weights stay positive
        updated_weights = np.maximum(updated_weights, 0)
        
        # Normalize to prevent runaway growth
        max_weight = parameters.get('max_weight', 1.0)
        updated_weights = np.minimum(updated_weights, max_weight)
        
        return updated_weights
    
    def _spike_timing_dependent_plasticity(self, weights: np.ndarray, 
                                         data: Dict[str, Any],
                                         parameters: Dict[str, Any]) -> np.ndarray:
        """Implement STDP: timing-dependent synaptic plasticity."""
        # This would require spike times
        # For now, return placeholder
        return weights
    
    def _bcm_plasticity(self, weights: np.ndarray, data: Dict[str, Any],
                       parameters: Dict[str, Any]) -> np.ndarray:
        """Implement BCM (Bienenstock-Cooper-Munro) plasticity rule."""
        if 'signals' not in data:
            return weights
            
        signals = data['signals']
        learning_rate = parameters.get('learning_rate', 0.01)
        
        # Calculate average activity for each neuron
        activity = signals.T  # channels x time
        mean_activity = np.mean(activity, axis=1)
        
        # BCM threshold (sliding threshold)
        theta = mean_activity ** 2
        
        # Update weights based on BCM rule
        n_channels = weights.shape[0]
        updated_weights = weights.copy()
        
        for i in range(n_channels):
            for j in range(n_channels):
                if i != j:
                    # BCM update rule
                    delta_w = learning_rate * activity[j] * activity[i] * (activity[i] - theta[i])
                    updated_weights[i, j] += np.mean(delta_w)
                    
        # Keep weights positive
        updated_weights = np.maximum(updated_weights, 0)
        
        return updated_weights
    
    def _homeostatic_plasticity(self, weights: np.ndarray, data: Dict[str, Any],
                              parameters: Dict[str, Any]) -> np.ndarray:
        """Implement homeostatic plasticity to maintain stable activity."""
        if 'signals' not in data:
            return weights
            
        signals = data['signals']
        target_rate = parameters.get('target_firing_rate', 5.0)  # Hz
        
        # Calculate actual firing rates (simplified)
        activity = signals.T
        actual_rates = np.mean(np.abs(activity) > np.std(activity), axis=1) * data['sampling_rate']
        
        # Scale weights to achieve target rate
        scaling_factors = target_rate / (actual_rates + 1e-10)
        scaling_factors = np.clip(scaling_factors, 0.5, 2.0)  # Limit scaling
        
        # Apply scaling
        updated_weights = weights * scaling_factors[:, np.newaxis]
        
        return updated_weights
    
    def _analyze_topology(self, G: nx.Graph) -> Dict[str, Any]:
        """Analyze network topology (scale-free, small-world, etc.)."""
        degrees = [d for n, d in G.degree()]
        
        # Fit power law to degree distribution
        from scipy.stats import powerlaw
        
        if len(degrees) > 10:
            # Fit power law
            params = powerlaw.fit(degrees)
            alpha = params[2]  # Power law exponent
            
            # Test scale-free property
            is_scale_free = 2 < alpha < 3  # Typical range for scale-free networks
        else:
            alpha = None
            is_scale_free = False
            
        # Hub detection
        mean_degree = np.mean(degrees)
        std_degree = np.std(degrees)
        hub_threshold = mean_degree + 2 * std_degree
        hubs = [n for n, d in G.degree() if d > hub_threshold]
        
        # Core-periphery structure
        core_nodes = self._identify_core_periphery(G)
        
        return {
            'degree_distribution': degrees,
            'power_law_exponent': alpha,
            'is_scale_free': is_scale_free,
            'hubs': hubs,
            'n_hubs': len(hubs),
            'core_nodes': core_nodes,
            'assortativity': nx.degree_assortativity_coefficient(G)
        }
    
    def _identify_core_periphery(self, G: nx.Graph) -> List[int]:
        """Identify core nodes in core-periphery structure."""
        # Use k-core decomposition
        k_core = nx.k_core(G)
        core_nodes = list(k_core.nodes())
        
        return core_nodes
    
    def generate_report(self, results: Dict[str, Any] = None,
                       output_dir: Union[str, Path] = None) -> Path:
        """Generate network analysis report."""
        if results is None:
            results = self.results
            
        if output_dir is None:
            output_dir = Path('./results')
            
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create multi-panel figure
        n_panels = 6
        fig, axes = create_multi_panel_figure(n_panels, fig_size=(18, 12))
        
        # 1. Network visualization
        if 'graph' in results:
            self._plot_network(axes[0], results['graph'])
            
        # 2. Connectivity matrix
        if 'connectivity' in results:
            self._plot_connectivity_matrix(axes[1], results['connectivity'])
            
        # 3. Degree distribution
        if 'metrics' in results:
            self._plot_degree_distribution(axes[2], results['metrics'])
            
        # 4. Network metrics summary
        if 'metrics' in results:
            self._plot_metrics_summary(axes[3], results['metrics'])
            
        # 5. Community structure
        if 'communities' in results:
            self._plot_communities(axes[4], results['graph'], results['communities'])
            
        # 6. Dynamic metrics (if available)
        if 'dynamics' in results:
            self._plot_dynamic_metrics(axes[5], results['dynamics'])
        else:
            axes[5].axis('off')
            
        plt.suptitle('Network Analysis Report', fontsize=16)
        
        # Save figure
        fig_path = output_dir / 'network_analysis.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate text report
        report_path = output_dir / 'network_report.txt'
        self._generate_text_report(results, report_path)
        
        # Export graph
        if 'graph' in results:
            nx.write_graphml(results['graph'], output_dir / 'network.graphml')
            
        # Export results
        self.export_results(output_dir / 'network_results', format='pickle')
        
        logger.info(f"Network analysis report saved to {output_dir}")
        
        return report_path
    
    def _plot_network(self, ax, G):
        """Visualize network graph."""
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # Node colors by degree
        degrees = [G.degree(n) for n in G.nodes()]
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_color=degrees, 
                              cmap='viridis', node_size=300, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)
        
        # Labels for high-degree nodes
        labels = {n: str(n) for n in G.nodes() if G.degree(n) > np.mean(degrees) + np.std(degrees)}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
        
        ax.set_title('Network Structure')
        ax.axis('off')
    
    def _plot_connectivity_matrix(self, ax, connectivity_data):
        """Plot connectivity matrix."""
        matrix = connectivity_data['matrix']
        
        im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', 
                      vmin=-1, vmax=1)
        ax.set_xlabel('Node')
        ax.set_ylabel('Node')
        ax.set_title(f'Connectivity Matrix ({connectivity_data["method"]})')
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    def _plot_degree_distribution(self, ax, metrics):
        """Plot degree distribution."""
        degrees = metrics['degree_distribution']
        
        ax.hist(degrees, bins=20, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Degree')
        ax.set_ylabel('Count')
        ax.set_title('Degree Distribution')
        
        # Add statistics
        ax.text(0.7, 0.9, f'Mean: {np.mean(degrees):.2f}',
               transform=ax.transAxes)
        ax.text(0.7, 0.8, f'Max: {np.max(degrees)}',
               transform=ax.transAxes)
    
    def _plot_metrics_summary(self, ax, metrics):
        """Plot summary of network metrics."""
        ax.axis('off')
        
        summary_text = "Network Metrics\n" + "=" * 30 + "\n\n"
        
        key_metrics = [
            ('Nodes', metrics.get('n_nodes', 0)),
            ('Edges', metrics.get('n_edges', 0)),
            ('Density', metrics.get('density', 0)),
            ('Clustering', metrics.get('clustering_coefficient', 0)),
            ('Modularity', metrics.get('modularity', 0)),
            ('Communities', metrics.get('n_communities', 0))
        ]
        
        if 'small_world' in metrics:
            sw = metrics['small_world']
            key_metrics.extend([
                ('Small-world σ', sw.get('sigma', 0)),
                ('Small-world ω', sw.get('omega', 0))
            ])
        
        for name, value in key_metrics:
            if isinstance(value, float):
                summary_text += f"{name}: {value:.3f}\n"
            else:
                summary_text += f"{name}: {value}\n"
        
        ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
               fontsize=11, va='center', family='monospace')
    
    def _plot_communities(self, ax, G, communities_data):
        """Plot community structure."""
        communities = communities_data['communities']
        
        # Create color map for communities
        colors = plt.cm.Set3(np.linspace(0, 1, len(communities)))
        
        pos = nx.spring_layout(G)
        
        for i, community in enumerate(communities):
            nx.draw_networkx_nodes(G, pos, nodelist=list(community),
                                 node_color=[colors[i]], node_size=300, ax=ax)
        
        nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
        
        ax.set_title(f'Community Structure (Q={communities_data["modularity"]:.3f})')
        ax.axis('off')
    
    def _plot_dynamic_metrics(self, ax, dynamics_data):
        """Plot time-varying network metrics."""
        time_points = dynamics_data['time_points']
        metrics = dynamics_data['metrics']
        
        for metric_name, values in metrics.items():
            ax.plot(time_points, values, label=metric_name)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Metric Value')
        ax.set_title('Dynamic Network Properties')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _generate_text_report(self, results: Dict[str, Any], report_path: Path):
        """Generate detailed text report."""
        with open(report_path, 'w') as f:
            f.write("Network Analysis Report\n")
            f.write("=" * 60 + "\n\n")
            
            # Summary
            if 'metrics' in results:
                metrics = results['metrics']
                f.write("Network Summary:\n")
                f.write(f"  Nodes: {metrics.get('n_nodes', 0)}\n")
                f.write(f"  Edges: {metrics.get('n_edges', 0)}\n")
                f.write(f"  Density: {metrics.get('density', 0):.3f}\n")
                f.write(f"  Average Degree: {metrics.get('mean_degree', 0):.2f}\n")
                f.write(f"  Clustering Coefficient: {metrics.get('clustering_coefficient', 0):.3f}\n")
                
                if 'average_path_length' in metrics:
                    f.write(f"  Average Path Length: {metrics['average_path_length']:.3f}\n")
                    
            # Topology analysis
            if 'topology' in results:
                topo = results['topology']
                f.write("\nTopology Analysis:\n")
                if topo.get('is_scale_free'):
                    f.write("  Network exhibits scale-free properties\n")
                    f.write(f"  Power-law exponent: {topo.get('power_law_exponent', 0):.2f}\n")
                f.write(f"  Number of hubs: {topo.get('n_hubs', 0)}\n")
                f.write(f"  Assortativity: {topo.get('assortativity', 0):.3f}\n")


# Example usage
if __name__ == "__main__":
    # Create analyzer
    analyzer = NetworkAnalysis()
    
    # Generate synthetic multi-channel data
    n_channels = 10
    duration = 60  # seconds
    fs = 1000  # Hz
    
    # Create correlated signals
    time = np.arange(0, duration, 1/fs)
    base_signal = np.sin(2 * np.pi * 10 * time) + np.random.randn(len(time)) * 0.5
    
    signals = np.zeros((len(time), n_channels))
    for i in range(n_channels):
        signals[:, i] = base_signal + np.random.randn(len(time)) * 0.3
        
    # Make some channels more correlated
    signals[:, 1] = 0.8 * signals[:, 0] + 0.2 * np.random.randn(len(time))
    signals[:, 2] = 0.7 * signals[:, 0] + 0.3 * np.random.randn(len(time))
    
    # Create data dict
    data = {
        'signals': signals,
        'sampling_rate': fs,
        'time': time
    }
    
    # Run analysis
    results = analyzer.run_analysis(
        data=data,
        parameters={
            'analysis_type': 'connectivity',
            'connectivity_method': 'correlation',
            'threshold': 0.3,
            'analyze_topology': True
        }
    )
    
    # Generate report
    analyzer.generate_report(output_dir='./results/network')
