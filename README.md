# In Vitro Electrophysiology Analysis Framework

## A Comprehensive Toolkit for Patch-Clamp and Multi-Electrode Array Data Analysis

### Overview

This repository presents a dual-implementation framework for in vitro electrophysiology data analysis, developed to demonstrate research capabilities in computational neuroscience. It features both educational (v1) and research-grade (v2) implementations, showcasing progression from fundamental concepts to production-ready analysis pipelines.

**Key Highlights:**
- 🔬 **Dual Architecture**: Educational v1 for learning/prototyping, Research-grade v2 for production
- 📊 **Multi-Format Support**: ABF, NWB, CSV, HDF5, Neo-compatible formats
- 🧮 **Comprehensive Analyses**: Action potentials, synaptic events, ion channels, network dynamics
- 🔧 **Modern Standards**: NWB compatibility, object-oriented design, extensive validation
- 📈 **Publication Ready**: Statistical validation, quality control, reproducible workflows

### Background & Motivation

This framework bridges computational approaches from diagnostic development (automated IHC/ISH assays) to electrophysiology analysis, demonstrating transferable skills in:
- Signal processing and time-series analysis
- Statistical validation and quality control
- Systematic method development and validation
- Research-grade software engineering

## Repository Structure

```plaintext
my-in-vitro-electrophysiology-analysis/
├── README.md                          # This file
├── QUICKSTART.md                      # Quick start guide
├── CONTRIBUTING.md                    # Contribution guidelines
├── CHANGELOG.md                       # Version history
├── MIGRATION_GUIDE.md                 # Migration from v1 to v2
├── requirements.txt                   # Python dependencies
├── LICENSE                            # License information
├── data/                              # Sample data directory
├── test_data/                         # Test datasets (CSV, NWB, DAT formats)
│   ├── README.md
│   ├── action_potential_test.csv
│   ├── synaptic_current_test.nwb
│   └── ...
├── scripts/                           # Analysis implementations
│   ├── patch_clamp_whole_cell_in_vitro/  # Patch-clamp analyses
│   │   ├── README.md                  # Detailed module documentation
│   │   ├── DEPENDENCIES.md            # Module-specific dependencies
│   │   ├── python/
│   │   │   ├── v1/                    # Educational implementations
│   │   │   │   ├── action_potential_analysis.py
│   │   │   │   ├── synaptic_current_analysis.py
│   │   │   │   ├── ion_channels_kinetics.py
│   │   │   │   └── ... (with config files)
│   │   │   └── v2/                    # Research-grade framework
│   │   │       ├── README.md
│   │   │       ├── core/              # Core infrastructure
│   │   │       ├── common/            # Shared utilities
│   │   │       ├── modules/           # Analysis modules
│   │   │       ├── utils/             # Helper functions
│   │   │       └── workflows/         # Complete pipelines
│   │   ├── matlab/
│   │   │   ├── v1/                    # MATLAB implementations
│   │   │   └── v2/                    # Enhanced MATLAB framework
│   │   └── notebooks/
│   │       └── v1/                    # Interactive Jupyter notebooks
│   ├── mea_in_vitro/                  # Multi-electrode array analyses
│   │   ├── python/
│   │   │   ├── advanced_spike_sorting_clustering.py
│   │   │   ├── connectivity_analysis.py
│   │   │   ├── network_dynamics_analysis.py
│   │   │   └── pharmacological_modulation_analysis.py
│   │   └── matlab/
│   ├── utils/                         # Validation and utility scripts
│   │   ├── validation.py
│   │   ├── validate_modules.py
│   │   └── validate_modules_matlab.m
│   └── test_data_generator.py         # Generate test datasets
```
## Technical Architecture

### v1 - Educational Implementation
- **Purpose**: Learning, prototyping, teaching
- **Design**: Simple, function-based, minimal dependencies
- **Best For**: Quick analyses, educational demonstrations, concept validation

### v2 - Research-Grade Framework
- **Purpose**: Production research, complex pipelines, extensible architecture
- **Design**: Object-oriented, multi-format support, comprehensive validation
- **Best For**: Publication-quality analysis, batch processing, custom workflows

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/clement-lo/my-in-vitro-electrophysiology-analysis-main.git
cd my-in-vitro-electrophysiology-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Example Usage

#### Simple Analysis (v1)
```python
# Quick action potential analysis
from scripts.patch_clamp_whole_cell_in_vitro.python.v1 import action_potential_analysis

# Load and analyze
data = action_potential_analysis.load_data('recording.csv')
spikes = action_potential_analysis.detect_action_potentials(data, threshold=-20)
action_potential_analysis.plot_results(data, spikes)
```

#### Advanced Pipeline (v2)
```python
# Research-grade analysis pipeline
from scripts.patch_clamp_whole_cell_in_vitro.python.v2.core.data_loader import DataLoader
from scripts.patch_clamp_whole_cell_in_vitro.python.v2.core.data_pipeline import DataPipeline
from scripts.patch_clamp_whole_cell_in_vitro.python.v2.action_potential_analysis_merged import run_complete_analysis

# Multi-format data loading
loader = DataLoader()
data = loader.load_file("recording.abf")  # Supports ABF, NWB, CSV, HDF5

# Complete analysis with validation
results = run_complete_analysis(
    data_file="recording.abf",
    output_dir="./results",
    config_file="config.yaml"
)
```

## Analysis Capabilities

### Patch-Clamp & Whole-Cell Analyses

#### 1. Action Potential Analysis
- **Detection**: Multiple algorithms (threshold, derivative, template matching)
- **Characterization**: Amplitude, width, AHP, threshold dynamics
- **Advanced Metrics**: Adaptation, accommodation, burst analysis
- **Formats**: CSV (v1), ABF/NWB/HDF5 (v2)

#### 2. Synaptic Current Analysis
- **Event Detection**: mEPSCs, mIPSCs, evoked responses
- **Kinetics**: Rise time, decay constants, charge transfer
- **Plasticity**: Paired-pulse, frequency-dependent changes
- **Input-Output**: Stimulus-response relationships

#### 3. Ion Channel Analysis
- **I-V Relationships**: Automated curve fitting
- **Kinetics**: Activation, inactivation, recovery
- **Conductance**: G-V curves, reversal potentials
- **Pharmacology**: Dose-response, Hill equation fitting

#### 4. Network Connectivity & Dynamics
- **Connectivity Mapping**: Cross-correlation, coherence analysis
- **Synchronization**: Phase locking, synchrony indices
- **Graph Theory**: Centrality, clustering, path analysis
- **State Transitions**: Network state identification

#### 5. Pharmacological Modulation
- **Dose-Response**: IC50/EC50 calculation
- **Time Course**: Drug wash-in/wash-out kinetics
- **Multi-Drug**: Interaction analysis, isobolograms
- **Statistical Validation**: Paired comparisons, effect sizes

#### 6. Time-Frequency Analysis
- **Spectral**: FFT, multitaper, wavelet transforms
- **Oscillations**: Peak detection, power analysis
- **Cross-Frequency**: Phase-amplitude coupling
- **Temporal Dynamics**: Time-varying spectral features

### Multi-Electrode Array (MEA) Analyses

- **Spike Sorting**: Kilosort-compatible, PCA/ICA clustering
- **Network Dynamics**: Population activity, avalanche analysis
- **Spatial Patterns**: Wave propagation, source localization
- **Connectivity**: Directed/undirected network inference

## 1. Getting Started

This section will guide you through setting up the repository on your machine and understanding the fundamental steps to run analyses.

### Prerequisites

Ensure you have the following installed on your system:

- Python 3.10 or higher
- Jupyter Notebook
- MATLAB (optional, for some analyses)
- Git (for version control)

## 2. Installation

To get started with this repository, follow the steps below:

1. **Clone the Repository**: Open Terminal (Mac) or Command Prompt (Windows), and run:

   ```bash
   git clone https://github.com/clement-lo/my-in-vitro-electrophysiology-analysis.git
   cd my-in-vitro-electrophysiology-analysis
   ```

2. **Create a Virtual Environment**:

   ```bash
   python3 -m venv env
   source env/bin/activate  # For Mac/Linux
   .\env\Scripts\activate  # For Windows
   ```

3. **Install Required Python Libraries**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**:
   Run `python --version` and `pip list` to confirm Python and required packages are installed correctly.

## 3. Usage Guide

This repository provides tools for analyzing electrophysiological data using Python scripts, Jupyter Notebooks, and MATLAB. Follow the instructions below to understand how to use each type of script.

### Python Scripts (.py)

1. **Navigate to the Python Scripts Directory**:
   ```bash
   cd scripts/python
   ```
2. **Run a Script**:
   For example, to run a synaptic input-output analysis script:
   ```bash
   python synaptic_input_output_analysis.py
   ```
3. **View Output**: The output will be saved in the `results/` directory, and visualizations will be generated as PNG files.

#### Jupyter Notebooks (.ipynb)

1. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```
2. **Open the Notebook**: Navigate to the `notebooks/` directory and open `Synaptic_Input_Output_Analysis.ipynb`.
3. **Run Cells**: Click "Run" for each cell sequentially or "Run All" from the "Cell" menu.

#### MATLAB Scripts

1. **Open MATLAB**.
2. **Navigate to the MATLAB Scripts Directory**: Use the "Current Folder" panel or type in the MATLAB command window:
   ```matlab
   cd('scripts/matlab')
   ```
3. **Run the Script**:
   ```matlab
   run('synaptic_input_output_analysis.m')
   ```
4. **View Output**: Plots and results will be displayed in the MATLAB environment.

## 4. Data Formats and Handling
This repository supports various data formats commonly used in in vitro electrophysiology research, including:

ABF (Axon Binary Format): Supported by libraries like PyABF for easy data handling.
Neo Data Format: Supported by libraries like Neo for standardized data handling.
MATLAB Files: For users with custom data formats, we provide functions to load and preprocess these data types.
Ensure your data is properly formatted and organized in the data/ directory before running any analysis.

## 5. Implementation Structure

### 5.1 Patch-Clamp & Whole-Cell Analyses (`scripts/patch_clamp_whole_cell_in_vitro/`)

This section features a dual-implementation approach:

#### **Dual Version Architecture**
- **v1 (Educational)**: Simple, clear implementations for learning and quick analyses
- **v2 (Research-Grade)**: Higher throughput framework with extensive features

#### **Available Implementations**

| Analysis Type | Python v1 | Python v2 | MATLAB v1 | MATLAB v2 | Notebooks |
|--------------|-----------|-----------|-----------|-----------|-----------|
| Action Potentials | ✅ | ✅ | ✅ | ✅ | ✅ |
| Synaptic Currents | ✅ | ✅ | ✅ | 🔧 | ✅ |
| Ion Channels | ✅ | ✅ | ✅ | 🔧 | ✅ |
| Network Connectivity | ✅ | ✅ | ✅ | 🔧 | ✅ |
| Pharmacology | ✅ | ✅ | ✅ | 🔧 | ✅ |
| Time-Frequency | ✅ | ✅ | ✅ | 🔧 | ✅ |

#### **Python Implementations**

**v1 Structure** (`python/v1/`):
```
├── action_potential_analysis.py          # AP detection and characterization
├── synaptic_current_analysis.py          # EPSC/IPSC analysis
├── ion_channels_kinetics.py              # Channel properties
├── network_connectivity_plasticity_analysis.py  # Connectivity metrics
├── pharmacological_modulation.py         # Drug effects
├── time_frequency_analysis.py            # Spectral analysis
└── *_config.yaml                         # Configuration files for each module
```

**v2 Structure** (`python/v2/`):
```
├── core/                                 # Framework infrastructure
│   ├── base_analyzer.py                  # Abstract base classes
│   ├── data_loader.py                    # Multi-format loading
│   └── config_manager.py                 # Configuration system
├── modules/                              # Analysis modules
│   ├── action_potential/                 # Comprehensive AP analysis
│   ├── synaptic/                        # Event detection & kinetics
│   ├── ion_channels/                    # State modeling
│   └── ...                              # Other specialized modules
├── common/                              # Shared utilities
│   ├── preprocessing/                   # Signal processing
│   ├── statistics/                      # Statistical tools
│   └── visualization/                   # Plotting utilities
└── workflows/                           # Complete pipelines
```

#### **MATLAB Implementations** (`matlab/v1/`, `matlab/v2/`)
- Parallel implementations for MATLAB users
- v1: Direct function-based approach
- v2: Object-oriented framework (in development)

#### **Interactive Notebooks** (`notebooks/v1/`)
- Step-by-step tutorials
- Visual exploration of concepts
- Ready-to-use analysis templates

### 5.2 Multi-Electrode Array Analyses (`scripts/mea_in_vitro/`)

MEA analyses are implemented as specialized tools without v1/v2 separation:

#### **Python Scripts**
- `advanced_spike_sorting_clustering.py` - Multi-channel spike sorting with PCA/ICA
- `connectivity_analysis.py` - Network connectivity inference
- `network_dynamics_analysis.py` - Population dynamics and synchronization
- `pharmacological_modulation_analysis.py` - Multi-site drug effect analysis

#### **Key Features**
- Optimized for high-channel-count data
- Spatial analysis capabilities
- Population-level metrics
- Parallel processing support

#### **Why No v1/v2 for MEA?**
MEA analysis is inherently complex and specialized, requiring sophisticated algorithms from the start. The implementations are already research-grade with built-in configurability.

## 6. Extending the Framework

### Adding Custom Analyses

#### For v1 (Simple approach):
```python
# Create new analysis in scripts/patch_clamp_whole_cell_in_vitro/python/v1/
def custom_analysis(data, params):
    # Your analysis logic
    return results
```

#### For v2 (Framework approach):
```python
# Extend base analyzer in v2
from scripts.patch_clamp_whole_cell_in_vitro.python.v2.core.base_analyzer import BaseAnalyzer

class CustomAnalyzer(BaseAnalyzer):
    def analyze(self, data, **kwargs):
        # Your sophisticated analysis
        return self.create_results(...)
```

### Contributing Guidelines
See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## 7. Dependencies & Integration

### Core Dependencies
```
numpy>=1.21.0          # Numerical computing
scipy>=1.7.0           # Scientific computing
matplotlib>=3.4.0      # Visualization
pandas>=1.3.0          # Data manipulation
pyabf>=2.3.0           # ABF file support
pynwb>=2.0.0           # NWB format support
neo>=0.10.0            # Multi-format compatibility
h5py>=3.0.0            # HDF5 support
pyyaml>=5.4.0          # Configuration files
tqdm>=4.62.0           # Progress bars
scikit-learn>=0.24.0   # Machine learning tools
networkx>=2.6.0        # Network analysis
```

### Library Integrations
- **Neo**: Unified data object model for electrophysiology
- **PyABF**: Native support for Axon Binary Format files
- **PyNWB**: Neurodata Without Borders standard compliance
- **SciPy**: Signal processing and statistics
- **NetworkX**: Graph theory and network analysis

## 8. Validation & Testing

### Automated Validation
```bash
# Run validation suite
python scripts/utils/validate_modules.py

# MATLAB validation
matlab -batch "run('scripts/utils/validate_modules_matlab.m')"
```

### Test Data Generation
```bash
# Generate test datasets
python scripts/test_data_generator.py --format all
```

### Quality Assurance
- Unit tests for core functionality
- Integration tests for workflows
- Validation against published methods
- Continuous integration ready

## 9. Documentation & Resources

### Available Documentation
- **Main README**: This file
- **Module README**: `scripts/patch_clamp_whole_cell_in_vitro/README.md`
- **v2 Framework**: `scripts/patch_clamp_whole_cell_in_vitro/python/v2/README.md`
- **Quick Start**: `QUICKSTART.md`
- **Migration Guide**: `MIGRATION_GUIDE.md`

### Interactive Learning
- **Jupyter Notebooks**: Step-by-step tutorials in `notebooks/v1/`
- **Example Workflows**: Complete analysis pipelines
- **Test Data**: Sample datasets in multiple formats

## 10. Future Development

### Planned Enhancements
- **v2 MATLAB**: Complete object-oriented MATLAB implementation
- **Deep Learning**: Integration with spike sorting neural networks
- **Cloud Support**: Distributed processing for large datasets
- **GUI Interface**: User-friendly graphical interface for v1 tools
- **Real-time Analysis**: Online processing capabilities

### Research Applications
This framework is designed to support cutting-edge research in:
- Synaptic plasticity mechanisms
- Neural circuit mapping
- Drug discovery and screening
- Disease model characterization
- Brain-computer interfaces

## 11. License

This repository is available under the GNU General Public License v3.0 (GPL-3.0) for academic and research use. See [LICENSE](LICENSE) for details.

For commercial licensing inquiries, please contact the repository maintainer.

## 12. Author & Contact

**Clement Lo**  
*BSc Applied Medical Sciences, MRes Translational Neuroscience (UCL)*

This repository demonstrates the application of systematic analysis approaches from diagnostic development to computational neuroscience, showcasing transferable skills in signal processing, statistical validation, and research software engineering.

## 13. References

### Core Libraries
- [Neo](https://neo.readthedocs.io/en/latest/) - Unified electrophysiology data handling
- [PyABF](https://github.com/swharden/pyabf) - Axon Binary Format support
- [PyNWB](https://pynwb.readthedocs.io/) - Neurodata Without Borders
- [SciPy](https://scipy.org/) - Scientific computing in Python
- [NetworkX](https://networkx.org/) - Network analysis

### Electrophysiology Methods
- Pernía-Andrade et al. (2012) - Spike sorting methodology
- Stimberg et al. (2014) - Brian 2 simulator
- Rossant et al. (2016) - Spike sorting review
- Jun et al. (2017) - Real-time spike sorting

### Standards & Best Practices
- [NWB](https://www.nwb.org/) - Neurodata standardization
- [FAIR](https://www.go-fair.org/) - Data principles
- [BIDS](https://bids.neuroimaging.io/) - Brain imaging data structure

---

*Repository developed to demonstrate computational neuroscience capabilities for research positions in neurophysiology and neuroengineering.*
