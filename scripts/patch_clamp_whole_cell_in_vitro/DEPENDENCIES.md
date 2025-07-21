# Cross-Version Dependencies and Relationships

This document describes the relationships between v1 and v2 implementations and their dependencies.

## Version Independence

**Key Principle**: v1 and v2 are designed to be independent implementations with no cross-dependencies.

### v1 - Educational Implementation
- **Philosophy**: Self-contained, simple, minimal dependencies
- **External Dependencies**: NumPy, SciPy, Matplotlib, Pandas
- **Internal Dependencies**: None - each module is standalone
- **Data Flow**: Direct file I/O, no shared components with v2

### v2 - Research Framework
- **Philosophy**: Modular, extensible, comprehensive
- **External Dependencies**: Extended set including PyABF, PyNWB, Neo
- **Internal Dependencies**: Structured hierarchy within v2 only
- **Data Flow**: Unified data pipeline, no reliance on v1

## Architectural Separation

```
patch_clamp_whole_cell_in_vitro/
├── python/
│   ├── v1/  ← No imports from v2
│   └── v2/  ← No imports from v1
```

## Shared Resources

While code is independent, these resources are shared:

### 1. Test Data
- Location: `/test_data/` at project root
- Used by: Both v1 and v2 for examples and testing
- Format: Standardized CSV and NWB files

### 2. Documentation
- Main README explains both versions
- Each version has dedicated documentation
- Notebooks demonstrate both approaches

### 3. Configuration Concepts
- v1: Simple YAML files per module
- v2: Unified configuration system
- No shared configuration files

## Migration Path

For users transitioning from v1 to v2:

### Data Migration
```python
# v1 style
data = pd.read_csv('data.csv')
spikes = detect_action_potentials(data['voltage'], threshold=-20)

# v2 style
loader = DataLoader()
data = loader.load_file('data.csv')
analyzer = ActionPotentialAnalyzer(config)
results = analyzer.analyze(data)
```

### Feature Mapping

| v1 Function | v2 Equivalent |
|-------------|---------------|
| `detect_action_potentials()` | `ActionPotentialAnalyzer.analyze()` |
| `analyze_io_relationship()` | `SynapticAnalyzer.analyze_io()` |
| `fit_activation_curve()` | `IonChannelAnalyzer.fit_activation()` |
| `calculate_connectivity()` | `NetworkAnalyzer.calculate_connectivity()` |
| `analyze_dose_response()` | `PharmacologyAnalyzer.dose_response()` |
| `compute_spectrogram()` | `TimeFrequencyAnalyzer.spectrogram()` |

## Dependency Management

### Installing v1 Only
```bash
pip install numpy scipy matplotlib pandas pyyaml
```

### Installing v2 Only
```bash
pip install -r python/v2/requirements.txt
```

### Installing Both
```bash
# Install v1 requirements
pip install numpy scipy matplotlib pandas pyyaml

# Install additional v2 requirements
pip install pyabf pynwb neo h5py tqdm
```

## Best Practices

### 1. No Cross-Version Imports
```python
# ❌ Don't do this in v2
from ..v1.action_potential_analysis import detect_action_potentials

# ✅ Each version is self-contained
from .modules.action_potential import ActionPotentialAnalyzer
```

### 2. Clear Version Selection
```python
# Be explicit about which version you're using
# For v1:
sys.path.append('path/to/v1')

# For v2:
sys.path.append('path/to/v2')
```

### 3. Parallel Development
- v1 remains stable for educational use
- v2 evolves for research needs
- No breaking changes affect the other version

## Common Pitfalls

### 1. Namespace Conflicts
- Both versions may have similarly named modules
- Use explicit imports or virtual environments

### 2. Data Format Assumptions
- v1 assumes simple CSV format
- v2 handles multiple formats
- Don't mix data loaders between versions

### 3. Configuration Confusion
- v1 configs are simple key-value pairs
- v2 configs are hierarchical with validation
- Keep configurations separate

## Future Considerations

### Maintaining Separation
- New features in v2 don't affect v1
- v1 remains frozen for stability
- Clear upgrade path documented

### Potential Shared Components
If sharing becomes necessary:
1. Create a `common/` directory
2. Version shared components carefully
3. Maintain backward compatibility

## Conclusion

The v1/v2 separation allows:
- Educational use without complexity
- Research use without compromises
- Clear progression path
- Independent evolution