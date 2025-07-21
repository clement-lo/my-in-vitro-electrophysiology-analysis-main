# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive v1 (educational) and v2 (research-grade) implementations
- Multi-format data support in v2 (ABF, NWB, CSV, HDF5, Neo)
- Object-oriented architecture for v2 framework
- Complete test suite with example data
- Configuration management system for v2
- Jupyter notebooks for interactive learning

### Changed
- Reorganized codebase structure to separate patch-clamp and MEA analyses
- Consolidated utilities into common modules
- Standardized module organization with clear v1/v2 separation

### Fixed
- Import paths and module dependencies
- Data validation and error handling
- Documentation accuracy and completeness

### Removed
- Obsolete merge documentation and roadmaps
- Duplicate module files in v2
- Temporary scripts and generated artifacts

## [1.0.0] - 2024-01-XX

### Initial Release
- Basic electrophysiology analysis tools for:
  - Action potential detection and analysis
  - Synaptic event detection and characterization
  - Ion channel kinetics modeling
  - Network connectivity analysis
  - Pharmacological modulation studies
  - Time-frequency analysis
- Support for CSV data format (v1)
- MATLAB implementations for core analyses