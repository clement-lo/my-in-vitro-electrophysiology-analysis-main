# Contributing to In Vitro Electrophysiology Analysis

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please be respectful and constructive in all interactions.

## How to Contribute

### Reporting Issues

1. Check existing issues to avoid duplicates
2. Use issue templates when available
3. Provide detailed information:
   - Steps to reproduce
   - Expected behavior
   - Actual behavior
   - Environment details (OS, Python version, etc.)

### Suggesting Enhancements

1. Check if the enhancement has already been suggested
2. Clearly describe the enhancement and its benefits
3. Provide examples of how it would be used

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests to ensure nothing is broken
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Development Guidelines

### Code Style

#### Python
- Follow PEP 8 style guide
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and small

#### MATLAB
- Use consistent indentation (4 spaces)
- Add comments for complex logic
- Use descriptive variable names

### Directory Structure

When adding new features:
- Educational/simple implementations go in `v1/`
- Research-grade implementations go in `v2/`
- Shared utilities go in `common/`
- Always include appropriate `__init__.py` files

### Testing

- Write tests for new features
- Ensure existing tests pass
- Aim for >80% code coverage
- Test with multiple data formats (if applicable)

### Documentation

- Update relevant README files
- Add docstrings to new functions
- Include usage examples
- Update CHANGELOG.md

## Version Guidelines

### v1 (Educational)
- Keep it simple and readable
- Minimize dependencies
- Focus on clarity over performance
- Include extensive comments

### v2 (Research-Grade)
- Follow object-oriented design principles
- Ensure extensibility
- Optimize for performance where needed
- Include comprehensive error handling

## Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Keep the first line under 50 characters
- Reference issues and pull requests when relevant
- Be descriptive but concise

Example:
```
Add wavelet-based spike detection

- Implement continuous wavelet transform method
- Add configuration options for wavelet parameters
- Include unit tests and documentation
- Fixes #123
```

## Review Process

1. All submissions require review
2. Changes may be requested for:
   - Code style consistency
   - Test coverage
   - Documentation completeness
   - Performance concerns
3. Be patient and constructive during review

## Questions?

Feel free to open an issue for any questions about contributing. We're here to help!

Thank you for contributing to make this project better! 🎉