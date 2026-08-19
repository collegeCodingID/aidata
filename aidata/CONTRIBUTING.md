# Contributing to AIDATA

Thank you for contributing. AIDATA is intentionally small, so changes should solve a real dataset or ML-engineering problem without unnecessarily expanding the core API.

## Development requirements

- Python 3.10+
- Git
- NumPy
- Pytest
- PyTorch only when working on the optional integration

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/aidata.git
cd aidata
python -m pip install -e ".[dev]"
```

Run tests:

```bash
pytest -q
```

## Before implementing a large feature

Open an issue first when a change affects:

- the file format
- public APIs
- compression
- indexing
- caching
- PyTorch worker behavior
- performance characteristics

Describe the use case and proposed API.

## Code rules

- Follow PEP 8.
- Add type hints to new public APIs where practical.
- Add docstrings to new public classes and methods.
- Keep functions focused.
- Avoid unnecessary dependencies in the core package.
- Do not silently change file-format semantics.

## Tests

New behavior should have tests. For format/parser changes, include malformed-input and boundary cases.

Useful commands:

```bash
pytest tests/test_basic.py -q
pytest tests/test_hardening.py -q
pytest tests/test_fuzzing.py -q
pytest tests/test_api_compatibility.py -q
pytest -q
```

For performance-sensitive changes, run the benchmarks and report the hardware and configuration.

## Pull requests

A good PR should contain:

- a focused change
- tests
- documentation when user behavior changes
- changelog entry when appropriate
- compatibility discussion for format/API changes
- benchmark evidence for I/O or training-path changes

## Commit messages

Prefer concise, descriptive commits, for example:

```text
Add strict trailing-zlib validation
Fix worker-local reader lifecycle
Document AIDATA v1 index layout
```

## Release process

Maintainers should:

1. update the package version
2. update `CHANGELOG.md`
3. run the complete test suite
4. build the package
5. inspect the wheel/sdist
6. publish to PyPI
7. create the matching git tag

Example build command:

```bash
python -m build
```

## Documentation

The main entry point is `README.md`. Detailed documentation lives in `docs/`.

If an API behavior changes, update both the API documentation and at least one runnable example where appropriate.
