# 7. CONTRIBUTING.md
# 🤝 Contributing to AIDATA

Thank you for your interest in contributing! This document will help you get started.

---

## 🚀 Quick Start

1. **Fork** the repository on GitHub
2. **Clone** your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/aidata.git
   cd aidata
   ```
3. **Install** in development mode:
   ```bash
   pip install -e ".[dev]"
   ```
4. **Run tests** to ensure everything works:
   ```bash
   pytest tests/ -v
   ```

---

## 📋 Development Setup

### Requirements

- Python >= 3.8
- Git

### Install Development Dependencies

```bash
pip install -e ".[dev]"
```

This installs:
- `pytest` for testing
- `numpy`, `torch`, `zstandard` for runtime

### Run Tests

```bash
# All tests
pytest tests/ -v

# Specific test
pytest tests/test_basic.py::test_write_read_roundtrip -v

# With coverage
pytest tests/ --cov=aidata --cov-report=html
```

---

## 🐛 Reporting Bugs

Before reporting, please:
1. Check existing issues
2. Update to the latest version
3. Try to isolate the problem

**Good bug reports include:**
- AIDATA version (`python -c "import aidata; print(aidata.__version__)"`)
- Python version
- Operating system
- Minimal code to reproduce
- Expected vs actual behavior
- Full error traceback

---

## 💡 Suggesting Features

Feature requests are welcome! Please:
1. Describe the use case
2. Explain why existing features don\'t solve it
3. Propose an API (if applicable)

---

## 📝 Code Style

- Follow **PEP 8**
- Use **type hints** for public APIs
- Add **docstrings** to all classes and methods
- Keep functions focused and small

### Example

```python
def get_batch(self, start: int, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Read a contiguous batch of samples.

    Parameters
    ----------
    start : int
        Starting sample index.
    batch_size : int
        Number of samples to read.

    Returns
    -------
    tuple
        (X_batch, y_batch) as NumPy arrays.

    Raises
    ------
    IndexError
        If start is out of range.
    ValueError
        If batch_size <= 0.
    """
```

---

## 🧪 Adding Tests

All new features must include tests. Place tests in `tests/`.

### Test Template

```python
def test_my_feature():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.aidata")
        
        # Setup
        X = np.random.rand(100, 10).astype(np.float32)
        y = np.random.randint(0, 2, size=100, dtype=np.int64)
        
        writer = AIDATAWriter(path)
        writer.write(X, y, verbose=False)
        
        # Test
        reader = AIDATAReader(path)
        assert len(reader) == 100
        
        # Cleanup (automatic with TemporaryDirectory)
```

---

## 📖 Documentation

Documentation improvements are highly valued! You can:
- Fix typos in README/docs
- Add examples
- Improve docstrings
- Write tutorials

Docs are in:
- `README.md` — Overview
- `docs/` — Detailed guides
- Docstrings in source code

---

## 🔄 Pull Request Process

1. **Create a branch:**
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Make your changes**

3. **Add tests** for new functionality

4. **Run the test suite:**
   ```bash
   pytest tests/ -v
   ```

5. **Update documentation** if needed

6. **Commit with clear messages:**
   ```bash
   git commit -m "Add feature: sample-level shuffling"
   ```

7. **Push and open a Pull Request:**
   ```bash
   git push origin feature/my-feature
   ```

### PR Checklist

- [ ] Tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Code follows style guide
- [ ] Commit messages are clear

---

## 🏷️ Release Process

Maintainers only:

1. Update version in `src/aidata/__init__.py`
2. Update `CHANGELOG.md`
3. Create a git tag:
   ```bash
   git tag v0.6.0
   git push origin v0.6.0
   ```
4. Build and publish:
   ```bash
   python -m build
   python -m twine upload dist/*
   ```

---

## 📜 Code of Conduct

- Be respectful and inclusive
- Accept constructive criticism gracefully
- Focus on what\'s best for the community
- Show empathy towards others

---

## ❓ Questions?

- Open a GitHub Discussion
- Comment on an existing issue
- Email the maintainers

Thank you for contributing! 🎉

Dev by: aditya praveen sharma ( valkariyon group )
'''