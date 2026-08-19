# AIDATA Documentation

AIDATA is a chunked numerical dataset format and Python library for machine-learning workloads.

## Documentation map

- [Getting started](getting-started.md) — install AIDATA and create/read a dataset.
- [API reference](api.md) — public classes, methods, and exceptions.
- [PyTorch integration](pytorch.md) — datasets, loaders, workers, and training patterns.
- [Format specification](format.md) — AIDATA v1 binary layout and validation rules.
- [Architecture](architecture.md) — how the writer, reader, cache, and integrations fit together.
- [Reliability and security](reliability.md) — corruption detection, atomic writes, and threat boundaries.
- [Benchmarking](benchmarking.md) — how to measure AIDATA fairly.
- [Roadmap](roadmap.md) — planned capabilities and design priorities.

## Design principle

AIDATA should remain small at its core. The core package owns dataset storage, integrity, indexed access, and training integration. Higher-level profiling, visualization, drift detection, and quality tooling should be layered on top rather than turning the core into an all-purpose data-science framework.

## Building the documentation website

The repository includes `mkdocs.yml` and a GitHub Pages workflow.

Local build:

```bash
pip install mkdocs-material
mkdocs serve
```

Production build:

```bash
mkdocs build --strict
```

Before publishing, replace `YOUR_USERNAME` in `mkdocs.yml`, `README.md`, `CITATION.cff`, and contribution examples with the real GitHub owner/repository.
