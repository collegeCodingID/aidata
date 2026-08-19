# Roadmap

The roadmap is intentionally ordered by user value and architectural impact, not by feature count.

## 0.6 — Data inspection foundation

- [ ] Dataset profiling
- [ ] Streaming statistics
- [ ] Dataset validation API
- [ ] CLI: `info`, `validate`, `profile`, `stats`
- [ ] CSV/NPY/NPZ conversion
- [ ] Better sampling utilities

## 0.7 — ML data tooling

- [ ] Dataset quality score
- [ ] Stratified splitting
- [ ] Balanced/weighted sampling
- [ ] Outlier detection integration
- [ ] Visualization companion package
- [ ] Optional compression backends

## 0.8 — Training and production workflows

- [ ] Distributed training sampler
- [ ] Better mmap/read-path experiments
- [ ] Dataset drift detection
- [ ] More extensive compatibility fixtures
- [ ] Large-index performance improvements

## 0.9 — Dataset lifecycle

- [ ] Dataset versioning
- [ ] Dataset diff
- [ ] Dataset manifests
- [ ] Reproducible dataset snapshots

## 1.0 — Stability

The 1.0 milestone should focus on stability rather than maximum feature count:

- [ ] stable public API
- [ ] explicit file-format compatibility policy
- [ ] long-term compatibility fixtures
- [ ] performance regression suite
- [ ] documentation coverage
- [ ] security review of parser boundaries

## Explicit non-goals

AIDATA should not become:

- a full pandas replacement
- a general-purpose database
- a model-training framework
- a model-serving platform
- a giant visualization framework

The project should stay focused on ML dataset storage and access.
