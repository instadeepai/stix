# Changelog

## Release 0.1.1 - 2026-09-11

### Added

- `THIRD_PARTY_NOTICES.md`, documenting the licences of direct dependencies,
  the proprietary NVIDIA CUDA/cuDNN packages pulled in by the `gpu` extra,
  and a dated dependency snapshot for full disclosure.

## Release 0.1.0 - 2026-09-11

### Added

- Initial release of `stix`, a unifying framework for flows and diffusions using stochastic interpolants.
- Core library (`src/stix`): `core`, `nn`, `sampling`, `training`, `typing` modules.
- Test suite covering sampling, training, and network components.
- Sphinx documentation and tutorial notebooks.
- CI workflows for tests/linters, docs build/deploy, and PyPI publishing.
