# Changelog

All notable changes to the project are documented in this file.

Version numbers are of the form `1.0.0`.

Each version section may have subsections for: _Added_, _Changed_, _Removed_, _Deprecated_, and _Fixed_.

## [0.4.0]

### Fixed

- Fixed an issue where the maximum generation length was not properly configured, leading to truncated translations.
- Fixed tests that could not run in isolation before because of global variables.

### Added

- Added a parameter `use_backbone_max_length` for `MultimodalEmbedderConfig`.
- Added configuration tests.

### Changed

- Changed allowed `max_length` and `num_beams` parameters:
  - for `multimodalhugs-train`: `generation_max_length` and `generation_num_beams` are expected
  - for `multimodalhugs-generate`: `--max_length` and `--num_beams` are expected
