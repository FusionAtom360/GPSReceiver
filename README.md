# GPS L1 C/A Software Receiver

This repository contains a Python reference implementation for GPS L1 C/A signal processing. This includes GPS PRN generation, deterministic synthetic IQ signal generation, and FFT-based coarse acquisition. 

## Repository layout

- `python/gpsrx/` - Python package, including:
    - `synthetic.py` - GPS C/A PRN generation and multi-satellite synthetic IQ
      channel generation with Doppler, C/N0, navigation data, and noise
    - `acquisition.py` - FFT-based coarse acquisition for Doppler and code phase
    - `config.py` - YAML-backed receiver configuration types and loading
- `python/tests/` - Python unit and regression tests for the implemented
  reference functionality
- `cpp/` - C++20 core library, command-line application, smoke tests, and
  optional benchmarks
- `data/` - permitted capture slices and dataset-fetch metadata
- `tools/` - capture fetching, correlation plotting, and observation
  comparison utilities
- `docs/` - report sources and design notes
- `.github/workflows/` - CI workflows

## Current functionality

The Python reference implementation currently provides:

- GPS PRN 1-32 generation using the standard G1/G2 linear-feedback shift
  registers
- Synthetic complex baseband samples for one or more satellites
- Configurable Doppler shifts, fractional code phases, C/N0 values, navigation
  bits, sampling rates, and seeded noise
- FFT-based coarse acquisition over a caller-supplied Doppler grid
- Noncoherent integration over multiple 1 ms code periods
- Detection metrics and estimates for PRN, Doppler, code phase, and peak
  magnitude
- Input validation for signal shape, finite values, search bins, offsets, and
  receiver parameters

## Development

Python tooling is configured in `pyproject.toml` and requires Python 3.11 or
newer. Install the package and development dependencies with:

```powershell
python -m pip install -e ".[dev]"
```

Run the Python tests with:

```powershell
python -m pytest
```

## Examples

Run the synthetic-channel example:

```powershell
python python/gpsrx/synthetic.py
```

Run the FFT-acquisition example:

```powershell
python python/gpsrx/acquisition.py
```

The examples generate synthetic data locally; they do not require a capture
file or live receiver input.

## Data and repository policy

Large captures are intentionally not committed. Use the dataset tooling in
`data/` to fetch approved captures and verify their checksums before using
them in local experiments.

## References

- [GPS Interface Specification IS-GPS-200](https://navcen.uscg.gov/sites/default/files/pdf/gps/IS-GPS-200N.pdf)
