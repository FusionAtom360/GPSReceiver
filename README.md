# GPS L1 C/A Software Receiver

This repository contains a Python reference implementation and a real-time C++20
port of a GPS L1 C/A software receiver.

## Repository layout

- `python/gpsrx/` - Python reference receiver package
- `python/tests/` - Python unit and regression tests
- `cpp/` - C++20 library, application, tests, and benchmarks
- `data/` - small permitted capture slices and dataset fetch metadata
- `tools/` - comparison, plotting, and metric scripts
- `docs/` - report sources and design notes
- `.github/workflows/` - CI and benchmark workflows

## Development

Python tooling is configured in `pyproject.toml`. The C++ project is configured
with CMake:

```powershell
python -m pip install -e ".[dev]"
python -m pytest
cmake -S cpp -B build
cmake --build build --config Release
ctest --test-dir build -C Release
```

Large captures are intentionally not committed. Use the dataset tooling in
`data/` to fetch approved captures and verify their checksums.

## References

- [GPS Interface Specification IS-GPS-200](https://navcen.uscg.gov/sites/default/files/pdf/gps/IS-GPS-200N.pdf)
