# Benchmarks

Timing benchmarks for the lfm-physics core routines.

## Running

```bash
pip install "lfm-physics[benchmark]"
cd benchmarks
pytest bench_evolver.py bench_fields.py -v --benchmark-sort=mean
```

Or for a quick text summary without pytest-benchmark:

```bash
python bench_evolver.py
python bench_fields.py
```

## Typical results (RTX 4060, 8 GB VRAM)

| Benchmark | Grid | Steps/sec |
|-----------|------|-----------|
| evolver (CPU, N=32) | 32³ | ~150 000 |
| evolver (CPU, N=64) | 64³ | ~18 000 |
| evolver (GPU, N=128) | 128³ | ~3 400 |
| evolver (GPU, N=256) | 256³ | ~350 |
| equilibrate_from_fields (N=64) | — | ~2 200/s |
| radial_profile (N=64) | — | ~40 000/s |
