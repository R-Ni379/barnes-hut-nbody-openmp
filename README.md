# Barnes–Hut N-Body Simulation (C + OpenMP)

Fast approximate N-body simulation with Barnes–Hut, parallelized using OpenMP.

## Build
gcc-14 -O3 -mcpu=native -fopenmp src/barnes_hut_openmp.c -o barnes_hut_openmp -lm


# Performance
For N = 10000 particles on my multi-core CPU (M3, Apple MacbookAir), the OpenMP version scales up to 4x faster than the serial baseline when using 8 threads

## Versions

- barnes_hut_basic: serial Barnes–Hut + velocity Verlet
- barnes_hut_openmp: OpenMP parallelized 
