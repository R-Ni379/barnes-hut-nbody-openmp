# Barnes–Hut N-Body Simulation (C + OpenMP)

Fast approximate N-body simulation with Barnes–Hut, parallelized using OpenMP.

## Build
gcc-14 -O3 -mcpu=native -fopenmp src/barnes_hut_openmp.c -o barnes_hut_openmp -lm

## Console input
./barnes_hut_openmp  N filename nsteps delta_t graphics n_threads

example: ./barnes_hut_openmp 2000 input_data/ellipse_N_02000.gal 80 0.0000125 0 8

- galsim: compiled program
- N: Number of stars/particles
- nsteps: number of time steps
- delta_t: timestep
- n_threads: number of threads to use

Note: Input .gal files re not included in this repository. The code expects binary files in that format.

## Comparison to reference solution

Reference solution
./barnes_hut_openmp 2000 input_data/ellipse_N_02000.gal 1000 0.000001 0 1


./utils/compare_gal_files/compare_gal_files 2000 result.gal ref_output_data/local_reference_verlet_N_2000.gal


## Performance
For N = 10000 particles on my multi-core CPU (M3, Apple MacbookAir), the OpenMP version scales up to 4x faster than the serial baseline when using 8 threads

## Versions

- barnes_hut_basic: serial Barnes–Hut + velocity Verlet
- barnes_hut_openmp: OpenMP parallelized 
