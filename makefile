.PHONY: all clean

all: barnes_hut_openmp barnes_hut_basic utils/compare_gal_files/compare_gal_files

barnes_hut_openmp: src/barnes_hut_openmp.c
	#gcc main version (OpenMP)
	gcc-14 -O3 -mcpu=native -fopenmp src/barnes_hut_openmp.c -o barnes_hut_openmp -lm

barnes_hut_basic: src/barnes_hut_basic.c
	#gcc basic version
	gcc-14 -O2 -mcpu=native src/barnes_hut_basic.c -o barnes_hut_basic -lm


clean:
	rm -f barnes_hut_openmp barnes_hut_basic 
