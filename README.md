Parallel programs for collatz and fractal calculation.

The programs are implemented on the TACC machines in the parallel lab of Texas State University.

OpenMP + CUDA Collatz - Hybrid OpenMP and CUDA program for collatz code.Used 512 threads per block on the GPU and did not specify the number of threads on the CPU, i.e., did not use num_threads().

OpenMP + CUDA + MPI Collatz - Hybrid OpenMP, CUDA and MPI program for collatz code. Used the completed CPU source code from Part 1 into a file named collatz_hyb.cpp and reused the collatz_hyb.cu file without making any changes to it. Then added MPI support to the CPU code in such a way that each node computes one nth of the sequences (using a blocked distribution), where n is the number of MPI processes. Within a compute node, the workload is still split among the CPU cores and the GPU according to the percentage given on the command line.

OpenMP + CUDA + MPI Fractal - Hybrid OpenMP, CUDA and MPI program for Fractal code. Used 512 threads per block on the GPU and did not specify the number of threads on the CPU, i.e., did not use num_threads().Parallelized the middle (for-row) loop using OpenMP.
