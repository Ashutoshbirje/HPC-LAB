#include <stdio.h>
#include <omp.h>

int main() {
    int num_threads;

    printf("Enter number of threads: ");
    scanf("%d", &num_threads);

    printf("\nSequential Hello, World:\n");
    for (int i = 0; i < num_threads; i++) {
        printf("Hello, World from thread %d (Sequential)\n", i);
    }

    omp_set_num_threads(num_threads);

    printf("\nParallel Hello, World using OpenMP:\n");
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        printf("Hello, World from thread %d (Parallel)\n", thread_id);
    }

    return 0;
}
