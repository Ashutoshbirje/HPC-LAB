#include <stdio.h>
#include <omp.h>

int main() {
    long long int num_steps;
    double step, x, pi, sum = 0.0;

    printf("Enter number of intervals (larger = better precision): ");
    scanf("%lld", &num_steps);

    step = 1.0 / (double) num_steps;

    double start = omp_get_wtime();

    #pragma omp parallel
    {
        double x, local_sum = 0.0;
        int id = omp_get_thread_num();
        int nthrds = omp_get_num_threads();

        for (long long i = id; i < num_steps; i += nthrds) {
            x = (i + 0.5) * step;
            local_sum += 4.0 / (1.0 + x * x);
        }

        #pragma omp atomic
        sum += local_sum;
    }

    pi = step * sum;

    double end = omp_get_wtime();

    printf("Calculated Pi = %.15f\n", pi);
    printf("Time taken: %lf seconds\n", end - start);

    return 0;
}
