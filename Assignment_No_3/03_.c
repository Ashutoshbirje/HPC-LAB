#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define SIZE 200

void static_schedule_add(int *a, int *res, int scalar, int chunk) {
    #pragma omp parallel for schedule(static, chunk)
    for (int i = 0; i < SIZE; i++) {
        res[i] = a[i] + scalar;
    }
}

void dynamic_schedule_add(int *a, int *res, int scalar, int chunk) {
    #pragma omp parallel for schedule(dynamic, chunk)
    for (int i = 0; i < SIZE; i++) {
        res[i] = a[i] + scalar;
    }
}

void nowait_demo(int *a, int *b, int *res) {
    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < SIZE/2; i++)
            res[i] = a[i] + b[i];

        #pragma omp for nowait
        for (int i = SIZE/2; i < SIZE; i++)
            res[i] = a[i] + b[i];
    }
}

int main() {
    int a[SIZE], res[SIZE], b[SIZE];
    for (int i = 0; i < SIZE; i++) {
        a[i] = i;
        b[i] = 2 * i;
    }

    int scalar = 5;
    int chunk_sizes[] = {1, 5, 10, 20, 50};

    for (int c = 0; c < 5; c++) {
        int chunk = chunk_sizes[c];

        double start = omp_get_wtime();
        static_schedule_add(a, res, scalar, chunk);
        double end = omp_get_wtime();
        printf("STATIC Chunk %d: %lf seconds\n", chunk, end - start);

        start = omp_get_wtime();
        dynamic_schedule_add(a, res, scalar, chunk);
        end = omp_get_wtime();
        printf("DYNAMIC Chunk %d: %lf seconds\n", chunk, end - start);
    }

    printf("\nNOWAIT Clause demo:\n");
    nowait_demo(a, b, res);

    return 0;
}
