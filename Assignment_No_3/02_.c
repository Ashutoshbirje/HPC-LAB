#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void matrix_add(int **a, int **b, int **c, int n) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            c[i][j] = a[i][j] + b[i][j];
}

int main() {
    int sizes[] = {250, 500, 750, 1000, 2000};
    
    for (int t = 2; t <= 8; t *= 2) {
        omp_set_num_threads(t);
        printf("\nThreads: %d\n", t);
        for (int s = 0; s < 5; s++) {
            int n = sizes[s];
            int **a = malloc(n * sizeof(int *));
            int **b = malloc(n * sizeof(int *));
            int **c = malloc(n * sizeof(int *));
            for (int i = 0; i < n; i++) {
                a[i] = malloc(n * sizeof(int));
                b[i] = malloc(n * sizeof(int));
                c[i] = malloc(n * sizeof(int));
                for (int j = 0; j < n; j++) {
                    a[i][j] = rand() % 100;
                    b[i][j] = rand() % 100;
                }
            }

            double start = omp_get_wtime();
            matrix_add(a, b, c, n);
            double end = omp_get_wtime();

            printf("Matrix Size: %d x %d, Time: %lf seconds\n", n, n, end - start);

            for (int i = 0; i < n; i++) {
                free(a[i]); free(b[i]); free(c[i]);
            }
            free(a); free(b); free(c);
        }
    }

    return 0;
}
