#include <stdio.h>
#include <omp.h>

#define N 3

int main() {
    int A[N][N] = {{1, 2, 3},
                   {4, 5, 6},
                   {7, 8, 9}};
    int v[N] = {1, 2, 3};
    int result[N] = {0};

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            result[i] += A[i][j] * v[j];
        }
    }

    printf("Result Vector:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", result[i]);
    }
    printf("\n");
    return 0;
}
