#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

int main() {
    int n;
    float scalar;

    printf("Enter size of vector: ");
    scanf("%d", &n);

    printf("Enter scalar value: ");
    scanf("%f", &scalar);

    float *vec = (float *)malloc(n * sizeof(float));
    float *result = (float *)malloc(n * sizeof(float));

    for (int i = 0; i < n; i++) {
        vec[i] = i * 1.0f;
    }

    double start = omp_get_wtime();

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        result[i] = vec[i] + scalar;
    }

    double end = omp_get_wtime();

    printf("Time taken: %lf seconds\n", end - start);

    printf("Sample result: \n");
    for (int i = 0; i < (n < 10 ? n : 10); i++) {
        printf("%.2f ", result[i]);
    }
    printf("\n");

    free(vec);
    free(result);

    return 0;
}
