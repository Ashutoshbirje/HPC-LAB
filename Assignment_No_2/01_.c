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
    for (int i = 0; i < n; i++) {
        result[i] = vec[i] + scalar;
    }
    double end = omp_get_wtime();
    double time = end - start;
    printf("Time taken: %lf seconds (Sequential) \n", time);

    double start1 = omp_get_wtime();
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        result[i] = vec[i] + scalar;
    }
    double end1 = omp_get_wtime();
    double time1 = end1 - start1;
    printf("Time taken: %lf seconds (Parallel) \n", time1);
    
    double speedup = time / time1;
    printf("Speedup: %.3f\n", speedup);


    printf("Sample result: \n");
    for (int i = 0; i < (n < 10 ? n : 10); i++) {
        printf("%.2f ", result[i]);
    }
    printf("\n");

    free(vec);
    free(result);

    return 0;
}
