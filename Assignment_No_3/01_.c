#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {
    int n;
    printf("Enter size of vectors: ");
    scanf("%d", &n);

    int *a = (int *)malloc(n * sizeof(int));
    int *b = (int *)malloc(n * sizeof(int));
    int dot_product = 0;

    printf("Enter elements of vector A:\n");
    for (int i = 0; i < n; i++) scanf("%d", &a[i]);

    printf("Enter elements of vector B:\n");
    for (int i = 0; i < n; i++) scanf("%d", &b[i]);

    double start = omp_get_wtime();

    #pragma omp parallel for reduction(+:dot_product)
    for (int i = 0; i < n; i++) {
        dot_product += a[i] * b[i];
    }

    double end = omp_get_wtime();

    printf("Minimum scalar product (dot product) = %d\n", dot_product);
    printf("Time taken: %lf seconds\n", end - start);

    free(a);
    free(b);
    return 0;
}
