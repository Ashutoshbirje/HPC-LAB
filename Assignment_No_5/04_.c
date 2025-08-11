#include <stdio.h>
#include <omp.h>

#define N 8

int main() {
    int arr[N] = {1, 2, 3, 4, 5, 6, 7, 8};
    int prefix[N];

    prefix[0] = arr[0];
    #pragma omp parallel for
    for (int i = 1; i < N; i++) {
        prefix[i] = prefix[i - 1] + arr[i];
    }

    for (int i = 1; i < N; i++) {
        prefix[i] = prefix[i - 1] + arr[i];
    }

    printf("Prefix Sum:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", prefix[i]);
    }
    printf("\n");
    return 0;
}
