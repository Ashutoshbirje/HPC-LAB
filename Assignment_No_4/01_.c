#include <stdio.h>
#include <omp.h>

#define N 20

int main() {
    int fib[N];
    fib[0] = 0;
    fib[1] = 1;

    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int i = 2; i < N; i++) {
                #pragma omp task firstprivate(i)
                {
                    #pragma omp taskwait
                    fib[i] = fib[i-1] + fib[i-2];
                }
            }
        }
    }

    printf("Fibonacci Sequence:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", fib[i]);
    }
    printf("\n");

    return 0;
}
