#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    int rank, size;
    int m, n, p; // A(m×n), B(n×p), C(m×p)
    double *A = NULL, *B = NULL, *C = NULL;
    double *local_A = NULL, *local_C = NULL;
    int rows_per_proc, remainder, local_rows;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter dimensions m n p for A(m×n) * B(n×p): ");
        fflush(stdout);
        scanf("%d %d %d", &m, &n, &p);

        // Allocate and initialize matrices
        A = (double*)malloc(m * n * sizeof(double));
        B = (double*)malloc(n * p * sizeof(double));
        C = (double*)malloc(m * p * sizeof(double));

        printf("Initializing matrices A and B...\n");
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                A[i*n + j] = rand() % 10; // random small values
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                B[i*p + j] = rand() % 10;
            }
        }
    }

    // Broadcast dimensions
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&p, 1, MPI_INT, 0, MPI_COMM_WORLD);

    rows_per_proc = m / size;
    remainder = m % size;
    local_rows = (rank < remainder) ? rows_per_proc + 1 : rows_per_proc;

    // Allocate local memory
    local_A = (double*)malloc(local_rows * n * sizeof(double));
    local_C = (double*)malloc(local_rows * p * sizeof(double));
    if (rank != 0) B = (double*)malloc(n * p * sizeof(double));

    // Broadcast B to all processes
    MPI_Bcast(B, n*p, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Scatter rows of A
    int *sendcounts = NULL, *displs = NULL;
    if (rank == 0) {
        sendcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        int offset = 0;
        for (int i = 0; i < size; i++) {
            int rows = (i < remainder) ? rows_per_proc + 1 : rows_per_proc;
            sendcounts[i] = rows * n;
            displs[i] = offset;
            offset += rows * n;
        }
    }

    MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE,
                 local_A, local_rows*n, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // Start timing
    double start = MPI_Wtime();

    // Compute local multiplication
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < p; j++) {
            local_C[i*p + j] = 0.0;
            for (int k = 0; k < n; k++) {
                local_C[i*p + j] += local_A[i*n + k] * B[k*p + j];
            }
        }
    }

    // End timing
    double end = MPI_Wtime();
    double local_time = end - start;
    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather results
    if (rank == 0) {
        int *recvcounts = sendcounts;
        int *rdispls = displs;
        MPI_Gatherv(local_C, local_rows*p, MPI_DOUBLE,
                    C, recvcounts, rdispls, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        printf("\nExecution Time with %d processes: %f seconds\n", size, max_time);

        // Print result matrix (for small sizes only)
        if (m <= 6 && n <= 6 && p <= 6) {
            printf("\nMatrix C = A * B:\n");
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < p; j++) {
                    printf("%6.1f ", C[i*p + j]);
                }
                printf("\n");
            }
        }

        free(A); free(B); free(C);
        free(sendcounts); free(displs);
    } else {
        MPI_Gatherv(local_C, local_rows*p, MPI_DOUBLE,
                    NULL, NULL, NULL, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        free(B);
    }

    free(local_A);
    free(local_C);

    MPI_Finalize();
    return 0;
}
