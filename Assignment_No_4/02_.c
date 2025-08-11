#include <stdio.h>
#include <omp.h>

#define BUFFER_SIZE 5
#define PRODUCE_COUNT 10

int buffer[BUFFER_SIZE];
int count = 0; 

void producer() {
    for (int i = 1; i <= PRODUCE_COUNT; i++) {
        #pragma omp critical
        {
            if (count < BUFFER_SIZE) {
                buffer[count] = i;
                count++;
                printf("Produced: %d (Buffer count: %d)\n", i, count);
            }
        }
    }
}

void consumer() {
    for (int i = 1; i <= PRODUCE_COUNT; i++) {
        #pragma omp critical
        {
            if (count > 0) {
                int item = buffer[count - 1];
                count--;
                printf("Consumed: %d (Buffer count: %d)\n", item, count);
            }
        }
    }
}

int main() {
    #pragma omp parallel sections
    {
        #pragma omp section
        producer();

        #pragma omp section
        consumer();
    }

    return 0;
}
