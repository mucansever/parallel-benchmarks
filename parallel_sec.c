#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>

#define SIZE 8
#define FILENAME "inputs/8x8.txt"

double get_wall_time() {
    struct timeval time;
    if (gettimeofday(&time, NULL)) {
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * 0.000001;
}

int main() {
    int i, j, k;

    double *a = (double*)malloc(SIZE * SIZE * sizeof(double));
    double *b = (double*)malloc(SIZE * SIZE * sizeof(double));
    double *c = (double*)calloc(SIZE * SIZE, sizeof(double));

    FILE *fp;
    fp = fopen(FILENAME, "r");
    if (fp == NULL) {
        printf("Cannot open file.\n");
        exit(1);
    }
    for (i = 0; i < SIZE * SIZE; i++) {
        fscanf(fp, "%lf", &a[i]);
    }
    for (i = 0; i < SIZE * SIZE; i++) {
        fscanf(fp, "%lf", &b[i]);
    }
    fclose(fp);

    double start_time = get_wall_time();

    #pragma omp parallel for private(i, j, k) shared(a, b, c)
    for (i = 0; i < SIZE; i++) {
        for (j = 0; j < SIZE; j++) {
            double sum = 0.0;
            for (k = 0; k < SIZE; k++) {
                sum += a[i * SIZE + k] * b[k * SIZE + j];
            }
            c[i * SIZE + j] = sum;
        }
    }

    #pragma omp parallel for private(j)
    for (i = 0; i < SIZE; i++) {
        for (j = 0; j < SIZE; j++) {
            // Perform any computations on c[i][j] within this parallel region
        }
    }

    double end_time = get_wall_time();
    double elapsed_time = end_time - start_time;

    printf("Matrix C:\n");
    for (i = 0; i < SIZE; i++) {
        for (j = 0; j < SIZE; j++) {
            printf("%lf ", c[i * SIZE + j]);
        }
        printf("\n");
    }

    printf("Elapsed time: %.6f seconds\n", elapsed_time);

    free(a);
    free(b);
    free(c);

    return 0;
}
