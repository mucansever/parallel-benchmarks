#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

void MatrixMultiply(int n, double *a, double *b, double *c);

double get_wall_time() {
    struct timeval time;
    if (gettimeofday(&time, NULL)) {
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * 0.000001;
}

int main(int argc, char **argv) {
    int n = 128; // Matrix size
    int npes, myrank;
    double *a, *b, *c;

    a = (double*)malloc(n * n * sizeof(double));
    b = (double*)malloc(n * n * sizeof(double));
    c = (double*)calloc(n * n, sizeof(double));

    if (myrank == 0) {
        FILE *fp;
        fp = fopen("inputs/128x128.txt", "r");
        if (fp == NULL) {
        printf("Cannot open file.\n");
        exit(1);
        }
        for (int i = 0; i < n * n; i++) {
        fscanf(fp, "%lf", &a[i]);
        }
        for (int i = 0; i < n * n; i++) {
        fscanf(fp, "%lf", &b[i]);
        }
        fclose(fp);
    }

    double start_time = get_wall_time();

    MatrixMultiply(n, a, b, c);

    double end_time = get_wall_time();
    double elapsed_time = end_time - start_time;

    // Print the result matrix c
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          printf("%.2f ", c[i * n + j]);
        }
        printf("\n");
      }

    if (myrank == 0) {
        printf("Elapsed time: %.6f seconds\n", elapsed_time);
    }

    free(a);
    free(b);
    free(c);

    return 0;
}

void MatrixMultiply(int n, double *a, double *b, double *c) { 
    int i, j, k; 
    for (i=0; i<n; i++) 
        for (j=0; j<n; j++) 
            for (k=0; k<n; k++) 
                c[i*n+j] += a[i*n+k]*b[k*n+j]; 
}