#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <sys/time.h>

void MatrixMultiply(int n, double *a, double *b, double *c);
void MatrixMatrixMultiply_NonBlocking(int n, double *a, double *b, double *c, MPI_Comm comm);

double get_wall_time() {
    struct timeval time;
    if (gettimeofday(&time, NULL)) {
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * 0.000001;
}

int main(int argc, char **argv) {
    int n = 64; // Matrix size
    int npes, myrank;
    double *a, *b, *c;
    MPI_Comm comm;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    int nlocal = n / sqrt(npes); // Compute local size

    a = (double*)malloc(nlocal * nlocal * sizeof(double));
    b = (double*)malloc(nlocal * nlocal * sizeof(double));
    c = (double*)calloc(nlocal * nlocal, sizeof(double));

    if (myrank == 0) {
        FILE *fp;
        fp = fopen("inputs/64x64.txt", "r");
        if (fp == NULL) {
            printf("Cannot open file.\n");
            exit(1);
        }
        for (int i = 0; i < nlocal * nlocal; i++) {
            fscanf(fp, "%lf", &a[i]);
        }
        for (int i = 0; i < nlocal * nlocal; i++) {
            fscanf(fp, "%lf", &b[i]);
        }
        fclose(fp);
    }

    double start_time = get_wall_time();

    MatrixMatrixMultiply_NonBlocking(n, a, b, c, MPI_COMM_WORLD);

    double end_time = get_wall_time();
    double elapsed_time = end_time - start_time;

    for (int i = 0; i < nlocal; i++) {
        for (int j = 0; j < nlocal; j++) {
            printf("%.2f ", c[i * nlocal + j]);
        }
        printf("\n");
    }

    if (myrank == 0) {
        printf("Elapsed time: %.6f seconds\n", elapsed_time);
    }

    free(a);
    free(b);
    free(c);

    MPI_Finalize();

    return 0;
}

void MatrixMatrixMultiply_NonBlocking(int n, double *a, double *b, double *c, MPI_Comm comm) { 
    int i, j, nlocal; 
    double *a_buffers[2], *b_buffers[2]; 
    int npes, dims[2], periods[2]; 
    int myrank, my2drank, mycoords[2]; 
    int uprank, downrank, leftrank, rightrank, coords[2]; 
    int shiftsource, shiftdest; 
    MPI_Status status; 
    MPI_Comm comm_2d; 
    MPI_Request reqs[4]; 
    /* Get the communicator related information */ 
    MPI_Comm_size(comm, &npes); 
    MPI_Comm_rank(comm, &myrank); 
    /* Set up the Cartesian topology */ 
    dims[0] = dims[1] = sqrt(npes); 
    /* Set the periods for wraparound connections */ 
    periods[0] = periods[1] = 1; 
    /* Create the Cartesian topology, with rank reordering */ 
    MPI_Cart_create(comm, 2, dims, periods, 1, &comm_2d); 
    /* Get the rank and coordinates with respect to the new topology */ 
    MPI_Comm_rank(comm_2d, &my2drank); 
    MPI_Cart_coords(comm_2d, my2drank, 2, mycoords); 
    /* Compute ranks of the up and left shifts */ 
    MPI_Cart_shift(comm_2d, 0, -1, &rightrank, &leftrank); 
    MPI_Cart_shift(comm_2d, 1, -1, &downrank, &uprank); 
    /* Determine the dimension of the local matrix block */ 
    nlocal = n/dims[0]; 
    /* Setup the a_buffers and b_buffers arrays */ 
    a_buffers[0] = a; 
    a_buffers[1] = (double *)malloc(nlocal*nlocal*sizeof(double)); 
    b_buffers[0] = b; 
    b_buffers[1] = (double *)malloc(nlocal*nlocal*sizeof(double)); 
    /* Perform the initial matrix alignment. First for A and then for B */ 
    MPI_Cart_shift(comm_2d, 0, -mycoords[0], &shiftsource, &shiftdest); 
    MPI_Sendrecv_replace(a_buffers[0], nlocal*nlocal, MPI_DOUBLE, 
    shiftdest, 1, shiftsource, 1, comm_2d, &status); 
    MPI_Cart_shift(comm_2d, 1, -mycoords[1], &shiftsource, &shiftdest); 
    MPI_Sendrecv_replace(b_buffers[0], nlocal*nlocal, MPI_DOUBLE, 
    shiftdest, 1, shiftsource, 1, comm_2d, &status); 
    /* Get into the main computation loop */ 
    for (i=0; i<dims[0]; i++) { 
        MPI_Isend(a_buffers[i%2], nlocal*nlocal, MPI_DOUBLE, 
        leftrank, 1, comm_2d, &reqs[0]); 
        MPI_Isend(b_buffers[i%2], nlocal*nlocal, MPI_DOUBLE, 
        uprank, 1, comm_2d, &reqs[1]); 
        MPI_Irecv(a_buffers[(i+1)%2], nlocal*nlocal, MPI_DOUBLE, 
        rightrank, 1, comm_2d, &reqs[2]); 
        MPI_Irecv(b_buffers[(i+1)%2], nlocal*nlocal, MPI_DOUBLE, 
        downrank, 1, comm_2d, &reqs[3]); 
        /* c = c + a*b */ 
        MatrixMultiply(nlocal, a_buffers[i%2], b_buffers[i%2], c); 
        for (j=0; j<4; j++) 
            MPI_Wait(&reqs[j], &status); 
    } 
    /* Restore the original distribution of a and b */ 
    MPI_Cart_shift(comm_2d, 0, +mycoords[0], &shiftsource, &shiftdest); 
    MPI_Sendrecv_replace(a_buffers[i%2], nlocal*nlocal, MPI_DOUBLE, 
    shiftdest, 1, shiftsource, 1, comm_2d, &status); 
    MPI_Cart_shift(comm_2d, 1, +mycoords[1], &shiftsource, &shiftdest); 
    MPI_Sendrecv_replace(b_buffers[i%2], nlocal*nlocal, MPI_DOUBLE, 
    shiftdest, 1, shiftsource, 1, comm_2d, &status); 
    MPI_Comm_free(&comm_2d); /* Free up communicator */ 
    free(a_buffers[1]); 
    free(b_buffers[1]); 
}

void MatrixMultiply(int n, double *a, double *b, double *c) { 
    int i, j, k; 
    for (i=0; i<n; i++) 
        for (j=0; j<n; j++) 
            for (k=0; k<n; k++) 
                c[i*n+j] += a[i*n+k]*b[k*n+j]; 
}