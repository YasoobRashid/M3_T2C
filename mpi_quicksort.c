#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void swap(int *a, int *b) {
    int t = *a;
    *a = *b;
    *b = t;
}

int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = (low - 1);

    for (int j = low; j <= high-1; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i+1], &arr[high]);
    return (i + 1);
}

void quicksort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);

        quicksort(arr, low, pi - 1);
        quicksort(arr, pi + 1, high);
    }
}

void merge(int *a, int n, int *b, int m, int *out) {
    int i = 0, j = 0, k = 0;
    while (i < n && j < m) {
        if (a[i] <= b[j])
            out[k++] = a[i++];
        else
            out[k++] = b[j++];
    }
    while (i < n)
        out[k++] = a[i++];
    while (j < m)
        out[k++] = b[j++];
}

int main(int argc, char* argv[]) {
    int rank, size;
    int n = 16;  // total number of elements
    int *data = NULL;
    int *sub_data;
    int sub_n;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    sub_n = n / size;
    sub_data = (int *)malloc(sub_n * sizeof(int));

    if (rank == 0) {
        data = (int *)malloc(n * sizeof(int));
        printf("Unsorted array:\n");
        for (int i = 0; i < n; i++) {
            data[i] = rand() % 100;
            printf("%d ", data[i]);
        }
        printf("\n");

        start_time = MPI_Wtime();  // Start timing
    }

    MPI_Scatter(data, sub_n, MPI_INT, sub_data, sub_n, MPI_INT, 0, MPI_COMM_WORLD);

    quicksort(sub_data, 0, sub_n - 1);

    MPI_Gather(sub_data, sub_n, MPI_INT, data, sub_n, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int *tmp = (int *)malloc(n * sizeof(int));
        for (int i = 0; i < size - 1; i++) {
            merge(data + i * sub_n, sub_n, data + (i + 1) * sub_n, sub_n, tmp);
            for (int j = 0; j < (sub_n * (i + 2)); j++)
                data[j] = tmp[j];
        }
        printf("Sorted array:\n");
        for (int i = 0; i < n; i++)
            printf("%d ", data[i]);
        printf("\n");

        end_time = MPI_Wtime();  // End timing

        // Execution time in microseconds
        printf("\nExecution Time: %.0f microseconds\n", (end_time - start_time) * 1000000);

        free(tmp);
        free(data);
    }

    free(sub_data);
    MPI_Finalize();
    return 0;
}
