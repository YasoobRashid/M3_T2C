#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <CL/cl.h>

#define MAX_SOURCE_SIZE (0x10000)
#define ARRAY_SIZE 16

void checkError(cl_int err, const char* msg) {
    if (err != CL_SUCCESS) {
        printf("Error: %s (%d)\n", msg, err);
        exit(1);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int *data = NULL;
    int chunk_size = ARRAY_SIZE / size;
    int *sub_data = (int*)malloc(chunk_size * sizeof(int));

    if (rank == 0) {
        data = (int*)malloc(ARRAY_SIZE * sizeof(int));
        printf("Unsorted array: ");
        for (int i = 0; i < ARRAY_SIZE; i++) {
            data[i] = rand() % 100;
            printf("%d ", data[i]);
        }
        printf("\n");
    }

    MPI_Scatter(data, chunk_size, MPI_INT, sub_data, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    // OpenCL setup
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_context context = NULL;
    cl_command_queue queue = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_mem buffer = NULL;
    cl_int err;

    err = clGetPlatformIDs(1, &platform_id, NULL);
    checkError(err, "Getting platform IDs");
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    checkError(err, "Getting device IDs");
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    checkError(err, "Creating context");
    queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
    checkError(err, "Creating command queue");

    FILE* fp = fopen("quicksort.cl", "r");
    if (!fp) {
        printf("Failed to load kernel.\n");
        exit(1);
    }
    char* source_str = (char*)malloc(MAX_SOURCE_SIZE);
    size_t source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    program = clCreateProgramWithSource(context, 1, (const char**)&source_str, &source_size, &err);
    checkError(err, "Creating program");
    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char*) malloc(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("%s\n", log);
        free(log);
        exit(1);
    }

    kernel = clCreateKernel(program, "quicksort", &err);
    checkError(err, "Creating kernel");

    buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, chunk_size * sizeof(int), NULL, &err);
    checkError(err, "Creating buffer");
    err = clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0, chunk_size * sizeof(int), sub_data, 0, NULL, NULL);
    checkError(err, "Copying data to buffer");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(int), &chunk_size);
    checkError(err, "Setting kernel arguments");

    size_t global_item_size = 1; // only one QuickSort per subarray
    size_t local_item_size = 1;

    double start = MPI_Wtime();
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
    checkError(err, "Running kernel");
    clFinish(queue);
    double end = MPI_Wtime();

    err = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, chunk_size * sizeof(int), sub_data, 0, NULL, NULL);
    checkError(err, "Reading back data");

    // Cleanup OpenCL
    clReleaseMemObject(buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(source_str);

    MPI_Gather(sub_data, chunk_size, MPI_INT, data, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Merge sorted subarrays
        int* temp = (int*)malloc(ARRAY_SIZE * sizeof(int));
        for (int i = 0; i < ARRAY_SIZE; i++) temp[i] = data[i];

        for (int i = 1; i < size; i++) {
            int* merged = (int*)malloc((i+1)*chunk_size * sizeof(int));
            int p = 0, q = 0, r = 0;
            while (p < i*chunk_size && q < chunk_size) {
                if (temp[p] < data[i*chunk_size + q])
                    merged[r++] = temp[p++];
                else
                    merged[r++] = data[i*chunk_size + q++];
            }
            while (p < i*chunk_size)
                merged[r++] = temp[p++];
            while (q < chunk_size)
                merged[r++] = data[i*chunk_size + q++];
            for (int j = 0; j < (i+1)*chunk_size; j++)
                temp[j] = merged[j];
            free(merged);
        }

        printf("Sorted array: ");
        for (int i = 0; i < ARRAY_SIZE; i++)
            printf("%d ", temp[i]);
        printf("\n");

        printf("Execution time: %.0f seconds\n");

        free(temp);
        free(data);
    }

    free(sub_data);
    MPI_Finalize();
    return 0;
}
