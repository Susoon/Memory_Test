#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/time.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#define true 1
#define false 0

#define GLOBAL 1
#define BATCH 0

#define THREAD_NUM 512
#define BLOCK_NUM 8
#define TERM 96
#define TOKEN 32
#define AREA_SIZE (1024*THREAD_NUM)

#define LOOP 11

uint64_t monotonic_time() {
    struct timespec timespec;
    clock_gettime(CLOCK_MONOTONIC, &timespec);
    return timespec.tv_sec * 1000 * 1000 * 1000 + timespec.tv_nsec;
}

__global__ void global_to_global(uint8_t *area1, uint8_t *area2, uint32_t knum, uint64_t *time, uint8_t *all_kernel_set)
{
    uint32_t area1_offset = knum * THREAD_NUM + blockIdx.x * THREAD_NUM + threadIdx.x * TERM;
    uint32_t area2_offset = knum * THREAD_NUM + blockIdx.x * THREAD_NUM + threadIdx.x * TOKEN;

    uint64_t start = 0;
    uint64_t end = 0;

    __shared__ uint8_t tmp_area[TOKEN * THREAD_NUM];

    for(int i = area2_offset; i < area2_offset + THREAD_NUM; i++){
        tmp_area[i] = 0;
    }

    //while(*all_kernel_set == false);
    
    if(threadIdx.x == 0)
        start = clock64();
    //asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    memcpy(area2 + area2_offset, area1 + area1_offset, TOKEN);
    //asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if(threadIdx.x == 0)
        end = clock64();

    if(threadIdx.x == 0)
        time[knum + blockIdx.x] = end - start;
}

__global__ void shared_to_global(uint8_t *area1, uint8_t *area2, uint32_t knum, uint64_t *time, uint8_t *all_kernel_set)
{
    uint32_t area1_offset = knum * THREAD_NUM + blockIdx.x * THREAD_NUM + threadIdx.x * TERM;
    uint32_t area2_offset = knum * THREAD_NUM + blockIdx.x * THREAD_NUM + threadIdx.x * TOKEN;

    uint64_t start = 0;
    uint64_t end = 0;

    //while(*all_kernel_set == false);

    __shared__ uint8_t tmp_area[TOKEN * THREAD_NUM];

    for(int i = area2_offset; i < area2_offset + THREAD_NUM; i++){
        tmp_area[i] = 0;
    }

    memcpy(tmp_area + area2_offset, area1 + area1_offset, TOKEN);

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    //start = clock64();
#if BATCH
    __syncthreads();
    if(threadIdx.x == THREAD_NUM - 1){
        memcpy(area2 + knum * THREAD_NUM + blockIdx.x * THREAD_NUM, tmp_area, TOKEN * THREAD_NUM);
    }
#else
    memcpy(area2 + area2_offset, tmp_area + area2_offset, TOKEN);
#endif
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    //end = clock64();

    time[knum * THREAD_NUM + blockIdx.x * THREAD_NUM + threadIdx.x] = end - start;
}

int main(void)
{
    uint8_t *h_area1;
    uint8_t *d_area1;
    uint8_t *d_area2;
    uint8_t *all_kernel_set;

    uint64_t start = 0;
    uint64_t end = 0;
    uint64_t mid1 = 0;
    uint64_t times[LOOP + 1] = {0};
    uint64_t *gpu_times;
    uint64_t *tmp_times;

    h_area1 = (uint8_t *)malloc(AREA_SIZE);

    cudaMalloc((void**)&d_area1, AREA_SIZE);
    cudaMemset(d_area1, 0, AREA_SIZE);

    cudaMalloc((void**)&d_area2, AREA_SIZE);
    cudaMemset(d_area2, 0, AREA_SIZE);

    cudaMalloc((void**)&all_kernel_set, sizeof(uint8_t));
    cudaMemset(all_kernel_set, 0, sizeof(uint8_t));

    cudaMalloc((void**)&gpu_times, BLOCK_NUM * sizeof(uint64_t));
    cudaMemset(gpu_times, 0, BLOCK_NUM * sizeof(uint64_t));

    tmp_times = (uint64_t *)malloc(BLOCK_NUM * sizeof(uint64_t));

    cudaStream_t *cuda_streams;
    cuda_streams = (cudaStream_t *)malloc(sizeof(cudaStream_t) * BLOCK_NUM);

    for(int i = 0; i < BLOCK_NUM; i++)
        cudaStreamCreateWithFlags(&(cuda_streams[i]), cudaStreamNonBlocking);

    for(int i = 0; i < LOOP; i++){
        srand(1234 + i);

        for(int j = 0; j < AREA_SIZE; j++){
            h_area1[j] = 0 + rand() % 255;
        }

        cudaMemcpy(d_area1, h_area1, AREA_SIZE, cudaMemcpyHostToDevice);
        sleep(1);

        start = monotonic_time();

        for(int j = 0; j < BLOCK_NUM; j++){
#if GLOBAL
            global_to_global <<< 1, THREAD_NUM, 0, cuda_streams[j] >>> (d_area1, d_area2, j, gpu_times, all_kernel_set);
#else
            shared_to_global <<< 1, THREAD_NUM, 0, cuda_streams[j] >>> (d_area1, d_area2, j, gpu_times, all_kernel_set);
#endif
        }
        cudaMemset(all_kernel_set, 1, sizeof(uint8_t));
/*
#if GLOBAL
        global_to_global <<< BLOCK_NUM, THREAD_NUM, 0, cuda_streams[0] >>> (d_area1, d_area2, 0, gpu_times, all_kernel_set);
#else
        shared_to_global <<< BLOCK_NUM, THREAD_NUM, 0, cuda_streams[0] >>> (d_area1, d_area2, 0, gpu_times, all_kernel_set);
#endif
*/
        mid1 = monotonic_time();
        cudaDeviceSynchronize();
        end = monotonic_time();

        cudaMemcpy(tmp_times, gpu_times, sizeof(uint64_t) * BLOCK_NUM, cudaMemcpyDeviceToHost);
  
        uint64_t tmp = 0;
        for(int j = 0; j < BLOCK_NUM; j++){
            tmp += tmp_times[j];
        }  

        times[i] = tmp_times[0] / BLOCK_NUM;
/*
        times[i] = end - start;
*/
        times[LOOP] += times[i];

        cudaMemset(d_area2, 0, AREA_SIZE);
        cudaMemset(all_kernel_set, 0, sizeof(uint8_t));
        sleep(1);
        //printf("%lu %lu\n", end-mid1, mid1-start);
    }

    for(int i = 0; i < LOOP; i++){
        printf("%dth Loop : %lu\n", i, times[i]);
    }

    printf("Avg : %lfns elapsed!\n", (double)times[LOOP] / LOOP);
}
