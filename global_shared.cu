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

#define GLOBAL 0
#define BATCH 1

#define THREAD_NUM 512
#define BLOCK_NUM 8
#define TERM 64
#define TOKEN 32
#define AREA_SIZE (1024*THREAD_NUM)

#define LOOP 10

#define bTos(b, s) do{                            \
        uint16_t tmp1 = (uint16_t)*(b);           \
        uint16_t tmp2 = (uint16_t)*((b) + 1);     \
        (s) = (tmp2) + (tmp1 << 8);               \
    }while(0)

#define bToi(b, i) do{                                                \
        uint32_t tmp1 = (uint32_t)*((b)    );                         \
        uint32_t tmp2 = (uint32_t)*((b) + 1);                         \
        uint32_t tmp3 = (uint32_t)*((b) + 2);                         \
        uint32_t tmp4 = (uint32_t)*((b) + 3);                         \
        (i) = (tmp4) + (tmp3 << 8) + (tmp2 << 16) + (tmp1 << 24);     \
    }while(0)

#define bTol(b, l) do{                                  \
        uint64_t tmp1 = (uint64_t)*((b)    );           \
        uint64_t tmp2 = (uint64_t)*((b) + 1);           \
        uint64_t tmp3 = (uint64_t)*((b) + 2);           \
        uint64_t tmp4 = (uint64_t)*((b) + 3);           \
        uint64_t tmp5 = (uint64_t)*((b) + 4);           \
        uint64_t tmp6 = (uint64_t)*((b) + 5);           \
        uint64_t tmp7 = (uint64_t)*((b) + 6);           \
        uint64_t tmp8 = (uint64_t)*((b) + 7);           \
        (l) = (tmp8) + (tmp7 << 8) + (tmp6 << 16) + (tmp5 << 24)             \
                + (tmp4 << 32) + (tmp3 << 40) + (tmp2 << 48) + (tmp1 << 56); \
    }while(0)


uint64_t monotonic_time() {
    struct timespec timespec;
    clock_gettime(CLOCK_MONOTONIC, &timespec);
    return timespec.tv_sec * 1000 * 1000 * 1000 + timespec.tv_nsec;
}

__global__ void global_to_global(uint8_t *area1, uint8_t *area2, uint32_t knum, uint64_t *time, uint8_t *all_kernel_set)
{
    uint32_t area1_offset = knum * THREAD_NUM + blockIdx.x * THREAD_NUM + threadIdx.x * TERM;
    uint32_t area2_offset = knum * THREAD_NUM + blockIdx.x * THREAD_NUM + threadIdx.x * TOKEN;

    uint8_t *my_area;

    uint64_t start = 0;
    uint64_t end = 0;

    uint16_t stmp = 0;
    uint32_t itmp = 0;
    uint64_t ltmp = 0;
    uint32_t dummy_hash = 0;
    uint32_t dummy_sign = 0;

    uint8_t *dummy_bucket = NULL;

    __shared__ uint8_t tmp_area[TOKEN * THREAD_NUM];

    for(int i = threadIdx.x * TOKEN; i < threadIdx.x * TOKEN + TOKEN; i++){
        tmp_area[i] = 0;
    }

    //while(*all_kernel_set == false);
    
    if(threadIdx.x == 0)
        start = clock64();
    
    my_area = area1 + area1_offset;

    bTos(my_area, stmp);
    my_area += sizeof(uint16_t);

    bToi(my_area, itmp);
    my_area += sizeof(uint32_t);
    
    bTol(my_area, ltmp);
    my_area += sizeof(uint64_t);

    dummy_hash = (uint32_t)(ltmp >> 32);
    dummy_sign = (uint32_t)((ltmp << 32) >> 32);

    dummy_bucket = &(area1[dummy_hash & (AREA_SIZE - 1)]);

    for(int i = 0; i < 8; i++){
        itmp = dummy_bucket[i];
        if(i & 1){
            dummy_bucket[i] = dummy_sign;
        }
    }

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

    uint8_t *my_area;

    uint64_t start = 0;
    uint64_t end = 0;

    uint16_t stmp = 0;
    uint32_t itmp = 0;
    uint64_t ltmp = 0;
    uint32_t dummy_hash = 0;
    uint32_t dummy_sign = 0;

    uint8_t *dummy_bucket = NULL;

    //while(*all_kernel_set == false);

    __shared__ uint8_t tmp_area[TOKEN * THREAD_NUM];
    __shared__ uint8_t tmp_ht[8 * THREAD_NUM];

    for(int i = threadIdx.x * TOKEN; i < threadIdx.x * TOKEN + TOKEN; i++){
        tmp_area[i] = 0;
    }

    if(threadIdx.x == THREAD_NUM - 1)
        start = clock64();

    memcpy(tmp_area + threadIdx.x * TOKEN, area1 + area1_offset, TOKEN);

    my_area = tmp_area + threadIdx.x * TOKEN;

    bTos(my_area, stmp);
    my_area += sizeof(uint16_t);

    bToi(my_area, itmp);
    my_area += sizeof(uint32_t);
    
    bTol(my_area, ltmp);
    my_area += sizeof(uint64_t);

    dummy_hash = (uint32_t)(ltmp >> 32);
    dummy_sign = (uint32_t)((ltmp << 32) >> 32);

    memcpy(tmp_ht + threadIdx.x * 8, &(area1[dummy_hash & (AREA_SIZE - 1)]), sizeof(uint8_t) * 8);

    dummy_bucket = &(tmp_ht[threadIdx.x * 8]);

    for(int i = 0; i < 8; i++){
        itmp = dummy_bucket[i];
        if(i & 1){
            dummy_bucket[i] = dummy_sign;
        }
    }

#if BATCH
    __syncthreads();
    if(threadIdx.x == THREAD_NUM - 1){
        memcpy(area2 + knum * THREAD_NUM + blockIdx.x * THREAD_NUM, tmp_area, TOKEN * (THREAD_NUM));
    }
    __syncthreads();
#else
    memcpy(area2 + area2_offset, tmp_area + threadIdx.x * TOKEN, TOKEN);
#endif
    if(threadIdx.x == THREAD_NUM - 1)
        end = clock64();

    if(threadIdx.x == THREAD_NUM - 1)
        time[knum + blockIdx.x] = end - start;
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
    uint64_t cpu_times[LOOP + 1] = {0};
    uint64_t *d_gpu_times;
    uint64_t gpu_times[LOOP + 1] = {0};
    uint64_t *tmp_times;

    h_area1 = (uint8_t *)malloc(AREA_SIZE);

    cudaMalloc((void**)&d_area1, AREA_SIZE);
    cudaMemset(d_area1, 0, AREA_SIZE);

    cudaMalloc((void**)&d_area2, AREA_SIZE);
    cudaMemset(d_area2, 0, AREA_SIZE);

    cudaMalloc((void**)&all_kernel_set, sizeof(uint8_t));
    cudaMemset(all_kernel_set, 0, sizeof(uint8_t));

    cudaMalloc((void**)&d_gpu_times, BLOCK_NUM * sizeof(uint64_t));
    cudaMemset(d_gpu_times, 0, BLOCK_NUM * sizeof(uint64_t));

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
            global_to_global <<< 1, THREAD_NUM, 0, cuda_streams[j] >>> (d_area1, d_area2, j, d_gpu_times, all_kernel_set);
#else
            shared_to_global <<< 1, THREAD_NUM, 0, cuda_streams[j] >>> (d_area1, d_area2, j, d_gpu_times, all_kernel_set);
#endif
        }
        cudaMemset(all_kernel_set, 1, sizeof(uint8_t));
/*
#if GLOBAL
        global_to_global <<< BLOCK_NUM, THREAD_NUM, 0, cuda_streams[0] >>> (d_area1, d_area2, 0, d_gpu_times, all_kernel_set);
#else
        shared_to_global <<< BLOCK_NUM, THREAD_NUM, 0, cuda_streams[0] >>> (d_area1, d_area2, 0, d_gpu_times, all_kernel_set);
#endif
*/
        mid1 = monotonic_time();
        cudaDeviceSynchronize();
        end = monotonic_time();

        cudaMemcpy(tmp_times, d_gpu_times, sizeof(uint64_t) * BLOCK_NUM, cudaMemcpyDeviceToHost);
  
        for(int j = 0; j < BLOCK_NUM; j++){
            gpu_times[i] += tmp_times[j];
        }  

        gpu_times[i] /= BLOCK_NUM;
        gpu_times[LOOP] += gpu_times[i];

        cpu_times[i] = end - start;
        cpu_times[LOOP] += cpu_times[i];

        cudaMemset(d_area2, 0, AREA_SIZE);
        cudaMemset(all_kernel_set, 0, sizeof(uint8_t));
        sleep(1);
        //printf("%lu %lu\n", end-mid1, mid1-start);
    }

    for(int i = 0; i < LOOP; i++){
        printf("%dth Loop CPU TIME : %lu GPU TIME : %lu\n", i, cpu_times[i], gpu_times[i]);
    }

    //printf("Avg : %lfns elapsed!\n", (double)cpu_times[LOOP] / LOOP);
}
