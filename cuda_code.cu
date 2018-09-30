/* CUDA code goes  here */
#include <stdio.h>
#include <GL/gl.h>
#include <cuda.h>
#include <cuda_gl_interop.h>

// cuda needs to be initialized after opengl
void CUDA_init()
{
    cudaDeviceProp prop;
    int dev;
    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 6;
    prop.minor = 0;
    if (cudaChooseDevice(&dev, &prop) != cudaSuccess)
        puts("failed to choose device");
    printf("cuda chose device %d\n",dev);
    if (cudaGLSetGLDevice(dev) != cudaSuccess)
        puts("failed to set gl device");
}

void *CUDA_registerBuffer(GLuint buf)
{
    cudaGraphicsResource *res = 0;
    if (cudaGraphicsGLRegisterBuffer(&res, buf, cudaGraphicsRegisterFlagsNone) != cudaSuccess)
        printf("Failed to register buffer %u\n", buf);
    return res;
}

void CUDA_unregisterBuffer(void *res)
{
    if (cudaGraphicsUnregisterResource((cudaGraphicsResource *) res) != cudaSuccess)
        puts("Failed to unregister resource for buffer");
}

void *CUDA_map(void *res)
{
    if (cudaGraphicsMapResources(1, (cudaGraphicsResource **) &res) != cudaSuccess) {
        puts("Failed to map resource");
        return 0;
    }
    void *devPtr = 0;
    size_t size;
    if (cudaGraphicsResourceGetMappedPointer(&devPtr, &size, (cudaGraphicsResource *) res) != cudaSuccess) {
        puts("Failed to get device pointer");
        return 0;
    }
    return devPtr;
}

void CUDA_unmap(void *res)
{
    if (cudaGraphicsUnmapResources(1,(cudaGraphicsResource **) &res) != cudaSuccess)
        puts("Failed to unmap resource");
}

__global__ void run(uchar4 *ptr, int w, int h, float t)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    if(x < w && y < h) {
        uchar4 bgra = {0x0,0x0,0x0,0x0};
        bgra.y = (unsigned char)(t + 255.99f*y/(float)h);
        bgra.z = (unsigned char)(t + 255.99f*x/(float)w);
        *(ptr + offset) = bgra;
    }
}

void CUDA_do_something(void *devPtr, int w, int h, float t)
{
    const int blockSize = 16; // 256 threads per block
    run<<<dim3(w / blockSize, h / blockSize), dim3(blockSize, blockSize)>>>((uchar4 *) devPtr, w, h, t);
}
