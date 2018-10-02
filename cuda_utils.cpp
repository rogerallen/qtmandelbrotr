#include <stdio.h>
#include <cstring>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include "cuda_utils.h"

// cuda needs to be initialized after opengl
void cuda_init()
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

// register buffer GLuint and return as a cudaGraphicsResource
void *cuda_register_buffer(GLuint buf)
{
    cudaGraphicsResource *res = nullptr;
    if (cudaGraphicsGLRegisterBuffer(&res, buf, cudaGraphicsRegisterFlagsNone) != cudaSuccess)
        printf("Failed to register buffer %u\n", buf);
    return res;
}

void cuda_unregister_resource(void *res)
{
    if (cudaGraphicsUnregisterResource(static_cast<cudaGraphicsResource *>(res)) != cudaSuccess)
        puts("Failed to unregister resource for buffer");
}

void *cuda_map_resource(void *res)
{
    cudaGraphicsResource *ptr = static_cast<cudaGraphicsResource *>(res);
    if (cudaGraphicsMapResources(1, &ptr) != cudaSuccess) {
        puts("Failed to map resource");
        return nullptr;
    }
    void *devPtr = nullptr;
    size_t size;
    if (cudaGraphicsResourceGetMappedPointer(&devPtr, &size, static_cast<cudaGraphicsResource *>(res)) != cudaSuccess) {
        puts("Failed to get device pointer");
        return nullptr;
    }
    return devPtr;
}

void cuda_unmap_resource(void *res)
{
    cudaGraphicsResource *ptr = static_cast<cudaGraphicsResource *>(res);
    if (cudaGraphicsUnmapResources(1, &ptr) != cudaSuccess)
        puts("Failed to unmap resource");
}
