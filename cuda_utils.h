#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H
#include <GL/gl.h>

void cuda_init();
void *cuda_register_buffer(GLuint buf);
void cuda_unregister_resource(void *res);
void *cuda_map_resource(void *res);
void cuda_unmap_resource(void *res);

#endif // CUDA_UTILS_H
