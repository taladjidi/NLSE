/**
 * Metal compute wrapper for NLSE solver.
 *
 * Provides a C-level API for:
 *  - Metal device/queue management
 *  - Shared-memory buffer allocation (zero-copy with numpy)
 *  - Compute kernel dispatch
 *
 * Compiled as: clang -O2 -framework Metal -framework Foundation
 *              -shared -o libmetal_nlse.dylib metal_wrapper.m
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <string.h>
#include <stdio.h>

// ---- Opaque context handle ----
typedef struct {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;

    // Pre-compiled pipeline states for each kernel
    id<MTLComputePipelineState> pso_square_mod;
    id<MTLComputePipelineState> pso_nl_prop;
    id<MTLComputePipelineState> pso_nl_prop_without_V;
    id<MTLComputePipelineState> pso_nl_prop_c;
    id<MTLComputePipelineState> pso_nl_prop_without_V_c;
    id<MTLComputePipelineState> pso_rabi_coupling_A1;
    id<MTLComputePipelineState> pso_rabi_coupling_A2;
    id<MTLComputePipelineState> pso_vortex;
    id<MTLComputePipelineState> pso_complex_multiply_inplace;
} MetalCtx;

// ---- Buffer handle ----
typedef struct {
    id<MTLBuffer> buffer;
    size_t size;
} MetalBuf;


// ---- Helper: create pipeline state from function name ----
static id<MTLComputePipelineState> _make_pso(MetalCtx *ctx, const char *name) {
    NSString *nsname = [NSString stringWithUTF8String:name];
    id<MTLFunction> func = [ctx->library newFunctionWithName:nsname];
    if (!func) {
        fprintf(stderr, "Metal: function '%s' not found in library\n", name);
        return nil;
    }
    NSError *error = nil;
    id<MTLComputePipelineState> pso =
        [ctx->device newComputePipelineStateWithFunction:func error:&error];
    if (!pso) {
        fprintf(stderr, "Metal: failed to create PSO for '%s': %s\n",
                name, [[error localizedDescription] UTF8String]);
        return nil;
    }
    return pso;
}


// ---- Helper: dispatch a compute kernel ----
static void _dispatch(MetalCtx *ctx, id<MTLComputePipelineState> pso,
                      id<MTLCommandBuffer> cmdBuf, uint32_t count,
                      void (^encode)(id<MTLComputeCommandEncoder>)) {
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    [enc setComputePipelineState:pso];
    encode(enc);
    NSUInteger threadGroupSize = pso.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > count) threadGroupSize = count;
    MTLSize grid = MTLSizeMake(count, 1, 1);
    MTLSize tg = MTLSizeMake(threadGroupSize, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
}


// ============================================================
// Public API
// ============================================================

MetalCtx* metal_init(const char *shader_source) {
    MetalCtx *ctx = (MetalCtx *)calloc(1, sizeof(MetalCtx));
    if (!ctx) return NULL;

    ctx->device = MTLCreateSystemDefaultDevice();
    if (!ctx->device) {
        fprintf(stderr, "Metal: no GPU device found\n");
        free(ctx);
        return NULL;
    }

    ctx->queue = [ctx->device newCommandQueue];

    // Compile shader source at runtime
    NSString *src = [NSString stringWithUTF8String:shader_source];
    NSError *error = nil;
    MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
    opts.fastMathEnabled = YES;
    ctx->library = [ctx->device newLibraryWithSource:src options:opts error:&error];
    if (!ctx->library) {
        fprintf(stderr, "Metal: shader compilation failed: %s\n",
                [[error localizedDescription] UTF8String]);
        free(ctx);
        return NULL;
    }

    // Pre-compile all pipeline states
    ctx->pso_square_mod = _make_pso(ctx, "square_mod");
    ctx->pso_nl_prop = _make_pso(ctx, "nl_prop");
    ctx->pso_nl_prop_without_V = _make_pso(ctx, "nl_prop_without_V");
    ctx->pso_nl_prop_c = _make_pso(ctx, "nl_prop_c");
    ctx->pso_nl_prop_without_V_c = _make_pso(ctx, "nl_prop_without_V_c");
    ctx->pso_rabi_coupling_A1 = _make_pso(ctx, "rabi_coupling_A1");
    ctx->pso_rabi_coupling_A2 = _make_pso(ctx, "rabi_coupling_A2");
    ctx->pso_vortex = _make_pso(ctx, "vortex");
    ctx->pso_complex_multiply_inplace = _make_pso(ctx, "complex_multiply_inplace");

    return ctx;
}

void metal_free(MetalCtx *ctx) {
    if (ctx) free(ctx);
}

const char* metal_device_name(MetalCtx *ctx) {
    return [[ctx->device name] UTF8String];
}

// ---- Buffer management ----

MetalBuf* metal_buf_alloc(MetalCtx *ctx, size_t nbytes) {
    MetalBuf *buf = (MetalBuf *)calloc(1, sizeof(MetalBuf));
    if (!buf) return NULL;
    buf->buffer = [ctx->device newBufferWithLength:nbytes
                                           options:MTLResourceStorageModeShared];
    buf->size = nbytes;
    return buf;
}

MetalBuf* metal_buf_from_ptr(MetalCtx *ctx, void *ptr, size_t nbytes) {
    MetalBuf *buf = (MetalBuf *)calloc(1, sizeof(MetalBuf));
    if (!buf) return NULL;
    // Create buffer wrapping existing memory (zero-copy)
    buf->buffer = [ctx->device newBufferWithBytesNoCopy:ptr
                                                 length:nbytes
                                                options:MTLResourceStorageModeShared
                                            deallocator:nil];
    if (!buf->buffer) {
        // Fallback: allocate and copy (page alignment requirement)
        buf->buffer = [ctx->device newBufferWithBytes:ptr
                                               length:nbytes
                                              options:MTLResourceStorageModeShared];
    }
    buf->size = nbytes;
    return buf;
}

void* metal_buf_ptr(MetalBuf *buf) {
    return [buf->buffer contents];
}

size_t metal_buf_size(MetalBuf *buf) {
    return buf->size;
}

void metal_buf_free(MetalBuf *buf) {
    if (buf) free(buf);
}

// Copy data between numpy and Metal buffer
void metal_buf_copy_from(MetalBuf *buf, const void *src, size_t nbytes) {
    memcpy([buf->buffer contents], src, nbytes);
}

void metal_buf_copy_to(MetalBuf *buf, void *dst, size_t nbytes) {
    memcpy(dst, [buf->buffer contents], nbytes);
}


// ============================================================
// Kernel dispatch functions
// ============================================================

void metal_square_mod(MetalCtx *ctx, MetalBuf *A, MetalBuf *A_sq, uint32_t count) {
    id<MTLCommandBuffer> cmdBuf = [ctx->queue commandBuffer];
    _dispatch(ctx, ctx->pso_square_mod, cmdBuf, count, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:A->buffer offset:0 atIndex:0];
        [enc setBuffer:A_sq->buffer offset:0 atIndex:1];
    });
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
}

void metal_nl_prop(MetalCtx *ctx, MetalBuf *A, MetalBuf *A_sq, MetalBuf *V,
                   float dz, float alpha, float g, float Isat, uint32_t count) {
    id<MTLCommandBuffer> cmdBuf = [ctx->queue commandBuffer];
    _dispatch(ctx, ctx->pso_nl_prop, cmdBuf, count, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:A->buffer offset:0 atIndex:0];
        [enc setBuffer:A_sq->buffer offset:0 atIndex:1];
        [enc setBuffer:V->buffer offset:0 atIndex:2];
        [enc setBytes:&dz length:sizeof(float) atIndex:3];
        [enc setBytes:&alpha length:sizeof(float) atIndex:4];
        [enc setBytes:&g length:sizeof(float) atIndex:5];
        [enc setBytes:&Isat length:sizeof(float) atIndex:6];
    });
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
}

void metal_nl_prop_without_V(MetalCtx *ctx, MetalBuf *A, MetalBuf *A_sq,
                              float dz, float alpha, float g, float Isat,
                              uint32_t count) {
    id<MTLCommandBuffer> cmdBuf = [ctx->queue commandBuffer];
    _dispatch(ctx, ctx->pso_nl_prop_without_V, cmdBuf, count, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:A->buffer offset:0 atIndex:0];
        [enc setBuffer:A_sq->buffer offset:0 atIndex:1];
        [enc setBytes:&dz length:sizeof(float) atIndex:2];
        [enc setBytes:&alpha length:sizeof(float) atIndex:3];
        [enc setBytes:&g length:sizeof(float) atIndex:4];
        [enc setBytes:&Isat length:sizeof(float) atIndex:5];
    });
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
}

void metal_nl_prop_c(MetalCtx *ctx, MetalBuf *A1, MetalBuf *A_sq_1,
                      MetalBuf *A_sq_2, MetalBuf *V,
                      float dz, float alpha, float g11, float g12,
                      float Isat1, float Isat2, uint32_t count) {
    id<MTLCommandBuffer> cmdBuf = [ctx->queue commandBuffer];
    _dispatch(ctx, ctx->pso_nl_prop_c, cmdBuf, count, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:A1->buffer offset:0 atIndex:0];
        [enc setBuffer:A_sq_1->buffer offset:0 atIndex:1];
        [enc setBuffer:A_sq_2->buffer offset:0 atIndex:2];
        [enc setBuffer:V->buffer offset:0 atIndex:3];
        [enc setBytes:&dz length:sizeof(float) atIndex:4];
        [enc setBytes:&alpha length:sizeof(float) atIndex:5];
        [enc setBytes:&g11 length:sizeof(float) atIndex:6];
        [enc setBytes:&g12 length:sizeof(float) atIndex:7];
        [enc setBytes:&Isat1 length:sizeof(float) atIndex:8];
        [enc setBytes:&Isat2 length:sizeof(float) atIndex:9];
    });
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
}

void metal_nl_prop_without_V_c(MetalCtx *ctx, MetalBuf *A1, MetalBuf *A_sq_1,
                                MetalBuf *A_sq_2,
                                float dz, float alpha, float g11, float g12,
                                float Isat1, float Isat2, uint32_t count) {
    id<MTLCommandBuffer> cmdBuf = [ctx->queue commandBuffer];
    _dispatch(ctx, ctx->pso_nl_prop_without_V_c, cmdBuf, count, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:A1->buffer offset:0 atIndex:0];
        [enc setBuffer:A_sq_1->buffer offset:0 atIndex:1];
        [enc setBuffer:A_sq_2->buffer offset:0 atIndex:2];
        [enc setBytes:&dz length:sizeof(float) atIndex:3];
        [enc setBytes:&alpha length:sizeof(float) atIndex:4];
        [enc setBytes:&g11 length:sizeof(float) atIndex:5];
        [enc setBytes:&g12 length:sizeof(float) atIndex:6];
        [enc setBytes:&Isat1 length:sizeof(float) atIndex:7];
        [enc setBytes:&Isat2 length:sizeof(float) atIndex:8];
    });
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
}

void metal_rabi_coupling(MetalCtx *ctx, MetalBuf *A1, MetalBuf *A2,
                          MetalBuf *A1_scratch,
                          float dz, float omega, uint32_t count) {
    float cos_val = cosf(omega * dz);
    float sin_val = sinf(omega * dz);
    id<MTLCommandBuffer> cmdBuf = [ctx->queue commandBuffer];
    // Pass 1: save A1_old and update A1
    _dispatch(ctx, ctx->pso_rabi_coupling_A1, cmdBuf, count, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:A1->buffer offset:0 atIndex:0];
        [enc setBuffer:A2->buffer offset:0 atIndex:1];
        [enc setBuffer:A1_scratch->buffer offset:0 atIndex:2];
        [enc setBytes:&cos_val length:sizeof(float) atIndex:3];
        [enc setBytes:&sin_val length:sizeof(float) atIndex:4];
    });
    // Pass 2: update A2 using A1_old
    _dispatch(ctx, ctx->pso_rabi_coupling_A2, cmdBuf, count, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:A2->buffer offset:0 atIndex:0];
        [enc setBuffer:A1_scratch->buffer offset:0 atIndex:1];
        [enc setBytes:&cos_val length:sizeof(float) atIndex:2];
        [enc setBytes:&sin_val length:sizeof(float) atIndex:3];
    });
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
}

void metal_vortex(MetalCtx *ctx, MetalBuf *im, MetalBuf *ii, MetalBuf *jj,
                   float i_pos, float j_pos, int ll, uint32_t count) {
    id<MTLCommandBuffer> cmdBuf = [ctx->queue commandBuffer];
    _dispatch(ctx, ctx->pso_vortex, cmdBuf, count, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:im->buffer offset:0 atIndex:0];
        [enc setBuffer:ii->buffer offset:0 atIndex:1];
        [enc setBuffer:jj->buffer offset:0 atIndex:2];
        [enc setBytes:&i_pos length:sizeof(float) atIndex:3];
        [enc setBytes:&j_pos length:sizeof(float) atIndex:4];
        [enc setBytes:&ll length:sizeof(int) atIndex:5];
    });
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
}

void metal_complex_multiply_inplace(MetalCtx *ctx, MetalBuf *A, MetalBuf *B,
                                     uint32_t count) {
    id<MTLCommandBuffer> cmdBuf = [ctx->queue commandBuffer];
    _dispatch(ctx, ctx->pso_complex_multiply_inplace, cmdBuf, count, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:A->buffer offset:0 atIndex:0];
        [enc setBuffer:B->buffer offset:0 atIndex:1];
    });
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
}
