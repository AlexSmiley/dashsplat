#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <vector>
struct Mesh {
  id<MTLBuffer> vb, ib;
  id<MTLTexture> photo, labels;
  int n;
};
struct Call {
  Mesh *m;
  float u[28];
};
struct Scene {
  id<MTLDevice> dev;
  id<MTLCommandQueue> q;
  id<MTLRenderPipelineState> first, second;
  id<MTLDepthStencilState> less;
  id<MTLTexture> depth, win, sum;
  int w, h;
  std::vector<Call> jobs;
};
static id<MTLTexture> tex(Scene *s, int w, int h, MTLPixelFormat pf,
                          MTLTextureUsage usage, bool priv = false) {
  auto d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pf
                                                              width:w
                                                             height:h
                                                          mipmapped:NO];
  d.storageMode = priv ? MTLStorageModePrivate : MTLStorageModeShared;
  d.usage = usage;
  return [s->dev newTextureWithDescriptor:d];
}
extern "C" void *ibr_init(const char *src, int w, int h) {
  @autoreleasepool {
    Scene *s = new Scene();
    s->w = w;
    s->h = h;
    s->dev = MTLCreateSystemDefaultDevice();
    s->q = [s->dev newCommandQueue];
    NSError *err = nil;
    auto lib = [s->dev newLibraryWithSource:[NSString stringWithUTF8String:src]
                                    options:nil
                                      error:&err];
    if (!lib) {
      NSLog(@"%@", err);
      return nullptr;
    }
    for (int p = 0; p < 2; p++) {
      auto d = [MTLRenderPipelineDescriptor new];
      d.vertexFunction = [lib newFunctionWithName:@"vert"];
      d.fragmentFunction = [lib newFunctionWithName:p ? @"second" : @"first"];
      auto c = d.colorAttachments[0];
      c.pixelFormat = p ? MTLPixelFormatRGBA16Float : MTLPixelFormatRGBA32Float;
      if (!p)
        d.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
      else {
        c.blendingEnabled = YES;
        c.sourceRGBBlendFactor = MTLBlendFactorOne;
        c.destinationRGBBlendFactor = MTLBlendFactorOne;
        c.sourceAlphaBlendFactor = MTLBlendFactorOne;
        c.destinationAlphaBlendFactor = MTLBlendFactorOne;
      }
      auto ps = [s->dev newRenderPipelineStateWithDescriptor:d error:&err];
      if (!ps) {
        NSLog(@"%@", err);
        return nullptr;
      }
      if (p)
        s->second = ps;
      else
        s->first = ps;
    }
    auto d = [MTLDepthStencilDescriptor new];
    d.depthCompareFunction = MTLCompareFunctionLess;
    d.depthWriteEnabled = YES;
    s->less = [s->dev newDepthStencilStateWithDescriptor:d];
    s->depth = tex(s, w, h, MTLPixelFormatDepth32Float,
                   MTLTextureUsageRenderTarget, true);
    s->win = tex(s, w, h, MTLPixelFormatRGBA32Float,
                 MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead);
    s->sum =
        tex(s, w, h, MTLPixelFormatRGBA16Float, MTLTextureUsageRenderTarget);
    return s;
  }
}
extern "C" void *ibr_mesh(Scene *s, const void *verts, int nv, const void *idx,
                          int ni, const void *rgb, int w, int h,
                          const void *labels, int mw, int mh) {
  @autoreleasepool {
    auto m = new Mesh();
    m->n = ni;
    m->vb = [s->dev newBufferWithBytes:verts
                                length:nv * 24
                               options:MTLResourceStorageModeShared];
    if (ni)
      m->ib = [s->dev newBufferWithBytes:idx
                                  length:ni * 4
                                 options:MTLResourceStorageModeShared];
    m->photo =
        tex(s, w, h, MTLPixelFormatRGBA8Unorm, MTLTextureUsageShaderRead);
    [m->photo replaceRegion:MTLRegionMake2D(0, 0, w, h)
                mipmapLevel:0
                  withBytes:rgb
                bytesPerRow:w * 4];
    m->labels =
        tex(s, mw, mh, MTLPixelFormatR16Uint, MTLTextureUsageShaderRead);
    [m->labels replaceRegion:MTLRegionMake2D(0, 0, mw, mh)
                 mipmapLevel:0
                   withBytes:labels
                 bytesPerRow:mw * 2];
    return m;
  }
}
extern "C" void ibr_free(Mesh *m) { delete m; }
extern "C" void ibr_add(Scene *s, Mesh *m, const float *u) {
  Call c;
  c.m = m;
  memcpy(c.u, u, sizeof(c.u));
  s->jobs.push_back(c);
}
extern "C" int ibr_render(Scene *s, void *win, void *sum) {
  @autoreleasepool {
    auto b = [s->q commandBuffer];
    for (int p = 0; p < 2; p++) {
      auto d = [MTLRenderPassDescriptor renderPassDescriptor];
      auto c = d.colorAttachments[0];
      c.texture = p ? s->sum : s->win;
      c.loadAction = MTLLoadActionClear;
      c.clearColor = MTLClearColorMake(p ? 0 : 1, 0, 0, 0);
      c.storeAction = MTLStoreActionStore;
      if (!p) {
        d.depthAttachment.texture = s->depth;
        d.depthAttachment.loadAction = MTLLoadActionClear;
        d.depthAttachment.clearDepth = 1;
        d.depthAttachment.storeAction = MTLStoreActionDontCare;
      }
      auto e = [b renderCommandEncoderWithDescriptor:d];
      [e setRenderPipelineState:p ? s->second : s->first];
      if (!p)
        [e setDepthStencilState:s->less];
      else
        [e setFragmentTexture:s->win atIndex:2];
      for (auto &j : s->jobs) {
        if (!j.m->n)
          continue;
        [e setVertexBuffer:j.m->vb offset:0 atIndex:0];
        [e setVertexBytes:j.u length:sizeof(j.u) atIndex:1];
        [e setFragmentBytes:j.u length:sizeof(j.u) atIndex:1];
        [e setFragmentTexture:j.m->photo atIndex:0];
        [e setFragmentTexture:j.m->labels atIndex:1];
        [e drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                      indexCount:j.m->n
                       indexType:MTLIndexTypeUInt32
                     indexBuffer:j.m->ib
               indexBufferOffset:0];
      }
      [e endEncoding];
    }
    [b commit];
    [b waitUntilCompleted];
    s->jobs.clear();
    if (b.error) {
      NSLog(@"%@", b.error);
      return 0;
    }
    auto r = MTLRegionMake2D(0, 0, s->w, s->h);
    [s->sum getBytes:sum bytesPerRow:s->w * 8 fromRegion:r mipmapLevel:0];
    [s->win getBytes:win bytesPerRow:s->w * 16 fromRegion:r mipmapLevel:0];
    return 1;
  }
}
