/* Adapted from MetalSplat rasterize.metal (84e8c3a).

MIT License

Copyright (c) 2026 Thibaut CHAUFFIER

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
// CUDA compositor for the fixed-geometry appearance fit. RGB and the three
// depth/coverage moments use identical weights. One thread per pixel.
#include <cuda_runtime.h>

__device__ float density(const float *uv, const float *co, int g, int x,
                         int y) {
  float dx = x + .5f - uv[2 * g], dy = y + .5f - uv[2 * g + 1];
  float p = -.5f * (co[3 * g] * dx * dx + 2.f * co[3 * g + 1] * dx * dy +
                    co[3 * g + 2] * dy * dy);
  return p > 0.f ? 0.f : expf(p);
}
__global__ void forward(int w, int h, const float *uv, const float *co,
                        const float *a, const float *c, const int *ids,
                        const int *bins, const float *bg, float *im,
                        float *trans, int *last) {
  int x = blockIdx.x * blockDim.x + threadIdx.x,
      y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h)
    return;
  int p = y * w + x, t = blockIdx.y * gridDim.x + blockIdx.x, k = -1;
  float T = 1.f, r[3] = {};
  for (int i = bins[2 * t]; i < bins[2 * t + 1]; ++i) {
    int g = ids[i];
    float alpha = fminf(.99f, a[g] * density(uv, co, g, x, y));
    if (alpha < 1.f / 255)
      continue;
    float next = T * (1 - alpha);
    if (next < 1e-4f)
      break;
    for (int j = 0; j < 3; ++j)
      r[j] += T * alpha * c[3 * g + j];
    T = next;
    k = i;
  }
  for (int j = 0; j < 3; ++j)
    im[3 * p + j] = r[j] + T * bg[j];
  trans[p] = T;
  last[p] = k;
}
__global__ void backward(int w, int h, const float *uv, const float *co,
                         const float *a, const float *c, const int *ids,
                         const int *bins, const float *bg, const float *trans,
                         const int *last, const float *di, float *da,
                         float *dc) {
  int x = blockIdx.x * blockDim.x + threadIdx.x,
      y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h)
    return;
  int p = y * w + x, t = blockIdx.y * gridDim.x + blockIdx.x;
  float T = trans[p], suffix[3] = {bg[0], bg[1], bg[2]};
  for (int i = last[p]; i >= bins[2 * t]; --i) {
    int g = ids[i];
    float d = density(uv, co, g, x, y), raw = a[g] * d,
          alpha = fminf(.99f, raw);
    if (alpha < 1.f / 255)
      continue;
    T /= 1 - alpha;
    float dot = 0;
    for (int j = 0; j < 3; ++j) {
      float v = c[g * 3 + j], grad = di[p * 3 + j];
      dot += (v - suffix[j]) * grad;
      atomicAdd(dc + g * 3 + j, T * alpha * grad);
      suffix[j] = alpha * v + (1 - alpha) * suffix[j];
    }
    if (raw < .99f)
      atomicAdd(da + g, T * d * dot);
  }
}
extern "C" int rs_fit_forward(int w, int h, int s, const float *uv,
                              const float *co, const float *a, const float *c,
                              const int *ids, const int *bins, const float *bg,
                              float *im, float *trans, int *last,
                              cudaStream_t stream) {
  forward<<<dim3((w + s - 1) / s, (h + s - 1) / s), dim3(s, s), 0, stream>>>(
      w, h, uv, co, a, c, ids, bins, bg, im, trans, last);
  return cudaGetLastError();
}
extern "C" int rs_fit_backward(int w, int h, int s, const float *uv,
                               const float *co, const float *a, const float *c,
                               const int *ids, const int *bins, const float *bg,
                               const float *tr, const int *last,
                               const float *di, float *da, float *dc,
                               cudaStream_t stream) {
  backward<<<dim3((w + s - 1) / s, (h + s - 1) / s), dim3(s, s), 0, stream>>>(
      w, h, uv, co, a, c, ids, bins, bg, tr, last, di, da, dc);
  return cudaGetLastError();
}
