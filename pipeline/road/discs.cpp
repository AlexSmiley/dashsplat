// Painter's algorithm, oriented screen-space 3DGS covariance, straight RGB
// alpha.
#include <algorithm>
#include <cmath>
#include <cstdint>
extern "C" void draw(float *canvas, int w, int h, const float *discs, int n) {
  for (int i = 0; i < n; i++) {
    const float *d = discs + 12 * i;
    int x0 = std::max(0, (int)floor(d[0] - d[5])),
        x1 = std::min(w - 1, (int)ceil(d[0] + d[5])),
        y0 = std::max(0, (int)floor(d[1] - d[6])),
        y1 = std::min(h - 1, (int)ceil(d[1] + d[6]));
    for (int y = y0; y <= y1; y++)
      for (int x = x0; x <= x1; x++) {
        float dx = x + .5f - d[0], dy = y + .5f - d[1],
              r = dx * dx * d[2] + 2 * dx * dy * d[3] + dy * dy * d[4];
        if (r > 9)
          continue;
        float a = d[7] * expf(-.5f * r);
        float *p = canvas + 3 * (y * w + x);
        for (int j = 0; j < 3; j++)
          p[j] += a * (d[8 + j] - p[j]);
      }
  }
}
