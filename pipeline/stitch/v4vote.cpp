#include <algorithm>
#include <cmath>
#include <cstdint>
using namespace std;
// Native 1440x900 pixels. Accumulate one observation per centre per exposure;
// no billboard or view-indexed colour.
extern "C" void vote(const float *a, int n, float *stats, const double *T,
                     const double *K, int W, int H, const uint8_t *rgb,
                     const uint8_t *w, const uint8_t *sky, const uint16_t *dep,
                     float base, int ground, int variant) {
  for (int i = 0; i < n; i++, a += 24, stats += 18) {
    float X = a[0] - T[3], Y = a[1] - T[7], Z = a[2] - T[11];
    float z = X * T[2] + Y * T[6] + Z * T[10];
    if (z < 1 || z > 213)
      continue;
    float xf = K[0] * (X * T[0] + Y * T[4] + Z * T[8]) / z + K[2] - .5,
          yf = K[1] * (X * T[1] + Y * T[5] + Z * T[9]) / z + K[3] - .5;
    if (xf < 2 || yf < 2 || xf > W - 4 || yf > H - 4)
      continue;
    int x = int(xf), y = int(yf), j = x + W * y;
    float dx = xf - x, dy = yf - y;
    int jc = int(roundf(xf)) + W * int(roundf(yf));
    float dw = base * (variant ? .42f : 1.f);
    if (sky[jc] > 190 && !ground) {
      stats[9] += dw;
      continue;
    }
    float q = w[jc] / 255.f, d = dep[jc] * .01f;
    if (q < .16 || d < 1)
      continue;
    float tol = ground ? .10 + .014 * z : .42 + .042 * z, err = d - z;
    if (err > tol * 2 && !ground) {
      stats[10] += q * dw;
      continue;
    }
    if (fabs(err) > tol * 1.6f)
      continue;
    float wt =
        dw * q * q * expf(-.65f * err * err / (tol * tol)) * min(1.f, 24.f / z);
    if (wt < .009)
      continue;
    float c[3], grad = 0;
    for (int k = 0; k < 3; k++) {
      auto pix = [&](int off) { return float(rgb[3 * (j + off) + k]); };
      c[k] = (1 - dy) * ((1 - dx) * pix(0) + dx * pix(1)) +
             dy * ((1 - dx) * pix(W) + dx * pix(W + 1));
      float gg = max(fabs(pix(1) - pix(-1)), fabs(pix(W) - pix(-W)));
      grad += gg / 3;
    }
    stats[0] += wt;
    float lum = (c[0] + c[1] + c[2]) / 3;
    for (int k = 0; k < 3; k++)
      stats[1 + k] += wt * c[k];
    stats[4] += wt * lum * lum;
    float closeness = wt / (1 + .037 * z);
    if (closeness > stats[5]) {
      stats[5] = closeness;
      for (int k = 0; k < 3; k++)
        stats[6 + k] = c[k];
    }
    stats[11] += wt * grad;
    stats[12] += wt * clamp(err, -.35f, .35f);
    stats[13] += wt * clamp(err, -.35f, .35f) * X / z;
    stats[14] += wt * clamp(err, -.35f, .35f) * Y / z;
    stats[15] += wt * clamp(err, -.35f, .35f) * Z / z;
    stats[16] += !variant;
    stats[17] += wt * min(grad, 60.f) * min(grad, 60.f);
  }
}
