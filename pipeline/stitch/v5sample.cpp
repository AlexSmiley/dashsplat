#include <algorithm>
#include <cmath>
#include <cstdint>
using namespace std;
// Row-timed physical KB4. RGB stays native; only semantic and metric
// observations use the depth lattice.
extern "C" void observe(const float *A, int n, float *S, const double *rows,
                        const double *K, int W, int H, const uint8_t *im,
                        const uint8_t *sem, const float *depth, float lens) {
  for (int i = 0; i < n; i++, A += 24, S += 18) {
    int row = 4;
    float u = 0, v = 0, dist = 0;
    bool valid = true;
    for (int k = 0; k < 2; k++) {
      auto T = rows + row * 16;
      float x = A[0] - T[3], y = A[1] - T[7], z = A[2] - T[11];
      float X = x * T[0] + y * T[4] + z * T[8],
            Y = x * T[1] + y * T[5] + z * T[9],
            Z = x * T[2] + y * T[6] + z * T[10];
      if (Z < .6) {
        valid = false;
        break;
      }
      float r = hypotf(X, Y), a = atan2f(r, Z), q = a * a,
            sc = a * (1 + q * (K[4] + q * (K[5] + q * (K[6] + q * K[7])))) /
                 max(1e-7f, r);
      u = K[0] * X * sc + K[2] - .5;
      v = K[1] * Y * sc + K[3] - .5;
      if (u < 1 || u >= W - 2 || v < 1 || v >= H - 2) {
        valid = false;
        break;
      }
      row = clamp(int((v + .5) * 8 / H), 0, 7);
      dist = sqrt(x * x + y * y + z * z);
    }
    if (!valid)
      continue;
    int j = int((u + .5) * 724 / W) + 724 * int((v + .5) * 469 / H);
    int cl = sem[j];
    if (cl == 250)
      continue;
    float d = depth[j], tol = .65 + .095 * dist;
    if (d > 0 && dist > d + 2 * tol)
      continue;
    if (cl == 10) {
      S[0] += 1;
      continue;
    }
    if (cl == 0 || cl == 1 || cl == 9) {
      S[1] += 1;
      continue;
    }
    if (d <= 0 || abs(d - dist) > tol * (cl == 2 ? 2.2 : 1.65))
      continue;
    float q = 1 / (1 + abs(d - dist) / tol);
    int ix = int(u), iy = int(v);
    float dx = u - ix, dy = v - iy;
    float rgb[3];
    for (int c = 0; c < 3; c++) {
      auto p = im + (ix + iy * W) * 3 + c;
      rgb[c] = (1 - dy) * ((1 - dx) * p[0] + dx * p[3]) +
               dy * ((1 - dx) * p[3 * W] + dx * p[3 * W + 3]);
    }
    float lum = (rgb[0] + rgb[1] + rgb[2]) / 3;
    S[2] += q;
    S[3] += q * lum * lum;
    for (int c = 0; c < 3; c++)
      S[4 + c] += q * rgb[c];
    float best = lens * q / (.8 + dist * .033);
    if (best > S[7]) {
      S[7] = best;
      for (int c = 0; c < 3; c++)
        S[8 + c] = rgb[c];
    }
    if (cl == 8)
      S[11] += q;
    if (cl == 2)
      S[12] += q;
    if (cl >= 5 && cl <= 7)
      S[13] += q;
    S[14] += 1;
    if (cl >= 13 && cl <= 18)
      S[15] += q;
  }
}
