// Sorted DC Gaussians. Analytic fisheye differential at the exposure row.
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
using namespace std;
struct V {
  float x, y, z;
};
V cam(const double *T, V p, bool pt) {
  if (pt) {
    p.x -= T[3];
    p.y -= T[7];
    p.z -= T[11];
  }
  return {float(T[0] * p.x + T[4] * p.y + T[8] * p.z),
          float(T[1] * p.x + T[5] * p.y + T[9] * p.z),
          float(T[2] * p.x + T[6] * p.y + T[10] * p.z)};
}
struct P {
  float x, y, fx[3], fy[3];
};
P proj(V v, const double *K, bool fish) {
  float x = v.x, y = v.y, z = v.z;
  P p;
  float fx = K[0], fy = K[1];
  float r = hypotf(x, y);
  if (!fish || r < 1e-5) {
    float iz = 1 / z;
    p = {fx * x * iz + (float)K[2],
         fy * y * iz + (float)K[3],
         {fx * iz, 0, -fx * x * iz * iz},
         {0, fy * iz, -fy * y * iz * iz}};
  } else {
    float t = atan2f(r, z), u = t * t,
          rad = t * (1 + u * (K[4] + u * (K[5] + u * (K[6] + u * K[7])))),
          der = 1 +
                u * (3 * K[4] + u * (5 * K[5] + u * (7 * K[6] + u * 9 * K[7])));
    float a = rad / r, dr = (der * z / (r * r + z * z) - a) / (r * r),
          dz = -der / (r * r + z * z);
    p = {fx * x * a + (float)K[2],
         fy * y * a + (float)K[3],
         {fx * (a + x * x * dr), fx * x * y * dr, fx * x * dz},
         {fy * x * y * dr, fy * (a + y * y * dr), fy * y * dz}};
  }
  return p;
}
extern "C" void raster(const float *pts, int count, float t, int live,
                       const double *row, const double *K, int fish, int w,
                       int h, float *out) {
  vector<array<float, 13>> records;
  records.reserve(count / 3);
  for (int i = 0; i < count; i++) {
    const auto *a = pts + i * 25;
    if (t < a[17] || t >= a[18])
      continue;
    V pos{a[0], a[1], a[2]};
    auto *T = row + 16 * 16;
    V v = cam(T, pos, true);
    if (v.z < .25)
      continue;
    P p = proj(v, K, fish);
    if (p.x < -70 || p.x > w + 70 || p.y < -70 || p.y > h + 70)
      continue;
    if (fish) {
      T = row + 16 * clamp(int(round(p.y / h * 32)), 0, 32);
      v = cam(T, pos, true);
      if (v.z < .25)
        continue;
      p = proj(v, K, fish);
    }
    float r = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (live < 2 && r > 135)
      continue;
    float qw = a[13], x = a[14], y = a[15], z = a[16];
    V axes[3] = {
        {1 - 2 * (y * y + z * z), 2 * (x * y + qw * z), 2 * (x * z - qw * y)},
        {2 * (x * y - qw * z), 1 - 2 * (x * x + z * z), 2 * (y * z + qw * x)},
        {a[3], a[4], a[5]}};
    float A = 0, B = 0, C = 0;
    for (int q = 0; q < 3; q++) {
      V e = cam(T, axes[q], false);
      float s = expf(a[10 + q]),
            xx = s * (p.fx[0] * e.x + p.fx[1] * e.y + p.fx[2] * e.z),
            yy = s * (p.fy[0] * e.x + p.fy[1] * e.y + p.fy[2] * e.z);
      A += xx * xx;
      B += xx * yy;
      C += yy * yy;
    }
    float det = A * C - B * B;
    A += .22f;
    C += .22f;
    float D = max(1e-6f, A * C - B * B),
          alpha = sqrtf(max(0.f, det) / D) / (1 + expf(-a[9]));
    if (live == 1 || live == 2)
      alpha *= clamp((t - a[17]) / .25f, 0.f, 1.f) *
               clamp((a[18] - t) / .25f, 0.f, 1.f);
    if (alpha < .002)
      continue;
    records.push_back({p.x, p.y, C / D, -B / D, A / D, min(120.f, 3 * sqrtf(A)),
                       min(120.f, 3 * sqrtf(C)), alpha,
                       (.5f + .28209479f * a[6]) * 255,
                       (.5f + .28209479f * a[7]) * 255,
                       (.5f + .28209479f * a[8]) * 255, r, a[24]});
  }
  sort(records.begin(), records.end(),
       [](auto &a, auto &b) { return a[11] > b[11]; });
  for (auto &d : records) {
    for (int y = max(0, (int)(d[1] - d[6]));
         y <= min(h - 1, (int)(d[1] + d[6])); y++)
      for (int x = max(0, (int)(d[0] - d[5]));
           x <= min(w - 1, (int)(d[0] + d[5])); x++) {
        float dx = x + .5f - d[0], dy = y + .5f - d[1],
              rr = dx * dx * d[2] + 2 * dx * dy * d[3] + dy * dy * d[4];
        if (rr > 9)
          continue;
        float a = expf(-.5f * rr) * d[7];
        auto *p = out + (y * w + x) * 6;
        for (int j = 0; j < 3; j++)
          p[j] += a * (d[8 + j] - p[j]);
        p[3] += a * (1 - p[3]);
        p[4] += a * (d[11] - p[4]);
        p[5] += a * (d[12] - p[5]);
      }
  }
}
