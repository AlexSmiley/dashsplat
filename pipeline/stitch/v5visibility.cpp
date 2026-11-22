#include "v2match.cpp"
// Only visibility evidence. Removed mover pixels are unknown; a parked car can
// still provide a surface vote.
extern "C" void verify_world(const float *a, int n, float *s, const double *T,
                             const double *K, const uint8_t *mask,
                             const float *dep, float odo) {
  for (int i = 0; i < n; i++, a += 24, s += 12) {
    float u, v;
    if (!project(T, K, a, u, v))
      continue;
    int x = int(u + .5), y = int(v + .5), j = y * 724 + x;
    auto row = T + 16 * clamp(y * 8 / 469, 0, 7);
    float dx = a[0] - row[3], dy = a[1] - row[7], dz = a[2] - row[11],
          r = sqrt(dx * dx + dy * dy + dz * dz);
    if (r < 1 || r > 145 || mask[j] == 250)
      continue;
    float d = dep[j], tol = .22 + .044 * r, w = 1 / (1 + .019 * r);
    int cl = mask[j];
    if (d > 1 && r > d + tol * 1.7)
      continue;
    if (cl == 10) {
      s[0] += w;
      continue;
    }
    if (cl == 0 || cl == 1 || cl == 9) {
      s[1] += w;
      continue;
    }
    if (d < 1)
      continue;
    if (d - r > tol * 1.5) {
      s[2] += w;
      continue;
    }
    if (abs(d - r) > tol)
      continue;
    s[3] += w;
    s[4] += 1;
    s[5] = min(s[5], odo);
    s[6] = max(s[6], odo);
    if (cl == 8)
      s[7] += w;
    if (cl >= 13 && cl <= 18)
      s[8] += w;
    if (cl >= 5 && cl <= 7)
      s[9] += w;
    if (cl == 2)
      s[10] += w;
    s[11] += w * abs(d - r) / tol;
  }
}
