#include <algorithm>
#include <cmath>
#include <cstdint>
using namespace std;
// Compare a physical ray's metric range hypotheses in two independent,
// row-timed captures. Grid coordinates are OpenCV centres. Physical KB4
// intrinsics are scaled to that grid.
static bool project(const double *rows, const double *K, const float *p,
                    float &u, float &v) {
  int band = 4;
  for (int k = 0; k < 2; k++) {
    auto T = rows + 16 * band;
    float x = p[0] - T[3], y = p[1] - T[7], z = p[2] - T[11];
    float X = T[0] * x + T[4] * y + T[8] * z,
          Y = T[1] * x + T[5] * y + T[9] * z,
          Z = T[2] * x + T[6] * y + T[10] * z;
    if (Z < .2)
      return false;
    float r = hypotf(X, Y), a = atan2f(r, Z), a2 = a * a;
    float f = a * (1 + a2 * (K[4] + a2 * (K[5] + a2 * (K[6] + a2 * K[7])))) /
              max(r, 1e-8f);
    u = K[0] * f * X + K[2] - .5;
    v = K[1] * f * Y + K[3] - .5;
    if (u < 3 || v < 3 || u >= 720 || v >= 465)
      return false;
    band = clamp(int((v + .5) * 8 / 469), 0, 7);
  }
  return true;
}
static float channel(const uint8_t *p, int x, int y, int c) {
  return p[(y * 724 + x) * 3 + c];
}
static float sample(const uint8_t *p, float x, float y, int c) {
  int u = x, v = y;
  float a = x - u, b = y - v;
  return (1 - b) *
             ((1 - a) * channel(p, u, v, c) + a * channel(p, u + 1, v, c)) +
         b * ((1 - a) * channel(p, u, v + 1, c) +
              a * channel(p, u + 1, v + 1, c));
}
extern "C" void sweep(const float *obs, int n, const double *K,
                      const double *rows, const uint8_t *images,
                      const uint8_t *masks, float *out) {
  const int stride = 724 * 469;
  float ratios[] = {.53,  .64,  .76,  .86, .94, 1,   1.08,
                    1.19, 1.36, 1.58, 1.9, 2.4, 3.1, 4.1};
  for (int i = 0; i < n; i++) {
    auto s = obs + 10 * i;
    int x = s[0], y = s[1];
    float r0 = s[8], trust = s[9];
    float vals[14], best = 1e6;
    int bj = 5;
    auto score = [&](float ratio) {
      float r = r0 * ratio;
      if (r < 1.2 || r > 205)
        return 1e4f;
      float p[] = {s[2] + r * s[5], s[3] + r * s[6], s[4] + r * s[7]};
      float sc[2];
      for (int j = 0; j < 2; j++) {
        float u, v;
        auto im = images + (j + 1) * stride * 3;
        if (!project(rows + 128 * j, K, p, u, v)) {
          sc[j] = 39;
          continue;
        }
        auto label = masks[j * stride + int(v + .5) * 724 + int(u + .5)];
        if (label == 0) {
          sc[j] = 30;
          continue;
        }
        if (label != 1) {
          sc[j] = 42;
          continue;
        }
        float diff = 0;
        for (int c = 0; c < 3; c++) {
          float a = channel(images, x, y, c), b = sample(im, u, v, c);
          diff += min(48.f, abs(a - b)) * .25;
          for (auto d : {-2, 2}) {
            float h = channel(images, x + d, y, c) - a,
                  l = sample(im, u + d, v, c) - b;
            diff += min(28.f, abs(h - l)) * .095;
            h = channel(images, x, y + d, c) - a;
            l = sample(im, u, v + d, c) - b;
            diff += min(28.f, abs(h - l)) * .095;
          }
        }
        sc[j] = diff;
      }
      return min(sc[0], sc[1]) * .62f + max(sc[0], sc[1]) * .38f +
             abs(logf(ratio)) * (1.7f + 4.f * trust);
    };
    for (int j = 0; j < 14; j++) {
      vals[j] = score(ratios[j]);
      if (vals[j] < best) {
        best = vals[j];
        bj = j;
      }
    }
    float br = ratios[bj];
    for (int dir : {-1, 1}) {
      int j = bj + dir;
      if (j < 0 || j >= 14)
        continue;
      for (float w : {.33f, .67f}) {
        float r = ratios[bj] * (1 - w) + ratios[j] * w, err = score(r);
        if (err < best) {
          best = err;
          br = r;
        }
      }
    }
    float alt = 1e4;
    for (int j = 0; j < 14; j++)
      if (abs(logf(ratios[j] / br)) > .20)
        alt = min(alt, vals[j]);
    float certainty = clamp((alt - best - 1.0f) / 5, 0.f, 1.f) *
                      clamp((30 - best) / 16, 0.f, 1.f);
    float blend = clamp((vals[5] - best - .6f) / 3, 0.f, 1.f) * certainty;
    out[i * 2] = r0 * expf(blend * logf(br));
    out[i * 2 + 1] = certainty;
  }
}

// Accumulate soft visibility/free-space log odds using independent, displaced
// exposures.
extern "C" void votes(const float *points, const int32_t *ids, int n,
                      const double *rows, const double *K, const uint8_t *lab,
                      const float *range, const float *trust, float *stats,
                      int lens) {
  auto mid = rows + 64;
  for (int k = 0; k < n; k++) {
    int id = ids[k];
    auto p = points + 4 * id;
    float u, v;
    if (!project(rows, K, p, u, v))
      continue;
    int ix = int(u + .5), iy = int(v + .5);
    int stride = 724;
    int j = ix + stride * iy;
    int l = lab[j];
    if (!l)
      continue;
    auto T = rows + 16 * clamp(int((v + .5) * 8 / 469), 0, 7);
    float X = p[0] - T[3], Y = p[1] - T[7], Z = p[2] - T[11];
    float r = sqrtf(X * X + Y * Y + Z * Z);
    int rad = clamp(int(K[0] * p[3] / r + .5), 1, 10);
    if (ix < rad || iy < rad || ix >= 724 - rad || iy >= 469 - rad)
      continue;
    bool same = lab[j - rad] == l && lab[j + rad] == l &&
                lab[j - 724 * rad] == l && lab[j + 724 * rad] == l;
    float d = min({range[j], range[j - rad], range[j + rad],
                   range[j - 724 * rad], range[j + 724 * rad]});
    float w = trust[j], tol = .28 + r * (lens == 0 ? .065 : .10);
    auto s = stats + 4 * id;
    if (l == 2 && same) {
      s[0] += 1;
      s[3] += 1;
    } else if (l == 3 && same && d > r + tol) {
      s[0] += .95;
      s[3] += 1;
    } else if (l == 1 && d > r + 2 * tol && same) {
      s[0] += w;
      s[2] += w;
    } else if (l == 1 && range[j] > 0 && fabsf(range[j] - r) < tol) {
      s[1] += max(.13f, w);
    }
  }
}

// Displaced physical photos vote on surface placement. Distinguish obscured (no
// vote), visible but inconsistent colour/depth, and a viewpoint matching both.
// Never use chase RGB.
extern "C" void posterior(const float *a, const int32_t *ids, int n,
                          const double *rows, const double *K,
                          const uint8_t *lab, const float *range,
                          const float *trust, const uint8_t *rgb, float *stats,
                          uint32_t *dirs, const float *sky_dist, int lens) {
  for (int k = 0; k < n; k++) {
    int id = ids[k];
    auto p = a + 24 * id;
    float u, v;
    if (!project(rows, K, p, u, v))
      continue;
    int ix = int(u + .5), iy = int(v + .5);
    int j = ix + 724 * iy, l = lab[j];
    if (!l)
      continue;
    auto T = rows + 16 * clamp(int((v + .5) * 8 / 469), 0, 7);
    float X = p[0] - T[3], Y = p[1] - T[7], Z = p[2] - T[11],
          r = sqrtf(X * X + Y * Y + Z * Z);
    float d = range[j], ww = trust[j];
    auto s = stats + 20 * id;
    int rad = clamp(int(K[0] * expf(p[10]) / r), 1, 9);
    bool interior = true;
    if (ix - rad < 0 || ix + rad >= 724 || iy - rad < 0 || iy + rad >= 469)
      continue;
    for (int off : {rad, -rad, 724 * rad, -724 * rad}) {
      if (lab[j + off] != l)
        interior = false;
    }
    if (l == 2) {
      if (interior)
        s[0] += 1;
      continue;
    }
    if (d <= 0 || ww < .035)
      continue;
    float tol = (.2 + r * (lens == 0 ? .06 : lens == 5 ? .22 : .10));
    if (r > d + tol * 1.7) {
      s[5] += .35 * ww;
      continue;
    }
    if (l == 3) {
      if (interior && r < d - tol)
        s[0] += 1.4f;
      continue;
    }
    if (r < d - tol * 1.5 && interior) {
      s[0] += ww;
      continue;
    }
    float err = 0;
    for (int c = 0; c < 3; c++)
      err +=
          fabsf((.5f + .28209479f * p[6 + c]) * 255 - sample(rgb, u, v, c)) / 3;
    float df = fabsf(r - d) / tol;
    float srf = expf(-.75 * df * df), col = expf(-.5f * err * err / 256);
    float w = ww * (lens == 5 ? .6f : 1);
    s[1] += w * srf * col;
    s[2] += w * srf;
    s[3] += w * srf * col * col;
    s[4] += w;
    float pix[3];
    for (int c = 0; c < 3; c++)
      pix[c] = sample(rgb, u, v, c);
    float obs = w * srf * (interior ? 1.f : .35f) / (1 + r / 48);
    if (obs > .001) {
      for (int c = 0; c < 3; c++)
        s[7 + c] += obs * pix[c];
      s[6] += obs;
      float strength = obs * clamp(float(K[0] / (r + 2)) * .05f, .5f, 3.f);
      if (strength > s[10]) {
        s[10] = strength;
        for (int c = 0; c < 3; c++)
          s[11 + c] = pix[c];
        s[14] = r / K[0];
      }
      float mn = (pix[0] + pix[1] + pix[2]) / 3;
      s[15] += obs * mn * mn;
    }
    // Multi-view silhouette: a good centre can still cast a long ellipse across
    // an observed sky gap.
    if (ww > .09 && df < 1.55 &&
        sky_dist[j] <
            3 * expf(max({p[10], p[11], p[12]})) * K[0] / max(.3f, r) + 4) {
      float qw = p[13], qx = p[14], qy = p[15], qz = p[16];
      float axes[3][3] = {{1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy + qw * qz),
                           2 * (qx * qz - qw * qy)},
                          {2 * (qx * qy - qw * qz), 1 - 2 * (qx * qx + qz * qz),
                           2 * (qy * qz + qw * qx)},
                          {p[3], p[4], p[5]}};
      s[19] += ww;
      for (int ax = 0; ax < 3; ax++) {
        float sig = 2.5f * expf(p[10 + ax]);
        bool skyhit = false;
        for (int sign : {-1, 1}) {
          float Xp[] = {p[0] + sign * sig * axes[ax][0],
                        p[1] + sign * sig * axes[ax][1],
                        p[2] + sign * sig * axes[ax][2]};
          float su, sv;
          if (project(rows, K, Xp, su, sv)) {
            int sj = int(sv + .5) * 724 + int(su + .5);
            skyhit |= (sky_dist[sj] < .01f);
          }
        }
        if (skyhit)
          s[16 + ax] += ww;
      }
    }
    if (df < 1.20 && err < 24) {
      int bin = (int(floorf((atan2f(Y, X) + 3.141593f) * 32 / 6.283186f)) & 31);
      dirs[id] |= 1u << bin;
    }
  }
}
