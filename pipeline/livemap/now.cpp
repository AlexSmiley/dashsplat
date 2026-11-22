// Latest-view fusion; retired generations remain on the playback tape, never in
// the hot set.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <vector>
using namespace std;
const int W = 724, H = 469, D = 2, X = (W + 1) / 2, Y = (H + 1) / 2;
const float INF = 1e9f;
struct V {
  float x, y, z;
  V operator+(V b) const { return {x + b.x, y + b.y, z + b.z}; }
  V operator-(V b) const { return {x - b.x, y - b.y, z - b.z}; }
  V operator*(float k) const { return {x * k, y * k, z * k}; }
};
float dot(V a, V b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float len(V p) { return sqrt(dot(p, p)); }
V cross(V a, V b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
V org(const double *T) { return {(float)T[3], (float)T[7], (float)T[11]}; }
V loc(const double *T, V p) {
  return {float(T[0] * p.x + T[4] * p.y + T[8] * p.z),
          float(T[1] * p.x + T[5] * p.y + T[9] * p.z),
          float(T[2] * p.x + T[6] * p.y + T[10] * p.z)};
}
bool proj(V p, const double *K, float &x, float &y) {
  if (p.z <= .10f)
    return false;
  float r = hypotf(p.x, p.y), t = atan2f(r, p.z), s = t * t,
        f = t * (1 + s * (K[4] + s * (K[5] + s * (K[6] + s * K[7])))) /
            max(r, .00001f);
  x = p.x * f * K[0] + K[2];
  y = p.y * f * K[1] + K[3];
  return true;
}
struct Obs {
  V p, n, u, s, rgb;
  float x, y, r, step, inst, expiry;
};
static_assert(sizeof(Obs) == 21 * 4);
struct Disc {
  V p, n, u, s, rgb;
  float radius, born, dead, tm[5], last;
  int inst, cam, frame, lastcam, lastframe;
};
struct Bucket {
  V ctr;
  vector<int> hot;
};
long long grid(int x, int y) { return (long long)x * 1000000 + y; }
struct World {
  vector<Disc> a;
  unordered_map<long long, Bucket> b;
  vector<int> moves[6];
  long long cnt[7] = {};
  int push(Disc d) {
    int id = a.size();
    a.push_back(d);
    int x = floor(d.p.x / 20), y = floor(d.p.y / 20);
    auto &b0 = b[grid(x, y)];
    b0.ctr = {(x + .5f) * 20, (y + .5f) * 20, 0};
    b0.hot.push_back(id);
    if (d.inst)
      moves[d.cam].push_back(id);
    return id;
  }
};
extern "C" void *now_init() { return new World(); }
extern "C" void now_free(void *p) { delete (World *)p; }
struct Hit {
  int id, k;
  float x, y, r, ax, ay, bx, by;
};
void paint(const Hit &s, vector<float> &z, vector<int> &ids, float mul) {
  float u = s.x / D, v = s.y / D;
  float A = (s.ax * s.ax + s.bx * s.bx) * mul * mul / 4 + .53f,
        C = (s.ay * s.ay + s.by * s.by) * mul * mul / 4 + .53f,
        B = (s.ax * s.ay + s.bx * s.by) * mul * mul / 4, det = A * C - B * B;
  int dx = min(18, (int)ceil(sqrt(A))), dy = min(18, (int)ceil(sqrt(C)));
  for (int yy = max(0, (int)v - dy); yy <= min(Y - 1, (int)v + dy); yy++)
    for (int xx = max(0, (int)u - dx); xx <= min(X - 1, (int)u + dx); xx++) {
      float x = xx + .5f - u, y = yy + .5f - v;
      if (C * x * x - 2 * B * x * y + A * y * y > det)
        continue;
      int k = yy * X + xx;
      if (s.r < z[k]) {
        z[k] = s.r;
        ids[k] = s.id;
      }
    }
}
void retire(World &g, Disc &d, float t, int reason) {
  if (d.dead > t + .25001f) {
    d.dead = t + .25f;
    g.cnt[reason]++;
  }
}
int visit(World &g, int i, float now, int cam, int frame) {
  Disc &d = g.a[i];
  int j = (d.tm[3] < INF) ? 4 : (d.tm[1] < INF ? 2 : 0);
  if (now - d.last <= .25f ||
      (cam == d.lastcam && frame == d.lastframe + 1 && now - d.last < .75f)) {
    d.tm[j] = now;
  } else if (j < 4) {
    d.tm[j + 1] = d.tm[j + 2] = now;
  } else {
    Disc copy = d;
    retire(g, d, now, 5);
    copy.born = now;
    copy.last = now;
    copy.lastcam = cam;
    copy.lastframe = frame;
    copy.tm[0] = now;
    for (int h = 1; h < 5; h++)
      copy.tm[h] = INF;
    return g.push(copy);
  }
  d.last = now;
  d.lastcam = cam;
  d.lastframe = frame;
  return i;
}
extern "C" void now_take(void *ptr, const float *obs, int count,
                         const double *rows, const double *K,
                         const float *depth, const unsigned short *dyn,
                         const unsigned char *image, float now, int cam,
                         int frame, const float *heading) {
  World &g = *(World *)ptr;
  V centre = org(rows + 16 * 16);
  for (int id : g.moves[cam])
    retire(g, g.a[id], now, 3);
  g.moves[cam].clear();
  vector<float> nearest(X * Y, INF), agreeZ(X * Y, INF);
  vector<int> nearestId(X * Y, -1), agreeId(X * Y, -1);
  vector<Hit> hits;
  float reach = 0;
  for (int k = 0; k < W * H; k++)
    reach = max(reach, depth[k]);
  for (auto &kv : g.b) {
    auto &b = kv.second;
    float xx = b.ctr.x - centre.x, yy = b.ctr.y - centre.y,
          dist = hypotf(xx, yy);
    if (b.hot.empty())
      continue;
    bool far = dist > 85 && xx * heading[0] + yy * heading[1] < -50;
    if (dist > max(reach + 18, 220.f) && !far)
      continue;
    unordered_map<long long, int> mem;
    int n = 0;
    for (int id : b.hot) {
      auto &d = g.a[id];
      if (d.dead <= now + .25001f)
        continue;
      // Keep a deterministic coarse subset of old rear memory without rewriting
      // its history.
      if (far && now - d.last > 5 && !d.inst && d.radius < .75f) {
        int nx = (int)floor((d.p.x - b.ctr.x) * 1.4) + 15,
            ny = (int)floor((d.p.y - b.ctr.y) * 1.4) + 15,
            nz = (int)floor(d.p.z * 1.4) + 10000,
            axis = abs(d.n.z) > .65f         ? 2
                   : abs(d.n.x) > abs(d.n.y) ? 1
                                             : 0;
        long long cell =
            ((((long long)nz * 32 + ny) * 32 + nx) * 3 + axis) * 8 +
            (int)(d.rgb.y / 32);
        if (mem.count(cell)) {
          retire(g, d, now, 4);
          continue;
        }
        mem[cell] = id;
      }
      b.hot[n++] = id;
      if (dist > max(reach + 18, 220.f))
        continue;
      const double *T = rows + 16 * 16;
      V p = loc(T, d.p - centre);
      float x, y;
      if (!proj(p, K, x, y) || x < -10 || x > W + 10 || y < -10 || y > H + 10)
        continue;
      T = rows + clamp((int)round(y / H * 32), 0, 32) * 16;
      p = loc(T, d.p - org(T));
      if (!proj(p, K, x, y) || x < 0 || x >= W || y < 0 || y >= H)
        continue;
      float r = len(p), ax, ay, bx, by;
      if (!proj(p + loc(T, d.u) * d.s.x, K, ax, ay) ||
          !proj(p + loc(T, cross(d.n, d.u)) * d.s.y, K, bx, by))
        continue;
      Hit s{id, int(y) * W + int(x), x, y, r, ax - x, ay - y, bx - x, by - y};
      hits.push_back(s);
      paint(s, nearest, nearestId, .9f);
    }
    b.hot.resize(n);
  }
  for (auto s : hits) {
    auto &d = g.a[s.id];
    float z = depth[s.k];
    int free = 0, seen = 0;
    for (auto q : array<array<float, 2>, 5>{
             {{0, 0}, {.65f, 0}, {-.65f, 0}, {0, .65f}, {0, -.65f}}}) {
      int u = int(s.x + q[0] * s.ax + q[1] * s.bx),
          v = int(s.y + q[0] * s.ay + q[1] * s.by);
      if (u < 0 || u >= W || v < 0 || v >= H)
        continue;
      float rz = depth[v * W + u];
      seen += rz != 0;
      free += rz < 0 || rz > s.r + max(.25f, .115f * s.r);
    }
    if ((z < 0 && free >= 2) || (seen >= 3 && free >= 4)) {
      retire(g, d, now, 0);
      continue;
    }
    if (z <= 0)
      continue;
    float tol = max(.20f, min(.16f, .075f + .0007f * z) * z);
    int k = int(s.y / D) * X + int(s.x / D);
    float delta = z - s.r;
    if (delta > tol) {
      retire(g, d, now, 0);
      continue;
    }
    if (d.inst) {
      if (abs(delta) <= tol)
        retire(g, d, now, 3);
      continue;
    }
    if (dyn[s.k])
      continue;
    if (delta < -tol) {
      auto c = image + 3 * s.k;
      float rgb =
          abs(c[0] - d.rgb.x) + abs(c[1] - d.rgb.y) + abs(c[2] - d.rgb.z);
      if (s.r <= nearest[k] + tol * .35f && s.r - z < max(.45f, .32f * z) &&
          rgb < 65) {
        retire(g, d, now, 1);
      }
      continue;
    }
    if (s.r > nearest[k] + tol)
      continue;
    s.id = visit(g, s.id, now, cam, frame);
    paint(s, agreeZ, agreeId, .96);
    g.cnt[6]++;
  }
  for (int i = 0; i < count; i++) {
    Obs o;
    memcpy(&o, obs + 21 * i, sizeof(o));
    int k =
        clamp((int)(o.y / D), 0, Y - 1) * X + clamp((int)(o.x / D), 0, X - 1);
    if (o.expiry <= now)
      continue;
    float radius = max(o.s.y, .4f * o.s.x);
    int parent = agreeId[k];
    if (!o.inst && parent >= 0 &&
        abs(o.r - agreeZ[k]) <=
            max(.20f, (.075f + min(.085f, .0007f * o.r)) * o.r)) {
      auto &d = g.a[parent];
      if (d.dead > now + .25001f) {
        if (radius * 1.3f >= d.radius)
          continue;
        retire(g, d, now, 2);
      }
    }
    Disc d{o.p,
           o.n,
           o.u,
           o.s,
           o.rgb,
           radius,
           now,
           (o.expiry < INF ? o.expiry + .25f : INF),
           {now, INF, INF, INF, INF},
           now,
           (int)o.inst,
           cam,
           frame,
           cam,
           frame};
    g.push(d);
  }
}
extern "C" long now_save(void *ptr, const char *out, const char *tags,
                         long long *cnt, float last) {
  World &g = *(World *)ptr;
  FILE *f = fopen(out, "wb"), *m = fopen(tags, "wb");
  if (!f || !m)
    return -1;
  for (auto &d : g.a) {
    float end = d.inst ? min(d.dead, last + .25f) : d.dead;
    float tm[7] = {d.born, end, d.tm[0], d.tm[1], d.tm[2], d.tm[3], d.tm[4]};
    fwrite(&d.p, 4, 15, f);
    fwrite(tm, 4, 7, f);
    int tag[3] = {d.cam, d.frame, d.inst};
    fwrite(tag, 4, 3, m);
  }
  fclose(f);
  fclose(m);
  memcpy(cnt, g.cnt, sizeof(g.cnt));
  return g.a.size();
}
