// Deterministic hot-set fusion. Distances here are along a physical unit ray.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>
using namespace std;
constexpr int W = 724, H = 469, S = 2, X = W / S, Y = (H + S - 1) / S;
constexpr float INF = 1e9;
struct V {
  float x, y, z;
  V operator+(V b) const { return {x + b.x, y + b.y, z + b.z}; }
  V operator-(V b) const { return {x - b.x, y - b.y, z - b.z}; }
  V operator*(float s) const { return {x * s, y * s, z * s}; }
};
float dot(V a, V b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float norm(V a) { return sqrtf(dot(a, a)); }
V cross(V a, V b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
V unit(V a) { return a * (1 / max(.00001f, norm(a))); }
V get(const float *p) { return {p[0], p[1], p[2]}; }
struct Obs {
  V p, n, u, s, c;
  float px, py, r, inc, land, leaf, ground, conditional;
};
static_assert(sizeof(Obs) == 23 * 4);
struct Run {
  float a, b;
};
struct Disc {
  Obs o;
  V src, neg;
  float t = INF, death = INF, first, last, odds = 0, weight = 0;
  int reason = 0, source = -1, lastcam = -1, lastframe = -1, viol = 0;
  bool pending = true, provisional = false;
  vector<Run> runs;
};
struct Spot {
  int id;
  float px, py, r, inc, ax, ay, bx, by;
};
struct Map {
  bool live;
  vector<Disc> d;
  unordered_map<long long, vector<int>> chunks;
  long long counts[8] = {};
};
extern "C" void *init(int live) { return new Map{(bool)live}; }
extern "C" void destroy(void *g) { delete (Map *)g; }
long long key(int x, int y) { return (long long)x * 1000000 + y; }
V origin(const double *T) { return {(float)T[3], (float)T[7], (float)T[11]}; }
V back(const double *T, V v) {
  return {float(T[0] * v.x + T[4] * v.y + T[8] * v.z),
          float(T[1] * v.x + T[5] * v.y + T[9] * v.z),
          float(T[2] * v.x + T[6] * v.y + T[10] * v.z)};
}
bool image(V p, const double *K, float &x, float &y) {
  if (p.z < .1)
    return false;
  float rr = hypotf(p.x, p.y), t = atan2f(rr, p.z), a = t * t;
  float f = t * (1 + a * (K[4] + a * (K[5] + a * (K[6] + a * K[7])))) /
            max(.00001f, rr);
  x = K[0] * f * p.x + K[2];
  y = K[1] * f * p.y + K[3];
  return true;
}
void visit(Disc &d, float t, int cam, int frame) {
  if (t < d.t || t >= d.death)
    return;
  float a = t, b = t;
  if (d.lastcam == cam && frame == d.lastframe + 1 && t >= d.last &&
      t - d.last < .48)
    a = d.last;
  auto &z = d.runs;
  auto it = lower_bound(z.begin(), z.end(), a - .25f,
                        [](const Run &r, float v) { return r.b < v; });
  while (it != z.end() && it->a <= b + .25f) {
    a = min(a, it->a);
    b = max(b, it->b);
    it = z.erase(it);
  }
  z.insert(it, {a, b});
  if (t >= d.last) {
    d.last = t;
    d.lastcam = cam;
    d.lastframe = frame;
  }
}
int px(float x, float y) {
  if (x < 0 || y < 0 || x >= W || y >= H)
    return -1;
  return int(y) * W + int(x);
}
int cell(float x, float y) { return int(y / S) * X + int(x / S); }
void stamp(Spot q, vector<int> &cover, vector<float> &nearest, float mul) {
  float ax = q.ax * mul / S, ay = q.ay * mul / S, bx = q.bx * mul / S,
        by = q.by * mul / S;
  float A = ax * ax + bx * bx + .4f, B = ax * ay + bx * by,
        C = ay * ay + by * by + .4f, D = A * C - B * B;
  float u = q.px / S, v = q.py / S;
  int dx = min(24, (int)ceil(sqrt(A))), dy = min(24, (int)ceil(sqrt(C)));
  for (int y = max(0, int(v) - dy); y <= min(Y - 1, int(v) + dy); y++)
    for (int x = max(0, int(u) - dx); x <= min(X - 1, int(u) + dx); x++) {
      float a = x + .5f - u, b = y + .5f - v;
      if (C * a * a - 2 * B * a * b + A * b * b > D)
        continue;
      int k = y * X + x;
      if (q.r < nearest[k] + .02f) {
        cover[k] = q.id;
        nearest[k] = q.r;
      }
    }
}
extern "C" void take(void *ptr, const float *obs, int n, const double *rows,
                     const double *K, const float *depth, const float *minimum,
                     const unsigned char *rgb, float now, float clock,
                     float readout, int cam, int frame, float reach) {
  Map &g = *(Map *)ptr;
  V cen = origin(rows + 16 * 16);
  int tx = floor(cen.x / 8), ty = floor(cen.y / 8), nr = ceil((reach + 1) / 8);
  vector<int> cover(X * Y, -1), nearestId(X * Y, -1);
  vector<float> zbuf(X * Y, INF), cz(X * Y, INF);
  vector<Spot> hits, replace;
  for (int x = tx - nr; x <= tx + nr; x++)
    for (int y = ty - nr; y <= ty + nr; y++) {
      auto it = g.chunks.find(key(x, y));
      if (it == g.chunks.end())
        continue;
      for (int id : it->second) {
        auto &d = g.d[id];
        if (d.death <= now)
          continue;
        if (d.pending && now - d.first > (d.o.r > 28 ? 4.5f : 2.5f)) {
          d.death = now;
          continue;
        }
        auto &o = d.o;
        V v = o.p - cen;
        float r = norm(v);
        if (r > reach + 1 || r < 1)
          continue;
        float u, yy;
        auto T = rows + 16 * 16;
        V pp = back(T, v);
        if (!image(pp, K, u, yy) || u < -8 || u > W + 8 || yy < -8 ||
            yy > H + 8)
          continue;
        T = rows + 16 * clamp(int(round(yy / H * 32)), 0, 32);
        v = o.p - origin(T);
        pp = back(T, v);
        if (!image(pp, K, u, yy) || px(u, yy) < 0)
          continue;
        r = norm(v);
        float inc = abs(dot(v, o.n)) / r, ax, ay, bx, by;
        if (!image(pp + back(T, o.u) * o.s.x, K, ax, ay) ||
            !image(pp + back(T, cross(o.n, o.u)) * o.s.y, K, bx, by))
          continue;
        Spot sp{id,
                u,
                yy,
                r,
                inc,
                clamp(ax - u, -35.f, 35.f),
                clamp(ay - yy, -35.f, 35.f),
                clamp(bx - u, -35.f, 35.f),
                clamp(by - yy, -35.f, 35.f)};
        hits.push_back(sp);
        if (!d.pending)
          stamp(sp, nearestId, zbuf, 1.15);
      }
    }
  for (auto sp : hits) {
    auto &d = g.d[sp.id];
    if (d.death <= now || (d.pending && d.o.land && d.o.ground <= now))
      continue;
    auto &o = d.o;
    float tau = max(.09f, .032f * sp.r) / max(.4f, sp.inc),
          dr = depth[px(sp.px, sp.py)] - sp.r;
    bool agree = false, through = sp.inc > .5f;
    int count = 0, cons = 0;
    int ij = px(sp.px, sp.py);
    auto color = rgb + 3 * ij;
    bool paint =
        abs(color[0] - o.c.x) + abs(color[1] - o.c.y) + abs(color[2] - o.c.z) <
        59;
    if (d.pending && !paint && sp.r > 22)
      continue;
    bool wall = sp.r > 28 && abs(o.n.z) < .72f && !o.land && !o.leaf &&
                sp.inc > .45f && paint;
    float gateR = wall ? .125f : .065f;
    if (wall)
      tau = max(tau, .095f * sp.r);
    for (auto t : array<array<float, 2>, 5>{
             {{0, 0}, {.58, 0}, {-.58, 0}, {0, .58}, {0, -.58}}}) {
      auto u = sp.px + t[0] * sp.ax + t[1] * sp.bx,
           v = sp.py + t[0] * sp.ay + t[1] * sp.by;
      int k = px(u, v);
      if (k < 0) {
        through = false;
        continue;
      }
      float z = depth[k];
      count += z > 0;
      cons += (z > 0 && abs(z - sp.r) * sp.inc < tau &&
               abs(z - sp.r) < max(.19f, gateR * sp.r));
      through &= minimum[k] > sp.r + 2 * tau;
    }
    agree = count >= 3 && cons >= 3 &&
            abs(dr) < max(.23f, (gateR + .004f) * sp.r) &&
            sp.r < zbuf[cell(sp.px, sp.py)] + max(.23f, .055f * sp.r);
    if (agree) {
      if (d.pending) {
        float baseline = norm(cen - d.src);
        V path = cen - d.src, ray = unit(o.p - d.src);
        float perp = norm(path - ray * dot(path, ray));
        if (baseline < max(.32f, .022f * sp.r) ||
            (sp.r > 23 && perp < (wall ? .010f : .029f) * sp.r))
          continue;
        d.pending = false;
        d.t = now;
        d.odds = 2.2f;
        g.counts[0]++;
        visit(d, now, cam, frame);
      } else {
        visit(d, g.live ? now : clock + (sp.py / H - .5f) * readout, cam,
              frame);
        d.odds = min(4.f, d.odds + 1.f);
        if (norm(cen - d.src) > .65f)
          d.provisional = false;
        g.counts[1]++;
      }
      stamp(sp, cover, cz, 1.32f);
      replace.push_back(sp);
    } else if (through && !d.pending && norm(cen - d.neg) > .65f &&
               sp.r <= zbuf[cell(sp.px, sp.py)] + .08f) {
      d.neg = cen;
      d.odds -= .4f;
      d.viol++;
      g.counts[2]++;
      if (d.odds <= 1.25f && d.viol >= (d.provisional ? 2 : 3)) {
        if (now + .25f < d.death) {
          d.reason = 1;
          d.death = now + .25f;
        }
        g.counts[3]++;
      }
    }
  }
  vector<int> fine(X * Y, -1);
  vector<float> fz(X * Y, INF);
  for (int i = 0; i < n; i++) {
    Obs a;
    memcpy(&a, obs + i * 23, sizeof(a));
    int k = cell(a.px + .5f, a.py + .5f);
    int id = cover[k];
    bool newborn = false;
    if (id >= 0) {
      auto &d = g.d[id];
      if (d.death <= now)
        id = -1;
      else {
        auto &o = d.o;
        V disp = a.p - o.p;
        float e = dot(disp, o.n);
        float lim = max(.14f, .057f * a.r);
        bool same = abs(e) < lim && abs(a.r - cz[k]) < 2 * lim &&
                    (a.leaf || o.leaf || abs(dot(a.n, o.n)) > .47f);
        if (same) {
          float size = max(o.s.y, .45f * o.s.x), ns = max(a.s.y, .45f * a.s.x);
          bool better = ns < size / 1.55f ||
                        (a.r < o.r * .60f && a.r < 21 && ns < size * 1.10f);
          if (!better || now - d.t < .25f) {
            if (!g.live && !d.pending) {
              float wt = a.inc / (a.r * a.r + .5f);
              float frac = min(.38f, wt / (wt + d.weight));
              if (ns < size * 1.5 && norm(disp) < max(.27f, .035f * a.r)) {
                o.c = o.c * (1 - frac) + a.c * frac;
                if (!o.land)
                  o.p = o.p + o.n * (clamp(e, -.15f, .15f) * frac);
                d.weight = min(d.weight + wt, wt * 18);
              }
            }
            continue;
          }
          if (abs(e) < lim * .45f && !a.leaf && !a.land)
            a.p = a.p - o.n * e;
          newborn = true;
        } else if (abs(a.r - cz[k]) < lim && !a.leaf)
          continue;
      }
    }
    if (a.land && a.ground <= now)
      continue;
    // The producer's pose-conditioned window already contains distinct images.
    // A clean near patch or frontal wall may be born provisionally on first
    // sight. Suppress another same-depth sheet even when its parent missed the
    // colour gate.
    int previous = nearestId[k];
    if (id < 0 && previous >= 0 && !newborn) {
      auto &pr = g.d[previous];
      if (pr.death > now && abs(a.r - zbuf[k]) < max(.3f, .10f * a.r) &&
          abs(dot(a.p - pr.o.p, pr.o.n)) < max(.2f, .06f * a.r) &&
          a.s.y > .63f * pr.o.s.y)
        continue;
    }
    bool strongObs = !a.conditional && (a.r < 24 || a.c.y > 60.f) && !a.land &&
                     ((a.r < 24 && a.inc > .30f) ||
                      (a.inc > .45f && abs(a.n.z) < .75f && !a.leaf));
    Disc d;
    d.source = cam;
    d.o = a;
    d.src = cen;
    d.first = now;
    d.last = now;
    d.neg = cen;
    d.weight = a.inc / (a.r * a.r + .5f);
    d.death = a.land && a.ground < INF ? a.ground + .25f : INF;
    d.pending = !(newborn || strongObs);
    d.provisional = strongObs && !newborn;
    if (!d.pending) {
      d.t = now;
      d.odds = newborn ? 2.2f : 1.4f;
      visit(d, now, cam, frame);
    }
    int j = g.d.size();
    g.d.push_back(std::move(d));
    g.chunks[key(floor(a.p.x / 8), floor(a.p.y / 8))].push_back(j);
    g.counts[4]++;
    float f = K[0] / max(1.f, a.r);
    Spot sp{j,   a.px + .5f, a.py + .5f,
            a.r, a.inc,      min(24.f, a.s.y * f),
            0,   0,          min(24.f, a.s.y * f)};
    stamp(sp, cover, cz, 1.1);
    if (newborn)
      stamp(sp, fine, fz, 1.55);
  }
  for (auto sp : replace) {
    auto &d = g.d[sp.id];
    if (d.death <= now + .25f || d.pending)
      continue;
    int yes = 0, tot = 0;
    for (float i : {-.64f, 0.f, .64f})
      for (float j : {-.64f, 0.f, .64f}) {
        float u = sp.px + i * sp.ax + j * sp.bx,
              v = sp.py + i * sp.ay + j * sp.by;
        if (px(u, v) < 0)
          continue;
        tot++;
        int k = cell(u, v);
        int other = fine[k] >= 0 ? fine[k] : cover[k];
        if (other >= 0 && other != sp.id) {
          auto &o = g.d[other];
          yes += !o.pending && o.death > now &&
                 ((o.o.s.y < d.o.s.y * .78f) || (o.o.r < d.o.r * .62f)) &&
                 abs(norm(o.o.p - cen) - sp.r) < max(.25f, .06f * sp.r);
        }
      }
    if (tot >= 5 && yes >= .8f * tot) {
      d.death = now + .25f;
      d.reason = 2;
      g.counts[5]++;
    }
  }
}
extern "C" long output(void *ptr, const char *file, const char *extra,
                       long long *counts) {
  Map &g = *(Map *)ptr;
  FILE *f = fopen(file, "wb"), *side = fopen(extra, "wb");
  FILE *flags = fopen((std::string(file) + ".meta").c_str(), "wb");
  long n = 0;
  for (auto &d : g.d) {
    if (d.pending || d.t >= d.death || (!g.live && d.death < INF))
      continue;
    auto o = d.o;
    auto z = d.runs;
    if (!g.live) {
      if (z.empty() || z[0].a > 0)
        z.insert(z.begin(), {0, 0});
    } else if (z.empty() || z[0].a > d.t + .25f)
      z.insert(z.begin(), {d.t, d.t});
    if (!g.live && z.size() > 3) {
      for (size_t i = 2; i + 1 < z.size(); i++) {
        int no = n;
        fwrite(&no, 4, 1, side);
        fwrite(&z[i], 8, 1, side);
      }
      z = {z[0], z[1], z.back()};
    }
    for (size_t h = 0; h < z.size(); h += 3) {
      float birth = h ? z[h].a : (g.live ? d.t : 0),
            end = h + 3 < z.size() ? min(d.death, z[h + 3].a) : d.death;
      if (birth >= end)
        continue;
      float tm[7] = {birth, end, clamp(z[h].b, birth, end), INF, INF, INF, INF};
      for (int j = 1; j < 3 && h + j < z.size(); j++) {
        if (z[h + j].a > end)
          break;
        tm[1 + 2 * j] = max(tm[2 * j], z[h + j].a);
        tm[2 + 2 * j] = clamp(z[h + j].b, tm[1 + 2 * j], end);
      }
      fwrite(&o, 4, 15, f);
      float alpha =
          g.live ? (o.leaf ? .60f : (o.land ? .65f : .73f))
                 : (d.provisional ? .42f
                                  : (o.leaf ? .69f : (o.land ? .68f : .82f)));
      fwrite(&alpha, 4, 1, f);
      fwrite(tm, 4, 7, f);
      unsigned char tag[4] = {
          (unsigned char)d.source, (unsigned char)o.land, (unsigned char)o.leaf,
          (unsigned char)(d.provisional |
                          ((end == d.death ? d.reason : 0) << 1))};
      fwrite(tag, 1, 4, flags);
      n++;
      g.counts[6] += (h > 0);
      g.counts[7] += (o.land != 0);
    }
  }
  fclose(f);
  fclose(side);
  fclose(flags);
  memcpy(counts, g.counts, sizeof(g.counts));
  return n;
}
