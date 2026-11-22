// Whole-drive ground fusion. One independent 50 m ENU tile per worker.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
using namespace std;
constexpr int M = 220, F = 880;
constexpr double P = 50. / F, S = 4 * P;
constexpr float INF = 1e9;
struct View {
  float u, v, z;
  bool ok;
};
struct Fine {
  float w = 0, best = 0, colors[3] = {}, raw[3] = {};
  uint8_t winner[3] = {}, winraw[3] = {}, winbase[3] = {};
};
struct Ref {
  float x = 0, y = 0, t = -1000, lum[4] = {}, col[3] = {}, sig[3][5] = {};
  int cam = -1;
};
struct Cell {
  Ref refs[3];
  int nref = 0, yes = 0, no = 0, level = -1, higher = 0, plane = 0, semno = 0,
      semantic = 0;
  float base[3] = {}, bw = 0;
  float best = 0, lat = 100, z = 0, nx = 0, ny = 0;
  int ids[16];
  uint16_t seen = 0;
  bool trusted = false, inferred = false;
  float major = 100, minor = 100, axis = 0, range = 100, candidate = 0,
        candz = 0, candnx = 0, candny = 0, candMajor = 0, candMinor = 0,
        candAxis = 0, candLat = 100, candRange = 100;
  int independent = 0, negclose = 0, winnerCam = -1, polished = 0;
  bool detail = false;
  float evidenceRange = 1.e8f;
  float lastX = 1e8, lastY = 1e8;
  bool blocked = false;
  uint8_t candidateRGB[16][3] = {};
  uint16_t candmask = 0;
};
struct Record {
  float x, y, z, nx, ny, sz, t, death = INF,
                                time[5] = {INF, INF, INF, INF, INF};
  uint8_t rgb[3], raw[3], level;
  int macro, bit;
  float sy = 0, theta = 0, alpha = .88;
};
struct Vis {
  float t, dt_x, dt_y, dt_xy;
  int cell;
  uint16_t mask;
  int packet;
};
struct Run {
  float a, b;
};
struct Tile {
  double x, y;
  bool live, v2;
  vector<Cell> C;
  vector<Fine> f;
  vector<Record> records;
  vector<Vis> visits;
  vector<int> changed;
  long long samples = 0, masked = 0, raises = 0;
  int pkt = 0;
  Tile(int ix, int iy, int v)
      : x(ix * 50.), y(iy * 50.), live(v & 1), v2(v >= 2), C(M * M),
        f(M * M * 16) {}
};
extern "C" void *create(int ix, int iy, int live) {
  return new Tile(ix, iy, live);
}
extern "C" void release(void *t) { delete (Tile *)t; }
inline float clampf(float x, float a, float b) { return min(b, max(a, x)); }
// row[r]: world-to-camera rotation (9), camera centre (3), upward plane n and d
// (4).
View project(double x, double y, float raised, const double *rows,
             const double *intr, int w, int h) {
  double u = 0, v = h * .5, z = 0;
  bool ok = true;
  for (int iter = 0; iter < 3; iter++) {
    int ri = max(0, min(16, (int)round(v * 16 / h)));
    const auto *a = rows + 16 * ri;
    z = -(a[12] * x + a[13] * y + a[15]) / a[14] + raised;
    double dx = x - a[9], dy = y - a[10], dz = z - a[11];
    double cx = a[0] * dx + a[1] * dy + a[2] * dz,
           cy = a[3] * dx + a[4] * dy + a[5] * dz,
           cz = a[6] * dx + a[7] * dy + a[8] * dz;
    double r = hypot(cx, cy), th = atan2(r, cz), q = th * th,
           k = th *
               (1 +
                q * (intr[4] + q * (intr[5] + q * (intr[6] + q * intr[7])))) /
               max(r, 1.e-12);
    u = intr[0] * cx * k + intr[2];
    v = intr[1] * cy * k + intr[3];
    ok = cz > 0 && th < intr[8] && dz < -.25;
  }
  return {(float)u, (float)v, (float)z, ok};
}
inline int bits(int v) { return __builtin_popcount(v); }
void emit(Tile &g, int m, int level, float birth) {
  auto &c = g.C[m];
  int nd = 1 << level, step = 4 / nd;
  float pitch = S / nd;
  int supporting = 0;
  for (int yy = -1; yy <= 1; yy++)
    for (int xx = -1; xx <= 1; xx++) {
      if (!xx && !yy)
        continue;
      int x = m % M + xx, y = m / M + yy;
      if (x < 0 || x >= M || y < 0 || y >= M) {
        supporting++;
        continue;
      }
      auto &nb = g.C[y * M + x];
      supporting += nb.trusted && nb.yes;
    }
  float border = g.v2 ? clampf((supporting + 1) / 8.f, .30f, 1.f) : 1;
  int old = c.level;
  if (old >= 0 && g.live)
    for (int j = 0; j < (1 << (2 * old)); j++)
      g.records[c.ids[j]].death = birth + .25f;
  for (int y = 0; y < nd; y++)
    for (int x = 0; x < nd; x++) {
      Record r{};
      int b0 = step * y * 4 + step * x;
      float w = 0, cs[3] = {}, raw[3] = {};
      for (int u = 0; u < step; u++)
        for (int v = 0; v < step; v++) {
          const auto &f = g.f[16 * m + b0 + u * 4 + v];
          if (!f.w)
            continue;
          float ww = 1;
          for (int i = 0; i < 3;
               i++) { // feathered top observations; retain paint edges if
                      // geometry displaced the secondary views.
            float soft = f.colors[i] / f.w;
            float wb = min(.82f, .38f + abs(f.winner[i] - soft) / 95.f);
            float local =
                c.bw ? clampf(c.base[i] / c.bw - f.winbase[i], -23, 23) : 0;
            if (g.v2)
              wb = clampf(.64f + abs(f.winner[i] - soft) / 100.f, .64f, .94f);
            if (g.v2 && g.live && c.winnerCam == 0 && c.range < 28) {
              wb = .98f;
              local = 0;
            }
            cs[i] += ww * (wb * (f.winner[i] + (g.v2 ? .48f : .82f) * local) +
                           (1 - wb) * soft);
            raw[i] += ww * (wb * f.winraw[i] + (1 - wb) * f.raw[i] / f.w);
          }
          w += ww;
        }
      if (!w)
        continue;
      r.macro = m;
      r.bit = (y * step + step / 2) * 4 + (x * step + step / 2);
      r.x = g.x + S * (m % M) + (x + .5f) * pitch;
      r.y = g.y + S * (m / M) + (y + .5f) * pitch;
      r.nx = c.nx;
      r.ny = c.ny;
      r.z = c.z - c.nx * ((x + .5f) * pitch - S * .5) -
            c.ny * ((y + .5f) * pitch - S * .5);
      r.sz = pitch * .80f;
      r.sy = r.sz;
      if (g.v2) {
        float kernel = pitch * (.66f + .06f * border);
        r.sz = hypot(kernel, c.major);
        r.sy = hypot(kernel, c.minor);
        r.theta = c.axis;
        r.alpha =
            min((c.inferred ? .78f : .90f) * (.44f + .56f * border),
                -expm1f(-4.6f * pitch * pitch / (6.2831853f * r.sz * r.sy)));
      }
      r.t = birth;
      r.level = level;
      r.death = INF;
      for (int i = 0; i < 3; i++) {
        r.rgb[i] = clampf(cs[i] / w + .5, 0, 255);
        r.raw[i] = clampf(raw[i] / w + .5, 0, 255);
      }
      c.ids[y * nd + x] = (int)g.records.size();
      g.records.push_back(r);
    }
  c.level = level;
}
extern "C" void see(void *ptr, int cam, float tm, const double *E,
                    const double *row, const double *intr, int W, int H,
                    const uint8_t *im, int w, int h, const uint8_t *mask,
                    const float *shade, const float *field) {
  Tile &g = *(Tile *)ptr;
  double ox = E[0], oy = E[1], cs = E[2], sn = E[3], ch = E[4];
  float side = g.v2 ? (cam == 0 ? 20.4f : 30.f) : 20.f;
  float xmin = ox - side, xmax = ox + side, ymin = oy - side, ymax = oy + side;
  if (g.v2 && cam == 0) {
    xmin = ymin = 1e8;
    xmax = ymax = -1e8;
    for (float L : {-2.f, 82.f})
      for (float V : {-side, side}) {
        float X = ox + L * cs - V * sn, Y = oy + L * sn + V * cs;
        xmin = min(xmin, X);
        xmax = max(xmax, X);
        ymin = min(ymin, Y);
        ymax = max(ymax, Y);
      }
  }
  int x0 = max(0, (int)floor((xmin - g.x) / S)),
      y0 = max(0, (int)floor((ymin - g.y) / S)),
      x1 = min(M, (int)ceil((xmax - g.x) / S)),
      y1 = min(M, (int)ceil((ymax - g.y) / S));
  if (x0 >= x1 || y0 >= y1)
    return;
  int width = x1 - x0 + 1;
  int planeN = width * (y1 - y0 + 1);
  vector<View> grid(3 * planeN);
  for (int y = y0; y <= y1; y++)
    for (int x = x0; x <= x1; x++) {
      float high = field ? field[min(y, M - 1) * M + min(x, M - 1)] : 0;
      float lateral = abs(-sn * (g.x + x * S - ox) + cs * (g.y + y * S - oy));
      high *= clampf((lateral - 2.8f) / .8f, 0, 1);
      for (int hp = 0; hp < 3; hp++)
        grid[hp * planeN + (y - y0) * width + x - x0] =
            project(g.x + x * S, g.y + y * S,
                    high + (hp == 1   ? .32
                            : hp == 2 ? .84
                                      : 0),
                    row, intr, W, H);
    }
  for (int y = y0; y < y1; y++)
    for (int x = x0; x < x1; x++) {
      int m = y * M + x;
      float wx = g.x + (x + .5) * S, wy = g.y + (y + .5) * S, dx = wx - ox,
            dy = wy - oy, dist = hypot(dx, dy), lat = abs(-sn * dx + cs * dy),
            lon = dx * cs + dy * sn;
      if ((!g.v2 &&
           (dist > 19.8 || lat > 10.8 || ch / max(.9f, dist) < .057f)) ||
          (g.v2 && (lat > 20 || dist > (cam == 0 ? 82. : 30.) ||
                    (cam == 0 && lon < 0))) ||
          (lat < 1.05 && lon > -.98 && lon < 3.75) || dist < .9)
        continue;
      auto p = &grid[(y - y0) * width + x - x0];
      View A = p[0], B = p[1], C = p[width], D = p[width + 1];
      if (!A.ok || !B.ok || !C.ok || !D.ok)
        continue;
      float uc = (A.u + B.u + C.u + D.u) * .25f / W,
            vc = (A.v + B.v + C.v + D.v) * .25f / H;
      if (uc < -.015 || uc > 1.015 || vc < -.015 || vc > 1.015)
        continue;
      uint16_t seen = 0;
      float colors[16][3], raw[16][3], ave[3] = {}, ls[4] = {}, lsw[4] = {},
                                       sem = 0, known = 0;
      for (int j = 0; j < 4; j++)
        for (int i = 0; i < 4; i++) {
          int bit = j * 4 + i;
          float u = (i + .5f) * .25f, v = (j + .5f) * .25f;
          float U = ((1 - v) * ((1 - u) * A.u + u * B.u) +
                     v * ((1 - u) * C.u + u * D.u)) /
                    W,
                V = ((1 - v) * ((1 - u) * A.v + u * B.v) +
                     v * ((1 - u) * C.v + u * D.v)) /
                    H;
          float xx = U * w - .5f, yy = V * h - .5f;
          int ix = floor(xx), iy = floor(yy);
          if (ix < 0 || iy < 0 || ix >= w - 1 || iy >= h - 1)
            continue;
          int ij = iy * w + ix, ma = mask[ij];
          if (!(ma & 1)) {
            g.masked++;
            continue;
          }
          seen |= 1 << bit;
          sem += (ma & 2) != 0;
          known += (ma & 4) != 0;
          float a = xx - ix, b = yy - iy;
          int lu = max(0, min(64, (int)(U * 65))),
              lv = max(0, min(48, (int)(V * 49)));
          for (int c = 0; c < 3; c++) {
            int ci = 2 - c;
            const auto *ip = im + ij * 3 + ci;
            float q = (1 - b) * ((1 - a) * ip[0] + a * ip[3]) +
                      b * ((1 - a) * ip[w * 3] + a * ip[w * 3 + 3]);
            raw[bit][c] = q;
            colors[bit][c] = clampf(q * shade[(lv * 65 + lu) * 3 + c], 0, 255);
            ave[c] += colors[bit][c];
          }
          int k = (j / 2) * 2 + i / 2;
          ls[k] += colors[bit][1];
          lsw[k]++;
        }
      if (!seen)
        continue;
      float readout = E[5];
      g.visits.push_back({tm + (A.v / H - .5f) * readout,
                          (B.v - A.v) * readout / H, (C.v - A.v) * readout / H,
                          (D.v - C.v - B.v + A.v) * readout / H, m, seen,
                          (int)E[6]});
      if (g.live && E[7] != 0)
        continue;
      auto &c = g.C[m];
      if (!c.seen)
        g.changed.push_back(m);
      c.seen |= seen;
      int n = bits(seen);
      if (n < 10)
        continue;
      for (auto &v : ave)
        v /= n;
      for (int z = 0; z < 4; z++)
        ls[z] = lsw[z] ? ls[z] / lsw[z] - ave[1] : 0;
      float r = ave[0], gr = ave[1], b = ave[2];
      bool grass = gr - r > 6 && gr > b * 1.04,
           bright = max(r, max(b, gr)) > 42,
           neutral = max(r, max(b, gr)) - min(r, min(b, gr)) < 60;
      bool stripe = (r > b * 1.17 && gr > b * 1.1 && abs(r - gr) < 45);
      bool raised = field && field[m] > .045;
      bool outer = lat > 5.4;
      float height =
                std::max(.45, row[8 * 16 + 11] - (A.z + B.z + C.z + D.z) * .25),
            cd = hypot(wx - row[8 * 16 + 9], wy - row[8 * 16 + 10]);
      // Ground-to-sensor differential, in decoded pixels per ENU metre. The
      // eigenvectors of J^T J are the two resolved directions on the road.
      float ux = (B.u + D.u - A.u - C.u) / (2 * S) * w / W,
            uy = (C.u + D.u - A.u - B.u) / (2 * S) * w / W;
      float vx = (B.v + D.v - A.v - C.v) / (2 * S) * h / H,
            vy = (C.v + D.v - A.v - B.v) / (2 * S) * h / H;
      float aa = ux * ux + vx * vx, bb = ux * uy + vx * vy,
            cc = uy * uy + vy * vy;
      float eig = .5f * (aa + cc + hypot(aa - cc, 2 * bb)),
            jac = abs(ux * vy - uy * vx);
      float across = 1 / sqrt(max(.001f, eig)),
            along = sqrt(eig) / max(.001f, jac);
      float axis = .5f * atan2(2 * bb, aa - cc) + 1.57079633f;
      float rot =
          g.v2 ? E[8] * (.34f + .66f * clampf((cd - 20) / 22, 0, 1)) +
                     clampf((cd - 23) / 28, 0, 1) * E[9] * .55f /
                         std::sqrt(std::max(1, std::min(6, c.independent)))
               : 0;
      float sigma = hypot(.5f * along, rot * cd * cd / height);
      float shortSigma = .50f * across;
      bool possible = bright && (neutral || stripe || sem > .6 * n) &&
                      (lat < 8.4 || raised || sem > .6 * n) &&
                      (!grass || raised || sem > .6 * n);
      if (g.v2) {
        possible = bright && (neutral || stripe || sem > .6 * n || grass);
        if (dist > 31 && known > .8 * n && sem < .4 * n)
          possible = false;
      }
      bool core = lat < 2.25 && !grass && neutral && ave[1] > 53 && dist < 12;
      bool anchor = core || sem > .78 * n;
      // At the verge require geometric baseline and agreement in chroma,
      // intensity, and the four mean-centred subpatches. Never fill colour from
      // an unseen cell. Wider tangent patch and two above-ground hypotheses.
      // Uniform elevated objects cannot win on colour agreement alone.
      float sig[3][5];
      int offsets[5] = {0, -1, 1, -width, width};
      for (int h0 = 0; h0 < 3; h0++)
        for (int j = 0; j < 5; j++) {
          sig[h0][j] = -1000;
          int pos = (y - y0) * width + x - x0 + offsets[j];
          if (pos < 0 || pos + width + 1 >= planeN)
            continue;
          auto A0 = grid[h0 * planeN + pos],
               D0 = grid[h0 * planeN + pos + width + 1];
          float U = (A0.u + D0.u) * .5f / W, V = (A0.v + D0.v) * .5f / H;
          int tx = U * w, ty = V * h;
          if (!A0.ok || !D0.ok || tx < 0 || tx >= w || ty < 0 || ty >= h ||
              !(mask[ty * w + tx] & 1))
            continue;
          int sid = (max(0, min(48, int(V * 49))) * 65 +
                     max(0, min(64, int(U * 65)))) *
                    3;
          sig[h0][j] = im[(ty * w + tx) * 3 + 1] * shade[sid + 1];
        }
      int planarEvidence = 0, aboveEvidence = 0;
      int agree = 0, conflict = 0;
      bool strong = false, fresh = true;
      for (auto &f : c.refs) {
        if (f.cam < 0)
          continue;
        float base = hypot(ox - f.x, oy - f.y);
        bool independent = (f.cam != cam || base > (outer ? 1.65f : .85f));
        if (!independent) {
          fresh = false;
          continue;
        }
        if (base > 1.5 || f.cam != cam) {
          float errs[3] = {};
          int nums[3] = {};
          for (int hh = 0; hh < 3; hh++)
            for (int j = 0; j < 5; j++)
              if (f.sig[hh][j] >= 0 && sig[hh][j] >= 0) {
                errs[hh] += abs(f.sig[hh][j] - sig[hh][j]);
                nums[hh]++;
              }
          for (int hh = 0; hh < 3; hh++)
            errs[hh] = nums[hh] >= 3 ? errs[hh] / nums[hh] : 999;
          float alt = min(errs[1], errs[2]);
          if (alt < 999 && errs[0] < 999) {
            aboveEvidence += (alt + 4.5f < errs[0] * .71f);
            planarEvidence += (errs[0] + 2.5f < alt * .88f);
          }
        }
        float l = f.col[1] - gr,
              chr = max(abs((r - gr) - (f.col[0] - f.col[1])),
                        abs((b - gr) - (f.col[2] - f.col[1]))),
              tex = 0;
        for (int q = 0; q < 4; q++)
          tex += abs(ls[q] - f.lum[q]);
        bool ok = abs(l) < (outer ? 13 : 24) && chr < (outer ? 9 : 15) &&
                  tex * .25 < (outer ? 11 : 19);
        if (ok) {
          agree++;
          strong |= base > 2.2 || f.cam != cam;
        } else
          conflict++;
      }
      if (fresh || g.pkt % 4 == 0) {
        auto &ref = c.refs[(c.nref++) % 3];
        ref.cam = cam;
        ref.x = ox;
        ref.y = oy;
        ref.t = tm;
        memcpy(ref.sig, sig, sizeof(sig));
        memcpy(ref.col, ave, 12);
        memcpy(ref.lum, ls, 16);
      }
      if (agree && strong && hypot(c.lastX - ox, c.lastY - oy) > 2.f) {
        c.independent++;
        c.lastX = ox;
        c.lastY = oy;
      }
      // Distant reprojections hit the wrong material at a turn. Their vote
      // cannot outweigh a subsequent close measured road observation.
      if (g.v2 && dist < .62f * c.evidenceRange) {
        c.higher /= 4;
        c.plane /= 4;
        c.semno /= 4;
        c.semantic /= 4;
        c.no /= 4;
        c.evidenceRange = dist;
      }
      bool nearRoad = g.v2 && dist < 23 && !grass && sem > .83f * n &&
                      (agree || lat < 4.2f) && aboveEvidence <= 1;
      if (nearRoad) {
        c.blocked = false;
        c.negclose = 0;
      }
      if (g.v2 && c.level == 0 && c.range > 28 && dist < 24 &&
          c.range > 1.35f * dist &&
          ((known > .8 * n && sem < .15 * n && (aboveEvidence || !agree)) ||
           (aboveEvidence >= 2 && !agree))) {
        if (++c.negclose >= 3)
          c.blocked = true;
      }
      if (known >= n * .8f) {
        if (sem < .15f * n)
          c.semno++;
        if (sem > .72f * n)
          c.semantic++;
      }
      c.higher += aboveEvidence;
      c.plane += planarEvidence;
      bool veto = (c.higher > c.plane * 2 + 2 && lat > 2.75 && !raised) ||
                  (c.semno > 2 * c.semantic + 1 && lat > 2.8 && !raised);
      if (g.v2 && c.semno > 2 * c.semantic + 1 && sem < .5 * n &&
          c.plane > 2 * c.higher + 3 && (grass || lat > 4.5))
        veto = (c.higher > c.plane * 2 + 2 && !raised);
      if (nearRoad)
        veto = false;
      if (veto)
        possible = false;
      if (!possible || (!agree && conflict)) {
        c.no++;
      }
      if (!possible)
        continue;
      int neigh = 0;
      for (int v0 : {-M, -1, 1, M}) {
        int nb = m + v0;
        if (nb >= 0 && nb < M * M && g.C[nb].trusted && g.C[nb].yes >= 2)
          neigh++;
      }
      bool paintedRoad = (lat < 3.8 && dist < 17 && (neutral || stripe) &&
                          neigh >= 2 && !(known > .8 * n && sem < .15 * n));
      bool confirmed = (sem > .78 * n && !veto) ||
                       (core && (agree || neigh >= 2)) || paintedRoad ||
                       (agree && (!outer || strong) &&
                        (lat < 4.5 || raised || sem > .6 * n ||
                         planarEvidence || c.plane > c.higher + 1));
      if (g.v2) {
        bool insideFront = cam == 0 && lon >= 4 && lat < 4.3;
        bool broad = agree && (strong || c.plane > c.higher + 1) &&
                     (planarEvidence || c.plane > c.higher + 1 ||
                      (insideFront && lat < 2.4 && !grass && !aboveEvidence));
        bool roadFront =
            insideFront && dist < 28 && !grass && (neutral || stripe) &&
            (known < .5 * n || sem > .4 * n) && (agree || neigh >= 2);
        confirmed = confirmed || broad || roadFront;
        if (cam == 0 && dist <= 28 && lat < 5.2f && c.trusted && !grass &&
            (known < .5f * n || sem > .35f * n) && (neigh >= 2 || agree) &&
            aboveEvidence <= 1)
          confirmed = true;
        if (grass && !raised && sem < .6 * n)
          confirmed =
              agree && strong && c.independent >= 2 && c.plane > c.higher + 4;
        if (dist > 31)
          confirmed = (sem > .68 * n && !grass) ||
                      (agree && strong && !aboveEvidence &&
                       (planarEvidence || c.plane >= c.higher + 2 ||
                        (insideFront && !grass && conflict == 0)));
        if (cam == 0 && dist > 24) {
          // A far disc is allowed only if road continues along the same
          // bearing. Test a credible depth interval, not one road-coloured
          // pixel past the kerb. Maskless extrapolation also needs baseline.
          int semanticRoad = 0, asphalt = 0, validRay = 0;
          for (float frac : {.36f, .62f, .82f, 1.f,
                             1.f + min(.10f, 2 * sigma / max(10.f, cd))}) {
            auto Q =
                project(ox + dx * frac, oy + dy * frac, 0, row, intr, W, H);
            int xx = (int)(Q.u * w / W), yy = (int)(Q.v * h / H);
            if (!Q.ok || xx < 0 || yy < 0 || xx >= w || yy >= h)
              continue;
            int pos = yy * w + xx, ma = mask[pos];
            if (!(ma & 1))
              continue;
            validRay++;
            semanticRoad += (ma & 2) != 0;
            float R = im[3 * pos + 2], G = im[3 * pos + 1], B = im[3 * pos];
            asphalt += (G > 42 && G < 192 && R > G * .52f && R < G * 1.24f &&
                        B > G * .83f && B < G * 1.25f);
          }
          bool aperture =
              known > .8f * n
                  ? (semanticRoad >= 4 && sem > .83f * n)
                  : (asphalt >= 4 && abs(lat) < 8.f && c.independent >= 2 &&
                     c.higher <= c.plane + 1 && E[10] * dist < .62f);
          bool straightConsensus = lat < 5.5f && E[10] * dist < .35f &&
                                   c.independent >= 3 && c.higher < c.plane + 2;
          confirmed &= validRay == 5 && aperture &&
                       sigma < (straightConsensus ? 5.3f : 3.1f);
        }
        if (dist > 15 && lat > 6 && (grass || sem < .3 * n))
          confirmed &= agree && c.plane > c.higher + 3;
        float q = 1.f / (.024f + sigma + shortSigma);
        if (n == 16 && q > c.candidate && !grass && dist < 30 &&
            (sem > .5 * n || lat < 6 || c.plane > c.higher)) {
          c.candidate = q;
          c.candmask = seen;
          c.candz = (A.z + B.z + C.z + D.z) * .25f;
          c.candnx = clampf(-(B.z + D.z - A.z - C.z) / (2 * S), -.28, .28);
          c.candny = clampf(-(C.z + D.z - A.z - B.z) / (2 * S), -.28, .28);
          c.candMajor = sigma;
          c.candMinor = shortSigma;
          c.candAxis = axis;
          c.candLat = lat;
          c.candRange = dist;
          for (int b = 0; b < 16; b++)
            for (int k = 0; k < 3; k++)
              c.candidateRGB[b][k] = clampf(colors[b][k] + .5, 0, 255);
        }
      }
      if (!confirmed)
        continue;
      if (g.v2 && c.blocked) {
        if (sem > .85 * n && agree && aboveEvidence == 0) {
          if (--c.negclose <= 0)
            c.blocked = false;
        }
        if (c.blocked)
          continue;
      }
      if (g.v2 && agree)
        c.negclose = 0;
      c.yes++;
      c.trusted |= anchor || paintedRoad || agree >= 2 || strong ||
                   (g.v2 && cam == 0 && neigh >= 2);
      c.lat = min(c.lat, lat);
      if (outer && c.no > c.yes * 2 + 1)
        continue;
      float edge = min(min(uc, 1 - uc), min(vc, 1 - vc));
      float qual = ch / (max(1.f, dist) * max(1.f, dist)) *
                   clampf(edge * 18, .25, 1) *
                   clampf(intr[0] / 660., .65, 1.32);
      if (g.v2) {
        qual = clampf(edge * 25, .55, 1) / (.03f + sigma + shortSigma);
        if (cam == 0 && dist >= 6 && dist <= 23)
          qual *= 1.12f;
      }
      float wt = qual * qual * qual;
      float bw = clampf(sqrt(qual), .055f, .15f);
      if (c.bw > 0)
        bw /= 1 + powf((ave[1] - c.base[1] / c.bw) / 27.f, 2);
      for (int v = 0; v < 3; v++)
        c.base[v] += bw * ave[v];
      c.bw += bw;
      bool top = qual > c.best;
      if (top || !c.best) {
        c.winnerCam = cam;
        if (g.v2 && cam == 0 && dist < 30) {
          float mi = 255, ma = 0;
          for (int b = 0; b < 16; b++)
            if (seen & (1 << b)) {
              mi = min(mi, colors[b][1]);
              ma = max(ma, colors[b][1]);
            }
          for (float q : {-.65f, .65f})
            for (int k : {0, 1}) {
              int ix = uc * w + q * (k ? uy : ux),
                  iy = vc * h + q * (k ? vy : vx);
              if (ix >= 0 && iy >= 0 && ix < w && iy < h &&
                  (mask[iy * w + ix] & 1)) {
                float z = im[(iy * w + ix) * 3 + 1] *
                          shade[(min(48, max(0, int(vc * 49))) * 65 +
                                 min(64, max(0, int(uc * 65)))) *
                                    3 +
                                1];
                mi = min(mi, z);
                ma = max(ma, z);
              }
            }
          c.detail |= ma - mi > 17;
        }
        c.major = clampf(sigma, .002f, 9.f);
        c.minor = clampf(shortSigma, .001f, .18f);
        c.axis = axis;
        c.range = dist;
        c.z = (A.z + B.z + C.z + D.z) * .25f;
        c.nx = -(B.z + D.z - A.z - C.z) / (2 * S);
        c.ny = -(C.z + D.z - A.z - B.z) / (2 * S);
        c.nx = clampf(c.nx, -.28, .28);
        c.ny = clampf(c.ny, -.28, .28);
      }
      c.best = max(c.best, qual);
      for (int bit = 0; bit < 16; bit++)
        if (seen & (1 << bit)) {
          auto &f = g.f[m * 16 + bit];
          float ww = wt;
          if (f.w > 0 && !anchor && outer &&
              abs(colors[bit][1] - f.colors[1] / f.w) > 25)
            ww *= .14f;
          f.w += ww;
          for (int z = 0; z < 3; z++) {
            f.colors[z] += ww * colors[bit][z];
            f.raw[z] += ww * raw[bit][z];
            if (qual > f.best) {
              f.winbase[z] = ave[z] + .5f;
              f.winner[z] = colors[bit][z] + .5f;
              f.winraw[z] = raw[bit][z] + .5f;
            }
          }
          f.best = max(f.best, qual);
        }
      g.samples += n;
      g.raises += raised;
    }
}
// Repair a bounded enclosed gap only if it already has a complete, unmasked
// colour observation and supported road on all primary sides. No future ring.
bool closeHole(Tile &g, int m) {
  auto &c = g.C[m];
  if (!g.v2 || c.trusted || c.blocked || c.higher > c.plane + 2 ||
      c.semno > 2 * c.semantic + 2)
    return false;
  // First require a closed rim on every cardinal side, not a mostly full
  // rectangular neighbourhood. An angular sector open to the verge fails.
  int sides = 0, diags = 0, ids[8], n = 0;
  for (int y = -1; y <= 1; y++)
    for (int x = -1; x <= 1; x++) {
      if (!x && !y)
        continue;
      for (int j = 1; j <= 4; j++) {
        int xx = m % M + j * x, yy = m / M + j * y;
        if (xx < 0 || yy < 0 || xx >= M || yy >= M)
          break;
        int id = yy * M + xx;
        auto &nb = g.C[id];
        if (nb.trusted && !nb.inferred && nb.yes) {
          ids[n++] = id;
          if (x && y)
            diags++;
          else
            sides++;
          break;
        }
      }
    }
  if (sides < 4 || diags < 3)
    return false;
  bool sampled = c.candidate && c.candmask == 65535;
  float sum = 0, col[3] = {}, lo = 1e9, hi = -1e9, lowY = 1e9, highY = -1e9;
  if (!sampled) {
    c.candz = c.candnx = c.candny = 0;
    for (int k = 0; k < n; k++) {
      auto &r = g.C[ids[k]];
      float dx = (m % M - ids[k] % M) * S, dy = (m / M - ids[k] / M) * S;
      float zz = r.z - dx * r.nx - dy * r.ny, w = 1 / hypot(dx, dy);
      if (!r.bw || r.range > 23)
        return false;
      lo = min(lo, zz);
      hi = max(hi, zz);
      lowY = min(lowY, r.base[1] / r.bw);
      highY = max(highY, r.base[1] / r.bw);
      sum += w;
      c.candz += w * zz;
      c.candnx += w * r.nx;
      c.candny += w * r.ny;
      c.candLat = min(c.candLat, r.lat);
      for (int j = 0; j < 3; j++)
        col[j] += w * r.base[j] / r.bw;
    }
    if (hi - lo > .075f || highY - lowY > 24.f) {
      c.candz = c.candnx = c.candny = 0;
      return false;
    }
    c.candz /= sum;
    c.candnx /= sum;
    c.candny /= sum;
    c.candMajor = .13f;
    c.candMinor = .065f;
    c.candRange = 18;
    c.candidate = 1.;
    for (int b = 0; b < 16; b++)
      for (int j = 0; j < 3; j++)
        c.candidateRGB[b][j] = clampf(col[j] / sum, 0, 255);
  }
  c.yes++;
  c.trusted = c.inferred = true;
  c.best = c.candidate;
  c.lat = c.candLat;
  c.major = c.candMajor;
  c.minor = c.candMinor;
  c.range = c.candRange;
  c.axis = c.candAxis;
  c.z = c.candz;
  c.nx = c.candnx;
  c.ny = c.candny;
  for (int i = 0; i < 16; i++) {
    auto &f = g.f[m * 16 + i];
    f.best = c.best;
    f.w = 1;
    for (int k = 0; k < 3; k++) {
      int a = c.candidateRGB[i][k];
      f.colors[k] = f.raw[k] = f.winraw[k] = f.winner[k] = f.winbase[k] = a;
    }
  }
  return true;
}
int v2level(const Cell &c) {
  if (c.lat <= 6.1f && c.minor < .039f &&
      ((c.winnerCam == 0 && c.range < 26.f) ||
       (c.major < .076f && c.range < 20.f)))
    return 2;
  if (c.lat <= 10.f && c.minor < .1f && c.major < .43f)
    return 1;
  return 0;
}
extern "C" void packet(void *ptr, float viewtime, float birth, int packet) {
  auto &g = *(Tile *)ptr;
  vector<int> rim;
  for (int m : g.changed) {
    auto &c = g.C[m];
    if (g.v2 && g.live && c.blocked && c.level >= 0 && birth < INF) {
      for (int j = 0; j < (1 << (2 * c.level)); j++)
        g.records[c.ids[j]].death =
            min(g.records[c.ids[j]].death, birth + .25f);
      c.level = -1;
    }
    if (g.live)
      closeHole(g, m);
    c.seen = 0;
    if (!g.live || !c.yes || !c.trusted || c.blocked || birth >= INF)
      continue;
    bool full = true;
    for (int i = 0; i < 16; i++)
      full &= g.f[m * 16 + i].w > 0;
    if (!full)
      continue;
    float minq = c.best;
    for (int b = 0; b < 16; b++)
      minq = min(minq, g.f[16 * m + b].best);
    int wanted = minq > .020f && c.lat <= 6.05 ? 2 : minq > .007 ? 1 : 0;
    if (g.v2)
      wanted = v2level(c);
    if (c.level < 0) {
      emit(g, m, wanted, birth);
      if (g.v2 && !c.inferred)
        rim.push_back(m);
    } else if (wanted > c.level && birth - g.records[c.ids[0]].t > .065 &&
               c.yes >= 2)
      emit(g, m, g.v2 ? wanted : c.level + 1, birth);
    else if (g.v2 && c.level == 0 && c.major > .02f &&
             g.records[c.ids[0]].sz > 2.8f * hypot(.16f, c.major) && c.yes >= 3)
      emit(g, m, 0, birth);
    else if (g.v2 && c.level == 2 && c.polished < 2 && c.detail &&
             c.winnerCam == 0 && c.yes >= 2) {
      auto &old = g.records[c.ids[0]];
      float drift = 0;
      for (int j = 0; j < 16; j++)
        drift += abs((int)g.f[16 * m + j].winner[1] -
                     (int)g.records[c.ids[j]].rgb[1]);
      bool sharper =
          old.sz >
          1.62f * hypot(.041f, c.major); // the *along* refinement, without
                                         // another lateral subdivision
      if (birth - old.t > .27f &&
          (sharper || (drift / 16 > 12 && c.range < 18.f))) {
        emit(g, m, 2, birth);
        c.polished++;
      }
    }
  }
  if (g.v2 && g.live && birth < INF)
    for (int m : rim)
      for (int dy = -2; dy <= 2; dy++)
        for (int dx = -2; dx <= 2; dx++) {
          int x = m % M + dx, y = m / M + dy;
          if (x < 0 || y < 0 || x >= M || y >= M)
            continue;
          int nb = y * M + x;
          if (g.C[nb].level < 0 && closeHole(g, nb))
            emit(g, nb, v2level(g.C[nb]), birth);
        }
  g.changed.clear();
  g.pkt++;
}
extern "C" long finish(void *ptr, const char *file, const char *side,
                       int *counts) {
  auto &g = *(Tile *)ptr;
  if (!g.live) {
    for (int m = 0; m < M * M; m++) {
      auto &c = g.C[m];
      closeHole(g, m);
      if (!c.trusted || !c.yes ||
          (c.lat > 3.5 && (c.yes * 2 < c.no || c.higher > 2 * c.plane + 3 ||
                           (c.semno > 2 * c.semantic + 1 &&
                            (!g.v2 || c.plane <= 2 * c.higher + 3)))))
        continue;
      bool full = true;
      for (int b = 0; b < 16; b++)
        full &= g.f[m * 16 + b].w > 0;
      if (!full)
        continue;
      int lv = c.lat <= 6.05 && c.best > .014 ? 2 : 1;
      if (g.v2)
        lv = v2level(c);
      emit(g, m, lv, 0);
    }
  }
  vector<vector<int>> by(M * M);
  vector<vector<Run>> runs(g.records.size());
  vector<int> lp(g.records.size(), -100);
  for (int i = 0; i < (int)g.records.size(); i++)
    by[g.records[i].macro].push_back(i);
  for (auto &v : g.visits) {
    for (int id : by[v.cell]) {
      auto &r = g.records[id];
      int nd = 1 << r.level, step = 4 / nd;
      float u = ((r.bit % 4) / step + .5f) / nd,
            vv = ((r.bit / 4) / step + .5f) / nd;
      float tv = v.t + v.dt_x * u + v.dt_y * vv + v.dt_xy * u * vv;
      if (tv < r.t || tv >= r.death || !(v.mask & (1 << r.bit)))
        continue;
      auto &z = runs[id];
      if (z.empty() || (tv - z.back().b >= .25f &&
                        (v.packet > lp[id] + 1 || tv - z.back().b > .42f)))
        z.push_back({tv, tv});
      else {
        z.back().a = std::min(z.back().a, tv);
        z.back().b = std::max(z.back().b, tv);
      }
      lp[id] = v.packet;
    }
  }
  FILE *fp = fopen(file, "wb");
  FILE *raw = fopen(side, "wb");
  FILE *extras = g.live
                     ? nullptr
                     : fopen((std::string(side) + ".extra-runs").c_str(), "wb");
  long written = 0;
  for (int id = 0; id < (int)g.records.size(); id++) {
    auto r = g.records[id];
    auto z = runs[id];
    if (z.empty() || z.front().a >= r.t + .25f)
      z.insert(z.begin(), {r.t, r.t});
    else
      z.front().a = r.t;
    if (!g.live && z.size() > 3) {
      counts[5]++;
      for (size_t j = 2; j + 1 < z.size(); ++j) {
        uint32_t idx = (uint32_t)written;
        fwrite(&idx, 4, 1, extras);
        fwrite(&z[j], sizeof(Run), 1, extras);
      }
      z = {z[0], z[1], z.back()};
    }
    for (unsigned h = 0; h < z.size(); h += 3) {
      float times[7] = {h ? z[h].a : r.t,
                        h + 3 < z.size() ? z[h + 3].a : r.death,
                        z[h].b,
                        INF,
                        INF,
                        INF,
                        INF};
      for (unsigned j = 1; j < 3 && h + j < z.size(); j++) {
        times[1 + 2 * j] = z[h + j].a;
        times[2 + 2 * j] = z[h + j].b;
      }
      float l = 1 / sqrt(1 + r.nx * r.nx + r.ny * r.ny), nx = r.nx * l,
            ny = r.ny * l, nz = l, w = sqrt((1 + nz) / 2);
      float sn = sin(r.theta * .5f), cs = cos(r.theta * .5f),
            qx = -ny / (2 * w), qy = nx / (2 * w);
      float a[24] = {r.x,
                     r.y,
                     r.z,
                     nx,
                     ny,
                     nz,
                     (r.rgb[0] / 255.f - .5f) / .28209479f,
                     (r.rgb[1] / 255.f - .5f) / .28209479f,
                     (r.rgb[2] / 255.f - .5f) / .28209479f,
                     logf(r.alpha / (1 - r.alpha)),
                     logf(r.sz),
                     logf(r.sy),
                     logf(.008f),
                     w * cs,
                     qx * cs + qy * sn,
                     qy * cs - qx * sn,
                     w * sn};
      memcpy(a + 17, times, sizeof(times));
      fwrite(a, sizeof(a), 1, fp);
      fwrite(r.raw, 3, 1, raw);
      counts[r.level]++;
      if (g.v2) {
        counts[6] += g.C[r.macro].inferred;
        counts[7] += r.sz > 2 * r.sy;
      }
      counts[3] += (h != 0);
      counts[4] += times[1] < INF;
      written++;
    }
  }
  fclose(fp);
  fclose(raw);
  if (extras)
    fclose(extras);
  return written;
}
