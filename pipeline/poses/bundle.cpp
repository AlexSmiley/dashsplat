#include <array>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <fstream>
#include <iostream>
#include <vector>
using std::array;
using std::vector;
// Frozen camera: fx,fy,cx,cy,k1..k4, R_ego_cam row-major, lever arm.
struct Obs {
  array<double, 20> c;
  double a, x, y, cy, sy, u, v;
  template <class T> bool eval(const T *f, const T *g, const T *p, T *r) const {
    T q[6];
    for (int j = 0; j < 6; j++)
      q[j] = f[j] * (1. - a) + g[j] * a;
    T xx = p[0] - T(x) - q[3], yy = p[1] - T(y) - q[4],
      d[3] = {T(cy) * xx + T(sy) * yy, -T(sy) * xx + T(cy) * yy, p[2] - q[5]},
      aa[3] = {-q[0], -q[1], -q[2]}, e[3], z[3];
    ceres::AngleAxisRotatePoint(aa, d, e);
    for (int j = 0; j < 3; j++)
      e[j] -= T(c[17 + j]);
    for (int j = 0; j < 3; j++)
      z[j] = T(c[8 + j]) * e[0] + T(c[11 + j]) * e[1] + T(c[14 + j]) * e[2];
    T rho = sqrt(z[0] * z[0] + z[1] * z[1] + T(1e-16)), th = atan2(rho, z[2]),
      h = th * th,
      sc = th * (T(1) + h * (c[4] + h * (c[5] + h * (c[6] + h * c[7])))) / rho;
    r[0] = c[0] * z[0] * sc + c[2] - u;
    r[1] = c[1] * z[1] * sc + c[3] - v;
    return true;
  }
  template <class T>
  bool operator()(const T *f, const T *g, const T *p, T *r) const {
    return eval(f, g, p, r);
  }
};
struct Same {
  Obs o;
  template <class T> bool operator()(const T *f, const T *p, T *r) const {
    return o.eval(f, f, p, r);
  }
};
struct Prior {
  double x, y, ang, sxy, sz;
  template <class T> bool operator()(const T *f, T *r) const {
    r[0] = f[0] / .055;
    r[1] = f[1] / .055;
    r[2] = f[2] / ang;
    r[3] = (f[3] - x) / sxy;
    r[4] = (f[4] - y) / sxy;
    r[5] = f[5] / sz;
    return true;
  }
};
struct RoadState {
  array<double, 12> p;
  template <class T> bool operator()(const T *f, T *r) const {
    for (int j = 0; j < 6; j++)
      r[j] = (f[j] - T(p[j])) / T(p[j + 6]);
    return true;
  }
};
// Each original patch centre has its own exposure, lever arm and local plane.
// Dynamic parameter blocks deduplicate the two bracketing knots at each end.
struct Patch {
  array<array<double, 16>, 2> sides;
  array<int, 4> blocks;
  double inv_noise;
  template <class T> bool operator()(T const *const *par, T *res) const {
    res[0] = T(0);
    res[1] = T(0);
    for (int end = 0; end < 2; end++) {
      auto &d = sides[end];
      T f[6];
      for (int j = 0; j < 6; j++)
        f[j] = (1. - d[0]) * par[blocks[end * 2]][j] +
               d[0] * par[blocks[end * 2 + 1]][j];
      T ray[3] = {T(d[6]), T(d[7]), T(d[8])},
        lever[3] = {T(d[9]), T(d[10]), T(d[11])}, u[3], o[3];
      ceres::AngleAxisRotatePoint(f, ray, u);
      ceres::AngleAxisRotatePoint(f, lever, o);
      T ux = d[3] * u[0] - d[4] * u[1], uy = d[4] * u[0] + d[3] * u[1],
        ox = f[3] + d[1] + d[3] * o[0] - d[4] * o[1],
        oy = f[4] + d[2] + d[4] * o[0] + d[3] * o[1], oz = f[5] + o[2];
      T dep = -(d[12] * ox + d[13] * oy + d[14] * oz + d[15]) /
              (d[12] * ux + d[13] * uy + d[14] * u[2]);
      T sign(end ? -inv_noise : inv_noise);
      res[0] += sign * (ox + dep * ux);
      res[1] += sign * (oy + dep * uy);
    }
    return true;
  }
};
struct Odometry {
  double dx, dy, y, d, dt;
  template <class T> bool operator()(const T *a, const T *b, T *r) const {
    T h = y + (a[2] + b[2]) * .5, x = dx + b[3] - a[3], yy = dy + b[4] - a[4];
    r[0] = (cos(h) * x + sin(h) * yy - d) / (.025 + .022 * d);
    r[1] = (-sin(h) * x + cos(h) * yy) / (.018 + .012 * d);
    r[2] = (b[2] - a[2]) / (.0016 + .0007 * d);
    r[3] = (b[5] - a[5] + d * (a[1] + b[1]) * .5) / (.023 + .025 * d);
    return true;
  }
};
struct Smooth {
  double dt1, dt2;
  template <class T>
  bool operator()(const T *a, const T *b, const T *c, T *r) const {
    double w = dt1 / (dt1 + dt2), d = std::max(.04, std::min(.28, dt1 + dt2));
    for (int j = 0; j < 6; j++) {
      double s = j < 2 ? .0013 + .0018 * d : (j == 2 ? .0014 : .010 + .055 * d);
      r[j] = (b[j] - (1 - w) * a[j] - w * c[j]) / s;
    }
    return true;
  }
};
template <class A> void rd(std::ifstream &i, A &x) {
  i.read(reinterpret_cast<char *>(&x), sizeof(x));
}
template <class A> void rvec(std::ifstream &i, vector<A> &x) {
  i.read(reinterpret_cast<char *>(x.data()), sizeof(A) * x.size());
}
int main(int argc, char **argv) {
  std::ifstream in(argv[1], std::ios::binary);
  array<int, 7> h;
  rd(in, h);
  auto [nf, np, no, ne, ns, it, threads] = h;
  double loss;
  rd(in, loss);
  array<array<double, 20>, 6> cams;
  rd(in, cams);
  vector<array<double, 6>> frame(nf);
  vector<array<double, 5>> pr(nf);
  vector<array<double, 3>> pts(np);
  rvec(in, frame);
  rvec(in, pr);
  rvec(in, pts);
  ceres::Problem problem;
  auto order = std::make_shared<ceres::ParameterBlockOrdering>();
  for (int i = 0; i < nf; i++) {
    auto v = pr[i];
    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<Prior, 6, 6>(
                                 new Prior{v[0], v[1], v[2], v[3], v[4]}),
                             new ceres::HuberLoss(2), frame[i].data());
    order->AddElementToGroup(frame[i].data(), 1);
  }
  for (auto &p : pts)
    order->AddElementToGroup(p.data(), 0);
  for (int i = 0; i < ne; i++) {
    array<double, 7> z;
    rd(in, z);
    auto [a, b, x, y, yaw, dd, dt] = z;
    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<Odometry, 4, 6, 6>(
                                 new Odometry{x, y, yaw, dd, dt}),
                             new ceres::HuberLoss(2), frame[a].data(),
                             frame[b].data());
  }
  for (int i = 0; i < ns; i++) {
    array<double, 5> z;
    rd(in, z);
    auto [a, b, c, u, v] = z;
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<Smooth, 6, 6, 6, 6>(new Smooth{u, v}),
        new ceres::HuberLoss(2), frame[a].data(), frame[b].data(),
        frame[c].data());
  }
  for (int i = 0; i < no; i++) {
    array<double, 10> z;
    rd(in, z);
    auto [a, b, id, cam, alpha, x, y, yaw, u, v] = z;
    Obs o{cams[cam], alpha, x, y, cos(yaw), sin(yaw), u, v};
    if (a == b)
      problem.AddResidualBlock(
          new ceres::AutoDiffCostFunction<Same, 2, 6, 3>(new Same{o}),
          new ceres::CauchyLoss(loss), frame[a].data(), pts[id].data());
    else
      problem.AddResidualBlock(
          new ceres::AutoDiffCostFunction<Obs, 2, 6, 6, 3>(new Obs{o}),
          new ceres::CauchyLoss(loss), frame[a].data(), frame[b].data(),
          pts[id].data());
  }
  // Optional direct-road posterior. Keep exactly the same feature observations,
  // robust losses, odometry factors and gauge priors as the v1 solve.
  if (argc >= 4) {
    std::ifstream road(argv[3], std::ios::binary);
    for (int i = 0; i < nf; ++i) {
      array<double, 12> z;
      rd(road, z);
      problem.AddResidualBlock(
          new ceres::AutoDiffCostFunction<RoadState, 6, 6>(new RoadState{z}),
          nullptr, frame[i].data());
    }
    if (!road) {
      std::cerr << "bad road constraints";
      return 2;
    }
  }
  if (argc == 5) {
    std::ifstream rays(argv[4], std::ios::binary);
    int count;
    rd(rays, count);
    for (int i = 0; i < count; i++) {
      array<int, 4> knots;
      rd(rays, knots);
      auto *patch = new Patch;
      rd(rays, patch->sides);
      rd(rays, patch->inv_noise);
      vector<double *> args;
      auto *cost = new ceres::DynamicAutoDiffCostFunction<Patch, 8>(patch);
      for (int j = 0; j < 4; j++) {
        double *p = frame[knots[j]].data();
        auto pos = std::find(args.begin(), args.end(), p);
        patch->blocks[j] = std::distance(args.begin(), pos);
        if (pos == args.end()) {
          args.push_back(p);
          cost->AddParameterBlock(6);
        }
      }
      cost->SetNumResiduals(2);
      problem.AddResidualBlock(cost, new ceres::CauchyLoss(2.2), args);
    }
    if (!rays) {
      std::cerr << "truncated patch stream";
      return 2;
    }
  }
  if (!in) {
    std::cerr << "bad binary problem";
    return 2;
  }
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.linear_solver_ordering = order;
  options.num_threads = threads;
  options.max_num_iterations = it;
  options.function_tolerance = 2e-5;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << std::endl;
  if (!summary.IsSolutionUsable())
    return 3;
  std::ofstream out(argv[2], std::ios::binary);
  out.write((char *)frame.data(), sizeof(frame[0]) * nf);
  out.write((char *)pts.data(), sizeof(pts[0]) * np);
}
