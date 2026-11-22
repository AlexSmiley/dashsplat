// CPU sparse-Schur bundle adjustment. The residual is in native image pixels.
#include <array>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
using std::array;
using std::vector;
template <int N> struct Prior {
  array<double, N> mu, s;
  template <class T> bool operator()(const T *x, T *r) const {
    for (int j = 0; j < N; j++)
      r[j] = (x[j] - mu[j]) / s[j];
    return true;
  }
};
struct Ratio {
  double s;
  template <class T> bool operator()(const T *x, T *r) const {
    r[0] = (x[0] - x[1]) / s;
    return true;
  }
};
struct Motion {
  double dx, dy, yaw, dist, sx, sy;
  template <class T> bool operator()(const T *a, const T *b, T *r) const {
    T h = T(yaw) + (a[2] + b[2]) / T(2);
    T x = T(dx) + b[3] - a[3], y = T(dy) + b[4] - a[4];
    r[0] = (cos(h) * x + sin(h) * y - T(dist)) / T(sx);
    r[1] = (-sin(h) * x + cos(h) * y) / T(sy);
    return true;
  }
};
struct Observation {
  array<double, 9> B;
  array<double, 6> base;
  double dx, dy, dt, h, u, v;
  template <class T>
  bool operator()(const T *cam, const T *frame, const T *pt, T *r) const {
    // Per-camera exposure relative to front clock, seconds; readout about
    // mid-row.
    T tau = T(dt) + cam[14] + cam[15] * T(v / h - .5);
    T yaw = T(base[2]) + tau * T(base[5]);
    T co = cos(yaw), si = sin(yaw);
    T wx = pt[0] - T(base[0]) - frame[3] - tau * T(dx),
      wy = pt[1] - T(base[1]) - frame[4] - tau * T(dy);
    T a[3] = {co * wx + si * wy, -si * wx + co * wy, pt[2] - frame[5]},
      rv[3] = {-frame[0], -frame[1], -frame[2]}, b[3];
    ceres::AngleAxisRotatePoint(rv, a, b);
    for (int i = 0; i < 3; i++)
      b[i] -= cam[11 + i];
    T c[3];
    for (int i = 0; i < 3; i++)
      c[i] = T(B[i]) * b[0] + T(B[3 + i]) * b[1] + T(B[6 + i]) * b[2];
    T rc[3] = {-cam[8], -cam[9], -cam[10]}, p[3];
    ceres::AngleAxisRotatePoint(rc, c, p);
    T rho = sqrt(p[0] * p[0] + p[1] * p[1] + T(1e-16)),
      theta = atan2(rho, p[2]), t2 = theta * theta;
    T s = theta *
          (T(1) + t2 * (cam[4] + t2 * (cam[5] + t2 * (cam[6] + t2 * cam[7])))) /
          rho;
    r[0] = cam[0] * s * p[0] + cam[2] - T(u);
    r[1] = cam[1] * s * p[1] + cam[3] - T(v);
    return true;
  }
};
template <size_t N> void read(std::ifstream &in, array<double, N> &d) {
  for (auto &z : d)
    in >> z;
}
template <size_t N>
void write(std::ofstream &out, const vector<array<double, N>> &d) {
  for (auto &z : d)
    out.write((const char *)z.data(), N * sizeof(double));
}
int main(int argc, char **argv) {
  std::ifstream in(argv[1]);
  int nc, nf, np, no, it, mode, threads, ne;
  double loss;
  in >> nc >> nf >> np >> no >> it >> loss >> mode >> threads >> ne;
  vector<array<double, 16>> cameras(nc);
  vector<array<double, 9>> B(nc);
  vector<double> h(nc);
  vector<array<double, 6>> base(nf), frame(nf);
  vector<array<double, 3>> pts(np);
  ceres::Problem problem;
  for (int c = 0; c < nc; c++) {
    array<double, 16> mu, s;
    read(in, cameras[c]);
    read(in, B[c]);
    read(in, mu);
    read(in, s);
    in >> h[c];
    auto *p = cameras[c].data();
    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<Prior<16>, 16, 16>(
                                 new Prior<16>{mu, s}),
                             nullptr, p);
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<Ratio, 1, 16>(new Ratio{mu[0] * .014}),
        nullptr, p);
    std::vector<int> fixed = {6, 7};
    if (mode == 0 || mode == 2) {
      fixed.push_back(14);
      fixed.push_back(15);
    }
    if (c == 0) {
      fixed.push_back(5);
      if (mode == 1)
        fixed.push_back(14);
    }
    problem.SetParameterLowerBound(p, 14, -.035);
    problem.SetParameterUpperBound(p, 14, .035);
    problem.SetParameterLowerBound(p, 15, -.028);
    problem.SetParameterUpperBound(p, 15, .028);
    problem.SetManifold(p, new ceres::SubsetManifold(16, fixed));
    for (int k = 0; k < 2; k++) {
      problem.SetParameterLowerBound(p, k, mu[k] * .48);
      problem.SetParameterUpperBound(p, k, mu[k] * 1.75);
    }
    if (mode == 2)
      problem.SetParameterBlockConstant(p);
  }
  for (int k = 0; k < nf; k++) {
    array<double, 6> s, zero{};
    read(in, base[k]);
    read(in, frame[k]);
    read(in, s);
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<Prior<6>, 6, 6>(new Prior<6>{zero, s}),
        nullptr, frame[k].data());
  }
  for (int i = 0; i < ne; i++) {
    int a, b;
    double dx, dy, y, d, sx, sy;
    in >> a >> b >> dx >> dy >> y >> d >> sx >> sy;
    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<Motion, 2, 6, 6>(
                                 new Motion{dx, dy, y, d, sx, sy}),
                             new ceres::HuberLoss(2), frame[a].data(),
                             frame[b].data());
  }
  for (auto &p : pts)
    read(in, p);
  for (int i = 0; i < no; i++) {
    int k, c, p;
    double dt, u, v;
    in >> k >> c >> p >> dt >> u >> v;
    auto a = base[k];
    auto *cost = new ceres::AutoDiffCostFunction<Observation, 2, 16, 6, 3>(
        new Observation{B[c], a, a[3], a[4], dt, h[c], u, v});
    problem.AddResidualBlock(cost, new ceres::CauchyLoss(loss),
                             cameras[c].data(), frame[k].data(), pts[p].data());
  }
  if (!in) {
    std::cerr << "invalid BA input\n";
    return 2;
  }
  ceres::Solver::Options opt;
  opt.num_threads = threads;
  opt.max_num_iterations = it;
  opt.linear_solver_type = ceres::SPARSE_SCHUR;
  opt.function_tolerance = 1e-6;
  opt.max_solver_time_in_seconds = 600;
  auto order = std::make_shared<ceres::ParameterBlockOrdering>();
  for (auto &p : pts)
    order->AddElementToGroup(p.data(), 0);
  for (auto &f : frame)
    order->AddElementToGroup(f.data(), 1);
  for (auto &c : cameras)
    order->AddElementToGroup(c.data(), 2);
  opt.linear_solver_ordering = order;
  ceres::Solver::Summary sum;
  ceres::Solve(opt, &problem, &sum);
  std::cout << sum.BriefReport() << std::endl;
  if (!sum.IsSolutionUsable())
    return 3;
  std::ofstream out(argv[2], std::ios::binary);
  write(out, cameras);
  write(out, frame);
  write(out, pts);
}
