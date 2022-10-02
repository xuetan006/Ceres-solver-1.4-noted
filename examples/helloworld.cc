// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: keir@google.com (Keir Mierle)
//
// A simple example of using the Ceres minimizer.
//
// Minimize 0.5 (10 - x)^2 using jacobian matrix computed using
// automatic differentiation.

#include "ceres/ceres.h"
#include "glog/logging.h"

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

// A templated cost functor that implements the residual r = 10 -
// x. The method operator() is templated so that we can then use an
// automatic differentiation wrapper around it to generate its
// derivatives.
// 第一部分：构建代价函数
struct CostFunctor {
  // 函数模板
  // 重载符号（），仿函数；传入待优化变量列表和承接残差的变量列表
  // 三个const：第一个是指针x指向的内容不可变，第二个const是指针x不能再指向别的内容
  // 第三个const修饰成员函数，成员函数有个隐式的this指针参数，通过this指针可以修改和访问
  // 类里面的成员变量，形式为：CostFunctor* const this。如果我们不想让别人通过this
  // 指针来修改成员变量，this指针又是隐式的，没办法显式的将this声明成指向常量的指针。
  // c++的做法是允许把const关键字放在成员函数的参数列表之后，表示this是一个指向常量
  // 的指针，像这样使用const的成员函数被称为常量成员函数，this形式变成了
  // const CostFunctor* const this。
  template <typename T> bool operator()(const T* const x, T* residual) const {
    // 残差计算
    residual[0] = 10.0 - x[0];
    return true;
  }
};

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  // The variable to solve for with its initial value. It will be
  // mutated in place by the solver.
  // 寻优参数 x 的初始值，为 5
  double x = 0.5;
  const double initial_x = x;

  // Build the problem.
  // 第二部分：构建优化问题
  Problem problem;

  // Set up the only cost function (also known as residual). This uses
  // auto-differentiation to obtain the derivative (jacobian).
  // 代价函数(类模板)赋值
  // 使用自动求导，将之前的代价函数结构体传入，第一个 1 是输出维度，即残差的维度，
  // 第二个 1 是输入维度，即待寻优参数 x 的维度。
  CostFunction* cost_function =
      new AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);
  //添加误差项，1.上一步实例化后的代价函数 2.核函数 3.待优化变量
  problem.AddResidualBlock(cost_function, NULL, &x);

  // Run the solver!
  // 第三部分： 配置并运行求解器
  Solver::Options options;
  // 是否输出到cout
  options.minimizer_progress_to_stdout = true;
  Solver::Summary summary;
  // 求解：1.求解器 2.实例化 problem 3.优化器
  Solve(options, &problem, &summary);

  // 输出优化的简要信息,迭代次数和每次的 cost
  std::cout << summary.BriefReport() << "\n";
  std::cout << "x : " << initial_x
            << " -> " << x << "\n";
  return 0;
}
