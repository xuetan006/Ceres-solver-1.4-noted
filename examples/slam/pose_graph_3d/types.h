// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2016 Google Inc. All rights reserved.
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
// Author: vitus@google.com (Michael Vitus)

#ifndef EXAMPLES_CERES_TYPES_H_
#define EXAMPLES_CERES_TYPES_H_

#include <istream>
#include <map>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"

namespace ceres {
namespace examples {

// 定义节点位姿结构体
struct Pose3d {
  // 位置，3维
  Eigen::Vector3d p;
  // 姿态：四元数表示，4维
  Eigen::Quaterniond q;

  // The name of the data type in the g2o file format.
  static std::string name() {
    return "VERTEX_SE3:QUAT";
  }

  // Eigen中的宏定义，该宏定义会重载new函数
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// 重载输入运算符，方便从文件中读取节点位姿数据
std::istream& operator>>(std::istream& input, Pose3d& pose) {
  input >> pose.p.x() >> pose.p.y() >> pose.p.z() >> pose.q.x() >>
      pose.q.y() >> pose.q.z() >> pose.q.w();
  // Normalize the quaternion to account for precision loss due to
  // serialization.
  pose.q.normalize();
  return input;
}

// 定义map容器，参数说明：
// int:键的类型
// Pose3d:映射值的类型。
// std::less<int>: 接受两个键值作为参数并返回一个bool值，Compare可以是函数指针或函数对象，默认为 less<T>，其返回与使用小于操作符(a<b)相同的结果。
// Eigen::aligned_allocator：分配器对象的类型，用于定义存储分配模型。在使用Eigen的时候，如果STL容器中的元素是Eigen数据库结构，比如下面用vector
// 容器存储Eigen::Matrix4f类型或用map存储Eigen::Vector4f数据类型时：
// vector<Eigen::Matrix4d>;
// std::map<int, Eigen::Vector4f>;
// 这么使用编译能通过，当运行时会报段错误。
// 注意在Pose3d中需要使用EIGEN_MAKE_ALIGNED_OPERATOR_NEW宏定义。这里涉及到了Eigen的内存对齐问题。
typedef std::map<int, Pose3d, std::less<int>,
                 Eigen::aligned_allocator<std::pair<const int, Pose3d> > >
    MapOfPoses;

// The constraint between two vertices in the pose graph. The constraint is the
// transformation from vertex id_begin to vertex id_end.
// 节点约束的结构体
struct Constraint3d {
  // 约束的起始节点和结束节点
  int id_begin;
  int id_end;

  // The transformation that represents the pose of the end frame E w.r.t. the
  // begin frame B. In other words, it transforms a vector in the E frame to
  // the B frame.
  // 两个节点间的约束
  Pose3d t_be;

  // The inverse of the covariance matrix for the measurement. The order of the
  // entries are x, y, z, delta orientation.
  // 信息矩阵，表征权重
  Eigen::Matrix<double, 6, 6> information;

  // The name of the data type in the g2o file format.
  static std::string name() {
    return "EDGE_SE3:QUAT";
  }

  // Eigen中的宏定义，该宏定义会重载new函数
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// 重载输入运算符，方便从文件中读取节点间约束数据
std::istream& operator>>(std::istream& input, Constraint3d& constraint) {
  Pose3d& t_be = constraint.t_be;
  input >> constraint.id_begin >> constraint.id_end >> t_be;

  for (int i = 0; i < 6 && input.good(); ++i) {
    for (int j = i; j < 6 && input.good(); ++j) {
      input >> constraint.information(i, j);
      if (i != j) {
        constraint.information(j, i) = constraint.information(i, j);
      }
    }
  }
  return input;
}

// 约束都存储在vector中，用到了eigen中的数据类型，所以需要用eigen提供的内存分配器Eigen::aligned_allocator
typedef std::vector<Constraint3d, Eigen::aligned_allocator<Constraint3d> >
    VectorOfConstraints;

}  // namespace examples
}  // namespace ceres

#endif  // EXAMPLES_CERES_TYPES_H_
