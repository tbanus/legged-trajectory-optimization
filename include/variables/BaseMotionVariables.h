#include <ifopt/bounds.h>
#include <ifopt/constraint_set.h>
#include <ifopt/cost_term.h>
#include <ifopt/ipopt_solver.h>
#include <ifopt/problem.h>
#include <ifopt/variable_set.h>

#include <cmath>
#include <eigen3/Eigen/Dense>
#include <iostream>  // Ensure std::cout is recognized
#include <vector>

#ifndef LTO_BASEMOTIONVARIABLES_H
#define LTO_BASEMOTIONVARIABLES_H

class BaseMotionVariables : public ifopt::VariableSet {
 public:
  BaseMotionVariables(int n_points, int dim = 3)
      : VariableSet(dim * n_points, "trajectory"),
        n_points_(n_points),
        dim_(dim) {
    variables_.resize(dim_ * n_points_);
    variables_.setZero();
    // Initialize with start point
    variables_[0] = 1.0;  // x0
    variables_[1] = 1.0;  // y0
    if (dim > 2) {
      variables_[2] = 1.0;  // z0 if dim is 3
    }
  }

  // Get variable values
  Eigen::VectorXd GetValues() const override { return variables_; }

  // Set variable values
  void SetVariables(const Eigen::VectorXd& x) override { variables_ = x; }

  // Get bounds for variables
  std::vector<ifopt::Bounds> GetBounds() const override {
    std::vector<ifopt::Bounds> bounds;

    for (int d = 0; d < dim_; ++d) {
      bounds.emplace_back(1, 1);  // Start point fixed
    }
    // }
    for (int i = 1; i < n_points_ - 1; ++i) {
      for (int d = 0; d < dim_; ++d) {
        bounds.emplace_back(ifopt::NoBound);  // Intermediate points
      }
    }
    for (int d = 0; d < dim_; ++d) {
      bounds.emplace_back(4, 4);  // End point fixed
    }
    // std::cout << "TrajVars bounds.size()" << bounds.size() << std::endl;

    return bounds;
  }

  int GetDim() const { return dim_; }

  void SetByLinearInterpolation(const VectorXd& initial_val,
                                const VectorXd& final_val) {
    // similar to python numpy linspace function to get intermediate points
    VectorXd dp = final_val - initial_val;

    for (int i = 0; i < n_points_; ++i) {
      variables_[dim_ * i] = initial_val[0] + i * dp[0] / (n_points_ - 1);
      variables_[dim_ * i + 1] = initial_val[1] + i * dp[1] / (n_points_ - 1);
      if (dim_ > 2) {
        variables_[dim_ * i + 2] = initial_val[2] + i * dp[2] / (n_points_ - 1);
      }
    }
  }

 private:
  int n_points_;
  Eigen::VectorXd variables_;
  int dim_;
};

#endif  // LTO_BASEMOTIONVARIABLES_H