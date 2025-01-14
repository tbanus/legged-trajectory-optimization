#include <ifopt/constraint_set.h>
#include <ifopt/cost_term.h>
#include <ifopt/ipopt_solver.h>
#include <ifopt/problem.h>
#include <ifopt/variable_set.h>

#include <cmath>
#include <eigen3/Eigen/Dense>
#include <iostream>  // Ensure std::cout is recognized
#include <vector>

#include "variables/ForceVariables.h"
#include "variables/BaseMotionVariables.h"

#ifndef LTO_DYNAMICSCONSTRAINT_H
#define LTO_DYNAMICSCONSTRAINT_H

class DynamicsConstraint : public ifopt::ConstraintSet {
 public:
  DynamicsConstraint(int n_points,
                     std::shared_ptr<BaseMotionVariables> traj_vars,
                     std::shared_ptr<ForceVariables> force_vars)
      : ConstraintSet((n_points - 1) * traj_vars->GetDim(), "dynamics"),
        n_points_(n_points),
        traj_vars_(traj_vars),
        force_vars_(force_vars),
        dim_(traj_vars->GetDim()) {
    // Define the state-space matrices A and B
    A_ = Eigen::MatrixXd::Identity(dim_, dim_);  // Identity matrix
    B_ = Eigen::MatrixXd::Zero(dim_, dim_);  // Identity matrix
    B_ = B_ * 0.5 * dt * dt / m;                 // Scaling factor
  }

  // Compute constraint values
  Eigen::VectorXd GetValues() const override {
    Eigen::VectorXd g((n_points_ - 1) * dim_);
    for (int i = 1; i < n_points_; ++i) {
      Eigen::VectorXd x_prev = traj_vars_->GetValues().segment(dim_ * (i - 1), dim_);
      Eigen::VectorXd x_curr = traj_vars_->GetValues().segment(dim_ * i, dim_);
      Eigen::VectorXd u = force_vars_->GetValues().segment(dim_ * (i - 1), dim_);
      Eigen::VectorXd dx = A_ * x_prev + B_ * u;
      g.segment(dim_ * (i - 1), dim_) = x_curr - x_prev - dx;
    }
    std::cout << "g.size() " << g.size() << std::endl;
    return g;
  }

  // Get bounds for constraints
  std::vector<ifopt::Bounds> GetBounds() const override {
    std::vector<ifopt::Bounds> bounds;
    for (int i = 0; i < (n_points_ - 1) * dim_; ++i) {
      bounds.emplace_back(0.0, 0.0);  // Equality constraint
    }
    std::cout << "bounds.size() " << bounds.size() << std::endl;
    return bounds;
  }

  void FillJacobianBlock(std::string var_set, Jacobian& jac) const override {
    if (var_set == "trajectory") {
      for (int i = 1; i < n_points_; ++i) {
        for (int d = 0; d < dim_; ++d) {
          int row = dim_ * (i - 1) + d;
          int col_prev = dim_ * (i - 1) + d;
          int col_curr = dim_ * i + d;
          if (row >= 0 && row < jac.rows() && col_prev >= 0 && col_prev < jac.cols()) {
            jac.coeffRef(row, col_prev) = -1.0;
          }
          if (row >= 0 && row < jac.rows() && col_curr >= 0 && col_curr < jac.cols()) {
            jac.coeffRef(row, col_curr) = 1.0;
          }
        }
      }
    } else if (var_set == "force") {
      for (int i = 1; i < n_points_; ++i) {
        for (int d = 0; d < dim_; ++d) {
          int row = dim_ * (i - 1) + d;
          int col = dim_ * (i - 1) + d;
          if (row >= 0 && row < jac.rows() && col >= 0 && col < jac.cols()) {
            jac.coeffRef(row, col) = B_(d, d);
          }
        }
      }
    }
    std::cout << __FILE__ << " var_set " << var_set << std::endl;
    std::cout << __FILE__ << " jac.size() " << jac.size() << std::endl;
  }

 private:
  int n_points_;
  std::shared_ptr<BaseMotionVariables> traj_vars_;
  std::shared_ptr<ForceVariables> force_vars_;
  Eigen::MatrixXd A_;
  Eigen::MatrixXd B_;
  int dim_;
  double dt = 1;  // Time step
  double m = 1.0;   // Mass
};

#endif  // LTO_DYNAMICSCONSTRAINT_H