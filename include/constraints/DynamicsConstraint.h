#ifndef LTO_DYNAMICSCONSTRAINT_H
#define LTO_DYNAMICSCONSTRAINT_H

#include <ifopt/constraint_set.h>
#include <Eigen/Dense>
#include <memory>
#include "variables/BaseMotionVariables.h"
#include "variables/ForceVariables.h"

class DynamicsConstraint : public ifopt::ConstraintSet {
public:
  DynamicsConstraint(int n_points,
                     std::shared_ptr<BaseMotionVariables> traj_vars,
                     std::shared_ptr<ForceVariables> force_vars)
      : ConstraintSet((n_points - 1) * 2 * traj_vars->GetDim(), "dynamics"),
        n_points_(n_points),
        traj_vars_(traj_vars),
        force_vars_(force_vars),
        dim_(traj_vars->GetDim())
  {
    // Time step and mass (could also be passed in via constructor).
    dt = 0.3;
    m  = 1.0;

    // State-space matrices
    A_ = Eigen::MatrixXd::Zero(2 * dim_, 2 * dim_);
    B_ = Eigen::MatrixXd::Zero(2 * dim_, dim_);
    g_ = Eigen::VectorXd::Zero(2 * dim_);

    // Fill A matrix
    for (int i = 0; i < dim_; ++i) {
      A_(i, i)         = 1.0;
      A_(i, dim_ + i)  = dt;
      A_(dim_ + i, dim_ + i) = 1.0;
    }

    // Fill B matrix (force to acceleration)
    B_.block(dim_, 0, dim_, dim_) = Eigen::MatrixXd::Identity(dim_, dim_) * (dt / m);

    // Gravity term: acts on velocity
    g_(2 * dim_ - 1) = -9.81 * dt;
  }

  // -- The critical part: if IFOPT requires GetValues() to be const, that's fine,
  //    because we are only creating and returning a local variable 'g'.
  Eigen::VectorXd GetValues() const override
  {
    // Number of constraints = (n_points_ - 1) * 2 * dim_
    Eigen::VectorXd g((n_points_ - 1) * 2 * dim_);

    for (int i = 1; i < n_points_; ++i) {
      // Positions at time-step (i-1) and i
      Eigen::VectorXd r_prev = traj_vars_->GetValues().segment(dim_ * (i - 1), dim_);
      Eigen::VectorXd r_curr = traj_vars_->GetValues().segment(dim_ * i,     dim_);

      // Forces at time-step (i-1)
      Eigen::VectorXd u = force_vars_->GetValues().segment(dim_ * (i - 1), dim_);

      // Approximate velocity at the midpoint
      Eigen::VectorXd dr_avg = (r_curr - r_prev) / dt;

      // Build the "previous" and "current" state vectors
      Eigen::VectorXd x_prev(2 * dim_), x_curr(2 * dim_);
      x_prev << r_prev, dr_avg;
      x_curr << r_curr, dr_avg;

      // Predicted change from the dynamics model
      Eigen::VectorXd dx = A_ * x_prev + B_ * u + g_;

      // The constraint: x_curr - x_prev - dx = 0
      g.segment(2 * dim_ * (i - 1), 2 * dim_) = x_curr - x_prev - dx;
    }

    return g;
  }

  std::vector<ifopt::Bounds> GetBounds() const override
  {
    std::vector<ifopt::Bounds> bounds;
    bounds.reserve((n_points_ - 1) * 2 * dim_);

    for (int i = 0; i < (n_points_ - 1) * 2 * dim_; ++i) {
      // We want each constraint = 0
      bounds.emplace_back(-0.1, 0.1);
    }

    return bounds;
  }

  // Jacobian
  void FillJacobianBlock(std::string var_set, Jacobian& jac) const override
  {
    if (var_set == "trajectory") {
      for (int i = 1; i < n_points_; ++i) {
        for (int d = 0; d < dim_; ++d) {
          int row_r     = 2 * dim_ * (i - 1) + d;
          int row_dr    = 2 * dim_ * (i - 1) + dim_ + d;
          int col_prev  = dim_ * (i - 1) + d;
          int col_curr  = dim_ * i + d;

          // Partial wrt r_prev
          if (row_r >= 0 && row_r < jac.rows() && col_prev >= 0 && col_prev < jac.cols()) {
            jac.coeffRef(row_r, col_prev) = -1.0;
          }

          // Partial wrt r_curr
          if (row_r >= 0 && row_r < jac.rows() && col_curr >= 0 && col_curr < jac.cols()) {
            jac.coeffRef(row_r, col_curr) = 1.0;
          }
          // row_dr is for velocity part if needed
        }
      }
    } else if (var_set == "force") {
      // For each force variable
      for (int i = 1; i < n_points_; ++i) {
        for (int d = 0; d < dim_; ++d) {
          int row = 2 * dim_ * (i - 1) + dim_ + d;
          int col = dim_ * (i - 1) + d;
          // B_ * u
          if (row >= 0 && row < jac.rows() && col >= 0 && col < jac.cols()) {
            jac.coeffRef(row, col) = B_(dim_ + d, d);
          }
        }
      }
    }
  }

private:
  int n_points_;
  std::shared_ptr<BaseMotionVariables> traj_vars_;
  std::shared_ptr<ForceVariables> force_vars_;

  // State-space matrices (size 2*dim_ x 2*dim_ for A, 2*dim_ x dim_ for B).
  Eigen::MatrixXd A_;
  Eigen::MatrixXd B_;

  // Gravity term (size 2*dim_).
  Eigen::VectorXd g_;

  int dim_;
  double dt;  // Time step
  double m;   // Mass
};

#endif  // LTO_DYNAMICSCONSTRAINT_H
