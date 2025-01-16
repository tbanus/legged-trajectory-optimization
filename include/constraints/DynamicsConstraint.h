#ifndef LTO_DYNAMICSCONSTRAINT_H
#define LTO_DYNAMICSCONSTRAINT_H

#include <ifopt/constraint_set.h>
#include <eigen3/Eigen/Dense>
#include <memory>
#include "variables/BaseMotionVariables.h"
#include "variables/ForceVariables.h"

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
    A_ = Eigen::MatrixXd::Zero(2 * dim_, 2 * dim_);
    B_ = Eigen::MatrixXd::Zero(2 * dim_, dim_);
    g_ = Eigen::VectorXd::Zero(2 * dim_);

    // Fill A matrix
    for (int i = 0; i < dim_; ++i) {
      A_(i, i) = 1.0;
      A_(i, dim_ + i) = dt;
      A_(dim_ + i, dim_ + i) = 1.0;
    }

    // Fill B matrix
    B_.block(dim_, 0, dim_, dim_) = Eigen::MatrixXd::Identity(dim_, dim_) * (dt / m);

    // Gravity vector
    g_.tail(1).setConstant(-9.81 * dt);
  }

  // Compute constraint values
  Eigen::VectorXd GetValues() const override {
    Eigen::VectorXd g((n_points_ - 1) * dim_);
    
    for (int i = 1; i < n_points_; ++i) {
      Eigen::VectorXd r_prev_2 ;

      r_prev_2.resize(3);
      r_prev_2.setZero();
      if (i>1){
      
        r_prev_2 = traj_vars_->GetValues().segment(dim_ * (i - 2), dim_);
      }
        
       
      Eigen::VectorXd r_prev = traj_vars_->GetValues().segment(dim_ * (i - 1), dim_);
      Eigen::VectorXd r_curr = traj_vars_->GetValues().segment(dim_ * i, dim_);
      Eigen::VectorXd u = force_vars_->GetValues().segment(dim_ * (i - 1), dim_);
      
      Eigen::VectorXd dr_avg_prev = (r_prev - r_prev_2) / dt;
      Eigen::VectorXd dr_avg = (r_curr - r_prev) / dt;

      Eigen::VectorXd x_prev(2 * dim_);
      x_prev << r_prev, dr_avg_prev;
      Eigen::VectorXd x_curr(2 * dim_);
      x_curr << r_curr, dr_avg;

      Eigen::VectorXd dx = A_ * x_prev + B_ * u + g_;

      // The constraint: x_curr - x_prev - dx = 0 (only position variables)
      g.segment(dim_ * (i - 1), dim_) = x_curr.head(dim_) - dx.head(dim_);
    }

    return g;
  }

  // Get bounds for constraints
  std::vector<ifopt::Bounds> GetBounds() const override {
    std::vector<ifopt::Bounds> bounds;
    for (int i = 0; i < (n_points_ - 1) * dim_; ++i) {
      
      bounds.emplace_back(ifopt::BoundZero);  // Equality constraint

    }
    return bounds;
  }

  void FillJacobianBlock(std::string var_set, Jacobian& jac) const override {
    if (var_set == "trajectory") {
      for (int i = 1; i < n_points_; ++i) {
        for (int d = 0; d < dim_; ++d) {
          int row_r = dim_ * (i - 1) + d;
          int col_prev_r = dim_ * (i - 1) + d;
          int col_curr_r = dim_ * i + d;
          if (row_r >= 0 && row_r < jac.rows() && col_prev_r >= 0 && col_prev_r < jac.cols()) {
            jac.coeffRef(row_r, col_prev_r) = -1.0;
          }
          if (row_r >= 0 && row_r < jac.rows() && col_curr_r >= 0 && col_curr_r < jac.cols()) {
            jac.coeffRef(row_r, col_curr_r) = 1.0;
          }
        }
      }
    } else if (var_set == "force") {
      for (int i = 1; i < n_points_; ++i) {
        for (int d = 0; d < dim_; ++d) {
          int row = dim_ * (i - 1) + d;
          int col = dim_ * (i - 1) + d;
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
  Eigen::MatrixXd A_;
  Eigen::MatrixXd B_;
  Eigen::VectorXd g_;
  int dim_;
  double dt = 0.2;  // Time step
  double m = 2.0;   // Mass
};

#endif  // LTO_DYNAMICSCONSTRAINT_H
