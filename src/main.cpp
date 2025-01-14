#include <ifopt/constraint_set.h>
#include <ifopt/cost_term.h>
#include <ifopt/ipopt_solver.h>
#include <ifopt/problem.h>
#include <ifopt/variable_set.h>
#include <matplotlibcpp.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "constraints/DynamicsConstraint.h"
#include "variables/BaseMotionVariables.h"
#include "variables/ForceVariables.h"

namespace plt = matplotlibcpp;

// Constraint set for dynamics

// Obstacle avoidance constraint (circular obstacle)
class ObstacleConstraint : public ifopt::ConstraintSet {
 public:
  ObstacleConstraint(int n_points, std::shared_ptr<BaseMotionVariables> vars,
                     double obstacle_x, double obstacle_y, double radius)
      : ConstraintSet(n_points - 2, "obstacle"),
        n_points_(n_points),
        variables_(vars),
        obs_x_(obstacle_x),
        obs_y_(obstacle_y),
        radius_(radius),
        dim_(vars->GetDim()) {}

  Eigen::VectorXd GetValues() const override {
    Eigen::VectorXd g(n_points_ - 2);
    for (int i = 1; i < n_points_ - 1; ++i) {
      double x = variables_->GetValues()(dim_ * i);
      double y = variables_->GetValues()(dim_ * i + 1);

      double distance_squared =
          (x - obs_x_) * (x - obs_x_) + (y - obs_y_) * (y - obs_y_);
      g(i - 1) = radius_ * radius_ -
                 distance_squared;  // Negative if inside the circle
    }
    return g;
  }

  std::vector<ifopt::Bounds> GetBounds() const override {
    std::vector<ifopt::Bounds> bounds;
    for (int i = 0; i < n_points_ - 2; ++i) {
      bounds.emplace_back(0.0,
                          ifopt::inf);  // Ensure points are outside the circle
    }
    return bounds;
  }

  void FillJacobianBlock(std::string var_set, Jacobian& jac) const override {
    if (var_set == "trajectory") {
      for (int i = 1; i < n_points_ - 1; ++i) {
        double x = variables_->GetValues()(dim_ * i);
        double y = variables_->GetValues()(dim_ * i + 1);

        if (dim_ * i < jac.cols() && i - 1 < jac.rows()) {
          jac.coeffRef(i - 1, dim_ * i) = -2.0 * (x - obs_x_);      // ∂g/∂x
          jac.coeffRef(i - 1, dim_ * i + 1) = -2.0 * (y - obs_y_);  // ∂g/∂y
        }
      }
    }
  }

 private:
  int n_points_;
  std::shared_ptr<BaseMotionVariables> variables_;
  double obs_x_, obs_y_, radius_;
  int dim_;
};

// Cost function for trajectory optimization
class TrajectoryCost : public ifopt::CostTerm {
 public:
  TrajectoryCost(std::shared_ptr<BaseMotionVariables> vars)
      : CostTerm("trajectory_cost"), variables_(vars) {}

  double GetCost() const override {
    double cost = 0.0;
    for (int i = 0; i < variables_->GetValues().size(); i++) {
      cost += variables_->GetValues()[i] * variables_->GetValues()[i];
    }
    return cost;
  }

  void FillJacobianBlock(std::string var_set, Jacobian& jac) const override {
    if (var_set == "trajectory") {
      for (int i = 0; i < variables_->GetValues().size(); i++) {
        jac.coeffRef(0, i) = 2.0 * variables_->GetValues()[i];
      }
    }
  }

 private:
  std::shared_ptr<BaseMotionVariables> variables_;
};
void VisualizeTrajectory(const Eigen::VectorXd& solution, int n_points, double obstacle_x, double obstacle_y, double radius) {
    std::vector<double> x_vals, y_vals;

    // Extract trajectory points
    for (int i = 0; i < n_points; ++i) {
        x_vals.push_back(solution[2 * i]);
        y_vals.push_back(solution[2 * i + 1]);
    }

    // Plot trajectory
    plt::plot(x_vals, y_vals, "-o"); // Trajectory line with markers

    // Highlight start and end points
    plt::scatter(std::vector<double>{x_vals.front()}, std::vector<double>{y_vals.front()}, 100);
    plt::scatter(std::vector<double>{x_vals.back()}, std::vector<double>{y_vals.back()}, 100);

    // Add labels for start and end points
    plt::text(x_vals.front(), y_vals.front(), "Start");
    plt::text(x_vals.back(), y_vals.back(), "End");

    // Draw circular obstacle
    std::vector<double> circle_x, circle_y;
    const int num_circle_points = 100;
    for (int i = 0; i < num_circle_points; ++i) {
        double angle = 2.0 * M_PI * i / num_circle_points;
        circle_x.push_back(obstacle_x + radius * std::cos(angle));
        circle_y.push_back(obstacle_y + radius * std::sin(angle));
    }
    plt::plot(circle_x, circle_y, "r-"); // Red circle for the obstacle

    // Set plot limits and labels
    plt::xlim(0.0, 5.0);
    plt::ylim(0.0, 5.0);
    plt::title("Trajectory Optimization with Circular Obstacle");
    plt::xlabel("X");
    plt::ylabel("Y");

    // Display the plot
    plt::show();
}



int main() {
  const int n_points = 100;
  const int dim = 3;

  auto base_motion_vars = std::make_shared<BaseMotionVariables>(n_points, dim);
  base_motion_vars->SetByLinearInterpolation(Eigen::Vector3d(1.0, 1.0, 1.0),
                                             Eigen::Vector3d(4.0, 4.0, 4.0));

  auto force_vars = std::make_shared<ForceVariables>(n_points, dim);
  force_vars->SetVariables(Eigen::VectorXd::Zero(dim * (n_points - 1)));

  auto dynamics_constraint = std::make_shared<DynamicsConstraint>(
      n_points, base_motion_vars, force_vars);
  auto obstacle_constraint = std::make_shared<ObstacleConstraint>(
      n_points, base_motion_vars, 2.5, 2.5, 1.0);  // Circular obstacle at (2.5, 2.5) with radius 1.0
  auto trajectory_cost = std::make_shared<TrajectoryCost>(base_motion_vars);

  ifopt::Problem nlp;
  nlp.AddVariableSet(base_motion_vars);
  nlp.AddVariableSet(force_vars);
  nlp.AddConstraintSet(dynamics_constraint);
  // nlp.AddConstraintSet(obstacle_constraint);
  // nlp.AddCostSet(trajectory_cost);

  ifopt::IpoptSolver solver;
  solver.SetOption("max_iter", 1000);
  solver.SetOption("tol", 1e-6);  // Set tolerance for convergence
  solver.SetOption("print_level", 5);  // Increase print level for more detailed output

  // Debugging statements
  std::cout << "Before solving: " << std::endl;
  std::cout << "nlp: " << &nlp << std::endl;
  std::cout << "solver: " << &solver << std::endl;

  solver.Solve(nlp);

  std::cout << "After solving: " << std::endl;

  Eigen::VectorXd solution = base_motion_vars->GetValues();
  std::cout << "Optimized trajectory:" << std::endl;
  for (int i = 0; i < n_points; ++i) {
    std::cout << "Point " << i << ": (";
    for (int d = 0; d < dim; ++d) {
      std::cout << solution[dim * i + d];
      if (d < dim - 1) std::cout << ", ";
    }
    std::cout << ")" << std::endl;
  }

  // Print force variables
  Eigen::VectorXd force_solution = force_vars->GetValues();
  std::cout << "Optimized forces:" << std::endl;
  for (int i = 0; i < n_points - 1; ++i) {
    std::cout << "Force " << i << ": (";
    for (int d = 0; d < dim; ++d) {
      std::cout << force_solution[dim * i + d];
      if (d < dim - 1) std::cout << ", ";
    }
    std::cout << ")" << std::endl;
  }

  // Visualization of the trajectory
  std::vector<double> x_vals, y_vals;
  for (int i = 0; i < n_points; ++i) {
    x_vals.push_back(solution[dim * i]);
    y_vals.push_back(solution[dim * i + 1]);
  }

  plt::plot(x_vals, y_vals, "-o");

  // Draw circular obstacle
  std::vector<double> circle_x, circle_y;
  const int num_circle_points = 100;
  for (int i = 0; i < num_circle_points; ++i) {
    double angle = 2.0 * M_PI * i / num_circle_points;
    circle_x.push_back(2.5 + 1.0 * std::cos(angle));
    circle_y.push_back(2.5 + 1.0 * std::sin(angle));
  }
  plt::plot(circle_x, circle_y, "r-");

  plt::xlim(0.0, 5.0);
  plt::ylim(0.0, 5.0);
  plt::title("Trajectory Optimization with Circular Obstacle");
  plt::xlabel("X");
  plt::ylabel("Y");
  plt::show();

  return 0;
}