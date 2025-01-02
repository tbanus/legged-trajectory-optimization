#include <ifopt/problem.h>
#include <ifopt/ipopt_solver.h>
#include <ifopt/constraint_set.h>
#include <ifopt/cost_term.h>
#include <ifopt/variable_set.h>
#include <vector>
#include <cmath>
#include <iostream> // Ensure std::cout is recognized
#include <matplotlibcpp.h> // Include matplotlib-cpp for plotting

namespace plt = matplotlibcpp;

// Variable set for x, y positions
class TrajectoryVariables : public ifopt::VariableSet {
public:
    TrajectoryVariables(int n_points)
        : VariableSet(2 * n_points, "trajectory"), n_points_(n_points) {
        variables_.resize(2 * n_points_);
        variables_.setZero();
        // Initialize with start point
        variables_[0] = 1.0; // x0
        variables_[1] = 1.0; // y0
    }

    // Get variable values
    Eigen::VectorXd GetValues() const override {
        return variables_;
    }

    // Set variable values
    void SetVariables(const Eigen::VectorXd& x) override {
        variables_ = x;
    }

    // Get bounds for variables
    std::vector<ifopt::Bounds> GetBounds() const override {
        std::vector<ifopt::Bounds> bounds;
        bounds.emplace_back(1.0, 1.0); // x0 fixed
        bounds.emplace_back(1.0, 1.0); // y0 fixed
        for (int i = 1; i < n_points_ - 1; ++i) {
            bounds.emplace_back(0.0, 5.0); // Intermediate x
            bounds.emplace_back(0.0, 5.0); // Intermediate y
        }
        bounds.emplace_back(4.0, 4.0); // x_goal fixed
        bounds.emplace_back(4.0, 4.0); // y_goal fixed
        return bounds;

    }

private:
    int n_points_;
    Eigen::VectorXd variables_;
};

// Constraint set for momentum
class MomentumConstraint : public ifopt::ConstraintSet {
public:
    MomentumConstraint(int n_points, std::shared_ptr<TrajectoryVariables> vars)
        : ConstraintSet(2 * (n_points - 1), "momentum"), n_points_(n_points), variables_(vars) {}

    // Compute constraint values
    Eigen::VectorXd GetValues() const override {
        Eigen::VectorXd g(2 * (n_points_ - 1));
        for (int i = 1; i < n_points_; ++i) {
            double dx = variables_->GetValues()(2 * i) - variables_->GetValues()(2 * (i - 1));
            double dy = variables_->GetValues()(2 * i + 1) - variables_->GetValues()(2 * (i - 1) + 1);
            g(2 * (i - 1)) = dx;
            g(2 * (i - 1) + 1) = dy;
        }
        return g;
    }

    // Get bounds for constraints
    std::vector<ifopt::Bounds> GetBounds() const override {
        std::vector<ifopt::Bounds> bounds;
        for (int i = 0; i < 2 * (n_points_ - 1); ++i) {
            bounds.emplace_back(-0.1, 0.1); // Momentum constraint
        }
        return bounds;
    }

    void FillJacobianBlock(std::string var_set, Jacobian& jac) const override {
        if (var_set == "trajectory") {
            for (int i = 1; i < n_points_; ++i) {
                jac.coeffRef(2 * (i - 1), 2 * i) = 1.0;   // dx w.r.t. x_i
                jac.coeffRef(2 * (i - 1), 2 * (i - 1)) = -1.0; // dx w.r.t. x_{i-1}
                jac.coeffRef(2 * (i - 1) + 1, 2 * i + 1) = 1.0; // dy w.r.t. y_i
                jac.coeffRef(2 * (i - 1) + 1, 2 * (i - 1) + 1) = -1.0; // dy w.r.t. y_{i-1}
            }
        }
    }

private:
    int n_points_;
    std::shared_ptr<TrajectoryVariables> variables_;
};
// Obstacle avoidance constraint (circular obstacle)
class ObstacleConstraint : public ifopt::ConstraintSet {
public:
    ObstacleConstraint(int n_points, std::shared_ptr<TrajectoryVariables> vars, double obstacle_x, double obstacle_y, double radius)
        : ConstraintSet(n_points - 2, "obstacle"), n_points_(n_points), variables_(vars), obs_x_(obstacle_x), obs_y_(obstacle_y), radius_(radius) {}

    // Compute constraint values
    Eigen::VectorXd GetValues() const override {
        Eigen::VectorXd g(n_points_ - 2);
        for (int i = 1; i < n_points_ - 1; ++i) {
            double x = variables_->GetValues()(2 * i);
            double y = variables_->GetValues()(2 * i + 1);

            // Ensure the point is outside the circle
            double distance = std::sqrt((x - obs_x_) * (x - obs_x_) + (y - obs_y_) * (y - obs_y_));
            g[i - 1] = distance - radius_; // Negative if inside the circle
        }
        return g;
    }

    // Get bounds for the constraints
    std::vector<ifopt::Bounds> GetBounds() const override {
        std::vector<ifopt::Bounds> bounds;
        for (int i = 0; i < n_points_ - 2; ++i) {
            bounds.emplace_back(0.0, ifopt::inf); // Ensure points are outside the circle
        }
        return bounds;
    }

    // Fill the Jacobian block
    void FillJacobianBlock(std::string var_set, Jacobian& jac) const override {
        if (var_set == "trajectory") {
            for (int i = 1; i < n_points_ - 1; ++i) {
                double x = variables_->GetValues()(2 * i);
                double y = variables_->GetValues()(2 * i + 1);
                double dx = x - obs_x_;
                double dy = y - obs_y_;
                double distance = std::sqrt(dx * dx + dy * dy);

                if (distance > 1e-6) { // Avoid division by zero
                    jac.coeffRef(i - 1, 2 * i) = dx / distance; // ∂g/∂x
                    jac.coeffRef(i - 1, 2 * i + 1) = dy / distance; // ∂g/∂y
                }
            }
        }
    }

private:
    int n_points_;
    std::shared_ptr<TrajectoryVariables> variables_;
    double obs_x_, obs_y_, radius_;
};


// Cost function for trajectory optimization
class TrajectoryCost : public ifopt::CostTerm {
public:
    TrajectoryCost(std::shared_ptr<TrajectoryVariables> vars)
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
    std::shared_ptr<TrajectoryVariables> variables_;
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
    const int n_points = 40;

    auto trajectory_vars = std::make_shared<TrajectoryVariables>(n_points);
    auto momentum_constraint = std::make_shared<MomentumConstraint>(n_points, trajectory_vars);
    auto obstacle_constraint = std::make_shared<ObstacleConstraint>(n_points, trajectory_vars, 2.5, 2.5, 1.0); // Circle at (2.5, 2.5) with radius 1.0
    auto trajectory_cost = std::make_shared<TrajectoryCost>(trajectory_vars);

    ifopt::Problem nlp;
    nlp.AddVariableSet(trajectory_vars);
    nlp.AddConstraintSet(momentum_constraint);
    nlp.AddConstraintSet(obstacle_constraint);
    nlp.AddCostSet(trajectory_cost);

    ifopt::IpoptSolver solver;
    solver.Solve(nlp);

    Eigen::VectorXd solution = trajectory_vars->GetValues();
    std::cout << "Optimized trajectory:" << std::endl;
    for (int i = 0; i < n_points; ++i) {
        std::cout << "Point " << i << ": (" << solution[2 * i] << ", " << solution[2 * i + 1] << ")" << std::endl;
    }

    VisualizeTrajectory(solution, n_points, 2.5, 2.5, 1.0);

    return 0;
}
