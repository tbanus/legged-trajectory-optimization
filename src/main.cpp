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

// Obstacle avoidance constraint
class ObstacleConstraint : public ifopt::ConstraintSet {
public:
    ObstacleConstraint(int n_points, std::shared_ptr<TrajectoryVariables> vars)
        : ConstraintSet(6 * (n_points - 2), "obstacle"), n_points_(n_points), variables_(vars) {}

    Eigen::VectorXd GetValues() const override {
        Eigen::VectorXd g(6 * (n_points_ - 2));
        for (int i = 1; i < n_points_ - 1; ++i) {
            double x = variables_->GetValues()(2 * i);
            double y = variables_->GetValues()(2 * i + 1);

            // x constraints
            g[6 * (i - 1)]     = x;              // x >= 0
            g[6 * (i - 1) + 1] = 5.0 - x;        // x <= 5
            g[6 * (i - 1) + 2] = std::max(2.0 - x, x - 3.0); // Ensure x is NOT in [2, 3]

            // y constraints
            g[6 * (i - 1) + 3] = y;              // y >= 0
            g[6 * (i - 1) + 4] = 5.0 - y;        // y <= 5
            g[6 * (i - 1) + 5] = std::max(2.0 - y, y - 3.0); // Ensure y is NOT in [2, 3]
        }
        return g;
    }

    std::vector<ifopt::Bounds> GetBounds() const override {
        std::vector<ifopt::Bounds> bounds;

        for (int i = 1; i < n_points_ - 1; ++i) {
            // Bounds for x and y positions
            bounds.emplace_back(0.0, ifopt::inf); // x >= 0
            bounds.emplace_back(0.0, ifopt::inf); // x <= 5
            bounds.emplace_back(0.0, ifopt::inf); // Ensure x is outside [2, 3]

            bounds.emplace_back(0.0, ifopt::inf); // y >= 0
            bounds.emplace_back(0.0, ifopt::inf); // y <= 5
            bounds.emplace_back(0.0, ifopt::inf); // Ensure y is outside [2, 3]
        }
        return bounds;
    }
    
void FillJacobianBlock(std::string var_set, Jacobian& jac) const override {
    if (var_set == "trajectory") {
        for (int i = 1; i < n_points_ - 1; ++i) {
            double x = variables_->GetValues()(2 * i);
            double y = variables_->GetValues()(2 * i + 1);

            jac.coeffRef(6 * (i - 1), 2 * i) = 1.0;        // ∂x / ∂x
            jac.coeffRef(6 * (i - 1) + 1, 2 * i) = -1.0;   // ∂(5.0 - x) / ∂x
            jac.coeffRef(6 * (i - 1) + 2, 2 * i) = (x < 2.0) ? -1.0 : 1.0; // x constraint gradient

            jac.coeffRef(6 * (i - 1) + 3, 2 * i + 1) = 1.0;       // ∂y / ∂y
            jac.coeffRef(6 * (i - 1) + 4, 2 * i + 1) = -1.0;      // ∂(5.0 - y) / ∂y
            jac.coeffRef(6 * (i - 1) + 5, 2 * i + 1) = (y < 2.0) ? -1.0 : 1.0; // y constraint gradient
        }
    }
}


private:
    int n_points_;
    std::shared_ptr<TrajectoryVariables> variables_;
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

int main() {
    const int n_points = 10;

    auto trajectory_vars = std::make_shared<TrajectoryVariables>(n_points);
    auto momentum_constraint = std::make_shared<MomentumConstraint>(n_points, trajectory_vars);
    auto obstacle_constraint = std::make_shared<ObstacleConstraint>(n_points, trajectory_vars);
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

    // Visualization of the trajectory
    std::vector<double> x_vals, y_vals;
    for (int i = 0; i < n_points; ++i) {
        x_vals.push_back(solution[2 * i]);
        y_vals.push_back(solution[2 * i + 1]);
    }

    plt::plot(x_vals, y_vals, "-o");

    // Draw obstacle
    std::vector<double> obstacle_x = {2.0, 3.0, 3.0, 2.0, 2.0};
    std::vector<double> obstacle_y = {2.0, 2.0, 3.0, 3.0, 2.0};
    plt::plot(obstacle_x, obstacle_y, "r-");

    plt::xlim(0.0, 5.0);
    plt::ylim(0.0, 5.0);
    plt::title("Trajectory Optimization");
    plt::xlabel("X");
    plt::ylabel("Y");
    plt::show();

    return 0;
}
