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

// Cost function for total distance
class DistanceCost : public ifopt::CostTerm {
public:
    DistanceCost(int n_points, std::shared_ptr<TrajectoryVariables> vars)
        : CostTerm("distance_cost"), n_points_(n_points), variables_(vars) {}

    double GetCost() const override {
        double cost = 0.0;
        for (int i = 1; i < n_points_; ++i) {
            double dx = variables_->GetValues()(2 * i) - variables_->GetValues()(2 * (i - 1));
            double dy = variables_->GetValues()(2 * i + 1) - variables_->GetValues()(2 * (i - 1) + 1);
            cost += std::sqrt(dx * dx + dy * dy);
        }
        return cost;
    }

    void FillJacobianBlock(std::string var_set, Jacobian& jac) const override {
        if (var_set == "trajectory") {
            const double epsilon = 1e-8; // Avoid division by zero
            for (int i = 1; i < n_points_; ++i) {
                double dx = variables_->GetValues()(2 * i) - variables_->GetValues()(2 * (i - 1));
                double dy = variables_->GetValues()(2 * i + 1) - variables_->GetValues()(2 * (i - 1) + 1);
                double dist = std::sqrt(dx * dx + dy * dy) + epsilon; // Add epsilon
                jac.coeffRef(0, 2 * i) = dx / dist;       // Partial w.r.t. x_i
                jac.coeffRef(0, 2 * (i - 1)) = -dx / dist; // Partial w.r.t. x_{i-1}
                jac.coeffRef(0, 2 * i + 1) = dy / dist;   // Partial w.r.t. y_i
                jac.coeffRef(0, 2 * (i - 1) + 1) = -dy / dist; // Partial w.r.t. y_{i-1}
            }
        }
    }

private:
    int n_points_;
    std::shared_ptr<TrajectoryVariables> variables_;
};

void PlotTrajectory(const Eigen::VectorXd& solution, int n_points) {
    std::vector<double> x_vals, y_vals;
    for (int i = 0; i < n_points; ++i) {
        x_vals.push_back(solution(2 * i));
        y_vals.push_back(solution(2 * i + 1));
    }
    plt::plot(x_vals, y_vals, "-o");
    plt::xlabel("X Position");
    plt::ylabel("Y Position");
    plt::title("Optimized Trajectory");
    plt::grid(true);
    plt::show();
}

int main() {
    int n_points = 10; // Number of discrete points in trajectory

    ifopt::Problem nlp;

    auto trajectory_vars = std::make_shared<TrajectoryVariables>(n_points);
    trajectory_vars->SetVariables(Eigen::VectorXd::LinSpaced(2 * n_points, 1.0, 4.0)); // Improved initial guess

    nlp.AddVariableSet(trajectory_vars);
    nlp.AddConstraintSet(std::make_shared<MomentumConstraint>(n_points, trajectory_vars));
    nlp.AddCostSet(std::make_shared<DistanceCost>(n_points, trajectory_vars));

    ifopt::IpoptSolver solver;
    solver.SetOption("print_level", 5); // Enable verbose output
    solver.Solve(nlp);

    Eigen::VectorXd solution = nlp.GetOptVariables()->GetValues();
    for (int i = 0; i < n_points; ++i) {
        std::cout << "Point " << i << ": (" << solution(2 * i) << ", " << solution(2 * i + 1) << ")" << std::endl;
    }

    PlotTrajectory(solution, n_points);

    return 0;
}
