#include <eigen3/Eigen/Dense>
#include <ifopt/variable_set.h>

#ifndef LTO_FORCEVARIABLES_H
#define LTO_FORCEVARIABLES_H

class ForceVariables : public ifopt::VariableSet {
public:
    ForceVariables(int n_points, int dim = 3)
        : VariableSet(dim * (n_points - 1), "force"), n_points_(n_points), dim_(dim) {
        variables_.resize(dim_ * (n_points_ - 1));
        variables_.setZero();
    }

    Eigen::VectorXd GetValues() const override {
        return variables_;
    }

    void SetVariables(const Eigen::VectorXd& x) override {
        variables_ = x;
    }

    std::vector<ifopt::Bounds> GetBounds() const override {
        std::vector<ifopt::Bounds> bounds;
        for (int i = 0; i < dim_ * (n_points_ - 1); ++i) {
            bounds.emplace_back(-100,100); // No bounds
        }
        // std::cout<<"ForceVars bounds.size()"<<bounds.size()<<std::endl;
        return bounds;
    }

private:
    int n_points_;
    int dim_;
    Eigen::VectorXd variables_;
};

#endif // LTO_FORCEVARIABLES_H