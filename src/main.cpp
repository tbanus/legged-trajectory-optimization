#include <ifopt/problem.h>
#include <ifopt/ipopt_solver.h>
#include <iostream>

// Base variables
class BaseVariables : public ifopt::VariableSet {
public:
  BaseVariables() : VariableSet(6,"base_vars") {}
  void SetVariables(const Eigen::VectorXd& x) override {}
  Eigen::VectorXd GetValues() const override { return Eigen::VectorXd::Zero(6); }
  VecBound GetBounds() const override { return VecBound(6, ifopt::NoBound); }
};

// Foot position variables
class FootPositionVariables : public ifopt::VariableSet {
public:
  FootPositionVariables() : VariableSet(12,"foot_pos") {}
  void SetVariables(const Eigen::VectorXd& x) override {}
  Eigen::VectorXd GetValues() const override { return Eigen::VectorXd::Zero(12); }
  VecBound GetBounds() const override { return VecBound(12, ifopt::NoBound); }
};

// Foot force variables
class FootForceVariables : public ifopt::VariableSet {
public:
  FootForceVariables() : VariableSet(12,"foot_force") {}
  void SetVariables(const Eigen::VectorXd& x) override {}
  Eigen::VectorXd GetValues() const override { return Eigen::VectorXd::Zero(12); }
  VecBound GetBounds() const override { return VecBound(12, ifopt::NoBound); }
};

// Foot contact schedule variables
class FootContactScheduleVariables : public ifopt::VariableSet {
public:
  FootContactScheduleVariables() : VariableSet(4,"foot_contact") {}
  void SetVariables(const Eigen::VectorXd& x) override {}
  Eigen::VectorXd GetValues() const override { return Eigen::VectorXd::Zero(4); }
  VecBound GetBounds() const override { return VecBound(4, ifopt::NoBound); }
};

// Dynamics constraint
class DynamicsConstraint : public ifopt::ConstraintSet {
public:
  DynamicsConstraint() : ConstraintSet(6,"dynamics_constr") {}
  Eigen::VectorXd GetValues() const override { return Eigen::VectorXd::Zero(6); }
  VecBound GetBounds() const override {
    return VecBound(6, ifopt::Bounds(0,0));
  }
  void FillJacobianBlock(std::string var_set, Jacobian& jac) const override {}
};

// Friction constraint
class FrictionConstraint : public ifopt::ConstraintSet {
public:
  FrictionConstraint() : ConstraintSet(4, "friction_constr") {}
  Eigen::VectorXd GetValues() const override {
    return Eigen::VectorXd::Zero(4);
  }
  VecBound GetBounds() const override {
    return VecBound(4, ifopt::Bounds(-0.0, 0.0));
  }
  void FillJacobianBlock(std::string var_set, Jacobian& jac) const override {}
};

// Bounding box constraint
class BoundingBoxConstraint : public ifopt::ConstraintSet {
public:
  BoundingBoxConstraint() : ConstraintSet(4, "bbox_constr") {}
  Eigen::VectorXd GetValues() const override {
    return Eigen::VectorXd::Zero(4);
  }
  VecBound GetBounds() const override {
    return VecBound(4, ifopt::NoBound);
  }
  void FillJacobianBlock(std::string var_set, Jacobian& jac) const override {}
};

// Example cost
class EffortCost : public ifopt::CostTerm {
public:
  EffortCost() : ifopt::CostTerm("effort_cost") {}
  double GetCost() const override { return 0.0; }
  void FillJacobianBlock(std::string var_set, Jacobian& jac) const override {}
};

int main()
{
  ifopt::Problem nlp;
  nlp.AddVariableSet(std::make_shared<BaseVariables>());
  nlp.AddVariableSet(std::make_shared<FootPositionVariables>());
  nlp.AddVariableSet(std::make_shared<FootForceVariables>());
  nlp.AddVariableSet(std::make_shared<FootContactScheduleVariables>());

  nlp.AddConstraintSet(std::make_shared<DynamicsConstraint>());
  nlp.AddConstraintSet(std::make_shared<FrictionConstraint>());
  nlp.AddConstraintSet(std::make_shared<BoundingBoxConstraint>());

  nlp.AddCostSet(std::make_shared<EffortCost>());

  ifopt::IpoptSolver solver;
  solver.SetOption("tol", 1e-6);
  solver.Solve(nlp);

  std::cout << "Solution: " 
            << nlp.GetOptVariables()->GetValues().transpose() 
            << std::endl;
  return 0;
}