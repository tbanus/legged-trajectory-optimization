#include <ifopt/ipopt_solver.h>
#include <towr/initialization/gait_generator.h>
#include <towr/models/robot_model.h>
#include <towr/parameters.h>
#include <towr/terrain/examples/height_map_examples.h>
#include <towr/variables/nodes_variables_all.h>
#include <towr/variables/spline_holder.h>
#include <towr/variables/phase_durations.h>
#include <towr/constraints/base_motion_constraint.h>
#include <towr/constraints/dynamic_constraint.h>
#include <towr/constraints/force_constraint.h>
#include <towr/constraints/range_of_motion_constraint.h>
#include <towr/constraints/swing_constraint.h>
#include <towr/constraints/terrain_constraint.h>
#include <towr/constraints/total_duration_constraint.h>
#include <towr/constraints/spline_acc_constraint.h>
#include <towr/costs/node_cost.h>
#include <cmath>
#include <iostream>
#include <plot_trajectories.h>
#include <Save2CSV.h>
using namespace towr;


int main() {
 
  int terrain_;
  int gait_combo_;
  int robot_;
  bool visualize_trajectory_;
  bool play_initialization_;
  double replay_speed_;
  bool plot_trajectory_;
  bool optimize_;
  bool publish_optimized_trajectory_;
  double total_duration_;
  bool optimize_phase_durations_;

  robot_ = RobotModel::Hyq;
  terrain_ = HeightMap::GapID;
  gait_combo_ = GaitGenerator::Combos::C0; 
  total_duration_ =1;
  visualize_trajectory_ = false;
  plot_trajectory_ = false;
  replay_speed_ = 1.0;  // realtime
  optimize_ = false;
  publish_optimized_trajectory_ = false;
  optimize_phase_durations_ = 0;
  int n_ee = 4;

  BaseState initial_base_;
  BaseState final_base_;
  std::vector<Eigen::Vector3d> initial_ee_W_;

  RobotModel model_;
  std::shared_ptr<HeightMap> terrain;
  Parameters params_;

  // terrain
  terrain = std::make_shared<Gap>();

  model_ = RobotModel(RobotModel::Anymal);
  auto nominal_stance_B = model_.kinematic_model_->GetNominalStanceInBase();

  double z_ground = 0.0;
  initial_ee_W_ =  nominal_stance_B;
  std::for_each(initial_ee_W_.begin(), initial_ee_W_.end(),
                [&](Eigen::Vector3d& p){ p.z() = z_ground; } // feet at 0 height
  );


  initial_base_.lin.at(kPos).z() = 0.5;

  
  final_base_.lin.at(towr::kPos) << 0.5, 0.0, 0.5;

  
  auto gait_gen_ = GaitGenerator::MakeGaitGenerator(n_ee);
  auto id_gait = static_cast<GaitGenerator::Combos>(gait_combo_);
  gait_gen_->SetCombo(id_gait);
  for (int ee = 0; ee < n_ee; ++ee) {
    initial_ee_W_.push_back(Eigen::Vector3d::Zero());

  }

    for (int ee=0; ee<n_ee; ++ee) {
      params_.ee_phase_durations_.push_back(gait_gen_->GetPhaseDurations(total_duration_, ee));
      params_.ee_in_contact_at_start_.push_back(gait_gen_->IsInContactAtStart(ee));
    }

 
  if (optimize_phase_durations_) params_.OptimizePhaseDurations();

  std::cout << "Total time ia: " << params_.GetTotalTime() << " seconds" << std::endl;
  std::cout << "Number of nodes: " << int(params_.GetBasePolyDurations().size() )+ 1 << std::endl;

  std::vector<NodesVariables::Ptr> vars;


  std::vector<NodesVariables::Ptr> base_motion;

  int n_nodes = params_.GetBasePolyDurations().size() + 1;
  static const std::string base_lin_nodes = "base-lin";
  static const std::string base_ang_nodes = "base-ang";
  static const std::string ee_motion_nodes = "ee-motion_";
  static const std::string ee_force_nodes = "ee-force_";
  static const std::string contact_schedule = "ee-schedule";
  auto spline_lin =
      std::make_shared<NodesVariablesAll>(n_nodes, k3D, base_lin_nodes);

  double x = final_base_.lin.p().x();
  double y = final_base_.lin.p().y();
  double z = terrain->GetHeight(x, y) -
             model_.kinematic_model_->GetNominalStanceInBase().front().z();
  Eigen::Vector3d final_pos(x, y, z);

  spline_lin->SetByLinearInterpolation(initial_base_.lin.p(), final_pos,
                                       params_.GetTotalTime());
  spline_lin->AddStartBound(kPos, {X, Y, Z}, initial_base_.lin.p());
  spline_lin->AddStartBound(kVel, {X, Y, Z}, initial_base_.lin.v());
  spline_lin->AddFinalBound(kPos, params_.bounds_final_lin_pos_,
                            final_base_.lin.p());
  spline_lin->AddFinalBound(kVel, params_.bounds_final_lin_vel_,
                            final_base_.lin.v());
  base_motion.push_back(spline_lin);

  auto spline_ang =
      std::make_shared<NodesVariablesAll>(n_nodes, k3D, base_ang_nodes);
  spline_ang->SetByLinearInterpolation(
      initial_base_.ang.p(), final_base_.ang.p(), params_.GetTotalTime());
  spline_ang->AddStartBound(kPos, {X, Y, Z}, initial_base_.ang.p());
  spline_ang->AddStartBound(kVel, {X, Y, Z}, initial_base_.ang.v());
  spline_ang->AddFinalBound(kPos, params_.bounds_final_ang_pos_,
                            final_base_.ang.p());
  spline_ang->AddFinalBound(kVel, params_.bounds_final_ang_vel_,
                            final_base_.ang.v());
  base_motion.push_back(spline_ang);

  vars.insert(vars.end(), base_motion.begin(), base_motion.end());


  std::vector<NodesVariablesPhaseBased::Ptr> ee_motion;
  double T = params_.GetTotalTime();
  for (int ee = 0; ee < params_.GetEECount(); ee++) {
    auto nodes = std::make_shared<NodesVariablesEEMotion>(
        params_.GetPhaseCount(ee), params_.ee_in_contact_at_start_.at(ee),
        ee_motion_nodes + std::to_string(ee),
        params_.ee_polynomials_per_swing_phase_);

    double yaw = final_base_.ang.p().z();
    Eigen::Vector3d euler(0.0, 0.0, yaw);
    Eigen::Matrix3d w_R_b = EulerConverter::GetRotationMatrixBaseToWorld(euler);
    Eigen::Vector3d final_ee_pos_W =
        final_base_.lin.p() +
        w_R_b * model_.kinematic_model_->GetNominalStanceInBase().at(ee);
    double x = final_ee_pos_W.x();
    double y = final_ee_pos_W.y();
    double z = terrain->GetHeight(x, y);
    nodes->SetByLinearInterpolation(initial_ee_W_.at(ee),
                                    Eigen::Vector3d(x, y, z), T);

    nodes->AddStartBound(kPos, {X, Y, Z}, initial_ee_W_.at(ee));
    ee_motion.push_back(nodes);
  }
  vars.insert(vars.end(), ee_motion.begin(), ee_motion.end());

  

  std::vector<NodesVariablesPhaseBased::Ptr> ee_force;

  for (int ee = 0; ee < params_.GetEECount(); ee++) {
    auto nodes = std::make_shared<NodesVariablesEEForce>(
        params_.GetPhaseCount(ee), params_.ee_in_contact_at_start_.at(ee),
        ee_force_nodes + std::to_string(ee), params_.force_polynomials_per_stance_phase_);

    // initialize with mass of robot distributed equally on all legs
    double m = model_.dynamic_model_->m();
    double g = model_.dynamic_model_->g();

    Eigen::Vector3d f_stance(0.0, 0.0, m * g / params_.GetEECount());
    nodes->SetByLinearInterpolation(f_stance, f_stance, T);  // stay constant
    ee_force.push_back(nodes);
  }
  vars.insert(vars.end(), ee_force.begin(), ee_force.end());



  std::vector<PhaseDurations::Ptr> contact_schedules;

  for (int ee=0; ee<params_.GetEECount(); ee++) {
    auto var = std::make_shared<PhaseDurations>(ee,
                                                params_.ee_phase_durations_.at(ee),
                                                params_.ee_in_contact_at_start_.at(ee),
                                                params_.bound_phase_duration_.first,
                                                params_.bound_phase_duration_.second);
    contact_schedules.push_back(var);
  }


 towr::SplineHolder spline_holder(base_motion.at(0), // linear
                               base_motion.at(1), // angular
                               params_.GetBasePolyDurations(),
                               ee_motion,
                               ee_force,
                               contact_schedules,
                               optimize_phase_durations_);


  std::vector<ifopt::ConstraintSet::Ptr> constraints;


  auto constraint = std::make_shared<DynamicConstraint>(model_.dynamic_model_,
                                                        params_.GetTotalTime(),
                                                        params_.dt_constraint_dynamic_,
                                                        spline_holder);
  constraints.push_back(constraint);

  
  
  for (int ee=0; ee<params_.GetEECount(); ee++) {
    auto rom = std::make_shared<RangeOfMotionConstraint>(model_.kinematic_model_,
                                                         params_.GetTotalTime(),
                                                         params_.dt_constraint_range_of_motion_,
                                                         ee,
                                                         spline_holder);
    constraints.push_back(rom);
  }
  
  
  auto baseRom = std::make_shared<BaseMotionConstraint>(params_.GetTotalTime(),
                                                 params_.dt_constraint_base_motion_,
                                                 spline_holder);
  
  constraints.push_back(baseRom);
  
  
  if(optimize_phase_durations_){
  for (int ee=0; ee<params_.GetEECount(); ee++) {
    auto duration_constraint = std::make_shared<TotalDurationConstraint>(T, ee);
    constraints.push_back(duration_constraint);
  }
  }
  
  for (int ee=0; ee<params_.GetEECount(); ee++) {
    auto c = std::make_shared<TerrainConstraint>(terrain, ee_motion_nodes + std::to_string(ee));
    constraints.push_back(c);
  }


   for (int ee=0; ee<params_.GetEECount(); ee++) {
    auto c = std::make_shared<ForceConstraint>(terrain,
                                               params_.force_limit_in_normal_direction_,
                                               ee);
    constraints.push_back(c);
  }


    for (int ee=0; ee<params_.GetEECount(); ee++) {
    auto swing = std::make_shared<SwingConstraint>(ee_motion_nodes + std::to_string(ee));
    constraints.push_back(swing);
  }


  constraints.push_back(std::make_shared<SplineAccConstraint>
                        (spline_holder.base_linear_, base_lin_nodes));

  constraints.push_back(std::make_shared<SplineAccConstraint>
                        (spline_holder.base_angular_, base_ang_nodes));



 std::vector<ifopt::CostTerm::Ptr> costs;
 double weight = 1.0;

  // for (int ee=0; ee<params_.GetEECount(); ee++)
  //   costs.push_back(std::make_shared<NodeCost>(ee_force_nodes + std::to_string(ee), kPos, Z, weight));


  // for (int ee=0; ee<params_.GetEECount(); ee++) {
  //   costs.push_back(std::make_shared<NodeCost>(ee_motion_nodes + std::to_string(ee), kVel, X, weight));
  //   costs.push_back(std::make_shared<NodeCost>(ee_motion_nodes + std::to_string(ee), kVel, Y, weight));
  // }


  ifopt::Problem nlp;
  for (auto var : vars) nlp.AddVariableSet(var);

  if( optimize_phase_durations_) for(auto cs : contact_schedules) nlp.AddVariableSet(cs);

  for (auto con : constraints)  nlp.AddConstraintSet(con); 
  for (auto cost : costs) nlp.AddCostSet(cost);
  auto solver = std::make_shared<ifopt::IpoptSolver>();
  solver->SetOption("jacobian_approximation",
                    "exact");  
  solver->SetOption("max_cpu_time", 200.0);
  solver->Solve(nlp);

  Save2CSV(spline_holder, "trajectory_data.csv");

  plot_trajectories(spline_holder);
  
}
