/******************************************************************************
Copyright (c) 2018, Alexander W. Winkler. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************************************************************/

#include <ifopt/constraint_set.h>
#include <ifopt/cost_term.h>
#include <ifopt/ipopt_solver.h>
#include <ifopt/variable_set.h>
#include <towr/initialization/gait_generator.h>
#include <towr/models/robot_model.h>
#include <towr/nlp_formulation.h>
#include <towr/parameters.h>
#include <towr/terrain/examples/height_map_examples.h>
#include <towr/terrain/height_map.h>
#include <towr/variables/nodes_variables_all.h>
#include <towr/variables/spline_holder.h>
#include <towr/nlp_formulation.h>
#include <towr/variables/variable_names.h>
#include <towr/variables/phase_durations.h>
#include <towr/constraints/base_motion_constraint.h>
#include <towr/constraints/dynamic_constraint.h>
#include <towr/constraints/force_constraint.h>
#include <towr/constraints/range_of_motion_constraint.h>
#include <towr/constraints/swing_constraint.h>
#include <towr/constraints/terrain_constraint.h>
#include <towr/constraints/total_duration_constraint.h>
#include <towr/constraints/spline_acc_constraint.h>
#include <ifopt/variable_set.h>
#include <ifopt/constraint_set.h>
#include <ifopt/cost_term.h>
#include <towr/variables/spline_holder.h>
#include <towr/models/robot_model.h>
#include <towr/terrain/height_map.h>
#include <towr/parameters.h>
#include <towr/costs/node_cost.h>
#include <towr/variables/nodes_variables_all.h>
#include <cmath>
#include <iostream>
#include <plot_trajectories.h>

using namespace towr;

// A minimal example how to build a trajectory optimization problem using TOWR.
//
// The more advanced example that includes ROS integration, GUI, rviz
// visualization and plotting can be found here:
// towr_ros/src/towr_ros_app.cc
int main() {
  // Parameters that define the motion. See c'tor for default values or
  // other values that can be modified.
  // First we define the initial phase durations, that can however be changed
  // by the optimizer. The number of swing and stance phases however is fixed.
  // alternating stance and swing:     ____-----_____-----_____-----_____

  // xpp::State3dEuler goal_geom_;
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
  terrain_ = HeightMap::FlatID;
  gait_combo_ = GaitGenerator::Combos::C0; 
  total_duration_ = 2.40;
  visualize_trajectory_ = false;
  plot_trajectory_ = false;
  replay_speed_ = 1.0;  // realtime
  optimize_ = false;
  publish_optimized_trajectory_ = false;
  optimize_phase_durations_ = false;
  int n_ee = 4;

  // NlpFormulation formulation;
  BaseState initial_base_;
  BaseState final_base_;
  std::vector<Eigen::Vector3d> initial_ee_W_;

  RobotModel model_;
  std::shared_ptr<HeightMap> terrain;
  Parameters params_;

  // terrain
  terrain = std::make_shared<FlatGround>(0.0);

  // Kinematic limits and dynamic parameters of the hopper
  model_ = RobotModel(RobotModel::Anymal);
  auto nominal_stance_B = model_.kinematic_model_->GetNominalStanceInBase();

  double z_ground = 0.0;
  initial_ee_W_ =  nominal_stance_B;
  std::for_each(initial_ee_W_.begin(), initial_ee_W_.end(),
                [&](Eigen::Vector3d& p){ p.z() = z_ground; } // feet at 0 height
  );


  // set the initial position of the hopper
  initial_base_.lin.at(kPos).z() = 0.5;

  // define the desired goal state of the hopper
  final_base_.lin.at(towr::kPos) << 1, 0.0, 0.5;

  // Instead of manually defining the initial durations for each foot and
  // step, for convenience we use a GaitGenerator with some predefined gaits
  // for a variety of robots (walk, trot, pace, ...).
  auto gait_gen_ = GaitGenerator::MakeGaitGenerator(n_ee);
  auto id_gait = static_cast<GaitGenerator::Combos>(gait_combo_);
  gait_gen_->SetCombo(id_gait);
  for (int ee = 0; ee < n_ee; ++ee) {
    initial_ee_W_.push_back(Eigen::Vector3d::Zero());
    // params_.ee_phase_durations_.push_back(
    //     gait_gen_->GetPhaseDurations(total_duration_, ee));
    // params_.ee_in_contact_at_start_.push_back(
    //     gait_gen_->IsInContactAtStart(ee));
  }
    // auto gait_gen_ = GaitGenerator::MakeGaitGenerator(n_ee);
    // auto id_gait   = static_cast<GaitGenerator::Combos>(msg.gait);
    // gait_gen_->SetCombo(id_gait);
    for (int ee=0; ee<n_ee; ++ee) {
      params_.ee_phase_durations_.push_back(gait_gen_->GetPhaseDurations(total_duration_, ee));
      params_.ee_in_contact_at_start_.push_back(gait_gen_->IsInContactAtStart(ee));
    }

  // add 

  // Here you can also add other constraints or change parameters
  // params.constraints_.push_back(Parameters::BaseRom);

  // increases optimization time, but sometimes helps find a solution for
  // more difficult terrain.
  if (optimize_phase_durations_) params_.OptimizePhaseDurations();

  // print total time 
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

  // Endeffector Motions

  std::vector<NodesVariablesPhaseBased::Ptr> ee_motion;
  double T = params_.GetTotalTime();
  for (int ee = 0; ee < params_.GetEECount(); ee++) {
    auto nodes = std::make_shared<NodesVariablesEEMotion>(
        params_.GetPhaseCount(ee), params_.ee_in_contact_at_start_.at(ee),
        ee_motion_nodes + std::to_string(ee),
        params_.ee_polynomials_per_swing_phase_);

    // initialize towards final footholds
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

  // std::cout << "params_.GetPhaseCount(" << 0 << "): " << params_.GetPhaseCount(0) << std::endl;
  

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

  // vars.insert(vars.end(), contact_schedules.begin(), contact_schedules.end());

  // stores these readily constructed spline
 towr::SplineHolder spline_holder(base_motion.at(0), // linear
                               base_motion.at(1), // angular
                               params_.GetBasePolyDurations(),
                               ee_motion,
                               ee_force,
                               contact_schedules,
                               false);
  // CONSTRAINTS


  std::vector<ifopt::ConstraintSet::Ptr> constraints;


   auto constraint = std::make_shared<DynamicConstraint>(model_.dynamic_model_,
                                                        params_.GetTotalTime(),
                                                        params_.dt_constraint_dynamic_,
                                                        spline_holder);
  constraints.push_back(constraint);
  // Initialize the nonlinear-programming problem with the variables,
  // constraints and costs.
  
  
  
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
  
  

  for (int ee=0; ee<params_.GetEECount(); ee++) {
    auto duration_constraint = std::make_shared<TotalDurationConstraint>(T, ee);
    // constraints.push_back(duration_constraint);
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
  for (auto con : constraints)  nlp.AddConstraintSet(con); 
  for (auto cost : costs) nlp.AddCostSet(cost);

  // You can add your own elements to the nlp as well, simply by calling:
  // nlp.AddVariablesSet(your_custom_variables);
  // nlp.AddConstraintSet(your_custom_constraints);

  // Choose ifopt solver (IPOPT or SNOPT), set some parameters and solve.
  // solver->SetOption("derivative_test", "first-order");
  auto solver = std::make_shared<ifopt::IpoptSolver>();
  solver->SetOption("jacobian_approximation",
                    "exact");  // "finite difference-values"
  solver->SetOption("max_cpu_time", 20.0);
  solver->Solve(nlp);

  // Can directly view the optimization variables through:
  // Eigen::VectorXd x = nlp.GetVariableValues()
  // However, it's more convenient to access the splines constructed from these
  // variables and query their values at specific times:
  using namespace std;
  cout.precision(2);
  nlp.PrintCurrent();  // view variable-set, constraint violations, indices,...
  cout << fixed;
  cout << "\n====================\nMonoped trajectory:\n====================\n";

  double t = 0.0;
  SplineHolder solution = spline_holder;
  while (t <= solution.base_linear_->GetTotalTime() + 1e-5) {
    cout << "t=" << t << "\n";
    cout << "Base linear position x,y,z:   \t";
    cout << solution.base_linear_->GetPoint(t).p().transpose() << "\t[m]"
         << endl;

    cout << "Base Euler roll, pitch, yaw:  \t";
    Eigen::Vector3d rad = solution.base_angular_->GetPoint(t).p();
    cout << (rad / M_PI * 180).transpose() << "\t[deg]" << endl;

    cout << "Foot position x,y,z:          \t";
    cout << solution.ee_motion_.at(0)->GetPoint(t).p().transpose() << "\t[m]"
         << endl;

    cout << "Contact force x,y,z:          \t";
    cout << solution.ee_force_.at(0)->GetPoint(t).p().transpose() << "\t[N]"
         << endl;

    bool contact = solution.phase_durations_.at(0)->IsContactPhase(t);
    std::string foot_in_contact = contact ? "yes" : "no";
    cout << "Foot in contact:              \t" + foot_in_contact << endl;

    cout << endl;

    t += 0.02;
  }

  plot_trajectories(solution);
  
}
