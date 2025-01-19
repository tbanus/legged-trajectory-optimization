#pragma once

#include <fstream>
#include <iostream>
#include <cmath>
#include <vector>
#include <towr/variables/spline_holder.h>

static void Save2CSV(const towr::SplineHolder& solution,
                     const std::string& csv_filename)
{
  std::ofstream file(csv_filename);

  // Build header (time, base pos/angles, then foot pos/force/contact per leg)
  file << "t,BasePosX,BasePosY,BasePosZ,Roll,Pitch,Yaw";
  int n_legs = solution.ee_motion_.size();
  for (int ee = 0; ee < n_legs; ++ee) {
    file << ",FootPosX_" << ee << ",FootPosY_" << ee << ",FootPosZ_" << ee
         << ",ForceX_" << ee << ",ForceY_" << ee << ",ForceZ_" << ee
         << ",InContact_" << ee;
  }
  file << "\n";

  // Write time-based data
  double t     = 0.0;
  double dt    = 0.01;
  double end_t = solution.base_linear_->GetTotalTime();

  while (t <= end_t + 1e-5) {
    auto base_lin     = solution.base_linear_->GetPoint(t).p();
    auto base_ang_rad = solution.base_angular_->GetPoint(t).p();
    auto base_ang_deg = base_ang_rad / M_PI * 180.0;

    file << t << ","
         << base_lin.x() << "," << base_lin.y() << "," << base_lin.z() << ","
         << base_ang_deg.x() << "," << base_ang_deg.y() << "," << base_ang_deg.z();

    // For each leg, log foot position, force, contact
    for (int ee = 0; ee < n_legs; ++ee) {
      auto foot_pos   = solution.ee_motion_.at(ee)->GetPoint(t).p();
      auto foot_force = solution.ee_force_.at(ee)->GetPoint(t).p();
      bool contact    = solution.phase_durations_.at(ee)->IsContactPhase(t);

      file << "," << foot_pos.x()
           << "," << foot_pos.y()
           << "," << foot_pos.z()
           << "," << foot_force.x()
           << "," << foot_force.y()
           << "," << foot_force.z()
           << "," << (contact ? "yes" : "no");
    }
    file << "\n";
    t += dt;
  }

  file.close();
}

// New function that saves phase durations to a separate file.
// Note: PhaseDurations doesn't have a GetPhaseCount() or GetPhaseDuration(),
//       so we retrieve and iterate over the vector returned by GetPhaseDurations().
static void SavePhaseDurations2CSV(const towr::SplineHolder& solution,
                                   const std::string& csv_filename)
{
  std::ofstream file(csv_filename);
  int n_legs = solution.ee_motion_.size();

  file << "PHASE DURATIONS (seconds):\n";
  for (int ee = 0; ee < n_legs; ++ee) {
    file << "Leg " << ee << ":";
    // Grab the durations vector
    auto durations = solution.phase_durations_.at(ee)->GetPhaseDurations();
    for (double phase_dur : durations) {
      file << " " << phase_dur;
    }
    file << "\n";
  }

  file.close();
}