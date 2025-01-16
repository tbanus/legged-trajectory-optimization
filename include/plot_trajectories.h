#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include "matplotlibcpp.h"
#include "towr/variables/spline_holder.h"

namespace plt = matplotlibcpp;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void save_plot(const std::string& filename, const std::string& title,
               const std::vector<double>& time_points,
               const std::vector<std::vector<double>>& data,
               const std::vector<std::string>& labels) {
    plt::figure_size(2400, 1600);
    for (size_t i = 0; i < data.size(); ++i) {
        plt::named_plot(labels[i], time_points, data[i]);
    }
    plt::title(title);
    plt::legend();
    plt::save(filename);
}

void plot_trajectories_each(const towr::SplineHolder& solution) {
    matplotlibcpp::backend("Agg");

    double t = 0.0;
    double total_time = solution.base_linear_->GetTotalTime();
    std::vector<double> time_points;

    std::vector<std::vector<double>> base_positions(3);  // X, Y, Z
    std::vector<std::vector<double>> base_angles(3);     // Roll, Pitch, Yaw
    std::vector<std::vector<std::vector<double>>> foot_positions(4, std::vector<std::vector<double>>(3)); // 4 feet, X/Y/Z
    std::vector<std::vector<std::vector<double>>> foot_forces(4, std::vector<std::vector<double>>(3));    // 4 feet, Force X/Y/Z
    std::vector<std::vector<double>> contact_phases(4); // Contact phases for 4 feet

    while (t <= total_time + 1e-5) {
        time_points.push_back(t);

        // Base positions
        Eigen::Vector3d base_pos = solution.base_linear_->GetPoint(t).p();
        base_positions[0].push_back(base_pos.x());
        base_positions[1].push_back(base_pos.y());
        base_positions[2].push_back(base_pos.z());

        // Base angles
        Eigen::Vector3d base_angle = solution.base_angular_->GetPoint(t).p();
        base_angles[0].push_back(base_angle.x() * 180.0 / M_PI);
        base_angles[1].push_back(base_angle.y() * 180.0 / M_PI);
        base_angles[2].push_back(base_angle.z() * 180.0 / M_PI);

        // Foot positions, forces, and contact phases
        for (size_t leg = 0; leg < 4; ++leg) {
            Eigen::Vector3d foot_pos = solution.ee_motion_.at(leg)->GetPoint(t).p();
            foot_positions[leg][0].push_back(foot_pos.x());
            foot_positions[leg][1].push_back(foot_pos.y());
            foot_positions[leg][2].push_back(foot_pos.z());

            Eigen::Vector3d foot_force = solution.ee_force_.at(leg)->GetPoint(t).p();
            foot_forces[leg][0].push_back(foot_force.x());
            foot_forces[leg][1].push_back(foot_force.y());
            foot_forces[leg][2].push_back(foot_force.z());

            bool is_contact = solution.phase_durations_.at(leg)->IsContactPhase(t);
            contact_phases[leg].push_back(is_contact ? 1.0 : 0.0);
        }

        t += 0.2;
    }

    // Plot base states
    save_plot("base_position.png", "Base Positions (X, Y, Z)", time_points, base_positions, {"Base X", "Base Y", "Base Z"});
    save_plot("base_angles.png", "Base Angles (Roll, Pitch, Yaw)", time_points, base_angles, {"Roll", "Pitch", "Yaw"});

    // Plot foot positions and forces for each leg
    for (size_t leg = 0; leg < 4; ++leg) {
        save_plot("foot_position_leg" + std::to_string(leg + 1) + ".png",
                  "Foot Positions (Leg " + std::to_string(leg + 1) + ")", time_points,
                  foot_positions[leg], {"Foot X", "Foot Y", "Foot Z"});

        save_plot("foot_force_leg" + std::to_string(leg + 1) + ".png",
                  "Foot Forces (Leg " + std::to_string(leg + 1) + ")", time_points,
                  foot_forces[leg], {"Force X", "Force Y", "Force Z"});

        save_plot("contact_phase_leg" + std::to_string(leg + 1) + ".png",
                  "Contact Phase (Leg " + std::to_string(leg + 1) + ")", time_points,
                  {contact_phases[leg]}, {"Contact"});
    }
}


void save_grouped_plot(const std::string& filename, const std::string& title,
                       const std::vector<double>& time_points,
                       const std::vector<std::vector<std::vector<double>>>& data,
                       const std::vector<std::string>& labels,
                       const std::vector<std::string>& subplot_titles,
                       const std::vector<std::string>& marker_styles = {"r*-", "g*-", "b*-", "k*-"} // Default markers
                       ) {
    size_t n_legs = data.size(); // Number of groups (legs)
    size_t n_components = data[0].size(); // Components per group (e.g., X/Y/Z)

    size_t rows = 2; // Fixed grid: 2x2 for 4 legs
    size_t cols = 2;

    plt::figure_size(2400, 1600);
    for (size_t leg = 0; leg < n_legs; ++leg) {
        plt::subplot(rows, cols, leg + 1);

        for (size_t component = 0; component < n_components; ++component) {
            // Check data consistency
            if (data[leg][component].size() != time_points.size()) {
                std::cerr << "Error: Data size mismatch for leg " << leg
                          << ", component " << component << std::endl;
                continue; // Skip plotting for this component
            }

            // Skip empty data
            if (data[leg][component].empty()) {
                std::cerr << "Skipping empty data for leg " << leg
                          << ", component " << component << std::endl;
                continue;
            }

            // Plot data with style and add label
            std::string style = marker_styles[component % marker_styles.size()];
            plt::plot(time_points, data[leg][component], style);
        }

        // Add a title to the subplot
        plt::title(subplot_titles[leg]);

  
    }

    // Add an overall title to the figure
    plt::suptitle(title);

    // Save the figure
    plt::save(filename);
}



void save_grouped_base_plot(const std::string& filename, const std::string& title,
                            const std::vector<double>& time_points,
                            const std::vector<std::vector<double>>& base_positions,
                            const std::vector<std::vector<double>>& base_angles) {
    plt::figure_size(2400, 1600);

    // Subplot 1: Base Linear Positions (X, Y, Z)
    plt::subplot(2, 1, 1);
    plt::plot(time_points, base_positions[0], "r*-");
    plt::plot(time_points, base_positions[1], "g*-");
    plt::plot(time_points, base_positions[2], "b*-");
    plt::title("Base Linear Positions (X, Y, Z)");
    // plt::legend({"Base X", "Base Y", "Base Z"});

    // Subplot 2: Base Angular Positions (Roll, Pitch, Yaw)
    plt::subplot(2, 1, 2);
    plt::plot(time_points, base_angles[0], "r*-");
    plt::plot(time_points, base_angles[1], "g*-");
    plt::plot(time_points, base_angles[2], "b*-");
    plt::title("Base Angular Positions (Roll, Pitch, Yaw)");
    // plt::legend({"Roll", "Pitch", "Yaw"});

    plt::suptitle(title); // Overall title
    plt::legend();
    plt::save(filename);
}

void plot_grouped_trajectories(const towr::SplineHolder& solution) {
    matplotlibcpp::backend("Agg");

    double t = 0.0;
    double total_time = solution.base_linear_->GetTotalTime();
    std::vector<double> time_points;

    std::vector<std::vector<double>> base_positions(3);  // X, Y, Z
    std::vector<std::vector<double>> base_angles(3);     // Roll, Pitch, Yaw
    std::vector<std::vector<std::vector<double>>> foot_positions(4, std::vector<std::vector<double>>(3)); // 4 feet, X/Y/Z
    std::vector<std::vector<std::vector<double>>> foot_forces(4, std::vector<std::vector<double>>(3));    // 4 feet, Force X/Y/Z
    std::vector<std::vector<double>> contact_phases(4); // Contact phases for 4 feet

    while (t <= total_time + 1e-5) {
        time_points.push_back(t);

        // Base positions
        Eigen::Vector3d base_pos = solution.base_linear_->GetPoint(t).p();
        base_positions[0].push_back(base_pos.x());
        base_positions[1].push_back(base_pos.y());
        base_positions[2].push_back(base_pos.z());

        // Base angles
        Eigen::Vector3d base_angle = solution.base_angular_->GetPoint(t).p();
        base_angles[0].push_back(base_angle.x() * 180.0 / M_PI);
        base_angles[1].push_back(base_angle.y() * 180.0 / M_PI);
        base_angles[2].push_back(base_angle.z() * 180.0 / M_PI);

        // Foot positions, forces, and contact phases
        for (size_t leg = 0; leg < 4; ++leg) {
            Eigen::Vector3d foot_pos = solution.ee_motion_.at(leg)->GetPoint(t).p();
            foot_positions[leg][0].push_back(foot_pos.x());
            foot_positions[leg][1].push_back(foot_pos.y());
            foot_positions[leg][2].push_back(foot_pos.z());

            Eigen::Vector3d foot_force = solution.ee_force_.at(leg)->GetPoint(t).p();
            foot_forces[leg][0].push_back(foot_force.x());
            foot_forces[leg][1].push_back(foot_force.y());
            foot_forces[leg][2].push_back(foot_force.z());

            bool is_contact = solution.phase_durations_.at(leg)->IsContactPhase(t);
            contact_phases[leg].push_back(is_contact ? 1.0 : 0.0);
        }

        t += 0.2;
    }

    // Save grouped base linear and angular positions plot
    save_grouped_base_plot("../figures/base_positions_and_angles.png",
                           "Base Linear and Angular Positions",
                           time_points, base_positions, base_angles);

    // Grouped plots for foot data
    save_grouped_plot("../figures/foot_positions.png", "Foot Positions for All Legs",
                      time_points, foot_positions, {"Foot X", "Foot Y", "Foot Z"},
                      {"Leg 1 Position", "Leg 2 Position", "Leg 3 Position", "Leg 4 Position"});

    save_grouped_plot("../figures/foot_forces.png", "Foot Forces for All Legs",
                      time_points, foot_forces, {"Force X", "Force Y", "Force Z"},
                      {"Leg 1 Force", "Leg 2 Force", "Leg 3 Force", "Leg 4 Force"});

    save_grouped_plot("../figures/contact_phases.png", "Contact Phases for All Legs",
                      time_points, {contact_phases}, {"Contact"},
                      {"Leg 1 Contact", "Leg 2 Contact", "Leg 3 Contact", "Leg 4 Contact"},
                      {"k*-"}); // Single marker style for contact phases
}

void plot_trajectories(const towr::SplineHolder& solution) {
    plot_grouped_trajectories(solution);
}
