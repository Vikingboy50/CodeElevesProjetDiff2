# -*- coding: utf-8 -*-
"""
Pollutant Search Robot Navigation

This script simulates a robot searching for pollutant sources using a potential field.
It uses gradient ascent, contour following, and filtering techniques.

(c) S. Bertrand
"""

import math
import Robot as rob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Timer as tmr
import Potential

# --- Configuration ---

# Robot initial state
x0 = -20.0
y0 = -20.0
theta0 = np.pi/4.0
robot = rob.Robot(x0, y0, theta0)

# Potential field (Environment)
P_iso = 200
pot = Potential.Potential(difficulty=3, random=True)
n_source = pot.difficulty

# Control loops parameters
kp_pos = 0.8
position_ctrl_period = 0.05
timer_position_ctrl = tmr.Timer(position_ctrl_period)

kp_orient = 6.0
orientation_ctrl_period = 0.01
timer_orientation_ctrl = tmr.Timer(orientation_ctrl_period)

# Waypoint Manager (Required for Simulation class, though not primarily used for navigation here)
wp_list = [[x0, y0]]
epsilon_wp = 0.2
wp_manager = rob.WPManager(wp_list, epsilon_wp)

# Simulation parameters
t0 = 0.0
tf = 200.0
dt = 0.01
simu = rob.RobotSimulation(robot, t0, tf, dt)

# --- State Variables ---

# Control inputs
v_ref = 0.0
theta_ref = 0.0
omega_ref = 0.0

# Navigation flags and counters
is_local_max_found = False  # True if the current local maximum is found
is_first_lap = True         # True if the robot is doing the first lap of the contour
is_loop_initialized = False # True if the start of the contour loop is defined
contour_step_counter = 0    # Counter for steps taken on the contour

# Contour loop tracking
x_loop_start = 0.0
y_loop_start = 0.0
dist_from_entry = 0.0

# Search algorithm variables
x_search_list = []
y_search_list = []
x_exclusion_center = 0.0
y_exclusion_center = 0.0

# Mission State Machine
# 0: Gradient Ascent / Contour Following
# 1: Search for next source (Filter -> Select Target -> Move)
# 2: Mission Complete
mission_state = 0 

# Search Sub-State Machine (when mission_state == 1)
# 0: Filter points
# 1: Move to target
# 2: Local gradient ascent
search_sub_state = 0

# Source tracking
source_positions = np.zeros((2, n_source))
is_target_selected = False
num_sources_found = 0
found_sources_log = []

# --- Main Simulation Loop ---

for t in simu.t: 
    # Update distance from the start of the loop if initialized
    if x_loop_start != 0:
        dist_from_entry = np.sqrt((x_loop_start - robot.x)**2 + (y_loop_start - robot.y)**2)

    # --- Position Control Loop ---
    if timer_position_ctrl.isEllapsed(t):

        potential_value = pot.value([robot.x, robot.y])
        
        # STATE 0: Gradient Ascent & Contour Following
        if mission_state == 0:
            # Calculate potential gradient
            epsilon = 0.1
            grad_x = (pot.value([robot.x + epsilon, robot.y]) - potential_value) / epsilon
            grad_y = (pot.value([robot.x, robot.y + epsilon]) - potential_value) / epsilon
            grad_norm = math.sqrt(grad_x**2 + grad_y**2)
                
            # Check if a peak is reached (low gradient, high potential)
            if grad_norm < 0.1 and potential_value > 0 and not is_local_max_found:
                source_positions[0, 0] = robot.x
                source_positions[1, 0] = robot.y
                is_local_max_found = True
                num_sources_found = 1
                found_sources_log.append([simu.currentIndex, robot.x, robot.y])
                if num_sources_found == n_source:
                    mission_state = 2
                
            # Velocity Logic
            if mission_state == 2:
                v_ref = 0.0
            elif not is_local_max_found:
                v_ref = 4.0 # Fast approach
            elif is_local_max_found:
                if not is_loop_initialized:
                    # Move away from peak to find contour start
                    v_ref = 1.0
                    if potential_value <= P_iso + 5.0:
                        x_loop_start = robot.x
                        y_loop_start = robot.y
                        is_loop_initialized = True
                else: # Loop initialized
                    if is_first_lap:
                        v_ref = 4.0 # Fast contour following
                        contour_step_counter += 1
                        # Check if lap is complete (enough steps taken and close to start)
                        if contour_step_counter >= 200 and dist_from_entry <= 1.0:
                            is_first_lap = False
                            # Save potential field points for filtering
                            x_search_list = np.copy(simu.x)
                            y_search_list = np.copy(simu.y)
                            # Filter to keep points near P_iso
                            mask = np.abs(simu.potential[:simu.currentIndex] - P_iso) < 3.0
                            x_search_list = x_search_list[:simu.currentIndex][mask]
                            y_search_list = y_search_list[:simu.currentIndex][mask]
                            
                            x_exclusion_center = x_loop_start
                            y_exclusion_center = y_loop_start
                            v_ref = 0.0
                            mission_state = 1
                    else:
                        v_ref = 0.0
            
            # Orientation Logic
            if is_local_max_found:
                # Contour following (tangent to gradient)
                Tx = - grad_y
                Ty = grad_x
                k_iso = 0.05
                ux = Tx - k_iso * (potential_value - P_iso) * grad_x
                uy = Ty - k_iso * (potential_value - P_iso) * grad_y
                theta_ref = math.atan2(uy, ux)
            else:
                # Gradient ascent
                if grad_norm < 0.001:
                    theta_ref = robot.theta
                else:
                    theta_ref = math.atan2(grad_y, grad_x)
        
        # STATE 1: Search for next source
        if mission_state == 1:
            # Sub-state 0: Filter points (Exclusion zone)
            if search_sub_state == 0:
                # Calculate distance from the last found source to the exclusion center
                d = np.sqrt((x_exclusion_center - source_positions[0, num_sources_found-1])**2 + 
                            (y_exclusion_center - source_positions[1, num_sources_found-1])**2)
                
                # Calculate distances of all points to the last found source
                distances_to_max = np.sqrt((x_search_list - source_positions[0, num_sources_found-1])**2 + 
                                           (y_search_list - source_positions[1, num_sources_found-1])**2)
            
                # Remove points within the exclusion ring (handling elliptical shapes)
                mask_circle = (distances_to_max < d * 0.9) | (distances_to_max > d * 1.1)
                x_search_list = x_search_list[mask_circle]
                y_search_list = y_search_list[mask_circle]
                search_sub_state = 1
            
            # Sub-state 1: Select target and move
            if search_sub_state == 1:
                if len(x_search_list) > 0:
                    if not is_target_selected:
                        random_index = np.random.randint(0, len(x_search_list))
                        target_x = x_search_list[random_index]
                        target_y = y_search_list[random_index]
                        x_exclusion_center = target_x
                        y_exclusion_center = target_y
                        is_target_selected = True
                    
                    dist_to_target = math.sqrt((target_x - robot.x)**2 + (target_y - robot.y)**2)
                    
                    if dist_to_target > 0.3:
                        v_ref = 4.0
                        theta_ref = math.atan2(target_y - robot.y, target_x - robot.x)
                    else:
                        v_ref = 0.0
                        is_target_selected = False
                        search_sub_state = 2
                else:
                    v_ref = 0.0
            
            # Sub-state 2: Local Gradient Ascent
            if search_sub_state == 2:
                epsilon = 0.1
                grad_x = (pot.value([robot.x + epsilon, robot.y]) - potential_value) / epsilon
                grad_y = (pot.value([robot.x, robot.y + epsilon]) - potential_value) / epsilon
                grad_norm = math.sqrt(grad_x**2 + grad_y**2)

                if grad_norm < 0.2:
                    # Check if it is a new source
                    is_new = True
                    for i in range(num_sources_found):
                        if np.sqrt((robot.x - source_positions[0, i])**2 + (robot.y - source_positions[1, i])**2) < 1.5:
                            is_new = False
                    
                    if is_new:
                        source_positions[0, num_sources_found] = robot.x
                        source_positions[1, num_sources_found] = robot.y
                        num_sources_found += 1
                        found_sources_log.append([simu.currentIndex, robot.x, robot.y])
                        if num_sources_found == n_source:
                            mission_state = 2
                            v_ref = 0.0
                        else:
                            v_ref = 0.0
                            search_sub_state = 0
                            is_target_selected = False
                    else:
                        # Already found, go back to searching
                        v_ref = 0.0
                        search_sub_state = 0
                        is_target_selected = False
                else:
                    v_ref = 2.0
                    theta_ref = math.atan2(grad_y, grad_x)

    # --- Orientation Control Loop ---
    if timer_orientation_ctrl.isEllapsed(t):
        angle_error = math.atan2(math.sin(theta_ref - robot.theta), math.cos(theta_ref - robot.theta))
        omega_ref = kp_orient * angle_error
    
    # Apply control inputs
    robot.setV(v_ref)
    robot.setOmega(omega_ref)    
    
    # Integrate motion
    robot.integrateMotion(dt)

    # Store data
    simu.addData(robot, wp_manager, v_ref, theta_ref, omega_ref, pot.value([robot.x, robot.y]))

# --- Post-Processing & Visualization ---

# Print comparison
real_sources = np.array(pot.mu)
print("\n--- Comparison of Found Sources vs Real Sources ---")
print("Real Sources (x, y):")
for i, source in enumerate(real_sources):
    print(f"  Source {i+1}: ({source[0]:.2f}, {source[1]:.2f})")

print("Found Sources (x, y):")
for i in range(num_sources_found):
    print(f"  Found {i+1}: ({source_positions[0,i]:.2f}, {source_positions[1,i]:.2f})")
print("---------------------------------------------------\n")

plt.close("all")

# Figure 1: Trajectory
fig, ax = simu.plotXY(1)
pot.plot(noFigure=None, fig=fig, ax=ax)
ax.plot(robot.x, robot.y, 'b*', markersize=10, label='Final Position')
ax.plot(source_positions[0, :num_sources_found], source_positions[1, :num_sources_found], 'r*', markersize=15, label='Sources Found')
ax.legend()

# Figure 2: XY Theta
simu.plotXYTheta(2)

# Figure 4: Potential
simu.plotPotential(4)

# Figure 5: 3D Potential
simu.plotPotential3D(5)

# Figure 6: Filtered Search Points
fig = plt.figure(6)
plt.clf()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-25, 25), ylim=(-25, 25))
pot.plot(fig=fig, ax=ax)
ax.plot(x_search_list, y_search_list, 'go', label='Search Points (P_iso)')
ax.plot(source_positions[0, :num_sources_found], source_positions[1, :num_sources_found], 'r*', markersize=15, label='Localized Max')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('Filtered Search Points at P_iso')
ax.legend()
ax.grid(True)
plt.pause(0.01)

# --- Animation ---
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-25, 25), ylim=(-25, 25))
pot.plot(fig=fig, ax=ax)
ax.grid()
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
 
robot_body, = ax.plot([], [], 'o-', lw=2)
robot_direction, = ax.plot([], [], '-', lw=1, color='k')
robot_path, = ax.plot([], [], '-', lw=1, color='g')
sources_plot, = ax.plot([], [], 'r*', markersize=15)
way_point, = ax.plot([], [], 'o-', lw=2, color='b')
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
potential_template = 'potential = %.1f'
potential_text = ax.text(0.05, 0.1, '', transform=ax.transAxes)
wp_area, = ax.plot([], [], ':', lw=1, color='b')
 
theta_wp_area = np.arange(0.0, 2.0*math.pi+2*math.pi/30.0, 2.0*math.pi/30.0)
x_wp_area = wp_manager.epsilonWP * np.cos(theta_wp_area)
y_wp_area = wp_manager.epsilonWP * np.sin(theta_wp_area)
 
def init_animation():
    robot_direction.set_data([], [])
    robot_body.set_data([], [])
    robot_path.set_data([], [])
    sources_plot.set_data([], [])
    way_point.set_data([], [])
    wp_area.set_data([], [])
    robot_body.set_color('r')
    robot_body.set_markersize(20)    
    time_text.set_text('')
    potential_text.set_text('')
    return robot_body, robot_direction, robot_path, sources_plot, way_point, time_text, potential_text, wp_area  

def animate(i):  
    robot_body.set_data(simu.x[i], simu.y[i])          
    robot_path.set_data(simu.x[:i], simu.y[:i])
    
    current_sources_x = [s[1] for s in found_sources_log if s[0] <= i]
    current_sources_y = [s[2] for s in found_sources_log if s[0] <= i]
    sources_plot.set_data(current_sources_x, current_sources_y)
    
    way_point.set_data(simu.xr[i], simu.yr[i])
    wp_area.set_data(simu.xr[i] + x_wp_area.transpose(), simu.yr[i] + y_wp_area.transpose())    
    
    this_x = [simu.x[i], simu.x[i] + 0.5*math.cos(simu.theta[i])]
    this_y = [simu.y[i], simu.y[i] + 0.5*math.sin(simu.theta[i])]
    robot_direction.set_data(this_x, this_y)
    
    time_text.set_text(time_template % (i * simu.dt))
    potential_text.set_text(potential_template % (pot.value([simu.x[i], simu.y[i]])))
    return robot_body, robot_direction, robot_path, sources_plot, way_point, time_text, potential_text, wp_area
 
step = 100 # Keep 1 point out of 100 for video
ani = animation.FuncAnimation(fig, animate, np.arange(1, len(simu.t), step),
    interval=4, blit=True, init_func=init_animation, repeat=False)
 
ani.save('robot.mp4', fps=15)