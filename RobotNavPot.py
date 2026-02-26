# -*- coding: utf-8 -*-
"""
Way Point navigtion

(c) S. Bertrand
"""

import math
import Robot as rob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Timer as tmr
import Potential
from scipy.signal import find_peaks

# robot
x0 = -20.0
y0 = -20.0
theta0 = np.pi/4.0
robot = rob.Robot(x0, y0, theta0)


# potential
pot = Potential.Potential(difficulty=2, random=False)


# position control loop: gain and timer
kpPos = 0.8
positionCtrlPeriod = 0.2#0.01
timerPositionCtrl = tmr.Timer(positionCtrlPeriod)

# orientation control loop: gain and timer
kpOrient = 2.5
orientationCtrlPeriod = 0.05#0.01
timerOrientationCtrl = tmr.Timer(orientationCtrlPeriod)



# list of way points list of [x coord, y coord]
WPlist = [ [x0,y0] ]
#threshold for change to next WP
epsilonWP = 0.2
# init WPManager
WPManager = rob.WPManager(WPlist, epsilonWP)


# duration of scenario and time step for numerical integration
t0 = 0.0
tf = 200.0
dt = 0.01
simu = rob.RobotSimulation(robot, t0, tf, dt)


# initialize control inputs
Vr = 0.0
thetar = 0.0
omegar = 0.0

# State Machine Variables
state = 0 # 0: Scan, 1: Approach, 2: Climb, 3: Finished
climb_state = 0 # 0: Orient, 1: Move, 2: Check, 3: Return

indices = None
targets_to_visit = []
final_maxima = []
current_target = None

climb_best_pot = -1000.0
climb_best_pos = [0.0, 0.0]
climb_start_pos = [0.0, 0.0]
climb_angle = 0.0
climb_fails = 0


# loop on simulation time
for t in simu.t: 
   


    # position control loop
    if timerPositionCtrl.isEllapsed(t):

        potentialValue = pot.value([robot.x, robot.y])
        
        # Default behavior
        Vr = 0.0
        
        if state == 0: # SCAN
            Vr = 2.0
            dCheck1 = math.sqrt((robot.x - 20)**2 + (robot.y - 20)**2)
            dCheck2 = math.sqrt((robot.x - -20)**2 + (robot.y - 20)**2)
            dCheck3 = math.sqrt((robot.x - 20)**2 + (robot.y - -20)**2)
            
            # Logic for triangle path
            if dCheck1 < 1.0:
                theta0 = math.atan2(20 - robot.y, -20 - robot.x)
            if dCheck2 < 1.0:
                theta0 = math.atan2(-20 - robot.y, 20 - robot.x)
            if dCheck3 < 1.0:
                # Scan finished
                valid_potential = simu.potential[0:simu.currentIndex]
                indices, _ = find_peaks(valid_potential)
                for idx in indices:
                    targets_to_visit.append([simu.x[idx], simu.y[idx]])
                state = 1 # Approach
            
            thetar = theta0

        elif state == 1: # APPROACH
            if current_target is None:
                if len(targets_to_visit) > 0:
                    # Find closest
                    dists = [math.sqrt((p[0] - robot.x)**2 + (p[1] - robot.y)**2) for p in targets_to_visit]
                    idx = np.argmin(dists)
                    current_target = targets_to_visit.pop(idx)
                else:
                    state = 3 # Finished
            
            if current_target is not None:
                Vr = 2.0
                thetar = math.atan2(current_target[1] - robot.y, current_target[0] - robot.x)
                dist = math.sqrt((robot.x - current_target[0])**2 + (robot.y - current_target[1])**2)
                if dist < 0.5:
                    state = 2 # Climb
                    climb_state = 0
                    climb_best_pot = potentialValue
                    climb_best_pos = [robot.x, robot.y]
                    climb_angle = 0.0
                    climb_fails = 0

        elif state == 2: # CLIMB (Gradient Ascent Physique)
            if climb_state == 0: # Orient
                Vr = 0.0
                thetar = climb_angle
                # Check alignment
                diff = climb_angle - robot.theta
                if abs(math.atan2(math.sin(diff), math.cos(diff))) < 0.1:
                    climb_state = 1
                    climb_start_pos = [robot.x, robot.y]
            
            elif climb_state == 1: # Move
                Vr = 1.0
                thetar = climb_angle
                dist = math.sqrt((robot.x - climb_start_pos[0])**2 + (robot.y - climb_start_pos[1])**2)
                if dist > 0.5: # Step size
                    climb_state = 2
            
            elif climb_state == 2: # Check
                Vr = 0.0
                if potentialValue > climb_best_pot:
                    climb_best_pot = potentialValue
                    climb_best_pos = [robot.x, robot.y]
                    climb_fails = 0
                    climb_state = 1 # Continue in same direction
                    climb_start_pos = [robot.x, robot.y]
                else:
                    climb_fails += 1
                    climb_state = 3 # Return
            
            elif climb_state == 3: # Return
                Vr = 1.0
                thetar = math.atan2(climb_best_pos[1] - robot.y, climb_best_pos[0] - robot.x)
                dist = math.sqrt((robot.x - climb_best_pos[0])**2 + (robot.y - climb_best_pos[1])**2)
                if dist < 0.1:
                    if climb_fails >= 4: # Tried all 4 directions
                        final_maxima.append(climb_best_pos)
                        current_target = None
                        state = 1 # Next target
                    else:
                        climb_angle += math.pi/2
                        climb_state = 0 # Try next angle
        
        
        if math.fabs(robot.theta-thetar)>math.pi:
            thetar = thetar + math.copysign(2*math.pi,robot.theta)        
        
        
        
    # orientation control loop
    if timerOrientationCtrl.isEllapsed(t):
        # angular velocity control input        
        omegar = kpOrient * (thetar - robot.theta)
    
    
    # assign control inputs to robot
    robot.setV(Vr)
    robot.setOmega(omegar)    
    
    # integrate motion
    robot.integrateMotion(dt)

    # store data to be plotted   
    simu.addData(robot, WPManager, Vr, thetar, omegar, pot.value([robot.x,robot.y]))
    
    
# end of loop on simulation time


# close all figures
plt.close("all")

print("Liste des maximums affinés par gradient :", final_maxima)

# generate plots
fig,ax = simu.plotXY(1, indices=indices)
pot.plot(noFigure=None, fig=fig, ax=ax)  # plot potential for verification of solution

simu.plotXYTheta(2)
#simu.plotVOmega(3)

simu.plotPotential(4)



simu.plotPotential3D(5)


# show plots
#plt.show()





# # Animation *********************************
# fig = plt.figure()
# ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-25, 25), ylim=(-25, 25))
# ax.grid()
# ax.set_xlabel('x (m)')
# ax.set_ylabel('y (m)')

# robotBody, = ax.plot([], [], 'o-', lw=2)
# robotDirection, = ax.plot([], [], '-', lw=1, color='k')
# wayPoint, = ax.plot([], [], 'o-', lw=2, color='b')
# time_template = 'time = %.1fs'
# time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
# potential_template = 'potential = %.1f'
# potential_text = ax.text(0.05, 0.1, '', transform=ax.transAxes)
# WPArea, = ax.plot([], [], ':', lw=1, color='b')

# thetaWPArea = np.arange(0.0,2.0*math.pi+2*math.pi/30.0, 2.0*math.pi/30.0)
# xWPArea = WPManager.epsilonWP*np.cos(thetaWPArea)
# yWPArea = WPManager.epsilonWP*np.sin(thetaWPArea)

# def initAnimation():
#     robotDirection.set_data([], [])
#     robotBody.set_data([], [])
#     wayPoint.set_data([], [])
#     WPArea.set_data([], [])
#     robotBody.set_color('r')
#     robotBody.set_markersize(20)    
#     time_text.set_text('')
#     potential_text.set_text('')
#     return robotBody,robotDirection, wayPoint, time_text, potential_text, WPArea  
    
# def animate(i):  
#     robotBody.set_data(simu.x[i], simu.y[i])          
#     wayPoint.set_data(simu.xr[i], simu.yr[i])
#     WPArea.set_data(simu.xr[i]+xWPArea.transpose(), simu.yr[i]+yWPArea.transpose())    
#     thisx = [simu.x[i], simu.x[i] + 0.5*math.cos(simu.theta[i])]
#     thisy = [simu.y[i], simu.y[i] + 0.5*math.sin(simu.theta[i])]
#     robotDirection.set_data(thisx, thisy)
#     time_text.set_text(time_template%(i*simu.dt))
#     potential_text.set_text(potential_template%(pot.value([simu.x[i],simu.y[i]])))
#     return robotBody,robotDirection, wayPoint, time_text, potential_text, WPArea

# ani = animation.FuncAnimation(fig, animate, np.arange(1, len(simu.t)),
#     interval=4, blit=True, init_func=initAnimation, repeat=False)
# #interval=25

# #ani.save('robot.mp4', fps=15)
