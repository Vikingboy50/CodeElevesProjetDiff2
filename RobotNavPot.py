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
pot = Potential.Potential(difficulty=2, random=True)


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

firstIter = True
firstScan = False
approachingMax = False
target_x = 0.0
target_y = 0.0
max_global = -100.0
indices = None
final_maxima = []
targets_to_visit = []


# loop on simulation time
for t in simu.t: 
   


    # position control loop
    if timerPositionCtrl.isEllapsed(t):

        potentialValue = pot.value([robot.x, robot.y])
        
        # velocity control input
        Vr = 2.0
        
        
        thetar = theta0
        if firstScan == False:

            dCheck1 = math.sqrt((robot.x - 20)**2 + (robot.y - 20)**2)
            dCheck2 = math.sqrt((robot.x - -20)**2 + (robot.y - 20)**2)
            dCheck3 = math.sqrt((robot.x - 20)**2 + (robot.y - -20)**2)
            # reference orientation
            if dCheck1 < 1.0:
                theta0 = math.atan2(20 - robot.y, -20 - robot.x)
            if dCheck2 < 1.0:
                theta0 = math.atan2(-20 - robot.y, 20 - robot.x)
            if dCheck3 < 1.0:
                # Récupération des données valides (sans les NaN de fin de tableau)
                valid_potential = simu.potential[0:simu.currentIndex]
                indices, _ = find_peaks(valid_potential)
                
                # Descente de gradient pour affiner la position des maximums
                for idx in indices:
                    mx = simu.x[idx]
                    my = simu.y[idx]
                    
                    # Gradient ascent (montée de gradient)
                    for _ in range(50):
                        delta = 0.01
                        grad_x = (pot.value([mx + delta, my]) - pot.value([mx - delta, my])) / (2 * delta)
                        grad_y = (pot.value([mx, my + delta]) - pot.value([mx, my - delta])) / (2 * delta)
                        
                        mx += 0.2 * grad_x
                        my += 0.2 * grad_y
                        
                        if (grad_x**2 + grad_y**2) < 0.001:
                            break
                    
                    # Ajout si nouveau maximum (filtre les doublons)
                    is_new = True
                    for p in final_maxima:
                        if math.sqrt((mx - p[0])**2 + (my - p[1])**2) < 1.0:
                            is_new = False
                            break
                    if is_new:
                        final_maxima.append([mx, my])
                
                targets_to_visit = list(final_maxima)
                
                if len(targets_to_visit) > 0:
                    # Trouver le pic le plus proche physiquement du robot
                    dists = [math.sqrt((p[0] - robot.x)**2 + (p[1] - robot.y)**2) for p in targets_to_visit]
                    best_idx_local = np.argmin(dists)
                    target = targets_to_visit.pop(best_idx_local)
                    
                    # Définir la cible et l'état
                    target_x = target[0]
                    target_y = target[1]
                    
                    firstScan = True
                    approachingMax = True
                else:
                    Vr = 0.0
        else:
            if approachingMax:
                # Aller vers le maximum local identifié
                theta0 = math.atan2(target_y - robot.y, target_x - robot.x)
                dist_to_target = math.sqrt((robot.x - target_x)**2 + (robot.y - target_y)**2)
                if dist_to_target < 0.5:
                    if len(targets_to_visit) > 0:
                        dists = [math.sqrt((p[0] - robot.x)**2 + (p[1] - robot.y)**2) for p in targets_to_visit]
                        best_idx_local = np.argmin(dists)
                        target = targets_to_visit.pop(best_idx_local)
                        target_x = target[0]
                        target_y = target[1]
                    else:
                        approachingMax = False
                        Vr = 0.0
            else:
                Vr = 0.0
        
        
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
