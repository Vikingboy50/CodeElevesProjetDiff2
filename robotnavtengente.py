# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 16:34:27 2026

@author: utilisateur
"""

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

# robot
x0 = -20.0
y0 = -20.0
theta0 = np.pi/4.0
robot = rob.Robot(x0, y0, theta0)
P_iso = 250

# potential
pot = Potential.Potential(difficulty=3, random=True)


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
maxLoc = False #Variable si on a valider la mission 1 (max localisé)
firstLap = True
LoopInit = False
farCheck = False

xStartLoop = 0.0
yStartLoop = 0.0

# loop on simulation time
for t in simu.t: 
   


    # position control loop
    if timerPositionCtrl.isEllapsed(t):

        potentialValue = pot.value([robot.x, robot.y])
        
        # Calcul du gradient de potentiel
        epsilon = 0.1
        grad_x = (pot.value([robot.x + epsilon, robot.y]) - potentialValue) / epsilon #dérivée partielle X
        grad_y = (pot.value([robot.x, robot.y + epsilon]) - potentialValue) / epsilon #dérivée partielle Y
            
        grad_norm = math.sqrt(grad_x*grad_x + grad_y*grad_y) #Norme2 des dérivées (grad 0 = sommet)
            
        # Arret si gradient faible et potentiel élevé (sommet atteint)
        if grad_norm < 0.1 and potentialValue > 0 and maxLoc == False:
            xPosMax, yPosMax = robot.x, robot.y
            maxLoc = True
            
        # vitesse plus faible lors de la mision 2 pour plus de précision et arret quand les deux objectifs sont atteint 
        if maxLoc == False:
            Vr = 2.0
        elif maxLoc == True and firstLap == True:
            Vr = 1
        else:
            Vr = 0
         
        if maxLoc == True:
           Tx = - grad_y
           Ty = grad_x
           k_iso = 0.05
           ux = Tx - k_iso * (potentialValue - P_iso) * grad_x
           uy = Ty - k_iso * (potentialValue - P_iso) * grad_y
           thetar = math.atan2(uy, ux)
            
        else:
            if grad_norm < 0.001:
                thetar = robot.theta
            else:
                thetar = math.atan2(grad_y, grad_x)
        
        
        
        
    # orientation control loop
    if timerOrientationCtrl.isEllapsed(t):
        # angular velocity control input     
        angle_error = math.atan2(math.sin(thetar - robot.theta), math.cos(thetar - robot.theta))
        omegar = kpOrient * angle_error
    
    
    # assign control inputs to robot
    robot.setV(Vr)
    robot.setOmega(omegar)    
    
    # integrate motion
    robot.integrateMotion(dt)

    # store data to be plotted   
    simu.addData(robot, WPManager, Vr, thetar, omegar, pot.value([robot.x,robot.y]))
    
    
# end of loop on simulation time
print(f" la source du polluant est a x={xPosMax}, y={yPosMax}, et a une valeur de pot={pot.value([xPosMax, yPosMax])}")

# close all figures
plt.close("all")

# generate plots
fig,ax = simu.plotXY(1)
pot.plot(noFigure=None, fig=fig, ax=ax)  # plot potential for verification of solution

simu.plotXYTheta(2)
#simu.plotVOmega(3)

simu.plotPotential(4)



simu.plotPotential3D(5)
