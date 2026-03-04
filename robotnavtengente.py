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
pot = Potential.Potential(difficulty=2, random=False)
nSource = pot.difficulty


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
n =0

# initialize control inputs
Vr = 0.0
thetar = 0.0
omegar = 0.0

firstIter = True
maxLoc = False #Variable si on a valider la mission 1 (max localisé)
firstLap = True
LoopInit = False
farCheck = False
histo2 = []
xStartLoop = 0.0
yStartLoop = 0.0
distance_from_entry = 0
XSearchList = []
YSearchList = []

State = 0
StateSearch = 0
posMax = np.zeros((2,nSource))
targetSelected = False
sourcesFound = 0

# loop on simulation time
for t in simu.t: 
    if xStartLoop != 0 :
        distance_from_entry = np.sqrt((xStartLoop - robot.x)**2+(yStartLoop-robot.y)**2)


    # position control loop
    if timerPositionCtrl.isEllapsed(t):

        potentialValue = pot.value([robot.x, robot.y])
        
        if State == 0:
            # Calcul du gradient de potentiel
            epsilon = 0.1
            grad_x = (pot.value([robot.x + epsilon, robot.y]) - potentialValue) / epsilon #dérivée partielle X
            grad_y = (pot.value([robot.x, robot.y + epsilon]) - potentialValue) / epsilon #dérivée partielle Y
                
            grad_norm = math.sqrt(grad_x*grad_x + grad_y*grad_y) #Norme2 des dérivées (grad 0 = sommet)
                
            # Arret si gradient faible et potentiel élevé (sommet atteint)
            if grad_norm < 0.1 and potentialValue > 0 and maxLoc == False:
                posMax[0,0] = robot.x
                posMax[1,0] = robot.y
                maxLoc = True
                sourcesFound = 1
                if sourcesFound == nSource:
                    State = 2
                
            # vitesse plus faible lors de la mision 2 pour plus de précision et arret quand les deux objectifs sont atteint 
            if State == 2:
                Vr = 0.0
            elif maxLoc == False:
                Vr = 2.0
            elif maxLoc == True and LoopInit == False :
                if potentialValue <= 250 :
                    xStartLoop = robot.x
                    yStartLoop = robot.y
                    LoopInit = True
            elif maxLoc == True and LoopInit == True and firstLap == True:
                n+=1
                if n>=300 and distance_from_entry <= 0.5:
                    firstLap  = False
                    XSearchList = np.copy(simu.x)
                    YSearchList = np.copy(simu.y)
                    # Filter XSearchList and YSearchList to keep only points where potential is approximately P_iso
                    mask = np.abs(simu.potential[:simu.currentIndex] - P_iso) < 3.0
                    XSearchList = XSearchList[:simu.currentIndex][mask]
                    YSearchList = YSearchList[:simu.currentIndex][mask]
                    XContour = XSearchList
                    YContour = YSearchList
                    xCercle = xStartLoop
                    yCercle = yStartLoop
                    Vr = 0.0
                    State = 1       
            elif maxLoc == True and firstLap == True:
                Vr = 2
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
        
        if State==1:
                if StateSearch == 0: #Enlèvment des points du cercle 
                    d = np.sqrt((xCercle - posMax[0,0])**2 + (yCercle - posMax[1,0])**2)
                    # Calculate distances of all points in the filtered list to the localized maximum
                    distances_to_max = np.sqrt((XSearchList - posMax[0,0])**2 + (YSearchList - posMax[1,0])**2)
                
                    # Remove points that are on the circle of radius d (approximately)
                    # We keep points where the distance to max is NOT close to d
                    mask_circle = np.abs(distances_to_max - d) > 0.1
                    XSearchList = XSearchList[mask_circle]
                    YSearchList = YSearchList[mask_circle]
                    StateSearch = 1
                if StateSearch == 1:
                    if len(XSearchList) > 0:
                        # Select a random index from the filtered list
                        if targetSelected == False:
                            randomIndex = np.random.randint(0, len(XSearchList))
                            targetX = XSearchList[randomIndex]
                            targetY = YSearchList[randomIndex]
                            targetSelected = True
                        
                        # Calculate distance and angle to the target point
                        distToTarget = math.sqrt((targetX - robot.x)**2 + (targetY - robot.y)**2)
                        
                        if distToTarget > 0.3:
                            Vr = 2.0
                            thetar = math.atan2(targetY - robot.y, targetX - robot.x)
                        else:
                            # Target reached, stop or pick another point in next iteration
                            Vr = 0.0
                            targetSelected = False
                            StateSearch = 2
                    else:
                        Vr = 0.0
                if StateSearch == 2:
                    # Gradient ascent to find the second maximum
                    epsilon = 0.1
                    grad_x = (pot.value([robot.x + epsilon, robot.y]) - potentialValue) / epsilon
                    grad_y = (pot.value([robot.x, robot.y + epsilon]) - potentialValue) / epsilon
                    grad_norm = math.sqrt(grad_x**2 + grad_y**2)

                    if grad_norm < 0.1:
                        # Check if new source
                        isNew = True
                        for i in range(sourcesFound):
                            if np.sqrt((robot.x - posMax[0,i])**2 + (robot.y - posMax[1,i])**2) < 1.0:
                                isNew = False
                        
                        if isNew:
                            posMax[0,sourcesFound] = robot.x
                            posMax[1,sourcesFound] = robot.y
                            sourcesFound += 1
                            if sourcesFound == nSource:
                                State = 2
                                Vr = 0.0
                            else:
                                Vr = 0.0
                                StateSearch = 1
                                targetSelected = False
                        else:
                            Vr = 0.0
                            StateSearch = 1
                            targetSelected = False
                    else:
                        Vr = 1.0
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
    print(f"State = {State}")
    print(f"StateSearch = {StateSearch}")
# end of loop on simulation time
#print(f" la source du polluant est a x={xPosMax}, y={yPosMax}, et a une valeur de pot={pot.value([xPosMax, yPosMax])}")

# close all figures
plt.close("all")

# generate plots
fig,ax = simu.plotXY(1)
pot.plot(noFigure=None, fig=fig, ax=ax)  # plot potential for verification of solution
ax.plot(robot.x, robot.y, 'b*', markersize=10, label='Final Position')
ax.legend()

simu.plotXYTheta(2)
#simu.plotVOmega(3)

simu.plotPotential(4)



simu.plotPotential3D(5)


# Generate 2D plot for XSearchList and YSearchList
plt.figure(6)
plt.clf()
plt.plot(XSearchList, YSearchList, 'go', label='Search Points (P_iso)')
plt.plot(posMax[0,:], posMax[1,:], 'r*', markersize=15, label='Localized Max')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Filtered Search Points at P_iso')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.pause(0.01)

# Animation *********************************
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-25, 25), ylim=(-25, 25))
ax.grid()
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
 
robotBody, = ax.plot([], [], 'o-', lw=2)
robotDirection, = ax.plot([], [], '-', lw=1, color='k')
wayPoint, = ax.plot([], [], 'o-', lw=2, color='b')
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
potential_template = 'potential = %.1f'
potential_text = ax.text(0.05, 0.1, '', transform=ax.transAxes)
WPArea, = ax.plot([], [], ':', lw=1, color='b')
 
thetaWPArea = np.arange(0.0,2.0*math.pi+2*math.pi/30.0, 2.0*math.pi/30.0)
xWPArea = WPManager.epsilonWP*np.cos(thetaWPArea)
yWPArea = WPManager.epsilonWP*np.sin(thetaWPArea)
 
def initAnimation():
    robotDirection.set_data([], [])
    robotBody.set_data([], [])
    wayPoint.set_data([], [])
    WPArea.set_data([], [])
    robotBody.set_color('r')
    robotBody.set_markersize(20)    
    time_text.set_text('')
    potential_text.set_text('')
    return robotBody,robotDirection, wayPoint, time_text, potential_text, WPArea  
def animate(i):  
    robotBody.set_data(simu.x[i], simu.y[i])          
    wayPoint.set_data(simu.xr[i], simu.yr[i])
    WPArea.set_data(simu.xr[i]+xWPArea.transpose(), simu.yr[i]+yWPArea.transpose())    
    thisx = [simu.x[i], simu.x[i] + 0.5*math.cos(simu.theta[i])]
    thisy = [simu.y[i], simu.y[i] + 0.5*math.sin(simu.theta[i])]
    robotDirection.set_data(thisx, thisy)
    time_text.set_text(time_template%(i*simu.dt))
    potential_text.set_text(potential_template%(pot.value([simu.x[i],simu.y[i]])))
    return robotBody,robotDirection, wayPoint, time_text, potential_text, WPArea
 
step = 100 # On ne garde qu'un point sur 100 pour la vidéo
ani = animation.FuncAnimation(fig, animate, np.arange(1, len(simu.t), step),
    interval=4, blit=True, init_func=initAnimation, repeat=False)
 
ani.save('robot.mp4', fps=15)