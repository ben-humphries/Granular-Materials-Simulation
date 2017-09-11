from __future__ import division
#from visual import *
from numpy import *
from random import random, randint


####PARAM####

R = 0.2         #radius of each ball
L = 4.0         #length of the walls
W = 4.0         #width of the walls
H = 8.0         #height of the box
IWH = 3.0           #height of the inner wall
M = 1.0         #mass of each ball
V = 0.1         #max initial velocity
T = 0.1         #wall thickness
TIMESPERSECOND = 100000           #times the loop is run per second
N = 100            #number of balls
g = array((0, -9.8, 0))*ones(N)[:,newaxis]          #N rows, each equal to [0, -9,8, 0]
e = 0.47            #coefficient of restitution
A = 4

time = 0



####INIT####

#scene.range = (L + W + IWH)/2

    ##Walls##

##walls = []
##walls = walls + [box(pos=(L + T/2.,H/2-IWH,0),axis = (-1,0,0),length = T, height = H, width = W, opacity=0.8)]          #the right wall
##walls = walls + [box(pos=(-L - T/2.,H/2-IWH,0),axis = (1,0,0),length = T, height = H, width = W, opacity=0.8)]          #the left wall
##walls = walls + [box(pos=(0,H-IWH + T/2.,0),axis = (0,-1,0),length = T, height = 2*L + T, width = W + T, opacity=0.8)]  #the top wall
##walls = walls + [box(pos=(0,H/2-IWH,-W/2.- T/2.),axis=(0,0,1),length=T, height = H + T, width = 2*L + T, opacity=0.8)]  #the back wall
##walls = walls + [box(pos=(0,H/2-IWH,W/2.+ T/2.),axis=(0,0,-1),length=T, height = H + T, width = 2*L + T,opacity=0)]     #the (invisible) front wall
##middleWall = box(pos=(0,-IWH/2,0),axis = (-1,0,0),length = T, height = IWH, width = W, opacity=0.8)
##bottomWall = box(pos=(0,-IWH - T/2.,0),axis = (0,1,0),length = T, height = 2*L + T, width = W + T, opacity=0.8)

wallposPos = array((L-R, H-IWH-R, W/2-R))*ones(N)[:,newaxis] #N rows each = [right, top, front] (adjusted for ball radius)
wallposNeg = array((-L+R, -IWH+R, -W/2+R))*ones(N)[:,newaxis] #N rows each = [left, bot, back] (adjusted for ball radius

    ##Balls##

#balls = []
poslist = []
vlist= []

while(len(poslist) < N):
    x = (L-2*R)*random() + R #x is between R and L-R
    x *= 2*randint(0,1)-1 #randomly place ball on one side of the box

    y = 2*IWH*random() - (IWH - R) #y is between -IWH+R and IWH-R
    z = (W-2*R)*random() - (W/2-R) #z is between -W/2+R and W/2-R

    #balls += [sphere(pos=(x,y,z), radius = R, color = color.red)] #add each ball to the list
    poslist += [(x,y,z)] #add each balls position to the list
    vlist += [(V*random() - V/2., V*random() - V/2., V*random() - V/2.)] #assign each ball a random velocity

pos = array(poslist) #convert to numpy array
v = array(vlist)

Dmatrix = 2*R*ones((N,N)) #NxN matrix filled with 2*R, used for checking overlaps later


####LOOP####

dt = 0.001

while 1:
    
    #rate(TIMESPERSECOND)
    time += dt

    x = pos[:,0] #all of the x coordinates of the balls
    rightlist = sort(nonzero(x>0)[0]).tolist() #coords in balls in the right half of the box
    leftlist = sort(nonzero(x<0)[0]).tolist()

    v += g*dt #update velocities
    pos += v*dt #update positions

        ##Overlapping Balls##
    
    r = pos-pos[:,newaxis] #convert all atom pairs (coords) from Nx1x3 and 1xNx3 to NxNx3
    rmag = sqrt(add.reduce(r*r, -1)) #find all distances between each pair of atoms
    hit = less_equal(rmag,Dmatrix)-identity(N) #hit = 1 if balls are colliding, else 0
    hitlist = sort(nonzero(hit.flat)[0]).tolist() #converts to N*Nx1 list containing all hits (0s and 1s). Indexed by i*N + j. 


        ##Ball Collisions##
    
    for ij in hitlist:
        i,j = divmod(ij, N) #divide and mod at the same time in order to get the index of each hit
        hitlist.remove(j*N+i) #remove equivalent pair from the list

        overlap = 2*R - rmag[j,i] #compute overlap
        direction = r[j,i]/rmag[j,i] #get direction of overlap
        deltav = v[i] - v[j] #relative velocity
        deltat = overlap/dot(deltav,direction) #compute amount to go back in time

        pos[i] -= v[i]*deltat #move balls back to where they first touched
        pos[j] -= v[j]*deltat

        vcm = (v[i] + v[j])/2 #velocity of cm frame (assumes equal masses of balls)
        vicm = v[i] - vcm
        vjcm = v[j] - vcm

        vicm -= (1+e)*dot(vicm,direction)*direction #reverse the component of velocity while dispelling some energy based on the coefficient of restitution
        vjcm -= (1+e)*dot(vjcm,direction)*direction
        v[i] = vicm + vcm #leave cm frame
        v[j] = vjcm + vcm

        pos[i] += v[i]*deltat #update velocities again based on post-collision velocity
        pos[j] += v[j]*deltat


        ##Wall Collisions##
        
    overlapping = greater_equal(pos,wallposPos) #true for balls overlapping right,top, or front walls
    overlap_v = v*overlapping #solely includes velocities of balls that are overlapping walls
    
    v = v - overlap_v - abs(overlap_v) # reverse velocity

    overlap_p = (pos-wallposPos)*overlapping #amount of overlap , solely including balls that are overlapping walls
    pos -= 2*overlap_p

    overlapping = less_equal(pos, wallposNeg)
    overlap_v = v*overlapping

    v = v - overlap_v + abs(overlap_v)

    overlap_p = (pos-wallposNeg)*overlapping
    pos -= 2*overlap_p

        ##Floor Oscillations##

    hitlist = sort(nonzero(overlapping[:,1]))[0].tolist() #coords of balls colliding with the floor
    for i in hitlist:
        v[i,1] += A*random()
        v[i,0] += (A/10)*(random() - 0.5)
        v[i,2] += (A/20)*(random() - 0.5)

    for i in rightlist:
        if (pos[i,0] < R) and (pos[i,1] < R):
            v[i,0] = abs(v[i,0]) #be sure to reflect back into right side
            pos[i,0] = pos[i,0] + 2*(R-pos[i,0])
            if (pos[i,1] > -R) and v[i,1]< 0: #collision with top of wall
                v[i,1] = -v[i,1] #also reverse y-component of velocity
                pos[i,1] = pos[i,1] + 2*(R-pos[i,1])

    for i in leftlist:
        if (pos[i,0] > -R) and (pos[i,1] < R):
            v[i,0] = -abs(v[i,0])
            pos[i,0] = pos[i,0] - 2*(R + pos[i,0])
            if (pos[i,1] > -R)and v[i,1]<0:
                v[i,1] = abs(v[i,1]) 
                pos[i,1] = pos[i,1] + 2*(R-pos[i,1])

        ##Update Ball Positions##
    if ((time + dt/4)%1.0 < dt/2) and time != 0:
        print(time,len(leftlist),len(rightlist))
    if (len(leftlist) == 0 or (len(rightlist) == 0)):
        print ('success! ',time)

##    for i in range(N):
##        balls[i].pos = pos[i]

