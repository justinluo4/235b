# Lab 1 - Justin Luo and Jitao Li

#DO NOT CHANGE ANYTHING OUTSIDE OF COMMENTS
#import robot/gripper controllers
import urx
from pyrobotiqur import RobotiqGripper
#import helpers
import time
import numpy as np

#set accel/vel limits-what units?
aj = 1      # rad/s²
vj = 0.5      # rad/s

al = 0.1    # m/s²
vl = 0.03   # m/s

#robot's ip address on our private network
UR_IP = "192.168.0.2" 

#initialize the robot, loosely based on our gripper
rob = urx.Robot(UR_IP)
rob.set_tcp((0, 0, 0.1, 0, 0, 0))
rob.set_payload(1.5, (0, 0, 0.07))
time.sleep(0.2)  #leave some time to robot to process the setup commands

#initialize the gripper and connect/activate it
g = RobotiqGripper(UR_IP, port=63352, timeout=20.0)
g.connect()
g.activate()
#END DO NOT CHANGE

#BEGIN CHANGES FOR FILL IN 1
#copy and paste code from the lab into this section
#open, close the gripper, then leave it halfway
#FILL IN 1
print("Opening gripper...")
g.open(speed=128, force=10)
print("Closing gripper...")
g.close(speed=200, force=200)
p = 50 # 50% (halfway)
print(f"Moving gripper to {p}%...")
g.move_percent(p, speed=128, force=128)
#END CHANGES FOR FILL IN 1


#DO NOT CHANGE ANYTHING OUTSIDE OF COMMENTS
pos_a = (
        0.0,
        -np.pi/2-.1,
        0.1,
        -np.pi/2,
        0.0,
        0.0
)

pos_b = (
        0.0,
        -np.pi/2,
        0.0,
        -np.pi/2,
        0.0,
        0.0
)
#END DO NOT CHANGE

#BEGIN CHANGES FOR FILL IN 2
#we want to start not at a singluarity to use cartesian commands smoothly
#FILL IN 2-pos should be pos_a or pos_b?
rob.movej(pos_a, aj, vj, wait=True)
rob.stopj(aj)
print("Tool pose at start: ",  rob.getl())
startPose = rob.getl()
#END CHANGES FOR FILL IN 2

#BEGIN CHANGES FOR FILL IN 3
#FILL IN 3
#we want to move so that we avoid commands outside the workspace
#copy and paste commands from the table in the lab so that this is respected
rob.translate((0, 0, -0.2), al, vl) # Move tool Z -0.2
rob.stopl(al)
rob.translate((0.4, 0, 0), al, vl)  # Move tool X +0.4
rob.stopl(al)
rob.translate((-0.2, 0, 0), al, vl) # Move tool X -0.2
rob.stopl(al)
rob.translate((-0.2, 0, 0), al, vl) # Move tool X -0.2
rob.stopl(al)
rob.movel((0, 0, -0.2, 0, 0, 0), al, vl, relative=True, wait=True)  # move relative to current pose
rob.stopl(al)

#END CHANGES FOR FILL IN 3


#DO NOT CHANGE ANYTHING OUTSIDE OF COMMENTS
#now the robot has recentered wrt to the x axis
print("Center tool pose is: ",  rob.getl())

rob.rx -= 0.1  # rotate tool around X axis
rob.ry -= 0.1  # rotate tool around Y axis
rob.rx += 0.1  # rotate tool around X axis
rob.ry += 0.1  # rotate tool around Y axis

curPos = rob.getl()
offset = (0, 0, 0.2, 0, 0, 0)
rob.movel((curPos[0]+offset[0],curPos[1]+offset[1],curPos[2]+offset[2],curPos[3]+offset[3],curPos[4]+offset[4],curPos[5]+offset[5]), al, vl, relative=False)  # move relative to current pose
rob.stopl(al)
secondPose = rob.getl()
#END DO NOT CHANGE


#BEGIN CHANGES FOR FILL IN 4
#FILL IN 4-pos should be pos_a or pos_b? HINT: only use pos_a/pos_b once
rob.movej(pos_b, aj, vj,wait=True)
#END CHANGES FOR FILL IN 4
rob.stopj(aj)

#DO NOT CHANGE ANYTHING OUTSIDE OF COMMENTS
homePose = rob.getl()
print("Current tool pose is: ",  rob.getl())
print("The start pose was: ",  startPose)
print("The second pose was: ",  secondPose)

rob.close()
#END DO NOT CHANGE

#Prelab questions

#1. See line 10-14 for the corresponding units. aj and vj should be used with joint commands, 
# such as rob.movej() or rob.speedj(). These commands control the robot by specifying angles for each joint directly.
# al and vl should be used with linear/Cartesian commands, 
# such as rob.movel(), rob.translate(), or rob.speedl(). These commands control the robot's tool center point (TCP) 
# moving in a straight line in 3D space.

#2. pos_a is used first because it's not at a singularity (slight joint offsets), allowing smooth cartesian movements. pos_b is a singularity configuration.

#3. secondPose should be approximately 0.2 meters lower (in the Z-axis) than startPose. The X and Y coordinates should remain roughly the same.

#Postlab questions

# 1. It rotated the tool -0.1 radians around the X axis in the base frame
# 2. Yes, we got the code blocks in order. 
# 3. No, it was off by a small amount. This could be due to physical inconsistencies, model mismatch, and numerical errors.
# 4. The e-stop should be used if the robot is endangering people/equipment. 
# Otherwise, ctrl-c should be used to terminate the program and switch to another. 