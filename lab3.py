# Prelab Questions
# 2.a Before starting the grasp, the gripper should be around 10cm back from the object. 
# Then, once the gripper is in position and at the correct width, it can move forward to grasp the object.
# 2.b First, we must define a grasp frame for the object (for a rectangular prism, this should be on the center with the jaws along the short side). 
# Then we use the object's pose in the world frame to calculate the grasp frame in the world frame. 
# 
# 



import numpy as np
import transformations as tf
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from lab2_helpers import get_all_frames, IK, transform_from_DH_modified
import cv2

import time
from aruco import ArucoDetector

def get_grasp_pose(T_ArUco, d):

    thickness = 0.05

    T_ArUco_g = np.eye(4)
    T_ArUco_g[:3, :3] = np.array([
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0],
    ])
    T_ArUco_g[:3, 3] = np.array([d / 2.0, d / 2.0, -thickness / 2.0])

    return T_ArUco @ T_ArUco_g

class Hanoi():
    def __init__(self, num_blocks):
        self.stacks = [[] for _ in range(3)]
        self.stacks[0] = [3, 4, 5]

    def move(self, source, target):
        target_id = 0
        source_id = self.stacks[source][-1]
        if len(target) == 0:
            target_id = target
        else:
            target_id = self.stacks[target][-1]
        
        self.stacks[target].append(self.stacks[source].pop())
        return source_id, target_id

    
class HanoiSolver():
    def __init__(self):
        import urx
        from pyrobotiqur import RobotiqGripper
        UR_IP = "192.168.0.2" 
        #initialize the robot, loosely based on our gripper
        self.rob = urx.Robot(UR_IP)
        self.rob.set_tcp((0, 0, 0.1, 0, 0, 0))
        self.rob.set_payload(1.5, (0, 0, 0.07))
        time.sleep(0.2)  #leave some time to robot to process the setup commands

        #initialize the gripper and connect/activate it
        self.g = RobotiqGripper(UR_IP, port=63352, timeout=20.0)
        self.g.connect()
        self.g.activate()
        self.num_blocks = 3
        self.block_sizes = [0.2, 0.2, 0.2, 0.06, 0.1, 0.12]
        
        self.markers_in_world = {}
        self.tool_frame = transform_from_DH_modified(0, 0.2, 0, 0)
        self.cam = cv2.VideoCapture(0)
        self.aruco_detector = ArucoDetector()


    def start_solve(self):
        self.solve(Hanoi(3), 3, 0, 1, 2)

    def make_move(self, hanoi, source, target):
        self.marker_search()
        ids = hanoi.move(source, target)
        self.place_on_top(ids[0], ids[1])
        return ids

    def solve(self, hanoi, n, source, target, auxiliary):
        if n == 1:
            self.make_move(hanoi, source, target)
            return
        self.solve(hanoi, n - 1, source, auxiliary, target)
        self.make_move(hanoi, source, target)
        self.solve(hanoi, n - 1, auxiliary, target, source)

    def place_on_top(self, id_1, id_2):
        if id_1 < 3:
            print(" cannot grab target")
            return False
        if id_1 not in self.markers_in_world or id_2 not in self.markers_in_world:
            print("One or more markers not found")
            return False
        T_1 = self.markers_in_world[id_1]
        T_2 = self.markers_in_world[id_2]
        T_1_g = get_grasp_pose(T_1, self.block_sizes[id_1])
        T_2_g = get_grasp_pose(T_2, 2*self.block_sizes[id_2] - self.block_sizes[id_1])
        T_2_g[2, 3] += 0.05
        self.pick_and_place(T_1_g, T_2_g)
        return True


    def pick_and_place(self, start_T, target_T):
        print(f"Move from {start_T} to {target_T}")
        j = IK(start_T)
        self.rob.movej(j, 0.05, 0.05, relative=False)
        self.rob.movel((0, 0, -0.05, 0, 0, 0), relative=True)
        print("Closing gripper...")
        self.g.close(speed=128, force=20)
        self.rob.movel((0, 0, 0.05, 0, 0, 0), relative=True)
        j = IK(target_T)
        self.rob.movej(j, 0.05, 0.05, relative=False)
        self.rob.movel((0, 0, -0.05, 0, 0, 0), relative=True)
        print("Opening gripper...")
        self.g.open(speed=128, force=10)
        self.rob.movel((0, 0, 0.05, 0, 0, 0), relative=True)


    def marker_search(self):
        check_locations = [
            (0.2, -0.2, 0.5),
            (0.0, -0.4, 0.5),
            (-0.2, 0.2, 0.5),
        ]
        self.markers_in_world = {}
        for location in check_locations:
            self.rob.movel((location[0], location[1], location[2], np.pi, 0, np.pi), 0.05, 0.05, relative=False)
            time.sleep(1)
            ret, frame = self.cam.read()
            if not ret:
                print("failed to grab frame")
                break
            self.markers_in_world.update(self.get_tags_in_world(frame))
        return self.markers_in_world

    def get_cam_frame(self, cur_j):
        T5 = get_all_frames(cur_j, )[5]
        T_cam = np.eye(4)
        T_cam[:3, :3] = np.array([
            [1.0,  0.0,  0.0],
            [0.0, 1.0,  0.0],
            [0.0,  0.0, 1.0],
        ])
        T_cam[:3, 3] = np.array([0.0, -0.1016, 0.0848])


        return T5 @ T_cam


    def get_tags_in_world(self, frame):
        seen_tags = self.aruco_detector.find_tags(frame)
        cur_j = self.rob.getj()
        tags_in_world = {}
        cam_frame = self.get_cam_frame(cur_j)
        for tag_id, T in seen_tags:
            tags_in_world[tag_id] = cam_frame @ T
        return tags_in_world


if __name__ == "__main__":
    # hanoi = Hanoi(3)
    # solver = HanoiSolver()
    # solver.start_solve()
    detector = ArucoDetector()

    frame = cv2.imread('aruco_detection_test_practice.png')
    #frame = cv2.imread('1.png')
    tags = detector.find_tags(frame)

    print(f"Found {len(tags)} tag(s):")
    # Camera is facing downwards with origin at (0.0, -0.4, 0.4) and RPY (π, 0, π)

    cam_T = tf.translation_matrix([0.0, -0.4, 0.4]) @ tf.euler_matrix(np.pi, 0, np.pi)
    for tag_id, T in tags:
        print(f"\nID: {tag_id}")
        print(f"  Position (x, y, z): {T[:3, 3]}")
        print(f"  Transform:\n{np.round(T, 4)}")
        world_T = cam_T @ T
        print(f"  world transform:\n{np.round(world_T, 4)}")
        grasp_T = get_grasp_pose(world_T, 0.06)
        print(f"  grasp transform:\n{np.round(grasp_T, 4)}")
