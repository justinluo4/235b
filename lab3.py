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
from lab2_helpers import get_all_frames, IK, transform_from_DH_modified, FK
import cv2
# from ta_utils import ExternalSafetyFilter
def ExternalSafetyFilter(joints):
    return True
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
    T_ArUco_g[:3, 3] = np.array([d / 2.0, d / 2.0, thickness / 2.0])
    T_ArUco_g[:3, 3] += T_ArUco[:3, 3]

    return T_ArUco_g

class Hanoi():
    def __init__(self, num_blocks):
        self.stacks = [[] for _ in range(3)]
        self.stacks[0] = [5, 4, 3]

    def move(self, source, target):
        target_id = 0
        source_id = self.stacks[source][-1]
        if len(self.stacks[target]) == 0:
            target_id = target
        else:
            target_id = self.stacks[target][-1]
        
        self.stacks[target].append(self.stacks[source].pop())
        return source_id, target_id

    
class RobotSim():
    def __init__(self, tool_frame=None, visualize=True, camera_offset=None,
                 camera_K=None, image_size=None, frustum_depth=0.35,
                 grasp_widths=None):
        self.joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.tool_frame = tool_frame
        self.visualize = visualize
        self.camera_offset = camera_offset
        self.camera_K = camera_K
        if image_size is None and camera_K is not None:
            image_size = (2.0 * camera_K[0, 2], 2.0 * camera_K[1, 2])
        self.image_size = image_size
        self.frustum_depth = frustum_depth
        self.grasp_widths = grasp_widths
        self.markers_in_world = {}
        self._fig = None
        self._ax = None

    def set_markers(self, markers_in_world):
        self.markers_in_world = dict(markers_in_world)
        if self.visualize:
            self._draw_arm()

    def movej(self, joints, accel, velocity, relative=False):
        self.joints = np.array(joints, dtype=float)
        if self.visualize:
            self._draw_arm()

    def getj(self):
        return self.joints

    def movel(self, pose, accel, velocity, relative=False):
        pass

    def getl(self):
        return FK(self.joints)

    def _ensure_figure(self):
        if self._fig is None or not plt.fignum_exists(self._fig.number):
            plt.ion()
            self._fig = plt.figure(figsize=(8, 7))
            self._ax = self._fig.add_subplot(111, projection='3d')

    def _draw_arm(self):
        self._ensure_figure()
        ax = self._ax
        ax.clear()

        frames = get_all_frames(self.joints, self.tool_frame)
        origins = np.array([T[:3, 3] for T in frames])

        ax.plot(origins[:, 0], origins[:, 1], origins[:, 2],
                '-o', color='black', linewidth=3, markersize=6,
                markerfacecolor='steelblue', label='links')

        axis_len = 0.08
        for i, T in enumerate(frames):
            o = T[:3, 3]
            x_ax = T[:3, 0] * axis_len
            y_ax = T[:3, 1] * axis_len
            z_ax = T[:3, 2] * axis_len
            ax.plot([o[0], o[0] + x_ax[0]], [o[1], o[1] + x_ax[1]],
                    [o[2], o[2] + x_ax[2]], color='red', linewidth=1.5)
            ax.plot([o[0], o[0] + y_ax[0]], [o[1], o[1] + y_ax[1]],
                    [o[2], o[2] + y_ax[2]], color='green', linewidth=1.5)
            ax.plot([o[0], o[0] + z_ax[0]], [o[1], o[1] + z_ax[1]],
                    [o[2], o[2] + z_ax[2]], color='blue', linewidth=1.5)

        ee = frames[-1][:3, 3]
        ax.scatter([ee[0]], [ee[1]], [ee[2]], color='crimson', s=60,
                   label='end effector')

        if self.camera_offset is not None:
            T_cam = frames[6] @ self.camera_offset
            c = T_cam[:3, 3]
            c_x = T_cam[:3, 0] * axis_len
            c_y = T_cam[:3, 1] * axis_len
            c_z = T_cam[:3, 2] * axis_len
            ax.plot([c[0], c[0] + c_x[0]], [c[1], c[1] + c_x[1]],
                    [c[2], c[2] + c_x[2]], color='red', linewidth=2)
            ax.plot([c[0], c[0] + c_y[0]], [c[1], c[1] + c_y[1]],
                    [c[2], c[2] + c_y[2]], color='green', linewidth=2)
            ax.plot([c[0], c[0] + c_z[0]], [c[1], c[1] + c_z[1]],
                    [c[2], c[2] + c_z[2]], color='blue', linewidth=2)
            ax.scatter([c[0]], [c[1]], [c[2]], color='purple', s=70,
                       marker='^', label='camera')

            fov_depth = 0.15
            fov_half = 0.08
            corners_cam = np.array([
                [ fov_half,  fov_half, fov_depth, 1.0],
                [-fov_half,  fov_half, fov_depth, 1.0],
                [-fov_half, -fov_half, fov_depth, 1.0],
                [ fov_half, -fov_half, fov_depth, 1.0],
            ]).T
            corners_world = (T_cam @ corners_cam)[:3].T
            for corner in corners_world:
                ax.plot([c[0], corner[0]], [c[1], corner[1]],
                        [c[2], corner[2]], color='purple',
                        linewidth=0.8, alpha=0.6)
            loop = np.vstack([corners_world, corners_world[0]])
            ax.plot(loop[:, 0], loop[:, 1], loop[:, 2],
                    color='purple', linewidth=0.8, alpha=0.6)

        grasp_legend_used = False
        for i, (tag_id, T) in enumerate(self.markers_in_world.items()):
            m = T[:3, 3]
            m_x = T[:3, 0] * axis_len
            m_y = T[:3, 1] * axis_len
            m_z = T[:3, 2] * axis_len
            ax.plot([m[0], m[0] + m_x[0]], [m[1], m[1] + m_x[1]],
                    [m[2], m[2] + m_x[2]], color='red', linewidth=1.5)
            ax.plot([m[0], m[0] + m_y[0]], [m[1], m[1] + m_y[1]],
                    [m[2], m[2] + m_y[2]], color='green', linewidth=1.5)
            ax.plot([m[0], m[0] + m_z[0]], [m[1], m[1] + m_z[1]],
                    [m[2], m[2] + m_z[2]], color='blue', linewidth=1.5)
            label = 'markers' if i == 0 else None
            ax.scatter([m[0]], [m[1]], [m[2]], color='orange', s=50,
                       marker='s', label=label)
            ax.text(m[0], m[1], m[2] + 0.03, f'id {tag_id}',
                    color='darkorange', fontsize=8, ha='center')

            d = None
            if self.grasp_widths is not None:
                try:
                    d = self.grasp_widths[tag_id]
                except (KeyError, IndexError, TypeError):
                    d = None
            if d is not None:
                T_grasp = get_grasp_pose(T, d)
                g = T_grasp[:3, 3]
                g_x = T_grasp[:3, 0] * axis_len
                g_y = T_grasp[:3, 1] * axis_len
                g_z = T_grasp[:3, 2] * axis_len
                ax.plot([g[0], g[0] + g_x[0]], [g[1], g[1] + g_x[1]],
                        [g[2], g[2] + g_x[2]], color='red', linewidth=1.5)
                ax.plot([g[0], g[0] + g_y[0]], [g[1], g[1] + g_y[1]],
                        [g[2], g[2] + g_y[2]], color='green', linewidth=1.5)
                ax.plot([g[0], g[0] + g_z[0]], [g[1], g[1] + g_z[1]],
                        [g[2], g[2] + g_z[2]], color='blue', linewidth=1.5)
                # Jaw line along grasp x-axis, spanning width d
                jaw_half = g_x / np.linalg.norm(g_x) * (d / 2.0) if np.linalg.norm(g_x) > 0 else np.zeros(3)
                jaw_a = g - jaw_half
                jaw_b = g + jaw_half
                ax.plot([jaw_a[0], jaw_b[0]],
                        [jaw_a[1], jaw_b[1]],
                        [jaw_a[2], jaw_b[2]],
                        color='magenta', linewidth=2, alpha=0.8)
                label_g = 'grasp pose' if not grasp_legend_used else None
                grasp_legend_used = True
                ax.scatter([g[0]], [g[1]], [g[2]], color='magenta', s=50,
                           marker='*', label=label_g)

        ax.plot([-1, 1], [0, 0], [0, 0], color='lightgray', linewidth=0.5)
        ax.plot([0, 0], [-1, 1], [0, 0], color='lightgray', linewidth=0.5)

        reach = 1.2
        ax.set_xlim(-reach, reach)
        ax.set_ylim(-reach, reach)
        ax.set_zlim(0, reach)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(
            f'UR10e pose  q(deg) = '
            f'{np.round(np.rad2deg(self.joints), 1).tolist()}'
        )
        try:
            ax.set_box_aspect((1, 1, 0.5))
        except Exception:
            pass
        ax.legend(loc='upper left', fontsize=8)

        self._fig.canvas.draw_idle()
        plt.pause(0.3)

class HanoiSolver():
    def __init__(self):
        # import urx
        # from pyrobotiqur import RobotiqGripper
        UR_IP = "192.168.0.2" 
        #initialize the robot, loosely based on our gripper
        # self.rob = urx.Robot(UR_IP)
        self.tool_frame = transform_from_DH_modified(0, 0.2, 0, 0)
        self.camera_offset = np.eye(4)
        self.camera_offset[:3, 3] = np.array([0.0, -0.1016, 0.0848])
        self.aruco_detector = ArucoDetector()
        self.block_sizes = [0.2, 0.2, 0.2, 0.06, 0.1, 0.12]
        self.rob = RobotSim(tool_frame=self.tool_frame,
                            camera_offset=self.camera_offset,
                            camera_K=self.aruco_detector.K,
                            grasp_widths=self.block_sizes)
        # self.rob.set_tcp((0, 0, 0.1, 0, 0, 0))
        # self.rob.set_payload(1.5, (0, 0, 0.07))
        time.sleep(0.2)  #leave some time to robot to process the setup commands

        #initialize the gripper and connect/activate it
        # self.g = RobotiqGripper(UR_IP, port=63352, timeout=20.0)
        # self.g.connect()
        # self.g.activate()
        self.num_blocks = 3

        self.markers_in_world = {}
        self.cam = cv2.VideoCapture(0)


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
        T_1_g[2, 3] += 0.05
        T_2_g[2, 3] += 0.10
        self.pick_and_place(T_1_g, T_2_g)
        return True


    def pick_and_place(self, start_T, target_T):
        print("Opening gripper...")
        # self.g.open(speed=128, force=10)
        print(f"Move from {start_T} to {target_T}")
        j = IK(start_T)[0]
        self.rob.movej(j, 0.5, 0.5, relative=False)
        self.rob.movel((0, 0, -0.05, 0, 0, 0), relative=True)
        print("Closing gripper...")
        # self.g.close(speed=128, force=20)
        self.rob.movel((0, 0, 0.05, 0, 0, 0), relative=True)
        j = IK(target_T)[0]
        self.rob.movej(j, 0.5, 0.5, relative=False)
        self.rob.movel((0, 0, -0.05, 0, 0, 0), relative=True)
        print("Opening gripper...")
        # self.g.open(speed=128, force=10)
        self.rob.movel((0, 0, 0.05, 0, 0, 0), relative=True)


    def marker_search(self):
        check_locations = [
            (0.3, -0.5, 0.5),
            (0.0, -0.7, 0.5),
            (-0.3, -0.7, 0.5),
        ]
        self.markers_in_world = {}
        for location in check_locations:
            look_T = tf.translation_matrix(list(location)) @ tf.euler_matrix(np.pi, 0, np.pi)
            j = IK(look_T)[0]
            print(f"Joint angles: {j}")
            assert ExternalSafetyFilter(j)
            self.rob.movej(j, 0.5, 0.5, relative=False)
            time.sleep(1)
            # ret, frame = self.cam.read()
            # if not ret:
            #     print("failed to grab frame")
            #     break
            frame = cv2.imread('aruco_detection_test_practice.png')
            self.markers_in_world.update(self.get_tags_in_world(frame))
            self.rob.set_markers(self.markers_in_world)
            input()
        for id, T in self.markers_in_world.items():
            print(f"Marker {id} in world: {T}")
        
        return self.markers_in_world

    def get_cam_frame(self, cur_j):
        T5 = get_all_frames(cur_j)[6]
        return T5 @ self.camera_offset


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
    solver = HanoiSolver()
    solver.start_solve()
    # detector = ArucoDetector()

    # frame = cv2.imread('aruco_detection_test_practice.png')
    # #frame = cv2.imread('1.png')
    # tags = detector.find_tags(frame)

    # print(f"Found {len(tags)} tag(s):")
    # # Camera is facing downwards with origin at (0.0, -0.4, 0.4) and RPY (π, 0, π)

    # cam_T = tf.translation_matrix([0.0, -0.4, 0.4]) @ tf.euler_matrix(np.pi, 0, np.pi)
    # for tag_id, T in tags:
    #     print(f"\nID: {tag_id}")
    #     print(f"  Position (x, y, z): {T[:3, 3]}")
    #     print(f"  Transform:\n{np.round(T, 4)}")
    #     world_T = cam_T @ T
    #     print(f"  world transform:\n{np.round(world_T, 4)}")
    #     grasp_T = get_grasp_pose(world_T, 0.06)
    #     print(f"  grasp transform:\n{np.round(grasp_T, 4)}")
