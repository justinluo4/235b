# Lab 1 - Justin Luo and Jitao Li
#Prelab Questions Part 1
# a. The difference between classical and modified conventions is that classical attaches the frame of link i to joint (i+1) (distal),
# while modified attaches the frame of link i to joint i (proximal). 
# b. Once the frames are assigned, we can use the four parameters of a given DH convention to describe the transformation from frame i-1 to frame i.
# We can then apply these transformations from the base to the end effector to get our final position.

import numpy as np
import transformations as tf
import pickle
import matplotlib.pyplot as plt
Lb = 0.181
a2 = 0.613
a3 = 0.572
d4 = 0.174
d5 = 0.120
LTP = 0.117

def transform_from_DH_classic(theta, d, a, alpha):
    z_rotation = tf.rotation_matrix(theta, [0, 0, 1])
    translation = tf.translation_matrix([a, 0.0, d])
    x_rotation = tf.rotation_matrix(alpha, [1, 0, 0])
    return z_rotation @ translation @ x_rotation

def transform_from_DH_modified(theta, d, a, alpha):
    z_rotation = tf.rotation_matrix(theta, [0, 0, 1])
    z_translation = tf.translation_matrix([0.0, 0.0, d])
    x_rotation = tf.rotation_matrix(alpha, [1, 0, 0])
    x_translation = tf.translation_matrix([a, 0.0, 0.0])
    return x_rotation @ x_translation @ z_rotation @ z_translation
    
def get_all_frames(joint_angles, tool_frame = None):
    q1, q2, q3, q4, q5, q6 = joint_angles
    transforms = [
        transform_from_DH_modified(q1, Lb, 0, 0),
        transform_from_DH_modified(q2 + np.pi, 0, 0, np.pi/2),
        transform_from_DH_modified(q3, 0, a2, 0),
        transform_from_DH_modified(q4, d4, a3, 0),
        transform_from_DH_modified(q5, d5, 0, -np.pi/2),
        transform_from_DH_modified(q6, 0, 0, np.pi/2),
        transform_from_DH_modified(0, LTP, 0, 0),
    ]
    if tool_frame is not None:
        transforms.append(tool_frame)

    frames = [np.eye(4)]
    for T in transforms:
        frames.append(frames[-1] @ T)
    return frames

def FK(joint_angles, tool_frame = None):
    frames = get_all_frames(joint_angles, tool_frame)
    
    return frames[-1]




def safety_check(joint_angles, tool_frame = None):
    for frame in get_all_frames(joint_angles, tool_frame):
        if frame[2, 3] < 0:
            return False

    return True

def plot_word(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    fig, ax = plt.subplots()
    tool_frame = transform_from_DH_modified(0, 0.3, 0, 0)
    for letter in data:
        for segment in letter:
            assert safety_check(segment[0], tool_frame)
            assert safety_check(segment[1], tool_frame)
            p1 = FK(segment[0], tool_frame)
            p2 = FK(segment[1], tool_frame)
            ax.plot([p1[0][3], p2[0][3]], [p1[1][3], p2[1][3]],
                    color='black', linewidth=2)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Word Plot')
    plt.show()






plot_word("JointAnglesPractice.pickle")

