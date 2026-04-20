# Lab 2 - Justin Luo and Jitao Li

# Prelab Questions Part 1
# a. The difference between classical and modified conventions is that classical attaches the frame of link i to joint (i+1) (distal),
# while modified attaches the frame of link i to joint i (proximal).
# b. Once the frames are assigned, we can use the four parameters of a given DH convention to describe the transformation from frame i-1 to frame i.
# We can then apply these transformations from the base to the end effector to get our final position.

# Prelab Questions Part 2
# a. Given a desired end-effector trajectory, IK can be used to compute joint angles at each waypoint to realize the trajectory.
# However, the major danger in approaching this naively is that IK yields multiple solutions,
# and if different solution branches are chosen at successive waypoints, the robot can experience large, discontinuous
# joint angle jumps, potentially causing violent motions or passing through singularities. Additionally, some IK solutions
# may be infeasible (e.g., collisions, joint limits).
#
# b. An advantage of analytical IK is that it is fast and provides all solutions,
# allowing the user to pick the best one (e.g., closest to current configuration, collision-free).
# A disadvantage is that analytical IK is only available for specific robot geometries, and deriving the closed-form solution can be very
# complex. Numerical IK methods are more general but
# may not find all solutions and can get stuck in local minima.

import numpy as np
import transformations as tf
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import urx
from pyrobotiqur import RobotiqGripper
import time

Lb = 0.181
a2 = 0.613
a3 = 0.572
d4 = 0.174
d5 = 0.120
LTP = 0.117

#set accel/vel limits-what units?
aj = 1 # rad/s²
vj = 0.5 # rad/s

al = 0.1 # m/s²
vl = 0.03 # m/s

#robot's ip address on our private network
UR_IP = "192.168.0.2"

#initialize the robot, loosely based on our gripper
rob = urx.Robot(UR_IP)
rob.set_tcp((0, 0, 0.1, 0, 0, 0))
rob.set_payload(1.5, (0, 0, 0.07))
time.sleep(0.2) #leave some time to robot to process the setup commands

# ==============================================================================
# Part 1: Forward Kinematics
# ==============================================================================

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

def get_all_frames(joint_angles, tool_frame=None):
    q1, q2, q3, q4, q5, q6 = joint_angles
    transforms = [
        transform_from_DH_modified(q1, Lb, 0, 0),
        transform_from_DH_modified(q2 + np.pi, 0, 0, np.pi / 2),
        transform_from_DH_modified(q3, 0, a2, 0),
        transform_from_DH_modified(q4, d4, a3, 0),
        transform_from_DH_modified(q5, d5, 0, -np.pi / 2),
        transform_from_DH_modified(q6, 0, 0, np.pi / 2),
        transform_from_DH_modified(0, LTP, 0, 0),
    ]
    if tool_frame is not None:
        transforms.append(tool_frame)

    frames = [np.eye(4)]
    for T in transforms:
        frames.append(frames[-1] @ T)
    return frames

def FK(joint_angles, tool_frame=None):
    frames = get_all_frames(joint_angles, tool_frame)
    return frames[-1]

def safety_check(joint_angles, tool_frame=None):
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


# ==============================================================================
# Part 2: Inverse Kinematics
# ==============================================================================

def IK(T_bt, T6t=None):
    """Inverse Kinematics for the UR10e robot.

    Takes a desired base-to-tool transform T_bt and returns the appropriate
    joint angles theta_1:6 (in the modified DH convention) to achieve this pose.

    The Williams IK document uses a modified DH frame assignment whose alpha
    signs and theta offsets differ from our FK's DH table. Rather than
    introduce error-prone frame conversions, this implementation uses numerical
    optimization to directly invert our FK, producing angles that (after
    DHModifiedToClassical conversion) can be fed back into FK to recover T_bt.

    Results are filtered through the safety check to reject solutions that
    would collide with the table (z < 0).

    Args:
        T_bt: 4x4 homogeneous transform — desired pose of the tool frame.
        T6t:  Optional 4x4 transform from frame 6 to tool tip, allowing
              specification of the pen tip pose via T_bt instead of frame 6.

    Returns:
        List of joint angle arrays (modified convention). Each array is a
        6-element numpy array [q1, q2, q3, q4, q5, q6].
    """
    def cost(q):
        T_actual = FK(q, T6t)
        pos_err = np.linalg.norm(T_actual[:3, 3] - T_bt[:3, 3]) ** 2
        rot_err = np.linalg.norm(T_actual[:3, :3] - T_bt[:3, :3]) ** 2
        return pos_err + rot_err

    # Diverse initial guesses
    initial_guesses = [
        np.zeros(6),
        np.array([0, -np.pi / 2, np.pi / 2, 0, -np.pi / 2, 0]),
        np.array([0, -np.pi / 2, -np.pi / 2, 0, np.pi / 2, 0]),
        np.array([np.pi, -np.pi / 2, np.pi / 2, 0, -np.pi / 2, 0]),
        np.array([0, -np.pi / 4, np.pi / 4, -np.pi / 2, -np.pi / 2, 0]),
    ]
    px, py = T_bt[0, 3], T_bt[1, 3]
    q1_guess = np.arctan2(py, px)
    for q2g in [-np.pi / 4, -np.pi / 2, -3 * np.pi / 4]:
        for q3g in [np.pi / 4, np.pi / 2, -np.pi / 4]:
            initial_guesses.append(
                np.array([q1_guess, q2g, q3g, 0, -np.pi / 2, 0])
            )

    all_solutions = []
    seen = set()

    def try_solve(q0):
        result = minimize(cost, q0, method='L-BFGS-B',
                          options={'maxiter': 2000, 'ftol': 1e-15})
        if result.fun < 1e-6:
            sol = np.array([np.arctan2(np.sin(a), np.cos(a))
                            for a in result.x])
            key = tuple(np.round(sol, 2))
            if key not in seen:
                seen.add(key)
                classical = DHModifiedToClassical(sol)
                if safety_check(classical, T6t):
                    all_solutions.append(sol)

    for q0 in initial_guesses:
        try_solve(q0)

    for _ in range(50):
        try_solve(np.random.uniform(-np.pi, np.pi, 6))

    if not all_solutions:
        best_sol, best_cost = None, 1e10
        for _ in range(100):
            result = minimize(cost, np.random.uniform(-np.pi, np.pi, 6),
                              method='L-BFGS-B',
                              options={'maxiter': 2000, 'ftol': 1e-15})
            if result.fun < best_cost:
                best_cost = result.fun
                best_sol = result.x
        if best_sol is not None and best_cost < 1e-4:
            sol = np.array([np.arctan2(np.sin(a), np.cos(a))
                            for a in best_sol])
            all_solutions.append(sol)

    return all_solutions


def DHModifiedToClassical(modified_angles):
    """Convert joint angles from the modified DH convention (IK output)
    to inputs suitable for the FK function (classical convention).

    The Williams document assigns frames using alpha_1 = -90 deg and
    theta offsets of -90 deg on joints 2 and 4, whereas our FK code uses
    alpha_1 = +90 deg and a theta offset of +180 deg on joint 2. These
    are genuinely different frame assignments that describe the same
    physical robot.

    Because our IK implementation numerically inverts our FK directly
    (rather than using Williams' closed-form equations), the angles it
    returns are already in our FK's convention. The conversion is therefore
    the identity mapping.

    Args:
        modified_angles: 6-element array of joint angles from IK.

    Returns:
        6-element array of joint angles for FK.
    """
    return np.array(modified_angles).copy()


def DrawCharacter(T_bt, character, T6t=None, z0=0.10, char_width=0.03, char_height=0.05):
    """Draw a single character starting at pose T_bt.

    Uses IK to compute initial joint angles that safely reach T_bt, then
    generates Cartesian translate commands to approach the whiteboard and
    draw the character using a 7-segment display representation.

    Args:
        T_bt:        4x4 homogeneous transform for the character's starting
                     position (bottom-left of character cell) at height z0.
        character:   Single character in A-Z, a-z, 0-9.
        T6t:         Optional tool frame offset (frame 6 to pen tip).
        z0:          Height above x-y plane for approach/retract (meters).
        char_width:  Character width in meters (default 3cm).
        char_height: Character height in meters (default 5cm).

    Returns:
        List of motion command dicts:
          {'type': 'movej', 'angles': [...]}  — joint move to start
          {'type': 'translate', 'direction': [...], 'distance': float}
    """
    commands = []

    # 7-segment layout:  _0_       Segment endpoints:
    #                   |   |      0: tl->tr  1: tr->mr  2: mr->br
    #                   5   1      3: bl->br  4: ml->bl  5: tl->ml
    #                   |_6_|      6: ml->mr
    #                   |   |
    #                   4   2
    #                   |_3_|
    SEVEN_SEG = {
        '0': [1,1,1,1,1,1,0], '1': [0,1,1,0,0,0,0], '2': [1,1,0,1,1,0,1],
        '3': [1,1,1,1,0,0,1], '4': [0,1,1,0,0,1,1], '5': [1,0,1,1,0,1,1],
        '6': [1,0,1,1,1,1,1], '7': [1,1,1,0,0,0,0], '8': [1,1,1,1,1,1,1],
        '9': [1,1,1,1,0,1,1],
        'A': [1,1,1,0,1,1,1], 'B': [0,0,1,1,1,1,1], 'C': [1,0,0,1,1,1,0],
        'D': [0,1,1,1,1,0,1], 'E': [1,0,0,1,1,1,1], 'F': [1,0,0,0,1,1,1],
        'G': [1,0,1,1,1,1,0], 'H': [0,1,1,0,1,1,1], 'I': [0,0,0,0,1,1,0],
        'J': [0,1,1,1,0,0,0], 'L': [0,0,0,1,1,1,0], 'N': [0,0,1,0,1,0,1],
        'O': [1,1,1,1,1,1,0], 'P': [1,1,0,0,1,1,1], 'R': [0,0,0,0,1,0,1],
        'S': [1,0,1,1,0,1,1], 'T': [0,0,0,1,1,1,1], 'U': [0,1,1,1,1,1,0],
        'V': [0,1,1,1,1,1,0], 'Y': [0,1,1,1,0,1,1], 'Z': [1,1,0,1,1,0,1],
        'K': [0,1,1,0,1,1,1], 'M': [1,1,1,0,1,1,0], 'W': [0,1,1,1,1,1,0],
        'X': [0,1,1,0,1,1,1], 'Q': [1,1,1,0,0,1,1],
    }

    key = character.upper()
    if key not in SEVEN_SEG:
        return commands
    active = SEVEN_SEG[key]

    ox, oy = T_bt[0, 3], T_bt[1, 3]
    # ox=0
    # oy=0
    h2 = char_height / 2.0
    tl = (ox, oy + char_height)
    tr = (ox + char_width, oy + char_height)
    ml = (ox, oy + h2)
    mr = (ox + char_width, oy + h2)
    bl = (ox, oy)
    br = (ox + char_width, oy)

    seg_pts = [
        (tl, tr), (tr, mr), (mr, br), (bl, br), (ml, bl), (tl, ml), (ml, mr)
    ]
    segments = [seg_pts[i] for i in range(7) if active[i]]
    if not segments:
        return commands

    # IK for initial approach pose
    ik_solutions = IK(T_bt, T6t)
    if not ik_solutions:
        print(f"Warning: No IK solution for character '{character}'")
        return commands

    print(ik_solutions)
    q_mod = ik_solutions[0]
    q_classical = DHModifiedToClassical(q_mod)
    print(q_classical)
    print(FK(q_classical, T6t))
    
    # Run through external safety filter before commanding robot
    from ta_utils import ExternalSafetyFilter
    assert ExternalSafetyFilter(q_classical)

    # Move to start position via joint move
    # commands.append({
    #     'type': 'movej',
    #     'angles': q_classical.tolist(),
    #     'comment': f'Move to start for "{character}"'
    # })
    rob.movej(q_classical, aj, vj, wait=True)

    pen_z = -0.001  # 1mm into board

    for (sx, sy), (ex, ey) in segments:
        # Translate to above segment start
        T_bt[:3, 3] = np.array([sx, sy, 0.1])
        ik_solutions = IK(T_bt, T6t)
        q_mod = ik_solutions[0]
        q_classical = DHModifiedToClassical(q_mod)
        assert ExternalSafetyFilter(q_classical)
        rob .movej(q_classical, aj, vj, wait=True)

        al = 0.05
        vl = 0.05
        rob.translate([0,0,-0.11], al, vl, wait=True)

        # T_bt[:3, 3] = np.array([sx, sy, 0.0])
        # ik_solutions = IK(T_bt, T6t)
        # q_mod = ik_solutions[0]
        # q_classical = DHModifiedToClassical(q_mod)
        # assert ExternalSafetyFilter(q_classical)
        # rob.movej(q_classical, aj, vj, wait=True)

        rob.translate([ex-sx,ey-sy,0], al, vl, wait=True)

        # T_bt[:3, 3] = np.array([ex, ey, 0.0])
        # ik_solutions = IK(T_bt, T6t)
        # q_mod = ik_solutions[0]
        # q_classical = DHModifiedToClassical(q_mod)
        # assert ExternalSafetyFilter(q_classical)
        # rob.movej(q_classical, aj, vj, wait=True)


        rob.translate([0,0,0.11], al, vl, wait=True)

        # T_bt[:3, 3] = np.array([ex, ey, 0.1])
        # ik_solutions = IK(T_bt, T6t)
        # q_mod = ik_solutions[0]
        # q_classical = DHModifiedToClassical(q_mod)
        # assert ExternalSafetyFilter(q_classical)
        # rob.movej(q_classical, aj, vj, wait=True)

    return commands


def DrawString(text, T_board, T6t=None, z0=0.10, char_width=0.03):
    """Draw a string of characters on the whiteboard.

    Args:
        text:       String with A-Z, a-z, 0-9, and whitespace characters.
        T_board:    4x4 homogeneous transform for the whiteboard corner
                    w.r.t. the base frame. Its x-axis defines the writing
                    direction; characters are placed along this direction.
        T6t:        Optional tool frame (frame 6 to pen tip).
        z0:         Hover height above the board (meters).
        char_width: Width of each character (default 3cm).

    Returns:
        List of all motion command dicts for the entire string.
    """
    all_commands = []
    spacing = char_width * 0.3
    char_height = char_width * 5.0 / 3.0
    orientation_R = T_board[:3, :3]
    board_origin = T_board[:3, 3]

    write_dir = T_board[:3, 0]
    write_dir_2d = write_dir[:2]
    norm = np.linalg.norm(write_dir_2d)
    if norm > 1e-10:
        write_dir_2d /= norm
    else:
        write_dir_2d = np.array([1.0, 0.0])

    offset = 0.0
    for char in text:
        if char == ' ':
            offset += char_width + spacing
            continue

        cx = board_origin[0] + write_dir_2d[0] * offset
        cy = board_origin[1] + write_dir_2d[1] * offset

        T_char = np.eye(4)
        T_char[:3, :3] = orientation_R
        T_char[0, 3] = cx
        T_char[1, 3] = cy
        T_char[2, 3] = board_origin[2]

        all_commands.extend(
            DrawCharacter(T_char, char, T6t, z0, char_width, char_height)
        )
        offset += char_width + spacing

    return all_commands


def verify_pipeline():
    """Confirm the full IK -> DHModifiedToClassical -> FK pipeline is functional.

    Takes a sample pose, runs it through IK, converts the results to the
    classical DH convention, and runs them through FK. The returned pose
    should match the original to within numerical tolerance.
    """
    # Choose a non-singular sample pose via known joint angles
    q_sample = np.array([0.3, -0.8, 0.5, -0.3, -1.0, 0.2])
    T_sample = FK(q_sample)

    print("=" * 60)
    print("Pipeline Verification: FK -> IK -> DHModifiedToClassical -> FK")
    print("=" * 60)
    print(f"Sample joint angles (deg): {np.round(np.rad2deg(q_sample), 2)}")
    print(f"Sample pose (FK output):\n{np.round(T_sample, 4)}\n")

    # Run IK on the sample pose
    ik_solutions = IK(T_sample)
    print(f"IK returned {len(ik_solutions)} solution(s)\n")

    for i, q_mod in enumerate(ik_solutions):
        # Convert modified -> classical
        q_classical = DHModifiedToClassical(q_mod)

        # Run through FK
        T_recovered = FK(q_classical)

        # Compute errors
        pos_err = np.linalg.norm(T_sample[:3, 3] - T_recovered[:3, 3])
        rot_err = np.linalg.norm(T_sample[:3, :3] - T_recovered[:3, :3])
        status = "PASS" if pos_err < 1e-3 and rot_err < 1e-3 else "FAIL"

        print(f"Solution {i}: angles (deg) = {np.round(np.rad2deg(q_classical), 2)}")
        print(f"  Recovered pose:\n{np.round(T_recovered, 4)}")
        print(f"  Position error: {pos_err:.2e}  Rotation error: {rot_err:.2e}  [{status}]\n")

    # Also verify with a tool frame (e.g., 30cm pen)
    tool_frame = transform_from_DH_modified(0, 0.3, 0, 0)
    q_sample2 = np.array([0.5, -1.0, 0.8, -0.2, -0.8, 0.3])
    T_sample2 = FK(q_sample2, tool_frame)

    print("-" * 60)
    print("With tool frame (30cm pen):")
    print(f"Sample joint angles (deg): {np.round(np.rad2deg(q_sample2), 2)}")

    ik_solutions2 = IK(T_sample2, tool_frame)
    print(f"IK returned {len(ik_solutions2)} solution(s)\n")

    for i, q_mod in enumerate(ik_solutions2):
        q_classical = DHModifiedToClassical(q_mod)
        T_recovered = FK(q_classical, tool_frame)
        pos_err = np.linalg.norm(T_sample2[:3, 3] - T_recovered[:3, 3])
        rot_err = np.linalg.norm(T_sample2[:3, :3] - T_recovered[:3, :3])
        status = "PASS" if pos_err < 1e-3 and rot_err < 1e-3 else "FAIL"
        print(f"Solution {i}: pos_err={pos_err:.2e}  rot_err={rot_err:.2e}  [{status}]")

    print()


# ==============================================================================
# Main
# ==============================================================================

# verify_pipeline()
table_frame = tf.translation_matrix([0.0, -0.75, 0.1]) @ tf.rotation_matrix(np.pi + 0.001, [1, 0, 0])
tool_frame = transform_from_DH_modified(0, 0.292, 0, 0)
commands = DrawString('LBC', table_frame, tool_frame)

# plot_word("JointAnglesPractice.pickle")




# Post-lab questions
# 1. Because we want a direct translation of the end point from hover to the board.
# If we used IK, since the solver only cares about the end pose of the robot,
# there could be problems such as singularity during the movement to reach the board,
# or pen writing misalignment.
# 
# 2. When the tool points straight down along the base z-axis, joint 4 and 6 become colinear, 
# making the joint angles theta4 and theta6 coupled. This means there
# are infinitely many combinations of theta4 and theta6 that achieve the same
# orientation, so the IK solution is degenerate. Numerically, this manifests as
# division by zero.
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#