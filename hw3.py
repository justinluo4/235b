import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import quaternion


def estimate_rotation(P, Q):

    P_mean = np.mean(P, axis=0)
    Q_mean = np.mean(Q, axis=0)
    P_centered = P - P_mean
    Q_centered = Q - Q_mean
    S = Q_centered.T @ P_centered
    N = np.array([
        [(S[0,0] + S[1,1] + S[2,2]),  (S[2,1] - S[1,2]),            (S[0,2] - S[2,0]),            (S[1,0] - S[0,1])          ],
        [(S[2,1] - S[1,2]),            (S[0,0] - S[1,1] - S[2,2]),   (S[0,1] + S[1,0]),            (S[0,2] + S[2,0])          ],
        [(S[0,2] - S[2,0]),            (S[0,1] + S[1,0]),            (-S[0,0] + S[1,1] - S[2,2]),   (S[1,2] + S[2,1])          ],
        [(S[1,0] - S[0,1]),            (S[0,2] + S[2,0]),            (S[1,2] + S[2,1]),            (-S[0,0] - S[1,1] + S[2,2]) ]
    ])
    eigenvalues, eigenvectors = np.linalg.eigh(N)
    q = eigenvectors[:, np.argmax(eigenvalues)]
    q = q / np.linalg.norm(q)
    if q[0] < 0: 
        q = -q
    return quaternion.from_float_array(q)

def estimate_translation(P, Q):

    P_mean = np.mean(P, axis=0)
    Q_mean = np.mean(Q, axis=0)

    t = Q_mean - P_mean
    return t

#starting with line 8
df = pd.read_csv('OptiTrack_wand_250_270m-1.csv', skiprows=7)

orientations = quaternion.from_float_array(df.iloc[:, [2, 3, 4, 5]].dropna().values)
translations = df.iloc[:, 6:9].dropna().values
translation_from_start = translations - translations[0]
orientation_from_start = orientations[0].inverse() * orientations
orientation_from_start = quaternion.as_float_array(orientation_from_start)
positions = np.array([df.iloc[:, 10 + 4*i:13 + 4*i].dropna().values for i in range(5)])
calculated_rotations = []
last_rotation = quaternion.from_float_array([1, 0, 0, 0])
for i in range(len(orientation_from_start)-1):
    calculated_rotations.append(quaternion.as_float_array(last_rotation))
    next_rotation = estimate_rotation(positions[:, i, :], positions[:, i+1, :])
    last_rotation = next_rotation * last_rotation
calculated_rotations.append(quaternion.as_float_array(last_rotation))
calculated_rotations = np.array(calculated_rotations)
calculated_rotations = calculated_rotations[:, [0, 3, 2, 1]]
calculated_rotations[:, 2:] *= -1
print(calculated_rotations)
print(orientation_from_start)

calculated_translations = []
for i in range(len(orientation_from_start)):
    calculated_translations.append(estimate_translation(positions[:, 0, :], positions[:, i, :]))
calculated_translations = np.array(calculated_translations)
print(translation_from_start)
print(calculated_translations)

all_translations = []
for i in range(len(positions[0])):
    all_translations.append(estimate_translation(positions[:, 0, :], positions[:, i, :]))
all_translations = np.array(calculated_translations)

#graph each component of the quaternion for calculated and actual rotations vs time on 4 subplots
x = np.arange(len(orientation_from_start))
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].plot(x, calculated_rotations[:, 0], label='Calculated Rotation X')
axs[0, 0].plot(x, orientation_from_start[:, 0], label='Actual Rotation X')
axs[0, 0].set_title('Calculated Rotation X vs Actual Rotation X')
axs[0, 0].legend()
axs[0, 1].plot(x, calculated_rotations[:, 1], label='Calculated Rotation Y')
axs[0, 1].plot(x, orientation_from_start[:, 1], label='Actual Rotation Y')
axs[0, 1].set_title('Calculated Rotation Y vs Actual Rotation Y')
axs[0, 1].legend()
axs[1, 0].plot(x, calculated_rotations[:, 2], label='Calculated Rotation Z')
axs[1, 0].plot(x, orientation_from_start[:, 2], label='Actual Rotation Z')
axs[1, 0].set_title('Calculated Rotation Z vs Actual Rotation Z')
axs[1, 0].legend()
axs[1, 1].plot(x, calculated_rotations[:, 3], label='Calculated Rotation W')
axs[1, 1].plot(x, orientation_from_start[:, 3], label='Actual Rotation W')
axs[1, 1].set_title('Calculated Rotation W vs Actual Rotation W')
axs[1, 1].legend()
plt.show()


t = np.arange(len(orientation_from_start))
fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
axs[0].plot(t, translation_from_start[:, 0], 'b-', label='Actual', linewidth=1.5)
axs[0].plot(t, calculated_translations[:, 0], 'r--', label='Calculated', linewidth=1.5)
axs[0].set_ylabel('X (m)')
axs[0].set_title('X Translation')
axs[0].legend()
axs[0].grid(True, alpha=0.3)
axs[1].plot(t, translation_from_start[:, 1], 'b-', label='Actual', linewidth=1.5)
axs[1].plot(t, calculated_translations[:, 1], 'r--', label='Calculated', linewidth=1.5)
axs[1].set_ylabel('Y (m)')
axs[1].set_title('Y Translation')
axs[1].legend()
axs[1].grid(True, alpha=0.3)
axs[2].plot(t, translation_from_start[:, 2], 'b-', label='Actual', linewidth=1.5)
axs[2].plot(t, calculated_translations[:, 2], 'r--', label='Calculated', linewidth=1.5)
axs[2].set_ylabel('Z (m)')
axs[2].set_xlabel('Frame')
axs[2].set_title('Z Translation')
axs[2].legend()
axs[2].grid(True, alpha=0.3)
fig.suptitle('Calculated vs Actual Translations Over Time')
plt.tight_layout()
plt.show()


fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
axs[0].plot(t, all_translations[:, 0], 'm-', linewidth=1.5)
axs[0].set_ylabel('X (m)')
axs[0].set_title('X Translation')
axs[0].grid(True, alpha=0.3)
axs[1].plot(t, all_translations[:, 1], 'm-', linewidth=1.5)
axs[1].set_ylabel('Y (m)')
axs[1].set_title('Y Translation')
axs[1].grid(True, alpha=0.3)
axs[2].plot(t, all_translations[:, 2], 'm-', linewidth=1.5)
axs[2].set_ylabel('Z (m)')
axs[2].set_xlabel('Frame')
axs[2].set_title('Z Translation')
axs[2].grid(True, alpha=0.3)
fig.suptitle('All Translations Over Time')
plt.tight_layout()
plt.show()





