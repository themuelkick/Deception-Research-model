import ezc3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Load C3D file
file = 'C:\\Users\\Michael\\pythonProject\\biodata.c3d'
c = ezc3d.c3d(file)

# Extract marker data
points = c['data']['points']  # shape: (4, num_markers, num_frames)
labels = c['parameters']['POINT']['LABELS']['value']
x, y, z = points[0, :, :], points[1, :, :], points[2, :, :]
n_markers, n_frames = x.shape

# Skeleton connections using your labels
connections = [
    # Spine and torso
    ('C7', 'CLAV'),
    ('CLAV', 'STRN'),
    ('STRN', 'T10'),
    # ('T10', 'PELV'),  # No PELV marker—can be estimated if needed

    # Head
    ('C7', 'LBHD'), ('C7', 'RBHD'),
    ('LBHD', 'LFHD'), ('RBHD', 'RFHD'),
    ('LFHD', 'RFHD'),

    # Left arm
    ('CLAV', 'LSHO'), ('LSHO', 'LELB'), ('LELB', 'LWRA'),
    ('LWRA', 'LWRB'), ('LWRA', 'LFIN'),

    # Right arm
    ('CLAV', 'RSHO'), ('RSHO', 'RELB'), ('RELB', 'RWRA'),
    ('RWRA', 'RWRB'), ('RWRA', 'RFIN'),

    # Left leg
    ('LASI', 'LTHI'), ('LTHI', 'LKNE'), ('LKNE', 'LTIB'),
    ('LTIB', 'LANK'), ('LANK', 'LTOE'), ('LANK', 'LHEE'),

    # Right leg
    ('RASI', 'RTHI'), ('RTHI', 'RKNE'), ('RKNE', 'RTIB'),
    ('RTIB', 'RANK'), ('RANK', 'RTOE'), ('RANK', 'RHEE'),
]

# Convert label pairs to index pairs
label_to_index = {label: i for i, label in enumerate(labels)}
connections_idx = [(label_to_index[a], label_to_index[b]) for a, b in connections]

# Set up 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(np.nanmin(x), np.nanmax(x))
ax.set_ylim(np.nanmin(y), np.nanmax(y))
ax.set_zlim(np.nanmin(z), np.nanmax(z))
ax.view_init(elev=15, azim=-70)
ax.set_title("3D Skeleton Animation")

# Plot markers
sc = ax.scatter(x[:, 0], y[:, 0], z[:, 0], c='r', s=40)

# Create line objects for skeleton
lines = []
for start_idx, end_idx in connections_idx:
    line, = ax.plot(
        [x[start_idx, 0], x[end_idx, 0]],
        [y[start_idx, 0], y[end_idx, 0]],
        [z[start_idx, 0], z[end_idx, 0]],
        'b-', lw=2
    )
    lines.append(line)

# Update function for animation
def update(frame):
    sc._offsets3d = (x[:, frame], y[:, frame], z[:, frame])
    for i, (start_idx, end_idx) in enumerate(connections_idx):
        lines[i].set_data([x[start_idx, frame], x[end_idx, frame]],
                          [y[start_idx, frame], y[end_idx, frame]])
        lines[i].set_3d_properties([z[start_idx, frame], z[end_idx, frame]])
    ax.set_title(f"Frame {frame}")
    return [sc] + lines

# Animate
ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=30, blit=False)
plt.show()
