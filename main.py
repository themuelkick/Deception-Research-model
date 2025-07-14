import ezc3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from scipy.spatial import ConvexHull

# Load C3D file
file = 'C:\\Users\\Michael\\pythonProject\\biodata.c3d'
c = ezc3d.c3d(file)

# Extract marker data
points = c['data']['points']
labels = c['parameters']['POINT']['LABELS']['value']
x, y, z = points[0, :, :], points[1, :, :], points[2, :, :]
n_markers, n_frames = x.shape
label_to_index = {label: i for i, label in enumerate(labels)}

# Improved triangle-ray intersection (Möller–Trumbore)
def ray_intersects_triangle(orig, dir, tri):
    v0, v1, v2 = tri
    eps = 1e-6
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(dir, edge2)
    a = np.dot(edge1, h)
    if -eps < a < eps:
        return False
    f = 1.0 / a
    s = orig - v0
    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0:
        return False
    q = np.cross(s, edge1)
    v = f * np.dot(dir, q)
    if v < 0.0 or u + v > 1.0:
        return False
    t = f * np.dot(edge2, q)
    return t > eps

# Build torso mesh
torso_markers = ['CLAV', 'STRN', 'C7', 'T10', 'RSHO', 'LSHO', 'RASI', 'LASI', 'RPSI', 'LPSI']
torso_indices = [label_to_index[m] for m in torso_markers if m in label_to_index]

# Build left arm triangles
left_arm_triangles = []
left_arm_labels = ['LSHO', 'LELB', 'LWRA', 'LWRB']
if all(label in label_to_index for label in left_arm_labels):
    lsho = label_to_index['LSHO']
    lelb = label_to_index['LELB']
    lwra = label_to_index['LWRA']
    lwrb = label_to_index['LWRB']
    left_arm_triangles = [
        (lsho, lelb, lwra),
        (lelb, lwra, lwrb),
        (lsho, lwra, lwrb),
        (lsho, lelb, lwrb)
    ]

# Catcher position (estimated from mound)
mound = np.mean([
    points[0:3, label_to_index['LASI'], 0],
    points[0:3, label_to_index['RASI'], 0]
], axis=0)
catcher_pos = mound.copy()
catcher_pos[0] += 18440  # 60.5 ft in mm

# Plot setup
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(np.nanmin(x), np.nanmax(x))
ax.set_ylim(np.nanmin(y), np.nanmax(y))
ax.set_zlim(np.nanmin(z), np.nanmax(z))
ax.set_xlabel("X (toward catcher)")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.view_init(elev=15, azim=-70)

# Initial markers
sc = ax.scatter(x[:, 0], y[:, 0], z[:, 0], c='r', s=40)
wrist_idx = label_to_index['RWRA']
wrist_dot = ax.scatter([x[wrist_idx, 0]], [y[wrist_idx, 0]], [z[wrist_idx, 0]], c='g', s=80)
ax.scatter([catcher_pos[0]], [catcher_pos[1]], [catcher_pos[2]], c='blue', s=100, label='Catcher')
ax.legend()

# Main update function
def update(frame):
    sc._offsets3d = (x[:, frame], y[:, frame], z[:, frame])
    wrist = np.array([x[wrist_idx, frame], y[wrist_idx, frame], z[wrist_idx, frame]])
    ray = wrist - catcher_pos

    # Torso occlusion
    torso_pts = np.stack([points[:3, idx, frame] for idx in torso_indices])
    try:
        torso_hull = ConvexHull(torso_pts)
        torso_blocked = any(
            ray_intersects_triangle(catcher_pos, ray, torso_pts[simplex])
            for simplex in torso_hull.simplices
        )
    except:
        torso_blocked = False

    # Left arm occlusion
    arm_blocked = False
    for i1, i2, i3 in left_arm_triangles:
        v0 = points[:3, i1, frame]
        v1 = points[:3, i2, frame]
        v2 = points[:3, i3, frame]
        triangle = np.array([v0, v1, v2])
        if ray_intersects_triangle(catcher_pos, ray, triangle):
            arm_blocked = True
            break

    visible = not (torso_blocked or arm_blocked)
    wrist_dot._offsets3d = ([wrist[0]], [wrist[1]], [wrist[2]])
    wrist_dot.set_color('g' if visible else 'r')
    ax.set_title(f"Frame {frame} - {'Visible' if visible else 'Hidden'}")
    print(f"Frame {frame:3} → {'Visible' if visible else 'Hidden'}")
    return [sc, wrist_dot]

# Run animation
ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=30, blit=False)
plt.show()
