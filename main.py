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

visibility_log = []

def estimate_release_frame(wrist_idx, head_idx, x, y, z, start_frame, end_frame, speed_thresh=20):
    max_speed = 0
    release_candidate = start_frame
    for frame in range(start_frame + 1, end_frame):
        prev = np.array([x[wrist_idx, frame - 1], y[wrist_idx, frame - 1], z[wrist_idx, frame - 1]])
        curr = np.array([x[wrist_idx, frame], y[wrist_idx, frame], z[wrist_idx, frame]])
        speed = np.linalg.norm(curr - prev)

        # Optional: Only accept if wrist has passed the head forward on X-axis
        if speed > speed_thresh:
            wrist_x = x[wrist_idx, frame]
            head_x = x[head_idx, frame]  # e.g. use C7 or STRN as reference
            if wrist_x > head_x:
                return frame

        # Save max speed in case no good candidate
        if speed > max_speed:
            max_speed = speed
            release_candidate = frame

    return release_candidate

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

def get_glove_occlusion_triangles(frame, label_to_index, points):
    if not all(label in label_to_index for label in ['LSHO', 'LELB', 'LWRA', 'LWRB']):
        return []

    lsho = points[:3, label_to_index['LSHO'], frame]
    lelb = points[:3, label_to_index['LELB'], frame]
    lwra = points[:3, label_to_index['LWRA'], frame]
    lwrb = points[:3, label_to_index['LWRB'], frame]

    forearm_vec = lwra - lelb
    forearm_vec /= np.linalg.norm(forearm_vec)

    perp1 = np.cross(forearm_vec, [1, 0, 0])
    if np.linalg.norm(perp1) < 1e-6:
        perp1 = np.cross(forearm_vec, [0, 1, 0])
    perp1 /= np.linalg.norm(perp1)
    perp2 = np.cross(forearm_vec, perp1)
    perp2 /= np.linalg.norm(perp2)

    offset_1 = lwra + 150 * perp1
    offset_2 = lwra - 150 * perp1
    offset_3 = lwra + 150 * perp2
    offset_4 = lwra - 150 * perp2

    triangles = [
        (lelb, lwra, offset_1),
        (lelb, lwra, offset_2),
        (lelb, lwra, offset_3),
        (lelb, lwra, offset_4),
        (lwra, offset_1, offset_2),
        (lwra, offset_3, offset_4),
    ]
    return triangles

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

    # Glove occlusion
    glove_triangles = get_glove_occlusion_triangles(frame, label_to_index, points)
    glove_blocked = any(
        ray_intersects_triangle(catcher_pos, ray, tri) for tri in glove_triangles
    )

    # Final visibility
    visible = not (torso_blocked or glove_blocked)
    visibility_log.append(visible)

    # Wrist dot color logic
    if frame < peak_frame:
        wrist_dot._offsets3d = ([wrist[0]], [wrist[1]], [wrist[2]])
        wrist_dot.set_color('r')  # Pre-leg-lift = red
    elif peak_frame <= frame <= release_frame:
        wrist_dot._offsets3d = ([wrist[0]], [wrist[1]], [wrist[2]])
        wrist_dot.set_color('g' if visible else 'r')  # Green if visible, red if hidden
    else:
        wrist_dot._offsets3d = ([], [], [])  # Make it disappear

    ax.set_title(f"Frame {frame} - {'Visible' if visible else 'Hidden'}")
    print(f"Frame {frame:3} → {'Visible' if visible else 'Hidden'}")
    return [sc, wrist_dot]




# --- Calculate peak leg lift frame ---
peak_frame = int(np.argmax(z[label_to_index['LKNE']]))  # Use RKNE for LHPs

# --- Estimate release frame ---
release_frame = estimate_release_frame(
    wrist_idx=wrist_idx,
    head_idx=label_to_index['C7'],  # or STRN
    x=x, y=y, z=z,
    start_frame=peak_frame,
    end_frame=n_frames,
    speed_thresh=20  # Adjust if needed
)

# Run animation
ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=30, blit=False)
plt.show()

# --- Compute Visibility Score from Peak Leg Lift ---

# Use lead knee for leg lift — usually LKNE for RHP
lkne_idx = label_to_index['LKNE']
lead_knee_z = z[lkne_idx]  # vertical position over time
peak_frame = int(np.argmax(lead_knee_z))

# Use C7 or STRN as reference marker for forward position
head_idx = label_to_index['C7']

release_frame = estimate_release_frame(
    wrist_idx=wrist_idx,
    head_idx=head_idx,
    x=x, y=y, z=z,
    start_frame=peak_frame,
    end_frame=n_frames,
    speed_thresh=20  # Tune as needed
)

visible_window = visibility_log[peak_frame:release_frame]
num_visible = sum(visible_window)
total = len(visible_window)
score_percent = 100 * num_visible / total if total > 0 else 0

print("\n--- Visibility Summary ---")
print(f"From frame {peak_frame} (peak leg lift) to {release_frame} (release)")
print(f"Visible Frames: {num_visible} / {total}")
print(f"Visibility Score: {score_percent:.1f}%")

