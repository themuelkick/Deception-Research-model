import streamlit as st
import ezc3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from io import BytesIO
import tempfile
from visibility_utils import (
    ray_intersects_triangle,
    get_glove_occlusion_triangles,
    estimate_release_frame,
    get_visibility_score
)

st.set_page_config(layout="wide")
st.title("Pitch Visibility Scoring App")



# Upload .c3d file
uploaded_file = st.file_uploader("Upload a .c3d biomechanics file", type=["c3d"])



if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".c3d") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Load C3D file from temp path
    c = ezc3d.c3d(tmp_path)

    points = c['data']['points']
    labels = c['parameters']['POINT']['LABELS']['value']
    x, y, z = points[0, :, :], points[1, :, :], points[2, :, :]
    n_markers, n_frames = x.shape
    label_to_index = {label: i for i, label in enumerate(labels)}

    # Automatically detect handedness
    st.markdown("### Detecting Handedness...")

    rwra_idx = None
    lwra_idx = None

    if 'RWRA' in label_to_index:
        rwra_idx = label_to_index['RWRA']
    if 'LWRA' in label_to_index:
        lwra_idx = label_to_index['LWRA']

    if rwra_idx is not None and lwra_idx is not None:
        # Estimate motion in X direction over the throw
        peak_frame_guess = int(np.argmax(z[label_to_index['LKNE']]) if 'LKNE' in label_to_index else 0)
        end_frame = min(peak_frame_guess + 60, x.shape[1])

        rwra_x = x[rwra_idx, peak_frame_guess:end_frame]
        lwra_x = x[lwra_idx, peak_frame_guess:end_frame]

        rwra_delta = rwra_x[-1] - rwra_x[0]
        lwra_delta = lwra_x[-1] - lwra_x[0]

        handedness = "RHP" if rwra_delta > lwra_delta else "LHP"
        st.success(f"✅ Automatically detected: **{handedness}**")
    else:
        handedness = st.selectbox("Select pitcher handedness", ["RHP", "LHP"])
        st.warning("Could not detect handedness automatically. Please select manually.")

    # Wrist marker (throwing hand)
    wrist_label = 'RWRA' if handedness == 'RHP' else 'LWRA'
    wrist_idx = label_to_index[wrist_label]

    # Peak leg lift
    knee_label = 'LKNE' if handedness == 'RHP' else 'RKNE'
    peak_frame = int(np.argmax(z[label_to_index[knee_label]]))

    # Estimate release
    head_idx = label_to_index['C7'] if 'C7' in label_to_index else label_to_index['STRN']
    release_frame = estimate_release_frame(
        wrist_idx=wrist_idx,
        head_idx=head_idx,
        x=x, y=y, z=z,
        start_frame=peak_frame,
        end_frame=n_frames
    )

    # Catcher position (estimated from mound)
    mound = np.mean([
        points[0:3, label_to_index['LASI'], 0],
        points[0:3, label_to_index['RASI'], 0]
    ], axis=0)
    catcher_pos = mound.copy()
    catcher_pos[0] += 18440  # 60.5 ft in mm

    # Torso indices
    torso_markers = ['CLAV', 'STRN', 'C7', 'T10', 'RSHO', 'LSHO', 'RASI', 'LASI', 'RPSI', 'LPSI']
    torso_indices = [label_to_index[m] for m in torso_markers if m in label_to_index]

    # Run visibility score
    visibility_log, score_percent = get_visibility_score(
        x, y, z, points,
        wrist_idx=wrist_idx,
        catcher_pos=catcher_pos,
        torso_indices=torso_indices,
        label_to_index=label_to_index,
        peak_frame=peak_frame,
        release_frame=release_frame
    )

    st.subheader("Visibility Summary")
    st.write(f"**From frame {peak_frame} (peak leg lift) to {release_frame} (release):**")
    st.write(f"**Visibility Score:** {score_percent:.1f}%")

    # Plot basic trajectory (optional)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x[wrist_idx], y[wrist_idx], z[wrist_idx], label="Throwing Hand")
    ax.scatter(catcher_pos[0], catcher_pos[1], catcher_pos[2], c='blue', label="Catcher")
    ax.set_title("Wrist Path to Catcher")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    st.pyplot(fig)

    import time

    # Add speed slider above playback loop
    speed = st.slider("Animation delay (seconds per frame)", min_value=0.000, max_value=0.030, value=0.005, step=0.001)

    if st.button("Play Throw Animation"):
        plot_placeholder = st.empty()

        for frame in range(n_frames):
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')

            # Plot all markers
            ax.scatter(x[:, frame], y[:, frame], z[:, frame], c='gray', s=20, label="Markers")

            # Highlight wrist marker
            wrist = [x[wrist_idx, frame], y[wrist_idx, frame], z[wrist_idx, frame]]
            ax.scatter(*wrist, c='red', s=60, label="Wrist")

            # Plot catcher
            ax.scatter(*catcher_pos, c='blue', s=80, label="Catcher")

            ax.set_xlim(np.nanmin(x), np.nanmax(x))
            ax.set_ylim(np.nanmin(y), np.nanmax(y))
            ax.set_zlim(np.nanmin(z), np.nanmax(z))
            ax.set_xlabel("X (to catcher)")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title(f"Frame {frame}")
            ax.view_init(elev=15, azim=-70)
            ax.legend()

            plot_placeholder.pyplot(fig)

            if speed > 0:
                time.sleep(speed)

