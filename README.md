# Deception-Research-model

# Pitch Visibility Analysis using Motion Capture (C3D)

This project analyzes baseball pitching mechanics using 3D motion capture data to determine how long the throwing hand (ball) is visible to the catcher throughout the delivery.

It leverages `.c3d` marker data, reconstructs a 3D skeleton, and animates the delivery while dynamically determining hand visibility based on head, torso and glove occlusion.

---

## Key Features

- Visualizes 3D skeletons from C3D motion capture data.
- Detects when the throwing hand is hidden or visible from the catcher’s point of view.
- Models glove-side occlusion and adjusts visibility logic for early delivery (glove containment).
- Uses ray-triangle intersection and convex hull occlusion for accurate line-of-sight analysis.
- 3D animated matplotlib output with real-time color-coded wrist tracking.

---
## Future Features
- Give visibility score as a part of a weighted deception score. 
## Project Structure

