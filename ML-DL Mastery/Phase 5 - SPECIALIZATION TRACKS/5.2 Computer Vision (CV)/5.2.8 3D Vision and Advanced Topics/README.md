# 5.2.8 3D Vision and Advanced Topics

Depth estimation, stereo vision, point clouds, PointNet, NeRF, 3D bounding boxes.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Stereo disparity concept |
| `working_example2.py` | Synthetic point cloud → PCA normal → perspective projection |
| `working_example.ipynb` | Interactive: sphere point cloud → perspective projection |

## Quick Reference

```python
import open3d as o3d

# Load and visualise point cloud
pcd = o3d.io.read_point_cloud("scene.ply")
o3d.visualization.draw_geometries([pcd])

# Normal estimation
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Depth map → point cloud (using camera intrinsics)
import torch
depth = torch.load("depth.pt")          # (H, W) in metres
fx, fy, cx, cy = 525, 525, 320, 240    # intrinsics
ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
Z = depth; X = (xs - cx) * Z / fx; Y = (ys - cy) * Z / fy
pts = torch.stack([X, Y, Z], dim=-1)   # (H, W, 3)
```

## Topic Map

| Topic | Key method | Framework |
|-------|-----------|-----------|
| Depth estimation | MiDaS, DPT | PyTorch |
| Point cloud | PointNet, PointNet++ | PyTorch |
| Stereo vision | SGM, RAFT-Stereo | OpenCV |
| NeRF | Volume rendering | tiny-cuda-nn |

## Learning Resources
- [NeRF paper](https://arxiv.org/abs/2003.08934)
- [Open3D docs](http://www.open3d.org/docs/release/)

Explore this topic with a small practical project or coding exercise.

## What to build

- Try a small hands-on exercise focused on this topic.
- Keep the code in `project.py` in this folder.
- Add notes, examples, or results inside this directory.

## Suggestions

1. Read the checklist topic and identify one practice task.
2. Write code in `project.py` that illustrates the main concept.
3. Run your code and iterate until it works.

## Notes

- Use Python and standard libraries when possible.
- For data topics, install `numpy`, `pandas`, `matplotlib` as needed.
