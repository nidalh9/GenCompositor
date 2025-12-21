#!/usr/bin/env python3
"""
Test script for 3D trajectory projection.
Run this to verify the 3D trajectory system works before using the full app.

Usage:
    cd /Users/nidalhulaihel/git/project/GenCompositor
    python infer/test_trajectory_3d.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infer.trajectory_3d import (
    load_trajectory_3d,
    interpolate_3d_trajectory,
    project_trajectory_to_views,
    get_camera_configs,
    Trajectory3D
)

def test_trajectory_file(filepath):
    """Test a 3D trajectory file and show projections."""
    print(f"\n{'='*60}")
    print(f"Testing: {os.path.basename(filepath)}")
    print('='*60)

    # Load trajectory
    traj = load_trajectory_3d(filepath)
    print(f"\nLoaded {len(traj.points)} 3D keypoints:")
    for i, pt in enumerate(traj.points):
        print(f"  {i+1}. ({pt[0]:6.2f}, {pt[1]:6.2f}, {pt[2]:6.2f})")

    # Interpolate to 49 frames
    if len(traj.points) < 49:
        interpolated = interpolate_3d_trajectory(traj.points, 49)
        traj = Trajectory3D(points=interpolated, num_frames=49)
        print(f"\nInterpolated to {len(traj.points)} frames")

    # Project to all views
    configs = get_camera_configs(distance=5.0, height=0.0, fov=60.0)
    projections = project_trajectory_to_views(traj, configs)

    print("\n2D Projections (first, middle, last frame):")
    print("-" * 50)
    print(f"{'Angle':>8} | {'Frame 0':>15} | {'Frame 24':>15} | {'Frame 48':>15}")
    print("-" * 50)

    for angle in [0, 90, 180, 270]:
        pts = projections[angle]
        print(f"{angle:>6}°  | ({pts[0][0]:>5}, {pts[0][1]:>4}) | ({pts[24][0]:>5}, {pts[24][1]:>4}) | ({pts[-1][0]:>5}, {pts[-1][1]:>4})")

    print("-" * 50)
    print("\nNote: Image center is (360, 240) for 720x480 resolution")

    return projections


def main():
    print("3D Trajectory Projection Test")
    print("=" * 60)
    print("\nCoordinate System:")
    print("  - Origin: Scene center (0, 0, 0)")
    print("  - Y-axis: Up (positive = up)")
    print("  - X-axis: Right (positive = right when viewed from 0°)")
    print("  - Z-axis: Depth (positive = towards 180° camera)")
    print("\nCamera Positions (at distance R=5):")
    print("  - 0°:   (0, 0, -5) - Front camera")
    print("  - 90°:  (5, 0, 0)  - Right camera")
    print("  - 180°: (0, 0, 5)  - Back camera")
    print("  - 270°: (-5, 0, 0) - Left camera")

    # Test all example trajectories
    trajectory_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                   "assets/trajectories_3d")

    if os.path.exists(trajectory_dir):
        for filename in sorted(os.listdir(trajectory_dir)):
            if filename.endswith('.txt'):
                filepath = os.path.join(trajectory_dir, filename)
                test_trajectory_file(filepath)
    else:
        print(f"\nTrajectory directory not found: {trajectory_dir}")
        print("Creating example trajectory inline...")

        # Create inline example
        example_points = [
            (-2.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (2.0, 0.0, 0.0),
        ]
        traj = Trajectory3D(points=example_points, num_frames=3)

        interpolated = interpolate_3d_trajectory(example_points, 49)
        traj = Trajectory3D(points=interpolated, num_frames=49)

        configs = get_camera_configs(distance=5.0, height=0.0, fov=60.0)
        projections = project_trajectory_to_views(traj, configs)

        print("\nExample: Parabola from left to right")
        for angle in [0, 90, 180, 270]:
            pts = projections[angle]
            print(f"  {angle}°: start={pts[0]}, middle={pts[24]}, end={pts[-1]}")


if __name__ == "__main__":
    main()
