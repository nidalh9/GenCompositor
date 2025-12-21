"""
trajectory_3d.py - 3D Trajectory and Camera Projection Module for GenCompositor

Coordinate System:
- Origin: Scene center (0, 0, 0)
- Y-axis: Up (positive Y is up)
- X-axis: Horizontal right (when viewed from 0 degrees)
- Z-axis: Depth (positive Z is towards 180 degrees camera)

Camera Arrangement (circular at distance R):
- 0 degrees:   Camera at (0, height, -R), looking at origin
- 90 degrees:  Camera at (R, height, 0), looking at origin
- 180 degrees: Camera at (0, height, R), looking at origin
- 270 degrees: Camera at (-R, height, 0), looking at origin
"""

import numpy as np
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Union


@dataclass
class Trajectory3D:
    """3D trajectory data structure"""
    points: List[Tuple[float, float, float]]  # (x, y, z) world coordinates
    num_frames: int


@dataclass
class CameraConfig:
    """Camera configuration for a single view"""
    angle: float           # 0, 90, 180, 270 degrees
    distance: float        # R - distance from origin
    height: float          # Camera Y position
    fov: float            # Field of view in degrees
    image_width: int       # Default: 720
    image_height: int      # Default: 480


class Camera:
    """Pinhole camera model for perspective projection"""

    def __init__(self, config: CameraConfig):
        self.config = config
        self._compute_matrices()

    def _compute_matrices(self):
        """Compute view and projection matrices"""
        angle_rad = math.radians(self.config.angle)
        R = self.config.distance

        # Camera position based on angle (circular arrangement)
        # At 0 degrees: camera is at (0, height, -R) looking at origin
        # At 90 degrees: camera is at (R, height, 0) looking at origin
        # etc.
        self.position = np.array([
            R * math.sin(angle_rad),
            self.config.height,
            -R * math.cos(angle_rad)
        ])

        # View matrix (look-at)
        self.view_matrix = self._look_at(
            eye=self.position,
            target=np.array([0.0, 0.0, 0.0]),
            up=np.array([0.0, 1.0, 0.0])
        )

        # Intrinsic matrix
        self.intrinsic_matrix = self._compute_intrinsics()

    def _look_at(self, eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
        """
        Compute view matrix using look-at formulation.

        Args:
            eye: Camera position in world coordinates
            target: Point the camera is looking at
            up: World up vector

        Returns:
            4x4 view matrix
        """
        # Forward direction (from eye to target)
        forward = target - eye
        forward = forward / np.linalg.norm(forward)

        # Right direction
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)

        # Recompute up to ensure orthogonality
        up_new = np.cross(right, forward)

        # Rotation matrix - camera looks along +Z in camera space
        # So we need: camera X = right, camera Y = up, camera Z = forward
        rotation = np.array([
            [right[0], right[1], right[2]],
            [-up_new[0], -up_new[1], -up_new[2]],  # Flip Y so +Y in world = -Y in image (top of image)
            [forward[0], forward[1], forward[2]]
        ])

        # Translation part
        translation = -rotation @ eye

        # Combine into 4x4 matrix
        view = np.eye(4)
        view[:3, :3] = rotation
        view[:3, 3] = translation

        return view

    def _compute_intrinsics(self) -> np.ndarray:
        """
        Compute camera intrinsic matrix.

        Returns:
            3x3 intrinsic matrix K
        """
        fov_rad = math.radians(self.config.fov)
        # Focal length in pixels
        f = self.config.image_height / (2 * math.tan(fov_rad / 2))

        # Principal point (image center)
        cx = self.config.image_width / 2
        cy = self.config.image_height / 2

        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ])
        return K

    def project_3d_to_2d(self, point_3d: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Project a 3D world point to 2D image coordinates.

        Args:
            point_3d: (x, y, z) world coordinates as numpy array

        Returns:
            (u, v) pixel coordinates, or None if point is behind camera
        """
        # Convert to homogeneous coordinates
        point_h = np.append(point_3d, 1.0)

        # World to camera transformation
        point_cam = self.view_matrix @ point_h

        # In our convention, camera looks along +z in camera space after transformation
        # Points in front of camera have positive z
        # Check if point is behind camera
        if point_cam[2] <= 0.01:  # Small epsilon to avoid division by zero
            return None

        # Project to image plane using intrinsics
        # Standard pinhole: u = f * X/Z + cx, v = f * Y/Z + cy
        u = int(self.intrinsic_matrix[0, 0] * point_cam[0] / point_cam[2] + self.intrinsic_matrix[0, 2])
        v = int(self.intrinsic_matrix[1, 1] * point_cam[1] / point_cam[2] + self.intrinsic_matrix[1, 2])

        # Clamp to image bounds
        u = max(0, min(u, self.config.image_width - 1))
        v = max(0, min(v, self.config.image_height - 1))

        return (u, v)

    def project_3d_to_2d_float(self, point_3d: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Project a 3D world point to 2D image coordinates (float precision).

        Args:
            point_3d: (x, y, z) world coordinates as numpy array

        Returns:
            (u, v) pixel coordinates as floats, or None if point is behind camera
        """
        point_h = np.append(point_3d, 1.0)
        point_cam = self.view_matrix @ point_h

        if point_cam[2] <= 0.01:
            return None

        u = self.intrinsic_matrix[0, 0] * point_cam[0] / point_cam[2] + self.intrinsic_matrix[0, 2]
        v = self.intrinsic_matrix[1, 1] * point_cam[1] / point_cam[2] + self.intrinsic_matrix[1, 2]

        return (u, v)

    def lift_2d_to_3d(
        self,
        u: float,
        v: float,
        depth: float
    ) -> Tuple[float, float, float]:
        """
        Inverse of project_3d_to_2d - lifts pixel + depth to world coordinates.

        This unprojects a 2D pixel coordinate plus depth value back to 3D world space.
        Uses the camera's intrinsics and view matrix to perform the transformation.

        Args:
            u: Pixel x coordinate
            v: Pixel y coordinate
            depth: Depth value in camera Z direction (meters)

        Returns:
            Tuple of (x_world, y_world, z_world) in world coordinates

        Raises:
            ValueError: If depth is invalid (<=0)
        """
        if depth <= 0.01:
            raise ValueError(f"Invalid depth: {depth}. Depth must be positive.")

        # Get intrinsics
        fx = self.intrinsic_matrix[0, 0]
        fy = self.intrinsic_matrix[1, 1]
        cx = self.intrinsic_matrix[0, 2]
        cy = self.intrinsic_matrix[1, 2]

        # Unproject to camera coordinates
        # From pinhole model: u = fx * X/Z + cx, v = fy * Y/Z + cy
        # Inverse: X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy
        X_cam = (u - cx) * depth / fx
        Y_cam = (v - cy) * depth / fy
        Z_cam = depth

        # Point in camera coordinates (homogeneous)
        point_cam = np.array([X_cam, Y_cam, Z_cam, 1.0])

        # Transform from camera to world coordinates
        # Since view_matrix transforms world->camera, we need its inverse
        view_inv = np.linalg.inv(self.view_matrix)
        point_world = view_inv @ point_cam

        return (float(point_world[0]), float(point_world[1]), float(point_world[2]))


# Default camera configurations for 4-view setup
DEFAULT_CAMERA_DISTANCE = 10.0
DEFAULT_CAMERA_HEIGHT = 0.0
DEFAULT_CAMERA_FOV = 60.0
DEFAULT_IMAGE_WIDTH = 720
DEFAULT_IMAGE_HEIGHT = 480

DEFAULT_CAMERA_CONFIGS = [
    CameraConfig(
        angle=0,
        distance=DEFAULT_CAMERA_DISTANCE,
        height=DEFAULT_CAMERA_HEIGHT,
        fov=DEFAULT_CAMERA_FOV,
        image_width=DEFAULT_IMAGE_WIDTH,
        image_height=DEFAULT_IMAGE_HEIGHT
    ),
    CameraConfig(
        angle=90,
        distance=DEFAULT_CAMERA_DISTANCE,
        height=DEFAULT_CAMERA_HEIGHT,
        fov=DEFAULT_CAMERA_FOV,
        image_width=DEFAULT_IMAGE_WIDTH,
        image_height=DEFAULT_IMAGE_HEIGHT
    ),
    CameraConfig(
        angle=180,
        distance=DEFAULT_CAMERA_DISTANCE,
        height=DEFAULT_CAMERA_HEIGHT,
        fov=DEFAULT_CAMERA_FOV,
        image_width=DEFAULT_IMAGE_WIDTH,
        image_height=DEFAULT_IMAGE_HEIGHT
    ),
    CameraConfig(
        angle=270,
        distance=DEFAULT_CAMERA_DISTANCE,
        height=DEFAULT_CAMERA_HEIGHT,
        fov=DEFAULT_CAMERA_FOV,
        image_width=DEFAULT_IMAGE_WIDTH,
        image_height=DEFAULT_IMAGE_HEIGHT
    ),
]


def get_camera_configs(
    distance: float = DEFAULT_CAMERA_DISTANCE,
    height: float = DEFAULT_CAMERA_HEIGHT,
    fov: float = DEFAULT_CAMERA_FOV,
    image_width: int = DEFAULT_IMAGE_WIDTH,
    image_height: int = DEFAULT_IMAGE_HEIGHT
) -> List[CameraConfig]:
    """
    Create camera configurations for 4-view setup with custom parameters.

    Args:
        distance: Distance from origin (R)
        height: Camera Y position
        fov: Field of view in degrees
        image_width: Output image width
        image_height: Output image height

    Returns:
        List of 4 CameraConfig objects for angles 0, 90, 180, 270
    """
    return [
        CameraConfig(angle=a, distance=distance, height=height, fov=fov,
                    image_width=image_width, image_height=image_height)
        for a in [0, 90, 180, 270]
    ]


def is_3d_trajectory(path: str) -> bool:
    """
    Check if a trajectory file is in 3D format.

    3D trajectory files start with '#3D' header.

    Args:
        path: Path to trajectory file

    Returns:
        True if file is 3D format, False otherwise
    """
    try:
        with open(path, 'r') as f:
            first_line = f.readline().strip()
            return first_line == '#3D'
    except Exception:
        return False


def load_trajectory_3d(path: str) -> Trajectory3D:
    """
    Load a 3D trajectory from file.

    File format:
        #3D
        x,y,z
        x,y,z
        ...

    Args:
        path: Path to trajectory file

    Returns:
        Trajectory3D object with loaded points
    """
    points = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split(',')
                if len(parts) >= 3:
                    x, y, z = map(float, parts[:3])
                    points.append((x, y, z))

    return Trajectory3D(points=points, num_frames=len(points))


def save_trajectory_3d(trajectory: Trajectory3D, path: str):
    """
    Save a 3D trajectory to file.

    Args:
        trajectory: Trajectory3D object to save
        path: Output file path
    """
    with open(path, 'w') as f:
        f.write('#3D\n')
        for point in trajectory.points:
            f.write(f'{point[0]},{point[1]},{point[2]}\n')


def save_trajectory_2d(points: List[Tuple[int, int]], path: str):
    """
    Save a 2D trajectory to file (legacy format).

    Args:
        points: List of (x, y) coordinates
        path: Output file path
    """
    with open(path, 'w') as f:
        for point in points:
            f.write(f'{point[0]},{point[1]}\n')


def interpolate_3d_trajectory(
    keypoints: List[Tuple[float, float, float]],
    num_frames: int
) -> List[Tuple[float, float, float]]:
    """
    Interpolate between 3D keypoints to get trajectory for all frames.
    Uses linear interpolation between consecutive keypoints.

    Args:
        keypoints: List of (x, y, z) keypoint coordinates
        num_frames: Target number of frames

    Returns:
        List of interpolated (x, y, z) coordinates with length num_frames
    """
    if not keypoints:
        return []

    if len(keypoints) == 1:
        return [keypoints[0]] * num_frames

    interpolated = []
    segments = len(keypoints) - 1
    points_per_segment = num_frames // segments

    for i in range(segments):
        start = np.array(keypoints[i])
        end = np.array(keypoints[i + 1])

        # Last segment gets remaining frames
        if i < segments - 1:
            n_points = points_per_segment
        else:
            n_points = num_frames - len(interpolated)

        for j in range(n_points):
            if n_points > 1:
                ratio = j / (n_points - 1)
            else:
                ratio = 0
            point = start + ratio * (end - start)
            interpolated.append(tuple(point))

    return interpolated[:num_frames]


def project_trajectory_to_view(
    trajectory_3d: Trajectory3D,
    camera_config: CameraConfig
) -> List[Tuple[int, int]]:
    """
    Project a 3D trajectory to a single 2D camera view.

    Args:
        trajectory_3d: 3D trajectory to project
        camera_config: Camera configuration for the view

    Returns:
        List of (u, v) 2D pixel coordinates
    """
    camera = Camera(camera_config)
    projected_points = []

    last_valid = (camera_config.image_width // 2, camera_config.image_height // 2)

    for point_3d in trajectory_3d.points:
        point_2d = camera.project_3d_to_2d(np.array(point_3d))
        if point_2d is None:
            # Point behind camera - use last valid point
            point_2d = last_valid
        else:
            last_valid = point_2d
        projected_points.append(point_2d)

    return projected_points


def project_trajectory_to_views(
    trajectory_3d: Trajectory3D,
    camera_configs: List[CameraConfig]
) -> Dict[float, List[Tuple[int, int]]]:
    """
    Project a 3D trajectory to multiple 2D camera views.

    Args:
        trajectory_3d: 3D trajectory to project
        camera_configs: List of camera configurations

    Returns:
        Dictionary mapping camera angle to list of 2D (u, v) coordinates
    """
    results = {}

    for config in camera_configs:
        projected_points = project_trajectory_to_view(trajectory_3d, config)
        results[config.angle] = projected_points

    return results


def parse_trajectory_3d_text(text: str) -> Trajectory3D:
    """
    Parse 3D trajectory from text input (e.g., from UI text area).

    Accepts format:
        x,y,z
        x,y,z
        ...

    Lines starting with '#' are ignored.

    Args:
        text: Multi-line text with x,y,z coordinates

    Returns:
        Trajectory3D object
    """
    points = []
    for line in text.strip().split('\n'):
        line = line.strip()
        if line and not line.startswith('#'):
            parts = line.split(',')
            if len(parts) >= 3:
                try:
                    x, y, z = map(float, parts[:3])
                    points.append((x, y, z))
                except ValueError:
                    continue

    return Trajectory3D(points=points, num_frames=len(points))


def trajectory_3d_to_text(trajectory: Trajectory3D, include_header: bool = True) -> str:
    """
    Convert a 3D trajectory to text format.

    Args:
        trajectory: Trajectory3D object
        include_header: Whether to include '#3D' header

    Returns:
        Multi-line text representation
    """
    lines = []
    if include_header:
        lines.append('#3D')
    for point in trajectory.points:
        lines.append(f'{point[0]},{point[1]},{point[2]}')
    return '\n'.join(lines)


# Convenience function for quick projection
def project_point_to_all_views(
    point_3d: Tuple[float, float, float],
    distance: float = DEFAULT_CAMERA_DISTANCE,
    height: float = DEFAULT_CAMERA_HEIGHT,
    fov: float = DEFAULT_CAMERA_FOV
) -> Dict[int, Tuple[int, int]]:
    """
    Project a single 3D point to all 4 camera views.

    Args:
        point_3d: (x, y, z) world coordinates
        distance: Camera distance from origin
        height: Camera height
        fov: Field of view in degrees

    Returns:
        Dictionary mapping angle (0, 90, 180, 270) to (u, v) pixel coordinates
    """
    configs = get_camera_configs(distance=distance, height=height, fov=fov)
    results = {}

    for config in configs:
        camera = Camera(config)
        point_2d = camera.project_3d_to_2d(np.array(point_3d))
        if point_2d is None:
            point_2d = (config.image_width // 2, config.image_height // 2)
        results[int(config.angle)] = point_2d

    return results


if __name__ == "__main__":
    # Test the projection system
    print("Testing 3D Trajectory Projection System")
    print("=" * 50)

    # Create a simple test trajectory - object moving from left to right at ground level
    test_points = [
        (-2.0, 0.0, 0.0),   # Start left
        (-1.0, 0.5, 0.0),   # Move right, rise up
        (0.0, 1.0, 0.0),    # Center, highest point
        (1.0, 0.5, 0.0),    # Move right, descend
        (2.0, 0.0, 0.0),    # End right
    ]

    trajectory = Trajectory3D(points=test_points, num_frames=5)

    # Project to all views
    configs = get_camera_configs(distance=5.0, height=0.0, fov=60.0)
    projections = project_trajectory_to_views(trajectory, configs)

    print("\n3D Trajectory Points:")
    for i, pt in enumerate(test_points):
        print(f"  Frame {i}: ({pt[0]:.1f}, {pt[1]:.1f}, {pt[2]:.1f})")

    print("\n2D Projections by Camera Angle:")
    for angle in [0, 90, 180, 270]:
        print(f"\n  Camera {angle} degrees:")
        for i, pt in enumerate(projections[angle]):
            print(f"    Frame {i}: ({pt[0]}, {pt[1]})")

    # Test interpolation
    print("\n" + "=" * 50)
    print("Testing Interpolation (5 keypoints -> 49 frames):")
    interpolated = interpolate_3d_trajectory(test_points, 49)
    print(f"  Input: {len(test_points)} keypoints")
    print(f"  Output: {len(interpolated)} frames")
    print(f"  First: {interpolated[0]}")
    print(f"  Middle: {interpolated[24]}")
    print(f"  Last: {interpolated[-1]}")
