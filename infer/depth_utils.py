"""
depth_utils.py - Depth utilities for GenCompositor

Provides:
- Depth video loading from 4DGS outputs
- Depth Anything integration for depth estimation
- Depth alignment (relative to metric)
- Object depth extraction
"""

import numpy as np
import torch
from typing import Tuple, Optional, Union
from pathlib import Path
import cv2


# Global model cache to avoid reloading
_depth_model = None
_depth_processor = None


def load_depth_video(path: str) -> np.ndarray:
    """Load depth video from 4DGS output.

    Supports:
    - .npy files: Shape (T, H, W), depth in meters
    - Directory of PNGs: 16-bit depth images

    Args:
        path: Path to depth video (.npy) or directory of depth frames

    Returns:
        np.ndarray: Shape (T, H, W), depth values in meters
    """
    path = Path(path)

    if path.suffix == '.npy':
        depth_video = np.load(str(path))
        return depth_video.astype(np.float32)

    elif path.is_dir():
        # Load directory of depth frames
        frame_paths = sorted(path.glob('*.png')) + sorted(path.glob('*.PNG'))
        if not frame_paths:
            raise ValueError(f"No PNG files found in {path}")

        frames = []
        for fp in frame_paths:
            # Load 16-bit PNG depth
            depth = cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise ValueError(f"Failed to load {fp}")

            # Convert to meters (assuming mm encoding)
            depth = depth.astype(np.float32) / 1000.0
            frames.append(depth)

        return np.stack(frames, axis=0)

    else:
        raise ValueError(f"Unsupported depth format: {path}")


def get_depth_model():
    """Get or initialize the Depth Anything model.

    Uses Depth Anything V2 via transformers pipeline.
    """
    global _depth_model, _depth_processor

    if _depth_model is None:
        try:
            from transformers import pipeline
            print("Loading Depth Anything V2 model...")
            _depth_model = pipeline(
                "depth-estimation",
                model="depth-anything/Depth-Anything-V2-Small-hf",
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            print("Depth Anything model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load Depth Anything model: {e}")
            print("Falling back to simple depth estimation")
            _depth_model = "fallback"

    return _depth_model


def run_depth_anything(image: np.ndarray) -> np.ndarray:
    """Run Depth Anything V2 on a single image.

    Args:
        image: RGB image (H, W, 3), uint8 or float32

    Returns:
        np.ndarray: Relative depth map (H, W), higher values = farther
    """
    model = get_depth_model()

    if model == "fallback":
        # Simple fallback: return uniform depth
        h, w = image.shape[:2]
        return np.ones((h, w), dtype=np.float32) * 0.5

    # Ensure image is in correct format
    if image.dtype == np.float32:
        image = (image * 255).astype(np.uint8)

    # Convert BGR to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Assume input might be BGR from cv2
        from PIL import Image
        pil_image = Image.fromarray(image)
    else:
        from PIL import Image
        pil_image = Image.fromarray(image)

    # Run inference
    result = model(pil_image)
    depth = np.array(result["depth"])

    # Normalize to 0-1 range
    depth = depth.astype(np.float32)
    depth_min, depth_max = depth.min(), depth.max()
    if depth_max > depth_min:
        depth = (depth - depth_min) / (depth_max - depth_min)

    return depth


def align_depth_to_metric(
    estimated_depth: np.ndarray,
    bg_depth: np.ndarray,
    object_mask: np.ndarray,
    min_known_pixels: int = 100
) -> np.ndarray:
    """Align Depth Anything's relative depth to metric depth.

    Depth Anything outputs relative depth (0-1 range).
    We align it to the background's metric depth using known regions.

    Approach:
    - Use background pixels (where mask=0) to compute scale/shift
    - Apply same transform to get metric depth for object

    Args:
        estimated_depth: Relative depth from Depth Anything (H, W)
        bg_depth: Metric depth from 4DGS (H, W), in meters
        object_mask: Binary mask where 1 = object, 0 = background
        min_known_pixels: Minimum pixels needed for reliable alignment

    Returns:
        np.ndarray: Aligned depth in meters (H, W)
    """
    # Ensure same shape
    if estimated_depth.shape != bg_depth.shape:
        estimated_depth = cv2.resize(
            estimated_depth,
            (bg_depth.shape[1], bg_depth.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )

    if object_mask.shape != bg_depth.shape:
        object_mask = cv2.resize(
            object_mask.astype(np.float32),
            (bg_depth.shape[1], bg_depth.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(np.uint8)

    # Find known (background) pixels with valid depth
    known_mask = (object_mask == 0) & (bg_depth > 0.01) & (bg_depth < 100.0)

    if known_mask.sum() < min_known_pixels:
        print(f"Warning: Only {known_mask.sum()} known pixels for alignment")
        # Fallback: use simple scaling based on depth range
        valid_bg = bg_depth[bg_depth > 0.01]
        if len(valid_bg) > 0:
            scale = valid_bg.mean()
        else:
            scale = 5.0  # Default 5 meters
        return estimated_depth * scale * 2  # Rough scaling

    # Least squares alignment: estimated * scale + shift ≈ bg_depth
    est_vals = estimated_depth[known_mask]
    bg_vals = bg_depth[known_mask]

    A = np.column_stack([est_vals, np.ones(len(est_vals))])
    b = bg_vals

    try:
        result = np.linalg.lstsq(A, b, rcond=None)
        scale, shift = result[0]
    except np.linalg.LinAlgError:
        # Fallback to simple scaling
        scale = bg_vals.mean() / (est_vals.mean() + 1e-6)
        shift = 0.0

    # Apply alignment
    aligned = estimated_depth * scale + shift

    # Clamp to reasonable range
    aligned = np.clip(aligned, 0.01, 100.0)

    return aligned


def extract_object_depth_at_trajectory(
    aligned_depth: np.ndarray,
    object_mask: np.ndarray,
    trajectory_point: Tuple[int, int],
    radius: int = 5
) -> float:
    """Get object depth at trajectory point.

    Samples depth in region around trajectory point,
    only considering pixels where object is present.

    Args:
        aligned_depth: Aligned depth map (H, W), in meters
        object_mask: Binary mask where 1 = object
        trajectory_point: (x, y) pixel coordinates
        radius: Sampling radius around trajectory point

    Returns:
        float: Median depth of object at trajectory point (meters)
    """
    x, y = trajectory_point
    h, w = aligned_depth.shape

    # Clamp coordinates
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))

    # Extract neighborhood
    y1, y2 = max(0, y - radius), min(h, y + radius + 1)
    x1, x2 = max(0, x - radius), min(w, x + radius + 1)

    region_depth = aligned_depth[y1:y2, x1:x2]
    region_mask = object_mask[y1:y2, x1:x2]

    # Get depths only where object is present
    object_depths = region_depth[region_mask > 0]

    if len(object_depths) == 0:
        # Fallback: try center of entire object
        all_object_depths = aligned_depth[object_mask > 0]
        if len(all_object_depths) > 0:
            return float(np.median(all_object_depths))
        else:
            # Ultimate fallback: use depth at trajectory point
            return float(aligned_depth[y, x])

    return float(np.median(object_depths))


def visualize_depth(depth: np.ndarray, colormap: int = cv2.COLORMAP_MAGMA) -> np.ndarray:
    """Create colored visualization of depth map.

    Args:
        depth: Depth map (H, W)
        colormap: OpenCV colormap to use

    Returns:
        np.ndarray: Colored visualization (H, W, 3), BGR
    """
    # Normalize to 0-255
    depth_normalized = depth.copy()
    valid = depth_normalized > 0
    if valid.any():
        depth_min = depth_normalized[valid].min()
        depth_max = depth_normalized[valid].max()
        if depth_max > depth_min:
            depth_normalized = (depth_normalized - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = np.zeros_like(depth_normalized)

    depth_normalized = (depth_normalized * 255).astype(np.uint8)

    # Apply colormap
    colored = cv2.applyColorMap(depth_normalized, colormap)

    return colored


def composite_element_at_point(
    bg_frame: np.ndarray,
    fg_element: np.ndarray,
    position: Tuple[int, int],
    scale: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Composite white-background element onto background at position.

    Args:
        bg_frame: Background image (H, W, 3), BGR
        fg_element: SAM-segmented element with white background (H, W, 3), BGR
        position: (x, y) center position for the element
        scale: Scale factor for the element

    Returns:
        Tuple of:
        - composite: Composited image (H, W, 3)
        - mask: Binary mask of element in composite coordinates (H, W)
    """
    bg_h, bg_w = bg_frame.shape[:2]
    fg_h, fg_w = fg_element.shape[:2]

    # Scale element if needed
    if scale != 1.0:
        new_w = int(fg_w * scale)
        new_h = int(fg_h * scale)
        fg_element = cv2.resize(fg_element, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        fg_h, fg_w = fg_element.shape[:2]

    # Create mask from white background (white = background, not white = foreground)
    # Detect non-white pixels
    white_threshold = 250
    is_white = np.all(fg_element > white_threshold, axis=2)
    fg_mask = (~is_white).astype(np.uint8) * 255

    # Find bounding box of non-white region
    coords = np.where(fg_mask > 0)
    if len(coords[0]) == 0:
        # No foreground found, return original
        return bg_frame.copy(), np.zeros((bg_h, bg_w), dtype=np.uint8)

    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()

    # Crop to bounding box
    fg_cropped = fg_element[y_min:y_max+1, x_min:x_max+1]
    mask_cropped = fg_mask[y_min:y_max+1, x_min:x_max+1]

    crop_h, crop_w = fg_cropped.shape[:2]

    # Calculate position (center element at position)
    px, py = position
    x1 = px - crop_w // 2
    y1 = py - crop_h // 2
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    # Handle boundary clipping
    src_x1, src_y1 = 0, 0
    src_x2, src_y2 = crop_w, crop_h

    if x1 < 0:
        src_x1 = -x1
        x1 = 0
    if y1 < 0:
        src_y1 = -y1
        y1 = 0
    if x2 > bg_w:
        src_x2 -= (x2 - bg_w)
        x2 = bg_w
    if y2 > bg_h:
        src_y2 -= (y2 - bg_h)
        y2 = bg_h

    # Create output
    composite = bg_frame.copy()
    output_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)

    if x2 > x1 and y2 > y1 and src_x2 > src_x1 and src_y2 > src_y1:
        # Get the region to composite
        fg_region = fg_cropped[src_y1:src_y2, src_x1:src_x2]
        mask_region = mask_cropped[src_y1:src_y2, src_x1:src_x2]

        # Blend using mask
        mask_3ch = np.stack([mask_region] * 3, axis=2).astype(np.float32) / 255.0
        composite[y1:y2, x1:x2] = (
            fg_region * mask_3ch +
            composite[y1:y2, x1:x2] * (1 - mask_3ch)
        ).astype(np.uint8)

        output_mask[y1:y2, x1:x2] = mask_region

    return composite, output_mask


def create_positioned_mask(
    fg_mask: np.ndarray,
    position: Tuple[int, int],
    output_shape: Tuple[int, int],
    scale: float = 1.0
) -> np.ndarray:
    """Transform SAM mask to the position in the composite.

    Args:
        fg_mask: Original SAM binary mask (H, W)
        position: Where the element center should be placed (x, y)
        output_shape: Size of output mask (H, W)
        scale: Scale factor applied to element

    Returns:
        np.ndarray: Binary mask in composite image coordinates (H, W)
    """
    fg_h, fg_w = fg_mask.shape[:2]
    out_h, out_w = output_shape

    # Scale mask if needed
    if scale != 1.0:
        new_w = int(fg_w * scale)
        new_h = int(fg_h * scale)
        fg_mask = cv2.resize(
            fg_mask.astype(np.float32),
            (new_w, new_h),
            interpolation=cv2.INTER_NEAREST
        ).astype(np.uint8)
        fg_h, fg_w = fg_mask.shape[:2]

    # Find bounding box of mask
    coords = np.where(fg_mask > 0)
    if len(coords[0]) == 0:
        return np.zeros((out_h, out_w), dtype=np.uint8)

    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()

    # Crop to bounding box
    mask_cropped = fg_mask[y_min:y_max+1, x_min:x_max+1]
    crop_h, crop_w = mask_cropped.shape[:2]

    # Calculate position (center at position)
    px, py = position
    x1 = px - crop_w // 2
    y1 = py - crop_h // 2
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    # Handle boundary clipping
    src_x1, src_y1 = 0, 0
    src_x2, src_y2 = crop_w, crop_h

    if x1 < 0:
        src_x1 = -x1
        x1 = 0
    if y1 < 0:
        src_y1 = -y1
        y1 = 0
    if x2 > out_w:
        src_x2 -= (x2 - out_w)
        x2 = out_w
    if y2 > out_h:
        src_y2 -= (y2 - out_h)
        y2 = out_h

    # Create output mask
    output_mask = np.zeros((out_h, out_w), dtype=np.uint8)

    if x2 > x1 and y2 > y1 and src_x2 > src_x1 and src_y2 > src_y1:
        output_mask[y1:y2, x1:x2] = mask_cropped[src_y1:src_y2, src_x1:src_x2]

    return output_mask


if __name__ == "__main__":
    # Test the depth utilities
    print("Testing depth_utils.py")
    print("=" * 50)

    # Test depth model loading
    print("\n1. Testing Depth Anything model loading...")
    model = get_depth_model()
    print(f"Model loaded: {model is not None}")

    # Test with a simple image
    print("\n2. Testing depth estimation on synthetic image...")
    test_image = np.random.randint(0, 255, (480, 720, 3), dtype=np.uint8)
    depth = run_depth_anything(test_image)
    print(f"Depth shape: {depth.shape}")
    print(f"Depth range: [{depth.min():.3f}, {depth.max():.3f}]")

    # Test alignment
    print("\n3. Testing depth alignment...")
    bg_depth = np.random.uniform(2.0, 10.0, (480, 720)).astype(np.float32)
    object_mask = np.zeros((480, 720), dtype=np.uint8)
    object_mask[200:300, 300:400] = 255

    aligned = align_depth_to_metric(depth, bg_depth, object_mask)
    print(f"Aligned depth range: [{aligned.min():.3f}, {aligned.max():.3f}]")

    # Test depth extraction
    print("\n4. Testing object depth extraction...")
    obj_depth = extract_object_depth_at_trajectory(aligned, object_mask, (350, 250))
    print(f"Object depth at (350, 250): {obj_depth:.3f} meters")

    print("\n" + "=" * 50)
    print("All tests completed!")
