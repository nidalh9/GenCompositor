import gradio as gr
import cv2
import os
import tempfile
import numpy as np
from pathlib import Path
import shutil
import json
import torch
from datetime import datetime
from diffusers import (
    CogVideoXDPMScheduler,
    CogvideoXBranchModel,
    CogVideoXTransformer3D3BModel_sep,
    CogVideoXI2VTriInpaintPipeline_sep,
)
from accelerate import Accelerator
from diffusers.utils import export_to_video, load_video
from PIL import Image
from torchvision.transforms import ToTensor
import torch.nn as nn
from diffusers.utils.torch_utils import is_compiled_module
import random
import logging
import sys 
sys.path.append("..")
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set TORCH_CUDA_ARCH_LIST to suppress warning
os.environ['TORCH_CUDA_ARCH_LIST'] = "7.0 7.5 8.0 8.6 8.9 9.0"

accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision="bf16")

def wrap_model(model):
    model = accelerator.prepare(model)
    return model

def unwrap_model(model):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

# Pre-load models
dtype = torch.bfloat16
model_path = "../ckpts/CogVideoX-5b-I2V"
inpainting_branch = "../ckpts/branch"
transformer_path = "../ckpts/model"

branch = CogvideoXBranchModel.from_pretrained(inpainting_branch, torch_dtype=dtype).to("cuda", dtype=dtype)
transformer = CogVideoXTransformer3D3BModel_sep.from_pretrained(
    transformer_path,
    subfolder="transformer",
    torch_dtype=dtype,
).to("cuda", dtype=dtype)
pipe = CogVideoXI2VTriInpaintPipeline_sep.from_pretrained(
    model_path,
    branch=unwrap_model(branch),
    transformer=unwrap_model(transformer),
    torch_dtype=dtype,
)
pipe.text_encoder.requires_grad_(False)
pipe.transformer.requires_grad_(False)
pipe.vae.requires_grad_(False)
pipe.branch.requires_grad_(False)
pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
pipe = wrap_model(pipe)
pipe.to("cuda")
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

# Initialize SAM2 models for point-based segmentation
sam2_checkpoint = "../ckpts/sam2.1_hiera_large.pt"
model_cfg = "../sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
image_predictor = SAM2ImagePredictor(sam2_image_model)

# -------- Helpers --------
def get_first_frame(video_path):
    """获取视频的第一帧"""
    if not video_path or not os.path.exists(video_path):
        return None

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret and frame is not None:
        return frame  # Keep in BGR for OpenCV compatibility
    return None

def visualize_3d_trajectory(traj_text, traj_file, cam_dist, cam_height):
    """Render 3D trajectory with camera positions using matplotlib"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from io import BytesIO
    import math

    try:
        # Parse trajectory
        points = []
        if traj_file is not None:
            # Handle Gradio file upload object
            file_path = traj_file.name if hasattr(traj_file, 'name') else traj_file
            logger.info(f"Reading trajectory from file: {file_path}")
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split(',')
                        if len(parts) >= 3:
                            x, y, z = map(float, parts[:3])
                            points.append((x, y, z))
        elif traj_text and traj_text.strip():
            logger.info(f"Parsing trajectory from text input")
            for line in traj_text.strip().split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split(',')
                    if len(parts) >= 3:
                        x, y, z = map(float, parts[:3])
                        points.append((x, y, z))

        logger.info(f"Parsed {len(points)} trajectory points")

        if not points:
            logger.warning("No trajectory points found to visualize")
            # Create a simple message image instead of returning None
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, 'No trajectory points found.\n\nPlease enter 3D coordinates in the format:\nx, y, z\n\nExample:\n-2.0, 0.0, 0.0\n0.0, 1.0, 0.0\n2.0, 0.0, 0.0',
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            plt.close(fig)
            from PIL import Image
            return np.array(Image.open(buf))

        # Create figure with 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Extract coordinates
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        zs = [p[2] for p in points]

        # Plot trajectory as line and points
        ax.plot(xs, zs, ys, 'b-', linewidth=2, label='Trajectory')
        ax.scatter(xs, zs, ys, c='blue', s=50, alpha=0.8)

        # Mark start and end
        ax.scatter([xs[0]], [zs[0]], [ys[0]], c='green', s=150, marker='o', label='Start')
        ax.scatter([xs[-1]], [zs[-1]], [ys[-1]], c='red', s=150, marker='s', label='End')

        # Draw cameras at 4 positions
        R = cam_dist
        h = cam_height
        camera_positions = {
            '0° (Front)': (0, -R, h),
            '90° (Right)': (R, 0, h),
            '180° (Back)': (0, R, h),
            '270° (Left)': (-R, 0, h)
        }
        camera_colors = {'0° (Front)': 'purple', '90° (Right)': 'orange', '180° (Back)': 'cyan', '270° (Left)': 'magenta'}

        for name, (cx, cz, cy) in camera_positions.items():
            ax.scatter([cx], [cz], [cy], c=camera_colors[name], s=200, marker='^', label=name)
            # Draw line from camera to origin
            ax.plot([cx, 0], [cz, 0], [cy, 0], color=camera_colors[name], linestyle='--', alpha=0.3)

        # Draw origin
        ax.scatter([0], [0], [0], c='black', s=100, marker='x', label='Origin')

        # Draw coordinate axes
        axis_len = max(R, max(abs(min(xs)), abs(max(xs)), abs(min(ys)), abs(max(ys)), abs(min(zs)), abs(max(zs)))) * 1.2
        ax.plot([0, axis_len], [0, 0], [0, 0], 'r-', linewidth=1, alpha=0.5)
        ax.plot([0, 0], [0, axis_len], [0, 0], 'b-', linewidth=1, alpha=0.5)
        ax.plot([0, 0], [0, 0], [0, axis_len], 'g-', linewidth=1, alpha=0.5)
        ax.text(axis_len, 0, 0, 'X', color='red')
        ax.text(0, axis_len, 0, 'Z', color='blue')
        ax.text(0, 0, axis_len, 'Y', color='green')

        ax.set_xlabel('X (Left/Right)')
        ax.set_ylabel('Z (Front/Back)')
        ax.set_zlabel('Y (Up/Down)')
        ax.set_title(f'3D Trajectory ({len(points)} points)\nCamera Distance: {R}, Height: {h}')
        ax.legend(loc='upper left', fontsize=8)

        # Set equal aspect ratio
        max_range = max(axis_len, R) * 1.1
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range/2, max_range])

        # Set viewing angle
        ax.view_init(elev=25, azim=45)

        plt.tight_layout()

        # Save to buffer and return as numpy array
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        # Convert to numpy array for Gradio
        from PIL import Image
        img = Image.open(buf)
        logger.info(f"Successfully created 3D trajectory visualization")
        return np.array(img)

    except Exception as e:
        logger.error(f"Error visualizing trajectory: {e}")
        import traceback
        traceback.print_exc()
        # Return an error image instead of None
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, f'Error visualizing trajectory:\n\n{str(e)}',
               ha='center', va='center', fontsize=12, transform=ax.transAxes, color='red')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        from PIL import Image
        return np.array(Image.open(buf))

def create_trajectory_draw_canvas(points, cam_dist, grid_size=5.0):
    """Create a top-down view canvas for drawing trajectory (fallback when no video)"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from io import BytesIO

    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw grid
    grid_range = max(grid_size, cam_dist) * 1.2
    ax.set_xlim(-grid_range, grid_range)
    ax.set_ylim(-grid_range, grid_range)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Draw axes
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)

    # Draw camera positions
    R = cam_dist
    cameras = [(0, -R, '0°'), (R, 0, '90°'), (0, R, '180°'), (-R, 0, '270°')]
    for cx, cz, label in cameras:
        ax.scatter([cx], [cz], s=100, marker='^', c='gray', alpha=0.5)
        ax.annotate(label, (cx, cz), textcoords="offset points", xytext=(5, 5), fontsize=8)

    # Draw trajectory points
    if points and len(points) > 0:
        xs = [p[0] for p in points]
        zs = [p[2] for p in points]
        ys = [p[1] for p in points]  # Y heights for color

        # Draw lines connecting points
        if len(points) > 1:
            ax.plot(xs, zs, 'b-', linewidth=2, alpha=0.7)

        # Draw points with color based on Y height
        scatter = ax.scatter(xs, zs, c=ys, cmap='coolwarm', s=100, edgecolors='black', linewidths=1, vmin=-5, vmax=5)
        plt.colorbar(scatter, ax=ax, label='Y Height', shrink=0.8)

        # Number the points
        for i, (x, z) in enumerate(zip(xs, zs)):
            ax.annotate(str(i+1), (x, z), textcoords="offset points", xytext=(5, 5), fontsize=10, fontweight='bold')

        # Mark start and end
        ax.scatter([xs[0]], [zs[0]], s=200, marker='o', facecolors='none', edgecolors='green', linewidths=3, label='Start')
        if len(xs) > 1:
            ax.scatter([xs[-1]], [zs[-1]], s=200, marker='s', facecolors='none', edgecolors='red', linewidths=3, label='End')

    ax.set_xlabel('X (Left ← → Right)')
    ax.set_ylabel('Z (Front ← → Back)')
    ax.set_title(f'Top-Down View (click to add points)\n{len(points) if points else 0} points')
    if points and len(points) > 0:
        ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plt.close(fig)

    from PIL import Image
    return np.array(Image.open(buf))

def draw_trajectory_on_frame(frame, points, cam_dist, cam_height, cam_fov):
    """Draw 3D trajectory points projected onto video frame (0° view)"""
    if frame is None:
        return None

    import math

    display = frame.copy()
    h, w = display.shape[:2]

    # Camera intrinsics for 0° view
    fov_rad = math.radians(cam_fov)
    focal_length = h / (2 * math.tan(fov_rad / 2))
    cx, cy = w / 2, h / 2

    # Camera position for 0° (front) - looking at origin from -Z
    cam_pos = np.array([0, cam_height, -cam_dist])

    def project_point(point_3d):
        """Project 3D point to 2D for 0° camera"""
        x, y, z = point_3d

        # Transform to camera space (camera at 0° looks along +Z)
        # Camera coordinate: right=+X, down=+Y, forward=+Z
        x_cam = x - cam_pos[0]
        y_cam = -(y - cam_pos[1])  # Flip Y (image Y is down)
        z_cam = z - cam_pos[2]  # Distance along viewing direction

        if z_cam <= 0.1:  # Behind camera
            return None

        # Perspective projection
        u = int(focal_length * x_cam / z_cam + cx)
        v = int(focal_length * y_cam / z_cam + cy)

        return (u, v)

    projected_points = []
    for pt in points:
        proj = project_point(pt)
        if proj:
            projected_points.append((proj, pt))

    # Draw trajectory line
    if len(projected_points) > 1:
        for i in range(len(projected_points) - 1):
            pt1 = projected_points[i][0]
            pt2 = projected_points[i + 1][0]
            cv2.line(display, pt1, pt2, (255, 255, 0), 2, cv2.LINE_AA)

    # Draw points with color based on Y height and numbers
    for i, (proj, orig) in enumerate(projected_points):
        x, y, z = orig
        # Color based on Y height: blue (low) -> green (0) -> red (high)
        if y < 0:
            color = (255, int(255 * (1 + y/3)), 0)  # Blue to green
        else:
            color = (int(255 * (1 - y/3)), 255, 0)  # Green to red
        color = (0, 255, 0)  # Simplified: green for all

        # Draw point
        cv2.circle(display, proj, 10, color, -1)
        cv2.circle(display, proj, 10, (0, 0, 0), 2)

        # Draw point number
        cv2.putText(display, str(i + 1), (proj[0] + 12, proj[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, str(i + 1), (proj[0] + 12, proj[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # Draw Y height indicator
        y_text = f"Y:{y:.1f}"
        cv2.putText(display, y_text, (proj[0] - 20, proj[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    # Mark start (green circle) and end (red square)
    if len(projected_points) > 0:
        start_pt = projected_points[0][0]
        cv2.circle(display, start_pt, 15, (0, 255, 0), 3)

    if len(projected_points) > 1:
        end_pt = projected_points[-1][0]
        cv2.rectangle(display, (end_pt[0]-12, end_pt[1]-12), (end_pt[0]+12, end_pt[1]+12), (0, 0, 255), 3)

    # Draw info overlay
    info_text = f"Points: {len(points)} | Cam Dist: {cam_dist} | FOV: {cam_fov}"
    cv2.putText(display, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(display, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    return display


def draw_trajectory_2d_on_frame(frame, points_2d):
    """Draw 2D trajectory points on video frame.

    Args:
        frame: RGB image (H, W, 3)
        points_2d: List of (x, y) pixel coordinates

    Returns:
        frame with trajectory drawn
    """
    if frame is None:
        return None

    display = frame.copy()

    # Draw trajectory line
    if len(points_2d) > 1:
        for i in range(len(points_2d) - 1):
            pt1 = (int(points_2d[i][0]), int(points_2d[i][1]))
            pt2 = (int(points_2d[i + 1][0]), int(points_2d[i + 1][1]))
            cv2.line(display, pt1, pt2, (255, 255, 0), 2, cv2.LINE_AA)

    # Draw points with numbers
    for i, (px, py) in enumerate(points_2d):
        pt = (int(px), int(py))

        # Draw point (green circle)
        cv2.circle(display, pt, 10, (0, 255, 0), -1)
        cv2.circle(display, pt, 10, (0, 0, 0), 2)

        # Draw point number
        cv2.putText(display, str(i + 1), (pt[0] + 12, pt[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, str(i + 1), (pt[0] + 12, pt[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Mark start (green outline) and end (red outline)
    if len(points_2d) > 0:
        start_pt = (int(points_2d[0][0]), int(points_2d[0][1]))
        cv2.circle(display, start_pt, 15, (0, 255, 0), 3)

        if len(points_2d) > 1:
            end_pt = (int(points_2d[-1][0]), int(points_2d[-1][1]))
            cv2.rectangle(display, (end_pt[0] - 12, end_pt[1] - 12),
                          (end_pt[0] + 12, end_pt[1] + 12), (255, 0, 0), 3)

    return display


def passthrough(video_path):
    return video_path

def on_gallery_select(evt: gr.SelectData):
    return evt.index

def pick_video_by_index(idx, videos):
    if idx is None:
        return None
    if isinstance(idx, (list, tuple)):
        idx = idx[0] if len(idx) > 0 else None
    if idx is None:
        return None
    if 0 <= idx < len(videos):
        return videos[idx]
    return None

def ensure_directory_exists(path):
    """确保目录存在，如果不存在则创建"""
    os.makedirs(path, exist_ok=True)
    return path

def cleanup_temp_files(paths):
    """清理临时文件"""
    for path in paths:
        try:
            if path and os.path.exists(path):
                if os.path.isfile(path):
                    os.unlink(path)
                else:
                    shutil.rmtree(path, ignore_errors=True)
        except Exception as e:
            logger.warning(f"清理临时文件失败 {path}: {e}")

def extract_frames(video_path, output_dir, start_frame=0, end_frame=48):
    """从视频中提取帧"""
    ensure_directory_exists(output_dir)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret or frame_count > end_frame:
            break
            
        if frame_count >= start_frame:
            frame_path = os.path.join(output_dir, f"{frame_count:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            
        frame_count += 1
        
    cap.release()
    return frame_count

def create_video_from_frames(frames_dir, output_path, fps=12):
    """从帧创建视频"""
    frame_names = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))])
    if not frame_names:
        return False
        
    first_frame = cv2.imread(os.path.join(frames_dir, frame_names[0]))
    height, width = first_frame.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_name in frame_names:
        frame = cv2.imread(os.path.join(frames_dir, frame_name))
        out.write(frame)
        
    out.release()
    return True

# -------- Video Processing Functions --------
def adjust_video_resolution(video_path, target_width=720, target_height=480, target_frames=49, target_fps=12):
    """调整视频分辨率、帧数和帧率：
       - 对短视频逐帧读取
       - 对长视频根据总帧数自动跳帧，最多写 target_frames 帧
    """
    if not video_path or not os.path.exists(video_path):
        return None, "Please provide a valid video file"

    temp_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_output_path = temp_output.name
    temp_output.close()

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "Unable to open video file"

        # 先看原视频总帧数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[adjust_video_resolution] total_frames = {total_frames}")

        # -------- 计算跳帧间隔 skip_interval --------
        skip_interval = 0
        if total_frames >= 200 and total_frames < 300:
            skip_interval = 1
        elif total_frames >= 300 and total_frames < 400:
            skip_interval = 2
        elif total_frames >= 400:
            # 对于400帧以上的视频，每增加100帧，跳帧间隔增加1
            # skip_interval = (total_frames // 100) - 2
            skip_interval = 3

        stride = skip_interval + 1
        if skip_interval > 0:
            print(f"[adjust_video_resolution] 启用跳帧功能，跳帧间隔: {skip_interval} (stride={stride})")
        else:
            print("[adjust_video_resolution] 不启用跳帧功能，逐帧读取 (stride=1)")
        # -------------------------------------------

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output_path, fourcc, target_fps, (target_width, target_height))

        sampled_frames = []  # 保存已经写出的帧（resize 后）

        written_frames = 0
        src_idx = 0  # 原视频帧索引

        # 先按 stride 从头采样，直到写满 target_frames 或读完视频
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if src_idx % stride == 0:
                resized_frame = cv2.resize(frame, (target_width, target_height))
                out.write(resized_frame)
                sampled_frames.append(resized_frame)
                written_frames += 1

                if written_frames >= target_frames:
                    break

            src_idx += 1

        cap.release()

        if len(sampled_frames) == 0:
            out.release()
            return None, "No frames written during adjustment"

        # 若采样帧数不足 target_frames，则用 ping-pong 方式补帧
        if written_frames < target_frames:
            n = len(sampled_frames)
            backward_indices = list(range(n - 1, -1, -1))  # 从后往前
            forward_indices = list(range(n))               # 从前往后
            patterns = [backward_indices, forward_indices]
            pattern_idx = 0  # 0 -> backward, 1 -> forward

            while written_frames < target_frames:
                seq = patterns[pattern_idx]
                for idx in seq:
                    if written_frames >= target_frames:
                        break
                    out.write(sampled_frames[idx])
                    written_frames += 1
                pattern_idx = 1 - pattern_idx  # 方向切换

        out.release()
        return temp_output_path, "Adjustment successful"
    except Exception as e:
        return None, f"Adjustment Failed: {str(e)}"


def process_foreground_video(fg_video, fg_points, run_folder=None):
    """使用SAM2处理前景视频，基于用户点击的点"""
    if not fg_video or not fg_points:
        return None, None, None, None, "Please provide a foreground video and at least one click point"

    logger.info(f"Begin processing foreground video: {fg_video}")
    logger.info(f"Foreground Point: {fg_points}")
    
    # Create run folder if not exists
    if run_folder is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = os.path.join("./results/runs", timestamp)
    ensure_directory_exists(run_folder)

    # 调整视频分辨率
    adjusted_fg_video, status = adjust_video_resolution(
        fg_video,
        target_width=720,
        target_height=480,
        target_frames=49,
        target_fps=12
    )

    if not adjusted_fg_video:
        return None, None, None, run_folder, f"Foreground video adjustment failed: {status}"

    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix="fg_processing_")
    frames_dir = os.path.join(temp_dir, "frames")
    mask_dir = os.path.join(temp_dir, "masks")
    element_dir = os.path.join(temp_dir, "elements")
    
    ensure_directory_exists(frames_dir)
    ensure_directory_exists(mask_dir)
    ensure_directory_exists(element_dir)
    
    try:
        # 提取视频帧
        frame_count = extract_frames(adjusted_fg_video, frames_dir, 0, 48)
        if frame_count == 0:
            return None, None, None, run_folder, "Unable to extract video frame"
            
        # 初始化视频预测器状态
        inference_state = video_predictor.init_state(video_path=frames_dir)
        
        # 处理第一帧
        first_frame_path = os.path.join(frames_dir, "00000.jpg")
        image = Image.open(first_frame_path).convert("RGB")
        image_predictor.set_image(np.array(image))

        # Convert clicked points to numpy arrays
        # fg_points is now a list of (x, y, label) tuples
        # Handle both old format (x, y) and new format (x, y, label)
        point_coords = []
        point_labels = []
        for point in fg_points:
            if len(point) == 3:
                x, y, label = point
                point_coords.append([x, y])
                point_labels.append(label)
            else:
                # Backward compatibility: old format (x, y) defaults to positive
                x, y = point
                point_coords.append([x, y])
                point_labels.append(1)

        point_coords = np.array(point_coords, dtype=np.float32)
        point_labels = np.array(point_labels, dtype=np.int32)

        logger.info(f"Point coords: {point_coords}, Labels: {point_labels}")

        # Get mask for first frame
        masks, scores, logits = image_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=None,
            multimask_output=False,
        )
        
        if masks.ndim == 4:
            masks = masks.squeeze(1)
            
        # 注册点到视频预测器
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            points=point_coords,
            labels=point_labels,
        )
        
        # 传播以获取所有帧的分割
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        
        # 处理所有帧，创建掩码和前景元素
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
        
        for frame_idx, frame_file in enumerate(frame_files):
            if frame_idx not in video_segments:
                continue
                
            frame_path = os.path.join(frames_dir, frame_file)
            frame = cv2.imread(frame_path)
            
            # 获取当前帧的掩码
            segments = video_segments[frame_idx]
            masks = list(segments.values())
            if not masks:
                continue
                
            mask = np.concatenate(masks, axis=0)[0]  # 假设单个对象
            mask_uint8 = (mask * 255).astype(np.uint8)
            
            # 保存掩码
            mask_path = os.path.join(mask_dir, f"{frame_idx:05d}.png")
            cv2.imwrite(mask_path, mask_uint8)
            
            # 创建前景元素（白色背景）
            element = frame.copy()
            element[mask == 0] = 255  # 将背景设为白色
            
            # 保存前景元素
            element_path = os.path.join(element_dir, f"{frame_idx:05d}.jpg")
            cv2.imwrite(element_path, element)
        
        # 创建前景元素视频
        element_video_path = os.path.join(temp_dir, "element_video.mp4")
        if not create_video_from_frames(element_dir, element_video_path, fps=12):
            return None, None, None, run_folder, "Failed to create foreground element video"
            
        # 创建掩码视频
        mask_video_path = os.path.join(temp_dir, "mask_video.mp4")
        if not create_video_from_frames(mask_dir, mask_video_path, fps=12):
            return None, None, None, run_folder, "Failed to create mask video"
        
        # Save segmented object video to run folder
        segmented_video_path = os.path.join(run_folder, "1_segmented_object.mp4")
        shutil.copy2(element_video_path, segmented_video_path)
        
        # Save first frame of segmented object
        cap = cv2.VideoCapture(element_video_path)
        ret, first_frame = cap.read()
        cap.release()
        if ret:
            first_frame_path = os.path.join(run_folder, "2_segmented_first_frame.jpg")
            cv2.imwrite(first_frame_path, first_frame)
        
        success_msg = f"Foreground processing successful\nRun folder: {run_folder}"
        return adjusted_fg_video, mask_video_path, element_video_path, run_folder, success_msg
            
    except Exception as e:
        error_msg = f"Foreground video processing failed: {str(e)}"
        logger.error(error_msg)
        return None, None, None, run_folder, error_msg
    finally:
        # 延迟清理 adjusted_fg_video，直到整个流程完成
        pass  # 移除 cleanup_temp_files([adjusted_fg_video])

def auto_adjust_background_video(video_path):
    """自动调整背景视频"""
    if not video_path:
        return None, "Please provide background video"

    adjusted_video, status = adjust_video_resolution(
        video_path,
        target_width=720,
        target_height=480,
        target_frames=49,
        target_fps=12
    )

    return adjusted_video, status

def auto_adjust_foreground_video(video_path):
    """自动调整前景视频"""
    if not video_path:
        return None, "Please provide foreground video"

    adjusted_video, status = adjust_video_resolution(
        video_path,
        target_width=720,
        target_height=480,
        target_frames=49,
        target_fps=12
    )

    return adjusted_video, status

def draw_and_save_trajectory(source_video_path, trajectory_points_json, trajectory_txt_path, target_width=720,
                             target_height=480):
    """绘制并保存轨迹，通过Gradio点击关键点连接成线，使用处理后的分辨率"""
    adjusted_video, _ = adjust_video_resolution(source_video_path, target_width, target_height)
    if not adjusted_video:
        return None, None, "Unable to adjust video resolution"

    cap = cv2.VideoCapture(adjusted_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return None, None, "Unable to read video file"

    first_frame = cv2.resize(first_frame, (target_width, target_height))

    trajectory_points = []
    if trajectory_points_json:
        try:
            trajectory_points = json.loads(trajectory_points_json)
        except:
            cap.release()
            return None, None, "无效的轨迹点数据"

    if not trajectory_points:
        cap.release()
        return None, None, "未选择任何关键点"

    if len(trajectory_points) == 1:
        prev_frame = first_frame
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_point = np.array([[trajectory_points[0]]], dtype=np.float32)
        flow_points = [tuple(trajectory_points[0])]

        for _ in range(total_frames - 1):
            ret, next_frame = cap.read()
            if not ret:
                break
            next_frame = cv2.resize(next_frame, (target_width, target_height))
            next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
            next_point, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, next_gray, prev_point, None
            )
            if status[0] == 1:
                new_point = tuple(map(int, next_point[0][0]))
                flow_points.append(new_point)
                prev_point = next_point
                prev_gray = next_gray
            else:
                flow_points.append(flow_points[-1])

        trajectory_points = flow_points[:total_frames]

    elif len(trajectory_points) >= 2:
        interpolated_points = []
        segments = len(trajectory_points) - 1
        points_per_segment = total_frames // segments
        for i in range(segments):
            start_point = trajectory_points[i]
            end_point = trajectory_points[i + 1]
            n_points = points_per_segment if i < segments - 1 else (total_frames - (segments - 1) * points_per_segment)
            for j in range(n_points):
                ratio = j / max(1, n_points - 1)
                x = int(start_point[0] + ratio * (end_point[0] - start_point[0]))
                y = int(start_point[1] + ratio * (end_point[1] - start_point[1]))
                interpolated_points.append((x, y))
        trajectory_points = interpolated_points[:total_frames]

    with open(trajectory_txt_path, 'w') as f:
        for point in trajectory_points:
            f.write(f"{point[0]},{point[1]}\n")

    vis_frame = first_frame.copy()
    for i in range(1, len(trajectory_points)):
        cv2.line(vis_frame, trajectory_points[i - 1], trajectory_points[i], (0, 0, 255), 2)
    for point in trajectory_points:
        cv2.circle(vis_frame, point, 5, (0, 255, 0), -1)

    vis_path = trajectory_txt_path.replace('.txt', '_vis.jpg')
    cv2.imwrite(vis_path, vis_frame)

    cap.release()
    if os.path.exists(adjusted_video):
        os.unlink(adjusted_video)
    return vis_path, trajectory_txt_path, f"轨迹已保存至 {trajectory_txt_path}"

def load_trajectory(trajectory_path, camera_angle=None, camera_distance=10.0, camera_height=0.0, camera_fov=60.0):
    """
    Load trajectory coordinates from file.
    Supports both 2D (legacy) and 3D formats.

    Args:
        trajectory_path: Path to trajectory file
        camera_angle: Required for 3D trajectories - angle in degrees (0, 90, 180, 270)
        camera_distance: Camera distance from origin (for 3D projection)
        camera_height: Camera Y position (for 3D projection)
        camera_fov: Field of view in degrees (for 3D projection)

    Returns:
        List of (x, y) 2D coordinates
    """
    # Import 3D trajectory module
    from infer.trajectory_3d import (
        is_3d_trajectory, load_trajectory_3d, Camera, CameraConfig
    )

    if is_3d_trajectory(trajectory_path):
        # 3D trajectory - requires projection
        if camera_angle is None:
            raise ValueError("3D trajectory requires camera_angle parameter for projection")

        trajectory_3d = load_trajectory_3d(trajectory_path)

        # Create camera config for this view
        config = CameraConfig(
            angle=camera_angle,
            distance=camera_distance,
            height=camera_height,
            fov=camera_fov,
            image_width=720,
            image_height=480
        )

        camera = Camera(config)
        trajectory = []
        default_point = (360, 240)  # Image center as fallback

        for point in trajectory_3d.points:
            point_2d = camera.project_3d_to_2d(np.array(point))
            if point_2d is None:
                # Point behind camera - use last valid or default
                point_2d = trajectory[-1] if trajectory else default_point
            trajectory.append(point_2d)

        return trajectory
    else:
        # Original 2D format
        trajectory = []
        with open(trajectory_path, 'r') as f:
            for line in f:
                if line.strip():
                    x, y = map(int, line.strip().split(','))
                    trajectory.append((x, y))
        return trajectory

def generate_mask_video_with_trajectory(fg_element_path, source_video_path, output_path, trajectory_path, scales=[1.0],
                                        target_width=720, target_height=480, alignment="center", run_folder=None):
    """生成带轨迹的掩膜视频 - 使用前景元素视频而不是掩码视频，支持alignment参数和动态scales"""
    trajectory = load_trajectory(trajectory_path)
    if not trajectory:
        return None, "Unable to load trajectory file"

    fg_cap = cv2.VideoCapture(fg_element_path)
    source_cap = cv2.VideoCapture(source_video_path)

    source_width = int(source_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(source_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = source_cap.get(cv2.CAP_PROP_FPS)

    frame_count = min(
        int(fg_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(source_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        len(trajectory)
    )

    # 过滤有效的 scales（非 None 且非 0），并沿用最后一个有效值填充后续未指定的 scales
    valid_scales = [s for s in scales if s is not None and s != 0]
    if not valid_scales:
        valid_scales = [1.0]  # 默认值
    else:
        last_valid = valid_scales[-1]
        valid_scales.extend([last_valid] * (len(scales) - len(valid_scales)))  # 填充未指定的 scales

    n_scales = len(valid_scales)
    if n_scales == 0:
        valid_scales = [1.0]
        n_scales = 1

    # 计算关键帧位置，仅考虑有效 scales
    if n_scales > 1:
        key_frames = [int(i * (frame_count - 1) / (n_scales - 1)) for i in range(n_scales)]
    else:
        key_frames = [0]

    # Use run folder or create new one
    if run_folder is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = os.path.join("./results/runs", timestamp)
    ensure_directory_exists(run_folder)
    
    # Create mask frames directory in run folder
    mask_frames_dir = os.path.join(run_folder, "mask_frames")
    ensure_directory_exists(mask_frames_dir)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_mask = cv2.VideoWriter(output_path, fourcc, fps, (source_width, source_height), isColor=False)

    frame_num = 0
    while frame_num < frame_count:
        ret_fg, fg_frame = fg_cap.read()
        ret_source, source_frame = source_cap.read()

        if not ret_fg or not ret_source:
            break

        center_x, center_y = trajectory[frame_num]

        # 计算当前 scale
        if n_scales == 1:
            current_scale = valid_scales[0]
        else:
            for j in range(1, n_scales):
                if frame_num <= key_frames[j]:
                    left = key_frames[j-1]
                    right = key_frames[j]
                    left_scale = valid_scales[j-1]
                    right_scale = valid_scales[j]
                    if left == right:
                        current_scale = left_scale
                    else:
                        ratio = (frame_num - left) / (right - left)
                        current_scale = left_scale + ratio * (right_scale - left_scale)
                    break
            else:
                current_scale = valid_scales[-1]

        gray = cv2.cvtColor(fg_frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros((source_height, source_width), dtype=np.uint8)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            if current_scale != 1.0:
                M = cv2.moments(largest_contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    scaling_matrix = np.array([[current_scale, 0, (1 - current_scale) * cx],
                                               [0, current_scale, (1 - current_scale) * cy]])
                    scaled_contour = cv2.transform(largest_contour, scaling_matrix).astype(np.int32)
                    x, y, w, h = cv2.boundingRect(scaled_contour)
                else:
                    scaled_contour = largest_contour.astype(np.int32)
            else:
                scaled_contour = largest_contour.astype(np.int32)

            if alignment == "center":
                start_x = center_x - w // 2
                start_y = center_y - h // 2
            elif alignment == "bottom":
                start_x = center_x - w // 2
                start_y = center_y - h
            else:
                raise ValueError(f"Unknown alignment: {alignment}")

            target_x1 = max(0, start_x)
            target_y1 = max(0, start_y)
            target_x2 = min(source_width, start_x + w)
            target_y2 = min(source_height, start_y + h)

            src_x1 = max(0, -start_x)
            src_y1 = max(0, -start_y)
            src_x2 = w - max(0, (start_x + w) - source_width)
            src_y2 = h - max(0, (start_y + h) - source_height)

            contour_offset = np.array([target_x1 - src_x1, target_y1 - src_y1])

            current_contour = scaled_contour if current_scale != 1.0 else largest_contour.astype(np.int32)
            shifted_contour = current_contour + contour_offset - np.array([x, y])

            cv2.fillPoly(final_mask, [shifted_contour.astype(np.int32)], 255)

        out_mask.write(final_mask)
        
        # 保存掩码帧到run folder
        mask_frame_path = os.path.join(mask_frames_dir, f"{frame_num:05d}.png")
        cv2.imwrite(mask_frame_path, final_mask)

        frame_num += 1

    fg_cap.release()
    source_cap.release()
    out_mask.release()
    
    # 保存掩码视频到run folder
    mask_video_path = os.path.join(run_folder, "3_mask_video.mp4")
    shutil.copy2(output_path, mask_video_path)
    
    success_message = (
        f"Mask video generated successfully\n"
        f"Mask saved to: {run_folder}\n"
        f"- Video: 3_mask_video.mp4\n"
        f"- Frames: mask_frames/ ({frame_num} frames)"
    )
    
    return output_path, success_message

def quick_freeze(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

def apply_consistent_gamma(frames, gamma=None, gamma_range=(0.5, 2)):
    frames_normalized = frames.float() / 255.0
    if gamma is None:
        gamma = random.uniform(*gamma_range)
    corrected_frames = torch.pow(frames_normalized, gamma) * 255.0
    return corrected_frames.to(frames.dtype), gamma

def get_gaussian_kernel(kernel_size, sigma, channels):
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.
    gaussian_kernel = torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels,
                                bias=False, padding=kernel_size // 2)
    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    return gaussian_filter

def read_video_with_mask(video_path, fg_video_path, masks, skip_frames_start=0, skip_frames_end=-1, fps=0):
    to_tensor = ToTensor()
    video = load_video(video_path)[skip_frames_start:skip_frames_end]
    mask = load_video(masks)[skip_frames_start:skip_frames_end]
    fg_video = load_video(fg_video_path)[skip_frames_start:skip_frames_end]
    if fps == 0:
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
    masked_video = []
    binary_masks = []
    fg_resized = []
    fgy_resized = []
    gaussian_filter = quick_freeze(get_gaussian_kernel(kernel_size=51, sigma=10, channels=1)).to("cuda")
    for frame, frame_mask, fg_frame in zip(video, mask, fg_video):
        frame_array = np.array(frame)
        fg_frame_array = np.array(fg_frame)
        target_height, target_width = frame_array.shape[0], frame_array.shape[1]
        frame_mask_array = np.array(frame_mask)
        if len(frame_mask_array.shape) == 3:
            frame_mask_array = cv2.cvtColor(frame_mask_array, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(frame_mask_array, 128, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_mask, 8, cv2.CV_32S)
        filtered_mask = np.zeros_like(cleaned_mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > target_height * target_width * 0.0001:
                filtered_mask[labels == i] = 255
        frame_tensor = to_tensor(filtered_mask / 255.0).unsqueeze(0).to("cuda").float()
        with torch.no_grad():
            filtered_tensor = gaussian_filter(frame_tensor)
            # filtered_tensor = frame_tensor
        binary_mask = (filtered_tensor.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        black_frame = np.zeros_like(frame_array)
        binary_mask_expanded = np.repeat(binary_mask[:, :, np.newaxis], 3, axis=2)
        masked_frame = np.where(binary_mask_expanded, black_frame, frame_array)
        masked_video.append(Image.fromarray(masked_frame.astype(np.uint8)).convert("RGB"))
        binary_mask_image = np.where(binary_mask, 255, 0).astype(np.uint8)
        binary_masks.append(Image.fromarray(binary_mask_image).convert("RGB"))
        height, width = fg_frame_array.shape[0], fg_frame_array.shape[1]
        fg_frame_resized = cv2.resize(fg_frame_array, (target_height, target_height), interpolation=cv2.INTER_CUBIC)
        pad_width = target_width - fg_frame_resized.shape[1]
        pad_left = pad_width // 2
        fg_frame_final = np.full((target_height, target_width, fg_frame_resized.shape[2]), 255,
                                dtype=fg_frame_resized.dtype)
        fg_frame_final[:, pad_left:pad_left + fg_frame_resized.shape[1], :] = fg_frame_resized
        fg_resized.append(Image.fromarray(fg_frame_final).convert("RGB"))
        gray = np.dot(fg_frame_final[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        gray_3ch = np.stack((gray, gray, gray), axis=-1)
        fgy_resized.append(Image.fromarray(gray_3ch))
    video = [item.convert("RGB") for item in video]
    return video, masked_video, binary_masks, fps, fg_resized, fgy_resized

def generate_video(
        output_path: str = "./output.mp4",
        video_path: str = "",
        mask_path: str = "",
        fg_video_path: str = "",
        num_inference_steps: int = 10,
        guidance_scale: float = 6.0,
        num_videos_per_prompt: int = 1,
        generate_type: str = "i2v_inpainting",
        seed: int = 42,
        inpainting_frames: int = 49,
        mask_background: bool = False,
        add_first: bool = False,
        first_frame_gt: bool = False,
        replace_gt: bool = False,
        mask_add: bool = True,
        down_sample_fps: int = 8,
        overlap_frames: int = 0,
        prev_clip_weight: float = 0.0,
        start_frame: int = 0,
        end_frame: int = 49,
        img_inpainting_model: str = None,
        llm_model: str = None,
        long_video: bool = False,
        dilate_size: int = -1,
        id_adapter_resample_learnable_path: str = None,
        run_folder: str = None,
):
    fps = 12
    # Using fixed prompt
    prompt = "A realistic scene with objects moving naturally"
    
    try:
        # Validate input paths
        if not os.path.exists(video_path):
            raise ValueError(f"Video path {video_path} does not exist")
        if not os.path.exists(mask_path):
            raise ValueError(f"Mask path {mask_path} does not exist")
        if not os.path.exists(fg_video_path):
            raise ValueError(f"Foreground video path {fg_video_path} does not exist")

        video, masked_video, binary_masks, fps, fg_video, fgy_video = read_video_with_mask(
            video_path, fg_video_path=fg_video_path, skip_frames_start=start_frame, skip_frames_end=end_frame,
            masks=mask_path, fps=fps
        )

        if generate_type == "i2v_inpainting":
            frames = inpainting_frames  # 一般是 49

            total_frames = len(video)
            if total_frames == 0:
                raise ValueError("No frames loaded from video/mask/foreground")

            # 只考虑全部帧作为候选
            max_use = total_frames

            # -------- 根据“原视频总帧数”决定跳帧间隔 --------
            skip_interval = 0
            if total_frames >= 200 and total_frames < 300:
                skip_interval = 1
            elif total_frames >= 300 and total_frames < 400:
                skip_interval = 2
            elif total_frames >= 400:
                # 对于400帧以上的视频，每增加100帧，跳帧间隔增加1
                skip_interval = (total_frames // 100) - 2
            stride = skip_interval + 1

            if skip_interval > 0:
                print(f"[generate_video] Enable frame skipping function, frame skipping interval: {skip_interval} (stride={stride}), total_frames={total_frames}")
            else:
                print(f"[generate_video] Disable frame skipping, use frame-by-frame (stride=1), total_frames={total_frames}")
            # ------------------------------------------------
            # 先按 stride 从 0 ~ max_use-1 采样一串候选索引
            indices = list(range(0, max_use, stride))

            # 如果采出来的索引比 49 多，就截到 49；不够 49 就循环补齐（改为 ping-pong）
            if len(indices) >= frames:
                indices = indices[:frames]
            else:
                if len(indices) == 0:
                    # 极端情况：max_use=0 已经在前面报错了，理论不会到这里
                    indices = [0] * frames
                else:
                    base = indices.copy()
                    # base 是一次从前到后的采样结果，例如 [0, stride, 2*stride, ...]
                    forward_indices = base
                    backward_indices = list(reversed(base))
                    patterns = [backward_indices, forward_indices]  # 先后->前，再前->后
                    pattern_idx = 0  # 0 表示先用 backward，再用 forward

                    while len(indices) < frames:
                        seq = patterns[pattern_idx]
                        for idx in seq:
                            if len(indices) >= frames:
                                break
                            indices.append(idx)
                        # 切换方向：backward <-> forward
                        pattern_idx = 1 - pattern_idx


            # 按选中的索引重排所有序列，保证长度一致且 = frames
            video = [video[i] for i in indices]
            masked_video = [masked_video[i] for i in indices]
            binary_masks = [binary_masks[i] for i in indices]
            fg_video = [fg_video[i] for i in indices]
            fgy_video = [fgy_video[i] for i in indices]


            
            # Save the downsampled mask by reading the clean mask video and downsampling it
            if run_folder and os.path.exists(mask_path):
                # Read the clean 100-frame mask video
                clean_mask_cap = cv2.VideoCapture(mask_path)
                clean_fps = int(clean_mask_cap.get(cv2.CAP_PROP_FPS))
                
                downsampled_mask_frames_dir = os.path.join(run_folder, "mask_frames_49")
                ensure_directory_exists(downsampled_mask_frames_dir)
                
                # Downsample the clean mask the same way as the video
                frame_idx = 0
                saved_idx = 0
                clean_masks = []
                
                while frame_idx < 100:
                    ret, mask_frame = clean_mask_cap.read()
                    if not ret:
                        break
                    
                    # Apply same downsampling logic
                    if frame_idx % int(fps // down_sample_fps) == 0 and saved_idx < frames:
                        if len(mask_frame.shape) == 3:
                            mask_gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
                        else:
                            mask_gray = mask_frame
                        
                        clean_masks.append(mask_gray)
                        mask_frame_path = os.path.join(downsampled_mask_frames_dir, f"{saved_idx:05d}.png")
                        cv2.imwrite(mask_frame_path, mask_gray)
                        saved_idx += 1
                    
                    frame_idx += 1
                
                clean_mask_cap.release()
                
                # Create downsampled mask video from clean masks
                downsampled_mask_video_path = os.path.join(run_folder, "4_mask_49frames.mp4")
                if len(clean_masks) > 0:
                    height, width = clean_masks[0].shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(downsampled_mask_video_path, fourcc, 12, (width, height), isColor=False)
                    for mask_frame in clean_masks:
                        out.write(mask_frame)
                    out.release()
                
                logger.info(f"Saved {len(clean_masks)}-frame clean mask to {run_folder}")
            
            inpaint_outputs = pipe(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                num_inference_steps=num_inference_steps,
                num_frames=frames,
                use_dynamic_cfg=True,
                guidance_scale=guidance_scale,
                generator=torch.Generator().manual_seed(seed),
                video=masked_video,
                masked_video=masked_video,
                mask_video=binary_masks,
                fg_video=fg_video,
                strength=1.0,
                replace_gt=replace_gt,
                mask_add=mask_add,
                id_pool_resample_learnable=False,
                output_type="np"
            ).frames[0]

            torch.cuda.empty_cache()
            video_generate = inpaint_outputs
            output_dir = './results'
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            export_to_video(video_generate, output_path, fps=12)
            
            # Save final composited video and first frame to run folder
            if run_folder:
                final_video_path = os.path.join(run_folder, "5_final_composited.mp4")
                shutil.copy2(output_path, final_video_path)
                
                # Save first frame of final video
                cap = cv2.VideoCapture(output_path)
                ret, first_frame = cap.read()
                cap.release()
                if ret:
                    final_first_frame_path = os.path.join(run_folder, "6_final_first_frame.jpg")
                    cv2.imwrite(final_first_frame_path, first_frame)
                
                return output_path, f"Video compositing successful\nAll outputs saved to: {run_folder}"
            
            return output_path, "Video compositing successful"
        else:
            raise NotImplementedError
    except Exception as e:
        return None, f"Video compositing failed: {str(e)}"
    finally:
        torch.cuda.empty_cache()

# -------- Your samples --------
sample_videos = [
    "../assets/bg/sora15.mp4",
    "../assets/bg/42858.mp4",
    "../assets/fg/44867.mp4",
    "../assets/fg/45871.mp4",
    "../assets/fg/47853.mp4",
    "../assets/fg/fg_butterfly.mp4",
]

thumbnails = [get_first_frame(p) for p in sample_videos]
cards = []
for i, (thumb, path) in enumerate(zip(thumbnails, sample_videos)):
    title = os.path.basename(path)
    if thumb is not None:
        thumb_rgb = thumb[:, :, ::-1]
        cards.append([thumb_rgb, f"{i + 1}. {title}"])

# -------- Gradio Interface --------
css = """
.composed-video {
    width: 100% !important;
    max-width: 100% !important;
    margin: 20px auto;
    display: block;
    border-radius: 8px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

.step-group {
    border: 2px solid #4A90E2;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 30px;
    background-color: #F9FAFB;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    overflow: hidden !important;
}

.rounded-button {
    border-radius: 12px !important;
    width: 180px !important;
    padding: 10px;
    font-weight: 600;
    transition: background-color 0.3s ease, transform 0.2s ease;
}

.rounded-button:hover {
    transform: translateY(-2px);
    background-color: #add8e6 !important;
}

.bg-button {
    border-radius: 12px !important;
    width: 600px !important;
    margin: 0 auto !important;
    display: block !important;
    padding: 12px;
    font-weight: 600;
}

.custom-button {
    background-color: #add8e6 !important;
    color: black !important;
    border: 1px solid #add8e6 !important;
    font-size: 24px;
}

.custom-button:hover {
    background-color: #add8e6 !important;
}

.step5-container {
    width: 100% !important;
    max-width: 100% !important;
    margin: 0 auto;
    padding: 20px;
}

.step5-video {
    width: 100% !important;
    max-width: 100% !important;
    border-radius: 8px;
}

.step5-title {
    text-align: center !important;
    font-size: 2.5em;
    color: #1F2937;
    margin-bottom: 20px;
    font-weight: 700;
}

.video-display {
    height: 360px !important;
    border-radius: 8px;
    border: 1px solid #E5E7EB;
    background-color: white !important;
}

.image-display {
    height: 360px !important;
    border-radius: 8px;
    border: 1px solid #E5E7EB;
    background-color: white !important;
}

.video-container, .image-container {
    background-color: white !important;
    padding: 10px;
    border-radius: 8px;
}

gradio-container video, gradio-container img {
    background-color: white !important;
    border-radius: 6px;
}

gradio-container {
    background-color: #F3F4F6 !important;
    font-family: 'Inter', sans-serif;
}

h1, h2, h3 {
    font-family: 'Inter', sans-serif;
    color: #1F2937;
    text-align: center;
    letter-spacing: 1px;
    margin-bottom: 30px;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
}

h1 {
    font-size: 2.5em;
    font-weight: 800;
    background: linear-gradient(45deg, #2563EB, #60A5FA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

h2 {
    font-size: 1.8em;
    font-weight: 700;
    border-bottom: 2px solid #4A90E2;
    padding-bottom: 10px;
}

.step-number {
    color: #FF4500; /* Deep orange color for Step numbers */
}

.bold-text {
    font-weight: bold;
}

/* 新增样式 */
.button-spacing {
    gap: 20px !important;
}

.gallery-column {
    flex: 4 !important;
    overflow: auto;
}

.preview-column {
    flex: 3 !important;
}

.quick-guide {
    font-size: 1.2em;
    font-weight: bold;
    text-align: left;
    margin-bottom: 10px;
}

.label-text {
    font-size: 1.2em;
    font-weight: 600;
    color: #1F2937;
    margin-bottom: 2px;
    margin-top: 5px;
    text-align: left;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown(
        """
        # GenCompositor Demo
        ### Create stunning video compositing with ease.

        <span class='quick-guide'>💡 Quick Guide</span>
        - We provide some video samples for you. You can choose to use them as foreground or background footage (set '**Yes**'), or upload your own videos (set '**No**').
        - **<span class='step-number'>Step 1</span>**: Click the "**Begin selecting object**" button, then click on the first frame of the foreground video to select the desired element. Click "**Segment foreground element**" to segment it.
        - **<span class='step-number'>Step 2</span>**: Click the "**Begin specifying trajectory**" button, then click on the first frame of the background video to design the desired trajectory. Click "**Complete trajectory drawing**" to finish.
        - **<span class='step-number'>Step 3</span>**: Set the "**Rescale parameters**" and "**Inference steps number**" (default: 10), then click "**Generate middle mask for compositing**" to create the mask video. Click "**Generate composited video**" to obtain the results.
        - **<span class='step-number'>(Optional)</span>**: To remove an object from the video, set "**Inference steps number**" (default: 10), click "**Generate mask for removal**" and then "**Generate removed video**" in the "Extended Application" section.
        """
    )

    processed_bg_video = gr.State(None)
    processed_fg_element = gr.State(None)
    processed_fg_mask = gr.State(None)
    processed_fg_source = gr.State(None)
    trajectory_file = gr.State(None)
    selected_sample_video = gr.State(None)
    trajectory_visualization = gr.State(None)
    trajectory_points = gr.State([])
    original_bg_frame = gr.State(None)
    final_mask_output_state = gr.State(None)
    is_drawing = gr.State(False)
    fg_points = gr.State([])  # List of (x, y, label) tuples - label: 1=positive, 0=negative
    is_drawing_fg = gr.State(False)
    fg_point_mode = gr.State("positive")  # "positive" or "negative"
    original_fg_frame = gr.State(None)
    inference_steps = gr.State(10)  # New state for inference steps
    run_folder = gr.State(None)  # Track the current run folder

    with gr.Group(elem_classes="step-group"):
        gr.Markdown("### Using our video samples?")
        use_samples = gr.Radio(choices=["Yes", "No"], value="Yes", label="", interactive=True)

        with gr.Row(visible=True) as samples_row:
            with gr.Column(scale=4, elem_classes="gallery-column"):
                gallery = gr.Gallery(
                    value=cards,
                    label="Video Samples. Click to select the sample video and it will play on the right.",
                    show_label=True,
                    columns=[3],
                    height="360px",
                )
            with gr.Column(scale=3, elem_classes="preview-column"):
                sample_preview = gr.Video(
                    label="Selected Video Preview. Click buttons below to determine whether it is foreground or background",
                    interactive=False,
                    height=360,
                    elem_classes="video-display"
                )
                with gr.Row():
                    select_fg_gallery = gr.Button(
                        "Select as foreground",
                        variant="primary",
                        elem_classes=["custom-button", "rounded-button"],
                        visible=True
                    )
                    select_bg_gallery = gr.Button(
                        "Select as background",
                        variant="primary",
                        elem_classes=["custom-button", "rounded-button"],
                        visible=True
                    )

    selected_idx = gr.State(None)

    def update_sample_preview(evt: gr.SelectData):
        idx = evt.index
        if idx is None:
            return None, None, None

        if 0 <= idx < len(sample_videos):
            video_path = sample_videos[idx]
            return video_path, video_path, idx
        return None, None, None

    gallery.select(
        fn=update_sample_preview,
        inputs=None,
        outputs=[selected_sample_video, sample_preview, selected_idx]
    )

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Group(elem_classes="step-group"):
                gr.Markdown("## <span class='step-number'>Step 1</span>: Foreground Video Processing")
                with gr.Row(visible=False) as fg_upload_row:
                    upload_fg_button = gr.UploadButton(
                        "Upload your foreground video",
                        file_types=[".mp4"],
                        variant="primary",
                        elem_classes=["custom-button", "bg-button"]
                    )
                fg_video_input = gr.Video(label="Foreground video")
                with gr.Row(elem_classes="button-spacing"):
                    btn_start_drawing_fg = gr.Button(
                        "Begin selecting object",
                        variant="primary",
                        elem_classes=["custom-button", "rounded-button"]
                    )
                    btn_clear_fg_points = gr.Button(
                        "Clear selected points",
                        elem_classes=["custom-button", "rounded-button"]
                    )
                with gr.Row(elem_classes="button-spacing"):
                    btn_positive_mode = gr.Button(
                        "✓ Positive (Include)",
                        variant="primary",
                        elem_classes=["custom-button", "rounded-button"]
                    )
                    btn_negative_mode = gr.Button(
                        "✗ Negative (Exclude)",
                        variant="secondary",
                        elem_classes=["custom-button", "rounded-button"]
                    )
                fg_status = gr.Textbox(label="Foreground video processing status", interactive=False)
                fg_first_frame = gr.Image(
                    label="Please click to select an object",
                    interactive=True,
                    width=900,
                    height=360,
                    elem_classes="image-display"
                )
                with gr.Row():
                    btn_process_fg = gr.Button(
                        "Segment foreground element",
                        variant="primary",
                        elem_classes=["custom-button", "rounded-button"]
                    )
                fg_element_output = gr.Video(
                    label="Foreground element video",
                    height=360,
                    elem_classes="video-display",
                    format="mp4"
                )

                def update_fg_video(video_path):
                    if not video_path:
                        return None, None, None, None, "Please provide foreground video", [], None
                    adjusted_video, status = auto_adjust_foreground_video(video_path)
                    frame = get_first_frame(adjusted_video)
                    frame_rgb = cv2.resize(frame, (720, 480))[:, :, ::-1] if frame is not None else None
                    return None, None, adjusted_video, frame_rgb, status, [], frame_rgb

                fg_video_input.change(
                    fn=update_fg_video,
                    inputs=[fg_video_input],
                    outputs=[fg_element_output, processed_fg_element, processed_fg_source, fg_first_frame, fg_status, fg_points, original_fg_frame]
                )

                select_fg_gallery.click(
                    fn=lambda idx: pick_video_by_index(idx, sample_videos),
                    inputs=selected_idx,
                    outputs=fg_video_input,
                )

                def process_fg_upload(uploaded_file):
                    return uploaded_file.name if uploaded_file else None

                upload_fg_button.upload(
                    fn=process_fg_upload,
                    inputs=[upload_fg_button],
                    outputs=[fg_video_input]
                )

                def start_drawing_fg():
                    return True, "Begin selecting object. Use Positive (green) to include, Negative (red) to exclude."

                btn_start_drawing_fg.click(
                    fn=start_drawing_fg,
                    inputs=[],
                    outputs=[is_drawing_fg, fg_status]
                )

                def set_positive_mode():
                    return "positive", "Mode: Positive (Include) - Click on parts to INCLUDE in mask"

                def set_negative_mode():
                    return "negative", "Mode: Negative (Exclude) - Click on parts to EXCLUDE from mask"

                btn_positive_mode.click(
                    fn=set_positive_mode,
                    inputs=[],
                    outputs=[fg_point_mode, fg_status]
                )

                btn_negative_mode.click(
                    fn=set_negative_mode,
                    inputs=[],
                    outputs=[fg_point_mode, fg_status]
                )

                def add_fg_point(evt: gr.SelectData, original_fg_frame, current_points, is_drawing_fg, point_mode):
                    if not is_drawing_fg:
                        return gr.update(), current_points, "Please click the 'Begin selecting objects' button first"
                    if original_fg_frame is None:
                        return gr.update(), current_points, "Please select foreground element first"

                    x, y = evt.index
                    x = min(max(0, int(x)), 720)
                    y = min(max(0, int(y)), 480)

                    # Label: 1 for positive (foreground), 0 for negative (background)
                    label = 1 if point_mode == "positive" else 0
                    new_points = current_points + [(x, y, label)]

                    display_frame = original_fg_frame.copy()
                    for point in new_points:
                        px, py, plabel = point
                        # Green for positive, Red for negative
                        color = (0, 255, 0) if plabel == 1 else (255, 0, 0)
                        cv2.circle(display_frame, (px, py), 8, color, -1)

                    pos_count = sum(1 for p in new_points if p[2] == 1)
                    neg_count = sum(1 for p in new_points if p[2] == 0)
                    mode_str = "positive" if label == 1 else "negative"
                    return display_frame, new_points, f"Added {mode_str} point ({x}, {y}). Total: {pos_count} positive, {neg_count} negative"

                fg_first_frame.select(
                    fn=add_fg_point,
                    inputs=[original_fg_frame, fg_points, is_drawing_fg, fg_point_mode],
                    outputs=[fg_first_frame, fg_points, fg_status]
                )

                def clear_fg_points(original_fg_frame):
                    if not original_fg_frame is None:
                        return original_fg_frame, [], False, "Selected point cleared"
                    return None, [], False, "Please select foreground element first"

                btn_clear_fg_points.click(
                    fn=clear_fg_points,
                    inputs=[original_fg_frame],
                    outputs=[fg_first_frame, fg_points, is_drawing_fg, fg_status]
                )

                def process_fg_and_update_state(fg_video, fg_points, current_run_folder):
                    source, mask, element, new_run_folder, status = process_foreground_video(fg_video, fg_points, current_run_folder)
                    debug_info = f"Path of element video: {element}, existing in: {os.path.exists(element) if element else False}"
                    status_with_debug = f"{status}\n{debug_info}"
                    return element, element, source, mask, new_run_folder, status_with_debug

                btn_process_fg.click(
                    fn=process_fg_and_update_state,
                    inputs=[fg_video_input, fg_points, run_folder],
                    outputs=[fg_element_output, processed_fg_element, processed_fg_source, processed_fg_mask, run_folder, fg_status]
                )

        with gr.Column(scale=3):
            with gr.Group(elem_classes="step-group"):
                gr.Markdown("## <span class='step-number'>Step 2</span>: Specify Trajectory in Background")
                with gr.Row(visible=False) as bg_upload_row:
                    upload_bg_button = gr.UploadButton(
                        "Upload your background video",
                        file_types=[".mp4"],
                        variant="primary",
                        elem_classes=["custom-button", "bg-button"]
                    )
                bg_video_input = gr.Video(label="Background video")
                with gr.Row(elem_classes="button-spacing"):
                    btn_start_drawing = gr.Button(
                        "Begin specifying trajectory",
                        variant="primary",
                        elem_classes=["custom-button", "rounded-button"]
                    )
                    btn_clear_points = gr.Button(
                        "Clear selected points",
                        variant="primary",
                        elem_classes=["custom-button", "rounded-button"]
                    )
                bg_status = gr.Textbox(label="Trajectory designing status", interactive=False)
                bg_first_frame = gr.Image(
                    label="Please click to select trajectory keypoints",
                    interactive=True,
                    width=900,
                    height=360,
                    elem_classes="image-display"
                )
                with gr.Row():
                    btn_draw_trajectory = gr.Button(
                        "Complete trajectory drawing",
                        variant="primary",
                        elem_classes=["custom-button", "rounded-button"]
                    )
                trajectory_vis_output = gr.Image(
                    label="Trajectory visualization",
                    interactive=False,
                    width=900,
                    height=360,
                    elem_classes="image-display"
                )

                def update_bg_video_and_frame(bg_video, original_bg_frame):
                    if not bg_video:
                        return None, None, "Please provide background video", [], None
                    adjusted_video, status = auto_adjust_background_video(bg_video)
                    frame = get_first_frame(adjusted_video)
                    frame_rgb = cv2.resize(frame, (720, 480))[:, :, ::-1] if frame is not None else None
                    return adjusted_video, frame_rgb, status, [], frame_rgb

                bg_video_input.change(
                    fn=update_bg_video_and_frame,
                    inputs=[bg_video_input, original_bg_frame],
                    outputs=[processed_bg_video, bg_first_frame, bg_status, trajectory_points, original_bg_frame]
                )

                select_bg_gallery.click(
                    fn=lambda idx: pick_video_by_index(idx, sample_videos),
                    inputs=selected_idx,
                    outputs=bg_video_input,
                )

                def process_bg_upload(uploaded_file):
                    return uploaded_file.name if uploaded_file else None

                upload_bg_button.upload(
                    fn=process_bg_upload,
                    inputs=[upload_bg_button],
                    outputs=[bg_video_input]
                )

                def start_drawing():
                    return True, "Begin specifying trajectory"

                btn_start_drawing.click(
                    fn=start_drawing,
                    inputs=[],
                    outputs=[is_drawing, bg_status]
                )

                def add_trajectory_point(evt: gr.SelectData, original_bg_frame, current_points, is_drawing):
                    if not is_drawing:
                        return gr.update(), current_points, "Please click the 'Begin specifying trajectory' button first"
                    if original_bg_frame is None:
                        return gr.update(), current_points, "Please select a background video first"

                    x, y = evt.index
                    x = min(max(0, int(x)), 720)
                    y = min(max(0, int(y)), 480)
                    new_points = current_points + [(x, y)]

                    display_frame = original_bg_frame.copy()

                    if len(new_points) > 1:
                        for i in range(1, len(new_points)):
                            cv2.line(display_frame, new_points[i - 1], new_points[i], (0, 0, 255), 3)

                    for point in new_points:
                        cv2.circle(display_frame, point, 8, (0, 255, 0), -1)

                    return display_frame, new_points, f"Added the point ({x}, {y}), a total of {len(new_points)} points"

                bg_first_frame.select(
                    fn=add_trajectory_point,
                    inputs=[original_bg_frame, trajectory_points, is_drawing],
                    outputs=[bg_first_frame, trajectory_points, bg_status]
                )

                def clear_points(original_bg_frame):
                    if not original_bg_frame is None:
                        return original_bg_frame, [], False, "Selected point cleared"
                    return None, [], False, "Please select background video first"

                btn_clear_points.click(
                    fn=clear_points,
                    inputs=[original_bg_frame],
                    outputs=[bg_first_frame, trajectory_points, is_drawing, bg_status]
                )

                def process_trajectory(bg_video, points):
                    if not bg_video:
                        return None, None, None, "Please provide background video"
                    if not points:
                        return None, None, None, "Please select at least one key point"
                    temp_trajectory = tempfile.NamedTemporaryFile(suffix=".txt", delete=False).name
                    points_json = json.dumps(points)
                    vis_path, trajectory_path, status = draw_and_save_trajectory(bg_video, points_json, temp_trajectory)
                    return vis_path, trajectory_path, vis_path, status

                btn_draw_trajectory.click(
                    fn=process_trajectory,
                    inputs=[processed_bg_video, trajectory_points],
                    outputs=[trajectory_vis_output, trajectory_file, trajectory_vis_output, bg_status]
                )

    def update_interface(use):
        if use == "Yes":
            return (
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False)
            )
        else:
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True)
            )

    use_samples.change(
        fn=update_interface,
        inputs=[use_samples],
        outputs=[samples_row, select_fg_gallery, upload_fg_button, select_bg_gallery, upload_bg_button, fg_upload_row, bg_upload_row]
    )

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Group(elem_classes="step-group"):
                gr.Markdown("## <span class='step-number'>Step 3</span>: Video Compositing")
                compose_status = gr.Textbox(label="Video compositing status", interactive=False)
                gr.Markdown(
                    "If the trajectory is the midpoint of desired mask, click the left; if the trajectory is the low point of desired mask, click the right",
                    elem_classes="label-text"
                )
                with gr.Row(elem_classes="button-spacing"):
                    btn_generate_mask = gr.Button(
                        "Generate middle mask for compositing",
                        variant="primary",
                        elem_classes=["custom-button", "rounded-button"]
                    )
                    btn_generate_upper_mask = gr.Button(
                        "Generate upper mask for compositing",
                        variant="primary",
                        elem_classes=["custom-button", "rounded-button"]
                    )
                with gr.Row():
                    rescale1 = gr.Number(
                        label="Rescale 1.",
                        value=0.4,
                        minimum=0.1,
                        maximum=5.0,
                    )
                    rescale2 = gr.Number(
                        label="Rescale 2 (optional)",
                        value=0.4,
                        minimum=0.1,
                        maximum=5.0,
                    )
                    rescale3 = gr.Number(
                        label="Rescale 3 (optional)",
                        value=0.4,
                        minimum=0.1,
                        maximum=5.0,
                    )
                    rescale4 = gr.Number(
                        label="Rescale 4 (optional)",
                        value=0.4,
                        minimum=0.1,
                        maximum=5.0,
                    )
                    rescale5 = gr.Number(
                        label="Rescale 5 (optional)",
                        value=0.4,
                        minimum=0.1,
                        maximum=5.0,
                    )
                with gr.Row():
                    final_mask_output = gr.Video(
                        label="Mask for compositing",
                        height=300,
                        elem_classes=["step5-video"],
                        format="mp4"
                    )
                with gr.Row():
                    inference_steps_input = gr.Number(
                        label="Inference steps number. Bigger value brings better performance but costs more time.",
                        value=10,
                        minimum=1,
                        maximum=100,
                        step=1,
                        scale=1
                    )
                with gr.Row(elem_classes="button-spacing"):
                    btn_compose_video = gr.Button(
                        "Generate composited video",
                        variant="primary",
                        elem_classes=["custom-button", "rounded-button"]
                    )
                with gr.Row():
                    composed_video_output = gr.Video(
                        label="Composited video",
                        height=300,
                        elem_classes=["step5-video"],
                        autoplay=True,
                        format="mp4"
                    )

                def generate_final_mask(fg_element, bg_video, trajectory_path, rescale_list, current_run_folder, alignment="center"):
                    scales = [r for r in rescale_list if r is not None and r != 0]
                    if not scales:
                        scales = [0.4]
                    if not fg_element or not bg_video or not trajectory_path:
                        return None, None, "Please provide complete foreground element video, background video, and specified trajectory"
                    temp_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
                    mask_video, status = generate_mask_video_with_trajectory(fg_element, bg_video, temp_output,
                                                                           trajectory_path, scales, alignment=alignment, run_folder=current_run_folder)
                    return mask_video, mask_video, status

                btn_generate_mask.click(
                    fn=lambda fg, bg, traj, r1, r2, r3, r4, r5, rf: generate_final_mask(fg, bg, traj, [r1, r2, r3, r4, r5], rf, "center"),
                    inputs=[processed_fg_element, processed_bg_video, trajectory_file, rescale1, rescale2, rescale3, rescale4, rescale5, run_folder],
                    outputs=[final_mask_output, final_mask_output_state, compose_status]
                )

                btn_generate_upper_mask.click(
                    fn=lambda fg, bg, traj, r1, r2, r3, r4, r5, rf: generate_final_mask(fg, bg, traj, [r1, r2, r3, r4, r5], rf, "bottom"),
                    inputs=[processed_fg_element, processed_bg_video, trajectory_file, rescale1, rescale2, rescale3, rescale4, rescale5, run_folder],
                    outputs=[final_mask_output, final_mask_output_state, compose_status]
                )

                def compose_video(fg_element, bg_video, mask_video, inference_steps, current_run_folder):
                    if not fg_element or not bg_video or not mask_video:
                        return None, "Please ensure that foreground element video, background video and final mask video are provided"
                    temp_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
                    try:
                        output_path, status = generate_video(
                            output_path=temp_output,
                            video_path=bg_video,
                            mask_path=mask_video,
                            fg_video_path=fg_element,
                            num_inference_steps=int(inference_steps),
                            guidance_scale=6.0,
                            num_videos_per_prompt=1,
                            generate_type="i2v_inpainting",
                            seed=42,
                            inpainting_frames=49,
                            mask_background=False,
                            add_first=False,
                            first_frame_gt=False,
                            replace_gt=False,
                            mask_add=True,
                            down_sample_fps=8,
                            overlap_frames=0,
                            prev_clip_weight=0.0,
                            start_frame=0,
                            end_frame=49,
                            img_inpainting_model=None,
                            llm_model=None,
                            long_video=False,
                            dilate_size=-1,
                            id_adapter_resample_learnable_path=None,
                            run_folder=current_run_folder,
                        )
                        return output_path, status
                    except Exception as e:
                        return None, f"Video compositing failed: {str(e)}"
                    finally:
                        torch.cuda.empty_cache()

                btn_compose_video.click(
                    fn=compose_video,
                    inputs=[processed_fg_element, processed_bg_video, final_mask_output_state, inference_steps_input, run_folder],
                    outputs=[composed_video_output, compose_status]
                )

        with gr.Column(scale=3):
            with gr.Group(elem_classes="step-group"):
                gr.Markdown("### <span class='step-number'>(Optional)</span> Extended Application: Video Inpainting and Removal")
                inpaint_status = gr.Textbox(label="Video element removal status", interactive=False)
                with gr.Row(elem_classes="button-spacing"):
                    btn_generate_mask_removal = gr.Button(
                        "Generate mask for removal",
                        variant="primary",
                        elem_classes=["custom-button", "rounded-button"]
                    )
                with gr.Row():
                    removal_mask_output = gr.Video(
                        label="Mask for removal",
                        height=300,
                        elem_classes=["step5-video"],
                        format="mp4"
                    )
                with gr.Row():
                    inference_steps_removal_input = gr.Number(
                        label="Inference steps number. Bigger value brings better performance but costs more time.",
                        value=10,
                        minimum=1,
                        maximum=100,
                        step=1,
                        scale=1
                    )
                with gr.Row(elem_classes="button-spacing"):
                    btn_generate_inpainted = gr.Button(
                        "Generate removed video",
                        variant="primary",
                        elem_classes=["custom-button", "rounded-button"]
                    )
                with gr.Row():
                    inpainted_video_output = gr.Video(
                        label="Removed video",
                        height=300,
                        elem_classes=["step5-video"],
                        autoplay=True,
                        format="mp4"
                    )

                def generate_mask_for_removal(processed_fg_mask):
                    if not processed_fg_mask:
                        return None, None, "Please segment foreground element first to generate a mask"
                    return processed_fg_mask, processed_fg_mask, "Mask for removal generated successfully"

                btn_generate_mask_removal.click(
                    fn=generate_mask_for_removal,
                    inputs=[processed_fg_mask],
                    outputs=[removal_mask_output, final_mask_output_state, inpaint_status]
                )

                def compose_inpainted_video(video, mask_video, inference_steps):
                    fg_element = "../assets/fg/white_video.mp4"
                    if not video or not mask_video:
                        return None, "Please ensure that video and mask are provided"
                    if not os.path.exists(video):
                        return None, f"Video path {video} does not exist"
                    if not os.path.exists(mask_video):
                        return None, f"Mask video path {mask_video} does not exist"
                    if not os.path.exists(fg_element):
                        return None, f"Foreground element path {fg_element} does not exist"
                    temp_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
                    try:
                        output_path, status = generate_video(
                            output_path=temp_output,
                            video_path=video,
                            mask_path=mask_video,
                            fg_video_path=fg_element,
                            num_inference_steps=int(inference_steps),
                            guidance_scale=6.0,
                            num_videos_per_prompt=1,
                            generate_type="i2v_inpainting",
                            seed=42,
                            inpainting_frames=49,
                            mask_background=False,
                            add_first=False,
                            first_frame_gt=False,
                            replace_gt=False,
                            mask_add=True,
                            down_sample_fps=8,
                            overlap_frames=0,
                            prev_clip_weight=0.0,
                            start_frame=0,
                            end_frame=49,
                            img_inpainting_model=None,
                            llm_model=None,
                            long_video=False,
                            dilate_size=-1,
                            id_adapter_resample_learnable_path=None,
                        )
                        return output_path, status
                    except Exception as e:
                        return None, f"Video inpainting failed: {str(e)}"
                    finally:
                        torch.cuda.empty_cache()

                btn_generate_inpainted.click(
                    fn=compose_inpainted_video,
                    inputs=[processed_fg_source, final_mask_output_state, inference_steps_removal_input],
                    outputs=[inpainted_video_output, inpaint_status]
                )

    # =============================================================================
    # 3D MULTI-VIEW COMPOSITING SECTION
    # =============================================================================

    with gr.Group(elem_classes="step-group"):
        gr.Markdown(
            """
            ## 3D Multi-View Compositing
            ### Composite foreground elements across 4 camera views using a single 3D trajectory

            <span class='quick-guide'>How it works:</span>
            1. Upload **4 background videos** (one for each camera angle: 0°, 90°, 180°, 270°)
            2. Upload **4 foreground videos** (one for each camera angle)
            3. Define a **3D trajectory** using world coordinates (x, y, z)
            4. The system will project the 3D trajectory to 2D for each view and composite all videos

            **To draw 3D trajectory interactively**, run in terminal: `python infer/draw_trajectory_3d.py`
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 2D Trajectory Input")
                gr.Markdown("*Object depth is estimated using Depth Anything V2 + background metric depth alignment*")
                with gr.Tabs():
                    with gr.TabItem("Draw Trajectory"):
                        gr.Markdown("**Draw on Background 0° (Front view). Click to add 2D points.**")
                        gr.Markdown("Object is composited, Depth Anything estimates depth, then aligned to 4DGS metric depth.")
                        traj_draw_points = gr.State([])  # List of (x, y) tuples - 2D only!
                        traj_draw_original_frame = gr.State(None)  # Store original bg frame
                        traj_draw_canvas = gr.Image(label="Click to add trajectory points (2D)", height=350, interactive=True)
                        with gr.Row():
                            btn_traj_draw_start = gr.Button("Load BG & Start", variant="primary", size="sm")
                            btn_traj_draw_undo = gr.Button("Undo Last", variant="secondary", size="sm")
                            btn_traj_draw_clear = gr.Button("Clear All", variant="secondary", size="sm")
                        traj_draw_status = gr.Textbox(label="Drawing Status", interactive=False, lines=1, show_label=False)

            with gr.Column(scale=1):
                gr.Markdown("### Camera Parameters")
                camera_distance = gr.Number(label="Camera Distance (R)", value=5.0, minimum=1.0, maximum=50.0)
                camera_height = gr.Number(label="Camera Height", value=0.0, minimum=-20.0, maximum=20.0)
                camera_fov = gr.Number(label="FOV (degrees)", value=60.0, minimum=30.0, maximum=120.0)
                mv_rescale = gr.Number(label="Rescale Factor", value=0.4, minimum=0.1, maximum=5.0)
                mv_inference_steps = gr.Number(label="Inference Steps", value=10, minimum=1, maximum=100, step=1)

            with gr.Column(scale=2):
                gr.Markdown("### 3D Trajectory Preview")
                trajectory_3d_plot = gr.Image(label="3D View with Camera Positions", height=350, interactive=False)

        gr.Markdown("### Background Videos (4 camera angles)")
        with gr.Row():
            mv_bg_0 = gr.Video(label="Background 0° (Front)", height=200)
            mv_bg_90 = gr.Video(label="Background 90° (Right)", height=200)
            mv_bg_180 = gr.Video(label="Background 180° (Back)", height=200)
            mv_bg_270 = gr.Video(label="Background 270° (Left)", height=200)

        gr.Markdown("### Depth Video (from 4DGS)")
        gr.Markdown("*Upload depth for View 0° only. Used to lift 2D trajectory to 3D.*")
        mv_depth_0 = gr.File(label="Depth 0° (.npy) - Shape: (num_frames, H, W), values in meters", file_types=[".npy"])

        gr.Markdown("### Foreground Videos (4 camera angles)")
        with gr.Row():
            mv_fg_0 = gr.Video(label="Foreground 0° (Front)", height=200)
            mv_fg_90 = gr.Video(label="Foreground 90° (Right)", height=200)
            mv_fg_180 = gr.Video(label="Foreground 180° (Back)", height=200)
            mv_fg_270 = gr.Video(label="Foreground 270° (Left)", height=200)

        gr.Markdown("### Foreground Object Selection")
        mv_fg_points_0 = gr.State([])
        mv_fg_points_90 = gr.State([])
        mv_fg_points_180 = gr.State([])
        mv_fg_points_270 = gr.State([])
        mv_fg_drawing = gr.State({0: False, 90: False, 180: False, 270: False})
        mv_fg_point_mode = gr.State({0: "positive", 90: "positive", 180: "positive", 270: "positive"})
        mv_fg_original_frames = gr.State({0: None, 90: None, 180: None, 270: None})
        mv_fg_segmented = gr.State({0: None, 90: None, 180: None, 270: None})

        with gr.Row():
            with gr.Column():
                gr.Markdown("**View 0° (Front)** - Click to select object")
                mv_fg_frame_0 = gr.Image(label=None, show_label=False, height=180, interactive=True)
                with gr.Row():
                    btn_mv_fg_start_0 = gr.Button("Start", variant="secondary", size="sm")
                    btn_mv_fg_clear_0 = gr.Button("Clear", variant="secondary", size="sm")
                with gr.Row():
                    btn_mv_pos_0 = gr.Button("+ Include", variant="primary", size="sm")
                    btn_mv_neg_0 = gr.Button("- Exclude", variant="secondary", size="sm")
                btn_mv_segment_0 = gr.Button("Segment (0°)", variant="primary", size="sm")
                mv_fg_status_0 = gr.Textbox(label="Status", interactive=False, lines=1, show_label=False)
                mv_fg_element_0 = gr.Video(label="Segmented Output (0°)", height=150)

            with gr.Column():
                gr.Markdown("**View 90° (Right)** - Click to select object")
                mv_fg_frame_90 = gr.Image(label=None, show_label=False, height=180, interactive=True)
                with gr.Row():
                    btn_mv_fg_start_90 = gr.Button("Start", variant="secondary", size="sm")
                    btn_mv_fg_clear_90 = gr.Button("Clear", variant="secondary", size="sm")
                with gr.Row():
                    btn_mv_pos_90 = gr.Button("+ Include", variant="primary", size="sm")
                    btn_mv_neg_90 = gr.Button("- Exclude", variant="secondary", size="sm")
                btn_mv_segment_90 = gr.Button("Segment (90°)", variant="primary", size="sm")
                mv_fg_status_90 = gr.Textbox(label="Status", interactive=False, lines=1, show_label=False)
                mv_fg_element_90 = gr.Video(label="Segmented Output (90°)", height=150)

            with gr.Column():
                gr.Markdown("**View 180° (Back)** - Click to select object")
                mv_fg_frame_180 = gr.Image(label=None, show_label=False, height=180, interactive=True)
                with gr.Row():
                    btn_mv_fg_start_180 = gr.Button("Start", variant="secondary", size="sm")
                    btn_mv_fg_clear_180 = gr.Button("Clear", variant="secondary", size="sm")
                with gr.Row():
                    btn_mv_pos_180 = gr.Button("+ Include", variant="primary", size="sm")
                    btn_mv_neg_180 = gr.Button("- Exclude", variant="secondary", size="sm")
                btn_mv_segment_180 = gr.Button("Segment (180°)", variant="primary", size="sm")
                mv_fg_status_180 = gr.Textbox(label="Status", interactive=False, lines=1, show_label=False)
                mv_fg_element_180 = gr.Video(label="Segmented Output (180°)", height=150)

            with gr.Column():
                gr.Markdown("**View 270° (Left)** - Click to select object")
                mv_fg_frame_270 = gr.Image(label=None, show_label=False, height=180, interactive=True)
                with gr.Row():
                    btn_mv_fg_start_270 = gr.Button("Start", variant="secondary", size="sm")
                    btn_mv_fg_clear_270 = gr.Button("Clear", variant="secondary", size="sm")
                with gr.Row():
                    btn_mv_pos_270 = gr.Button("+ Include", variant="primary", size="sm")
                    btn_mv_neg_270 = gr.Button("- Exclude", variant="secondary", size="sm")
                btn_mv_segment_270 = gr.Button("Segment (270°)", variant="primary", size="sm")
                mv_fg_status_270 = gr.Textbox(label="Status", interactive=False, lines=1, show_label=False)
                mv_fg_element_270 = gr.Video(label="Segmented Output (270°)", height=150)

        btn_process_multiview = gr.Button("Process All 4 Views", variant="primary", elem_classes=["custom-button", "rounded-button"])
        mv_status = gr.Textbox(label="Multi-view Processing Status", interactive=False, lines=4)

        gr.Markdown("### Output Videos")
        with gr.Row():
            mv_output_0 = gr.Video(label="Output 0°", height=250, autoplay=True)
            mv_output_90 = gr.Video(label="Output 90°", height=250, autoplay=True)
            mv_output_180 = gr.Video(label="Output 180°", height=250, autoplay=True)
            mv_output_270 = gr.Video(label="Output 270°", height=250, autoplay=True)

        # Trajectory drawing handlers
        traj_draw_is_drawing = gr.State(False)

        def traj_draw_start(bg_video):
            """Initialize the drawing canvas with background video frame"""
            if not bg_video:
                return None, None, [], False, "Please upload Background 0° video first"

            # Get first frame from background video
            frame = get_first_frame(bg_video)
            if frame is None:
                return None, None, [], False, "Could not read background video"

            frame_rgb = cv2.resize(frame, (720, 480))
            frame_rgb = frame_rgb[:, :, ::-1]  # BGR to RGB

            # Draw empty trajectory overlay
            canvas = draw_trajectory_2d_on_frame(frame_rgb, [])

            return canvas, frame_rgb, [], True, "Click on the video to add 2D trajectory points. Depth will be inferred during processing."

        def traj_draw_add_point(evt: gr.SelectData, original_frame, points, is_drawing):
            """Add a point when canvas is clicked - store only 2D pixel coordinates"""
            if not is_drawing or original_frame is None:
                return original_frame, points, "Click 'Load BG & Start' first"

            # Get click position in image coordinates
            px, py = evt.index
            h, w = 480, 720

            # Clamp to image bounds
            px = max(0, min(w - 1, px))
            py = max(0, min(h - 1, py))

            # Store 2D point
            new_points = points + [(px, py)]
            canvas = draw_trajectory_2d_on_frame(original_frame, new_points)

            return canvas, new_points, f"Point {len(new_points)}: ({px}, {py})"

        def traj_draw_undo(original_frame, points):
            """Remove the last point"""
            if original_frame is None:
                return None, [], "No canvas loaded"
            if not points:
                return draw_trajectory_2d_on_frame(original_frame, []), [], "No points to undo"
            new_points = points[:-1]
            canvas = draw_trajectory_2d_on_frame(original_frame, new_points)
            return canvas, new_points, f"Removed last point. {len(new_points)} points remaining."

        def traj_draw_clear(original_frame):
            """Clear all points"""
            if original_frame is None:
                return None, None, [], False, "No canvas loaded"
            canvas = draw_trajectory_2d_on_frame(original_frame, [])
            return canvas, original_frame, [], False, "Cleared. Click 'Load BG & Start' to begin again."

        btn_traj_draw_start.click(
            fn=traj_draw_start,
            inputs=[mv_bg_0],
            outputs=[traj_draw_canvas, traj_draw_original_frame, traj_draw_points, traj_draw_is_drawing, traj_draw_status]
        )

        traj_draw_canvas.select(
            fn=traj_draw_add_point,
            inputs=[traj_draw_original_frame, traj_draw_points, traj_draw_is_drawing],
            outputs=[traj_draw_canvas, traj_draw_points, traj_draw_status]
        )

        btn_traj_draw_undo.click(
            fn=traj_draw_undo,
            inputs=[traj_draw_original_frame, traj_draw_points],
            outputs=[traj_draw_canvas, traj_draw_points, traj_draw_status]
        )

        btn_traj_draw_clear.click(
            fn=traj_draw_clear,
            inputs=[traj_draw_original_frame],
            outputs=[traj_draw_canvas, traj_draw_original_frame, traj_draw_points, traj_draw_is_drawing, traj_draw_status]
        )

        # Helper: get first frame and store original
        def get_first_frame_for_mv(video_path, original_frames, angle):
            if not video_path:
                return None, original_frames, []
            frame = get_first_frame(video_path)
            frame_rgb = cv2.resize(frame, (720, 480))[:, :, ::-1] if frame is not None else None
            new_originals = original_frames.copy() if original_frames else {0: None, 90: None, 180: None, 270: None}
            new_originals[angle] = frame_rgb
            return frame_rgb, new_originals, []

        mv_fg_0.change(fn=lambda v, o: get_first_frame_for_mv(v, o, 0), inputs=[mv_fg_0, mv_fg_original_frames], outputs=[mv_fg_frame_0, mv_fg_original_frames, mv_fg_points_0])
        mv_fg_90.change(fn=lambda v, o: get_first_frame_for_mv(v, o, 90), inputs=[mv_fg_90, mv_fg_original_frames], outputs=[mv_fg_frame_90, mv_fg_original_frames, mv_fg_points_90])
        mv_fg_180.change(fn=lambda v, o: get_first_frame_for_mv(v, o, 180), inputs=[mv_fg_180, mv_fg_original_frames], outputs=[mv_fg_frame_180, mv_fg_original_frames, mv_fg_points_180])
        mv_fg_270.change(fn=lambda v, o: get_first_frame_for_mv(v, o, 270), inputs=[mv_fg_270, mv_fg_original_frames], outputs=[mv_fg_frame_270, mv_fg_original_frames, mv_fg_points_270])

        # Start drawing
        def mv_start_drawing(drawing_state, angle):
            new_state = drawing_state.copy() if drawing_state else {0: False, 90: False, 180: False, 270: False}
            new_state[angle] = True
            return new_state, f"Selection started. Green=Include, Red=Exclude"

        btn_mv_fg_start_0.click(fn=lambda s: mv_start_drawing(s, 0), inputs=[mv_fg_drawing], outputs=[mv_fg_drawing, mv_fg_status_0])
        btn_mv_fg_start_90.click(fn=lambda s: mv_start_drawing(s, 90), inputs=[mv_fg_drawing], outputs=[mv_fg_drawing, mv_fg_status_90])
        btn_mv_fg_start_180.click(fn=lambda s: mv_start_drawing(s, 180), inputs=[mv_fg_drawing], outputs=[mv_fg_drawing, mv_fg_status_180])
        btn_mv_fg_start_270.click(fn=lambda s: mv_start_drawing(s, 270), inputs=[mv_fg_drawing], outputs=[mv_fg_drawing, mv_fg_status_270])

        # Clear points
        def mv_clear_points(original_frames, drawing_state, angle):
            frame = original_frames.get(angle) if original_frames else None
            new_drawing = drawing_state.copy() if drawing_state else {0: False, 90: False, 180: False, 270: False}
            new_drawing[angle] = False
            return frame, [], new_drawing, "Points cleared"

        btn_mv_fg_clear_0.click(fn=lambda o, d: mv_clear_points(o, d, 0), inputs=[mv_fg_original_frames, mv_fg_drawing], outputs=[mv_fg_frame_0, mv_fg_points_0, mv_fg_drawing, mv_fg_status_0])
        btn_mv_fg_clear_90.click(fn=lambda o, d: mv_clear_points(o, d, 90), inputs=[mv_fg_original_frames, mv_fg_drawing], outputs=[mv_fg_frame_90, mv_fg_points_90, mv_fg_drawing, mv_fg_status_90])
        btn_mv_fg_clear_180.click(fn=lambda o, d: mv_clear_points(o, d, 180), inputs=[mv_fg_original_frames, mv_fg_drawing], outputs=[mv_fg_frame_180, mv_fg_points_180, mv_fg_drawing, mv_fg_status_180])
        btn_mv_fg_clear_270.click(fn=lambda o, d: mv_clear_points(o, d, 270), inputs=[mv_fg_original_frames, mv_fg_drawing], outputs=[mv_fg_frame_270, mv_fg_points_270, mv_fg_drawing, mv_fg_status_270])

        # Set point mode (positive/negative)
        def mv_set_mode(mode_state, angle, mode):
            new_state = mode_state.copy() if mode_state else {0: "positive", 90: "positive", 180: "positive", 270: "positive"}
            new_state[angle] = mode
            return new_state, f"Mode: {'Include' if mode == 'positive' else 'Exclude'}"

        btn_mv_pos_0.click(fn=lambda m: mv_set_mode(m, 0, "positive"), inputs=[mv_fg_point_mode], outputs=[mv_fg_point_mode, mv_fg_status_0])
        btn_mv_neg_0.click(fn=lambda m: mv_set_mode(m, 0, "negative"), inputs=[mv_fg_point_mode], outputs=[mv_fg_point_mode, mv_fg_status_0])
        btn_mv_pos_90.click(fn=lambda m: mv_set_mode(m, 90, "positive"), inputs=[mv_fg_point_mode], outputs=[mv_fg_point_mode, mv_fg_status_90])
        btn_mv_neg_90.click(fn=lambda m: mv_set_mode(m, 90, "negative"), inputs=[mv_fg_point_mode], outputs=[mv_fg_point_mode, mv_fg_status_90])
        btn_mv_pos_180.click(fn=lambda m: mv_set_mode(m, 180, "positive"), inputs=[mv_fg_point_mode], outputs=[mv_fg_point_mode, mv_fg_status_180])
        btn_mv_neg_180.click(fn=lambda m: mv_set_mode(m, 180, "negative"), inputs=[mv_fg_point_mode], outputs=[mv_fg_point_mode, mv_fg_status_180])
        btn_mv_pos_270.click(fn=lambda m: mv_set_mode(m, 270, "positive"), inputs=[mv_fg_point_mode], outputs=[mv_fg_point_mode, mv_fg_status_270])
        btn_mv_neg_270.click(fn=lambda m: mv_set_mode(m, 270, "negative"), inputs=[mv_fg_point_mode], outputs=[mv_fg_point_mode, mv_fg_status_270])

        # Add points with labels
        def mv_add_fg_point(evt: gr.SelectData, frame, points, drawing_state, mode_state, original_frames, angle):
            if drawing_state is None or not drawing_state.get(angle, False) or frame is None:
                return frame, points if points is not None else [], f"Click 'Start' first"
            x, y = evt.index
            x = min(max(0, int(x)), 720)
            y = min(max(0, int(y)), 480)

            mode = mode_state.get(angle, "positive") if mode_state else "positive"
            label = 1 if mode == "positive" else 0
            new_points = (points if points is not None else []) + [(x, y, label)]

            # Redraw on original frame
            orig_frame = original_frames.get(angle) if original_frames else frame
            display = orig_frame.copy() if orig_frame is not None else frame.copy()
            for pt in new_points:
                px, py, plabel = pt
                color = (0, 255, 0) if plabel == 1 else (255, 0, 0)
                cv2.circle(display, (px, py), 8, color, -1)

            pos_count = sum(1 for p in new_points if p[2] == 1)
            neg_count = sum(1 for p in new_points if p[2] == 0)
            return display, new_points, f"+{pos_count} -{neg_count} points"

        def make_mv_add_fg_point_handler(angle):
            def handler(evt: gr.SelectData, frame, points, drawing_state, mode_state, original_frames):
                return mv_add_fg_point(evt, frame, points, drawing_state, mode_state, original_frames, angle)
            return handler

        mv_fg_frame_0.select(fn=make_mv_add_fg_point_handler(0), inputs=[mv_fg_frame_0, mv_fg_points_0, mv_fg_drawing, mv_fg_point_mode, mv_fg_original_frames], outputs=[mv_fg_frame_0, mv_fg_points_0, mv_fg_status_0])
        mv_fg_frame_90.select(fn=make_mv_add_fg_point_handler(90), inputs=[mv_fg_frame_90, mv_fg_points_90, mv_fg_drawing, mv_fg_point_mode, mv_fg_original_frames], outputs=[mv_fg_frame_90, mv_fg_points_90, mv_fg_status_90])
        mv_fg_frame_180.select(fn=make_mv_add_fg_point_handler(180), inputs=[mv_fg_frame_180, mv_fg_points_180, mv_fg_drawing, mv_fg_point_mode, mv_fg_original_frames], outputs=[mv_fg_frame_180, mv_fg_points_180, mv_fg_status_180])
        mv_fg_frame_270.select(fn=make_mv_add_fg_point_handler(270), inputs=[mv_fg_frame_270, mv_fg_points_270, mv_fg_drawing, mv_fg_point_mode, mv_fg_original_frames], outputs=[mv_fg_frame_270, mv_fg_points_270, mv_fg_status_270])

        # Segment single view
        def mv_segment_single(fg_video, fg_points, segmented_state, angle):
            if not fg_video or not fg_points:
                return None, segmented_state, "Need video and points"
            try:
                source, mask, element, _, status = process_foreground_video(fg_video, fg_points)
                new_segmented = segmented_state.copy() if segmented_state else {0: None, 90: None, 180: None, 270: None}
                new_segmented[angle] = element
                return element, new_segmented, f"Segmented OK"
            except Exception as e:
                return None, segmented_state, f"Error: {str(e)}"

        btn_mv_segment_0.click(fn=lambda v, p, s: mv_segment_single(v, p, s, 0), inputs=[mv_fg_0, mv_fg_points_0, mv_fg_segmented], outputs=[mv_fg_element_0, mv_fg_segmented, mv_fg_status_0])
        btn_mv_segment_90.click(fn=lambda v, p, s: mv_segment_single(v, p, s, 90), inputs=[mv_fg_90, mv_fg_points_90, mv_fg_segmented], outputs=[mv_fg_element_90, mv_fg_segmented, mv_fg_status_90])
        btn_mv_segment_180.click(fn=lambda v, p, s: mv_segment_single(v, p, s, 180), inputs=[mv_fg_180, mv_fg_points_180, mv_fg_segmented], outputs=[mv_fg_element_180, mv_fg_segmented, mv_fg_status_180])
        btn_mv_segment_270.click(fn=lambda v, p, s: mv_segment_single(v, p, s, 270), inputs=[mv_fg_270, mv_fg_points_270, mv_fg_segmented], outputs=[mv_fg_element_270, mv_fg_segmented, mv_fg_status_270])

        def process_multiview_batch(traj_points_2d, bg_0, bg_90, bg_180, bg_270,
                                    depth_0,
                                    fg_0, fg_90, fg_180, fg_270,
                                    pts_0, pts_90, pts_180, pts_270, seg_0, seg_90, seg_180, seg_270,
                                    cam_dist, cam_h, cam_fov, rescale, steps):
            """Process multi-view compositing with depth completion.

            Pipeline:
            1. Draw 2D trajectory on View 0
            2. Sample depth from View 0 depth video
            3. Lift to 3D and project to other views
            """
            print("[DEBUG] ========== process_multiview_batch called ==========")
            print(f"[DEBUG] Inputs: depth_0={depth_0}, bg_0={bg_0 is not None}, seg_0={seg_0}")
            from infer.trajectory_3d import (Camera, CameraConfig, get_camera_configs,
                                             save_trajectory_2d, interpolate_3d_trajectory, Trajectory3D,
                                             project_trajectory_to_views)
            from infer.depth_utils import (load_depth_video, run_depth_anything,
                                          align_depth_to_metric, extract_object_depth_at_trajectory,
                                          composite_element_at_point)

            results = {0: None, 90: None, 180: None, 270: None}
            status_lines = []

            # Validate 2D trajectory
            if not traj_points_2d or len(traj_points_2d) == 0:
                return None, None, None, None, "Error: Please draw a 2D trajectory first (click on Background 0°)"

            status_lines.append(f"2D Trajectory: {len(traj_points_2d)} keypoints")

            # Interpolate 2D trajectory to 49 frames if needed
            if len(traj_points_2d) < 49:
                # Linear interpolation for 2D points
                import numpy as np
                points = np.array(traj_points_2d)
                if len(points) == 1:
                    traj_2d_interpolated = [tuple(points[0])] * 49
                else:
                    t_orig = np.linspace(0, 1, len(points))
                    t_new = np.linspace(0, 1, 49)
                    x_interp = np.interp(t_new, t_orig, points[:, 0])
                    y_interp = np.interp(t_new, t_orig, points[:, 1])
                    traj_2d_interpolated = [(int(x), int(y)) for x, y in zip(x_interp, y_interp)]
                status_lines.append("Interpolated to 49 frames")
            else:
                traj_2d_interpolated = [(int(p[0]), int(p[1])) for p in traj_points_2d[:49]]

            print(f"[DEBUG] Original 2D trajectory (View 0) - First 3 points: {traj_2d_interpolated[:3]}")
            print(f"[DEBUG] Original 2D trajectory (View 0) - Last point: {traj_2d_interpolated[-1]}")

            # Load depth video for view 0 (required for depth completion)
            depth_video_0 = None
            if depth_0 is not None:
                print(f"[DEBUG] Attempting to load depth video from: {depth_0.name}")
                try:
                    depth_video_0 = load_depth_video(depth_0.name)
                    status_lines.append(f"Loaded depth video 0°: shape {depth_video_0.shape}")
                    print(f"[DEBUG] Depth video loaded successfully: shape {depth_video_0.shape}")
                except Exception as e:
                    status_lines.append(f"Warning: Could not load depth 0°: {e}")
                    print(f"[DEBUG] ERROR loading depth video: {e}")
            else:
                print("[DEBUG] No depth_0 provided")

            # Create camera configurations
            camera_configs = get_camera_configs(distance=cam_dist, height=cam_h, fov=cam_fov)
            print(f"[DEBUG] Camera config: distance={cam_dist}, height={cam_h}, fov={cam_fov}")
            cameras = [Camera(cfg, debug=True) for cfg in camera_configs]

            # Use pre-segmented videos if available
            segmented_vids = {0: seg_0, 90: seg_90, 180: seg_180, 270: seg_270}

            # If we have depth video and can do depth completion, lift to 3D
            print(f"[DEBUG] Depth completion check: depth_video_0={depth_video_0 is not None}")
            if depth_video_0 is not None:
                # Get element video to run depth estimation on
                element_0_path = segmented_vids.get(0)
                use_depth_anything = element_0_path is not None and bg_0 is not None
                print(f"[DEBUG] Depth Anything check: element_0_path={element_0_path}, bg_0={bg_0 is not None}, use_depth_anything={use_depth_anything}")

                if use_depth_anything:
                    status_lines.append("Using Depth Anything for intelligent depth completion...")
                    print("[DEBUG] Will use Depth Anything for depth completion")
                else:
                    status_lines.append("No pre-segmented element for view 0°, using background depth directly")
                    print("[DEBUG] Skipping Depth Anything - missing element or background")

                try:
                    # Load video frames if using Depth Anything
                    bg_frames = None
                    element_frames = None

                    if use_depth_anything:
                        # Load background video frames for view 0
                        print(f"[DEBUG] Loading background video from: {bg_0}")
                        bg_cap = cv2.VideoCapture(bg_0)
                        if not bg_cap.isOpened():
                            print(f"[DEBUG] ERROR: Could not open background video: {bg_0}")
                        bg_frames = []
                        while True:
                            ret, frame = bg_cap.read()
                            if not ret:
                                break
                            bg_frames.append(frame)
                        bg_cap.release()
                        print(f"[DEBUG] Loaded {len(bg_frames)} background frames")
                        status_lines.append(f"Loaded {len(bg_frames)} background frames for view 0°")

                        # Load element video frames for view 0
                        print(f"[DEBUG] Loading element video from: {element_0_path}")
                        element_cap = cv2.VideoCapture(element_0_path)
                        if not element_cap.isOpened():
                            print(f"[DEBUG] ERROR: Could not open element video: {element_0_path}")
                        element_frames = []
                        while True:
                            ret, frame = element_cap.read()
                            if not ret:
                                break
                            element_frames.append(frame)
                        element_cap.release()
                        print(f"[DEBUG] Loaded {len(element_frames)} element frames")
                        status_lines.append(f"Loaded {len(element_frames)} element frames for view 0°")

                    # Lift 2D trajectory to 3D using depth
                    print(f"[DEBUG] Starting 2D->3D lifting for {len(traj_2d_interpolated)} trajectory points")
                    print(f"[DEBUG] Depth video shape: {depth_video_0.shape}")
                    points_3d = []
                    for frame_idx, (px, py) in enumerate(traj_2d_interpolated):
                        # Get background depth at this frame
                        if frame_idx < len(depth_video_0):
                            bg_depth_frame = depth_video_0[frame_idx]
                        else:
                            bg_depth_frame = depth_video_0[-1]  # Use last frame if trajectory is longer

                        depth_val = None

                        # Use Depth Anything pipeline if we have element frames
                        if use_depth_anything and bg_frames and element_frames:
                            if frame_idx == 0:
                                print(f"[DEBUG] Frame 0: Using Depth Anything pipeline")
                                print(f"[DEBUG] Frame 0: bg_frame shape={bg_frames[0].shape}, element_frame shape={element_frames[0].shape}")
                            try:
                                # Get current background and element frames
                                bg_frame_idx = min(frame_idx, len(bg_frames) - 1)
                                elem_frame_idx = min(frame_idx, len(element_frames) - 1)

                                bg_frame = bg_frames[bg_frame_idx]
                                element_frame = element_frames[elem_frame_idx]

                                # Resize element to match rescale factor if needed
                                if rescale != 1.0:
                                    new_h = int(element_frame.shape[0] * rescale)
                                    new_w = int(element_frame.shape[1] * rescale)
                                    element_frame = cv2.resize(element_frame, (new_w, new_h))
                                    if frame_idx == 0:
                                        print(f"[DEBUG] Frame 0: Rescaled element to {new_w}x{new_h}")

                                # Composite element onto background at trajectory point
                                composite, object_mask = composite_element_at_point(
                                    bg_frame, element_frame, position=(px, py), scale=1.0
                                )
                                if frame_idx == 0:
                                    print(f"[DEBUG] Frame 0: Composite shape={composite.shape}, mask sum={object_mask.sum()}")

                                # Only run Depth Anything if we have a valid object mask
                                if object_mask.sum() > 100:
                                    if frame_idx == 0:
                                        print(f"[DEBUG] Frame 0: Running Depth Anything...")
                                    # Run Depth Anything on composite
                                    # Convert BGR to RGB for Depth Anything
                                    composite_rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
                                    estimated_depth = run_depth_anything(composite_rgb)
                                    if frame_idx == 0:
                                        print(f"[DEBUG] Frame 0: Depth Anything output shape={estimated_depth.shape}, range=[{estimated_depth.min():.3f}, {estimated_depth.max():.3f}]")

                                    # Resize bg_depth_frame to match estimated_depth if needed
                                    if bg_depth_frame.shape != estimated_depth.shape:
                                        bg_depth_resized = cv2.resize(
                                            bg_depth_frame,
                                            (estimated_depth.shape[1], estimated_depth.shape[0]),
                                            interpolation=cv2.INTER_LINEAR
                                        )
                                    else:
                                        bg_depth_resized = bg_depth_frame

                                    # Resize object_mask to match
                                    if object_mask.shape != estimated_depth.shape:
                                        object_mask_resized = cv2.resize(
                                            object_mask.astype(np.float32),
                                            (estimated_depth.shape[1], estimated_depth.shape[0]),
                                            interpolation=cv2.INTER_NEAREST
                                        ).astype(np.uint8)
                                    else:
                                        object_mask_resized = object_mask

                                    # Align estimated depth to metric depth
                                    if frame_idx == 0:
                                        print(f"[DEBUG] Frame 0: Aligning depth to metric...")
                                    aligned_depth = align_depth_to_metric(
                                        estimated_depth, bg_depth_resized, object_mask_resized
                                    )
                                    if frame_idx == 0:
                                        print(f"[DEBUG] Frame 0: Aligned depth range=[{aligned_depth.min():.3f}, {aligned_depth.max():.3f}]")

                                    # Scale trajectory point to aligned depth resolution
                                    scale_x = aligned_depth.shape[1] / 720
                                    scale_y = aligned_depth.shape[0] / 480
                                    traj_pt_scaled = (int(px * scale_x), int(py * scale_y))

                                    # Extract object depth at trajectory point
                                    depth_val = extract_object_depth_at_trajectory(
                                        aligned_depth, object_mask_resized, traj_pt_scaled
                                    )
                                    if frame_idx == 0:
                                        print(f"[DEBUG] Frame 0: Extracted depth at ({px}, {py}) = {depth_val:.3f}m")

                                    if frame_idx == 0:
                                        status_lines.append(f"Depth Anything estimated object depth: {depth_val:.2f}m")
                                else:
                                    if frame_idx == 0:
                                        print(f"[DEBUG] Frame 0: Object mask too small ({object_mask.sum()}), skipping Depth Anything")
                            except Exception as da_error:
                                if frame_idx == 0:
                                    print(f"[DEBUG] Frame 0: Depth Anything ERROR: {da_error}")
                                    import traceback
                                    traceback.print_exc()
                                    status_lines.append(f"Depth Anything fallback: {str(da_error)[:50]}")
                                depth_val = None
                        else:
                            if frame_idx == 0:
                                print(f"[DEBUG] Frame 0: Skipping Depth Anything (use_depth_anything={use_depth_anything}, bg_frames={bg_frames is not None and len(bg_frames) if bg_frames else None}, element_frames={element_frames is not None and len(element_frames) if element_frames else None})")

                        # Fallback: sample depth from background depth directly
                        if depth_val is None:
                            if frame_idx == 0:
                                print(f"[DEBUG] Frame 0: Using fallback - sampling background depth directly")
                            h, w = bg_depth_frame.shape[:2]
                            scale_x = w / 720
                            scale_y = h / 480
                            depth_x = int(px * scale_x)
                            depth_y = int(py * scale_y)
                            depth_x = max(0, min(w-1, depth_x))
                            depth_y = max(0, min(h-1, depth_y))

                            # Get depth value (with small neighborhood for robustness)
                            radius = 3
                            y1 = max(0, depth_y - radius)
                            y2 = min(h, depth_y + radius + 1)
                            x1 = max(0, depth_x - radius)
                            x2 = min(w, depth_x + radius + 1)
                            neighborhood = bg_depth_frame[y1:y2, x1:x2]
                            valid_depths = neighborhood[neighborhood > 0.01]
                            if len(valid_depths) > 0:
                                depth_val = float(np.median(valid_depths))
                            else:
                                depth_val = 5.0  # Default depth

                        # Lift to 3D
                        try:
                            x_world, y_world, z_world = cameras[0].lift_2d_to_3d(px, py, depth_val)
                            points_3d.append((x_world, y_world, z_world))
                            if frame_idx < 3 or frame_idx == len(traj_2d_interpolated) - 1:
                                print(f"[DEBUG] Frame {frame_idx}: 2D=({px}, {py}), depth={depth_val:.3f}m -> 3D=({x_world:.3f}, {y_world:.3f}, {z_world:.3f})")
                        except ValueError as ve:
                            print(f"[DEBUG] Frame {frame_idx}: ValueError in lift_2d_to_3d: {ve}")
                            # Use previous point or default
                            if points_3d:
                                points_3d.append(points_3d[-1])
                            else:
                                points_3d.append((0.0, 0.0, 0.0))

                    status_lines.append(f"Lifted to 3D: {len(points_3d)} points")
                    print(f"[DEBUG] 3D trajectory sample (first 3 points): {points_3d[:3]}")
                    print(f"[DEBUG] 3D trajectory sample (last point): {points_3d[-1]}")

                    # Create 3D trajectory and project to all views
                    trajectory_3d = Trajectory3D(points=points_3d, num_frames=len(points_3d))
                    projected = project_trajectory_to_views(trajectory_3d, camera_configs)
                    status_lines.append("Projected to all 4 views")

                    # Debug: Show projected trajectories for each view
                    for angle in [0, 90, 180, 270]:
                        traj = projected[angle]
                        print(f"[DEBUG] View {angle}°: First 3 projected points: {traj[:3]}")
                        print(f"[DEBUG] View {angle}°: Last projected point: {traj[-1]}")

                except Exception as e:
                    import traceback
                    status_lines.append(f"Warning: Depth completion failed: {e}")
                    status_lines.append(traceback.format_exc()[:200])
                    # Fallback: use same 2D trajectory for all views
                    projected = {0: traj_2d_interpolated, 90: traj_2d_interpolated,
                                180: traj_2d_interpolated, 270: traj_2d_interpolated}
            else:
                # No depth video - use same 2D trajectory for all views (fallback)
                status_lines.append("No depth video provided - using 2D trajectory for view 0° only")
                projected = {0: traj_2d_interpolated, 90: traj_2d_interpolated,
                            180: traj_2d_interpolated, 270: traj_2d_interpolated}

            # Process each view
            for angle, bg_vid, fg_vid, fg_pts in [(0, bg_0, fg_0, pts_0), (90, bg_90, fg_90, pts_90),
                                                   (180, bg_180, fg_180, pts_180), (270, bg_270, fg_270, pts_270)]:
                if not bg_vid:
                    status_lines.append(f"Skipping {angle}° - missing background video")
                    continue

                # Check if we have pre-segmented video for this angle
                pre_segmented = segmented_vids.get(angle)

                if not pre_segmented and not fg_vid:
                    status_lines.append(f"Skipping {angle}° - missing foreground video")
                    continue

                try:
                    status_lines.append(f"Processing {angle}°...")
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    view_folder = os.path.join("./results/runs", f"{timestamp}_view{angle}")
                    ensure_directory_exists(view_folder)

                    # Use pre-segmented video if available, otherwise segment now
                    if pre_segmented:
                        element = pre_segmented
                        status_lines.append(f"  {angle}°: Using pre-segmented video")
                    else:
                        fg_pts_use = fg_pts if fg_pts and len(fg_pts) > 0 else [(360, 240, 1)]
                        source, mask, element, _, _ = process_foreground_video(fg_vid, fg_pts_use, view_folder)
                        if not element:
                            status_lines.append(f"  {angle}°: FG processing failed")
                            continue

                    adjusted_bg, _ = auto_adjust_background_video(bg_vid)
                    traj_path = os.path.join(view_folder, "trajectory_2d.txt")
                    save_trajectory_2d(projected[angle], traj_path)

                    mask_path = os.path.join(view_folder, "mask_video.mp4")
                    mask_video, _ = generate_mask_video_with_trajectory(element, adjusted_bg, mask_path, traj_path,
                                                                        scales=[rescale], alignment="center", run_folder=view_folder)
                    if not mask_video:
                        status_lines.append(f"  {angle}°: Mask failed")
                        continue

                    out_path = os.path.join(view_folder, "output.mp4")
                    output, _ = generate_video(output_path=out_path, video_path=adjusted_bg, mask_path=mask_video,
                                               fg_video_path=element, num_inference_steps=int(steps), guidance_scale=6.0,
                                               num_videos_per_prompt=1, generate_type="i2v_inpainting", seed=42,
                                               inpainting_frames=49, mask_background=False, add_first=False,
                                               first_frame_gt=False, replace_gt=False, mask_add=True, down_sample_fps=8,
                                               overlap_frames=0, prev_clip_weight=0.0, start_frame=0, end_frame=49,
                                               img_inpainting_model=None, llm_model=None, long_video=False,
                                               dilate_size=-1, id_adapter_resample_learnable_path=None, run_folder=view_folder)
                    if output:
                        results[angle] = output
                        status_lines.append(f"  {angle}°: Done!")
                    torch.cuda.empty_cache()
                except Exception as e:
                    status_lines.append(f"  {angle}°: Error - {str(e)}")
                    torch.cuda.empty_cache()

            status_lines.append("\nComplete!")
            return results[0], results[90], results[180], results[270], "\n".join(status_lines)

        btn_process_multiview.click(
            fn=process_multiview_batch,
            inputs=[traj_draw_points, mv_bg_0, mv_bg_90, mv_bg_180, mv_bg_270,
                    mv_depth_0,
                    mv_fg_0, mv_fg_90, mv_fg_180, mv_fg_270, mv_fg_points_0, mv_fg_points_90,
                    mv_fg_points_180, mv_fg_points_270, mv_fg_element_0, mv_fg_element_90,
                    mv_fg_element_180, mv_fg_element_270, camera_distance, camera_height, camera_fov,
                    mv_rescale, mv_inference_steps],
            outputs=[mv_output_0, mv_output_90, mv_output_180, mv_output_270, mv_status]
        )

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    allowed_paths=["../assets"],
    share=True
)