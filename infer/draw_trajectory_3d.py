#!/usr/bin/env python3
"""
3D Trajectory Drawing Tool for GenCompositor

This tool opens an interactive 3D matplotlib window where you can:
1. Click to add points in 3D space
2. Rotate the view to position points accurately
3. Save the trajectory to a file

Usage:
    cd /Users/nidalhulaihel/git/project/GenCompositor
    python infer/draw_trajectory_3d.py [output_file.txt]

Controls:
    - Left click: Add a point at the clicked location
    - Right click: Remove the last point
    - Mouse drag: Rotate the 3D view
    - Scroll: Zoom in/out
    - Press 'Enter': Save and exit
    - Press 'Escape': Exit without saving
    - Press 'c': Clear all points
    - Press 'p': Print current points
    - Press 'v': Cycle through preset views (front/side/top/iso)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Trajectory3DDrawer:
    def __init__(self, output_file="trajectory_3d.txt"):
        self.output_file = output_file
        self.points = []
        self.current_y = 0.0  # Default Y (height) for new points

        # Create figure and 3D axes
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Set up the scene
        self.setup_scene()

        # Connect event handlers
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # Store for visualization
        self.point_scatter = None
        self.line_plot = None
        self.camera_markers = []

        # View presets
        self.view_index = 0
        self.views = [
            (20, -60, "Isometric"),      # Default isometric view
            (0, -90, "Front (0°)"),       # Front view (looking from 0° camera)
            (0, 0, "Right (90°)"),        # Right view (looking from 90° camera)
            (0, 90, "Back (180°)"),       # Back view
            (0, 180, "Left (270°)"),      # Left view
            (90, -90, "Top-down"),        # Top-down view
        ]

        self.update_plot()

    def setup_scene(self):
        """Set up the 3D scene with grid and camera markers."""
        # Set axis limits
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)  # Y is up
        self.ax.set_zlim(-5, 5)

        # Labels
        self.ax.set_xlabel('X (Left/Right)', fontsize=10)
        self.ax.set_ylabel('Z (Forward/Back)', fontsize=10)
        self.ax.set_zlabel('Y (Up/Down)', fontsize=10)

        # Note: matplotlib 3D uses (x, y, z) but we want Y=up
        # So we'll swap Y and Z in the display

        # Draw ground plane grid
        xx, zz = np.meshgrid(np.linspace(-5, 5, 11), np.linspace(-5, 5, 11))
        yy = np.zeros_like(xx)
        self.ax.plot_wireframe(xx, zz, yy, alpha=0.2, color='gray')

        # Draw origin
        self.ax.scatter([0], [0], [0], color='black', s=100, marker='+', linewidths=2)

        # Draw camera positions (at distance 5)
        camera_distance = 5.0
        cameras = [
            (0, -camera_distance, 0, '0°\n(Front)', 'red'),
            (camera_distance, 0, 0, '90°\n(Right)', 'green'),
            (0, camera_distance, 0, '180°\n(Back)', 'blue'),
            (-camera_distance, 0, 0, '270°\n(Left)', 'orange'),
        ]

        for x, z, y, label, color in cameras:
            self.ax.scatter([x], [z], [y], color=color, s=200, marker='^', alpha=0.7)
            self.ax.text(x*1.2, z*1.2, y+0.3, label, fontsize=8, ha='center', color=color)

        # Title with instructions
        self.ax.set_title(
            "3D Trajectory Drawing Tool\n"
            "Left-click: Add point | Right-click: Undo | Enter: Save | Esc: Cancel | c: Clear | v: Change view",
            fontsize=10
        )

    def on_click(self, event):
        """Handle mouse click events."""
        if event.inaxes != self.ax:
            return

        if event.button == 1:  # Left click - add point
            # Get the current view angle
            elev = self.ax.elev
            azim = self.ax.azim

            # For 3D clicking, we'll place the point on the ground plane (y=current_y)
            # and use the mouse position to determine x and z

            # This is approximate - 3D picking in matplotlib is tricky
            # We'll use a simpler approach: place points based on the 2D projection

            # Get click coordinates in data space (approximate)
            # For now, we'll use keyboard to set height and click for x,z

            if hasattr(event, 'xdata') and hasattr(event, 'ydata'):
                if event.xdata is not None and event.ydata is not None:
                    # Place point at clicked x,z with current Y height
                    # Note: in mpl 3D, the display coords don't directly map
                    # So we'll use a different approach - keyboard input for now
                    pass

            # Alternative: Add point at predefined position based on current view
            # Let's use a more intuitive method - snap to grid
            self.add_point_interactive()

        elif event.button == 3:  # Right click - remove last point
            if self.points:
                self.points.pop()
                print(f"Removed last point. {len(self.points)} points remaining.")
                self.update_plot()

    def add_point_interactive(self):
        """Add a point interactively using current cursor position."""
        # Since 3D clicking is complex, we'll prompt for coordinates
        # But also show a visual helper

        # For now, add a point at (0, 0, current_y) and let user drag it
        # Or we use the simpler keyboard-based input shown in on_key

        print("\nTo add a point, press number keys:")
        print("  1-9: Add point at preset positions")
        print("  'a': Enter custom coordinates")
        print(f"  Current Y (height): {self.current_y}")

    def on_key(self, event):
        """Handle keyboard events."""
        if event.key == 'enter':
            self.save_and_exit()
        elif event.key == 'escape':
            print("Exiting without saving.")
            plt.close(self.fig)
        elif event.key == 'c':
            self.points = []
            print("Cleared all points.")
            self.update_plot()
        elif event.key == 'p':
            self.print_points()
        elif event.key == 'v':
            self.cycle_view()
        elif event.key == 'a':
            self.add_custom_point()
        elif event.key == 'up':
            self.current_y += 0.5
            print(f"Y (height) set to: {self.current_y}")
        elif event.key == 'down':
            self.current_y -= 0.5
            print(f"Y (height) set to: {self.current_y}")
        elif event.key in '123456789':
            self.add_preset_point(int(event.key))
        elif event.key == '0':
            # Add point at origin
            self.points.append((0.0, self.current_y, 0.0))
            print(f"Added point at origin: (0, {self.current_y}, 0)")
            self.update_plot()

    def add_preset_point(self, num):
        """Add a point at a preset position based on number key."""
        # Preset positions in a 3x3 grid at current Y height
        presets = {
            1: (-2, -2), 2: (0, -2), 3: (2, -2),
            4: (-2, 0),  5: (0, 0),  6: (2, 0),
            7: (-2, 2),  8: (0, 2),  9: (2, 2),
        }
        if num in presets:
            x, z = presets[num]
            self.points.append((float(x), self.current_y, float(z)))
            print(f"Added point {len(self.points)}: ({x}, {self.current_y}, {z})")
            self.update_plot()

    def add_custom_point(self):
        """Add a point with custom coordinates via terminal input."""
        try:
            coords = input("Enter coordinates (x, y, z) or (x, z): ").strip()
            parts = [float(p.strip()) for p in coords.replace(',', ' ').split()]

            if len(parts) == 2:
                x, z = parts
                y = self.current_y
            elif len(parts) == 3:
                x, y, z = parts
            else:
                print("Invalid input. Use 'x, y, z' or 'x, z' format.")
                return

            self.points.append((x, y, z))
            print(f"Added point {len(self.points)}: ({x}, {y}, {z})")
            self.update_plot()

        except ValueError:
            print("Invalid coordinates. Please enter numbers.")
        except EOFError:
            pass

    def cycle_view(self):
        """Cycle through preset camera views."""
        self.view_index = (self.view_index + 1) % len(self.views)
        elev, azim, name = self.views[self.view_index]
        self.ax.view_init(elev=elev, azim=azim)
        print(f"View: {name}")
        self.fig.canvas.draw_idle()

    def print_points(self):
        """Print all current points."""
        print(f"\nCurrent trajectory ({len(self.points)} points):")
        for i, (x, y, z) in enumerate(self.points):
            print(f"  {i+1}. ({x:.2f}, {y:.2f}, {z:.2f})")

    def update_plot(self):
        """Update the 3D plot with current points."""
        # Remove old plot elements
        if self.point_scatter:
            self.point_scatter.remove()
            self.point_scatter = None
        if self.line_plot:
            for line in self.line_plot:
                line.remove()
            self.line_plot = None

        if self.points:
            # Extract coordinates (swap y and z for matplotlib display)
            xs = [p[0] for p in self.points]
            ys = [p[2] for p in self.points]  # z -> y in display
            zs = [p[1] for p in self.points]  # y -> z in display

            # Plot points
            self.point_scatter = self.ax.scatter(
                xs, ys, zs,
                c=range(len(self.points)),
                cmap='viridis',
                s=100,
                edgecolors='black',
                linewidths=1,
                depthshade=True
            )

            # Plot connecting lines
            if len(self.points) > 1:
                self.line_plot = self.ax.plot(
                    xs, ys, zs,
                    'b-',
                    linewidth=2,
                    alpha=0.7
                )

            # Add point labels
            for i, (x, y, z) in enumerate(self.points):
                self.ax.text(x, z, y + 0.2, str(i+1), fontsize=8, ha='center')

        self.fig.canvas.draw_idle()

    def save_and_exit(self):
        """Save trajectory to file and exit."""
        if not self.points:
            print("No points to save!")
            return

        # Ensure output directory exists
        output_dir = os.path.dirname(self.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save to file
        with open(self.output_file, 'w') as f:
            f.write('#3D\n')
            f.write('# 3D Trajectory created with draw_trajectory_3d.py\n')
            f.write('# Format: x, y, z (y = up)\n')
            for x, y, z in self.points:
                f.write(f'{x:.2f}, {y:.2f}, {z:.2f}\n')

        print(f"\nTrajectory saved to: {self.output_file}")
        print(f"Total points: {len(self.points)}")
        self.print_points()

        plt.close(self.fig)

    def run(self):
        """Start the interactive drawing session."""
        print("\n" + "="*60)
        print("3D Trajectory Drawing Tool")
        print("="*60)
        print("\nControls:")
        print("  Number keys (0-9): Add points at preset grid positions")
        print("  'a': Add point with custom coordinates")
        print("  Up/Down arrows: Adjust Y (height) for new points")
        print("  Right-click: Remove last point")
        print("  'v': Cycle through camera views")
        print("  'c': Clear all points")
        print("  'p': Print current points")
        print("  Enter: Save and exit")
        print("  Escape: Exit without saving")
        print("\nGrid positions (at current Y height):")
        print("  7  8  9      (-2,2) (0,2) (2,2)")
        print("  4  5  6  =>  (-2,0) (0,0) (2,0)")
        print("  1  2  3      (-2,-2) (0,-2) (2,-2)")
        print(f"\nCurrent Y (height): {self.current_y}")
        print(f"Output file: {self.output_file}")
        print("="*60 + "\n")

        plt.show()


def main():
    # Default output file
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
    else:
        # Save to trajectories_3d folder
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "assets/trajectories_3d"
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, "custom_trajectory.txt")

    drawer = Trajectory3DDrawer(output_file)
    drawer.run()


if __name__ == "__main__":
    main()
