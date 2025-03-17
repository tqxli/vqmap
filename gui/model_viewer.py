from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGridLayout,
    QPushButton,
    QLabel,
    QScrollArea,
    QHBoxLayout,
)
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QEvent
from matplotlib import pyplot as plt
import numpy as np
import io

from .utils import MatplotlibCanvas, visualize_pose_sequence, AnimatedSequenceCanvas


class CodeButton(QPushButton):
    """Custom button for displaying code index"""

    def __init__(self, i, j, parent=None):
        super().__init__(parent)
        self.i = i
        self.j = j
        self.setFixedSize(60, 60)

    def set_sequence(self, sequence, skeleton):
        self.sequence = sequence
        self.skeleton = skeleton

        if sequence is not None:
            fig = plt.Figure(figsize=(2, 2), dpi=72)
            ax = fig.add_subplot(111, projection="3d")
            visualize_pose_sequence(ax, sequence, skeleton)

            # Convert figure to QPixmap
            buf = io.BytesIO()
            fig.savefig(
                buf, format="png", transparent=True, bbox_inches="tight", pad_inches=0.1
            )
            buf.seek(0)
            image = QImage.fromData(buf.getvalue())
            pixmap = QPixmap.fromImage(image)
            # Set as icon
            icon = QIcon(pixmap)
            self.setIcon(icon)
            self.setIconSize(QSize(self.width() - 15, self.height() - 15))
            # Close figure to free memory
            plt.close(fig)


class ModelViewer(QWidget):
    code_selected = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None
        self.code_sequences = None

        # Main layout
        layout = QVBoxLayout(self)

        # Add a description label
        desc = QLabel(
            "Decoded kinematic patterns from each discrete VQ-MAP code"
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Create horizontal layout for grid and visualization
        h_layout = QHBoxLayout()
        layout.addLayout(h_layout)

        # Left side - Scroll area for the grid (80%)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        h_layout.addWidget(scroll, 7)  # Stretch factor 7 (70%)

        # Container for the grid
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setSpacing(3)  # Reduced spacing
        scroll.setWidget(self.grid_container)

        # Right side - Visualization area (20%)
        viz_panel = QWidget()
        viz_layout = QVBoxLayout(viz_panel)
        h_layout.addWidget(viz_panel, 3)  # Stretch factor 1 (20%)

        self.viz_layout = viz_layout

    def set_model(self, model):
        """Set the model to visualize"""
        self.model = model
        self.code_sequences = None

        # depending on the model, we have arbitary number, different skeletons to visualize
        self.skeleton_names = self.model.skeleton_names

        # make the canvas here
        plot_args = {
            "linewidth": 1.0,
            "marker_size": 20,
            "alpha": 1.0,
            "coord_limits": 1.0,
        }
        self.animated_canvas = {
            skeleton_name: AnimatedSequenceCanvas(
                self, width=2, height=2, plot_args=plot_args
            )
            for skeleton_name in self.skeleton_names
        }
        for skeleton_name, canvas in self.animated_canvas.items():
            self.viz_layout.addWidget(canvas)

    def extract_and_display_codes(self):
        """Extract code kinematics and display them in grid"""
        if not self.model:
            return

        # Clear the grid
        for i in reversed(range(self.grid_layout.count())):
            self.grid_layout.itemAt(i).widget().deleteLater()

        # Extract code kinematics
        self.code_sequences = self.model.sample_latent_codes()

        # Create grid of buttons with titles
        for i in range(self.model.N):
            for j in range(self.model.M):
                # Create a container widget for each cell
                cell_widget = QWidget()
                cell_layout = QVBoxLayout(cell_widget)
                cell_layout.setContentsMargins(2, 2, 2, 2)
                cell_layout.setSpacing(2)
                
                # Create title label
                title = QLabel(f"({i},{j})")
                title.setAlignment(Qt.AlignCenter)
                title.setStyleSheet("font-size: 7pt;")
                cell_layout.addWidget(title)
                
                # Create button
                btn = CodeButton(i, j)
                sequence = self.code_sequences[self.model.default_skeleton_name][(i, j)]
                skeleton = self.model.default_skeleton
                btn.set_sequence(sequence, skeleton)
                btn.clicked.connect(self.show_code_sequence)
                cell_layout.addWidget(btn)
                
                # Add the container to the grid
                self.grid_layout.addWidget(cell_widget, i, j)

        # Display the first code sequence
        if self.code_sequences:
            first_key = (0, 0)
            self.display_sequence(first_key[0], first_key[1])

    def show_code_sequence(self):
        """Display the sequence for the clicked button"""
        sender = self.sender()
        if isinstance(sender, CodeButton):
            self.display_sequence(sender.i, sender.j)
            self.code_selected.emit(sender.i, sender.j)

    def display_sequence(self, i, j):
        """Display a specific code sequence"""
        if (
            self.code_sequences
            and (i, j) in self.code_sequences[self.skeleton_names[0]]
        ):
            for skeleton_name in self.skeleton_names:
                if (i, j) not in self.code_sequences[skeleton_name]:
                    continue

                sequence = self.code_sequences[skeleton_name][(i, j)]
                self.animated_canvas[skeleton_name].set_sequence(
                    sequence,
                    self.model.skeletons[skeleton_name],
                    f"Selected Code ({i},{j})",
                )
