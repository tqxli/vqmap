import os

import numpy as np
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (QFileDialog, QGridLayout, QHBoxLayout, QLabel,
                             QProgressBar, QPushButton, QScrollArea,
                             QVBoxLayout, QWidget)

from .utils import (AnimatedSequenceCanvas, MatplotlibCanvas,
                    visualize_pose_sequence)


class CodeCell(QPushButton):
    """Custom button for displaying code with usage statistics"""

    def __init__(self, i, j, parent=None):
        super().__init__(parent)
        self.i = i
        self.j = j
        self.setText(f"({i},{j})")
        self.setFixedSize(50, 50)
        self.usage_count = 0
        self.usage_percent = 0

    def update_stats(self, count, total):
        """Update usage statistics for this code cell"""
        self.usage_count = count
        self.usage_percent = (count / total * 100) if total > 0 else 0
        self.setText(f"({self.i},{self.j})\n{self.usage_percent:.1f}%")

        # Create style string with font properties
        style = "font-size: 12px; color: black;"
        
        # Add background color based on usage
        if self.usage_percent > 2:
            style += "background-color: #99ff99;"  # Green for high usage
        elif self.usage_percent > 0.5:
            style += "background-color: #ffff99;"  # Yellow for medium
        
        # Apply all styles at once
        self.setStyleSheet(style)


class EmbeddingThread(QThread):
    result = pyqtSignal(dict)

    def __init__(self, model, config_path):
        super().__init__()
        self.model = model
        self.config_path = config_path
        self.skeleton_name = model.default_skeleton_name

        config_root = "../configs"
        config_name = os.path.basename(config_path)
        with initialize(config_path=config_root, version_base=None):
            cfg = compose(config_name=f"dataset/{config_name}")
        self.cfg = cfg

    def run(self):
        """Run the embedding process in a separate thread"""
        try:
            # Set priority when the thread is actually running
            self.setPriority(QThread.HighestPriority)
            
            datasets = self.model.load_dataset_from_cfg(self.cfg)
            codes, pose3d, skeleton_name = self.model.embed(datasets)
            results = self.model.compute_code_statistics(codes, pose3d, skeleton_name)

            torch.cuda.empty_cache()

            self.skeleton_name = skeleton_name
            self.result.emit(results)
        except Exception as e:
            print(f"Error in embedding: {str(e)}")


class DataAnalyzer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None
        self.code_sequences = None
        self.embedding_results = None
        self.code_cells = {}

        # Main layout
        layout = QVBoxLayout(self)

        # Add a description label
        desc = QLabel("Embed pose datasets into discrete VQ codes")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Dataset selection and analysis controls
        controls_layout = QHBoxLayout()

        controls_layout.addWidget(QLabel("Dataset Config:"))
        self.config_path = QLabel("No file selected")
        self.config_path.setWordWrap(True)
        self.config_path.setFixedWidth(250)
        controls_layout.addWidget(self.config_path)

        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.browse_config)
        controls_layout.addWidget(self.browse_btn)

        self.embed_btn = QPushButton("Embed Dataset")
        self.embed_btn.clicked.connect(self.embed_dataset)
        self.embed_btn.setEnabled(False)
        controls_layout.addWidget(self.embed_btn)

        # Add pause/resume button for animations
        self.pause_btn = QPushButton("Pause Animations")
        self.pause_btn.clicked.connect(self.toggle_animations)
        self.animations_paused = False
        controls_layout.addWidget(self.pause_btn)

        layout.addLayout(controls_layout)

        # Add a horizontal split
        split_line = QLabel()
        split_line.setFrameShape(QLabel.HLine)
        split_line.setFrameShadow(QLabel.Sunken)
        layout.addWidget(split_line)

        # Create horizontal layout for grid and visualization
        h_layout = QHBoxLayout()
        layout.addLayout(h_layout)

        # Left side - scrollable grid of code cells (80%)
        grid_container = QWidget()
        self.grid_layout = QGridLayout(grid_container)
        self.grid_layout.setSpacing(3)  # Reduced spacing

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(grid_container)
        h_layout.addWidget(scroll, 7)  # Stretch factor 7 (70%)

        # Right side - visualization area (20%)
        viz_panel = QWidget()
        viz_layout = QVBoxLayout(viz_panel)
        h_layout.addWidget(viz_panel, 3)  # Stretch factor 3 (30%)

        # Selected code info
        self.selected_code_label = QLabel("Selected Code: None")
        viz_layout.addWidget(self.selected_code_label)

        # Canvas for sequence visualization
        # self.canvas = MatplotlibCanvas(self, width=6, height=6)
        # viz_layout.addWidget(self.canvas)
        n_rows = 3
        n_cols = 2
        plot_args = {
            "linewidth": 0.5,
            "marker_size": 10,
            "alpha": 1.0,
            "coord_limits": 1.0,
        }
        self.canvas = AnimatedSequenceCanvas(
            self, width=2, height=2, plot_args=plot_args
        )
        viz_layout.addWidget(self.canvas)

        # Add a 2x5 grid of animated canvases for examples
        examples_label = QLabel("Example Sequences:")
        viz_layout.addWidget(examples_label)

        examples_grid = QGridLayout()
        viz_layout.addLayout(examples_grid)

        # Create animated canvases for the examples grid
        self.example_canvases = []
        for i in range(n_rows):
            row = []
            for j in range(n_cols):
                canvas = AnimatedSequenceCanvas(
                    self, width=2, height=2, plot_args=plot_args
                )
                examples_grid.addWidget(canvas, i, j)
                row.append(canvas)
            self.example_canvases.append(row)

    def set_model(self, model):
        """Set the model to use"""
        self.model = model
        self.browse_btn.setEnabled(True)
        self.update_ui_state()
        self.setup_grid()

        self.code_sequences = self.model.sample_latent_codes()

    def setup_grid(self):
        """Set up the grid based on model dimensions"""
        # Clear existing grid
        for i in reversed(range(self.grid_layout.count())):
            self.grid_layout.itemAt(i).widget().deleteLater()

        # Create new cells
        self.code_cells = {}
        if self.model:
            for i in range(self.model.N):
                for j in range(self.model.M):
                    cell = CodeCell(i, j)
                    cell.clicked.connect(self.cell_clicked)
                    cell.setObjectName("grid_cell")
                    self.grid_layout.addWidget(cell, i, j)
                    self.code_cells[(i, j)] = cell

    def browse_config(self):
        """Browse for a dataset config file"""
        default_dir = os.path.join(os.path.dirname(__file__), "../configs/dataset")
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Dataset Config",
            default_dir,
            "YAML Files (*.yaml);;All Files (*)",
        )
        if file_path:
            # Get the filename for display
            filename = os.path.basename(file_path)
            # Set the full path as tooltip
            self.config_path.setToolTip(file_path)
            # Display just the filename
            self.config_path.setText(filename)
            self.update_ui_state()

    def update_ui_state(self):
        """Update UI based on current state"""
        enable_state = self.model is not None and self.config_path.text() != "No file selected"
        self.embed_btn.setEnabled(enable_state)
        # change color to green if enabled, red if disabled
        self.embed_btn.setStyleSheet(
            "background-color: #99ff99;" if enable_state else "background-color: #ff9999;"
        )

    def embed_dataset(self):
        """Start the embedding process"""
        if not self.model or self.config_path.text() == "No file selected":
            return

        # Setup and start the embedding thread
        # Update button to show loading status instead of using progress bar
        self.embed_btn.setText("Embedding in progress...")
        self.embed_btn.setEnabled(False)
        self.embed_btn.setStyleSheet("background-color: #ffcc66;")  # Orange for in-progress
        self.browse_btn.setEnabled(False)

        self.thread = EmbeddingThread(self.model, self.config_path.toolTip())
        self.thread.result.connect(self.show_results)
        self.thread.start()

    def show_results(self, results):
        """Display the embedding results"""
        self.embedding_results = results
        # Reset button state to normal
        self.embed_btn.setText("Embed Dataset")
        self.embed_btn.setEnabled(True)
        self.embed_btn.setStyleSheet("")  # Reset to default style
        self.browse_btn.setEnabled(True)

        # Update grid cells with statistics
        code_counts = results["code_counts"]
        total_samples = results["total_samples"]

        for (i, j), cell in self.code_cells.items():
            if i < code_counts.shape[0] and j < code_counts.shape[1]:
                count = code_counts[i, j]
                cell.update_stats(count, total_samples)

        # Clear example visualizations
        for row in self.example_canvases:
            for canvas in row:
                canvas.set_sequence(None, None)

        self.canvas.axes.clear()
        # self.canvas.draw()

        # Notify user
        self.selected_code_label.setText(
            f"Dataset: {results['dataset_name']} - Click on a code cell to view examples"
        )

    def cell_clicked(self):
        """Handle code cell click event"""
        sender = self.sender()
        if not isinstance(sender, CodeCell) or not self.embedding_results:
            return

        i, j = sender.i, sender.j
        self.selected_code_label.setText(
            f"Code: ({i}, {j}) - {sender.usage_percent:.1f}% ({sender.usage_count} instances)"
        )

        # Show the code's representative sequence
        skeleton = self.model.skeletons[self.thread.skeleton_name]
        if (
            self.code_sequences is not None
            and (i, j) in self.code_sequences[self.thread.skeleton_name]
        ):
            sequence = self.code_sequences[self.thread.skeleton_name][(i, j)]
            self.canvas.set_sequence(sequence, skeleton)

        # Show example sequences if available
        example_sequences = None
        if self.embedding_results and "code_to_sequences" in self.embedding_results:
            code_to_sequences = self.embedding_results["code_to_sequences"]
            if (i, j) in code_to_sequences:
                example_sequences = code_to_sequences[(i, j)]

        # Display examples
        if example_sequences is not None:
            self.display_example_sequences(example_sequences)

    def display_example_sequences(self, sequences):
        """Display example sequences in the grid"""
        # Clear all example plots
        for row in self.example_canvases:
            for canvas in row:
                canvas.set_sequence(None, None)

        # Limit to 10 sequences maximum
        n_rows, n_cols = len(self.example_canvases), len(self.example_canvases[0])
        sequences = sequences[: n_rows * n_cols]

        # Display each sequence
        skeleton = self.model.skeletons[self.thread.skeleton_name]
        for idx, sequence in enumerate(sequences):
            i, j = divmod(idx, n_cols)
            if i < n_rows and j < n_cols:
                canvas = self.example_canvases[i][j]
                canvas.set_sequence(sequence, skeleton)

    def toggle_animations(self):
        """Toggle pause/resume for all sequence animations"""
        self.animations_paused = not self.animations_paused

        # Update button text based on state
        self.pause_btn.setText(
            "Resume Animations" if self.animations_paused else "Pause Animations"
        )

        # Pause/resume all example canvas animations
        for row in self.example_canvases:
            for canvas in row:
                if hasattr(canvas, "timer"):
                    if self.animations_paused:
                        canvas.pause_animation()
                    else:
                        canvas.resume_animation()
