from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGridLayout,
    QPushButton,
    QLabel,
    QScrollArea,
    QHBoxLayout,
    QComboBox,
    QLineEdit,
    QColorDialog,
    QInputDialog,
    QFileDialog,
    QFrame,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QAction,
    QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap, QIcon, QColor
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QEvent
from matplotlib import pyplot as plt
import numpy as np
import io
import json
import os
from matplotlib.colors import to_hex, to_rgb
import random

from .utils import MatplotlibCanvas, visualize_pose_sequence, AnimatedSequenceCanvas


class CodeButton(QPushButton):
    """Custom button for displaying code index"""

    def __init__(self, i, j, parent=None):
        super().__init__(parent)
        self.i = i
        self.j = j
        self.setText(f"({i},{j})")
        self.setFixedSize(50, 50)
        self.annotation_label = None
        self.annotation_color = None

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
            
            # Apply styling
            self.update_style()
            
            # Close figure to free memory
            plt.close(fig)
        else:
            # Reset if no sequence
            self.setIcon(QIcon())  # Clear icon
            self.update_style()
    
    def annotate(self, label, color):
        """Set annotation for this button"""
        self.annotation_label = label
        self.annotation_color = color
        self.update_style()
        self.setToolTip(f"Code ({self.i},{self.j}): {label}")
    
    def remove_annotation(self):
        """Remove annotation from this button"""
        self.annotation_label = None
        self.annotation_color = None
        self.update_style()
        self.setToolTip(f"Code ({self.i},{self.j})")
    
    def update_style(self):
        """Update styling based on annotation state"""
        # Base style for text
        text_style = """
            text-align: center;
            padding-top: 3px;
            font-size: 6pt;
        """
        
        # If annotated, apply color
        if self.annotation_color:
            # Get suitable text color (black or white) based on background brightness
            bg_color = QColor(self.annotation_color)
            text_color = "#000000" if bg_color.lightness() > 128 else "#FFFFFF"
            
            self.setStyleSheet(f"""
                QPushButton {{ 
                    {text_style}
                    color: {text_color};
                    background-color: {self.annotation_color};
                    border: 2px solid #444444;
                }}
            """)
        else:
            # Default styling
            self.setStyleSheet(f"""
                QPushButton {{ 
                    {text_style}
                    color: black;
                    background-color: rgba(255, 255, 255, 120);
                }}
            """)


class ModelViewer(QWidget):
    code_selected = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None
        self.code_sequences = None
        self.annotations = {}  # Dictionary to store annotations: {(i,j): "label"}
        self.label_colors = {}  # Dictionary to map labels to colors
        self.checkpoint_dir = None  # Will store the directory of the loaded model

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

        # Add compact annotation panel below the grid and viz panels
        annotation_panel = QFrame()
        annotation_panel.setFrameStyle(QFrame.StyledPanel)
        annotation_panel.setMaximumHeight(150)  # Limit total height
        annotation_layout = QVBoxLayout(annotation_panel)
        annotation_layout.setSpacing(2)
        annotation_layout.setContentsMargins(5, 5, 5, 5)  # Smaller margins
        
        # First row: title, label selector, color controls in one row
        top_row = QHBoxLayout()
        
        annotation_title = QLabel("Code Annotation:")
        annotation_title.setStyleSheet("font-weight: bold;")
        top_row.addWidget(annotation_title)
        
        # Label selector
        self.label_selector = QComboBox()
        self.label_selector.setEditable(True)
        self.label_selector.setPlaceholderText("Select or enter label")
        self.label_selector.setMaximumWidth(200)  # Limit width
        top_row.addWidget(self.label_selector, 1)
        
        # Color indicator
        self.color_indicator = QLabel()
        self.color_indicator.setFixedSize(16, 16)
        self.color_indicator.setStyleSheet("background-color: #CCCCCC; border: 1px solid black;")
        top_row.addWidget(self.color_indicator)
        
        # Add label button (smaller)
        add_label_btn = QPushButton("Add")
        add_label_btn.setMaximumWidth(50)
        add_label_btn.clicked.connect(self.add_new_label)
        top_row.addWidget(add_label_btn)
        
        # Apply annotation button (smaller)
        apply_button = QPushButton("Apply to Selected")
        apply_button.clicked.connect(self.apply_annotation)
        top_row.addWidget(apply_button)
        
        annotation_layout.addLayout(top_row)
        
        # Second row: Combined annotation list and buttons
        bottom_section = QHBoxLayout()
        
        # List of existing annotations - take 70% of width
        self.annotation_list = QListWidget()
        self.annotation_list.setMaximumHeight(90)
        self.annotation_list.itemClicked.connect(self.annotation_selected)
        self.annotation_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.annotation_list.customContextMenuRequested.connect(self.show_annotation_context_menu)
        bottom_section.addWidget(self.annotation_list, 7)  # 70% width
        
        # Right side buttons in vertical layout - take 30% of width
        buttons_layout = QVBoxLayout()
        buttons_layout.setSpacing(2)  # Minimal spacing
        
        save_btn = QPushButton("Save Annotations")
        save_btn.clicked.connect(self.save_annotations)
        buttons_layout.addWidget(save_btn)
        
        load_btn = QPushButton("Load Annotations")
        load_btn.clicked.connect(self.load_annotations)
        buttons_layout.addWidget(load_btn)
        
        export_btn = QPushButton("Export to Text")
        export_btn.clicked.connect(self.export_annotations)
        buttons_layout.addWidget(export_btn)
        
        bottom_section.addLayout(buttons_layout, 3)  # 30% width
        
        annotation_layout.addLayout(bottom_section)
        
        layout.addWidget(annotation_panel)
        
        # Initialize with default colors
        self.initialize_default_labels()

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

        # Set checkpoint directory for saving/loading annotations
        if hasattr(model, 'checkpoint_path'):
            self.checkpoint_dir = os.path.dirname(model.checkpoint_path)
            
            # Try to load existing annotations
            annotation_file = os.path.join(self.checkpoint_dir, "code_annotations.json")
            if os.path.exists(annotation_file):
                try:
                    with open(annotation_file, 'r') as f:
                        data = json.load(f)
                        
                    # Load annotations
                    self.annotations = {}
                    for key, label in data.get("annotations", {}).items():
                        i, j = map(int, key.split(","))
                        self.annotations[(i, j)] = label
                        
                    # Load colors
                    self.label_colors = data.get("colors", {})
                    
                    # Update UI
                    self.label_selector.clear()
                    self.label_selector.addItems(sorted(self.label_colors.keys()))
                    
                except Exception:
                    # If loading fails, initialize with defaults
                    self.initialize_default_labels()
            else:
                self.initialize_default_labels()
        else:
            self.initialize_default_labels()

    def extract_and_display_codes(self):
        """Extract code kinematics and display them in grid"""
        if not self.model:
            return
        
        # Clear the grid
        for i in reversed(range(self.grid_layout.count())): 
            widget = self.grid_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
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
                title = QLabel(f"Code ({i},{j})")
                title.setAlignment(Qt.AlignCenter)
                title.setStyleSheet("font-size: 7pt;")
                cell_layout.addWidget(title)
                
                # Create button
                btn = CodeButton(i, j, self)
                sequence = self.code_sequences[self.model.default_skeleton_name][(i, j)]
                skeleton = self.model.default_skeleton
                btn.set_sequence(sequence, skeleton)
                btn.clicked.connect(self.show_code_sequence)
                cell_layout.addWidget(btn)
                
                # Add the container to the grid
                self.grid_layout.addWidget(cell_widget, i, j)
        
        # Set initial display and update colors based on existing annotations
        if self.code_sequences:
            first_key = (0, 0)
            self.display_sequence(first_key[0], first_key[1])
            self.selected_code = first_key
            self.update_grid_colors()
            self.update_annotation_list()

    def show_code_sequence(self):
        """Display the sequence for the clicked button"""
        sender = self.sender()
        if isinstance(sender, CodeButton):
            i, j = sender.i, sender.j
            self.selected_code = (i, j)  # Track selected code
            self.display_sequence(i, j)
            self.code_selected.emit(i, j)
            
            # Update annotation UI for this code
            if (i, j) in self.annotations:
                label = self.annotations[(i, j)]
                self.label_selector.setCurrentText(label)
            else:
                self.label_selector.setCurrentText("")
            self.update_color_indicator()

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

    def initialize_default_labels(self):
        """Initialize default labels with colors"""
        default_labels = [
            "Rear", "Locomotion", "Groom", "Idle", "Crouch",
        ]
        
        for label in default_labels:
            # Generate a random, visually distinct color
            color = QColor.fromHsv(random.randint(0, 359), 
                                  random.randint(180, 230), 
                                  random.randint(180, 230))
            self.label_colors[label] = color.name()
            
        # Update the combo box
        self.label_selector.clear()
        self.label_selector.addItems(sorted(self.label_colors.keys()))
        
        # Update color indicator for current selection
        self.update_color_indicator()
        
    def update_color_indicator(self):
        """Update the color indicator to match the selected label"""
        current_label = self.label_selector.currentText()
        if current_label in self.label_colors:
            self.color_indicator.setStyleSheet(
                f"background-color: {self.label_colors[current_label]}; border: 1px solid black;")
        else:
            self.color_indicator.setStyleSheet("background-color: #CCCCCC; border: 1px solid black;")
    
    def add_new_label(self):
        """Add a new label with a user-selected color"""
        label = self.label_selector.currentText().strip()
        if not label:
            return
            
        # If label already exists, just update the color
        if label not in self.label_colors:
            # Let user pick a color
            color = QColorDialog.getColor()
            if color.isValid():
                self.label_colors[label] = color.name()
                
                # Add to combo box if not already there
                if self.label_selector.findText(label) == -1:
                    self.label_selector.addItem(label)
                self.label_selector.setCurrentText(label)
        
        self.update_color_indicator()
        self.update_grid_colors()
    
    def apply_annotation(self):
        """Apply the selected label to the currently selected code"""
        # Make sure we have a selected code cell
        if not hasattr(self, 'selected_code') or not self.selected_code:
            return
            
        i, j = self.selected_code
        label = self.label_selector.currentText().strip()
        
        if not label:
            # Remove annotation if empty label
            if (i, j) in self.annotations:
                del self.annotations[(i, j)]
        else:
            # Add label to new colors if needed
            if label not in self.label_colors:
                # Generate a random color
                color = QColor.fromHsv(random.randint(0, 359), 200, 200)
                self.label_colors[label] = color.name()
                if self.label_selector.findText(label) == -1:
                    self.label_selector.addItem(label)
            
            # Apply annotation
            self.annotations[(i, j)] = label
        
        # Update the grid visualization
        self.update_annotation_list()
        self.update_grid_colors()
    
    def update_annotation_list(self):
        """Update the list widget with current annotations"""
        self.annotation_list.clear()
        
        # Group annotations by label
        by_label = {}
        for (i, j), label in self.annotations.items():
            if label not in by_label:
                by_label[label] = []
            by_label[label].append((i, j))
        
        # Add to list widget
        for label, codes in sorted(by_label.items()):
            # Sort codes
            codes.sort()
            # Format text
            codes_str = ", ".join([f"({i},{j})" for i, j in codes])
            display_text = f"{label}: {codes_str}"
            
            # Create item with color
            item = QListWidgetItem(display_text)
            color = self.label_colors.get(label, "#CCCCCC")
            item.setBackground(QColor(color))
            
            # Make text readable against background
            text_color = "#000000" if QColor(color).lightness() > 128 else "#FFFFFF"
            item.setForeground(QColor(text_color))
            
            self.annotation_list.addItem(item)
    
    def update_grid_colors(self):
        """Update colors of grid cells based on annotations"""
        if not hasattr(self, 'grid_layout'):
            return
            
        # Iterate through all grid items
        for i in range(self.grid_layout.rowCount()):
            for j in range(self.grid_layout.columnCount()):
                item = self.grid_layout.itemAtPosition(i, j)
                if not item:
                    continue
                    
                # Get the widget (could be a cell container or a button)
                widget = item.widget()
                
                # Find the button - it's either the widget directly or a child
                btn = None
                if isinstance(widget, CodeButton):
                    btn = widget
                else:
                    # Look for a CodeButton within this widget
                    for child in widget.findChildren(CodeButton):
                        btn = child
                        break
                
                if btn:
                    # Apply color based on annotation
                    if (i, j) in self.annotations:
                        label = self.annotations[(i, j)]
                        color = self.label_colors.get(label, "#CCCCCC")
                        btn.annotate(label, color)
                    else:
                        btn.remove_annotation()
    
    def annotation_selected(self, item):
        """When an annotation is selected from the list"""
        text = item.text()
        # Extract label from text
        if ":" in text:
            label = text.split(":")[0].strip()
            self.label_selector.setCurrentText(label)
            self.update_color_indicator()
    
    def show_annotation_context_menu(self, position):
        """Show context menu for annotation list items"""
        item = self.annotation_list.itemAt(position)
        if not item:
            return
            
        text = item.text()
        if ":" not in text:
            return
            
        label = text.split(":")[0].strip()
        
        # Create context menu
        menu = QMenu()
        change_color = QAction("Change Color", self)
        change_color.triggered.connect(lambda: self.change_label_color(label))
        menu.addAction(change_color)
        
        delete_label = QAction("Delete Label", self)
        delete_label.triggered.connect(lambda: self.delete_label(label))
        menu.addAction(delete_label)
        
        menu.exec_(self.annotation_list.mapToGlobal(position))
    
    def change_label_color(self, label):
        """Change the color for a specific label"""
        if label in self.label_colors:
            color = QColorDialog.getColor(QColor(self.label_colors[label]))
            if color.isValid():
                self.label_colors[label] = color.name()
                self.update_annotation_list()
                self.update_grid_colors()
                self.update_color_indicator()
    
    def delete_label(self, label):
        """Delete a label and all its annotations"""
        if label in self.label_colors:
            # Remove all annotations with this label
            self.annotations = {k: v for k, v in self.annotations.items() if v != label}
            # Remove label from colors
            del self.label_colors[label]
            # Update UI
            self.update_annotation_list()
            self.update_grid_colors()
            
            # Update combo box
            index = self.label_selector.findText(label)
            if index >= 0:
                self.label_selector.removeItem(index)
    
    def save_annotations(self):
        """Save annotations to a JSON file in the checkpoint directory"""
        if not self.checkpoint_dir:
            self.checkpoint_dir = QFileDialog.getExistingDirectory(
                self, "Select Checkpoint Directory", "")
            if not self.checkpoint_dir:
                return
        
        # Create annotations data structure
        data = {
            "annotations": {f"{i},{j}": label for (i, j), label in self.annotations.items()},
            "colors": self.label_colors
        }
        
        # Save to file
        filename = os.path.join(self.checkpoint_dir, "code_annotations.json")
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
        # Inform user
        QMessageBox.information(self, "Saved", f"Annotations saved to {filename}")
    
    def load_annotations(self):
        """Load annotations from a JSON file"""
        if not self.checkpoint_dir:
            self.checkpoint_dir = QFileDialog.getExistingDirectory(
                self, "Select Checkpoint Directory", "")
            if not self.checkpoint_dir:
                return
                
        filename = os.path.join(self.checkpoint_dir, "code_annotations.json")
        
        if not os.path.exists(filename):
            QMessageBox.warning(self, "Not Found", "No annotations file found in this directory")
            return
            
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                
            # Load annotations
            self.annotations = {}
            for key, label in data.get("annotations", {}).items():
                i, j = map(int, key.split(","))
                self.annotations[(i, j)] = label
                
            # Load colors
            self.label_colors = data.get("colors", {})
            
            # Update UI
            self.label_selector.clear()
            self.label_selector.addItems(sorted(self.label_colors.keys()))
            self.update_annotation_list()
            self.update_grid_colors()
            self.update_color_indicator()
            
            QMessageBox.information(self, "Loaded", "Annotations loaded successfully")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load annotations: {str(e)}")
    
    def export_annotations(self):
        """Export annotations to a text file"""
        if not self.annotations:
            QMessageBox.warning(self, "No Data", "No annotations to export")
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Annotations", "", "Text Files (*.txt);;All Files (*)")
            
        if not filename:
            return
            
        try:
            with open(filename, 'w') as f:
                f.write("# Code Annotations\n\n")
                
                # Group by label
                by_label = {}
                for (i, j), label in self.annotations.items():
                    if label not in by_label:
                        by_label[label] = []
                    by_label[label].append((i, j))
                
                # Write each label group
                for label, codes in sorted(by_label.items()):
                    codes.sort()
                    f.write(f"## {label}\n")
                    for i, j in codes:
                        f.write(f"Code ({i},{j})\n")
                    f.write("\n")
                    
            QMessageBox.information(self, "Exported", f"Annotations exported to {filename}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export: {str(e)}")
