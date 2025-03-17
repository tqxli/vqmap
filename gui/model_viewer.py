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
        
        # If annotated, apply color and bold text
        if self.annotation_color:
            # Get suitable text color (black or white) based on background brightness
            bg_color = QColor(self.annotation_color)
            text_color = "#000000" if bg_color.lightness() > 128 else "#FFFFFF"
            
            self.setStyleSheet(f"""
                QPushButton {{ 
                    {text_style}
                    color: {text_color};
                    background-color: {self.annotation_color};
                    font-weight: bold;
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
        
        # First row: title and label palette
        top_row = QVBoxLayout()
        top_row.setSpacing(2)
        
        header_row = QHBoxLayout()
        annotation_title = QLabel("Code Annotation:")
        annotation_title.setStyleSheet("font-weight: bold;")
        header_row.addWidget(annotation_title)
        
        # Add new label button
        add_label_btn = QPushButton("+ New Label")
        add_label_btn.setMaximumWidth(100)
        add_label_btn.clicked.connect(self.create_new_label)
        header_row.addWidget(add_label_btn)
        
        # Apply annotation button
        apply_button = QPushButton("Apply to Selected")
        apply_button.clicked.connect(self.apply_annotation)
        # set this button to orange color
        apply_button.setStyleSheet("background-color: #FFA500; color: white;")
        # move the button away from add label button
        header_row.addStretch(2)
        header_row.addWidget(apply_button)
        
        top_row.addLayout(header_row)
        
        # Label palette - horizontal scrollable area for labels
        palette_scroll = QScrollArea()
        palette_scroll.setWidgetResizable(True)
        palette_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        palette_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        palette_scroll.setMaximumHeight(30)
        
        # Container for label buttons
        palette_widget = QWidget()
        self.palette_layout = QHBoxLayout(palette_widget)
        self.palette_layout.setSpacing(4)
        self.palette_layout.setContentsMargins(2, 0, 2, 0)
        self.palette_layout.addStretch(1)  # Push buttons to the left
        
        palette_scroll.setWidget(palette_widget)
        top_row.addWidget(palette_scroll)
        
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
        
        # Store the currently selected label and initialize label buttons
        self.selected_label = None
        self.label_buttons = {}  # To keep track of label buttons

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
                    self.update_label_buttons()
                    
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
            
            # Update selected label for this code
            if (i, j) in self.annotations:
                label = self.annotations[(i, j)]
                self.select_label(label)
            else:
                # Unselect all
                for btn in self.label_buttons.values():
                    btn.setChecked(False)
                self.selected_label = None

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
        
        # Create palette buttons
        for label in sorted(self.label_colors.keys()):
            self.add_label_to_palette(label)
        
        # Update color indicator for current selection
        self.update_color_indicator()

    def update_label_buttons(self):
        """Update the label buttons based on the current labels"""
        # Clear existing buttons
        for btn in self.label_buttons.values():
            btn.deleteLater()
        self.label_buttons = {}
        
        # Create palette buttons
        for label in sorted(self.label_colors.keys()):
            self.add_label_to_palette(label)
    
    def select_label(self, label):
        """Select a label from the label buttons"""
        self.selected_label = label
        self.update_color_indicator()
    
    def update_color_indicator(self):
        """Update the color indicator to match the selected label"""
        # Reset all buttons to unselected state
        for label, btn in self.label_buttons.items():
            # Use normal style for unselected buttons
            checked = (label == self.selected_label)
            btn.setChecked(checked)
    
    def create_new_label(self):
        """Create a new label with user-selected name and color"""
        label, ok = QInputDialog.getText(self, "New Label", "Enter label name:")
        if ok and label.strip():
            label = label.strip()
            
            # Don't allow duplicates
            if label in self.label_colors:
                QMessageBox.warning(self, "Duplicate", "This label already exists")
                return
            
            # Let user pick a color
            color = QColorDialog.getColor()
            if color.isValid():
                self.label_colors[label] = color.name()
                self.add_label_to_palette(label)
                self.select_label(label)
    
    def apply_annotation(self):
        """Apply the selected label to the currently selected code"""
        # Make sure we have a selected code cell
        if not hasattr(self, 'selected_code') or not self.selected_code:
            return
            
        i, j = self.selected_code
        
        if not self.selected_label:
            # Remove annotation if no label selected
            if (i, j) in self.annotations:
                del self.annotations[(i, j)]
        else:
            # Apply annotation with selected label
            self.annotations[(i, j)] = self.selected_label
        
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

    def annotation_selected(self, item):
        """Handle selection of an annotation from the list"""
        if not item:
            return
        
        text = item.text()
        # Get the label part (before the colon)
        if ":" in text:
            label = text.split(":")[0].strip()
            
            # Select this label in the palette
            if label in self.label_buttons:
                self.select_label(label)
            
            # Find a code with this label to select (first one in the list)
            for (i, j), anno_label in self.annotations.items():
                if anno_label == label:
                    self.selected_code = (i, j)
                    self.display_sequence(i, j)
                    self.code_selected.emit(i, j)
                    break

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
        
        # Add menu actions
        change_color = QAction("Change Color", self)
        change_color.triggered.connect(lambda: self.change_label_color(label))
        menu.addAction(change_color)
        
        rename_label = QAction("Rename Label", self)
        rename_label.triggered.connect(lambda: self.rename_label(label))
        menu.addAction(rename_label)
        
        delete_label = QAction("Delete Label", self)
        delete_label.triggered.connect(lambda: self.delete_label(label))
        menu.addAction(delete_label)
        
        # Add action to select all codes with this label
        select_all = QAction("Select All with This Label", self)
        select_all.triggered.connect(lambda: self.select_codes_with_label(label))
        menu.addAction(select_all)
        
        # Display the menu at the right position
        menu.exec_(self.annotation_list.mapToGlobal(position))

    def select_codes_with_label(self, label):
        """Select all codes that have the specified label"""
        # Get all codes with this label
        codes = [(i, j) for (i, j), l in self.annotations.items() if l == label]
        
        if codes:
            # Select the first code to display its sequence
            self.selected_code = codes[0]
            i, j = codes[0]
            self.display_sequence(i, j)
            self.code_selected.emit(i, j)
            
            # Select the label in the palette
            self.select_label(label)
            
            # Provide feedback about how many codes were found
            count = len(codes)
            QMessageBox.information(self, "Selection", f"Found {count} codes with label '{label}'")

    def save_annotations(self):
        """Save annotations to a JSON file in the checkpoint directory"""
        if not self.checkpoint_dir:
            self.checkpoint_dir = QFileDialog.getExistingDirectory(
                self, "Select Checkpoint Directory", "")
            if not self.checkpoint_dir:
                return
        
        if not self.annotations:
            QMessageBox.warning(self, "No Data", "No annotations to save")
            return
        
        try:
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
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save annotations: {str(e)}")

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
            
            # Clear existing palette buttons
            for btn in self.label_buttons.values():
                btn.deleteLater()
            self.label_buttons = {}
            
            # Recreate palette
            for label in sorted(self.label_colors.keys()):
                self.add_label_to_palette(label)
            
            # Update UI
            self.update_annotation_list()
            self.update_grid_colors()
            
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

    def add_label_to_palette(self, label):
        """Add a label button to the palette"""
        if label in self.label_buttons:
            return  # Already exists
        
        color = self.label_colors.get(label, "#CCCCCC")
        
        # Create a colored button with the label
        btn = QPushButton(label)
        btn.setCheckable(True)
        btn.setMinimumWidth(80)
        btn.setMaximumHeight(24)
        
        # Set color based on label
        text_color = "#000000" if QColor(color).lightness() > 128 else "#FFFFFF"
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: {text_color};
                border: 1px solid #888888;
                border-radius: 3px;
                padding: 2px 8px;
            }}
            QPushButton:checked {{
                border: 2px solid #000000;
                font-weight: bold;
            }}
            QPushButton:hover {{
                border: 2px solid #444444;
            }}
        """)
        
        # Connect click event
        btn.clicked.connect(lambda checked, lbl=label: self.select_label(lbl))
        
        # Add to layout before the stretch
        self.palette_layout.insertWidget(self.palette_layout.count()-1, btn)
        
        # Store reference to button
        self.label_buttons[label] = btn

    def update_grid_colors(self):
        """Update the grid button colors based on annotations"""
        if not hasattr(self, 'model') or not self.model:
            return
        
        # Loop through grid cells
        for i in range(self.model.N):
            for j in range(self.model.M):
                # Find the button in the container widget
                item = self.grid_layout.itemAtPosition(i, j)
                if not item:
                    continue
                
                container = item.widget()
                if not container:
                    continue
                
                # Find the CodeButton inside the container (second widget in VBoxLayout)
                button = None
                for child_idx in range(container.layout().count()):
                    child = container.layout().itemAt(child_idx).widget()
                    if isinstance(child, CodeButton):
                        button = child
                        break
                
                if not button:
                    continue
                
                # Update the button's annotation
                coords = (i, j)
                if coords in self.annotations:
                    label = self.annotations[coords]
                    color = self.label_colors.get(label, "#CCCCCC")
                    button.annotate(label, color)
                else:
                    button.remove_annotation()
