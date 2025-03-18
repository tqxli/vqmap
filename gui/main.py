import sys
import os
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QTabWidget,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QWidget,
)
from PyQt5.QtCore import Qt

from .model_viewer import ModelViewer
from .data_analyzer import DataAnalyzer
from .utils import load_model


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("VQ-MAP Visualization GUI")
        self.resize(1200, 800)

        # Main layout
        main_widget = QWidget()
        main_widget.setStyleSheet("font-family: 'monospace';")
        main_layout = QVBoxLayout(main_widget)

        # Model selection area
        model_selection_layout = QHBoxLayout()
        checkpoint_label = QLabel("Model Checkpoint:")
        checkpoint_label.setStyleSheet("font-weight: bold;")
        model_selection_layout.addWidget(checkpoint_label)

        self.model_path_label = QLabel("No model selected")
        self.model_path_label.setWordWrap(True)
        model_selection_layout.addWidget(self.model_path_label)

        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_model)
        model_selection_layout.addWidget(self.browse_button)

        self.load_button = QPushButton("Load Model")
        self.load_button.clicked.connect(self.load_selected_model)
        self.load_button.setEnabled(False)

        model_selection_layout.addWidget(self.load_button)
        main_layout.addLayout(model_selection_layout)

        # Tab Widget for different functionalities
        self.tabs = QTabWidget()

        # Model Viewer Tab
        self.model_viewer = ModelViewer()
        self.tabs.addTab(self.model_viewer, "Code Kinematics")

        # Combined Data Analyzer Tab (merges embedding and sequence visualization)
        self.data_analyzer = DataAnalyzer()
        self.tabs.addTab(self.data_analyzer, "Dataset Embedding")

        main_layout.addWidget(self.tabs)
        self.setCentralWidget(main_widget)

    def browse_model(self):
        """Open file dialog to select a model checkpoint"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model Checkpoint",
            "",
            "Checkpoint Files (*.pth);;All Files (*)",
        )

        if file_path:
            self.model_path_label.setText(file_path)
            self.load_button.setEnabled(True)

    def load_selected_model(self):
        """Load the selected model and update all tabs"""
        model_path = self.model_path_label.text()
        if model_path == "No model selected":
            return

        model = load_model(model_path)
        if model:
            self.model_viewer.set_model(model)
            self.data_analyzer.set_model(model)

            # Enable the grid visualization immediately
            self.model_viewer.extract_and_display_codes()

            # Switch to the model viewer tab
            self.tabs.setCurrentIndex(0)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()

    window.setStyleSheet(
        """
        #grid_cell {
            background-color: #cccfce;
            color: white;
            border-radius: 4px;
            padding: 2px 4px;
            border: 1px solid #a0a0a0;
        }
        
        #grid_cell:hover {
            background-color: #b8bab9;
        }
        
        #grid_cell:pressed {
            background-color: #9a9c9b;
            color: #e0e0e0;
            border: 1px inset #808080;
        }
        
    """
    )

    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    # Check how many threads QThreadPool uses by default
    # from PyQt5.QtCore import QThreadPool
    # print(QThreadPool.globalInstance().maxThreadCount())

    # You can change this if needed
    # QThreadPool.globalInstance().setMaxThreadCount(10)  # Set to 10

    main()
