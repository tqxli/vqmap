import hashlib
import os
import sys
import tempfile
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import instantiate
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5.QtCore import Qt, QTimer, QUrl, pyqtSignal
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (QHBoxLayout, QLabel, QMessageBox, QPushButton,
                             QVBoxLayout, QWidget)
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from csbev.core.inference import load_checkpoint
from csbev.dataset.loader import prepare_dataloaders, prepare_datasets
from csbev.utils.run import move_data_to_device
from csbev.utils.visualization import (make_pose_seq_overlay, reset_rcparams,
                                       visualize_pose)


class MatplotlibCanvas(FigureCanvasQTAgg):
    """A canvas that updates itself when new data arrives"""

    def __init__(self, parent=None, width=4, height=4, dpi=200):
        reset_rcparams()
        self.fig = plt.figure(figsize=(width, height), dpi=dpi)
        self.axes = ax = self.fig.add_subplot(111, projection="3d")

        # Initialize the canvas
        # make the panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # make the grid lines transparent
        ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
        ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
        ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        super().__init__(self.fig)
        self.setParent(parent)


class AnimatedSequenceCanvas(QWidget):
    """Canvas that shows an animated sequence of poses"""

    def __init__(self, parent=None, width=3, height=3, dpi=72, plot_args={}):
        super().__init__(parent)
        self.setMinimumSize(width * dpi, height * dpi)

        # Setup figure and canvas
        reset_rcparams()
        self.fig = plt.Figure(figsize=(width, height), dpi=dpi)
        self.axes = ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasQTAgg(self.fig)

        # Initialize the canvas
        # make the panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # make the grid lines transparent
        ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
        ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
        ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Setup animation properties
        self.sequence = None
        self.skeleton = None
        self.current_frame = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.fps = 30  # Frames per second
        self.auto_loop = True  # Add auto-loop functionality
        
        # Cache management
        self.cache_dir = os.path.join(tempfile.gettempdir(), "pose_animation_cache")
        self.max_cache_size_mb = 500  # Maximum cache size in MB
        self.video_path = None
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Title label
        self.title_label = QLabel("")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 12px; color: black;")
        layout.addWidget(self.title_label)

        # Canvas for visualization
        layout.addWidget(self.canvas)

        # Simplified controls - just frame counter, no play button
        controls_layout = QHBoxLayout()

        self.frame_label = QLabel("0/0")
        controls_layout.addWidget(self.frame_label)

        layout.addLayout(controls_layout)

        self.plot_args = plot_args

    def set_sequence(self, sequence, skeleton, title=None):
        """Set the sequence to animate"""
        # Stop any existing animation
        if self.timer.isActive():
            self.timer.stop()

        self.sequence = sequence
        self.skeleton = skeleton
        self.current_frame = 0
        
        # Handle video path
        self.video_path = None

        if title:
            self.title_label.setText(title)

        if sequence is not None:
            self.frame_label.setText(f"0/{len(sequence)-1}")
            
            # Create a hash of the sequence to use as a unique identifier
            data_hash = hashlib.md5(sequence.tobytes()).hexdigest()
            self.video_path = os.path.join(self.cache_dir, f"pose_{data_hash}.mp4")
            
            # Check if we need to create a video
            if not os.path.exists(self.video_path):
                self.create_animation_video(self.video_path)
            else:
                # Touch the file to update access time
                os.utime(self.video_path, None)
            
            # Setup video playback if needed
            if hasattr(self, 'media_player'):
                self.setup_video_playback()
            else:
                # Draw first frame
                self.draw_frame(0)
                # Automatically start playing
                self.timer.start(1000 // self.fps)
                
            # Clean up cache if needed
            self.cleanup_video_cache()
        else:
            self.axes.clear()
            self.canvas.draw()
            self.frame_label.setText("0/0")

    def create_animation_video(self, output_path):
        """Create and save an animation video of the sequence"""
        import matplotlib
        from matplotlib.animation import FuncAnimation

        # Create a new figure for the animation
        temp_fig = plt.figure(figsize=self.fig.get_size_inches())
        temp_ax = temp_fig.add_subplot(111, projection='3d')
        
        # Configure the temp axis to match our display settings
        temp_ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        temp_ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        temp_ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        temp_ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
        temp_ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
        temp_ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
        
        # Initialize frame function
        def update(frame):
            temp_ax.clear()
            pose = self.sequence[frame]
            visualize_pose(pose=pose, skeleton=self.skeleton, ax=temp_ax, **self.plot_args)
            return []
        
        # Create the animation
        anim = FuncAnimation(
            temp_fig, update, frames=len(self.sequence), 
            interval=1000//self.fps, blit=True
        )
        
        # Save the animation
        writer = matplotlib.animation.FFMpegWriter(fps=self.fps, bitrate=5000)
        anim.save(output_path, writer=writer)
        
        # Close the temporary figure
        plt.close(temp_fig)

    def setup_video_playback(self):
        """Set up video playback using media player"""
        # Set up the media content
        self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(os.path.abspath(self.video_path))))
        self.video_widget.show()
        self.media_player.play()

    def handle_state_changed(self, state):
        """Handle media player state changes"""
        # If media playback has ended and auto_loop is true, restart
        if state == QMediaPlayer.StoppedState and self.auto_loop:
            self.media_player.play()

    def cleanup_video_cache(self, force_all=False):
        """Clean up video cache files"""
        # Get all cache files with their info
        cache_files = []
        total_size = 0
        
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.mp4'):
                filepath = os.path.join(self.cache_dir, filename)
                file_size = os.path.getsize(filepath)
                file_time = os.path.getmtime(filepath)
                cache_files.append((filepath, file_size, file_time))
                total_size += file_size
        
        # Convert total size to MB
        total_size_mb = total_size / (1024 * 1024)
        
        # If force_all is True, delete everything
        if force_all:
            for filepath, _, _ in cache_files:
                try:
                    os.remove(filepath)
                except Exception as e:
                    print(f"Failed to delete {filepath}: {e}")
            return
            
        # Check if we need to clean up based on size limit
        if total_size_mb > self.max_cache_size_mb:
            # Sort by access time (oldest first)
            cache_files.sort(key=lambda x: x[2])
            
            # Remove files until we're under the limit
            target_size = self.max_cache_size_mb * 0.8 * 1024 * 1024  # Target 80% of max
            current_size = total_size
            
            while current_size > target_size and cache_files:
                filepath, file_size, _ = cache_files.pop(0)  # Remove oldest
                
                # Don't delete the current video
                if self.video_path and filepath == self.video_path:
                    continue
                    
                try:
                    os.remove(filepath)
                    current_size -= file_size
                    print(f"Removed cache file: {filepath}")
                except Exception as e:
                    print(f"Failed to delete {filepath}: {e}")

    def closeEvent(self, event):
        """Called when the widget is closed"""
        # Clean up resources
        if hasattr(self, 'media_player'):
            self.media_player.stop()
        
        # Stop any animation timer
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()
        
        # Continue with normal close event
        super().closeEvent(event)

    def draw_frame(self, frame_idx):
        """Draw a specific frame of the sequence"""
        if self.sequence is None or frame_idx >= len(self.sequence):
            return
        
        self.axes.clear()
        pose = self.sequence[frame_idx]
        
        visualize_pose(
            pose=pose,
            skeleton=self.skeleton,
            ax=self.axes,
            **self.plot_args
        )
        
        # Draw the updated figure
        self.canvas.draw()

    def update_frame(self):
        """Update to the next frame in the animation"""
        if self.sequence is None or len(self.sequence) == 0:
            return

        self.current_frame = (self.current_frame + 1) % len(self.sequence)
        self.draw_frame(self.current_frame)
        self.frame_label.setText(f"{self.current_frame}/{len(self.sequence)-1}")

        # When reaching the end, restart if auto_loop is enabled
        if self.current_frame == len(self.sequence) - 1 and not self.auto_loop:
            self.timer.stop()

    def stop_animation(self):
        """Stop the animation"""
        if hasattr(self, 'media_player') and self.media_player:
            self.media_player.stop()
        else:
            self.timer.stop()
            self.current_frame = 0
            if self.sequence is not None:
                self.frame_label.setText(f"0/{len(self.sequence)-1}")
                self.draw_frame(0)

    def pause_animation(self):
        """Pause the animation"""
        if hasattr(self, 'media_player') and self.media_player:
            self.media_player.pause()
        else:
            self.timer.stop()

    def resume_animation(self):
        """Resume the animation"""
        if hasattr(self, 'media_player') and self.media_player:
            self.media_player.play()
        else:
            if self.sequence is not None and len(self.sequence) > 0:
                self.timer.start(1000 // self.fps)


class ModelWrapper:
    def __init__(self, cfg_model, model, checkpoint_path):
        self.cfg_model = cfg_model
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        assert hasattr(
            self.model, "sample_latent_codes"
        ), "Model has no function 'sample_latent_codes' to decode latent codes"

        # retrieve needed parameters from config
        self.codebook_size = codebook_size = cfg_model.model.bottleneck.codebook_size
        self.N, self.M = codebook_size[0], codebook_size[1]

        self.skeletons = {
            tag: instantiate(cfg_model.datasets[tag].skeleton)
            for tag in cfg_model.datasets.keys()
        }
        # self.skeletons = {}
        # for tag in cfg_model.datasets.keys():
            # skeleton = instantiate(cfg_model.datasets[tag].skeleton)
            # self.skeletons[skeleton.skeleton_name] = skeleton
        self.skeleton_names = list(self.skeletons.keys())
        self.default_skeleton = self.skeletons[self.skeleton_names[0]]
        self.default_skeleton_name = self.skeleton_names[0]

        self.result_root = os.path.join(cfg_model.expdir, cfg_model.expname, "analysis")
        if not os.path.exists(self.result_root):
            os.makedirs(self.result_root, exist_ok=True)

    def make_dirs(self, folder: str):
        folder = os.path.join(self.result_root, folder)
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        return folder

    def sample_latent_codes(self):
        def _sample_latent_codes(skeleton_name):
            _, trajs, indices = self.model.sample_latent_codes(tag=skeleton_name)
            trajs = trajs.detach().cpu()
            skeleton = self.skeletons[skeleton_name]
            trajs = trajs.reshape(
                trajs.shape[0], skeleton.n_keypoints, skeleton.datadim, -1
            )
            trajs = trajs.permute(0, 3, 1, 2).numpy()

            sequences = {index_2d: traj for (index_2d, traj) in zip(indices, trajs)}
            return sequences

        sequences_all = {
            skeleton_name: _sample_latent_codes(skeleton_name)
            for skeleton_name in self.skeleton_names
        }

        return sequences_all

    def load_dataset_from_cfg(self, cfg, splits=["full"]):
        datasets, skeletons = prepare_datasets(cfg, splits=splits)
        return datasets[splits[0]]

    def embed(self, datasets):
        # need to load datasets again, even if they have already been embedded
        # as we need the original pose3d data for visualization
        dataset_name = list(datasets.keys())[0]
        dataset = datasets[dataset_name]
        pose3d = dataset.pose3d.clone().detach().cpu().numpy()

        savedir = self.make_dirs("embeddings")
        savepath = os.path.join(savedir, f"{dataset_name}.pt")
        if os.path.exists(savepath):
            codes = torch.load(savepath, weights_only=True, map_location="cpu")
            print(f"Loaded embeddings {codes.shape} for {dataset_name} from {savepath}")
            return codes.detach().cpu().numpy(), pose3d, dataset_name

        dataloader = DataLoader(dataset, shuffle=False, batch_size=32, pin_memory=True)
        codes = []
        for batch in tqdm(dataloader, desc="Embedding dataset"):
            batch = move_data_to_device(batch, self.device)
            batch["tag_in"] = batch["tag_out"] = dataset_name
            z, (_, info), _ = self.model.encode(batch)
            mapped_codes = (
                torch.stack([_info[1] for _info in info], dim=1).detach().cpu()
            )  # [num_codes, num_codebooks]
            codes.append(mapped_codes)
        codes = torch.cat(codes, dim=0)
        codes = codes.detach().cpu()
        print(f"Saving embeddings {codes.shape} for {dataset_name} to {savedir}")
        torch.save(codes, savepath)
        return codes.numpy(), pose3d, dataset_name

    def compute_code_statistics(
        self, codes, pose3d, dataset_name, n_sequences=10, duration=25
    ):
        n_frames = pose3d.shape[0]
        n_codes = codes.shape[0]
        assert n_frames % n_codes == 0
        code_duration = int(n_frames / n_codes)
        start_offset = duration // 2
        end_offset = duration - start_offset

        code_counts = torch.zeros((self.N, self.M), dtype=torch.int32)
        code_frame_mapper = defaultdict(list)
        for frame_idx, code in enumerate(codes):
            code_counts[code[0], code[1]] += 1
            code_frame_mapper[(code[0], code[1])].append(frame_idx)

        # sample random sequences for each code
        code_to_sequences = {}
        for code_idx in tqdm(code_frame_mapper.keys(), desc="Sampling sequences"):
            n_codes_tot = code_counts[code_idx[0], code_idx[1]]
            rand_code_indices = np.random.permutation(code_frame_mapper[code_idx])
            rand_frame_indices = rand_code_indices * code_duration

            rand_frame_ranges = []
            count = 0
            while len(rand_frame_ranges) < n_sequences and count < n_codes_tot:
                frame_idx = rand_frame_indices[count]
                if frame_idx - start_offset >= 0 and frame_idx + end_offset <= n_frames:
                    rand_frame_ranges.append(
                        (frame_idx - start_offset, frame_idx + end_offset)
                    )
                count += 1
            # map frame ranges to sequences
            rand_sequences = [
                pose3d[frame_range[0] : frame_range[1], :, :]
                for frame_range in rand_frame_ranges
            ]
            rand_sequences = np.stack(
                rand_sequences, axis=0
            )  # [n_sequences, duration, n_keypoints, 3]
            code_to_sequences[code_idx] = rand_sequences

        results = {
            "code_counts": code_counts,
            "total_samples": codes.shape[0],
            "dataset_name": dataset_name,
            # 'code_to_frames': code_frame_mapper,
            "code_to_sequences": code_to_sequences,
        }
        return results


def load_model(model_path):
    """Load a pretrained model from checkpoint file"""
    try:
        # from checkpoint path, load model weights and hydra config file
        cfg_ckpt, checkpoint = load_checkpoint(model_path)
        cfg_expdir = cfg_ckpt.expdir
        cur_expdir = "/".join(model_path.split("/")[:-4])
        if cfg_expdir != cur_expdir:
            cfg_ckpt.expdir = cur_expdir

        # Instantiate model
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = instantiate(cfg_ckpt.model)
        model.load_state_dict(checkpoint)
        model = model.to(device)
        model = model.eval()

        # Instantiate model wrapper for easier visualization
        model_wrapper = ModelWrapper(cfg_ckpt, model, model_path)
        return model_wrapper

    except Exception as e:
        QMessageBox.critical(None, "Error", f"Failed to load model: {str(e)}")
        return None


def visualize_pose_sequence(ax, sequence, skeleton, title=None):
    """Visualize a keypoint pose sequence on the given axes"""
    # Clear previous plot
    ax.clear()

    ax = make_pose_seq_overlay(
        poseseq=sequence,
        skeleton=skeleton,
        ax=ax,
        n_samples=sequence.shape[0] // 4,
        savename=None,  # do not save
    )
    ax.set_title(title)
    return ax
