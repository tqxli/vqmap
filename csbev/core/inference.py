from copy import deepcopy
import os
from typing import Dict, List
from loguru import logger
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
import torch
from torch.utils.data import default_collate
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
from tqdm import tqdm
import matplotlib.pyplot as plt
from lightning import seed_everything
import yaml

from csbev.dataset.loader import prepare_datasets, prepare_dataloaders
from csbev.dataset.skeleton import SkeletonProfile
from csbev.utils.visualization import visualize_pose, make_pose_seq_overlay
from csbev.utils.run import move_data_to_device
from csbev.utils.metrics import compute_mpjpe


class InferenceHelper:
    def __init__(
        self,
        cfg: DictConfig,
        skeletons: Dict[str, SkeletonProfile],
        dataset_scales: Dict[str, float],
        dataloaders: Dict[str, Dict[str, torch.utils.data.DataLoader]],
        model: torch.nn.Module,
        device: str = "cuda:0",
    ):
        """Helper class for running inference & analysis using a trained model.

        Args:
            cfg (DictConfig): inference configuration.
            model (torch.nn.Module): trained model.
        """
        self.cfg = cfg
        self.mode_names = list(cfg.modes.keys())
        self.mode_args = {k: {} if v is None else v for k, v in cfg.modes.items()}

        self.skeletons = skeletons
        self.dataset_scales = dataset_scales
        self.dataloaders = dataloaders
        self.tags = list(self.dataloaders.keys())
        self.output_tags = list(model.decoder.output_layers.keys())

        self.model = model
        self.model.eval()

        self.device = device

        self.result_root = cfg.result_root
        if not os.path.exists(self.result_root):
            os.makedirs(self.result_root, exist_ok=True)

    def run(self):
        for mode in self.mode_names:
            getattr(self, mode)(**self.mode_args[mode])

    def make_dirs(self, folder: str):
        folder = os.path.join(self.result_root, folder)
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        return folder

    def embed(self):
        """Embed behavioral data into discrete representations/codes.
        """
        savedir = self.make_dirs("embeddings")
        for tag in self.tags:
            codes = []
            for batch in tqdm(self.dataloaders[tag]):
                batch = move_data_to_device(batch, self.device)
                batch["tag_in"] = batch["tag_out"] = tag
                z, (_, info), _ = self.model.encode(batch)
                mapped_codes = (
                    torch.stack([_info[1] for _info in info], dim=1).detach().cpu()
                )  # [num_codes, num_codebooks]
                codes.append(mapped_codes)
            codes = torch.cat(codes, dim=0)
            print(f"Saving embeddings {codes.shape} for {tag} to {savedir}")
            torch.save(codes, os.path.join(savedir, f"{tag}.pt"))

    def cross_conversion(self):
        savedir = self.make_dirs("cross_conversion")
        for tag in self.tags:
            converted = {tag_out: [] for tag_out in self.output_tags if tag_out != tag}
            for batch in tqdm(self.dataloaders[tag]):
                batch = move_data_to_device(batch, self.device)
                batch["tag_in"] = tag
                
                for tag_out in self.output_tags:
                    if tag_out == tag:
                        continue
                    
                    batch["tag_out"] = tag_out
                
                    out = self.model.reconstruct(batch).detach().cpu()
                    converted[tag_out].append(out)
            
            for tag_out, outputs in converted.items():
                outputs = torch.cat(outputs, dim=0).permute(0, 2, 1).flatten(0, 1)
                print(f"Saving {outputs.shape} for {tag} to {tag_out}")
                torch.save(outputs, os.path.join(savedir, f"{tag}_to_{tag_out}.pt"))

    def sample(
        self,
        repeats: int = 1,
        sort_by_similarity: bool = False,
        sample_index: List[int] = [0, 0],
    ):
        """Decode codebook vectors into short trajectories for basic visualization.

        Args:
            sort_by_similarity (bool, optional): Whether to sort the codebook vectors by similarity. Defaults to False.
            sample_index (List[int], optional): Reference indices for sorting. Defaults to [0, 0].
        """
        assert hasattr(
            self.model, "sample_latent_codes"
        ), "Not a model with quantized bottleneck. Double check."

        savedir = self.make_dirs("latent_space")

        for tag in self.output_tags:
            codevecs, traj, combos = self.model.sample_latent_codes(tag, repeats=repeats)
            

            codebook_size = self.model.bottleneck.codebook_size
            h, w = codebook_size

            if sort_by_similarity:
                codebooks = [
                    self.model.bottleneck.sub_quantizers[i]
                    .codebook.clone()
                    .detach()
                    .cpu()
                    for i in range(2)
                ]
                codebooks = [
                    cb / torch.norm(cb, dim=1).unsqueeze(1) for cb in codebooks
                ]
                argsort_h = torch.argsort(
                    codebooks[0][sample_index[0]] @ codebooks[0].T, descending=True
                )
                argsort_w = torch.argsort(
                    codebooks[1][sample_index[1]] @ codebooks[1].T, descending=True
                )
                argsort = [
                    idx_h * w + idx_w for idx_h in argsort_h for idx_w in argsort_w
                ]
                traj = [traj[idx] for idx in argsort]
                combos = [combos[idx] for idx in argsort]

            fig = plt.figure(figsize=(w * 2, h * 2), dpi=200)

            for idx, combo in enumerate(combos):
                if self.skeletons[tag].datadim == 2:
                    ax = fig.add_subplot(h, w, idx + 1)
                elif self.skeletons[tag].datadim == 3:
                    ax = fig.add_subplot(h, w, idx + 1, projection="3d")
                ax.set_title(f"{combo}")
                poseseq = traj[idx].reshape(
                    self.skeletons[tag].n_keypoints, self.skeletons[tag].datadim, -1
                )
                poseseq = poseseq.permute(2, 0, 1).detach().cpu().numpy()

                ax = make_pose_seq_overlay(
                    poseseq=poseseq,
                    skeleton=self.skeletons[tag],
                    n_samples=5,
                    alpha_min=0.2,
                    linewidth=1.0,
                    marker_size=20,
                    coord_limits=0.8 if self.skeletons[tag].datadim == 3 else 1.8,
                    ax=ax,
                    savename=None,
                )
            plt.tight_layout()
            savename = f"{tag}_r{repeats}.png" if not sort_by_similarity else f"{tag}_r{repeats}_sorted.png"
            fig.savefig(os.path.join(savedir, savename))
            
            plt.close(fig)

    def sample_3codebooks(self, repeats=1):
        savedir = self.make_dirs("latent_space_3codebooks")

        for tag in self.output_tags:
            codevecs, traj, combos = self.model.sample_latent_codes(tag, repeats=repeats)
            
            codebook_size = self.model.bottleneck.codebook_size
            h, w, d = codebook_size

            traj = traj.reshape(h*w, d, *traj.shape[1:])
            combos = np.array(combos)
            combos = combos.reshape(h*w, d, *combos.shape[1:])

            for idx_d in range(d):
                fig = plt.figure(figsize=(w * 2, h * 2), dpi=200)
                
                for idx, combo in enumerate(combos[:, idx_d]):
                    ax = fig.add_subplot(h, w, idx + 1, projection="3d")
                    ax.set_title(f"{combo}")
                    poseseq = traj[idx, idx_d].reshape(
                        self.skeletons[tag].n_keypoints, self.skeletons[tag].datadim, -1
                    )
                    poseseq = poseseq.permute(2, 0, 1).detach().cpu().numpy()

                    ax = make_pose_seq_overlay(
                        poseseq=poseseq,
                        skeleton=self.skeletons[tag],
                        n_samples=5,
                        alpha_min=0.2,
                        linewidth=1.0,
                        marker_size=20,
                        coord_limits=0.8,
                        ax=ax,
                        savename=None,
                    )
                plt.tight_layout()
                savename = f"{tag}_r{repeats}_{idx_d}.png"
                fig.savefig(os.path.join(savedir, savename))
                
                plt.close(fig)

    def vis_reconstruction(self, n_samples: int = 2, fps: int = 50):
        """Visualize the reconstruction of the input poses.
        """
        savedir = self.make_dirs("reconstruction")
        for dataset_name, dataloader in self.dataloaders.items():
            dataset = dataloader.dataset
            if isinstance(dataset, torch.utils.data.Subset):
                dataset = dataset.dataset

            data_indices = np.random.choice(len(dataset), n_samples, replace=False)
            samples = [dataset[i] for i in data_indices]
            batch = default_collate(samples)
            
            batch["tag_in"] = dataset_name

            inputs = batch["x"].detach().cpu().numpy()
            outputs, skeletons_out = [], []
            for tag in self.output_tags:
                skeleton = self.skeletons[tag]
                batch["tag_out"] = tag

                batch = move_data_to_device(batch, self.device)

                out = self.model.reconstruct(batch)
                out = out.detach().cpu().permute(0, 2, 1)
                out = out.reshape(
                    *out.shape[:2], skeleton.n_keypoints, skeleton.datadim
                ).numpy()
                outputs.append(out)
                skeletons_out.append(skeleton)

            # make video for reconstruction
            self.make_video(
                n_samples,
                inputs,
                outputs,
                skeleton_input=self.skeletons[dataset_name],
                skeleton_output=skeletons_out,
                output_names=self.output_tags,
                savedir=savedir,
                video_name=f"reconstruction_{dataset_name}.mp4",
                fps=fps,
            )

    def benchmark(self):
        savedir = self.make_dirs("benchmark")
        for tag in self.output_tags:
            skeleton = self.skeletons[tag]
            trunk_index = skeleton.body_regions.index("Trunk")
            tests = {
                "TrunkOnly": np.where(skeleton.body_region_indices == trunk_index)[0],
                "LimbsOnly": np.where(skeleton.body_region_indices != trunk_index)[0],
            }

            joints_perm = np.random.permutation(skeleton.n_keypoints)
            for num_keep in list(np.arange(0, skeleton.n_keypoints, 5))[1:] + [
                skeleton.n_keypoints
            ]:
                tests[f"RandPerm{num_keep}"] = joints_perm[:num_keep]
                
            connectivity = torch.Tensor(skeleton.connectivity).long()

            benchmark_metrics = {}
            for batch in tqdm(self.dataloaders[tag]):
                batch = move_data_to_device(batch, self.device)

                # reconstruct keypoint subsets
                for testname, keep_indices in tests.items():
                    batch_masked = self._mask_keypoints(batch, keep_indices)
                    batch_masked["tag_in"] = batch_masked["tag_out"] = tag
                    out = self.model.reconstruct(batch_masked)
                    out = out.detach().cpu()

                    mpjpe = compute_mpjpe(
                        out / self.dataset_scales[tag],
                        batch["x"].detach().cpu() / self.dataset_scales[tag],
                    )
                    benchmark_metrics[testname] = mpjpe
            
                # reconstruct synthetic poses from averaging
                batch_syn = deepcopy(batch)
                batch_syn["x"] = batch_syn["x"][:, :, connectivity].mean(dim=-2)
                batch_syn["be"] = torch.max(batch_syn["be"][:, connectivity], dim=-1).values
                batch_syn["tag_in"] = batch_syn["tag_out"] = tag
                out = self.model.reconstruct(batch_syn)
                out = out.detach().cpu()
                mpjpe = compute_mpjpe(
                    out / self.dataset_scales[tag],
                    batch["x"].detach().cpu() / self.dataset_scales[tag],
                )
                benchmark_metrics[f"NovelPose{len(connectivity)}"] = mpjpe

            # torch.save(benchmark_metrics, os.path.join(savedir, f"{tag}.pt"))
            with open(os.path.join(savedir, f"{tag}.yaml"), "w") as f:
                yaml.dump(benchmark_metrics, f)

            for testname, mpjpe in benchmark_metrics.items():
                print(f"{tag} {testname}: {mpjpe:.2f} mm")

    def beh_stability(self):
        """Check how stable the discrete representatons wrt behavior-agnostic augmentation (e.g., mirror LR)
        """
        savedir = self.make_dirs("beh_stability")
        metrics = {}
        for tag in self.tags:
            matches, tot = 0, 0

            for batch in tqdm(self.dataloaders[tag]):
                batch = move_data_to_device(batch, self.device)
                batch["tag_in"] = batch["tag_out"] = tag

                # embed original
                mapped_codes = (
                    torch.stack(
                        [_info[1] for _info in self.model.encode(batch)[1][1]], dim=1
                    )
                    .detach()
                    .cpu()
                )  # [T, num_codebooks]

                # embed augmented
                batch_aug = self.aug_behavior(batch)
                batch_aug["tag_in"] = batch_aug["tag_out"] = tag
                mapped_codes_aug = (
                    torch.stack(
                        [_info[1] for _info in self.model.encode(batch_aug)[1][1]],
                        dim=1,
                    )
                    .detach()
                    .cpu()
                )

                match = (mapped_codes == mapped_codes_aug).float()

                tot += match.shape[0]
                matches += match.sum(dim=0).detach().cpu()

            unchanged = matches / tot * 100
            print(
                f"{tag} BehStability: {[f'{unchanged[i]:.2f}%' for i in range(unchanged.shape[0])]}"
            )
        
            metrics[tag] = unchanged.numpy().tolist()
        
        with open(os.path.join(savedir, "beh_stability.yaml"), "w") as f:
            yaml.dump(metrics, f)

    def _mask_keypoints(self, batch: Dict, keep_indices: List[int]):
        batch_masked = {
            "x": deepcopy(batch["x"])[:, :, keep_indices],
            "be": deepcopy(batch["be"])[:, keep_indices],
        }
        return batch_masked

    def aug_behavior(self, batch: Dict[str, torch.Tensor]):
        batch_aug = {
            "x": deepcopy(batch["x"]),
            "be": deepcopy(batch["be"]),
        }
        batch_aug["x"][..., 1] = -batch_aug["x"][..., 1]
        return batch_aug

    def make_video(
        self,
        n_samples: int,
        inputs: np.ndarray,
        outputs: List[np.ndarray],
        skeleton_input: SkeletonProfile,
        skeleton_output: SkeletonProfile,
        output_names: List[str],
        savedir: str,
        video_name: str = "reconstruction.mp4",
        fps: int = 50,
    ):
        plot_args = {
            "linewidth": 1.0,
            "marker_size": 20,
            "coord_limits": 0.8 if skeleton_input.datadim == 3 else 1.8,
            "alpha": 1.0,
        }
        n_rows = len(outputs) + 1
        fig = plt.figure(figsize=(2 * n_samples, 2 * n_rows), dpi=200)
        ax_input = [
            fig.add_subplot(n_rows, n_samples, idx + 1, projection="3d") if skeleton_input.datadim == 3 else fig.add_subplot(n_rows, n_samples, idx+1)
            for idx in range(n_samples)
        ]
        ax_recon = [
            [
                fig.add_subplot(
                    n_rows, n_samples, n_samples * (i + 1) + idx + 1, projection="3d"
                ) if skeleton_output[i].datadim == 3 else fig.add_subplot(n_rows, n_samples, n_samples * (i + 1) + idx + 1)
                for idx in range(n_samples)
            ]
            for i in range(len(outputs))
        ]

        def animate(i):
            for idx in range(n_samples):
                ax_input[idx].clear()
                visualize_pose(
                    pose=inputs[idx][i],
                    skeleton=skeleton_input,
                    ax=ax_input[idx],
                    **plot_args,
                )
                ax_input[idx].set_title("Input")

                for row in range(len(outputs)):
                    ax_recon[row][idx].clear()
                    visualize_pose(
                        pose=outputs[row][idx][i],
                        skeleton=skeleton_output[row],
                        ax=ax_recon[row][idx],
                        **plot_args,
                    )
                    ax_recon[row][idx].set_title(f"->{output_names[row]}")

            fig.suptitle(f"Frame {i}")

        anim = FuncAnimation(fig, animate, frames=outputs[0].shape[1])
        ffmpeg_writer = FFMpegWriter(fps=fps)
        anim.save(os.path.join(savedir, video_name), writer=ffmpeg_writer)


def load_checkpoint(checkpoint_path: str):
    assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"
    state_dict = torch.load(checkpoint_path)
    checkpoint = state_dict["model"]
    cfg = state_dict["config"]
    cfg = OmegaConf.create(cfg)
    
    epoch = state_dict["metadata"]["epoch"]
    logger.info(f"Loaded a checkpoint trained for {epoch+1} epochs.")

    return cfg, checkpoint


def run_inference(
    cfg: DictConfig, cfg_ckpt: DictConfig, checkpoint: Dict[str, torch.Tensor],
):
    """Run inference with an existing model checkpoint.

    Args:
        cfg (DictConfig): inference configuration.
        cfg_ckpt (DictConfig): cfg loaded from the checkpoint for reproducibility.
        checkpoint (Dict[str, torch.Tensor]): model state dict.
    """
    if cfg.get("seed", None) is not None:
        seed_everything(cfg.seed)
    
    # examine inference modes
    cfg_modes = cfg.modes
    modes = list(cfg_modes.keys())
    for mode in modes:
        assert hasattr(InferenceHelper, mode), f"Invalid inference mode: {mode}"
    logger.info(f"Running inference with modes: {modes}")

    # Prepare datasets and dataloaders
    # if not additionally specified, use the same datasets as training
    if cfg.datasets is None:
        cfg.datasets = cfg_ckpt.datasets
    logger.info(f"Datasets: {list(cfg.datasets.keys())}")
    splits = cfg.get("splits", ["full"])
    datasets, skeletons = prepare_datasets(cfg, splits=splits)

    skeletons_ckpt = {
        tag: instantiate(cfg_ckpt.datasets[tag].skeleton)
        for tag in cfg_ckpt.datasets.keys()
    }
    for k, v in skeletons_ckpt.items():
        if k not in skeletons:
            skeletons[k] = v
    
    # skeletons = {**skeletons, **skeletons_ckpt}

    # Save the dataset scales used during normalization for later computation of absolute errors in mm
    dataset_scales = {}
    for tag, dataset in datasets[splits[0]].items():
        if isinstance(dataset, torch.utils.data.Subset):
            dataset_scales[tag] = dataset.dataset.scale
        else:
            dataset_scales[tag] = dataset.scale
    dataloaders = prepare_dataloaders(datasets, cfg.dataloader)

    # Instantiate model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = instantiate(cfg_ckpt.model)
    model.load_state_dict(checkpoint)
    model.to(device)
    
    cfg.result_root = os.path.join(cfg_ckpt.expdir, cfg_ckpt.expname, cfg.result_root)

    # Instantiate InferenceHelper
    helper = InferenceHelper(
        cfg=cfg,
        skeletons=skeletons,
        dataset_scales=dataset_scales,
        dataloaders=dataloaders[splits[0]],
        model=model,
    )
    helper.run()


@hydra.main(
    config_path="../../configs", config_name="inference.yaml", version_base=None
)
def main(cfg: DictConfig):
    """Main entrypoint for performing inference & analyses using a pretrained model.
    
    python -m csbev.core.inference \
        checkpoint_path=/path/to/checkpoint.pth \
        dataset=mouse_demo
    
    """
    # load from existing model checkpoint
    checkpoint_path = cfg.checkpoint_path
    cfg_ckpt, checkpoint = load_checkpoint(checkpoint_path)
    
    cfg_expdir = cfg_ckpt.expdir
    cur_expdir = "/".join(checkpoint_path.split("/")[:-4])
    if cfg_expdir != cur_expdir:
        cfg_ckpt.expdir = cur_expdir

    run_inference(cfg, cfg_ckpt, checkpoint)


if __name__ == "__main__":
    main()
