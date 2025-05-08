import math
import os
import time
from typing import Literal
import torch
import tyro
from gsplat import rasterization
import matplotlib

matplotlib.use("TkAgg")  # To avoid conflict with cv2
from tqdm import tqdm
from lseg import LSegNet
import torch.nn as nn

from utils import (
    get_viewmat_from_colmap_image,
    load_checkpoint,
    prune_by_gradients,
    test_proper_pruning,
)


class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()
        self.encoder = nn.Parameter(torch.randn(512, 16))
        self.decoder = nn.Parameter(torch.randn(16, 512))

    def forward(self, x):
        x = x @ self.encoder
        y = x @ self.decoder
        return x, y


encoder_decoder = EncoderDecoder().to("cuda")
encoder_decoder.load_state_dict(torch.load("./encoder_decoder.ckpt"))


def create_feature_field_lseg(splats, batch_size=1, use_cpu=False):
    device = "cpu" if use_cpu else "cuda"

    net = LSegNet(
        backbone="clip_vitl16_384",
        features=256,
        crop_size=480,
        arch_option=0,
        block_depth=0,
        activation="lrelu",
    )
    # Load pre-trained weights
    net.load_state_dict(
        torch.load("./checkpoints/lseg_minimal_e200.ckpt", map_location=device)
    )
    net.eval()
    net.to(device)

    means = splats["means"]
    colors_dc = splats["features_dc"]
    colors_rest = splats["features_rest"]
    colors_all = torch.cat([colors_dc, colors_rest], dim=1)

    colors = colors_dc[:, 0, :]  # * 0
    colors_0 = colors_dc[:, 0, :] * 0
    colors.to(device)
    colors_0.to(device)

    colmap_project = splats["colmap_project"]

    opacities = torch.sigmoid(splats["opacity"])
    scales = torch.exp(splats["scaling"])
    quats = splats["rotation"]
    K = splats["camera_matrix"]
    colors.requires_grad = True
    colors_0.requires_grad = True

    gaussian_features = torch.zeros(colors.shape[0], 16, device=colors.device)
    gaussian_denoms = torch.ones(colors.shape[0], device=colors.device) * 1e-12

    t1 = time.time()

    colors_feats = torch.zeros(
        colors.shape[0], 16, device=colors.device, requires_grad=True
    )
    colors_feats_0 = torch.zeros(
        colors.shape[0], 3, device=colors.device, requires_grad=True
    )

    images = sorted(colmap_project.images.values(), key=lambda x: x.name)

    for batch_start in tqdm(
        range(0, len(images), batch_size),
        desc="Feature backprojection (batches)",
    ):
        batch = images[batch_start : batch_start + batch_size]
        for image in batch:
            viewmat = get_viewmat_from_colmap_image(image)

            width = int(K[0, 2] * 2)
            height = int(K[1, 2] * 2)

            with torch.no_grad():
                output, _, meta = rasterization(
                    means,
                    quats,
                    scales,
                    opacities,
                    colors_all,
                    viewmat[None],
                    K[None],
                    width=width,
                    height=height,
                    sh_degree=3,
                )

                output = torch.nn.functional.interpolate(
                    output.permute(0, 3, 1, 2).to(device),
                    size=(480, 480),
                    mode="bilinear",
                )
                output.to(device)
                feats = net.forward(output)
                feats = torch.nn.functional.normalize(feats, dim=1)
                feats = torch.nn.functional.interpolate(
                    feats, size=(height, width), mode="bilinear"
                )[0]
                feats = feats.permute(1, 2, 0)
                feats = feats @ encoder_decoder.encoder  # 512 -> 16

            output_for_grad, _, meta = rasterization(
                means,
                quats,
                scales,
                opacities,
                colors_feats,
                viewmat[None],
                K[None],
                width=width,
                height=height,
            )

            target = (output_for_grad[0].to(device) * feats).sum()
            target.to(device)
            target.backward()
            colors_feats_copy = colors_feats.grad.clone()
            colors_feats.grad.zero_()

            output_for_grad, _, meta = rasterization(
                means,
                quats,
                scales,
                opacities,
                colors_feats_0,
                viewmat[None],
                K[None],
                width=width,
                height=height,
            )

            target_0 = (output_for_grad[0]).sum()
            target_0.to(device)
            target_0.backward()

            gaussian_features += colors_feats_copy
            gaussian_denoms += colors_feats_0.grad[:, 0]
            colors_feats_0.grad.zero_()

            # Clean up unused variables and free GPU memory
            del (
                viewmat,
                meta,
                _,
                output,
                feats,
                output_for_grad,
                colors_feats_copy,
                target,
                target_0,
            )
            torch.cuda.empty_cache()
    gaussian_features = gaussian_features / gaussian_denoms[..., None]
    gaussian_features = gaussian_features / gaussian_features.norm(dim=-1, keepdim=True)
    # Replace nan values with 0
    gaussian_features[torch.isnan(gaussian_features)] = 0
    t2 = time.time()
    print("Time taken for feature backprojection", t2 - t1)
    return gaussian_features


def main(
    data_dir: str = "./data/garden",  # colmap path
    checkpoint: str = "./data/garden/ckpts/ckpt_29999_rank0.pt",  # checkpoint path, can generate from original 3DGS repo
    results_dir: str = "./results/garden",  # output path
    rasterizer: Literal[
        "inria", "gsplat"
    ] = "gsplat",  # Original or GSplat for checkpoints
    data_factor: int = 4,
    feature_field_batch_count: int = 1,  # Number of batches to process for feature field
    run_feature_field_on_cpu: bool = False,  # Run feature field on CPU
):

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this demo")

    torch.set_default_device("cuda")

    os.makedirs(results_dir, exist_ok=True)
    splats = load_checkpoint(
        checkpoint, data_dir, rasterizer=rasterizer, data_factor=data_factor
    )
    splats_optimized = prune_by_gradients(splats)
    test_proper_pruning(splats, splats_optimized)
    splats = splats_optimized
    features = create_feature_field_lseg(
        splats, feature_field_batch_count, run_feature_field_on_cpu
    )
    print(features.shape)
    torch.save(features, f"{results_dir}/features_lseg_compressed.pt")


if __name__ == "__main__":
    tyro.cli(main)
