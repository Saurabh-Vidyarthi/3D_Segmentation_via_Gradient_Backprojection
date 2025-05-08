from copy import deepcopy
from typing import Literal
import tyro
import os
import torch
import cv2
import imageio  # To generate gifs
import pycolmap_scene_manager as pycolmap
from gsplat import rasterization
import numpy as np
import clip
import matplotlib

matplotlib.use("TkAgg")

from lseg import LSegNet
from utils import (
    create_checkerboard,
    prune_by_gradients,
    test_proper_pruning,
    get_viewmat_from_colmap_image,
    load_checkpoint,
)


def get_mask3d_lseg(splats, features, prompt, neg_prompt, threshold=None):

    net = LSegNet(
        backbone="clip_vitl16_384",
        features=256,
        crop_size=480,
        arch_option=0,
        block_depth=0,
        activation="lrelu",
    )
    # Load pre-trained weights
    net.load_state_dict(torch.load("checkpoint/lseg_minimal_e200.ckpt"))
    net.eval()
    net.cuda()

    # Preprocess the text prompt
    clip_text_encoder = net.clip_pretrained.encode_text

    prompts = [prompt] + neg_prompt.split(";")

    prompt = clip.tokenize(prompts)
    prompt = prompt.cuda()

    text_feat = clip_text_encoder(prompt)  # N, 512, N - number of prompts
    text_feat_norm = torch.nn.functional.normalize(text_feat, dim=1)

    features = torch.nn.functional.normalize(features, dim=1)
    score = features @ text_feat_norm.float().T
    mask_3d = score[:, 0] > score[:, 1:].max(dim=1)[0]
    if threshold is not None:
        mask_3d = mask_3d & (score[:, 0] > threshold)
    mask_3d_inv = ~mask_3d

    return mask_3d, mask_3d_inv


def apply_mask3d(splats, mask3d, mask3d_inverted):
    if mask3d_inverted == None:
        mask3d_inverted = ~mask3d
    extracted = deepcopy(splats)
    deleted = deepcopy(splats)
    masked = deepcopy(splats)



    # Increment only where mask3d is True
    offset = 100*torch.tensor([100.9, 36.2, 400.1], device=splats["means"].device, dtype=splats["means"].dtype)
    extracted["means"] = extracted["means"].clone()  # Ensure we don't modify in-place
    # print(extracted["means"])
    extracted["means"][mask3d,...] += offset  # Add the offset only where mask is 1
    extracted["means"][mask3d,...] += offset  # Add the offset only where mask is 1
    # temp = extracted["means"]
    # temp[mask3d]
    print(extracted["means"], extracted["means"][mask3d])
    

    #extracted["means"] = extracted["means"][mask3d]
    extracted["features_dc"] = extracted["features_dc"][mask3d]
    extracted["features_rest"] = extracted["features_rest"][mask3d]
    extracted["scaling"] = extracted["scaling"][mask3d]
    extracted["rotation"] = extracted["rotation"][mask3d]
    extracted["opacity"] = extracted["opacity"][mask3d]

    deleted["means"] = deleted["means"][mask3d_inverted]
    deleted["features_dc"] = deleted["features_dc"][mask3d_inverted]
    deleted["features_rest"] = deleted["features_rest"][mask3d_inverted]
    deleted["scaling"] = deleted["scaling"][mask3d_inverted]
    deleted["rotation"] = deleted["rotation"][mask3d_inverted]
    deleted["opacity"] = deleted["opacity"][mask3d_inverted]

    masked["features_dc"][mask3d] = 1  # (1 - 0.5) / 0.2820947917738781
    masked["features_dc"][~mask3d] = 0  # (0 - 0.5) / 0.2820947917738781
    masked["features_rest"][~mask3d] = 0

    return extracted, deleted, masked


def render_to_gif(
    output_path: str,
    splats,
    feedback: bool = False,
    use_checkerboard_background: bool = False,
    no_sh: bool = False,
):
    if feedback:
        cv2.destroyAllWindows()
        cv2.namedWindow("Rendering", cv2.WINDOW_NORMAL)
    frames = []
    means = splats["means"]
    colors_dc = splats["features_dc"]
    colors_rest = splats["features_rest"]
    colors = torch.cat([colors_dc, colors_rest], dim=1)
    if no_sh == True:
        colors = colors_dc[:, 0, :]
    opacities = torch.sigmoid(splats["opacity"])
    scales = torch.exp(splats["scaling"])
    quats = splats["rotation"]
    K = splats["camera_matrix"]
    aux_dir = output_path + ".images"
    os.makedirs(aux_dir, exist_ok=True)
    for image in sorted(splats["colmap_project"].images.values(), key=lambda x: x.name):
        viewmat = get_viewmat_from_colmap_image(image)
        output, alphas, meta = rasterization(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmat[None],
            K[None],
            width=K[0, 2] * 2,
            height=K[1, 2] * 2,
            sh_degree=3 if not no_sh else None,
        )
        frame = np.clip(output[0].detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
        if use_checkerboard_background:
            checkerboard = create_checkerboard(frame.shape[1], frame.shape[0])
            alphas = alphas[0].detach().cpu().numpy()
            frame = frame * alphas + checkerboard * (1 - alphas)
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        frames.append(frame)
        if feedback:
            cv2.imshow("Rendering", frame[..., ::-1])
            cv2.imwrite(f"{aux_dir}/{image.name}", frame[..., ::-1])
            cv2.waitKey(1)
    imageio.mimsave(output_path, frames, fps=10, loop=0)
    if feedback:
        cv2.destroyAllWindows()


def render_mask_2d_to_gif(
    splats,
    features,
    prompt,
    neg_prompt,
    output_path: str,
    feedback: bool = False,
):
    if feedback:
        cv2.destroyAllWindows()
        cv2.namedWindow("Rendering", cv2.WINDOW_NORMAL)
    frames = []
    means = splats["means"]
    colors_dc = splats["features_dc"]
    colors_rest = splats["features_rest"]
    colors = torch.cat([colors_dc, colors_rest], dim=1)
    opacities = torch.sigmoid(splats["opacity"])
    scales = torch.exp(splats["scaling"])
    quats = splats["rotation"]
    K = splats["camera_matrix"]
    aux_dir = output_path + ".images"
    os.makedirs(aux_dir, exist_ok=True)

    net = LSegNet(
        backbone="clip_vitl16_384",
        features=256,
        crop_size=480,
        arch_option=0,
        block_depth=0,
        activation="lrelu",
    )
    # Load pre-trained weights
    net.load_state_dict(torch.load("checkpoint/lseg_minimal_e200.ckpt"))
    net.eval()
    net.cuda()

    # Preprocess the text prompt
    clip_text_encoder = net.clip_pretrained.encode_text

    prompts = [prompt] + neg_prompt.split(";")

    prompt = clip.tokenize(prompts)
    prompt = prompt.cuda()

    text_feat = clip_text_encoder(prompt)  # N, 512, N - number of prompts
    text_feat_norm = torch.nn.functional.normalize(text_feat, dim=1)

    # features = torch.nn.functional.normalize(features, dim=1)

    for image in sorted(splats["colmap_project"].images.values(), key=lambda x: x.name):
        viewmat = get_viewmat_from_colmap_image(image)
        output, alphas, meta = rasterization(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats=viewmat[None],
            Ks=K[None],
            width=K[0, 2] * 2,
            height=K[1, 2] * 2,
            sh_degree=3,
        )
        feats_rendered, _, _ = rasterization(
            means,
            quats,
            scales,
            opacities,
            features,
            viewmats=viewmat[None],
            Ks=K[None],
            width=K[0, 2] * 2,
            height=K[1, 2] * 2,
            # sh_degree=3,
        )
        feats_rendered = feats_rendered[0]
        feats_rendered = torch.nn.functional.normalize(feats_rendered, dim=-1)
        score = feats_rendered @ text_feat_norm.float().T
        mask2d = score[..., 0] > score[..., 1:].max(dim=2)[0]
        # print(mask2d.shape)
        mask2d = mask2d[..., None].detach().cpu().numpy()
        frame = np.clip(output[0].detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
        frame = frame * (
            0.75 + 0.25 * mask2d * np.array([255, 0, 0]) + (1 - mask2d) * 0.25
        )
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        frames.append(frame)
        if feedback:
            cv2.imshow("Rendering", frame[..., ::-1])
            cv2.imwrite(f"{aux_dir}/{image.name}", frame[..., ::-1])
            cv2.waitKey(1)
    imageio.mimsave(output_path, frames, fps=10, loop=0)
    if feedback:
        cv2.destroyAllWindows()


def save_to_ckpt(
    output_path: str,
    splats,
):
    # Save Torch Checkpoint
    checkpoint_data = {
        "splats": {
            "means": splats["means"],
            "quats": splats["rotation"],
            "scales": splats["scaling"],
            "opacities": splats["opacity"],
            "sh0": splats["features_dc"],
            "shN": splats["features_rest"],
        }
    }
    torch.save(checkpoint_data, output_path)



def shift_extracted_gaussians(splats, mask3d, shift_value=1, axis=0): # axis=[x,y,z]==[0,1,2]
    shifted_splat = splats.copy()
    shifted_means  = splats["means"].clone()
    shifted_means[mask3d,0]-=2.02
    shifted_means[mask3d,1]-=0.22
    shifted_means[mask3d,2]+=0.01
    shifted_splat["means"]=shifted_means
 
    print("Renderign shifted")
    results_dir: str = "./results/garden",
 
    render_to_gif(
        f"{results_dir}/shifted.gif",
        shifted_splat,
        True,
        use_checkerboard_background=False,
    )






# def main(
#     data_dir: str = "./data/garden",  # colmap path
#     checkpoint: str = "./data/garden/ckpts/ckpt_29999_rank0.pt",  # checkpoint path, can generate from original 3DGS repo
#     results_dir: str = "./results/garden",  # output path
#     rasterizer: Literal[
#         "inria", "gsplat"
#     ] = "gsplat",  # Original or gsplat for checkpoints
#     prompt: str = "Table",
#     neg_prompt: str = "Vase;Other",
#     data_factor: int = 4,
#     show_visual_feedback: bool = True,
#     export_checkpoint: bool = True,
# ):

#     if not torch.cuda.is_available():
#         raise RuntimeError("CUDA is required for this demo")

#     torch.set_default_device("cuda")

#     os.makedirs(results_dir, exist_ok=True)
#     splats = load_checkpoint(
#         checkpoint, data_dir, rasterizer=rasterizer, data_factor=data_factor
#     )
#     splats_optimized = prune_by_gradients(splats)
#     test_proper_pruning(splats, splats_optimized)
#     splats = splats_optimized
#     features = torch.load(f"{results_dir}/features_lseg.pt")
#     mask3d, mask3d_inv = get_mask3d_lseg(splats, features, prompt, neg_prompt)
#     extracted, deleted, masked = apply_mask3d(splats, mask3d, mask3d_inv)

#     print("Information : ---------------------")
#     print(type(mask3d))
#     print(type(extracted))

#     render_mask_2d_to_gif(
#         splats,
#         features,
#         prompt,
#         neg_prompt,
#         f"{results_dir}/mask2d_mean_shift.gif",
#         show_visual_feedback,
#     )

#     render_to_gif(
#         f"{results_dir}/extracted_mean_shift.gif",
#         extracted,
#         show_visual_feedback,
#         use_checkerboard_background=True,
#     )
#     render_to_gif(f"{results_dir}/deleted_mean_shift.gif", deleted, show_visual_feedback)

#     if export_checkpoint:
#         save_to_ckpt(f"{results_dir}/extracted.pt", extracted)
#         save_to_ckpt(f"{results_dir}/deleted.pt", deleted)


# if __name__ == "__main__":
#     tyro.cli(main)










def main(
    data_dir: str = "./data/rushi",  # colmap path
    checkpoint: str = "/home/open/SKV_Mid_Rv/3dgs-gradient-backprojection/data/rushi/ckpts/chkpnt30000.pth",  # checkpoint path, can generate from original 3DGS repo
    results_dir: str = "./results/rushi",
    rasterizer: Literal[
        "inria", "gsplat"
    ] = "inria",  # Original or gsplat for checkpoints
    prompt: str = "Person",
    neg_prompt: str = "Floor;Tree;Ground;Other",
    data_factor: int = 4,
    show_visual_feedback: bool = True,
    export_checkpoint: bool = True,
):
 
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this demo")
 
    torch.set_default_device("cuda")
 
    os.makedirs(results_dir, exist_ok=True)
    splats = load_checkpoint(
        checkpoint, data_dir, rasterizer=rasterizer, data_factor=data_factor
    )
    # splats_optimized = prune_by_gradients(splats)
    # test_proper_pruning(splats, splats_optimized)
    # splats = splats_optimized
    features = torch.load(f"{results_dir}/features_lseg.pt")
    mask3d, mask3d_inv = get_mask3d_lseg(splats, features, prompt, neg_prompt)
    shift_extracted_gaussians(splats, mask3d)
    extracted, deleted, masked = apply_mask3d(splats, mask3d, mask3d_inv)
    render_to_gif(
        f"{results_dir}/extracted.gif",
        splats,
        show_visual_feedback,
        use_checkerboard_background=False,
    )
    render_to_gif(f"{results_dir}/deleted.gif", deleted, show_visual_feedback)

if __name__ == "__main__":
    tyro.cli(main)