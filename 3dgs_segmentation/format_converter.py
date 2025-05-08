




# import torch
# import numpy as np
# from plyfile import PlyData

# # If you want CUDA support, set your device accordingly:
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def detach_tensors_from_dict(tensor_dict):
#     """
#     Example helper function that detaches and moves any tensor to CPU.
#     Adjust or remove if not needed.
#     """
#     for k, v in tensor_dict.items():
#         if isinstance(v, torch.Tensor):
#             tensor_dict[k] = v.detach().cpu()

# def load_gaussian_splats_from_input_ply_file(input_path: str):
#     """
#     Reads a .ply file containing 3D splats (positions, colors, scales, rotations, etc.)
#     and returns a dictionary of PyTorch tensors plus metadata.

#     :param input_path: Path to the input .ply file.
#     :return: A tuple (splats, metadata) where:
#              - splats is a dict of torch.Tensor objects
#              - metadata is a dict for any extra info
#     """
#     # Read the PLY data
#     ply_data = PlyData.read(input_path)

#     # -- Inspect the elements and their properties (for debug) --
#     print("Elements and Properties in the .ply File:\n")
#     for element in ply_data.elements:
#         print(f"Element: {element.name} ({len(element.data)} entries)")
#         for prop in element.properties:
#             print(f"  Property: {prop.name} ({prop.dtype})")

#     # Usually, the 'vertex' element holds your data
#     vertices = ply_data["vertex"].data

#     # 1) Extract x, y, z coordinates
#     x = vertices['x']
#     y = vertices['y']
#     z = vertices['z']
#     means = np.stack([x, y, z], axis=1)

#     # 2) Direct color features (f_dc_0, f_dc_1, f_dc_2)
#     f_dc_0 = vertices['f_dc_0']
#     f_dc_1 = vertices['f_dc_1']
#     f_dc_2 = vertices['f_dc_2']
#     colors_dc = np.stack([f_dc_0, f_dc_1, f_dc_2], axis=1)

#     # 3) Scaling factors (scale_0, scale_1, scale_2)
#     s0 = vertices['scale_0']
#     s1 = vertices['scale_1']
#     s2 = vertices['scale_2']
#     scaling = np.stack([s0, s1, s2], axis=1)

#     # 4) Rotation quaternions (rot_0, rot_1, rot_2, rot_3)
#     rot_0 = vertices['rot_0']
#     rot_1 = vertices['rot_1']
#     rot_2 = vertices['rot_2']
#     rot_3 = vertices['rot_3']
#     quats = np.stack([rot_0, rot_1, rot_2, rot_3], axis=1)

#     # 5) Opacity
#     opacity = vertices['opacity']
#     opacity = np.array(opacity)

#     # 6) "colors_rest" data (f_rest_0 through f_rest_44)
#     #    Adjust the range (45) if your file has more or fewer fields.
#     colors_rest_list = []
#     for i in range(45):
#         arr = vertices[f'f_rest_{i}']
#         colors_rest_list.append(arr)
#     colors_rest = np.stack(colors_rest_list, axis=1)

#     # Optionally, store any metadata you need
#     metadata = {}

#     # -- Convert to Torch tensors --
#     means = torch.tensor(means, dtype=torch.float32, device=device)
#     sh0 = torch.tensor(colors_dc, dtype=torch.float32, device=device)
#     sh_rest = torch.tensor(colors_rest, dtype=torch.float32, device=device)
#     scale = torch.tensor(scaling, dtype=torch.float32, device=device)
#     quaternion = torch.tensor(quats, dtype=torch.float32, device=device)
#     quaternion = torch.nn.functional.normalize(quaternion, dim=1)  # Normalize quaternions
#     opacity = torch.tensor(opacity, dtype=torch.float32, device=device)

#     print("sh0 (originally colors_dc) shape:", sh0.shape)
#     print("sh_rest (originally colors_rest) shape:", sh_rest.shape)

#     # -- Reshape color data as needed --
#     #
#     # If your usage expects "sh0" to have shape (N, 1, 3)...
#     sh0 = sh0[:, None, :].clone()
#     #
#     # If your usage expects "sh_rest" to have shape (N, 15, 3)...
#     sh_rest = sh_rest.reshape(-1, 15, 3).clone()

#     print("--- After reshape ---")
#     print("means:", means.shape)
#     print("sh0:", sh0.shape)
#     print("sh_rest:", sh_rest.shape)
#     print("scale:", scale.shape)
#     print("quaternion:", quaternion.shape)
#     print("opacity:", opacity.shape)

#     # -- Package everything in a dictionary --
#     #    NOTE: We rename fields to match what your loader wants: 'sh0' instead of 'features_dc', etc.
#     splats = {
#         "active_sh_degree": 3,
#         "xyz": means,         # If your code later expects "xyz", or rename it if needed
#         "sh0": sh0,
#         "sh_rest": sh_rest,
#         "scale": scale,
#         "quaternion": quaternion,
#         "opacity": opacity,
#     }

#     # Detach and move to CPU if desired:
#     detach_tensors_from_dict(splats)

#     return splats, metadata

# if __name__ == "__main__":
#     # Example usage: read 'point_cloud.ply' and save the results to 'gaussians.pt'
#     ply_input = "point_cloud.ply"
#     pt_output = "gaussians.pt"

#     splats, metadata = load_gaussian_splats_from_input_ply_file(ply_input)

#     # Save the dictionary to a .pt file
#     torch.save({"splats": splats, "metadata": metadata}, pt_output)
#     print(f"\nSaved splats to '{pt_output}'.")






# import torch
# import numpy as np
# from plyfile import PlyData

# # If you want CUDA support, set your device accordingly:
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def detach_tensors_from_dict(tensor_dict):
#     """
#     Example helper function that detaches and moves any tensor to CPU.
#     Adjust or remove if not needed.
#     """
#     for k, v in tensor_dict.items():
#         if isinstance(v, torch.Tensor):
#             tensor_dict[k] = v.detach().cpu()

# def load_gaussian_splats_from_input_ply_file(input_path: str):
#     """
#     Reads a .ply file containing 3D splats (positions, colors, scales, rotations, etc.)
#     and returns a dictionary of PyTorch tensors plus metadata.

#     :param input_path: Path to the input .ply file.
#     :return: A tuple (splats, metadata) where:
#              - splats is a dict of torch.Tensor objects with the correct key names
#              - metadata is a dict for any extra info
#     """
#     # Read the PLY data
#     ply_data = PlyData.read(input_path)

#     # -- Inspect the elements and their properties (for debug) --
#     print("Elements and Properties in the .ply File:\n")
#     for element in ply_data.elements:
#         print(f"Element: {element.name} ({len(element.data)} entries)")
#         for prop in element.properties:
#             print(f"  Property: {prop.name} ({prop.dtype})")

#     # Usually, the 'vertex' element holds your data
#     vertices = ply_data["vertex"].data

#     # 1) Extract x, y, z coordinates
#     x = vertices['x']
#     y = vertices['y']
#     z = vertices['z']
#     means_np = np.stack([x, y, z], axis=1)

#     # 2) Direct color features (f_dc_0, f_dc_1, f_dc_2)
#     #    This will become "sh0".
#     f_dc_0 = vertices['f_dc_0']
#     f_dc_1 = vertices['f_dc_1']
#     f_dc_2 = vertices['f_dc_2']
#     sh0_np = np.stack([f_dc_0, f_dc_1, f_dc_2], axis=1)

#     # 3) Scaling factors (scale_0, scale_1, scale_2)
#     #    This will become "scaling".
#     s0 = vertices['scale_0']
#     s1 = vertices['scale_1']
#     s2 = vertices['scale_2']
#     scaling_np = np.stack([s0, s1, s2], axis=1)

#     # 4) Rotation quaternions (rot_0, rot_1, rot_2, rot_3)
#     #    This will become "rotation".
#     rot_0 = vertices['rot_0']
#     rot_1 = vertices['rot_1']
#     rot_2 = vertices['rot_2']
#     rot_3 = vertices['rot_3']
#     rotation_np = np.stack([rot_0, rot_1, rot_2, rot_3], axis=1)

#     # 5) Opacity
#     opacity_np = np.array(vertices['opacity'])

#     # 6) "colors_rest" data (f_rest_0 through f_rest_44)
#     #    This will become "sh_rest".
#     colors_rest_list = []
#     for i in range(45):
#         arr = vertices[f'f_rest_{i}']
#         colors_rest_list.append(arr)
#     sh_rest_np = np.stack(colors_rest_list, axis=1)

#     # Optionally, store any metadata you need
#     metadata = {}

#     # -- Convert to Torch tensors --
#     means = torch.tensor(means_np, dtype=torch.float32, device=device)
#     sh0 = torch.tensor(sh0_np, dtype=torch.float32, device=device)
#     sh_rest = torch.tensor(sh_rest_np, dtype=torch.float32, device=device)
#     scaling = torch.tensor(scaling_np, dtype=torch.float32, device=device)
#     rotation = torch.tensor(rotation_np, dtype=torch.float32, device=device)
#     rotation = torch.nn.functional.normalize(rotation, dim=1)  # Normalize quaternions
#     opacity = torch.tensor(opacity_np, dtype=torch.float32, device=device)

#     print("sh0 shape:", sh0.shape)         # (N, 3)
#     print("sh_rest shape:", sh_rest.shape) # (N, 45)
#     print("means shape:", means.shape)     # (N, 3)
#     print("scaling shape:", scaling.shape) # (N, 3)
#     print("rotation shape:", rotation.shape)  # (N, 4)
#     print("opacity shape:", opacity.shape)    # (N,)

#     # -- Reshape color data as needed --
#     # If your pipeline expects sh0 with shape (N, 1, 3), do:
#     sh0 = sh0[:, None, :].clone()  # (N, 1, 3)

#     # If your pipeline expects sh_rest with shape (N, 15, 3), do:
#     sh_rest = sh_rest.reshape(-1, 15, 3).clone()  # (N, 15, 3)

#     # -- Package everything in a dictionary with EXACT key names your code wants --
#     splats = {
#         "means": means,          # Key your code looks for
#         "sh0": sh0,              # Key your code looks for -> "features_dc": model_params["sh0"]
#         "sh_rest": sh_rest,      # Key your code looks for -> "features_rest": model_params["sh_rest"]
#         "scaling": scaling,      # Key your code looks for -> "scaling": model_params["scaling"]
#         "rotation": rotation,    # Key your code looks for -> "rotation": model_params["quaternion"] (sometimes named quaternion)
#         "opacity": opacity,      # Key your code looks for
#         "active_sh_degree": 3,   # Some pipelines need this
#     }

#     # Detach and move to CPU if desired:
#     detach_tensors_from_dict(splats)

#     return splats, metadata

# if __name__ == "__main__":
#     # Example usage: read 'point_cloud.ply' and save the results to 'gaussians.pt'
#     ply_input = "point_cloud.ply"
#     pt_output = "gaussians.pt"

#     splats, metadata = load_gaussian_splats_from_input_ply_file(ply_input)

#     # Save the dictionary to a .pt file
#     torch.save({"splats": splats, "metadata": metadata}, pt_output)
#     print(f"\nSaved splats to '{pt_output}'.")




import torch
import numpy as np
from plyfile import PlyData

# If you want CUDA support, set your device accordingly:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def detach_tensors_from_dict(tensor_dict):
    """
    Detaches and moves any tensor to CPU.
    """
    for k, v in tensor_dict.items():
        if isinstance(v, torch.Tensor):
            tensor_dict[k] = v.detach().cpu()

def load_gaussian_splats_from_input_ply_file(input_path: str):
    """
    Reads a .ply file containing 3D splats (positions, colors, scales, rotations, etc.)
    and returns a dictionary of PyTorch tensors plus metadata.

    The keys produced here match what your downstream code expects:
      - means
      - sh0
      - shN
      - scales
      - quats
      - opacities
    """
    # Read the PLY data
    ply_data = PlyData.read(input_path)

    # Debug print of .ply structure (optional)
    print("Elements and Properties in the .ply File:")
    for element in ply_data.elements:
        print(f"  Element: {element.name} ({len(element.data)} entries)")
        for prop in element.properties:
            print(f"    Property: {prop.name} ({prop.dtype})")

    # Usually, the 'vertex' element holds your data
    vertices = ply_data["vertex"].data

    # 1) Extract x, y, z => means
    x = vertices['x']
    y = vertices['y']
    z = vertices['z']
    means_np = np.stack([x, y, z], axis=1)

    # 2) DC color features => sh0
    f_dc_0 = vertices['f_dc_0']
    f_dc_1 = vertices['f_dc_1']
    f_dc_2 = vertices['f_dc_2']
    sh0_np = np.stack([f_dc_0, f_dc_1, f_dc_2], axis=1)

    # 3) Scale factors => scales
    scale_0 = vertices['scale_0']
    scale_1 = vertices['scale_1']
    scale_2 = vertices['scale_2']
    scales_np = np.stack([scale_0, scale_1, scale_2], axis=1)

    # 4) Rotation quaternions => quats
    rot_0 = vertices['rot_0']
    rot_1 = vertices['rot_1']
    rot_2 = vertices['rot_2']
    rot_3 = vertices['rot_3']
    quats_np = np.stack([rot_0, rot_1, rot_2, rot_3], axis=1)

    # 5) Opacities => opacities
    opacities_np = np.array(vertices['opacity'])

    # 6) Additional color data => shN (f_rest_0..f_rest_44)
    colors_rest_list = []
    for i in range(45):
        arr = vertices[f'f_rest_{i}']
        colors_rest_list.append(arr)
    shN_np = np.stack(colors_rest_list, axis=1)

    # Convert everything to Torch
    means = torch.tensor(means_np, dtype=torch.float32, device=device)
    sh0 = torch.tensor(sh0_np, dtype=torch.float32, device=device)
    shN = torch.tensor(shN_np, dtype=torch.float32, device=device)
    scales = torch.tensor(scales_np, dtype=torch.float32, device=device)
    quats = torch.tensor(quats_np, dtype=torch.float32, device=device)
    quats = torch.nn.functional.normalize(quats, dim=1)
    opacities = torch.tensor(opacities_np, dtype=torch.float32, device=device)

    # Reshape if your pipeline needs (N,1,3) for sh0 or (N,15,3) for shN, etc.
    sh0 = sh0[:, None, :]  # => (N, 1, 3)
    shN = shN.reshape(-1, 15, 3)  # => (N, 15, 3)

    # Package everything. EXACT key names your loader wants:
    splats = {
        "means": means,
        "sh0": sh0,
        "shN": shN,
        "scales": scales,
        "quats": quats,
        "opacities": opacities,
        "active_sh_degree": 3,  # if needed
    }

    # Detach + CPU if desired
    detach_tensors_from_dict(splats)

    return splats, {}

if __name__ == "__main__":
    # Example usage: read 'point_cloud.ply' => 'gaussians.pt'
    ply_input = "point_cloud.ply"
    pt_output = "gaussians.pt"

    splats, metadata = load_gaussian_splats_from_input_ply_file(ply_input)

    # Save the dictionary to a .pt file
    torch.save({"splats": splats, "metadata": metadata}, pt_output)
    print(f"\nSaved splats to '{pt_output}'.")
