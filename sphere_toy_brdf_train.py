"""
Train a BRDF prediction model to fit to a sphere image.
"""

import math
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, cast

# Graphing
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from torch import nn
from tqdm import tqdm

from arguments import (
    BRDFOptmizationParams,
    ModelParams,
    OptimizationParams,
    PipelineParams,
)
from gaussian_renderer.ever import camera2rays
from neural_brdf import (
    Direct_BRDF_Normal_Predictor,
    FullModelOutput,
    Tiny_BRDF_Normal_Predictor,
    eval_blinn_phong_outgoing_radiance,
    transform_normals_to_world_space,
)
from raytracing import (
    generate_spherical_rays,
)
from scene.cameras import MiniCam
from slangtorch_kernel_compilation import sphere_raytracing_kernels
from utils.general_utils import safe_state
from utils.graphics_utils import focal2fov, getProjectionMatrix
from utils.tensor_utils import (
    chw_tensor_to_hwc,
    convert_pc_tensor_to_matplotlib_image_format,
    nchw_tensor_to_npc,
    p_by_c_tensor_to_chw,
    pretty_display_normal_tensor,
)


def plot_rays_and_sphere(
    sphere_center: torch.Tensor,
    sphere_radius: float,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.set_title("Generated Ray Origins and Directions")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.invert_yaxis()

    # Plot output normals:
    # https://matplotlib.org/stable/gallery/mplot3d/quiver3d.html
    ax.quiver3D(
        rays_o[:, 0],
        rays_o[:, 1],
        rays_o[:, 2],
        rays_d[:, 0],
        rays_d[:, 1],
        rays_d[:, 2],
        length=1,
        cmap="tab10",
        arrow_length_ratio=0.2,
    )
    ax.scatter(rays_o[:, 0], rays_o[:, 1], rays_o[:, 2], marker="x", c="red", s=60)
    sphere_center_list = sphere_center.ravel().tolist()

    # Make sphere data (from https://matplotlib.org/stable/gallery/mplot3d/surface3d_2.html)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = sphere_radius * np.outer(np.cos(u), np.sin(v)) + sphere_center_list[0]
    y = sphere_radius * np.outer(np.sin(u), np.sin(v)) + sphere_center_list[1]
    z = sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + sphere_center_list[2]

    ax.plot_surface(x, y, z, alpha=0.7, color="y", label="Sphere")
    ax.scatter(*sphere_center_list, marker="X", s=60, label="Sphere Center")
    ax.set_aspect("equal")


def create_plt_img_comparison_plot(
    current_guess_plt_image: torch.Tensor,
    target_plt_image: torch.Tensor,
    figsize=(9, 3),
):
    figure = plt.figure(figsize=figsize, dpi=300)

    ax = plt.subplot(1, 3, 1)
    plt.title("Rendered Image")
    ax.imshow(current_guess_plt_image)

    ax = plt.subplot(1, 3, 2)
    plt.title("L2 Difference")
    img = ax.imshow(
        torch.norm(
            current_guess_plt_image - target_plt_image, p=1, dim=-1, keepdim=True
        )
    )
    plt.colorbar(img, shrink=0.6)

    ax = plt.subplot(1, 3, 3)
    plt.title("Golden Image")
    ax.imshow(target_plt_image)

    return figure


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Manual Renderer Parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    brdf_optim_params = BRDFOptmizationParams(parser)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    brdf_args = cast(
        BRDFOptmizationParams, brdf_optim_params.extract(args)
    )  # NOTE: Lying to the type checker, but it's close enough.

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Load Gaussians
    model_params: ModelParams = cast(ModelParams, lp.extract(args))

    # torch.autograd.set_detect_anomaly(args.detect_anomaly)
    torch.autograd.set_detect_anomaly(
        True
    )  # always looking for grad anomalies in toy problems

    # Set a global image width and height that is used for instanciating the neural network, etc.
    global_image_height: int = 250
    global_image_width: int = 250
    # NOTE: Using a global batch size of 1.

    # More constants (affecting how much incoming light we use)
    incoming_light_sphere_points = 5
    outgoing_light_sphere_points = 30

    # How many points will we be calculating incoming light for? (P dimension)
    num_points = global_image_height * global_image_width

    # How many points are we using to simulate the probe for incoming / outgoing light
    num_probe_points = 100

    print(
        f"Rendering images at a {global_image_width} x {global_image_height} resolution (W x H)."
    )

    if brdf_args.randomly_sample_loss:
        print(
            f"Sampling {brdf_args.point_batch_size} pixels per iteration for loss calculation."
        )

    # Render our sphere
    fovx = math.radians(39.6)  # Square picture
    fovy = math.radians(39.6)

    # Define camera (50 mm) placed right at the origin
    # https://www.nikonians.org/reviews/fov-tables
    # Other options
    # fovx = math.radians(39.6)
    # fovy = math.radians(27.0)
    # fovx = focal2fov(4328.59, 5025)  # real cameras.json values
    # fovy = focal2fov(4285.45, 3312)  # real cameras.json values
    # fovx = math.radians(60.26544932324648)
    # fovy = math.radians(42.25543987670916)

    # Values copied from raytracing.py
    znear = 0.01
    zfar = 1000
    world_view_transform = torch.zeros((4, 4))
    camera_pos = torch.tensor([0, 0, 0], dtype=torch.float)
    rotation_matrix = torch.eye(3)

    world_view_transform[:3, :3] = rotation_matrix
    world_view_transform[3, :3] = -(torch.linalg.inv(rotation_matrix) @ camera_pos)
    world_view_transform[3, 3] = 1.0
    proj_matrix = getProjectionMatrix(znear, zfar, fovx, fovy).transpose(0, 1)
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(proj_matrix.unsqueeze(0)).squeeze(0)
    )

    rendering_cam = MiniCam(
        global_image_width,
        global_image_height,
        fovy,
        fovx,
        znear,
        zfar,
        world_view_transform,
        full_proj_transform,
    )
    # rendering_cam.model = ProjectionType.PERSPECTIVE
    camera_rays_o, camera_rays_d = camera2rays(rendering_cam)  # (HW, 3)

    # Define sphere
    camera_direction = torch.tensor([0.0, 0.0, 1.0])  # Camera looks along pos z axis
    sphere_distance = 2
    sphere_center = camera_pos + camera_direction * sphere_distance
    sphere_center = sphere_center.cuda().reshape(1, 3)
    sphere_radius = 0.4

    sphere_t_vals = torch.full(
        (camera_rays_o.size(0), 1), float("inf"), device="cuda"
    )  # (HW, 1) Any non-intersections default to "inf" as their t value

    # Intersect sphere
    num_pixels = camera_rays_o.size(0)
    block_size = 1024
    sphere_raytracing_kernels.intersect_sphere(
        ray_origins=camera_rays_o,
        ray_directions=camera_rays_d,
        sphere_center=sphere_center,
        sphere_radius=sphere_radius,
        T_values=sphere_t_vals,
    ).launchRaw(
        blockSize=(block_size, 1, 1), gridSize=(num_pixels // block_size + 1, 1, 1)
    )

    T_vals_image = p_by_c_tensor_to_chw(
        sphere_t_vals, global_image_height, global_image_width
    ).squeeze(0)  # (H, W)
    plt.imshow(T_vals_image.cpu(), cmap="gray")
    plt.title("Sphere T Values")
    plt.colorbar()
    graphs_save_folder = Path(__file__).parent / "graphs"
    sphere_t_values_graph_save_path = graphs_save_folder / "sphere_t_values.png"
    plt.savefig(sphere_t_values_graph_save_path)

    # Debug rays + sphere
    # cpu_rays_o, cpu_rays_d = rays_o.cpu(), rays_d.cpu()
    # plot_rays_and_sphere(sphere_center, sphere_radius, cpu_rays_o, cpu_rays_d)
    # plt.savefig(graphs_save_folder / "rays_o_d_debug.png")

    # plt.show()

    # Convert ray t-values into positions
    # TODO: can do downsampling around here to simulate probe points
    sphere_hit_positions = (
        sphere_t_vals * camera_rays_d + camera_rays_o
    )  # (HW, 3) - note that some values are inf here

    # graphing_sphere_hit_positions = sphere_hit_positions.cpu().T # (3, HW)

    # fig = plt.figure()
    # ax = fig.add_subplot(projection="3d")
    # ax.scatter(*graphing_sphere_hit_positions)
    # plt.title("Sphere Point Scatter Plot")

    # Generate our incoming light directions to sample
    incoming_rays_d, _ = generate_spherical_rays(
        torch.zeros((3,)), num_points=incoming_light_sphere_points
    )  # (R, 3)

    light_direction = (0, 0, 1)
    light_color = (1, 0, 1)

    # Allocate output tensor
    incoming_light = torch.zeros(
        (num_points, incoming_light_sphere_points, 3),
        device="cuda",
        dtype=torch.float,
    )  # (P, R, 3)

    block_size_x = 512
    block_size_y = 2
    sphere_raytracing_kernels.gather_incoming_light_for_sphere_using_directional_light(
        sphere_query_positions=sphere_hit_positions,
        incoming_light_directions=incoming_rays_d,
        sphere_center=sphere_center.ravel().tolist(),
        light_direction=light_direction,
        light_color=light_color,
        output=incoming_light,
    ).launchRaw(
        blockSize=(block_size_x, block_size_y, 1),
        gridSize=(
            incoming_light.size(0) // block_size_x + 1,
            incoming_light.size(1) // block_size_y + 1,
            1,
        ),
    )

    # Graph a subset of incoming light directions
    # light_dir_indices = [0, 14, 25, 30, 50, 99]
    light_dir_indices = [0, 1, 2, 3, 4]
    fig = plt.figure(figsize=(len(light_dir_indices) * 5, 5))
    axs: list[tuple[Axes]] = fig.subplots(
        2, len(light_dir_indices)
    )  # type annotation is a simplication
    for i, light_dir_idx in enumerate(light_dir_indices):
        light_direction_vec = incoming_rays_d[light_dir_idx]
        incoming_light_slice = incoming_light[:, light_dir_idx]  # (P, 3)
        incoming_light_slice_pic = chw_tensor_to_hwc(
            p_by_c_tensor_to_chw(
                incoming_light_slice, global_image_height, global_image_width
            )
        )  # (H, W, 3)
        axs[0][i].imshow(incoming_light_slice_pic.clip(0, 1).cpu())
        axs[0][i].set_title(
            f"Dir {light_dir_idx}: {', '.join(f'{num:.2f}' for num in light_direction_vec.tolist())}"
        )

        non_zero_mask = incoming_light_slice_pic.mean(dim=-1) != 0  # avg val
        axs[1][i].imshow(non_zero_mask.cpu())

    plt.suptitle(
        f"Plots of Incoming light for light direction {light_direction}, light color {light_color}"
    )

    # Basic setup right now - constant Kd with no specular
    golden_normals = torch.nn.functional.normalize(
        (sphere_hit_positions - sphere_center).nan_to_num(
            nan=None, posinf=-1, neginf=-1
        ),
        dim=-1,
    )  # (HW, 3), give inf hits dummy normals
    golden_Kd = torch.full_like(golden_normals, 1.0)  # (HW, 3)
    golden_Ks = torch.full_like(golden_normals, 0.0)  # (HW, 3)
    golden_shininess = torch.full_like(golden_normals, 1.0)[:, 0]  # (HW, 1)

    # Images we'll show to our model
    target_output = eval_blinn_phong_outgoing_radiance(
        incoming_light,
        incoming_rays_d,
        camera_rays_d,
        golden_normals,
        golden_Kd,
        golden_Ks,
        golden_shininess,
    )  # (HW, 3)
    target_image = p_by_c_tensor_to_chw(
        target_output, global_image_height, global_image_width
    )

    # Graph color
    plt.figure()
    plt.imshow(
        convert_pc_tensor_to_matplotlib_image_format(
            target_output, global_image_height, global_image_width
        )
    )
    plt.title("Target Image for optimziation")

    # Graph normals
    plt.figure()
    plt.imshow(
        convert_pc_tensor_to_matplotlib_image_format(
            (golden_normals / 2) + 0.5, global_image_height, global_image_width
        )
    )
    plt.title("Sphere Normals (xyz -> rgb)")

    # TODO: Experiement with light probe, outgoing light matching, SHs, etc.
    # TODO: Add depth and use proper neural network

    # Instanciate the BRDF_normal_predictor
    # brdf_normal_model = Tiny_BRDF_Normal_Predictor(
    #     global_image_height, global_image_width
    # )
    brdf_normal_model = Direct_BRDF_Normal_Predictor(
        global_image_height, global_image_width, num_cameras=1
    )
    brdf_normal_model = brdf_normal_model.cuda()
    brdf_normal_model.train()

    # Training Config
    loss_fn = nn.MSELoss(reduction="none")
    color_penalty = 0.5

    lr = 0.001  # TODO: Use scheduler
    optimizer = torch.optim.AdamW(
        brdf_normal_model.parameters(),
        lr=lr,
    )
    # Main training loop:
    for step_num in tqdm(range(brdf_args.training_steps)):
        optimizer.zero_grad()
        torch.cuda.empty_cache()

        # Ask for our BRDF values
        model_output = cast(
            FullModelOutput, brdf_normal_model(target_output.unsqueeze(0))
        )

        # Collect Model Outputs
        Kd = nchw_tensor_to_npc(model_output["brdf"]["diffuse"])  # (1, P, 3)
        Ks = nchw_tensor_to_npc(model_output["brdf"]["specular"])  # (1, P, 3)
        spec_c = nchw_tensor_to_npc(model_output["brdf"]["specular_c"]).squeeze(
            -1
        )  # (1, P, )

        camera_normals_unnormed = nchw_tensor_to_npc(
            model_output["normal"]
        )  # (1, P, 3)

        ###### BRDF reconstruction ######
        camera_normals_normed = nn.functional.normalize(camera_normals_unnormed, dim=-1)

        # TODO: Add in world normal transformation with a rendering cam?
        # world_normals = transform_normals_to_world_space(
        #     camera_normals_normed, rendering_cam
        # )

        outgoing_radiance = eval_blinn_phong_outgoing_radiance(
            incoming_light,
            incoming_rays_d,
            camera_rays_d,
            # camera_normals_normed[0],
            golden_normals,
            Kd[0],
            Ks[0],
            spec_c[0],
        )  # (HW, 3)

        # Combine both active rendered image difference and outgoing color diff.
        full_diff = loss_fn(outgoing_radiance, target_output)
        loss = torch.mean(full_diff)  # Reduce our diff down to one loss number.

        loss.backward()

        optimizer.step()

        if step_num % brdf_args.image_reporting_interval == 0:
            # Report loss and other metrics
            tqdm.write(f"========Step {step_num}:========")
            tqdm.write(f"{outgoing_radiance = }")
            tqdm.write(f"{loss = }")

    print("Final Results:")
    print(f"{golden_shininess - spec_c[0] = }")
    print(f"{golden_Kd - Kd  = }")
    print(f"{golden_Ks - Ks  = }")
    print(f"{golden_normals - camera_normals_normed  = }")

    outgoing_radiance_plt_image = convert_pc_tensor_to_matplotlib_image_format(
        outgoing_radiance, global_image_height, global_image_width
    )  # (H, W, 3)

    target_plt_image = convert_pc_tensor_to_matplotlib_image_format(
        target_output, global_image_height, global_image_width
    )

    # plt.imsave(
    #     Path(model_params._model_path) / "toy_training_test_output_image.png",
    #     outgoing_radiance_image.permute(1, 2, 0).detach().cpu().clip(0, 1),
    # )

    # Plot two rendered images and their L1 differences
    create_plt_img_comparison_plot(outgoing_radiance_plt_image, target_plt_image)
    plt.suptitle("Comparison of Golden vs Output Toy Image.")

    # Plot Kd, Ks, shininess, and Normals rendered images and their L1 differences
    create_plt_img_comparison_plot(
        convert_pc_tensor_to_matplotlib_image_format(
            Kd[0], global_image_height, global_image_width
        ),
        convert_pc_tensor_to_matplotlib_image_format(
            golden_Kd, global_image_height, global_image_width
        ),
        figsize=(6, 2),
    )
    plt.suptitle("Comparison of Golden vs Output Toy Kd")

    create_plt_img_comparison_plot(
        convert_pc_tensor_to_matplotlib_image_format(
            Ks[0], global_image_height, global_image_width
        ),
        convert_pc_tensor_to_matplotlib_image_format(
            golden_Ks, global_image_height, global_image_width
        ),
        figsize=(6, 2),
    )
    plt.suptitle("Comparison of Golden vs Output Toy Ks")

    # create_plt_img_comparison_plot(
    #     convert_pc_tensor_to_matplotlib_image_format(
    #         pretty_display_normal_tensor(camera_normals_normed),
    #         global_image_height,
    #         global_image_width,
    #     ),
    #     convert_pc_tensor_to_matplotlib_image_format(
    #         pretty_display_normal_tensor(golden_normals),
    #         global_image_height,
    #         global_image_width,
    #     ),
    # )
    # plt.suptitle("Comparison of Golden vs Output Toy Normals")

    # Plot two rendered images and their L1 differences
    plt.show()

    # plt.savefig(Path(model_params._model_path) / "toy_training_comparison_graph.png")

    # All done
