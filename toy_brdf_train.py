"""
Train a BRDF prediction model with fake data (fake incoming light, fake BRDF values for each pixel).
"""

from arguments import (
    ModelParams,
    PipelineParams,
    OptimizationParams,
    BRDFOptmizationParams,
)
from argparse import ArgumentParser
from brdf_train import nchw_tensor_to_p_by_c, p_by_c_tensor_to_chw
from neural_brdf import (
    BRDF_normal_predictor,
    transform_normals_to_world_space,
    eval_blinn_phong_outgoing_radiance,
    FullModelOutput,
)
from raytracing import (
    build_gaussian_renderer,
    depth_map_to_xyz,
    get_cameras,
    load_gaussian_model,
    render_gaussians,
    generate_spherical_rays,
    gather_incoming_light_at_points,
    plot_incoming_light_and_outgoing_radiance,
)
from utils.general_utils import safe_state
import sys
from tqdm import tqdm

import torch
from torch import nn

from pathlib import Path
import os

from typing import cast

from torch.utils.tensorboard.writer import SummaryWriter

# Graphing
import matplotlib

matplotlib.use("Agg")  # headless mode
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Manual Renderer Parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    brdf_optim_params = BRDFOptmizationParams(parser)
    parser.add_argument(
        "--start_ever_checkpoint",
        type=str,
        default=None,
        help="Checkpoint to resume ever model from.",
    )
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    brdf_args = cast(
        BRDFOptmizationParams, brdf_optim_params.extract(args)
    )  # NOTE: Lying to the type checker, but it's close enough.

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Load Gaussians
    model_params: ModelParams = cast(ModelParams, lp.extract(args))

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # Set a global image width and height that is used for instanciating the neural network, etc.
    global_image_height: int = 50
    global_image_width: int = 50

    print(
        f"Rendering images at a {global_image_width} x {global_image_height} resolution (W x H)."
    )

    if brdf_args.randomly_sample_loss:
        print(
            f"Sampling {brdf_args.point_batch_size} pixels per iteration for loss calculation."
        )

    # More constants (affecting how much incoming light we use)
    incoming_light_sphere_divisions = 20

    # Calculate how big our incoming light features will be when input into our model.
    test_sphere_o, test_sphere_d = generate_spherical_rays(
        torch.zeros((3,)), incoming_light_sphere_divisions
    )  # (N, 3)

    # How many points will we be calculating incoming light for? (P dimension)
    num_points = (
        global_image_height * global_image_width
        if not brdf_args.randomly_sample_loss
        else brdf_args.point_batch_size
    )

    # Generate fake incoming light data that we'll always use
    constant_incoming_light = True
    constant_spec_brdf = True
    constant_spec_shininess = True
    constant_diff_brdf = True
    constant_normals = True

    # Neater way to specify random toy tensors
    def rand_n_by_3(const: bool, num_points: int):
        if const:
            random_tensor = torch.rand(1, 3).expand((num_points, -1)).contiguous()
        else:
            random_tensor = torch.rand(num_points, 3)

        return random_tensor.cuda()

    toy_incoming_light_dirs = (
        test_sphere_d.unsqueeze(0)
        .expand(num_points, test_sphere_o.size(0), 3)
        .contiguous()
        .cuda()
    )  # (P, N, 3)

    # TODO: Boring incoming light that is constant for every point for now
    if constant_incoming_light:
        toy_incoming_light_color = (
            torch.rand((1, test_sphere_o.size(0), 3))
            .expand(num_points, -1, -1)
            .contiguous()
            .cuda()
        )  # (P, N, 3)
    else:
        toy_incoming_light_color = torch.rand(
            (num_points, test_sphere_o.size(0), 3)
        ).cuda()  # (P, N, 3)

    # Generate our fake "golden" BRDF data
    golden_diffuse_color = rand_n_by_3(constant_diff_brdf, num_points)
    golden_specular_color = rand_n_by_3(constant_spec_brdf, num_points)

    max_spec_c = 16
    if constant_spec_shininess:
        golden_specular_c = (
            torch.rand((1,)).expand(num_points).contiguous().cuda() * max_spec_c
        )  # (P,)
    else:
        golden_specular_c = torch.rand((num_points,)).cuda() * max_spec_c

    golden_normals = nn.functional.normalize(
        rand_n_by_3(constant_normals, num_points), dim=1
    )  # (P, 3)

    # Generate our rendered image to show to the model

    # Generate fake outgoing directions by perturbing the normal vector (trying to keep everything on the same side)
    outgoing_dir_residual = (
        torch.rand((1, 3)).cuda() / 10
    )  # divisor is arbitrary, just trying to keep change small enough
    fake_outgoing_dirs = golden_normals + outgoing_dir_residual  # (P, 3)

    rendered_colors = eval_blinn_phong_outgoing_radiance(
        toy_incoming_light_color,
        toy_incoming_light_dirs,
        fake_outgoing_dirs,
        golden_normals,
        golden_diffuse_color,
        golden_specular_color,
        golden_specular_c,
    )  # (P, 3)
    print(f"{rendered_colors = }")

    # Create model inputs
    rendered_image_rgb = p_by_c_tensor_to_chw(
        rendered_colors, global_image_height, global_image_width
    )  # (3, H, W)
    rendered_image_depth = torch.ones(
        (1, global_image_height, global_image_width)
    ).cuda()  # (1, H, W), Constant depth

    rendered_image = torch.cat(
        (rendered_image_rgb, rendered_image_depth), dim=0
    )  # (N, C, H, W)

    # Save image using matplotlib for visualziation
    plt.imsave(
        Path(model_params.model_path) / "toy_training_test_golden_image.png",
        rendered_image_rgb.permute(1, 2, 0).cpu().clip(0, 1),
    )

    # Instanciate the BRDF_normal_predictor
    brdf_normal_model = BRDF_normal_predictor(global_image_height, global_image_width)
    brdf_normal_model = brdf_normal_model.cuda()
    brdf_normal_model.train()

    # Training Config
    loss_fn = nn.MSELoss()
    color_penalty = 0.5

    lr = 0.001
    grad_norm_clip = 1
    optimizer = torch.optim.AdamW(
        brdf_normal_model.parameters(),
        lr=lr,
    )

    # TODO: Learning rate scheduler
    # TODO: etc, etc.

    # Main training loop:
    for step_num in tqdm(range(brdf_args.training_steps)):
        optimizer.zero_grad()
        torch.cuda.empty_cache()

        # Ask for our BRDF values
        model_output = cast(
            FullModelOutput, brdf_normal_model(rendered_image.unsqueeze(0))
        )
        # print(f"{model_output = }")

        # Collect Model Outputs
        Kd = nchw_tensor_to_p_by_c(model_output["brdf"]["diffuse"])  # (P, 3)
        Ks = nchw_tensor_to_p_by_c(model_output["brdf"]["specular"])  # (P, 3)
        spec_c = nchw_tensor_to_p_by_c(model_output["brdf"]["specular_c"]).squeeze(
            1
        )  # (P, )

        # TODO: Think about ways to improve normal optimization.
        camera_normals_unnormed = nchw_tensor_to_p_by_c(
            model_output["normal"]
        )  # (P, 3)

        # Select only random points if we're sampling:
        rand_points = None
        if brdf_args.randomly_sample_loss:
            # uniformly choose a few points at a time to calculate loss with on low VRAM configs.
            multinom_weights = torch.ones(
                (global_image_height * global_image_width,)
            ).cuda()
            rand_points = torch.multinomial(
                multinom_weights, brdf_args.point_batch_size, replacement=False
            )  # (P, )

            # Select only those indices for all relevant tensors
            rendered_colors = rendered_colors[rand_points, :]

            Kd = Kd[rand_points, :]
            Ks = Ks[rand_points, :]
            spec_c = spec_c[rand_points]
            camera_normals_unnormed = camera_normals_unnormed[rand_points, :]

        ###### BRDF reconstruction ######
        camera_normals_normed = nn.functional.normalize(camera_normals_unnormed, dim=1)

        # TODO: Add in world normal transformation with a rendering cam?
        # world_normals = transform_normals_to_world_space(
        #     camera_normals_normed, rendering_cam
        # )

        # TODO: Add in a functional camera position?
        # camera_pos = rendering_cam.camera_center.cuda()  # (3,)
        # outgoing_directions = nn.functional.normalize(
        #     (camera_pos - all_points_xyz), dim=1
        # )  # (P, 3)

        outgoing_radiance = eval_blinn_phong_outgoing_radiance(
            toy_incoming_light_color,
            toy_incoming_light_dirs,
            fake_outgoing_dirs,
            camera_normals_normed,  # TODO: Use toy world normals?
            Kd,
            Ks,
            spec_c,
        )  # (P, 3)

        loss = loss_fn(outgoing_radiance, rendered_colors)
        loss.backward()

        optimizer.step()

        if step_num % brdf_args.image_reporting_interval == 0:
            # Report loss and other metrics
            tqdm.write(f"========Step {step_num}:========")
            tqdm.write(f"{outgoing_radiance = }")
            tqdm.write(f"{loss = }")
            printfn = tqdm.write
            printfn(f"{camera_normals_unnormed = }")
            printfn(f"{camera_normals_normed = }")

    print("Final Results:")
    print(f"{golden_specular_c - spec_c = }")
    print(f"{golden_diffuse_color - Kd  = }")
    print(f"{golden_specular_color - Ks  = }")
    print(f"{golden_normals - camera_normals_normed  = }")

    outgoing_radiance_image = p_by_c_tensor_to_chw(
        outgoing_radiance, global_image_height, global_image_width
    )  # (3, H, W)
    plt.imsave(
        Path(model_params.model_path) / "toy_training_test_output_image.png",
        outgoing_radiance_image.permute(1, 2, 0).detach().cpu().clip(0, 1),
    )

    # Plot two images and their L1 differences
    figure = plt.figure(figsize=(9, 3), dpi=300)

    ax = plt.subplot(1, 3, 1)
    plt.title("Rendered Image")
    ax.imshow(outgoing_radiance_image.permute(1, 2, 0).detach().cpu().clip(0, 1))

    ax = plt.subplot(1, 3, 2)
    plt.title("L2 Difference")
    img = ax.imshow(
        torch.norm(
            outgoing_radiance_image - rendered_image_rgb, p=1, dim=0, keepdim=True
        )
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .clip(0, 1)
    )
    plt.colorbar(img, shrink=0.6)

    ax = plt.subplot(1, 3, 3)
    plt.title("Golden Image")
    ax.imshow(rendered_image_rgb.permute(1, 2, 0).cpu().clip(0, 1))

    plt.suptitle("Comparison of Golden vs Output Toy Image.")
    plt.savefig(Path(model_params.model_path) / "toy_training_comparison_graph.png")

    # All done
