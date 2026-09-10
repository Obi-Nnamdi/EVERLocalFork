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
from utils.tensor_utils import p_by_c_tensor_to_chw, unsqueezed_chw_tensor_to_p_by_c, nchw_tensor_to_npc, size_of_tensor_bytes
from batch_eval_blinn_phong_brdf_mem_save import (
    batch_eval_blinn_phong_outgoing_radiance_with_probe_mem_save,
    batch_eval_blinn_phong_varying_outgoing_radiance_with_probe,
)
from neural_brdf import (
    Tiny_BRDF_Normal_Predictor,
    eval_blinn_phong_outgoing_radiance,
    FullModelOutput,
)

from raytracing import (
    generate_spherical_rays,
)
from utils.general_utils import safe_state
import sys
from tqdm import tqdm
import time

import torch
from torch import nn

from pathlib import Path

from typing import cast


# Graphing
import matplotlib

matplotlib.use("Agg")  # headless mode
import matplotlib.pyplot as plt

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
    global_image_height: int = 20
    global_image_width: int = 20
    # NOTE: Using a global batch size of 1.

    # More constants (affecting how much incoming light we use)
    incoming_light_sphere_divisions = 10
    outgoing_light_sphere_divisions = 5

    # How many points will we be calculating incoming light for? (P dimension)
    num_points = (
        global_image_height * global_image_width
    )

    # How many points are we using to simulate the probe for incoming / outgoing light
    num_probe_points = 100

    print(
        f"Rendering images at a {global_image_width} x {global_image_height} resolution (W x H)."
    )

    if brdf_args.randomly_sample_loss:
        print(
            f"Sampling {brdf_args.point_batch_size} pixels per iteration for loss calculation."
        )


    # Calculate how big our incoming light features will be when input into our model.
    test_incoming_sphere_d, test_incoming_sphere_o = generate_spherical_rays(
        torch.zeros((3,)), incoming_light_sphere_divisions
    )  # (N, 3)

    # Calculate how big our incoming light features will be when input into our model.
    test_outgoing_sphere_d, test_outgoing_sphere_o = generate_spherical_rays(
        torch.zeros((3,)), outgoing_light_sphere_divisions
    )  # (O, 3)



    # Generate a simple point query tensor (strech probe point indices across actual points dim)
    light_query_mapping = torch.linspace(0, num_probe_points - 1, num_points, dtype=torch.int32)[None, :].cuda() # (1, P)

    # Generate fake incoming light data that we'll always use
    constant_incoming_light = True
    constant_outgoing_light = True
    constant_spec_brdf = True
    constant_spec_shininess = True
    constant_diff_brdf = True
    constant_normals = True


    probe_incoming_light_dirs = test_incoming_sphere_d

    # Boring incoming light that is constant for every point for now
    if constant_incoming_light:
        probe_incoming_light_color = (
            torch.rand((1, test_incoming_sphere_o.size(0), 3))
            .expand(num_probe_points, -1, -1)
            .contiguous()
            .cuda()
        )  # (P, N, 3)
    else:
        probe_incoming_light_color = torch.rand(
            (num_probe_points, test_incoming_sphere_o.size(0), 3)
        ).cuda()  # (P, N, 3)


    probe_outgoing_light_dirs = -test_outgoing_sphere_d # Invert sphere directions for outgoing light

    if constant_outgoing_light:
        probe_outgoing_light_color = (
            torch.rand((1, test_outgoing_sphere_o.size(0), 3))
            .expand(num_probe_points, -1, -1)
            .contiguous()
            .cuda()
        )  # (P, O, 3)
    else:
        probe_outgoing_light_color = torch.rand(
            (num_probe_points, test_outgoing_sphere_o.size(0), 3)
        ).cuda()  # (P, O, 3)

    # Neater way to specify random toy tensors
    def rand_1_by_n_by_3(const: bool, num_points: int):
        if const:
            random_tensor = torch.rand(1, 3).expand((num_points, -1)).contiguous()
        else:
            random_tensor = torch.rand(num_points, 3)

        return random_tensor.cuda()[None, :]

    # Generate our fake "golden" BRDF data
    golden_diffuse_color = rand_1_by_n_by_3(constant_diff_brdf, num_points)
    golden_specular_color = rand_1_by_n_by_3(constant_spec_brdf, num_points)

    max_spec_c = 16
    if constant_spec_shininess:
        golden_specular_c = (
            torch.rand((1,)).expand(num_points).contiguous().cuda() * max_spec_c
        )  # (P,)
    else:
        golden_specular_c = torch.rand((num_points,)).cuda() * max_spec_c

    golden_specular_c = golden_specular_c[None, :] # (1, P,)

    golden_normals = nn.functional.normalize(
        rand_1_by_n_by_3(constant_normals, num_points), dim=1
    )  # (P, 3)

    # Generate our rendered image to show to the model

    # Generate fake outgoing directions by perturbing the normal vector (trying to keep everything on the same side)
    outgoing_dir_residual = (
        torch.rand((1, 1, 3)).cuda() / 10
    )  # divisor is arbitrary, just trying to keep change small enough
    fake_outgoing_dirs = golden_normals + outgoing_dir_residual  # (P, 3)

    rendered_colors = batch_eval_blinn_phong_outgoing_radiance_with_probe_mem_save(
        probe_incoming_light_color,
        probe_incoming_light_dirs,
        light_query_mapping,
        fake_outgoing_dirs,
        golden_normals,
        golden_diffuse_color,
        golden_specular_color,
        golden_specular_c,
    )[0]  # (P, 3)
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
        Path(model_params._model_path) / "toy_training_test_golden_image.png",
        rendered_image_rgb.permute(1, 2, 0).cpu().clip(0, 1),
    )

    # Instanciate the BRDF_normal_predictor
    brdf_normal_model = Tiny_BRDF_Normal_Predictor(
        global_image_height, global_image_width
    )
    brdf_normal_model = brdf_normal_model.cuda()
    brdf_normal_model.train()

    # Training Config
    loss_fn = nn.MSELoss()
    color_penalty = 0.5

    lr = 0.001
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

        # Collect Model Outputs
        Kd = nchw_tensor_to_npc(model_output["brdf"]["diffuse"])  # (1, P, 3)
        Ks = nchw_tensor_to_npc(model_output["brdf"]["specular"])  # (1, P, 3)
        spec_c = nchw_tensor_to_npc(
            model_output["brdf"]["specular_c"]
        ).squeeze(
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

        # TODO: Add in a functional camera position?
        # camera_pos = rendering_cam.camera_center.cuda()  # (3,)
        # outgoing_directions = nn.functional.normalize(
        #     (camera_pos - all_points_xyz), dim=1
        # )  # (P, 3)

        outgoing_radiance = batch_eval_blinn_phong_outgoing_radiance_with_probe_mem_save(
            probe_incoming_light_color,
            probe_incoming_light_dirs,
            light_query_mapping,
            fake_outgoing_dirs,
            camera_normals_normed,
            Kd,
            Ks,
            spec_c,
        )[0]  # (P, 3)

        outgoing_color_diff = batch_eval_blinn_phong_varying_outgoing_radiance_with_probe(
            probe_incoming_light_color,
            probe_incoming_light_dirs,
            probe_outgoing_light_color,
            probe_outgoing_light_dirs,
            light_query_mapping,
            camera_normals_normed,
            Kd,
            Ks,
            spec_c,
        ) # (P, 3)

        # Combine both active rendered image difference and outgoing color diff.
        # loss = loss_fn(outgoing_radiance, rendered_colors) + torch.mean(outgoing_color_diff)
        loss = torch.mean(outgoing_color_diff)

        loss.backward()


        optimizer.step()

        if step_num % brdf_args.image_reporting_interval == 0:
            # Report loss and other metrics
            tqdm.write(f"========Step {step_num}:========")
            tqdm.write(f"{outgoing_radiance = }")
            tqdm.write(f"{outgoing_color_diff = }")
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
        Path(model_params._model_path) / "toy_training_test_output_image.png",
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
    plt.savefig(Path(model_params._model_path) / "toy_training_comparison_graph.png")

    # All done
