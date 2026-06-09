from arguments import ModelParams, PipelineParams, OptimizationParams
from argparse import ArgumentParser
from neural_brdf import (
    BRDF_normal_predictor,
    Blinn_Phong_BRDF,
    transform_normals_to_world_space,
)
from raytracing import (
    build_gaussian_renderer,
    depth_map_to_xyz,
    gather_incoming_light_at_point,
    get_cameras,
    get_rendering_cam,
    load_gaussian_model,
    render_gaussians,
    generate_spherical_rays,
    gather_incoming_light_at_points,
)
from utils.general_utils import safe_state
import sys
from tqdm import tqdm

import torch
from torch import nn

from pathlib import Path
import os

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Manual Renderer Parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations", nargs="+", type=int, default=[7_000, 30_000]
    )
    parser.add_argument(
        "--save_iterations", nargs="+", type=int, default=[7_000, 30_000]
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    # args.checkpoint_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Load Gaussians
    model_params: ModelParams = lp.extract(args)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    gaussians = load_gaussian_model(
        model_params, op.extract(args), args.start_checkpoint
    )
    print(f"Loaded Gaussian, Active SH Degree: {gaussians.active_sh_degree}")

    rendering_cameras = get_cameras(model_params)
    num_cameras = len(rendering_cameras)

    # Set a global image width and height that is used for instanciating the neural network, etc.
    preview_factor = 4
    global_image_height = rendering_cameras[0].image_height // preview_factor
    global_image_width = rendering_cameras[0].image_width // preview_factor

    # Set up our initial renderer
    ever_renderer = build_gaussian_renderer(
        gaussians, rendering_cameras[0], pp.extract(args)
    )

    # More constants (affecting how much incoming light we use)
    incoming_light_sphere_divisions = 4

    # Calculate how big our incoming light features will be when input into our model.
    test_sphere_o, _ = generate_spherical_rays(
        torch.zeros((3,)), incoming_light_sphere_divisions
    )
    incoming_light_size = test_sphere_o.size(0) * 3  # N vectors that have [r, g, b]

    # Instanciate the BRDF_normal_predictor
    brdf_normal_model = BRDF_normal_predictor(
        global_image_height, global_image_width, incoming_light_size
    )
    brdf_normal_model = brdf_normal_model.cuda()
    brdf_normal_model.train()

    # Training Config
    loss_fn = nn.MSELoss()
    color_penalty = 0.5

    lr = 0.0001
    grad_norm_clip = 1
    optimizer = torch.optim.AdamW(
        brdf_normal_model.parameters(),
        lr=lr,
    )
    # optimizer = torch.optim.SGD(
    #     brdf_normal_model.parameters(),
    #     lr=lr,
    # )
    training_steps = 500
    point_batch_size = 32
    # TODO: Learning rate scheduler
    # TODO: etc, etc.

    # reserved_mem = torch.cuda.memory.memory_reserved() / 1024 / 1024 / 1024
    # allocated_mem = torch.cuda.memory.memory_allocated() / 1024 / 1024 / 1024
    # print(f"{reserved_mem = }")
    # Main training loop:
    for i in tqdm(range(training_steps)):
        optimizer.zero_grad()
        torch.cuda.empty_cache()

        # TODO: Have random Camera index and chosen point
        camera_index = 1

        # Set up the camera we're rendering with
        # TODO: Can just render all images from all cameras once, right?
        rendering_cam = rendering_cameras[camera_index]
        # print(f"{global_image_width = }")
        # print(f"{global_image_height = }")
        rendering_cam.image_width = global_image_width
        rendering_cam.image_height = global_image_height

        rendered_image = render_gaussians(
            ever_renderer, rendering_cam, None, include_depth=True
        )  # (C, H, W)
        # TODO: Save the image output for debug purposes?
        # print("Rendered Image.")

        # Separate RGB and Depth Images
        rgb_image = rendered_image[:3, :, :].permute(1, 2, 0)  # (H, W, C)
        depth_map = rendered_image[3, :, :]  # (H, W)

        # TODO: Iterate over all points?
        rand_row = torch.randint(0, global_image_height, (1,)).item()
        rand_col = torch.randint(0, global_image_width, (1,)).item()
        # chosen_point = (10, 10)
        chosen_point = (int(rand_row), int(rand_col))

        rendered_color = rgb_image[chosen_point]
        print(f"Color at {chosen_point}: {rendered_color}")

        # Get Incoming Light
        rays_o, rays_d = ever_renderer.get_rays(rendering_cam)
        xyz_map = depth_map_to_xyz(rays_o, rays_d, depth_map)  # (H, W, 3)

        # Querying Spherical Directions
        incoming_light_single, _, incoming_light_dirs_single = (
            gather_incoming_light_at_point(
                xyz_map[chosen_point],
                ever_renderer,
                tmin=0.01,
                sphere_divisions=incoming_light_sphere_divisions,
            )
        )

        # Ask for our BRDF values
        model_output = brdf_normal_model(
            rendered_image.unsqueeze(0), incoming_light_single
        )

        print(f"{model_output = }")

        Kd = model_output["brdf"]["diffuse"]
        # Ks = torch.tensor([0.2, 0.2, 0.2]).cuda()
        spec_c = torch.tensor(2.0).cuda()
        Ks = model_output["brdf"]["specular"]
        # spec_c = model_output["brdf"]["specular_c"]

        learned_brdf = Blinn_Phong_BRDF(Kd, Ks, spec_c)
        pred_normal = nn.functional.normalize(model_output["normal"])
        # pred_normal = nn.functional.normalize(torch.tensor([[0, 0, 1.0]]).cuda())

        world_normal = transform_normals_to_world_space(pred_normal, rendering_cam)
        print(f"{world_normal = }")
        print(f"{Kd = }")
        print(f"{Ks = }")
        print(f"{spec_c = }")

        # BRDF reconstruction - basic Diffuse BRDF with albedo
        camera_pos = rendering_cam.camera_center.cuda()  # (3,)

        outgoing_dir = nn.functional.normalize(
            (camera_pos - xyz_map[chosen_point]).reshape(1, 3)
        )

        outgoing_radiance = learned_brdf.construct_outgoing_radiance(
            incoming_light_single,
            incoming_light_dirs_single,
            outgoing_dir,
            world_normal,
        )
        # outgoing_radiance = Kd

        print(f"{outgoing_radiance = }")
        print(f"{rendered_color - outgoing_radiance = }")

        # Calculate loss and update
        loss = loss_fn(outgoing_radiance, rendered_color) + color_penalty * (
            torch.norm(Ks) + torch.norm(Kd)
        )
        print(f"{loss = }")
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(brdf_normal_model.parameters(), grad_norm_clip)

        parameters = brdf_normal_model.fc1.parameters()
        norm_type = 2
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]), norm_type)

        print(f"{total_norm = }")

        optimizer.step()

    # TODO: Save model, optimizer, and epoch for resuming
    model_checkpoint_dir = "brdf_models"
    model_save_path = (
        Path(model_params.model_path) / model_checkpoint_dir / "brdf_model.pt"
    )
    os.makedirs(model_save_path.parent, exist_ok=True)
    torch.save(brdf_normal_model.state_dict(), model_save_path)
    print(f"Saved model at {model_save_path.absolute()}")
    # All done
