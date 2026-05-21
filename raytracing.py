from arguments import ModelParams, OptimizationParams
from scene import GaussianModel
from utils.system_utils import searchForMaxIteration

from scene.cameras import MiniCam

import torch


import os


def load_gaussian_model(
    model_params: ModelParams,
    opt_params: OptimizationParams,
    checkpoint: os.PathLike | None,
) -> GaussianModel:
    first_iter = 0
    gaussians = GaussianModel(
        model_params.sh_degree,
        model_params.use_neural_network,
        model_params.max_opacity,
    )

    if checkpoint is not None:
        gaussian_params, first_iter = torch.load(checkpoint)
        gaussians.restore(gaussian_params, opt_params)

    # Search for a valid EVER-comaptible .ply
    else:
        loaded_iter = searchForMaxIteration(
            os.path.join(model_params.model_path, "point_cloud")
        )
        print("Loading trained model at iteration {}".format(loaded_iter))

        gaussians.load_ply(
            os.path.join(
                model_params.model_path,
                "point_cloud",
                "iteration_" + str(loaded_iter),
                "point_cloud.ply",
            )
        )

    return gaussians


def render_gaussian_model(gaussians: GaussianModel, camera: MiniCam) -> torch.Tensor:
    bg_color = [1, 1, 1] if model_params.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    gaussians.training_setup(opt_params)
    torch.cuda.empty_cache()

    camera_index = 1
    # Get the cameras.json file that enumerates all the cameras and their positions
    rendering_cam = get_rendering_cam(model_params, camera_index)

    # Debug resulting camera
    # print(f"{rendering_cam.camera_center = }")
    # print(f"{rendering_cam.FoVy = }")
    # print(f"{rendering_cam.FoVx = }")
    # print(f"{rendering_cam.zfar = }")
    # print(f"{rendering_cam.znear = }")
    # print(f"{rendering_cam.image_height = }")
    # print(f"{rendering_cam.image_width = }")
    # print(f"{rendering_cam.world_view_transform = }")
    # print(f"{rendering_cam.full_proj_transform = }")

    preview_factor = 4
    image_width = rendering_cam.image_width
    image_height = rendering_cam.image_height

    rendering_cam.image_width = image_width // preview_factor
    rendering_cam.image_height = image_height // preview_factor

    print(f"{rendering_cam = }")

    renderer = FastRenderer(rendering_cam, gaussians, pipe_params.enable_GLO)
    renderer.set_camera(rendering_cam)


    # Just take in a renderer?
    # Gather incoming light?

    # Render sphere
    rays_o, rays_d = renderer.get_rays(rendering_cam)

    sphere_center_translation = torch.zeros((1, 3), device="cuda")
    avg_position = torch.mean(rays_o, dim=0).reshape(1, 3)
    avg_direction = torch.mean(rays_d, dim=0).reshape(1, 3)

    sphere_center = avg_position + avg_direction * 2
    # Trying a custom value for sphere center
    # sphere_center = torch.tensor([.12, -5.57, -7.3947], device="cuda").reshape(1,3)
    sphere_radius = 0.5
    # print(f"{sphere_center  = }")
    # print(f"{avg_position  = }")
    # print(f"{avg_direction  = }")

    sphere_intersect_st = time.time()
    T_vals = torch.full(
        (rays_o.size(0), 1), float("inf"), device="cuda"
    )  # Any non-intersections default to "inf" as their t value

    # Slang kernel params for launching processes
    block_size = 64
    num_pixels = rays_o.size(0)

    # Intersect the sphere
    kernels.intersect_sphere(
        ray_origins=rays_o,
        ray_directions=rays_d,
        sphere_center=sphere_center,
        sphere_radius=sphere_radius,
        T_values=T_vals,
    ).launchRaw(
        blockSize=(block_size, 1, 1), gridSize=(num_pixels // block_size + 1, 1, 1)
    )

    # T_vals = intersect_sphere(rays_o, rays_d, sphere_center, sphere_radius) # (h*w) x 1

    # Compute bounce rays
    bounce_ray_o = torch.full_like(rays_o, float("inf"))
    bounce_ray_d = torch.full_like(rays_d, 0)

    kernels.bounce_off_sphere(
        ray_origins=rays_o,
        ray_directions=rays_d,
        sphere_center=sphere_center,
        T_values=T_vals,
        bounce_ray_origins=bounce_ray_o,
        bounce_ray_directions=bounce_ray_d,
    ).launchRaw(
        blockSize=(block_size, 1, 1), gridSize=(num_pixels // block_size + 1, 1, 1)
    )

    # bounce_ray_o, bounce_ray_d = bounce_off_sphere(rays_o, rays_d, T_vals, sphere_center)
    print(f"Took {time.time() - sphere_intersect_st}s to intersect sphere")

    # See camera, spehre, and bounce rays if debugging.
    # plot_rays_and_sphere(rays_o, rays_d, sphere_center, sphere_radius, T_vals, bounce_ray_o, bounce_ray_d)

    # Render bounce rays
    # TODO: Can use rendering t_max for creating sphere?
    bounced_ray_output = renderer.trace_rays(
        bounce_ray_o, bounce_ray_d, rendering_cam, 0, 1e7
    )
    bounce_image = bounced_ray_output["color"][:, :3].T.reshape(
        3, rendering_cam.image_height, rendering_cam.image_width
    )

    T_vals = T_vals.reshape(
        1, rendering_cam.image_height, rendering_cam.image_width
    )  # (1, h, w)

    # Render quickly
    st = time.time()

    # TODO: Use opposite of t_min (t_max?) for creating chromesphere?
    # t_max parameter would need to become a tensor basically...kinda weird
    torch.cuda.empty_cache()
    base_image = renderer.render(
        rendering_cam, gaussians, background, include_depth=True
    )  # (4, h, w)
    color_image = base_image[:3, :, :]  # (3, h, w)
    depth_image = base_image[3, :, :]  # (h, w)

    # Add bounce lighting to the image
    masked_image = color_image * torch.isinf(
        T_vals
    )  # Mask out part that hits the sphere
    image = masked_image + bounce_image  # Add in bounce lighting

    # debug_depth_image(model_params, camera_index, rays_o, rays_d, color_image, depth_image)

    print(f"Took {time.time()-st}s to render frame.")

    # Save un-scaled base image for visualization
    base_image = (
        (torch.clamp(base_image, min=0, max=1.0) * 255)
        .byte()
        .permute(1, 2, 0)
        .contiguous()
        .cpu()
        .numpy()
    )

    base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)

    cv2.imwrite(
        Path(model_params.model_path) / f"base_image_camera_{camera_index}.png",
        base_image,
    )

    # Convert to 8-bit and save as png
    image = (
        (torch.clamp(image, min=0, max=1.0) * 255)
        .byte()
        .permute(1, 2, 0)
        .contiguous()
        .cpu()
        .numpy()
    )
    image = cv2.resize(image, (image_width, image_height))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    cv2.imwrite(
        Path(model_params.model_path)
        / f"chromesphere_output_camera_{camera_index}.png",
        image,
    )

    torch.cuda.empty_cache()
    return image
