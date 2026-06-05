from arguments import ModelParams, PipelineParams, OptimizationParams
from argparse import ArgumentParser
from raytracing import (
    build_gaussian_renderer,
    depth_map_to_xyz,
    gather_incoming_light_at_point,
    get_rendering_cam,
    load_gaussian_model,
    render_gaussians,
)
from utils.general_utils import safe_state
import sys

from scene.cameras import MiniCam
from gaussian_renderer.ever import get_ray_directions
import torch
from torch import nn
import math

# TODO: Could be turned into a more general BRDF class when implementing more complex models
class Blinn_Phong_BRDF:
    """
    Implemented from here: https://rodolphe-vaillant.fr/entry/85/phong-illumination-model-cheat-sheet
    """

    def __init__(
        self, Kd: torch.Tensor, Ks: torch.Tensor, spec_reflect_c: torch.Tensor
    ) -> None:
        """
        Initialize a Blinn Phong BRDF with Diffuse, Specular, and specular reflection size values.
        """
        self.Kd = Kd
        self.Ks = Ks
        # Control spectular reflection size
        self.spec_reflect_c = spec_reflect_c

    def compute_diffuse(
        self,
        incoming_light: torch.Tensor,
        incoming_light_dirs: torch.Tensor,
        normal: torch.Tensor,
    ) -> torch.Tensor:
        """
        incoming_light: (N, 3) R,G,B values of incoming light for each direction
        incoming_light_dirs: (N, 3) Directions oriented towards the light source in world space
        normal: (1, 3) Surface normal in world space
        """

        # Compute dot products of normal and all light directions
        diffuse_term = torch.sum(
            incoming_light_dirs * normal, dim=1, keepdim=True
        )  # (N, 1)

        # TODO: Don't want negative dot product values (light directions are spherical), but I'm choosing to not clamp for now
        # (could uncomment this line, but I assume everything passed in has a positive dot product)
        # light_intensity_dot_products = torch.clamp(light_intensity_dot_products, min = 0)

        # Combine diffuse color of material with the incoming light
        diffuse_intensity = self.Kd * incoming_light  # (N, 3)

        # Final Diffuse colors for each light source
        return diffuse_term * diffuse_intensity

    def compute_specular(
        self,
        incoming_light: torch.Tensor,
        incoming_light_dirs: torch.Tensor,
        normal: torch.Tensor,
        outgoing_dir: torch.Tensor,
    ) -> torch.Tensor:
        """
        Uses Blinn model of specular reflection.

        incoming_light: (N, 3) R,G,B values of incoming light for each direction
        incoming_light_dirs: (N, 3) Directions oriented towards the light source in world space
        normal: (1, 3) Surface normal in world space
        outgoing_dir: (1, 3) Outgoing (view) direction of radiance, in world space (head at the camera, tip at the surface point)
        """

        # Compute specular term
        halfway_vecs = torch.nn.functional.normalize(
            (incoming_light_dirs + outgoing_dir)
        )  # (N, 3)

        specular_term = torch.pow(
            torch.sum(halfway_vecs * normal, dim=1, keepdim=True), self.spec_reflect_c
        )  # (N, 1)

        # TODO: Don't want negative dot product values (light directions are spherical), but I'm choosing to not clamp for now
        # (could uncomment this line, but I assume everything passed in has a positive dot product)
        # light_intensity_dot_products = torch.clamp(light_intensity_dot_products, min = 0)

        # Combine diffuse color of material with the incoming light
        specular_intensity = self.Ks * incoming_light  # (N, 3)

        # Final Diffuse colors for each light source
        return specular_term * specular_intensity

    # TODO: Construct as a general function
    def construct_outgoing_radiance(
        self,
        incoming_light: torch.Tensor,
        incoming_light_dirs: torch.Tensor,
        outgoing_dir: torch.Tensor,
        normal_dir: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns color as a (3,) tensor.

        Sources:

        https://en.wikipedia.org/wiki/Rendering_equation
        https://15462.courses.cs.cmu.edu/spring2024content/lectures/15_brdfs/15_brdfs_slides.pdf
        https://www.scratchapixel.com/lessons/3d-basic-rendering/phong-shader-BRDF/phong-illumination-models-brdf.html
        """
        # Compute light attenuation based on viewing direction as found in rendering equation
        # <w_i, n>
        light_weakening_factors = torch.sum(incoming_light_dirs * normal_dir, dim=1)

        # Only choose light sources that are on the same side as the surface normal:
        # I guess averaging should be done across all positive surface-normal * light direction dot products?
        positive_light_contributions = light_weakening_factors >= 0

        relevant_incoming_light = incoming_light[positive_light_contributions, :]
        relevant_incoming_light_dirs = incoming_light_dirs[
            positive_light_contributions, :
        ]

        # TODO: is this needed?
        relevant_light_weakening_factors = light_weakening_factors[positive_light_contributions]

        # print(f"{relevant_incoming_light = }")
        # print(f"{relevant_incoming_light_dirs = }")

        diffuse_component = self.compute_diffuse(
            relevant_incoming_light, relevant_incoming_light_dirs, normal_dir
        )
        specular_component = self.compute_specular(
            relevant_incoming_light,
            relevant_incoming_light_dirs,
            normal_dir,
            outgoing_dir,
        )

        # TODO: Proper intergration of diffuse and specular terms across all the lights? Should I just average?
        # Should I take weighted average w/ something else? (i.e. normal weighting)

        # Simple Diffuse + Specular combination
        full_lighting = diffuse_component + specular_component

        # Average result to get our output
        # TODO: Might be worth looking into the dw_i term in the rendering equation and maybe thinking of just multipling a sort of area patch term?
        # Really not sure
        return torch.mean(full_lighting, dim=0)


class BRDF_normal_predictor(nn.Module):

    def __init__(
        self, img_height: int, img_width: int, incoming_light_size: int
    ) -> None:
        super().__init__()

        # Take in an RGB-D image and run multiple conv-net layers on it
        # TODO: Add dropouts, pooling, etc.
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=28, kernel_size=3, padding="same"),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=28, out_channels=4, kernel_size=3, padding="same"),
            nn.ReLU(),
        )
        # TODO: Define basic architecture

        self.img_height = img_height
        self.img_width = img_width
        self.incoming_light_size = incoming_light_size

        self.diffuse_brdf_size = 3
        self.spec_brdf_size = 4
        self.normal_size = 3

        # One network for the BRDFs, another for the normals.
        self.fc1 = nn.Sequential(
            nn.Linear(
                in_features=(self.img_height * self.img_width * 4)
                + self.incoming_light_size,
                out_features=self.diffuse_brdf_size + self.spec_brdf_size,
            ),
            nn.Softplus(beta=10),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(
                in_features=(self.img_height * self.img_width * 4)
                + self.incoming_light_size,
                out_features=self.normal_size,
            )
        )
        # TODO: support multiple types of BRDFs eventually?

    def forward(
        self, image: torch.Tensor, incoming_light: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Input:
            image: (N, C, H, W)
            incoming_light: (R, C) for one point
            TODO: support incoming light for multiple points?

        """

        img_features = self.conv1(image)
        img_features = self.conv2(img_features)

        img_features = img_features.reshape(1, -1)  # (N, C * H * W)

        # Concatenate incoming light to be used for MLP
        # There's only one pixel so for now this is how it's being used,
        # But should be explored how to improve this.
        # TODO: Transform incoming light in some way?
        img_and_light_features = torch.cat(
            [img_features, incoming_light.reshape(1, -1)], dim=1
        )

        brdf_predictions = self.fc1(img_and_light_features)
        normal_predictions = self.fc2(img_and_light_features)

        # TODO: refine arguments
        # Best way to structure this...all as one tensor or as multiple?
        output_dict = {
            "brdf": {
                "diffuse": brdf_predictions[:, : self.diffuse_brdf_size],
                # RGB + specular C
                "specular": brdf_predictions[
                    :,
                    self.diffuse_brdf_size : self.diffuse_brdf_size
                    + self.spec_brdf_size,
                ],
            },
            "normal": normal_predictions,
        }

        return output_dict


def transform_normals_to_world_space(
    camera_normals: torch.Tensor, camera: MiniCam
) -> torch.Tensor:
    """
    Inputs:
        camera_normals: (N, 3) normal directions in camera space
        camera: MiniCam object representing the camera.

    Outputs:
        world_normal: (N, 3) normal directions in world space
    """
    # Pulling code from "get_rays" in gaussian_renderer/ever.py.
    world_to_camera_rot = camera.world_view_transform[:3, :3]

    camera_to_world_rot = world_to_camera_rot.T

    # No scaling component of the camera to world matrix, can simply rotate the ray directions
    # without worrying about normal scaling issues
    world_normals = camera_normals @ camera_to_world_rot

    return world_normals


def test_normal_transformation(view: MiniCam, rays_d: torch.Tensor):
    """
    Compare the output of `transform_normals_to_world_space` to the actual camera rays used to render.
    Mainly just a sanity check.
    """
    # Adapted from "camera2rays" in gaussian_renderer/ever.py
    w = view.image_width
    h = view.image_height

    fx = 0.5 * w / math.tan(0.5 * view.FoVx)  # original focal length
    fy = 0.5 * h / math.tan(0.5 * view.FoVy)  # original focal length
    directions = get_ray_directions(h, w, [fx, fy]).cuda()
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    directions = directions.view(-1, 3)

    transformed_directions = transform_normals_to_world_space(directions, rendering_cam)

    print(f"{directions[0:10] = }")
    print(f"{transformed_directions[0:10] = }")
    print(f"{rays_d[0:10] = }")


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

    camera_index = 1
    rendering_cam = get_rendering_cam(model_params, camera_index)

    # Reduce Camera resolution
    preview_factor = 4
    image_width = rendering_cam.image_width
    image_height = rendering_cam.image_height

    rendering_cam.image_width = image_width // preview_factor
    rendering_cam.image_height = image_height // preview_factor

    # Build Renderer and Render
    renderer = build_gaussian_renderer(gaussians, rendering_cam, pp.extract(args))
    rendered_image = render_gaussians(
        renderer, rendering_cam, None, include_depth=True
    )  # (C, H, W)
    print("Rendered Image.")

    # Separate RGB and Depth Images
    rgb_image = rendered_image[:3, :, :].permute(1, 2, 0)  # (H, W, C)
    depth_map = rendered_image[3, :, :]  # (H, W)

    chosen_point = (0, 0)

    rendered_color = rgb_image[chosen_point]
    print(f"Color at {chosen_point}: {rendered_color}")

    # Get Incoming Light
    # TODO: How to handle any t_max occurences? Will have to look into that
    # since this alone doesn't work (maybe a threshold?).
    # print("T_MAX Depth Map:")
    # print(torch.sum(depth_map == 1e7))
    # TODO: Maybe simplify into another helper?
    rays_o, rays_d = renderer.get_rays(rendering_cam)
    xyz_map = depth_map_to_xyz(rays_o, rays_d, depth_map)

    # Querying Spherical Directions
    incoming_light, _, incoming_light_dirs = gather_incoming_light_at_point(
        xyz_map[chosen_point], renderer, tmin=0.01, sphere_divisions=4
    )

    # print(f"{incoming_light = }")
    # print(f"{incoming_light_dirs = }")

    # BRDF reconstruction - basic Diffuse BRDF with albedo
    camera_pos = rays_o[0, :]  # (3,)

    outgoing_dir = torch.nn.functional.normalize(
        (camera_pos - xyz_map[chosen_point]).reshape(1, 3)
    )

    normal_predictor = BRDF_normal_predictor(
        rendering_cam.image_height,
        rendering_cam.image_width,
        incoming_light.size(0) * incoming_light.size(1),
    )
    normal_predictor = normal_predictor.cuda()
    output = normal_predictor(rendered_image.unsqueeze(0), incoming_light)

    Kd = output["brdf"]["diffuse"]
    Ks = output["brdf"]["specular"][:, :3]
    spec_c = output["brdf"]["specular"][:, 3]

    print(f"{output = }")

    learned_brdf = Blinn_Phong_BRDF(Kd, Ks, spec_c)
    normal = nn.functional.normalize(output["normal"])
    
    world_normal = transform_normals_to_world_space(normal, rendering_cam)

    # test_normal_transformation(transform_normals_to_world_space, rendering_cam, rays_d)

    outgoing_radiance = learned_brdf.construct_outgoing_radiance(
        incoming_light, incoming_light_dirs, outgoing_dir, world_normal
    )

    print(f"{outgoing_radiance = }")

    print(f"{outgoing_radiance - rendered_color = }")

    # TODO: Calculate loss and such...
    loss_fn = nn.MSELoss()
    loss = loss_fn(outgoing_radiance, rendered_color)
    print(f"{loss = }")

    # TODO: Zero grad, take a step, update, etc.

    # All done
