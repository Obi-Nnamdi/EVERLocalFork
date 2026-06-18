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
from typing import Any

from scene.cameras import MiniCam
from gaussian_renderer.ever import get_ray_directions
import torch
from torch import nn
from torch.autograd import Function
import math
import time

# Import and build our custom slangtorch kernel for evaluating BRDFs.
from pathlib import Path
import slangtorch

kernels = slangtorch.loadModule(
    str(Path(__file__).parent / "ever/splinetracers/slang/brdf_eval.slang")
)


class EvalBlinnPhongBRDF(Function):
    """
    Evaluate the given Blinn-Phong BRDFs with the given incoming light, matching each point to its corresponding incoming light.
    """

    @staticmethod
    def forward(
        ctx: Any,
        incoming_light: torch.Tensor,
        incoming_light_dirs: torch.Tensor,
        outgoing_dir: torch.Tensor,
        normals: torch.Tensor,
        diffuse_K: torch.Tensor,
        specular_K: torch.Tensor,
        spec_reflect_c: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inputs:
            incoming_light: (P, N, 3) R,G,B values of incoming light for each direction
            incoming_light_dirs: (P, N, 3) Directions oriented towards the light source in world space
            normal: (P, 3) Surface normals in world space for each of P points
            outgoing_dir: (1, 3) Outgoing (view) direction of radiance, in world space (head at the camera, tip at the surface point)
            diffuse_K: (P, 3)
            specular_K: (P, 3)
            spec_reflect_c: (P,)

        Outputs:
            color: (P, N, 3) R,G,B values of lighting contributions at each of P points for all of the N directions
        """
        # TODO: Remove the outer "P" dimension from incoming_light_dirs since light directions are always the same for each point
        # (Might not be worth it since we already get that large tensor from trace_rays, but could save memory potentially)
        # TODO: Fill with an arbitrary value that can be masked later (nan? inf?).
        output = torch.full_like(incoming_light, float("inf"))

        brdf_eval_kernel = kernels.eval_outgoing_radiance_blinn_phong(
            incoming_light=incoming_light,
            incoming_light_dirs=incoming_light_dirs,
            outgoing_dir=outgoing_dir,
            normals=normals,
            diffuse_K=diffuse_K,
            specular_K=specular_K,
            spec_reflect_c=spec_reflect_c,
            output=output,
        )

        # Max thread count is 1024 (32^2), higher values raise an error.
        # TODO: Worth exploring block size x-y tradeoffs? I.e. 64/16 vs 32/32.
        # https://forums.developer.nvidia.com/t/what-is-the-maximum-number-of-blocks-i-can-use/201587
        block_size_x = 64
        block_size_y = 16
        brdf_eval_kernel.launchRaw(
            blockSize=(block_size_x, block_size_y, 1),
            gridSize=(
                EvalBlinnPhongBRDF.calc_block_size(
                    incoming_light.shape[0], block_size_x
                ),
                EvalBlinnPhongBRDF.calc_block_size(
                    incoming_light.shape[1], block_size_y
                ),
                1,
            ),
        )

        # Save all inputs for our backward pass
        ctx.save_for_backward(
            incoming_light,
            incoming_light_dirs,
            outgoing_dir,
            normals,
            diffuse_K,
            specular_K,
            spec_reflect_c,
            output,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # TODO: Might need to clone grad_output?
        # Note: When using DiffTensorView, grad_output gets 'consumed' during the reverse-mode.
        # If grad_output may be reused, consider calling grad_output = grad_output.clone()
        (
            incoming_light,
            incoming_light_dirs,
            outgoing_dir,
            normals,
            diffuse_K,
            specular_K,
            spec_reflect_c,
            output,
        ) = ctx.saved_tensors

        # Create gradients for all tensors that have them (BRDF and normal parameters)
        normals_grad = torch.zeros_like(normals)
        diffuse_K_grad = torch.zeros_like(diffuse_K)
        specular_K_grad = torch.zeros_like(specular_K)
        spec_reflect_c_grad = torch.zeros_like(spec_reflect_c)

        # Create backwards kernel and run it
        brdf_eval_kernel_bwd = kernels.eval_outgoing_radiance_blinn_phong.bwd(
            incoming_light=incoming_light,
            incoming_light_dirs=incoming_light_dirs,
            outgoing_dir=outgoing_dir,
            normals=(normals, normals_grad),
            diffuse_K=(diffuse_K, diffuse_K_grad),
            specular_K=(specular_K, specular_K_grad),
            spec_reflect_c=(spec_reflect_c, spec_reflect_c_grad),
            output=(output, grad_output),
        )

        block_size_x = 64
        block_size_y = 16
        brdf_eval_kernel_bwd.launchRaw(
            blockSize=(block_size_x, block_size_y, 1),
            gridSize=(
                EvalBlinnPhongBRDF.calc_block_size(
                    incoming_light.shape[0], block_size_x
                ),
                EvalBlinnPhongBRDF.calc_block_size(
                    incoming_light.shape[1], block_size_y
                ),
                1,
            ),
        )

        return (
            None,
            None,
            None,
            normals_grad,
            diffuse_K_grad,
            specular_K_grad,
            spec_reflect_c_grad,
        )

    @staticmethod
    def calc_block_size(dim_size: int, block_size: int) -> int:
        return (dim_size + (block_size - 1)) // block_size


def eval_blinn_phong_outgoing_radiance(
    incoming_light: torch.Tensor,
    incoming_light_dirs: torch.Tensor,
    outgoing_dir: torch.Tensor,
    normals: torch.Tensor,
    diffuse_K: torch.Tensor,
    specular_K: torch.Tensor,
    spec_reflect_c: torch.Tensor,
):
    """
    Evaluate the given Blinn-Phong BRDFs (specified with diffuse, specular, and specular_c coeffs) with the given incoming light,
    matching each point to its corresponding incoming light.

    Uses :class:`EvalBlinPhongBRDF` in the backend.

    Inputs:
        incoming_light: (P, N, 3) RGB values of incoming light for each direction
        incoming_light_dirs: (P, N, 3) Directions oriented towards the light source in world space
        normal: (P, 3) Surface normals in world space for each of P points
        outgoing_dir: (1, 3) Outgoing (view) direction of radiance, in world space (head at the camera, tip at the surface point)
        diffuse_K: (P, 3) RGB values of diffuse coeffients (0 - 1).
        specular_K: (P, 3) RGB values of specular coeffients (0 - 1).
        spec_reflect_c: (P,) Surface "shininess" specified as the specular exponent.

    Outputs:
        color: (P, 3) R,G,B values of outgoing radiance at each of P points as observed by outgoing_dir
    """

    # (P, N, 3) R,G,B values of lighting contributions at each of P points for all of the N directions
    outgoing_radiance: torch.Tensor = EvalBlinnPhongBRDF.apply(
        incoming_light,
        incoming_light_dirs,
        outgoing_dir,
        normals,
        diffuse_K,
        specular_K,
        spec_reflect_c,
    )  # pyright: ignore[reportAssignmentType]

    # Get Directions of light that didn't contribute (not on same side as normal)
    outgoing_radiance_inf_mask = outgoing_radiance.isposinf()
    positive_masked_radiance = outgoing_radiance.masked_fill(
        outgoing_radiance_inf_mask, 0.0
    )  # (P, N, 3)

    # Take masked mean across positive (not inf) normal lighting directions (collapsing "N" dimension)
    return torch.sum(positive_masked_radiance, dim=1) / torch.sum(
        ~outgoing_radiance_inf_mask, dim=1
    )


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
        Inputs:
            Kd: (P, 3)
            Ks: (P, 3)
            spec_reflect_c: (P,)
        """

        P = Kd.size(0)
        assert Kd.size(1) == 3
        assert Ks.size(1) == 3
        assert spec_reflect_c.size(0) == P

        self.Kd = Kd  # (P, 3)
        self.Ks = Ks  # (P, 3)
        # Control spectular reflection size
        self.spec_reflect_c = spec_reflect_c  # (P,)

    def compute_diffuse(
        self,
        incoming_light: torch.Tensor,
        incoming_light_dirs: torch.Tensor,
        normal: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inputs:
            incoming_light: (N, 3) R,G,B values of incoming light for each direction
            incoming_light_dirs: (N, 3) Directions oriented towards the light source in world space
            normal: (P, 3) Surface normals in world space for each of P points
        Outputs:
            diffuse_color: (N, P, 3) R,G,B values of diffuse lighting contributions at each of P points for all of the N directions
        """

        # Compute dot products of normal and all light directions
        diffuse_terms = incoming_light_dirs @ normal.T  # (N, 3) @ (3, P) --> (N, P)
        diffuse_terms = diffuse_terms.unsqueeze(-1)  # (N, P, 1)

        # TODO: Don't want negative dot product values (light directions are spherical), but I'm choosing to not clamp for now
        # (could uncomment this line, but I assume everything passed in has a positive dot product)
        # light_intensity_dot_products = torch.clamp(light_intensity_dot_products, min = 0)

        # Combine diffuse color of material with the incoming light
        N = incoming_light.size(0)
        P = normal.size(0)
        expanded_kd = self.Kd.unsqueeze(0).expand(N, -1, -1)  # (N (new), P, 3)
        expanded_incoming_light = incoming_light.unsqueeze(1).expand(
            -1, P, -1
        )  # (N, P (new), 3)

        diffuse_intensity = expanded_kd * expanded_incoming_light  # (N, P, 3)

        # Final Diffuse colors for each light source
        return diffuse_terms * diffuse_intensity  # (N, P, 3)

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
        normal: (P, 3) Surface normal in world space
        outgoing_dir: (1, 3) Outgoing (view) direction of radiance, in world space (head at the camera, tip at the surface point)

        Outputs:
            specular_color: (N, P, 3) R,G,B values of specular lighting contributions at each of P points for all of the N directions
        """

        N = incoming_light.size(0)
        P = normal.size(0)

        # Compute specular term
        halfway_vecs = torch.nn.functional.normalize(
            (incoming_light_dirs + outgoing_dir)
        )  # (N, 3)

        # Compute dot products of normal and all halfway directions
        specular_term_intermediate = (
            halfway_vecs @ normal.T
        )  # (N, 3) @ (3, P) --> (N, P)
        specular_term = torch.pow(
            specular_term_intermediate,
            self.spec_reflect_c.unsqueeze(0).expand(N, -1),
        )  # (N, P), [specular exp reshaped to (N, P)]
        specular_term = specular_term.unsqueeze(-1)  # (N, P, 1)

        # TODO: Don't want negative dot product values (light directions are spherical), but I'm choosing to not clamp for now
        # (could uncomment this line, but I assume everything passed in has a positive dot product)
        # light_intensity_dot_products = torch.clamp(light_intensity_dot_products, min = 0)

        # Combine diffuse color of material with the incoming light

        specular_intensity = self.Ks.unsqueeze(0).expand(
            N, -1, -1
        ) * incoming_light.unsqueeze(1).expand(
            -1, P, -1
        )  # (N (new), P, 3) * (N, P (new), 3)

        # Final colors for each light source
        return specular_term * specular_intensity  # (N, P, 1) * (N, P, 3) => (N, P, 3)

    # TODO: Construct as a general function
    def construct_outgoing_radiance(
        self,
        incoming_light: torch.Tensor,
        incoming_light_dirs: torch.Tensor,
        outgoing_dir: torch.Tensor,
        normals: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inputs:
            incoming_light: (P, N, 3) R,G,B values of incoming light for each direction
            incoming_light_dirs: (P, N, 3) Directions oriented towards the light source in world space
            normal: (P, 3) Surface normals in world space for each of P points
            outgoing_dir: (1, 3) Outgoing (view) direction of radiance, in world space (head at the camera, tip at the surface point)

        Outputs:



        Returns color as a (P, 3) tensor for every point.

        Sources:

        https://en.wikipedia.org/wiki/Rendering_equation
        https://15462.courses.cs.cmu.edu/spring2024content/lectures/15_brdfs/15_brdfs_slides.pdf
        https://www.scratchapixel.com/lessons/3d-basic-rendering/phong-shader-BRDF/phong-illumination-models-brdf.html
        """

        diffuse_component = self.compute_diffuse(
            incoming_light, incoming_light_dirs, normals
        )
        specular_component = self.compute_specular(
            incoming_light,
            incoming_light_dirs,
            normals,
            outgoing_dir,
        )

        # TODO: Proper intergration of diffuse and specular terms across all the lights? Should I just average?
        # Should I take weighted average w/ something else? (i.e. normal weighting)

        # Simple Diffuse + Specular combination
        full_lighting = diffuse_component + specular_component  # (N, P, 3)

        # Compute light attenuation based on viewing direction as found in rendering equation
        # <w_i, n>
        light_weakening_factors = (
            incoming_light_dirs @ normals.T
        )  # (N, 3) @ (3, P) --> (N, P)
        light_weakening_factors = light_weakening_factors.unsqueeze(-1)  # (N, P, 1)

        # Only choose light sources that are on the same side as the surface normal.
        positive_light_contributions = light_weakening_factors >= 0  # (N, P, 1)

        # We now only keep the positive lighting contributions (same side hemisphere) out of all the lighting contributions
        positive_masked_lighting = full_lighting.masked_fill(
            ~positive_light_contributions, 0.0
        )  # (N, P, 3)

        # Manually take the mean across masked elements
        return torch.sum(positive_masked_lighting, dim=0) / torch.sum(
            positive_light_contributions, dim=0
        )  # (P, 3)
        # TODO: Might be worth looking into the dw_i term in the rendering equation and maybe thinking of just multipling a sort of area patch term?


class BRDF_normal_predictor(nn.Module):

    def __init__(self, img_height: int, img_width: int) -> None:
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

        self.diffuse_brdf_size = 3
        self.spec_brdf_size = 4
        self.normal_size = 3
        self.max_spec_c = 16

        # Layer Norm before output
        # self.norm1 = nn.LayerNorm(self.img_height * self.img_width * 4)

        # Final up-res into everything we're predicting
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=self.diffuse_brdf_size
                + self.spec_brdf_size
                + self.normal_size,
                kernel_size=3,
                padding="same",
            ),
            # nn.Sigmoid(),
            nn.Softplus(beta=10),
            # nn.LeakyReLU(0.01),
            # nn.ReLU(),
        )

        # TODO: support multiple types of BRDFs eventually?

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Input:
            image: (N, C, H, W)

        Output:
            brdf:
                diffuse: (N, 3, H, W)
                specular: (N, 3, H, W)
                specular_c: (N, 1, H, W)
            normal:
                (N, 3, H, W)
        """

        N = image.size(0)

        img_features = self.conv1(image)
        img_features = self.conv2(img_features)
        img_features = self.conv3(img_features)
        # TODO: Use a norm?

        # TODO: Activation functions for these values?
        # "Soft Capping" Trick: https://pytorch.org/blog/flexattention/
        # TODO: Remove softplus for this exp? We want it to be positive so I don't mind keeping it.
        # spec_c = brdf_predictions[:, -1]
        # spec_c = spec_c / self.max_spec_c
        # spec_c = nn.functional.tanh(spec_c)
        # spec_c = spec_c * self.max_spec_c

        # TODO: refine arguments
        # Best way to structure this...all as one tensor or as multiple?
        output_dict = {
            "brdf": {
                "diffuse": img_features[:, : self.diffuse_brdf_size],
                # RGB + specular C
                "specular": img_features[
                    :,
                    self.diffuse_brdf_size : self.diffuse_brdf_size
                    + self.spec_brdf_size
                    - 1,
                ],
                "specular_c": img_features[
                    :,
                    self.diffuse_brdf_size
                    + self.spec_brdf_size
                    - 1 : self.diffuse_brdf_size
                    + self.spec_brdf_size,
                ],
            },
            "normal": img_features[:, -self.normal_size :],
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
    # Basic Test to ensure Neural net works:
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
        rendering_cam.image_height, rendering_cam.image_width
    )
    normal_predictor = normal_predictor.cuda()
    output = normal_predictor(
        rendered_image.unsqueeze(0)
    )  # Kd, Ks, Normal are (N, 3, H, W)

    chosen_point = ((0, 0), (0, 0))  # P = 2

    Kd = output["brdf"]["diffuse"][
        0, :, chosen_point[0], chosen_point[1]
    ].T.clone()  # (P, 3) [clone to allow retaining of gradients]
    Ks = output["brdf"]["specular"][
        0, :, chosen_point[0], chosen_point[1]
    ].T.clone()  # (P, 3)
    spec_c = output["brdf"]["specular_c"][
        0, :, chosen_point[0], chosen_point[1]
    ].T.squeeze(
        1
    )  # (P,)
    normal = output["normal"][
        0, :, chosen_point[0], chosen_point[1]
    ].T.clone()  # (P, 3)

    # print(f"{output = }")
    print(f"{Kd = }")
    print(f"{Kd.shape  = }")
    print(f"{Ks.shape  = }")
    print(f"{spec_c.shape  = }")
    print(f"{normal.shape  = }")

    learned_brdf = Blinn_Phong_BRDF(Kd, Ks, spec_c)
    normal = nn.functional.normalize(normal)

    world_normal = transform_normals_to_world_space(normal, rendering_cam)

    # test_normal_transformation(transform_normals_to_world_space, rendering_cam, rays_d)

    torch.cuda.synchronize()
    start_time = time.process_time()
    outgoing_radiance_pytorch = learned_brdf.construct_outgoing_radiance(
        incoming_light, incoming_light_dirs, outgoing_dir, world_normal
    )

    torch.cuda.synchronize()
    print(f"Radiance (Pytorch) completed in {time.process_time() - start_time:.6f}s")

    P = Kd.size(0)
    N = incoming_light.size(0)

    # Same incoming light for every single point, similar to what we have now.
    incoming_light = incoming_light.unsqueeze(0).expand(P, N, 3).contiguous()
    incoming_light_dirs = incoming_light_dirs.unsqueeze(0).expand(P, N, 3).contiguous()

    torch.cuda.synchronize()
    start_time = time.process_time()
    outgoing_radiance_slang = eval_blinn_phong_outgoing_radiance(
        incoming_light,
        incoming_light_dirs,
        outgoing_dir,
        world_normal,
        Kd,
        Ks,
        spec_c,
    )
    torch.cuda.synchronize()
    print(f"Radiance (Slang) completed in {time.process_time() - start_time:.6f}s")

    # Test forward pass computation between slang and pytorch versions
    print(f"{outgoing_radiance_pytorch = }")
    print(f"{outgoing_radiance_slang = }")
    print(f"{outgoing_radiance_slang.shape = }")
    print(f"{outgoing_radiance_pytorch - outgoing_radiance_slang = }")

    Kd.retain_grad()
    Ks.retain_grad()
    normal.retain_grad()
    # Test gradient computation (slang)
    avg_radiance = torch.mean(outgoing_radiance_slang)
    avg_radiance.backward(retain_graph=True)
    print(f"Slang Grad: {Kd.grad.cpu() = }")
    print(f"Slang Grad: {Ks.grad.cpu() = }")
    print(f"Slang Grad: {normal.grad.cpu() = }")

    # Clear gradients
    Kd.grad = None
    Ks.grad = None
    normal.grad = None

    # Test gradient computation (pytorch)
    avg_radiance = torch.mean(outgoing_radiance_pytorch)
    avg_radiance.backward()
    print(f"Pytorch Grad: {Kd.grad.cpu() = }")
    print(f"Pytorch Grad: {Ks.grad.cpu() = }")  # Gives a NaN result....
    print(f"Pytorch Grad: {normal.grad.cpu() = }")

    # Test loss calculation...
    print(f"{outgoing_radiance_slang - rendered_color = }")

    loss_fn = nn.MSELoss()
    loss = loss_fn(outgoing_radiance_slang, rendered_color)
    print(f"{loss = }")

    # All done
