from arguments import ModelParams, PipelineParams, OptimizationParams
from argparse import ArgumentParser
from extra_model_architectures import Blinn_Phong_BRDF
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
from typing import TypedDict, Any

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
        outgoing_directions: torch.Tensor,
        normals: torch.Tensor,
        diffuse_K: torch.Tensor,
        specular_K: torch.Tensor,
        spec_reflect_c: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inputs:
            incoming_light: (P, N, 3) R,G,B values of incoming light for each direction
            incoming_light_dirs: (N, 3) Directions oriented towards the light source in world space
            normal: (P, 3) Surface normals in world space for each of P points
            outgoing_directions: (P, 3) Outgoing (view) direction of radiance, in world space (head at the camera, tip at the surface point) for each point.
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
            outgoing_directions=outgoing_directions,
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
            outgoing_directions,
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
            outgoing_directions,
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
            outgoing_directions=outgoing_directions,
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
    incoming_light_colors: torch.Tensor,
    incoming_light_dirs: torch.Tensor,
    outgoing_directions: torch.Tensor,
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
        incoming_light_colors: (P, N, 3) RGB values of incoming light for each direction
        incoming_light_dirs: (N, 3) Normalized directions oriented towards the light source in world space
        normal: (P, 3) Surface normals in world space for each of P points
        outgoing_directions: (P, 3) Outgoing (view) normalized direction of radiance, in world space (head at the camera, tip at the surface point) for each point.
        diffuse_K: (P, 3) RGB values of diffuse coeffients (0 - 1 scale, but values can be greater or lower).
        specular_K: (P, 3) RGB values of specular coeffients (0 - 1).
        spec_reflect_c: (P,) Surface "shininess" specified as the specular exponent.

    Outputs:
        color: (P, 3) R,G,B values of outgoing radiance at each of P points as observed by the associated outgoing direction.
    """

    # Assertions for debugging.
    assert incoming_light_colors.dim() == 3
    assert incoming_light_colors.size(-1) == 3
    P, N, _ = incoming_light_colors.shape

    assert incoming_light_dirs.size(0) == N
    assert incoming_light_dirs.size(1) == 3

    assert normals.size(-1) == 3
    assert normals.size(0) == P
    assert outgoing_directions.is_same_size(normals)
    assert diffuse_K.is_same_size(normals)
    assert specular_K.is_same_size(diffuse_K)

    assert spec_reflect_c.dim() == 1
    assert spec_reflect_c.numel() == P

    # (P, N, 3) R,G,B values of lighting contributions at each of P points for all of the N directions
    outgoing_radiance: torch.Tensor = EvalBlinnPhongBRDF.apply(
        incoming_light_colors,
        incoming_light_dirs,
        outgoing_directions,
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


class BRDFModelOutput(TypedDict):
    diffuse: torch.Tensor
    specular: torch.Tensor
    specular_c: torch.Tensor


class FullModelOutput(TypedDict):
    # All tensors returned as shape (N, C, H, W).
    brdf: BRDFModelOutput
    normal: torch.Tensor


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
        self.normal_size = 2  # (x, y) components of tangent plane vector
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
            )
        )
        self.brdf_activation_function = nn.Softplus(beta=10)  # matches EVER method.
        self.spec_c_activation_function = nn.Sigmoid()
        self.normal_soft_cap = 20  # Soft capped from [-20, 20]

        # TODO: support multiple types of BRDFs eventually?

    def forward(self, image: torch.Tensor) -> FullModelOutput:
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

        # Extract each of our value types
        diffuse_brdf_features = img_features[:, : self.diffuse_brdf_size]
        specular_brdf_features = img_features[
            :,
            self.diffuse_brdf_size : self.diffuse_brdf_size + self.spec_brdf_size - 1,
        ]
        specular_c_features = img_features[
            :,
            self.diffuse_brdf_size
            + self.spec_brdf_size
            - 1 : self.diffuse_brdf_size
            + self.spec_brdf_size,
        ]
        normal_features = img_features[:, -self.normal_size :]

        # Apply Activation functions
        diffuse_brdf_features = self.brdf_activation_function(diffuse_brdf_features)
        specular_brdf_features = self.brdf_activation_function(specular_brdf_features)

        specular_c_features = (
            self.spec_c_activation_function(specular_c_features) * self.max_spec_c
        )  # Clips specular c value to be from 0 -> 16

        # "Soft Capping" Trick to prevent normals from growing too large: https://pytorch.org/blog/flexattention/
        # TODO: Can try replacing with linear activation function to see if there's a difference in performance
        normal_features = normal_features / self.normal_soft_cap
        normal_features = nn.functional.tanh(normal_features)
        normal_features = normal_features * self.normal_soft_cap

        # TODO: refine arguments
        # Best way to structure this...all as one tensor or as multiple?

        output_dict = {
            "brdf": {
                "diffuse": diffuse_brdf_features,
                # RGB + specular C
                "specular": specular_brdf_features,
                "specular_c": specular_c_features,
            },
            "normal": create_normals_from_tangent_space(
                normal_features
            ),  # TODO: don't do this automatically? Basically just adds an extra 1 dimension so it's not too bad though
        }

        return output_dict


def create_normals_from_tangent_space(tangent_normals: torch.Tensor) -> torch.Tensor:
    """
    Inputs:
        tangent_normals: (N, 2, [X, X...]) normal directions as (x, y) components in tangent space
    Outputs:
        normals: (N, 3, [X, X...]) normal directions
    """
    # https://learnopengl.com/Advanced-Lighting/Normal-Mapping
    # Normals are always pointing towards (0, 0, 1) in camera space
    z_normal_component = torch.ones_like(tangent_normals[:, :1])  # (N, 1)
    unnormed_normals = torch.cat((tangent_normals, z_normal_component), dim=1)
    return unnormed_normals


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


def batch_transform_normals_to_world_space(
    camera_normals: torch.Tensor, cameras_to_world_tensor: torch.Tensor
) -> torch.Tensor:
    """
    Inputs:
        camera_normals: (B, N, 3) normal directions in all of camera space, with the first dimension being len(cameras).
        camera_to_world_tensor: (B, 3, 3) stacked tensor giving the C2W rotation transformation from camera space to world space.

    Outputs:
        world_normals: (B, N, 3) normal directions in world space
    """

    # No scaling component of the camera to world matrix, can simply rotate the ray directions
    # without worrying about normal scaling issues
    world_normals = camera_normals.bmm(cameras_to_world_tensor)

    return world_normals


def get_stacked_camera_to_world_rotation_tensor(cameras: list[MiniCam]) -> torch.Tensor:
    """
    Get the camera to world rotation tensors stacked across the first dimension for a list of cameras.
    Inputs:
        cameras: list of B MiniCam objects representing the cameras.

    Outputs:
        cameras_to_world_tensor: (B, 3, 3) stacked tensor giving the C2W rotation transformation from camera space to world space.
    """
    camera_to_world_rot_list = [
        camera.world_view_transform[:3, :3].T for camera in cameras
    ]
    cameras_to_world_tensor = torch.stack(camera_to_world_rot_list, dim=0)  # (B, 3, 3)

    return cameras_to_world_tensor


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
    outgoing_dir = outgoing_dir.expand(P, 3).contiguous()
    outgoing_dir[0] = torch.Tensor(
        [0, 0, 1]
    ).cuda()  # If not handeled properly in Slang (safe operations), introduces a NaN.

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
