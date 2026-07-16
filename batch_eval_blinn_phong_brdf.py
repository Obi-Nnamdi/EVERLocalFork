from typing import Any, Optional

import torch
from torch.autograd import Function

# Import and build our custom slangtorch kernel for evaluating BRDFs.
from pathlib import Path
import slangtorch

kernels = slangtorch.loadModule(
    str(Path(__file__).parent / "ever/splinetracers/slang/brdf_eval.slang")
)


class BatchEvalBlinnPhongBRDF(Function):
    """
    Evaluate the given Blinn-Phong BRDFs with the given incoming light, matching each point to its corresponding incoming light.
    """

    @staticmethod
    def forward(
        ctx: Any,
        probe_incoming_light: torch.Tensor,
        probe_incoming_light_dirs: torch.Tensor,
        incoming_light_probe_query: torch.Tensor,
        outgoing_directions: torch.Tensor,
        normals: torch.Tensor,
        diffuse_K: torch.Tensor,
        specular_K: torch.Tensor,
        spec_reflect_c: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inputs:
            probe_incoming_light: (P, N, 3) R,G,B values of incoming light for each direction in the light probe
            probe_incoming_light_dirs: (N, 3) Directions oriented towards the light source in world space (assumed to be the same for each point)
            incoming_light_probe_query: (B, HW) Correspondence between each of the HW points and their probe number.
            outgoing_directions: (B, HW, 3) Outgoing (view) direction of radiance, in world space (head at the camera, tip at the surface point) for each point.
            normals: (B, HW, 3) Surface normals in world space for each of HW points
            diffuse_K: (B, HW, 3)
            specular_K: (B, HW, 3)
            spec_reflect_c: (B, HW,)

        Outputs:
            color: (B, HW, N, 3) R,G,B values of lighting contributions at each of P points for all of the N directions for every camera
        """
        B, HW, _ = normals.shape
        _, N, _ = probe_incoming_light.shape
        output = torch.full((B, HW, N, 3), float("nan"), device="cuda") # using nan to take advantage of torch.nanmean

        brdf_eval_kernel = (
            kernels.eval_outgoing_radiance_blinn_phong_with_incoming_light_cache(
                probe_incoming_light=probe_incoming_light,
                probe_incoming_light_dirs=probe_incoming_light_dirs,
                incoming_light_probe_query=incoming_light_probe_query,
                outgoing_directions=outgoing_directions,
                normals=normals,
                diffuse_K=diffuse_K,
                specular_K=specular_K,
                spec_reflect_c=spec_reflect_c,
                output=output,
            )
        )

        # Max thread count is 1024 (32^2), higher values raise an error.
        # TODO: Worth exploring block size x-y tradeoffs? I.e. 64/16 vs 32/32.
        # https://forums.developer.nvidia.com/t/what-is-the-maximum-number-of-blocks-i-can-use/201587
        block_size_x = 2  # Batch dim
        block_size_y = 64  # Point / HW dim
        block_size_z = 8  # light dim
        brdf_eval_kernel.launchRaw(
            blockSize=(block_size_x, block_size_y, block_size_z),
            gridSize=(
                BatchEvalBlinnPhongBRDF.calc_grid_size(normals.shape[0], block_size_x),
                BatchEvalBlinnPhongBRDF.calc_grid_size(normals.shape[1], block_size_y),
                BatchEvalBlinnPhongBRDF.calc_grid_size(
                    probe_incoming_light.shape[1], block_size_z
                ),
            ),
        )

        # Save all inputs for our backward pass
        ctx.save_for_backward(
            probe_incoming_light,
            probe_incoming_light_dirs,
            incoming_light_probe_query,
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
            probe_incoming_light,
            probe_incoming_light_dirs,
            incoming_light_probe_query,
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
        brdf_eval_kernel_bwd = (
            kernels.eval_outgoing_radiance_blinn_phong_with_incoming_light_cache.bwd(
                probe_incoming_light=probe_incoming_light,
                probe_incoming_light_dirs=probe_incoming_light_dirs,
                incoming_light_probe_query=incoming_light_probe_query,
                outgoing_directions=outgoing_directions,
                normals=(normals, normals_grad),
                diffuse_K=(diffuse_K, diffuse_K_grad),
                specular_K=(specular_K, specular_K_grad),
                spec_reflect_c=(spec_reflect_c, spec_reflect_c_grad),
                output=(output, grad_output),
            )
        )

        block_size_x = 2  # Batch dim
        block_size_y = 64  # Point / HW dim
        block_size_z = 8  # light dim
        brdf_eval_kernel_bwd.launchRaw(
            blockSize=(block_size_x, block_size_y, block_size_z),
            gridSize=(
                BatchEvalBlinnPhongBRDF.calc_grid_size(normals.shape[0], block_size_x),
                BatchEvalBlinnPhongBRDF.calc_grid_size(normals.shape[1], block_size_y),
                BatchEvalBlinnPhongBRDF.calc_grid_size(
                    probe_incoming_light.shape[1], block_size_z
                ),
            ),
        )

        return (
            None,
            None,
            None,
            None,
            normals_grad,
            diffuse_K_grad,
            specular_K_grad,
            spec_reflect_c_grad,
        )

    @staticmethod
    def calc_grid_size(dim_size: int, block_size: int) -> int:
        return (dim_size + (block_size - 1)) // block_size


def batch_eval_blinn_phong_outgoing_radiance_with_probe(
    probe_incoming_light_colors: torch.Tensor,
    probe_incoming_light_dirs: torch.Tensor,
    incoming_light_probe_query: torch.Tensor,
    outgoing_directions: torch.Tensor,
    normals: torch.Tensor,
    diffuse_K: torch.Tensor,
    specular_K: torch.Tensor,
    spec_reflect_c: torch.Tensor,
    sub_batch_size: Optional[int] = None,
):
    """
    Evaluate the given Blinn-Phong BRDFs (specified with diffuse, specular, and specular_c coeffs) with the given incoming light probe,
    matching each point to its corresponding incoming light.

    Uses :class:`BatchEvalBlinPhongBRDF` in the backend.

    Inputs:
        probe_incoming_light: (P, N, 3) R,G,B values of incoming light for each direction in the light probe
        probe_incoming_light_dirs: (N, 3) Directions oriented towards the light source in world space (assumed to be the same for each point)
        incoming_light_probe_query: (B, HW) Correspondence between each of the HW points and their probe number.
        outgoing_directions: (B, HW, 3) Outgoing (view) direction of radiance, in world space (head at the camera, tip at the surface point) for each point.
        normals: (B, HW, 3) Surface normals in world space for each of HW points
        diffuse_K: (B, HW, 3)
        specular_K: (B, HW, 3)
        spec_reflect_c: (B, HW,)

    Outputs:
        color: (B, HW, 3) R,G,B values of outgoing radiance at each of P points as observed by the associated outgoing direction.
    """

    # Assertions for debugging.
    assert probe_incoming_light_colors.dim() == 3
    assert probe_incoming_light_colors.size(-1) == 3
    P, N, _ = probe_incoming_light_colors.shape

    assert probe_incoming_light_dirs.size(0) == N
    assert probe_incoming_light_dirs.size(1) == 3

    assert normals.size(-1) == 3
    B, HW, _ = normals.shape

    assert outgoing_directions.is_same_size(normals)
    assert diffuse_K.is_same_size(normals)
    assert specular_K.is_same_size(diffuse_K)

    assert spec_reflect_c.dim() == 2
    assert spec_reflect_c.size(0) == B

    if sub_batch_size is None:
        sub_batch_size = calc_optimal_batch_size_for_brdf_eval(N, HW)

    all_colors = torch.empty((B, HW, 3), dtype=torch.float, device="cuda")
    for i in range(0, B, sub_batch_size):
        # (SB, HW, N, 3) R,G,B values of lighting contributions at each of HW points for all of the N directions
        outgoing_radiance: torch.Tensor = BatchEvalBlinnPhongBRDF.apply(
            probe_incoming_light_colors,
            probe_incoming_light_dirs,
            incoming_light_probe_query[i : i + sub_batch_size],
            outgoing_directions[i : i + sub_batch_size],
            normals[i : i + sub_batch_size],
            diffuse_K[i : i + sub_batch_size],
            specular_K[i : i + sub_batch_size],
            spec_reflect_c[i : i + sub_batch_size],
        )  # pyright: ignore[reportAssignmentType]

        # Get Directions of light that didn't contribute and take masked mean (not nan) normal lighting directions (collapsing "N" dimension)
        all_colors[i: i + sub_batch_size] = torch.nanmean(outgoing_radiance, dim=2) # (SB, HW, 3)

    return all_colors


def calc_optimal_batch_size_for_brdf_eval(ray_dim: int, point_dim: int):
    torch.cuda.synchronize()
    free_vram, _ = torch.cuda.mem_get_info()

    # How much memory would a 1 batch size (1, HW, N, 3) tensor take up? Then we can see how big our batch size can be.
    non_batched_mem = point_dim * ray_dim * 3 * torch.float32.itemsize

    sub_batch_size = free_vram // (
        non_batched_mem * 4
    )  # Multiply non-batched mem to allow for overhead
    assert sub_batch_size > 0
    return sub_batch_size
