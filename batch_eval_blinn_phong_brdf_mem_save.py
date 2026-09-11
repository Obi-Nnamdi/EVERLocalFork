from typing import Any, cast

import torch
from torch.autograd import Function

# Import and build our custom slangtorch kernel for evaluating BRDFs.

from slangtorch_kernel_compilation import (
    MAX_INCOMING_LIGHT_DIRECTIONS_FOR_LOOP_EVAL,
    MAX_NUMEL_FOR_SLANGTORCH,
    USE_CHECKPOINTING_FOR_INCOMING_LIGHT_PROBE_BACKWARD_PASS,
    brdf_eval_kernels,
)

class BatchEvalBlinnPhongBRDFMemSave(Function):
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
        output = torch.zeros(
            (B, HW, 3), device="cuda", dtype=torch.float
        )  # using nan to take advantage of torch.nanmean

        # TODO: Tracing an error where having a batch size that's bigger or around this value can cause errors because the tensor can't be populated
        # (uint saturation?)
        assert output.numel() < MAX_NUMEL_FOR_SLANGTORCH

        brdf_eval_kernel = brdf_eval_kernels.eval_outgoing_radiance_blinn_phong_with_incoming_light_cache_mem_save(
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

        # Max thread count is 1024 (32^2), higher values raise an error.
        # TODO: Worth exploring block size x-y tradeoffs? I.e. 64/16 vs 32/32.
        # https://forums.developer.nvidia.com/t/what-is-the-maximum-number-of-blocks-i-can-use/201587
        # Note that because of thread block limitations, the batch and point dimensions are swapped here:  https://forums.developer.nvidia.com/t/maximum-block-per-grid/246841
        block_size_x = 256  # Point / HW dim
        block_size_y = 4  # Batch dim
        block_size_z = 1  # No light dim
        brdf_eval_kernel.launchRaw(
            blockSize=(block_size_x, block_size_y, block_size_z),
            gridSize=(
                calc_grid_size(normals.shape[1], block_size_x),
                calc_grid_size(normals.shape[0], block_size_y),
                1,
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
        brdf_eval_kernel_bwd = brdf_eval_kernels.eval_outgoing_radiance_blinn_phong_with_incoming_light_cache_mem_save.bwd(
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

        block_size_x = 256  # Point / HW dim
        block_size_y = 2  # Batch dim
        block_size_z = 1  # No light dim
        brdf_eval_kernel_bwd.launchRaw(
            blockSize=(block_size_x, block_size_y, block_size_z),
            gridSize=(
                calc_grid_size(normals.shape[1], block_size_x),
                calc_grid_size(normals.shape[0], block_size_y),
                1,
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


class BatchEvalBlinnPhongBRDFMultiOutgoingLight(Function):
    """
    Evaluate the given Blinn-Phong BRDFs with the given incoming light, matching each point to its corresponding incoming light.
    """

    @staticmethod
    def forward(
        ctx: Any,
        probe_incoming_light: torch.Tensor,
        probe_incoming_light_dirs: torch.Tensor,
        probe_outgoing_light: torch.Tensor,
        probe_outgoing_light_dirs: torch.Tensor,
        light_probe_query: torch.Tensor,
        normals: torch.Tensor,
        diffuse_K: torch.Tensor,
        specular_K: torch.Tensor,
        spec_reflect_c: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inputs:
            probe_incoming_light: (P, N, 3) R,G,B values of incoming light for each direction in the light probe
            probe_incoming_light_dirs: (N, 3) Directions oriented towards the light source in world space (assumed to be the same for each point)
            probe_outgoing_light: (P, O, 3) R,G,B values of outgoing light for each direction
            probe_outgoing_light_dirs: (O, 3) All extra outgoing (view) directions of radiance (essentially just reversed probe_incoming_light_dirs)
            light_probe_query: (B, HW) Correspondence between each of the HW points and their probe number.
            normals: (B, HW, 3) Surface normals in world space for each of HW points
            diffuse_K: (B, HW, 3)
            specular_K: (B, HW, 3)
            spec_reflect_c: (B, HW,)

        Outputs:
            color_diff: (B, HW, 3) R,G,B values that specify the difference in color between the outgoing light probe and the rendered BRDF colors
            for each point (specified by Batch Size B and HW), summed across all the probe outgoing light directions.
        """

        B, HW, _ = normals.shape
        _, N, _ = probe_incoming_light.shape
        output = torch.zeros((B, HW, 3), device="cuda", dtype=torch.float)

        # TODO: Tracing an error where having a batch size that's bigger or around this value can cause errors because the tensor can't be populated
        # (uint saturation?)
        assert output.numel() < MAX_NUMEL_FOR_SLANGTORCH

        brdf_eval_kernel = (
            brdf_eval_kernels.eval_outgoing_radiance_at_multiple_directions_with_probe(
                probe_incoming_light=probe_incoming_light,
                probe_incoming_light_dirs=probe_incoming_light_dirs,
                probe_outgoing_light=probe_outgoing_light,
                probe_outgoing_light_dirs=probe_outgoing_light_dirs,
                light_probe_query=light_probe_query,
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
        # Note that because of thread block limitations, the batch and point dimensions are swapped here:  https://forums.developer.nvidia.com/t/maximum-block-per-grid/246841
        block_size_x = 256  # Point / HW dim
        block_size_y = 4  # Batch dim
        block_size_z = 1  # No light dim
        brdf_eval_kernel.launchRaw(
            blockSize=(block_size_x, block_size_y, block_size_z),
            gridSize=(
                calc_grid_size(normals.shape[1], block_size_x),
                calc_grid_size(normals.shape[0], block_size_y),
                1,
            ),
        )

        # Save all inputs for our backward pass
        ctx.save_for_backward(
            probe_incoming_light,
            probe_incoming_light_dirs,
            probe_outgoing_light,
            probe_outgoing_light_dirs,
            light_probe_query,
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
            probe_outgoing_light,
            probe_outgoing_light_dirs,
            light_probe_query,
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
        brdf_eval_kernel_bwd = brdf_eval_kernels.eval_outgoing_radiance_at_multiple_directions_with_probe_bwd(
            probe_incoming_light=probe_incoming_light,
            probe_incoming_light_dirs=probe_incoming_light_dirs,
            probe_outgoing_light=probe_outgoing_light,
            probe_outgoing_light_dirs=probe_outgoing_light_dirs,
            light_probe_query=light_probe_query,
            normals=normals,
            normals_grad=normals_grad,
            diffuse_K=diffuse_K,
            diffuse_K_grad=diffuse_K_grad,
            specular_K=specular_K,
            specular_K_grad=specular_K_grad,
            spec_reflect_c=spec_reflect_c,
            spec_reflect_c_grad=spec_reflect_c_grad,
            output=output,
            output_grad=grad_output,
        )

        block_size_x = 256  # Point / HW dim
        block_size_y = 2  # Batch dim
        block_size_z = 1  # No light dim
        brdf_eval_kernel_bwd.launchRaw(
            blockSize=(block_size_x, block_size_y, block_size_z),
            gridSize=(
                calc_grid_size(normals.shape[1], block_size_x),
                calc_grid_size(normals.shape[0], block_size_y),
                1,
            ),
        )

        # print(f"{normals_grad = }")
        # print(f"{diffuse_K_grad = }")
        # print(f"{specular_K_grad = }")
        # print(f"{spec_reflect_c_grad = }")

        return (
            None,
            None,
            None,
            None,
            None,
            normals_grad,
            diffuse_K_grad,
            specular_K_grad,
            spec_reflect_c_grad,
        )


def calc_grid_size(dim_size: int, block_size: int) -> int:
    return max((dim_size + (block_size - 1)) // block_size, 1)


def batch_eval_blinn_phong_outgoing_radiance_with_probe_mem_save(
    probe_incoming_light_colors: torch.Tensor,
    probe_incoming_light_dirs: torch.Tensor,
    incoming_light_probe_query: torch.Tensor,
    outgoing_directions: torch.Tensor,
    normals: torch.Tensor,
    diffuse_K: torch.Tensor,
    specular_K: torch.Tensor,
    spec_reflect_c: torch.Tensor,
) -> torch.Tensor:
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

    if USE_CHECKPOINTING_FOR_INCOMING_LIGHT_PROBE_BACKWARD_PASS:
        # We've specialized our BRDF eval logic to only use up to this constant amount of incoming light directions.
        assert N <= MAX_INCOMING_LIGHT_DIRECTIONS_FOR_LOOP_EVAL

    assert probe_incoming_light_dirs.size(0) == N
    assert probe_incoming_light_dirs.size(1) == 3

    assert normals.size(-1) == 3
    B, HW, _ = normals.shape

    assert outgoing_directions.is_same_size(normals)
    assert diffuse_K.is_same_size(normals)
    assert specular_K.is_same_size(diffuse_K)

    assert spec_reflect_c.dim() == 2
    assert spec_reflect_c.size(0) == B

    return cast(
        torch.Tensor,
        BatchEvalBlinnPhongBRDFMemSave.apply(
            probe_incoming_light_colors,
            probe_incoming_light_dirs,
            incoming_light_probe_query,
            outgoing_directions,
            normals,
            diffuse_K,
            specular_K,
            spec_reflect_c,
        ),
    )


def batch_eval_blinn_phong_varying_outgoing_radiance_with_probe(
    probe_incoming_light: torch.Tensor,
    probe_incoming_light_dirs: torch.Tensor,
    probe_outgoing_light: torch.Tensor,
    probe_outgoing_light_dirs: torch.Tensor,
    light_probe_query: torch.Tensor,
    normals: torch.Tensor,
    diffuse_K: torch.Tensor,
    specular_K: torch.Tensor,
    spec_reflect_c: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluate the given Blinn-Phong BRDFs with the given incoming light, matching each point to its corresponding outgoing light.
    Computes all intermediate light ray contributions within the kernel to save memory.

    Inputs:
        probe_incoming_light: (P, N, 3) R,G,B values of incoming light for each direction in the light probe
        probe_incoming_light_dirs: (N, 3) Directions oriented towards the light source in world space (assumed to be the same for each point)
        probe_outgoing_light: (P, O, 3) R,G,B values of outgoing light for each direction
        probe_outgoing_light_dirs: (O, 3) All extra outgoing (view) directions of radiance (essentially just reversed probe_incoming_light_dirs)
        light_probe_query: (B, HW) Correspondence between each of the HW points and their probe number.
        normals: (B, HW, 3) Surface normals in world space for each of HW points
        diffuse_K: (B, HW, 3)
        specular_K: (B, HW, 3)
        spec_reflect_c: (B, HW,)

    Outputs:
        color_diff: (B, HW, 3) R,G,B values that specify the difference in color between the outgoing light probe and the rendered BRDF colors
        for each point (specified by Batch Size B and HW), summed across all the probe outgoing light directions.
    """

    # Assertions for debugging.
    assert probe_incoming_light.dim() == 3
    assert probe_incoming_light.size(-1) == 3
    P, N, _ = probe_incoming_light.shape

    assert probe_outgoing_light.dim() == 3
    assert probe_outgoing_light.size(-1) == 3
    assert probe_outgoing_light.size(0) == probe_incoming_light.size(0)
    _, O, _ = probe_outgoing_light.shape

    if USE_CHECKPOINTING_FOR_INCOMING_LIGHT_PROBE_BACKWARD_PASS:
        # We've specialized our BRDF eval logic to only use up to this constant amount of incoming/outgoing light directions.
        assert N <= MAX_INCOMING_LIGHT_DIRECTIONS_FOR_LOOP_EVAL

    assert probe_incoming_light_dirs.size(0) == N
    assert probe_incoming_light_dirs.size(1) == 3

    assert probe_outgoing_light_dirs.size(0) == O
    assert probe_outgoing_light_dirs.size(1) == 3

    assert normals.size(-1) == 3
    B, HW, _ = normals.shape

    assert diffuse_K.is_same_size(normals)
    assert specular_K.is_same_size(diffuse_K)

    assert spec_reflect_c.dim() == 2
    assert spec_reflect_c.size(0) == B

    return cast(
        torch.Tensor,
        BatchEvalBlinnPhongBRDFMultiOutgoingLight.apply(
            probe_incoming_light,
            probe_incoming_light_dirs,
            probe_outgoing_light,
            probe_outgoing_light_dirs,
            light_probe_query,
            normals,
            diffuse_K,
            specular_K,
            spec_reflect_c,
        ),
    )
