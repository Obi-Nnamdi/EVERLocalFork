import math
import numpy as np
import torch
from scipy.special import sph_harm_y
from utils.sh_utils import eval_sh

def normalization_constant_sh_coeff(n: int, m: int) -> float:
    assert abs(m) <= n
    if m == 0:
        numerator = 2 * n + 1
        denominator = 4 * math.pi
        return math.sqrt(numerator / denominator)

    else:
        numerator = 2 * n + 1 * math.factorial(n - abs(m))
        denominator = 4 * math.pi * math.factorial(n + abs(m))
        return math.sqrt(numerator / denominator)


def real_spherical_harmonics(
    n: int, m: int, theta: torch.Tensor, phi: torch.Tensor
) -> np.ndarray:
    """
    Use numpy's conventions to evaluate the real spherical harmonics:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.sph_harm_y.html

    Parameters:
        n: degree of the spherical harmonic (e.g, "l" in usual literature)
        m: order of the spherical harmonic
        theta: polar angle(s), from 0 -> pi
        phi: azimuthal angle(s), from 0 -> 2pi

    Adapted from the PlenOctree paper, appendix B.1: https://arxiv.org/abs/2103.14024.
    """

    assert torch.all(theta >= 0) and torch.all(theta <= math.pi)
    assert torch.all(phi >= 0) and torch.all(phi <= 2 * math.pi)
    assert abs(m) <= n
    theta_numpy = theta.cpu().numpy()
    phi_numpy = phi.cpu().numpy()

    y_nm_func = lambda n, m: sph_harm_y(n, m, theta_numpy, phi_numpy)

    # Define real-valued spherical harmonics according to formula
    if m == 0:
        return sph_harm_y(n, m, theta_numpy, phi_numpy).real

    elif m > 0:
        # No need to add in the Condon-Shortly phase term again [(-1)**m], it's already part of
        # scipy's spherical harmonic function:
        sqrt_2 = math.sqrt(2)
        return sqrt_2 * sph_harm_y(n, m, theta_numpy, phi_numpy).real

    else:
        # m < 0
        # Again, no need for additional phase term
        sqrt_2 = math.sqrt(2)
        return sqrt_2 * sph_harm_y(n, abs(m), theta_numpy, phi_numpy).imag


def generate_SH_coeff_matrices(
    max_n: int, theta: torch.Tensor, phi: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates "B" and "L" matrix as seen by
    https://kvttt.github.io/files/Spherical_Harmonics_Fitting.pdf#page=1.83,
    section 2.1.

    Parameters:
        max_n: max degree of the spherical harmonics
        theta: (n,) polar angle(s), from 0 -> pi
        phi: (n,) azimuthal angle(s), from 0 -> 2pi

    Returns:
        tuple of (B_matrix, L_matrix), each with shape (num_angles, degrees)
    """

    assert max_n >= 0
    B_matrix = torch.empty(
        0,
    )
    l_values = (
        []
    )  # Holds "weight" for each SH to encourage stronger regularization for high freqs
    for n in range(max_n + 1):
        for m in range(-n, n + 1):
            # Create column vector of SH values and append them to our B matrix
            real_sh_vals = torch.from_numpy(real_spherical_harmonics(n, m, theta, phi))
            real_sh_vals = real_sh_vals.reshape(-1, 1)
            B_matrix = torch.cat([B_matrix, real_sh_vals], dim=1)
            l_values.append(n * (n + 1))

    #
    L_matrix = torch.diag(torch.tensor(l_values)).to(dtype=torch.float32)
    return B_matrix.to(dtype=torch.float32), L_matrix


def solve_for_real_SH_coeffs(
    observation_vals: torch.Tensor,
    B_matrix: torch.Tensor,
    L_matrix: torch.Tensor,
    lambda_val: float,
) -> torch.Tensor:
    """
    Solve for the SH coeffs that would be needed to reconstruct observation_vals using the provided SH_coeffs from B_matrix and
    regularization from L_matrix and lambda_val.

    Parameters:
        observation_vals: (B, n, k) vector.
        B_matrix: (n, d) matrix.
        L_matrix: (d, d) matrix.
        lambda: float denoting amount of regularization to put on coeffs.

    Returns:
        solutions: (B, d, k), where each batch has its own SH coeff solutions.
    """

    assert lambda_val >= 0
    assert observation_vals.dim() == 3

    batch_dim = observation_vals.size(0)

    # Define A
    lambda_L_mat = math.sqrt(lambda_val) * L_matrix
    stacked_BL_mat = torch.cat([B_matrix, lambda_L_mat], dim=0)  # (n + d, d)
    stacked_BL_mat = stacked_BL_mat[None, :].expand(batch_dim, -1, -1)  # (B, n + d, d)

    # Define B
    d = L_matrix.size(0)
    n = B_matrix.size(0)
    zero_padding = torch.zeros(
        (batch_dim, d, observation_vals.size(-1)),
        dtype=observation_vals.dtype,
        device=observation_vals.device,
    )

    padded_observations = torch.cat(
        [observation_vals, zero_padding], dim=1
    )  # (B, n + d, k)

    # Solve system of eq: min_x||Ax - b||
    solutions, _, _, _ = torch.linalg.lstsq(stacked_BL_mat, padded_observations)

    # Remove the extra stuff created for the regularization
    return solutions[:, :n, :]


def fit_SH_to_data(
    observations: torch.Tensor,
    theta_vals: torch.Tensor,
    phi_vals: torch.Tensor,
    lambda_val: float,
    max_SH_degree: int,
    return_residuals=False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fit SHs to batched data captured at various (theta, phi) angles.

    Parameters:
        observations: (B, n, k) vector, where each row is a new observation.
        theta_vals: (n,) vector of theta values for each observation (0 -> pi), polar angle
        phi_vals: (n,) vector of phi values for each observation (0 -> 2pi), aziumuthal angle
        lambda: float denoting amount of regularization to put on coeffs.
        max_SH_degree: maximum degree of Spherical Harmonics to fit. All spherical harmonic orders are used.

    Returns:
        coefficients: (B, (max_sh_degree + 1) ** 2, k) of Spherical harmonic coefficients that fit the observations.

    """

    B_matrix, L_matrix = generate_SH_coeff_matrices(max_SH_degree, theta_vals, phi_vals)

    B_matrix = B_matrix.to(observations.device)
    L_matrix = L_matrix.to(observations.device)

    result = solve_for_real_SH_coeffs(observations, B_matrix, L_matrix, lambda_val)

    if return_residuals:
        residuals = (B_matrix[None, :, :] @ result) - observations
        return result, residuals
    return result, torch.empty(0)


if __name__ == "__main__":
    torch.manual_seed(0)

    num_theta_vals = 40
    channels = 3
    batches = 10

    theta_vals = torch.linspace(0, torch.pi, num_theta_vals)
    phi_vals = torch.linspace(0, torch.pi * 2, num_theta_vals)

    dummy_observation_vals = torch.atleast_2d(
        torch.cos(theta_vals) + torch.sin(phi_vals) + torch.cos(theta_vals * phi_vals)
    ).T.expand(
        -1, channels
    )  # (dummy RGB)
    dummy_observation_vals = dummy_observation_vals[None, :].expand(batches, -1, -1)

    print(f"{dummy_observation_vals[0, :, 0] = }")

    # with no regularization (lambda = 0), high frequency values grow incredibly large!
    lambda_val = 0.05

    max_n = 4
    expected_sh_degrees = (max_n + 1) ** 2
    result, _ = fit_SH_to_data(
        dummy_observation_vals.cuda(),
        theta_vals.cuda(),
        phi_vals.cuda(),
        lambda_val,
        max_n,
    )
    assert result.shape == (batches, expected_sh_degrees, channels)

    # Trying to make x,y,z evaluation equivalent to theta and phi
    x_vals = torch.cos(phi_vals) * torch.sin(theta_vals)
    y_vals = torch.sin(phi_vals) * torch.sin(theta_vals)
    z_vals = torch.cos(theta_vals)

    xyz_dirs = torch.stack((x_vals, y_vals, z_vals), dim=1).cuda()
    assert xyz_dirs.shape == (num_theta_vals, channels)

    evaluating_sh_coeffs = result.permute(
        0, 2, 1
    )  # (B, deg, channels) -> (B, channels, deg)
    evaluating_sh_coeffs = evaluating_sh_coeffs[0:1].expand(num_theta_vals, -1, -1)
    assert evaluating_sh_coeffs.shape == (num_theta_vals, channels, expected_sh_degrees)

    cartesian_max_n = 4
    cartesian_expected_sh_degrees = (cartesian_max_n + 1) ** 2
    cartesian_answer = eval_sh(cartesian_max_n, evaluating_sh_coeffs, xyz_dirs)
    assert cartesian_answer.shape == (num_theta_vals, channels)

    # Generate spherical coordinates result
    B_matrix, _ = generate_SH_coeff_matrices(max_n, theta_vals, phi_vals)

    start_degree = 0
    end_degree = cartesian_expected_sh_degrees
    truncated_B_matrix = B_matrix[:, start_degree:end_degree].cuda()
    spherical_coords_answer = truncated_B_matrix @ result[0, start_degree:end_degree, 0]

    print(f"{cartesian_answer[:, 0] = }")
    print(f"{spherical_coords_answer = }")
    # TODO: Some small differences, largely chalking them up to numerical error with
    # sin and cos, but could use double precision
    print(f"{spherical_coords_answer - cartesian_answer[:, 0] = }")
    print(f"{spherical_coords_answer - cartesian_answer[:, 0] = }")
    torch.testing.assert_close(spherical_coords_answer, cartesian_answer[:, 0])
