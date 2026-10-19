import math
import numpy as np
import torch
from scipy.special import sph_harm_y, sph_legendre_p, sph_harm_y_all


def normalization_constant_sh_coeff(n: int, m: int) -> float:
    assert abs(m) <= n
    if m == 0:
        numerator = 2 * n + 1
        denominator = 4 * math.pi
        return math.sqrt(numerator / denominator)

    else:
        numerator = 2 * n + 1 * math.factorial(n - abs(m))
        denominator = 4 * math.pi * math.factorial(n + abs(m))
        return ((-1) ** (m)) * math.sqrt(2) * math.sqrt(numerator / denominator)


def real_spherical_harmonics(
    n: int, m: int, theta: torch.Tensor, phi: torch.Tensor
) -> np.ndarray:
    """

    Parameters:
        n: degree of the spherical harmonic (e.g, "l" in usual literature)
        m: order of the spherical harmonic
        theta: polar angle(s), from 0 -> 2pi
        phi: azimuthal angle(s), from 0 -> pi

    Adapted from the following resource:
    https://peakd.com/hive-128780/@mes/220d6572
    """
    assert abs(m) <= n
    theta = theta.cpu()
    phi = phi.cpu()

    y_nm_func = lambda n, m: sph_harm_y(n, m, theta, phi)

    # Define real-valued spherical harmonics according to formula
    if m == 0:
        return y_nm_func(n, 0).real

    elif m > 0:
        neg_one_power = ((-1) ** m)
        outside_factor = neg_one_power / math.sqrt(2)
        inner_factor = y_nm_func(n, m) + neg_one_power * y_nm_func(n, -m)
        return (outside_factor * inner_factor).real

    else:
        # m < 0
        neg_one_plus_power = ((-1) ** (m + 1))
        outside_factor = neg_one_plus_power / (math.sqrt(2) * 1j)
        abs_m = abs(m)

        neg_one_abs_power = ((-1) ** abs_m)
        inner_factor = y_nm_func(n, abs_m) - neg_one_abs_power * y_nm_func(n, -abs_m)

        return (outside_factor * inner_factor).real


def generate_SH_coeff_matrices(max_n: int, theta: torch.Tensor, phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates "B" and "L" matrix as seen by
    https://kvttt.github.io/files/Spherical_Harmonics_Fitting.pdf#page=1.83,
    section 2.1.

    Parameters:
        max_n: max degree of the spherical harmonics
        theta: polar angle(s), from 0 -> 2pi
        phi: azimuthal angle(s), from 0 -> pi

    Returns:
        tuple of (B_matrix, L_matrix).
    """
    assert max_n >= 0
    B_matrix = torch.empty(0,)
    l_values = [] # Holds "weight" for each SH to encourage stronger regularization for high freqs
    for n in range (max_n + 1):
        for m in range(-n, n + 1):
            # Create column vector of SH values and append them to our B matrix
            real_sh_vals = torch.from_numpy(real_spherical_harmonics(n, m, theta, phi))
            real_sh_vals = real_sh_vals.reshape(-1, 1)
            B_matrix = torch.cat([B_matrix, real_sh_vals], dim=1)
            l_values.append(n * (n + 1))

    # 
    L_matrix = torch.diag(torch.tensor(l_values)).to(dtype=torch.float32)
    return B_matrix.to(dtype=torch.float32), L_matrix

def solve_for_real_SH_coeffs(observation_vals: torch.Tensor, B_matrix: torch.Tensor, L_matrix: torch.Tensor, lambda_val: float) -> torch.Tensor:
    """
    Solve for the SH coeffs that would be needed to reconstruct observation_vals using the provided SH_coeffs from B_matrix and
    regularization from L_matrix and lambda_val.

    Parameters:
        observation_vals: (n, k) vector.
        B_matrix: (n, d) matrix.
        L_matrix: (d, d) matrix.
        lambda: float denoting amount of regularization to put on coeffs.

    Returns:
        solutions: (n, k)
    """
    assert lambda_val >= 0
    lambda_L_mat = math.sqrt(lambda_val) * L_matrix
    stacked_BL_mat = torch.cat([B_matrix, lambda_L_mat], dim=0) # (n + d, d)

    d = L_matrix.size(0)
    n = B_matrix.size(0)
    zero_padding = torch.zeros((d, observation_vals.size(1)), dtype=observation_vals.dtype, device=observation_vals.device)

    padded_observations = torch.cat([observation_vals, zero_padding], dim=0) # (n + d, 1)

    # Solve system of eq: min_x||Ax - b||
    solutions, _, _, _ = torch.linalg.lstsq(stacked_BL_mat, padded_observations)

    # Remove the extra stuff created for the regularization
    return solutions[:n]


def fit_SH_to_data(
    observations: torch.Tensor,
    theta_vals: torch.Tensor,
    phi_vals: torch.Tensor,
    lambda_val: float,
    max_SH_degree: int,
) -> torch.Tensor:
    """
    Fit SHs to data captured at various (theta, phi) angles.

    Parameters:
        observations: (n, k) vector, where each row is a new observation.
        theta_vals: (n,) vector of theta values for each observation
        phi_vals: (n,) vector of phi values for each observation
        lambda: float denoting amount of regularization to put on coeffs.
        max_SH_degree: maximum degree of Spherical Harmonics to fit. All spherical harmonic orders are used.

    Returns:


    """
    B_matrix, L_matrix = generate_SH_coeff_matrices(max_SH_degree, theta_vals, phi_vals)

    B_matrix = B_matrix.to(observations.device)
    L_matrix = L_matrix.to(observations.device)

    result = solve_for_real_SH_coeffs(observations, B_matrix, L_matrix, lambda_val)
    return result


if __name__ == "__main__":
    torch.manual_seed(0)

    num_theta_vals = 100
    theta_vals = torch.rand((num_theta_vals,)) * torch.pi * 2
    phi_vals = torch.rand((num_theta_vals,)) * torch.pi * 2

    dummy_observation_vals = torch.atleast_2d(
        2 * torch.cos(theta_vals) + 4 * torch.sin(phi_vals)
    ).T.expand(
        -1, 3
    )  # (dummy RGB)
    # dummy_observation_vals = torch.ones_like(dummy_observation_vals) * 10
    lambda_val = 10

    max_n = 2
    result = fit_SH_to_data(
        dummy_observation_vals.cuda(),
        theta_vals.cuda(),
        phi_vals.cuda(),
        lambda_val,
        max_n,
    )

    print(f"{result = }")

    # residuals = B_matrix @ result - dummy_observation_vals
    # print(f"{torch.sum(torch.pow(residuals, 2)) = }")
