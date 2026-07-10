import torch


def nchw_tensor_to_p_by_c(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts a (N, C, H, W) tensor into a (P, C) tensor, assuming that N (first dim) is 1, and there a P = H * W points.
    E.g. (1, 3, H, W) -> (H * W, 3)
    E.g. (1, 1, H, W) -> (H * W, 1)
    """
    N, C, H, W = input_tensor.shape
    input_tensor = input_tensor.squeeze(0)  # (C, H, W)
    input_tensor = input_tensor.reshape(C, H * W)
    input_tensor = input_tensor.T  # (H * W, C)

    return input_tensor


def p_by_c_tensor_to_chw(input_tensor: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Converts a (P, C) tensor into a (C, H, W) tensor, assuming that P was created by collapsing the (H,W) dimensions.
    E.g. (H * W, 3) -> (1, 3, H, W)
    """
    P, C = input_tensor.shape
    input_tensor = input_tensor.reshape(H, W, C)  # (H, W, C)
    input_tensor = input_tensor.permute(2, 0, 1)  # (C, H, W)

    return input_tensor


def pretty_display_normal_tensor(normal_tensor: torch.Tensor) -> torch.Tensor:
    """
    Cleans up a normal tensor by mapping it from (-1, 1) -> (0, 1)
    """
    return (normal_tensor / 2) + 0.5


def size_of_tensor_bytes(tensor: torch.Tensor):
    """
    Returns size of a tensor in bytes.
    https://discuss.pytorch.org/t/how-to-know-the-memory-allocated-for-a-tensor-on-gpu/28537/2
    """
    return tensor.element_size() * tensor.numel()