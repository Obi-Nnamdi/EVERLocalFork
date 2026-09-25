# Gets copied to build directory using CMake.
from .fast_cu_knn import run_knn

# Explicitly define what is exposed to the user when they do:
# from my_package import *
__all__ = [
    "run_knn",
]
