# TODO: Load Gaussian Model, be able to render it with a basic camera position
# From there, be able to intersect rays with a sphere, 

import os
from pathlib import Path
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
import sys
from scene import Scene, GaussianModel, camera_to_JSON
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from icecream import ic
import random
import math
import cv2
import json
import traceback
from utils.system_utils import searchForMaxIteration
import time
from gaussian_renderer.fast_renderer import FastRenderer
from gaussian_renderer import render, network_gui
from scene.cameras import MiniCam
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov

from scene.dataset_readers import ProjectionType

def load_gaussian_model(model_params: ModelParams, opt_params: OptimizationParams, checkpoint: os.PathLike | None) -> GaussianModel:
    first_iter = 0
    gaussians = GaussianModel(model_params.sh_degree, model_params.use_neural_network, model_params.max_opacity)

    if checkpoint is not None:
        (gaussian_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(gaussian_params, opt_params)

    # Search for a valid EVER-comaptible .ply
    else:
        loaded_iter = searchForMaxIteration(os.path.join(model_params.model_path, "point_cloud"))
        print("Loading trained model at iteration {}".format(loaded_iter))

        gaussians.load_ply(os.path.join(model_params.model_path,
                                                       "point_cloud",
                                                       "iteration_" + str(loaded_iter),
                                                       "point_cloud.ply"))
        
    return gaussians

def render_gaussian_model(gaussians: GaussianModel, model_params: ModelParams, opt_params: OptimizationParams, pipe_params: PipelineParams) -> torch.Tensor:
    bg_color = [1, 1, 1] if model_params.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    gaussians.training_setup(opt_params)
    torch.cuda.empty_cache()

    camera_index = 120
    # Get the cameras.json file that enumerates all the cameras and their positions
    rendering_cam = get_rendering_cam(model_params, camera_index)
    
    # TODO: Do rendering
    preview_factor = 4
    image_width = rendering_cam.image_width
    image_height = rendering_cam.image_height

    rendering_cam.image_width = image_width // preview_factor
    rendering_cam.image_height = image_height // preview_factor

    print(f"{rendering_cam = }")

    renderer = FastRenderer(rendering_cam, gaussians, pipe_params.enable_GLO)
    renderer.set_camera(rendering_cam)

    # Render quickly
    st = time.time()

    # TODO: Use opposite of t_min (t_max?) for creating chromesphere?
    image = renderer.render(rendering_cam, gaussians, background)

    print(f"Took {time.time()-st}s to render frame.")

    # Convert to 8-bit and save as png
    image = (torch.clamp(image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
    image = cv2.resize(image, (image_width, image_height))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    cv2.imwrite(Path(model_params.model_path) / "test_output.png", image)
    
    torch.cuda.empty_cache()

    



def get_rendering_cam(model_params: ModelParams, camera_index: int) -> MiniCam:
    camera_file = Path(model_params.model_path) / "cameras.json"
    with open(camera_file) as f:
        camera_arr = json.load(f)
        # print(f"Cameras: {camera_arr}")
        # print(f"Cameras[0]: {camera_arr[0]}")
        chosen_cam = camera_arr[camera_index]

        # Some more intermediate values:
        # Rt = np.zeros((4, 4))
        # Rt[:3, :3] = camera.R.transpose()
        # Rt[:3, 3] = camera.T
        # Rt[3, 3] = 1.0

        # W2C = np.linalg.inv(Rt)
        # pos = W2C[:3, 3]
        # rot = W2C[:3, :3]

        world_to_camera = torch.zeros((4, 4))
        world_to_camera[:3, 3] = torch.tensor(chosen_cam['position'])
        world_to_camera[:3, :3] = torch.tensor(chosen_cam['rotation'])
        world_to_camera[3, 3] = 1.0

        fovy = focal2fov(chosen_cam['fy'], chosen_cam['height'])
        fovx = focal2fov(chosen_cam['fx'], chosen_cam['width'])

        world_view_transform = world_to_camera.transpose(0, 1).to(device="cuda")
        z_near = 0.01
        z_far = 100
        proj_matrix = getProjectionMatrix(znear=z_near, zfar=z_far, fovX=fovx, fovY=fovy).transpose(0,1).to(device="cuda")
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(proj_matrix.unsqueeze(0))).squeeze(0)

        rendering_cam = MiniCam(chosen_cam['width'], chosen_cam['height'], fovy, fovx, 
                                0.01, 100, world_view_transform, full_proj_transform)
        
        rendering_cam.model = ProjectionType.PERSPECTIVE
                                
    return rendering_cam




if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Manual Renderer Parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    # args.checkpoint_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Load Gaussians
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    gaussians = load_gaussian_model(lp.extract(args), op.extract(args), args.start_checkpoint)

    print(f"Loaded Gaussian, Active SH Degree: {gaussians.active_sh_degree}")

    render_gaussian_model(gaussians,lp.extract(args), op.extract(args), pp.extract(args))

    # All done
