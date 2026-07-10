#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math
import os
from pathlib import Path
import sys
from argparse import ArgumentParser, Namespace
from typing import Any

class GroupParams:
    pass
    def __getattr__(self, attr: str) -> Any:
        """
        Allows 'correct' indexing of parameters that start with underscores.
        """
        if attr.startswith("_"):
            clipped_attr = attr[1:]
            try:
                return object.__getattribute__(self, clipped_attr)
            except AttributeError:
                raise AttributeError


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, action="store_true"
                    )
                elif t == list or t == tuple:
                    group.add_argument(
                        "--" + key,
                        ("-" + key[0:1]),
                        nargs="+",
                        type=type(value[0]),
                        default=value,
                    )
                else:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, type=t
                    )
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                elif t == list or t == tuple:
                    group.add_argument(
                        "--" + key, nargs="+", type=type(value[0]), default=value
                    )
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cpu"
        self.render_spline = False
        self.use_neural_network = False
        self.eval = False
        self.num_additional_pts = 10000
        self.additional_size_multi = 1.0
        self.num_spline_frames = 480
        self.glo_latent_dim = 64
        self.max_opacity = 0.99
        self.tmin = 0.2
        self.enable_mip_splatting = False  # Add the missing attribute
        self.low_pass_2d_kernel_size = 3 # Add the missing attribute
        self.low_pass_3d_kernel_size = 3 # Add the missing attribute

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = True
        self.compute_cov3D_python = False
        self.enable_GLO = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000

        self.betas = [0.9, 0.999]

        self.position_lr_final = 4e-7 #0.0000004
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.position_lr_init = 4e-5 #0.00004

        self.glo_lr = 0.01
        self.glo_network_lr = 0.00005

        self.feature_lr = 0.0025
        self.feature_rest_lr = 0.00025
        self.bg_lr = 0.0
        self.opacity_lr = 0.0125
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.min_opacity = 0.005
        self.min_split_opacity = 0.01
        self.percent_dense = 0.0025
        self.lambda_dssim = 0.2
        self.max_lambda_dssim = 0.8  # Add the missing attribute
        self.max_dssim_iteration = 15000  # Add the missing attribute

        self.lambda_anisotropic = 1e-1
        self.lambda_distortion = 0
        self.sh_up_interval = 2000

        self.densification_interval = 200
        self.opacity_reset_interval = 300000
        self.densify_from_iter = 1500
        self.densify_until_iter = 16_000

        self.densify_grad_threshold: float = 2.5e-7

        self.clone_grad_threshold: float = 1e-1

        self.center_pixel = False
        self.fallback_xy_grad = False

        self.random_background = False

        # Bilateral grid parameters
        self.use_bilateral_grid = False
        self.bilateral_grid_shape = [16, 16, 8]
        self.bilateral_grid_lr = 0.003  # Match gsplat's default

        # Default to 10.0 (gsplat's value) but keep it configurable
        # Explicitly set as float to ensure command line arguments work with decimal values
        self.lambda_tv: float = 10.0

        super().__init__(parser, "Optimization Parameters")


class BRDFOptmizationParams(ParamGroup):
    """
    Parameters for BRDF Optimization.
    """

    def __init__(self, parser: ArgumentParser):
        self.training_steps = 500 * 20  # How many steps to train BRDF model for
        self.resume_from = ""  # Checkpoint file to resume BRDF training model from.

        self.checkpoint_interval = 250  # How often to save model checkpoints.
        self.image_reporting_interval = (
            250  # How often to save output model images to the tensorboard.
        )

        self.incoming_light_divisions = 20  # How many sphere divisions should be used for calculating the incoming light

        # Have loss calculated only at randomly sampled points (specified by point_batch_size) to avoid high VRAM costs.
        self.randomly_sample_loss = False
        self.point_batch_size = 2048 * 12

        # How far to downsample the original resolution that the dataset cameras were rendered at.
        self.preview_factor = 4
        super().__init__(parser, "BRDF Optimization Parameters")

    def extract(self, args):
        params = super().extract(args)
        return params


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
