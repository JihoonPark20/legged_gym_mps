"""
Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Gym utilities, adapted for the Genesis + PyTorch MPS backend.
Only the argument/device parsing helpers from the original Isaac Gym
gymutil are kept; the Isaac-specific drawing and sim-config helpers
were removed.
"""

from __future__ import print_function, division, absolute_import

import argparse


def parse_device_str(device_str):
    device = 'cpu'
    device_id = 0

    if device_str == 'cpu' or device_str == 'mps':
        device = device_str
        device_id = 0
    else:
        device_args = device_str.split(':')
        assert len(device_args) == 2 and device_args[0] == 'mps', f'Invalid device string "{device_str}"'
        device, device_id_s = device_args
        try:
            device_id = int(device_id_s)
        except ValueError:
            raise ValueError(f'Invalid device string "{device_str}". Cannot parse "{device_id}"" as a valid device id')
    return device, device_id


def parse_arguments(description="Legged Gym (Genesis)", headless=False, no_graphics=False, custom_parameters=[]):
    parser = argparse.ArgumentParser(description=description)
    if headless:
        parser.add_argument('--headless', action='store_true', help='Run headless without creating a viewer window')
    if no_graphics:
        parser.add_argument('--nographics', action='store_true',
                            help='Disable graphics context creation, no viewer window is created, and no headless rendering is available')
    parser.add_argument('--sim_device', type=str, default="mps", help='Physics Device in PyTorch-like syntax (cpu, mps)')
    parser.add_argument('--graphics_device_id', type=int, default=0, help='Graphics Device ID')

    for argument in custom_parameters:
        if ("name" in argument) and ("type" in argument or "action" in argument):
            help_str = ""
            if "help" in argument:
                help_str = argument["help"]

            if "type" in argument:
                if "default" in argument:
                    parser.add_argument(argument["name"], type=argument["type"], default=argument["default"], help=help_str)
                else:
                    parser.add_argument(argument["name"], type=argument["type"], help=help_str)
            elif "action" in argument:
                parser.add_argument(argument["name"], action=argument["action"], help=help_str)

        else:
            print()
            print("ERROR: command line argument name, type/action must be defined, argument not added to parser")
            print("supported keys: name, type, default, action, help")
            print()

    args = parser.parse_args()

    args.sim_device_type, args.compute_device_id = parse_device_str(args.sim_device)
    args.use_gpu = (args.sim_device_type == 'mps')
    args.physics_engine = "genesis"

    # Using --nographics implies --headless
    if no_graphics and args.nographics:
        args.headless = True

    return args


def parse_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        if v == 1:
            return True
        elif v == 0:
            return False
    if isinstance(v, str):
        if v.lower() in ("true", "yes", "t", "y", "1"):
            return True
        elif v.lower() in ("false", "no", "f", "n", "0"):
            return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
