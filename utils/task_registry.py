import os
from datetime import datetime
from typing import Tuple
import torch
import numpy as np
from shutil import copyfile
import ntpath

from envs.vec_env import VecEnv
from runner import OnConstraintPolicyRunner

from global_config import ROOT_DIR, ENVS_DIR
from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params
from configs import LeggedRobotCfg, LeggedRobotCfgPPO

from runner import OnPolicyRunner

class TaskRegistry():
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}
    
    def register(self, name: str, task_class: VecEnv, env_cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgPPO):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg
    
    def get_task_class(self, name: str) -> VecEnv:
        return self.task_classes[name]
    
    def get_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        # copy seed
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg


    def save_cfgs(self, name, train_cfg):
        """
        Save all task-related configuration files to the log directory.

        Args:
            name (str): Task name used to locate the configuration files.
            train_cfg (object): Training configuration object, used to determine which files to save.
        """
        # Ensure the log directory exists
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

        # Determine the correct Python file based on the runner class name
        if train_cfg.runner.runner_class_name == "OnPolicyRunner":
            robot_file = "no_constrains_legged_robot.py"
        else:
            robot_file = "legged_robot.py"

        # Define the file paths to be saved (from save_config_files logic)
        save_items = [
            os.path.join(ENVS_DIR, robot_file),  # Path to the selected robot file
            os.path.join(ROOT_DIR, "configs", "legged_robot_config.py"),  # Path to legged_robot_config.py
            os.path.join(ROOT_DIR, "configs", f"{name}_config.py"),  # Path to task-specific constraint config file
        ]

        # Add the task-specific Python file path (if it exists)
        py_root = os.path.join(ENVS_DIR, f"{name}.py")
        if os.path.exists(py_root):
            save_items.append(py_root)

        # Additional files to save (from original save_cfgs logic)
        additional_items = [
            os.path.join(ROOT_DIR, "configs", f"{name}_config.py"),  # Task-specific constraint config
        ]
        save_items.extend(additional_items)

        # Save all files
        for save_item in save_items:
            if os.path.exists(save_item):  # Check if the file exists
                base_file_name = ntpath.basename(save_item)  # Get the file name
                destination_path = os.path.join(self.log_dir, base_file_name)  # Destination path
                copyfile(save_item, destination_path)  # Copy the file
                print(f"Saved: {destination_path}")
            else:
                print(f"Warning: {save_item} does not exist and will not be copied.")



    def make_env(self, name, args=None, env_cfg=None) -> Tuple[VecEnv, LeggedRobotCfg]:
        """ Creates an environment either from a registered namme or from the provided config file.

        Args:
            name (string): Name of a registered env.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            env_cfg (Dict, optional): Environment config file used to override the registered config. Defaults to None.

        Raises:
            ValueError: Error if no registered env corresponds to 'name' 

        Returns:
            isaacgym.VecTaskPython: The created environment
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        # check if there is a registered env with that name
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
        if env_cfg is None:
            # load config files
            env_cfg, _ = self.get_cfgs(name)
        # override cfg from args (if specified)
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
        set_seed(env_cfg.seed)
        # parse sim params (convert to dict first)
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)
        env = task_class(   cfg=env_cfg,
                            sim_params=sim_params,
                            physics_engine=args.physics_engine,
                            sim_device=args.sim_device,
                            headless=args.headless)
        return env, env_cfg

    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default") -> Tuple[OnConstraintPolicyRunner, LeggedRobotCfgPPO]:
        """ Creates the training algorithm  either from a registered namme or from the provided config file.

        Args:
            env (isaacgym.VecTaskPython): The environment to train (TODO: remove from within the algorithm)
            name (string, optional): Name of a registered env. If None, the config file will be used instead. Defaults to None.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            train_cfg (Dict, optional): Training config file. If None 'name' will be used to get the config file. Defaults to None.
            log_root (str, optional): Logging directory for Tensorboard. Set to 'None' to avoid logging (at test time for example). 
                                      Logs will be saved in <log_root>/<date_time>_<run_name>. Defaults to "default"=<path_to_LEGGED_GYM>/logs/<experiment_name>.

        Raises:
            ValueError: Error if neither 'name' or 'train_cfg' are provided
            Warning: If both 'name' or 'train_cfg' are provided 'name' is ignored

        Returns:
            PPO: The created algorithm
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        # if config files are passed use them, otherwise load from the name
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            # load config files
            _, train_cfg = self.get_cfgs(name)
        else:
            if name is not None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")
        # override cfg from args (if specified)
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)

        if log_root=="default":
            log_root = os.path.join(ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
            self.log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        elif log_root is None:
            self.log_dir = None
        else:
            self.log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)

        train_cfg_dict = class_to_dict(train_cfg)
        if train_cfg.runner.runner_class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, train_cfg_dict, self.log_dir, device=args.rl_device)
        else:
            runner_class = eval(train_cfg.runner.runner_class_name)
            runner = runner_class(env, train_cfg_dict, self.log_dir, device=args.rl_device)
        #save resume path before creating a new log_dir
        resume = train_cfg.runner.resume
        if train_cfg.runner.runner_class_name == "OnPolicyRunner":
            if resume:
                # load previously trained model
                resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
                print(f"Loading model from: {resume_path}")
                runner.load(resume_path)
        return runner, train_cfg

# make global task registry
task_registry = TaskRegistry()
