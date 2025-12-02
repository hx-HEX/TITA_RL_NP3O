import numpy as np
import os
from datetime import datetime
from configs.tita_constraint_config import TitaConstraintRoughCfg, TitaConstraintRoughCfgPPO
from configs.tita_wheel_constraint_config import TitaConstraintWheelCfg, TitaConstraintWheelCfgPPO
from configs.tita_fusion_constraint_config import TitaFusionCfg, TitaFusionCfgPPO
from configs.tita_flat_config import TitaFlatCfg, TitaFlatCfgPPO
from configs.tita_rough_config import TitaRoughCfg, TitaRoughCfgPPO
from envs.no_constrains_legged_robot import Tita
from configs.tita_feet_constraint_config import TitaConstraintFeetCfg,TitaConstraintFeetCfgPPO
from configs.tron1_wheel_constraint_config import Tron1ConstraintWheelCfg, Tron1ConstraintWheelCfgPPO
from configs.tron1_feet_constraint_config import Tron1ConstraintFeetCfg, Tron1ConstraintFeetCfgPPO
from configs.tron1_unified_constraint_config import Tron1ConstraintUnifiedCfg, Tron1ConstraintUnifiedCfgPPO

from global_config import ROOT_DIR, ENVS_DIR
import isaacgym
from utils.helpers import get_args
from envs import LeggedRobot, Tron1Robot, Tron1FeetRobot,Tron1UnifiedRobot
from utils.task_registry import task_registry

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)

    # Define the log path and task configuration folder path
    logs_path = os.path.join(ROOT_DIR, "logs")
    task_config_folder = os.path.join(logs_path, f"{args.task}")

    # Check if the task configuration folder exists and save configurations
    if os.path.exists(task_config_folder) and os.path.isdir(task_config_folder):
        print(f"Task configuration folder exists: {task_config_folder}, saving configuration files.")
        task_registry.save_cfgs(name=args.task, train_cfg=train_cfg)
    else:
        print(f"Task configuration folder does not exist: {task_config_folder}, skipping configuration saving.")

    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':

    task_registry.register("tita_constraint",LeggedRobot,TitaConstraintRoughCfg(),TitaConstraintRoughCfgPPO())
    task_registry.register("tita_wheel_constraint",LeggedRobot,TitaConstraintWheelCfg(),TitaConstraintWheelCfgPPO())
    task_registry.register("tita_feet_constraint",LeggedRobot,TitaConstraintFeetCfg(),TitaConstraintFeetCfgPPO())
    task_registry.register("tita_fusion_constraint", LeggedRobot, TitaFusionCfg(), TitaFusionCfgPPO())
    task_registry.register("tita_flat", Tita, TitaFlatCfg(), TitaFlatCfgPPO())
    task_registry.register("tita_rough", Tita, TitaRoughCfg(), TitaRoughCfgPPO())

    task_registry.register("tron1_wheel_constraint", Tron1Robot, Tron1ConstraintWheelCfg(), Tron1ConstraintWheelCfgPPO())
    task_registry.register("tron1_feet_constraint", Tron1FeetRobot, Tron1ConstraintFeetCfg(), Tron1ConstraintFeetCfgPPO())
    task_registry.register("tron1_unified_constraint", Tron1UnifiedRobot, Tron1ConstraintUnifiedCfg(), Tron1ConstraintUnifiedCfgPPO())

    args = get_args()
    train(args)
