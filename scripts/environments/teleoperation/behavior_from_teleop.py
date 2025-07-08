# behavior_clone_from_teleop.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from rl_games.torch_runner import Runner
from rl_games.common.algo_observer import IsaacAlgoObserver

import argparse
import sys
from distutils.util import strtobool

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--teleop", type=str, default=None, help="Path to teleoperation data.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# === CONFIG ===
BATCH_SIZE = 256
EPOCHS = 10
LR = 3e-4

# === LOAD TELEOP DATA ===
print("[INFO] Loading teleop data from", args_cli.teleop)
data = torch.load(args_cli.teleop)
flat_data = [step for episode in data for step in episode]
obs_tensor = torch.stack([d["obs"] for d in flat_data])
action_tensor = torch.stack([d["action"] for d in flat_data])

print(f"[INFO] Loaded {len(flat_data)} transitions")

# === DEFINE DATASET ===
class BCDataset(Dataset):
    def __init__(self, obs, actions):
        self.obs = obs
        self.actions = actions

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx]

dataset = BCDataset(obs_tensor, action_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === LOAD AGENT FROM CONFIG ===
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.envs import DirectRLEnvCfg  # or the correct cfg base
import isaaclab_tasks.direct.factory  # your task registration

@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg, agent_cfg):
    # Setup dummy RL-Games runner to get agent model
    agent_cfg["params"]["config"]["multi_gpu"] = False
    agent_cfg["params"]["config"]["device"] = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)
    runner.reset()
    agent = runner.algo
    actor = agent.model.sac_network.actor
    actor.train()

    optimizer = optim.Adam(actor.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    # specify directory for logging experiments
    config_name = args_cli.task
    log_root_path = os.path.join("logs", "teleoperation", config_name)
    log_root_path = os.path.abspath(log_root_path)

    # === TRAIN ===
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for obs_batch, act_batch in dataloader:
            obs_batch, act_batch = obs_batch.cuda(), act_batch.cuda()
            pred_actions = actor(obs_batch)
            loss = loss_fn(pred_actions, act_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {epoch_loss / len(dataloader):.6f}")

    # === SAVE ===
    os.makedirs(log_root_path, exist_ok=True)
    filename = os.path.join(log_root_path, "teleop_multi_episode_trajectory.pth")
    print("[INFO] Saving checkpoint to", filename)
    runner.algo.save(filename)

    print("[INFO] Behavior cloning finished.")

if __name__ == "__main__":
    main()
