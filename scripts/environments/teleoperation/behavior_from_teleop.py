import argparse
import sys
import os
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from isaaclab.app import AppLauncher

# === CLI ===
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--teleop", type=str, required=True, help="Path to teleop data .pth")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from rl_games.torch_runner import Runner
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver

# === Load teleop data ===
print("[INFO] Loading teleop data from", args_cli.teleop)
teleop_data = torch.load(args_cli.teleop)
flat_data = [step for episode in teleop_data for step in episode]
obs_tensor = torch.stack([d["obs"] for d in flat_data])
state_tensor = torch.stack([d["state"] for d in flat_data])
action_tensor = torch.stack([d["action"] for d in flat_data])
if action_tensor.ndim == 3 and action_tensor.shape[1] == 1:
    action_tensor = action_tensor.squeeze(1)

print(f"[INFO] Loaded {len(flat_data)} transitions")


# === Dataset and Dataloader ===
class BCDataset(Dataset):
    def __init__(self, obs, state, actions):
        self.obs = obs
        self.state = state
        self.actions = actions

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.state[idx], self.actions[idx]

dataset = BCDataset(obs_tensor, state_tensor, action_tensor)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

from isaaclab_rl.rl_games import RlGamesVecEnvWrapper, RlGamesGpuEnv
from isaaclab.envs import DirectRLEnvCfg, DirectMARLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config
import isaaclab_tasks.direct.factory  # ensure tasks are registered

# === Hydra entry point ===
@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg | DirectMARLEnvCfg | ManagerBasedRLEnvCfg, agent_cfg: dict):
    # Device setup
    device = env_cfg.sim.device = agent_cfg["params"]["config"].get("device", "cuda:0")
    agent_cfg["params"]["config"]["multi_gpu"] = False

    # Build gym env
    import gymnasium as gym
    env = gym.make(args_cli.task, cfg=env_cfg)
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", float("inf"))
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", float("inf"))
    env = RlGamesVecEnvWrapper(env, device, clip_obs, clip_actions)

    # Register for RL-Games
    vecenv.register("IsaacRlgWrapper", lambda name, num_actors, **kwargs: RlGamesGpuEnv(name, num_actors, **kwargs))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # Num actors
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    # Load agent using Runner (automatically loads correct type)
    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)
    runner.reset()

    agent = runner.algo_factory.create(runner.algo_name, base_name='run', params=runner.params)
    actor = agent.model
    actor.train()
    print("actor:", actor)

    # Train with behavior cloning
    optimizer = optim.Adam(actor.parameters(), lr=3e-4)
    loss_fn = nn.MSELoss()

    num_epochs = 400

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for obs_batch, state_batch, act_batch in dataloader:
            obs_batch, act_batch = obs_batch.to(device), act_batch.to(device)
            batch_size = obs_batch.shape[0]

            # Create dummy RNN states (usually 2 tensors: h and c)
            # This assumes 2-layer LSTM with hidden_size 1024 — modify if different
            num_layers = 2
            hidden_size = 1024
            rnn_states = (
                torch.zeros((num_layers, batch_size, hidden_size), device=device),
                torch.zeros((num_layers, batch_size, hidden_size), device=device),
            )

            input_dict = {
                "obs": obs_batch,
                "rnn_states": rnn_states,
                "is_train": True
            }

            # Get action distribution output
            # out = actor.eval_act(input_dict)
            # print(f"out: {out}")

            # pred_actions = out["actions"]

            # action_output = agent.get_action(input_dict)
            # pred_actions = action_output["actions"]

            pred_actions = actor.a2c_network(input_dict)[0]

            loss = loss_fn(pred_actions, act_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {epoch_loss / len(dataloader):.6f}")

    # Save model
    save_dir = os.path.join("logs", "teleoperation", args_cli.task)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "teleop_factory")
    print("[INFO] Saving checkpoint to", path)
    agent.save(path)

    # Done
    env.close()
    simulation_app.close()
    print("[INFO] Behavior cloning complete.")


if __name__ == "__main__":
    main()
