import torch
from stable_baselines3 import PPO
import gym
import os
import sys
import numpy as np
from pyvirtualdisplay import Display
from gym.wrappers import RecordVideo
import wandb
# Setup progetto e path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from env.custom_hopper import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Avvia un display virtuale per evitare errori di rendering
display = Display(visible=0, size=(1400, 900))
display.start()

# Inizializza wandb
wandb.init(project="RL_gym_hopper", name="PPO_test_comparison_v4", config={
    "episodes": 50,
    "environments": ["source->source", "source->target","target->target"]
})

# Parametri di test
test_episodes = 50

# Configurazioni: (etichetta, modello_path, env_id)
configs = [
    ("source->source", "models/PPO_1M_SOURCE_V4.mdl", "CustomHopper-source-v0"),
    ("source->target", "models/PPO_1M_SOURCE_V4.mdl", "CustomHopper-target-v0"),
    ("target->target", "models/PPO_1M_TARGET_V4.mdl", "CustomHopper-target-v0"),
]

all_results = {}
box_data = []

for label, model_path, env_id in configs:
    print(f"\nTesting: {label}")

    # Carica modello
    model = PPO.load(model_path)

    # Registra solo il video dell’ultima ep. del primo test
    env = gym.make(env_id)

    env = RecordVideo(env, video_folder=f"./videos/{label}/", episode_trigger=lambda ep: ep == test_episodes - 1)

    returns = []

    for ep in range(test_episodes):
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward

        returns.append(total_reward)
        box_data.append([label, total_reward])
        print(f"Episode {ep + 1}: {total_reward:.2f}")

    avg_return = np.mean(returns)
    std_return = np.std(returns)
    all_results[label] = (avg_return, std_return)

    wandb.log({
        f"{label}/avg_return": avg_return,
        f"{label}/std_return": std_return
    })

    env.close()
    print(f"Video salvato nella cartella ./videos/{label}")

# Log grafico a barre
bar_table = wandb.Table(data=[[k, v[0]] for k, v in all_results.items()], columns=["config", "avg_return"])
wandb.log({
    "Average Return (Bar Chart)": wandb.plot.bar(
        bar_table,
        "config", "avg_return",
        title="Average Return per Configuration"
    )
})
# Converti a DataFrame per facilità
df = pd.DataFrame(box_data, columns=["Configuration", "Reward"])

# Plot con seaborn/matplotlib
plt.figure(figsize=(10, 8))
sns.boxplot(x="Configuration", y="Reward", data=df,palette=["green"] * df["Configuration"].nunique())
plt.title("Reward Distribution per Configuration")
plt.ylabel("Reward")
plt.grid(True)

# Salva e logga con wandb
plt.savefig("boxplot_v4.png")
wandb.log({"Reward Distribution (BoxPlot)": wandb.Image("boxplot_v4.png")})



wandb.finish()


