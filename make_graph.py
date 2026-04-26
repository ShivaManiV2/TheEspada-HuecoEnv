import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the real data
df = pd.read_csv("data/training_adaptive_survival.csv")

# Clean, professional plot
fig, ax = plt.subplots(figsize=(10, 6))

# Raw data + smoothed line
if df['survival_rate'].nunique() > 1:
    df['smooth'] = df['survival_rate'].rolling(window=30, min_periods=1).mean()
    ax.plot(df['episode'], df['survival_rate'], color='#93c5fd', alpha=0.3, lw=1, label='Per Episode')
    ax.plot(df['episode'], df['smooth'], color='#2563eb', lw=2.5, label='Rolling Average (30 ep)')
else:
    ax.plot(df['episode'], df['survival_rate'], color='#2563eb', lw=2.5, label='Survival Rate')

ax.set_title('Qwen3-1.7B GRPO Training — Agent Survival Rate', fontsize=15, fontweight='bold', pad=12)
ax.set_xlabel('Training Episode', fontsize=12)
ax.set_ylabel('Survival Rate', fontsize=12)
ax.set_ylim(0.0, 1.05)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.2)
fig.tight_layout()

os.makedirs("assets", exist_ok=True)
plt.savefig("assets/survival_plot.png", dpi=300, bbox_inches='tight')
print("Graph saved to assets/survival_plot.png")
