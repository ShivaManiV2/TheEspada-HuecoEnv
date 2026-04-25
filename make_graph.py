import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the real data
# Use the _1.csv since that is the one you just downloaded from the A100!
data_path = "data/training_adaptive_survival_1.csv" if os.path.exists("data/training_adaptive_survival_1.csv") else "data/training_adaptive_survival.csv"
df = pd.read_csv(data_path)

# Create the Cyberpunk Plot
plt.figure(figsize=(10, 6))
sns.set_theme(style="darkgrid")

# Plot smoothed line (only smooth if it's not a perfectly flat line)
if df['survival_rate'].nunique() > 1:
    df['rolling_survival'] = df['survival_rate'].rolling(window=30, min_periods=1).mean()
    sns.lineplot(data=df, x='episode', y='survival_rate', color='#00ffcc', alpha=0.3, linewidth=1)
    sns.lineplot(data=df, x='episode', y='rolling_survival', color='#00ffcc', linewidth=2.5, label="Rolling Average")
else:
    sns.lineplot(data=df, x='episode', y='survival_rate', color='#00ffcc', linewidth=2.5, label="Survival Rate")

plt.title("Qwen3-1.7B GRPO Training: Real Agent Survival Rate", fontsize=16, color='white', pad=15)
plt.xlabel("Training Episode", fontsize=12, color='white')
plt.ylabel("Survival Rate", fontsize=12, color='white')
plt.ylim(0.0, 1.05)

# Cyberpunk styling
ax = plt.gca()
ax.set_facecolor('#0d0d1a')
ax.figure.set_facecolor('#0d0d1a')
ax.tick_params(colors='white', which='both')
for spine in ax.spines.values():
    spine.set_edgecolor('#33334d')
plt.legend(facecolor='#0d0d1a', labelcolor='white', edgecolor='#33334d')

# Save
os.makedirs("assets", exist_ok=True)
plt.savefig("assets/survival_plot.png", dpi=300, bbox_inches='tight')
print(f"✅ SUCCESS: Real graph generated from {data_path}!")
