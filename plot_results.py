import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data
df = pd.read_csv("data/training_adaptive_survival.csv")

# Create plot
plt.figure(figsize=(10, 6))
sns.set_theme(style="darkgrid")

# Since survival_rate might be 1.0 for many, let's plot a rolling average to make it look smooth
if len(df) > 50:
    df['rolling_survival'] = df['survival_rate'].rolling(window=50, min_periods=1).mean()
    sns.lineplot(data=df, x='episode', y='rolling_survival', color='#00ffcc', linewidth=2)
else:
    sns.lineplot(data=df, x='episode', y='survival_rate', color='#00ffcc', linewidth=2)

plt.title("HuecoEnv Agent Survival Rate over Episodes", fontsize=16, color='white')
plt.xlabel("Training Episode", fontsize=12, color='white')
plt.ylabel("Survival Rate", fontsize=12, color='white')

# Make the plot look cyberpunk
ax = plt.gca()
ax.set_facecolor('#0d0d1a')
ax.figure.set_facecolor('#0d0d1a')
ax.tick_params(colors='white', which='both')
for spine in ax.spines.values():
    spine.set_edgecolor('#33334d')

# Ensure directory exists
os.makedirs("assets", exist_ok=True)
plt.savefig("assets/survival_plot.png", dpi=300, bbox_inches='tight')
print("Plot saved to assets/survival_plot.png")
