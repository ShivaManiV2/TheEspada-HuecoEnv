import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    print("Generating Presentation Graph...")
    
    baseline_path = "data/simulation_adaptive_survival.csv"
    trained_path = "data/training_adaptive_survival.csv"
    
    if not os.path.exists(baseline_path):
        print(f"Error: Could not find {baseline_path}. Run simulate.py first!")
        return
        
    baseline_data = pd.read_csv(baseline_path)
    
    plt.figure(figsize=(10, 6))
    plt.plot(baseline_data['episode'], baseline_data['survival_rate'], label='Untrained Baseline', color='red', linestyle='--')
    
    if os.path.exists(trained_path):
        trained_data = pd.read_csv(trained_path)
        plt.plot(trained_data['episode'], trained_data['survival_rate'], label='TRL Trained Agent', color='blue', linewidth=2)
        print("Found training data! Adding to graph.")
    else:
        print("Training data not found yet. Plotting baseline only.")

    plt.title('Environment Recovery: Untrained vs Trained Agents')
    plt.xlabel('Episode')
    plt.ylabel('Survival Rate')
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.grid(True)

    plt.savefig('Slide_4_Graph.png', dpi=300, bbox_inches='tight')
    print("Success! Graph saved as Slide_4_Graph.png")

if __name__ == "__main__":
    main()
