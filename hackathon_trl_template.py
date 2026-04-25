"""
HuecoEnv — Hackathon TRL Training Template
=============================================
Use this file during the GPU phase of the hackathon. 
It shows you exactly where to paste the Unsloth/TRL code provided by the organizers
without breaking your environment or your CSV data logging.
"""

import os
import csv
import json

# Import your working environment wrapper!
from train import HuecoGymWrapper

def main():
    print("Starting HuecoEnv RL Training...")
    
    # =====================================================================
    # PHASE 1: MODEL SETUP (Paste Hackathon Code Here)
    # =====================================================================
    # The hackathon guide mentions a "dedicated tutorial flow around TRL and Wordle".
    # They will give you code that looks something like this:
    #
    # from unsloth import FastLanguageModel
    # from trl import GRPOConfig, GRPOTrainer
    # 
    # model, tokenizer = FastLanguageModel.from_pretrained(
    #     model_name = "meta-llama/Llama-3.1-8B-Instruct",
    #     max_seq_length = 2048,
    #     load_in_4bit = True,
    # )
    # =====================================================================
    print("[1] Model initialized (Placeholder)")


    # =====================================================================
    # PHASE 2: ENVIRONMENT HOOKUP
    # =====================================================================
    # Do not change this! This connects your complex JSON world to the TRL trainer.
    task_name = "adaptive_survival"
    wrapper = HuecoGymWrapper(task_name=task_name)
    print(f"[2] Environment hooked up for task: {task_name}")


    # =====================================================================
    # PHASE 3: THE TRAINING LOOP
    # =====================================================================
    # You will use the TRL GRPOTrainer or PPOTrainer here. 
    # It usually wraps around the environment. If you have to write a custom
    # rollout loop, it looks like this:
    
    num_episodes = 500
    training_log = []
    
    for ep in range(1, num_episodes + 1):
        obs = wrapper.reset()
        
        # --- TRL MAGIC HAPPENS HERE ---
        # The trainer will generate actions and call wrapper.step()
        # Example:
        # action = model.generate(obs)
        # next_obs, reward, done, info = wrapper.step(action)
        # trainer.step()
        # ------------------------------
        
        # At the end of the episode, we MUST call wrapper.end_episode() 
        # to trigger the Environment Brain and get the survival rate!
        brain_info = wrapper.end_episode()
        survival_rate = brain_info.get("survival_rate", 0.0)
        
        print(f"Episode {ep}/{num_episodes} | Survival Rate: {survival_rate:.2%}")
        
        # Save to our log so we can graph it later
        training_log.append({
            "episode": ep,
            "survival_rate": survival_rate
        })


    # =====================================================================
    # PHASE 4: SAVE THE GRAPH DATA
    # =====================================================================
    # We save this so you can plot it against your Untrained Baseline!
    os.makedirs("data", exist_ok=True)
    csv_path = f"data/training_{task_name}.csv"
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "survival_rate"])
        for row in training_log:
            writer.writerow([row["episode"], row["survival_rate"]])
            
    print(f"\n[4] Training complete! Data saved to {csv_path}")
    print("Now run make_graph.py to generate your presentation graph!")

if __name__ == "__main__":
    main()
