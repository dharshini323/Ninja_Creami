import pickle
import json
from collections import defaultdict

# --- Load Q-tables ---
with open("qtable_agent1.pkl", "rb") as f:
    Q1 = pickle.load(f)

with open("qtable_agent2.pkl", "rb") as f:
    Q2 = pickle.load(f)

# --- Combine Q-tables ---
# Strategy: average Q-values for each action in shared states, keep unique states
combined_Q = {}

# Add all states from Q1
for state, actions in Q1.items():
    if state in Q2:
        combined_actions = {a: (actions.get(a,0) + Q2[state].get(a,0))/2 for a in set(actions) | set(Q2[state])}
        combined_Q[state] = combined_actions
    else:
        combined_Q[state] = actions

# Add states unique to Q2
for state, actions in Q2.items():
    if state not in combined_Q:
        combined_Q[state] = actions

# --- Save combined Q-table ---
with open("qtable_combined.pkl", "wb") as f:
    pickle.dump(combined_Q, f)

print(f"Combined Q-table saved with {len(combined_Q)} states.")

# --- Compute win/loss/draw probabilities ---
wins, losses, draws = 0, 0, 0
with open("training_data/game_summaries.jsonl", "r") as f:
    for line in f:
        game = json.loads(line)
        if game["winner"] == "agent1":
            wins += 1
        elif game["winner"] == "agent2":
            losses += 1
        else:
            draws += 1

total_games = wins + losses + draws
print("Win/Loss/Draw probabilities:")
print(f"Win: {wins/total_games:.2f}")
print(f"Loss: {losses/total_games:.2f}")
print(f"Draw: {draws/total_games:.2f}")
