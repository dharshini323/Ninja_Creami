import os
import uuid
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque

from case_closed_game import Game, Direction, GameResult

#My imports
import random, pickle, heapq, os
import json
from datetime import datetime

# Flask API server setup
app = Flask(__name__)

GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}

game_lock = Lock()
 
PARTICIPANT = "ParticipantX"
AGENT_NAME = "AgentX"


@app.route("/", methods=["GET"])
def info():
    """Basic health/info endpoint used by the judge to check connectivity.

    Returns participant and agent_name (so Judge.check_latency can create Agent objects).
    """
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200


def _update_local_game_from_post(data: dict):
    """Update the local GLOBAL_GAME using the JSON posted by the judge.

    The judge posts a dictionary with keys matching the Judge.send_state payload
    (board, agent1_trail, agent2_trail, agent1_length, agent2_length, agent1_alive,
    agent2_alive, agent1_boosts, agent2_boosts, turn_count).
    """
    with game_lock:
        LAST_POSTED_STATE.clear()
        LAST_POSTED_STATE.update(data)

        if "board" in data:
            try:
                GLOBAL_GAME.board.grid = data["board"]
            except Exception:
                pass

        if "agent1_trail" in data:
            GLOBAL_GAME.agent1.trail = deque(tuple(p) for p in data["agent1_trail"]) 
        if "agent2_trail" in data:
            GLOBAL_GAME.agent2.trail = deque(tuple(p) for p in data["agent2_trail"]) 
        if "agent1_length" in data:
            GLOBAL_GAME.agent1.length = int(data["agent1_length"])
        if "agent2_length" in data:
            GLOBAL_GAME.agent2.length = int(data["agent2_length"])
        if "agent1_alive" in data:
            GLOBAL_GAME.agent1.alive = bool(data["agent1_alive"])
        if "agent2_alive" in data:
            GLOBAL_GAME.agent2.alive = bool(data["agent2_alive"])
        if "agent1_boosts" in data:
            GLOBAL_GAME.agent1.boosts_remaining = int(data["agent1_boosts"])
        if "agent2_boosts" in data:
            GLOBAL_GAME.agent2.boosts_remaining = int(data["agent2_boosts"])
        if "turn_count" in data:
            GLOBAL_GAME.turns = int(data["turn_count"])


@app.route("/send-state", methods=["POST"])
def receive_state():
    """Judge calls this to push the current game state to the agent server.

    The agent should update its local representation and return 200.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    _update_local_game_from_post(data)
    return jsonify({"status": "state received"}), 200


@app.route("/send-move", methods=["GET"])

def send_move():
    """Judge calls this (GET) to request the agent's move for the current tick.

    Query params the judge sends (optional): player_number, attempt_number,
    random_moves_left, turn_count. Agents can use this to decide.
    
    Return format: {"move": "DIRECTION"} or {"move": "DIRECTION:BOOST"}
    where DIRECTION is UP, DOWN, LEFT, or RIGHT
    and :BOOST is optional to use a speed boost (move twice)
    """
    player_number = request.args.get("player_number", default=1, type=int)

    with game_lock:
        state = dict(LAST_POSTED_STATE)   
        my_agent = GLOBAL_GAME.agent1 if player_number == 1 else GLOBAL_GAME.agent2
        boosts_remaining = my_agent.boosts_remaining
   
    # -----------------your code here-------------------
    
    # 1. Configuration (Exploitation Mode)
    epsilon = 0.0 # Set to 0.0 for final submission: pure exploitation
    
    # 2. Local BFS Helper Function (Must be defined here due to constraint)
    def calculate_open_area(start_pos, board, width, height):
        """Calculates the size of the largest reachable open area (BFS)."""
        q = deque([start_pos])
        visited = {start_pos}
        area = 0
        moves_dir = [(0, -1), (0, 1), (-1, 0), (1, 0)] 

        while q:
            x, y = q.popleft()
            area += 1

            for dx, dy in moves_dir:
                # Torus (Wraparound) check
                nx = (x + dx) % width
                ny = (y + dy) % height
                next_pos = (nx, ny)

                # 0 is EMPTY cell
                if next_pos not in visited and board[ny][nx] == 0:
                    visited.add(next_pos)
                    q.append(next_pos)
        return area

    # Safety: ensure we have a board
    board_state = state.get("board")
    if not board_state:
        # Emergency fallback
        return jsonify({"move": "UP"}), 200

    HEIGHT, WIDTH = len(board_state), len(board_state[0])

    # Get current and opponent info robustly
    cur_pos = (0, 0)
    if hasattr(my_agent, "trail") and my_agent.trail:
        cur_pos = tuple(my_agent.trail[-1])

    # Build grid marking trails (0=Empty, 1=My Trail, 2=Opponent Trail)
    # Using a local copy of the board to mark opponent's predicted move as blocked
    board = [[0] * WIDTH for _ in range(HEIGHT)]
    
    # Add all trails (using raw state data for robustness)
    for (x, y) in state.get("agent1_trail", []):
        if 0 <= x < WIDTH and 0 <= y < HEIGHT: board[y][x] = 1
    for (x, y) in state.get("agent2_trail", []):
        if 0 <= x < WIDTH and 0 <= y < HEIGHT: board[y][x] = 2

    # Moves and safe move helpers
    moves = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
    safe_moves = []
    
    for m, (dx, dy) in moves.items():
        nx = (cur_pos[0] + dx) % WIDTH
        ny = (cur_pos[1] + dy) % HEIGHT
        if board[ny][nx] == 0:
            safe_moves.append(m)

    # 3. Load Combined Q-table (Must handle potential file error)
    qfile = "qtable_combined.pkl" # Assuming this is your final file name
    try:
        with open(qfile, "rb") as fh:
            Q = pickle.load(fh)
    except Exception:
        Q = {} # Fallback to empty Q-table if file is missing

    # Build a compact state representation (must match training state)
    opp_agent = GLOBAL_GAME.agent2 if player_number == 1 else GLOBAL_GAME.agent1
    opp_pos = tuple(opp_agent.trail[-1]) if hasattr(opp_agent, "trail") and opp_agent.trail else (0, 0)

    rel_x = cur_pos[0] - opp_pos[0]
    rel_y = cur_pos[1] - opp_pos[1]
    boosts = int(boosts_remaining)
    n_safe = len(safe_moves)
    danger = 1 if n_safe < 2 else 0
    state_tuple = (rel_x, rel_y, boosts, danger)


    # 4. Q-Greedy Action Selection with BFS Tie-Breaker
    action = "UP" # Default safety move

    if state_tuple not in Q or epsilon > 0.0:
        # Use simple move logic if Q-state is unseen or exploring (epsilon=0.0 will skip this)
        if safe_moves:
             action = random.choice(safe_moves)
        else:
             action = random.choice(list(moves.keys())) # Truly random move
    
    else:
        # Pure Exploitation (epsilon = 0.0)
        q_candidates = {a: Q[state_tuple].get(a, 0.0) for a in safe_moves}
        
        if not q_candidates:
             # Emergency: If no safe Q-moves, take any safe direction
             action = random.choice(safe_moves) if safe_moves else random.choice(list(moves.keys()))
        else:
            # Find the highest Q-value
            best_val = max(q_candidates.values())
            # Find ALL actions that match the best Q-value (the tie)
            best_actions = [a for a, v in q_candidates.items() if v == best_val]
            
            # Apply Open Space (BFS) as a Tie-Breaker
            if len(best_actions) > 1:
                best_area = -1
                best_move_by_area = best_actions[0] 

                for a in best_actions:
                    dx, dy = moves[a]
                    next_pos = (cur_pos[0] + dx, cur_pos[1] + dy)
                    nx = next_pos[0] % WIDTH
                    ny = next_pos[1] % HEIGHT
                    
                    # Calculate open area from the resulting position
                    area = calculate_open_area((nx, ny), board, WIDTH, HEIGHT)
                    
                    if area > best_area:
                        best_area = area
                        best_move_by_area = a
                
                action = best_move_by_area
            else:
                # No tie, just use the single best Q-action
                action = best_actions[0]
    
    # 5. Final Move Decision (Action + Boost)
    move = action
    
    # Greedy Boost Logic: Use a boost if it leads to a significantly larger area
    # (Checking the best Q-action's resulting area against a threshold)
    if boosts_remaining > 0 and len(safe_moves) > 3:
        dx, dy = moves[action]
        nx = (cur_pos[0] + dx) % WIDTH
        ny = (cur_pos[1] + dy) % HEIGHT
        area_after_move = calculate_open_area((nx, ny), board, WIDTH, HEIGHT)
        
        # Threshold: boost if the area is large (e.g., > 40 cells)
        if area_after_move > 40: 
            move = action + ":BOOST"

    # -----------------end code here--------------------


    return jsonify({"move": move}), 200


# @app.route("/end", methods=["POST"])
# def end_game():
#     """Judge notifies agent that the match finished and provides final state.

#     We use this to apply a strong terminal reward, finalize training logs,
#     and persist the Q-table for long-term learning.
#     """
#     data = request.get_json()
#     if data:
#         _update_local_game_from_post(data)

#     try:
#         # --- Final Q-learning update ---
#         # Reload Q-table (shared file)
#         try:
#             Q = pickle.load(open("qtable.pkl", "rb"))
#         except Exception:
#             Q = {}

#         # Determine outcome and assign terminal reward
#         agent1_alive = bool(data.get("agent1_alive", True))
#         agent2_alive = bool(data.get("agent2_alive", True))

#         # Assign rewards
#         if agent1_alive and not agent2_alive:
#             final_reward = 50     # strong reward for winning
#         elif not agent1_alive and agent2_alive:
#             final_reward = -50    # big penalty for losing
#         else:
#             final_reward = 0      # draw

#         # Apply terminal reward to last known state
#         if "agent1_trail" in data and data["agent1_trail"]:
#             last_pos = tuple(data["agent1_trail"][-1])
#             state_tuple = (last_pos[0], last_pos[1])
#             if state_tuple in Q:
#                 for a in Q[state_tuple]:
#                     Q[state_tuple][a] += 0.1 * (final_reward - Q[state_tuple][a])

#         # Persist updated Q-table
#         pickle.dump(Q, open("qtable.pkl", "wb"))

#         # --- Log summary for analysis ---
#         os.makedirs("training_data", exist_ok=True)
#         log_filename = "training_data/game_summaries.jsonl"
#         summary = {
#             "timestamp": datetime.now().isoformat(),
#             "winner": "agent1" if agent1_alive else "agent2" if agent2_alive else "draw",
#             "turns": data.get("turn_count", None),
#             "final_reward": final_reward,
#             "agent1_alive": agent1_alive,
#             "agent2_alive": agent2_alive,
#         }
#         with open(log_filename, "a") as f:
#             f.write(json.dumps(summary) + "\n")

#         print(f"[INFO] Game ended. Winner: {summary['winner']} Reward: {final_reward}")

#     except Exception as e:
#         print(f"[ERROR] end_game(): {e}")

#     return jsonify({"status": "acknowledged"}), 200


@app.route("/end", methods=["POST"])
def end_game():
    """Judge notifies agent that the match finished and provides final state.

    We update local state for record-keeping and return OK.
    """
    data = request.get_json()
    if data:
        _update_local_game_from_post(data)
    return jsonify({"status": "acknowledged"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5008"))
    app.run(host="0.0.0.0", port=port, debug=True)
