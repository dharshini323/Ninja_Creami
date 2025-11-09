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
    # Basic defaults / hyperparams
    alpha = 0.1          # learning rate
    gamma = 0.9          # discount factor
    epsilon = 0.05        # exploration probability (you can vary per-run)

    # Safety: ensure we have a board
    board_state = state.get("board")
    if not board_state:
        return jsonify({"move": "UP"}), 200

    HEIGHT, WIDTH = len(board_state), len(board_state[0])

    # Get current and opponent info robustly from trails
    cur_pos = (0, 0)
    if hasattr(my_agent, "trail") and my_agent.trail:
        cur_pos = tuple(my_agent.trail[-1])
    opp_agent = GLOBAL_GAME.agent2 if player_number == 1 else GLOBAL_GAME.agent1
    opp_pos = (0, 0)
    if hasattr(opp_agent, "trail") and opp_agent.trail:
        opp_pos = tuple(opp_agent.trail[-1])
    opp_dir = getattr(opp_agent, "last_move", None)

    boosts_remaining = getattr(my_agent, "boosts_remaining", 0)

    # Build grid marking trails and predicted opponent cell
    board = [[0] * WIDTH for _ in range(HEIGHT)]
    if hasattr(my_agent, "trail"):
        for (x, y) in my_agent.trail:
            if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                board[y][x] = 1
    if hasattr(opp_agent, "trail"):
        for (x, y) in opp_agent.trail:
            if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                board[y][x] = 2

    # Predict opponent next cell to block
    dir_map = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
    if opp_dir in dir_map:
        dx, dy = dir_map[opp_dir]
        nx, ny = opp_pos[0] + dx, opp_pos[1] + dy
        if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
            board[ny][nx] = 2

    # Moves and neighbor helpers
    moves = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
    def neighbors_free(x, y):
        out = []
        for m, (dx, dy) in moves.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < WIDTH and 0 <= ny < HEIGHT and board[ny][nx] == 0:
                out.append((nx, ny, m))
        return out

    def bfs_safe_moves(start):
        sx, sy = start
        safe = []
        for m, (dx, dy) in moves.items():
            nx, ny = sx + dx, sy + dy
            if 0 <= nx < WIDTH and 0 <= ny < HEIGHT and board[ny][nx] == 0:
                safe.append(m)
        # if no legal moves, return all (judge will handle collisions)
        return safe if safe else list(moves.keys())

    safe_moves = bfs_safe_moves(cur_pos)

    # Build a compact state representation for Q
    def build_state_tuple():
        rel_x = cur_pos[0] - opp_pos[0]
        rel_y = cur_pos[1] - opp_pos[1]
        boosts = int(boosts_remaining)
        n_safe = len(safe_moves)
        danger = 1 if n_safe < 2 else 0
        # keep tuple small to keep Q-table manageable
        return (rel_x, rel_y, boosts, danger)

    state_tuple = build_state_tuple()

    # Q-table file (per-player)
    qfile = f"qtable_agent{player_number}.pkl"
    try:
        with open(qfile, "rb") as fh:
            Q = pickle.load(fh)
    except Exception:
        Q = {}

    # Ensure state entry
    if state_tuple not in Q:
        Q[state_tuple] = {a: 0.0 for a in moves.keys()}

    # ε-greedy action selection (choose from safe moves only)
    if random.random() < epsilon:
        action = random.choice(safe_moves)
    else:
        # pick best among safe_moves (fall back to any if unseen)
        q_candidates = {a: Q[state_tuple].get(a, 0.0) for a in safe_moves}
        # if tie, max picks first — break ties randomly
        best_val = max(q_candidates.values())
        best_actions = [a for a, v in q_candidates.items() if v == best_val]
        action = random.choice(best_actions)

    # Compute reward (single unified reward for this timestep)
    reward = 0.0
    # survival
    reward += 1.0 if getattr(my_agent, "alive", True) else -10.0
    # prefer more options
    reward += len(safe_moves) * 0.2
    # penalty if few options
    if len(safe_moves) < 2:
        reward -= 2.0
    # small bias to open space (count free neighbors)
    x0, y0 = cur_pos
    open_space = sum(1 for _, (dx, dy) in moves.items()
                     if 0 <= x0 + dx < WIDTH and 0 <= y0 + dy < HEIGHT and board[y0 + dy][x0 + dx] == 0)
    reward += open_space * 0.1
    # encourage moderate aggression (close to opponent) but not too much
    dist = abs(cur_pos[0] - opp_pos[0]) + abs(cur_pos[1] - opp_pos[1])
    reward += max(0, (10 - dist)) * 0.03
    # tiny noise to avoid repeat-determinism
    reward += random.uniform(-0.02, 0.02)

    # Bellman Q-update (use action without :BOOST)
    new_state_tuple = build_state_tuple()  # after move would be updated by environment; we use same approx.
    if new_state_tuple not in Q:
        Q[new_state_tuple] = {a: 0.0 for a in moves.keys()}
    best_future_q = max(Q[new_state_tuple].values()) if Q[new_state_tuple] else 0.0
    Q[state_tuple][action] += alpha * (reward + gamma * best_future_q - Q[state_tuple][action])

    # Decide boost (do not change action key used for Q)
    move = action
    if boosts_remaining > 0 and random.random() < 0.25 and len(safe_moves) > 2:
        move = action + ":BOOST"

    # Persist Q-table safely
    try:
        with open(qfile, "wb") as fh:
            pickle.dump(Q, fh)
    except Exception as e:
        print(f"[Q SAVE ERROR] {e}")

    # --- Training data logging (JSONL) ---
    try:
        os.makedirs("training_data", exist_ok=True)
        log_filename = f"training_data/agent_{player_number}_log.jsonl"
        timestamp = datetime.now().isoformat()

        log_entry = {
            "timestamp": timestamp,
            "player_number": player_number,
            "turn": state.get("turn_count", 0),
            "position": list(cur_pos),
            "action": action,
            "move": move,
            "reward": reward,
            "boosts_remaining": boosts_remaining,
            "alive": getattr(my_agent, "alive", True),
            "safe_moves": len(safe_moves)
        }

        with open(log_filename, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    except Exception as e:
        print(f"[WARNING] Logging failed: {e}")

    # -----------------end code here--------------------


    return jsonify({"move": move}), 200

@app.route("/end", methods=["POST"])
def end_game():
    """Judge notifies agent that the match finished and provides final state.

    We use this to apply a strong terminal reward, finalize training logs,
    and persist the Q-table for long-term learning.
    """
    data = request.get_json()
    if data:
        _update_local_game_from_post(data)

    try:
        # --- Final Q-learning update ---
        # Reload Q-table (shared file)
        try:
            Q = pickle.load(open("qtable.pkl", "rb"))
        except Exception:
            Q = {}

        # Determine outcome and assign terminal reward
        agent1_alive = bool(data.get("agent1_alive", True))
        agent2_alive = bool(data.get("agent2_alive", True))

        # Assign rewards
        if agent1_alive and not agent2_alive:
            final_reward = 50     # strong reward for winning
        elif not agent1_alive and agent2_alive:
            final_reward = -50    # big penalty for losing
        else:
            final_reward = 0      # draw

        # Apply terminal reward to last known state
        if "agent1_trail" in data and data["agent1_trail"]:
            last_pos = tuple(data["agent1_trail"][-1])
            state_tuple = (last_pos[0], last_pos[1])
            if state_tuple in Q:
                for a in Q[state_tuple]:
                    Q[state_tuple][a] += 0.1 * (final_reward - Q[state_tuple][a])

        # Persist updated Q-table
        pickle.dump(Q, open("qtable.pkl", "wb"))

        # --- Log summary for analysis ---
        os.makedirs("training_data", exist_ok=True)
        log_filename = "training_data/game_summaries.jsonl"
        summary = {
            "timestamp": datetime.now().isoformat(),
            "winner": "agent1" if agent1_alive else "agent2" if agent2_alive else "draw",
            "turns": data.get("turn_count", None),
            "final_reward": final_reward,
            "agent1_alive": agent1_alive,
            "agent2_alive": agent2_alive,
        }
        with open(log_filename, "a") as f:
            f.write(json.dumps(summary) + "\n")

        print(f"[INFO] Game ended. Winner: {summary['winner']} Reward: {final_reward}")

    except Exception as e:
        print(f"[ERROR] end_game(): {e}")

    return jsonify({"status": "acknowledged"}), 200



'''
@app.route("/end", methods=["POST"])
def end_game():
    """Judge notifies agent that the match finished and provides final state.

    We update local state for record-keeping and return OK.
    """
    data = request.get_json()
    if data:
        _update_local_game_from_post(data)
    return jsonify({"status": "acknowledged"}), 200
'''


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5009"))
    app.run(host="0.0.0.0", port=port, debug=True)
