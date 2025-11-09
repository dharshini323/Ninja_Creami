import os
import uuid
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque

from case_closed_game import Game, Direction, GameResult

#My imports
import heapq

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
    # Simple example: always go RIGHT (replace this with your logic)
    # To use a boost: move = "RIGHT:BOOST"
    move = "RIGHT"
    
    # Example: Use boost if available and it's late in the game
    # turn_count = state.get("turn_count", 0)
    # if boosts_remaining > 0 and turn_count > 50:
    #     move = "RIGHT:BOOST"

        # -----------------your code here-------------------

    # Safety check for missing board
    board_state = state.get("board")
    if not board_state:
        return jsonify({"move": "UP"}), 200

    HEIGHT, WIDTH = len(board_state), len(board_state[0])

    # Current and opponent positions from trails
    cur_pos = my_agent.trail[-1] if my_agent.trail else (0, 0)
    opp_agent = GLOBAL_GAME.agent2 if player_number == 1 else GLOBAL_GAME.agent1
    opp_pos = opp_agent.trail[-1] if opp_agent.trail else (0, 0)
    # Estimate opponent's last direction if possible
    if len(opp_agent.trail) >= 2:
        (x1, y1), (x2, y2) = list(opp_agent.trail)[-2:]
        if x2 > x1:
            opp_dir = "RIGHT"
        elif x2 < x1:
            opp_dir = "LEFT"
        elif y2 > y1:
            opp_dir = "DOWN"
        else:
            opp_dir = "UP"
    else:
        opp_dir = "UP"


    # Build board marking trails
    board = [[0] * WIDTH for _ in range(HEIGHT)]
    for (x, y) in my_agent.trail:
        if 0 <= y < HEIGHT and 0 <= x < WIDTH:
            board[y][x] = 1
    for (x, y) in opp_agent.trail:
        if 0 <= y < HEIGHT and 0 <= x < WIDTH:
            board[y][x] = 2

    # Predict opponent’s next move
    dir_map = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
    dx, dy = dir_map.get(opp_dir, (0, -1))
    nx, ny = opp_pos[0] + dx, opp_pos[1] + dy
    if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
        board[ny][nx] = 2

    # Neighbor function
    def neighbors(x, y):
        result = []
        for move, (dx, dy) in dir_map.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < WIDTH and 0 <= ny < HEIGHT and board[ny][nx] == 0:
                result.append((nx, ny, move))
        return result

    # Simple A* to choose direction
    def a_star(start):
        frontier = [(0, start, [])]
        visited = set()
        while frontier:
            _, (x, y), path = heapq.heappop(frontier)
            if (x, y) in visited:
                continue
            visited.add((x, y))
            if len(path) >= 1:
                return path[0]
            for nx, ny, move in neighbors(x, y):
                if (nx, ny) not in visited:
                    h = abs(nx - WIDTH // 2) + abs(ny - HEIGHT // 2)
                    heapq.heappush(frontier, (len(path) + 1 + h, (nx, ny), path + [move]))
        return "UP"

    move = a_star(cur_pos)

    # Use boost only when safe
    if boosts_remaining > 0:
        safe_neighbors = neighbors(cur_pos[0], cur_pos[1])
        if len(safe_neighbors) >= 2:
            move += ":BOOST"

    # -----------------end code here--------------------


    # -----------------end code here--------------------

    return jsonify({"move": move}), 200


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
