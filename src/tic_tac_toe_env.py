import numpy as np
import gymnasium as gym
from gymnasium import spaces

class TicTacToeEnv(gym.Env):
    """
    Simple Tic-Tac-Toe environment for RL and self-play.
    Board is 3x3, two players (X: agent, O: opponent).
    Observations: board state as 3x3 array (0: empty, 1: X, 2: O).
    Actions: integer 0-8, placing X in cell (row-major order).
    Rewards: +1 for win, 0 for draw, -1 for loss.
    Episode ends when win, loss, or draw.
    Opponent can be random or heuristic (currently random).
    """
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(9)  # cell index 0-8
        # Observation: 3x3 board, values 0 (empty), 1 (X), 2 (O)
        self.observation_space = spaces.Box(low=0, high=2, shape=(3,3), dtype=np.int32)
        self.board = np.zeros((3,3), dtype=np.int32)
        self.done = False
        self.current_player = 1  # 1: agent (X), 2: opponent (O)
        self.last_action = None
        self._rng = np.random.default_rng()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng()
        self.board = np.zeros((3,3), dtype=np.int32)
        self.done = False
        self.current_player = 1
        self.last_action = None
        obs = self.board.copy()
        return obs, {}

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")
        # Convert action (0-8) to (row, col)
        row, col = divmod(int(action), 3)
        info = {}
        # Validate move
        if self.board[row, col] != 0:
            # Invalid move: penalty, end episode
            reward = -1.0
            self.done = True
            info["invalid_move"] = True
            obs = self.board.copy()
            return obs, reward, self.done, False, info
        # Agent plays X
        self.board[row, col] = 1
        self.last_action = (row, col)
        # Check for win/draw
        result = self._get_result()
        if result == 1:
            reward = 1.0  # agent wins
            self.done = True
            info["result"] = "win"
        elif result == 0 and np.all(self.board != 0):
            reward = 0.0  # draw
            self.done = True
            info["result"] = "draw"
        else:
            # Opponent plays O (random)
            opp_action = self._opponent_action()
            if opp_action is not None:
                r2, c2 = opp_action
                self.board[r2, c2] = 2
            # Check for loss/draw
            result = self._get_result()
            if result == 2:
                reward = -1.0  # agent loses
                self.done = True
                info["result"] = "loss"
            elif result == 0 and np.all(self.board != 0):
                reward = 0.0  # draw
                self.done = True
                info["result"] = "draw"
            else:
                reward = 0.0
                self.done = False
        obs = self.board.copy()
        return obs, reward, self.done, False, info

    def _get_result(self):
        # Check rows, cols, diags for win/loss
        for player in [1,2]:
            for i in range(3):
                if np.all(self.board[i,:] == player) or np.all(self.board[:,i] == player):
                    return player
            if np.all(np.diag(self.board) == player) or np.all(np.diag(np.fliplr(self.board)) == player):
                return player
        return 0  # 0: no winner yet

    def _opponent_action(self):
        # Random available move for O
        empty_cells = list(zip(*np.where(self.board == 0)))
        if not empty_cells:
            return None
        idx = self._rng.integers(len(empty_cells))
        return empty_cells[idx]

    def render(self, mode="human"):
        # Print board
        symbols = {0: ".", 1: "X", 2: "O"}
        rows = [" ".join(symbols[self.board[r,c]] for c in range(3)) for r in range(3)]
        board_str = "\n".join(rows)
        print(board_str)
