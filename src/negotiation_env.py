import gymnasium as gym
from gymnasium import spaces

class NegotiationEnv(gym.Env):
    """
    Simple negotiation environment (phase 3).
    Two agents negotiate for splitting a resource (e.g., 10 tokens).
    Observations: textual description of offers and history.
    Actions: {'offer': int} (propose number of tokens for self).
    Rewards: 1 if agreement, 0 otherwise.
    Episode ends if agreement reached or max_steps exceeded.
    """
    def __init__(self, total_tokens=10, max_steps=8):
        super().__init__()
        self.total_tokens = total_tokens
        self.max_steps = max_steps
        self.action_space = spaces.Dict({'offer': spaces.Discrete(total_tokens+1)})
        self.observation_space = spaces.Text()
        self.steps = 0
        self.current_offer = None
        self.history = []
        self.agreed = False

    def reset(self, seed=None, options=None):
        self.steps = 0
        self.current_offer = None
        self.history = []
        self.agreed = False
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # Accept action as dict or integer
        if isinstance(action, dict) and 'offer' in action:
            offer = int(action['offer'])
        elif isinstance(action, int):
            offer = int(action)
        else:
            offer = 0
        self.current_offer = offer
        self.history.append({'step': self.steps, 'offer': offer})
        self.steps += 1

        # Simple rule: opponent accepts split if offer <= total_tokens/2
        if offer <= self.total_tokens // 2:
            self.agreed = True
            reward = 1
            done = True
        else:
            reward = 0
            done = self.steps >= self.max_steps

        obs = self._get_obs()
        info = {
            'current_offer': offer,
            'history': list(self.history),
            'agreed': self.agreed,
            'step': self.steps
        }
        return obs, reward, done, False, info

    def _get_obs(self):
        desc = f"Negotiation step {self.steps}. "
        if self.current_offer is None:
            desc += f"You may propose a split of {self.total_tokens} tokens."
        else:
            desc += f"Last offer: {self.current_offer} tokens for you. "
            if self.agreed:
                desc += "Agreement reached!"
            else:
                desc += "No agreement yet."
        return desc

    def render(self, mode='human'):
        print(self._get_obs())
        print(f"History: {self.history}")
