import os
import openai
from agent import Agent
from utils import safe_json_parse

class PromptedLLMAgent(Agent):
    """
    Agent that uses a prompted LLM (OpenAI API) to select actions given text observations.
    The policy prompt describes the task; the LLM emits an action as JSON.

    Args:
        action_space: RL env action space (discrete).
        system_prompt: Task/system prompt for LLM.
        model: OpenAI model name (default: 'gpt-3.5-turbo').
        api_key: OpenAI API key (default: from env OPENAI_API_KEY).
    """
    def __init__(self, action_space, system_prompt=None, model="gpt-3.5-turbo", api_key=None):
        self.action_space = action_space
        self.system_prompt = system_prompt or "You are an RL agent. Given a text observation, output a JSON action."
        self.model = model
        openai.api_key = api_key or os.getenv("OPENAI_API_KEY")

    def act(self, observation):
        """
        Given a text observation, prompt the LLM and parse its output as an action.
        Returns:
            action (int): Action index (for discrete spaces)
        """
        prompt = f"Observation: {observation}\nOutput your action as a JSON: {{\"action\": <int>}}"
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=32,
                temperature=0.3
            )
            output = response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            # Fallback: random action
            return self.action_space.sample()

        parsed = safe_json_parse(output)
        if parsed and "action" in parsed:
            action = parsed["action"]
            # Validate action for discrete action space
            if hasattr(self.action_space, "n") and isinstance(action, int) and 0 <= action < self.action_space.n:
                return action
        # Fallback: random action
        return self.action_space.sample()
