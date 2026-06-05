import os
import openai
from agent import Agent
from utils import safe_json_parse
import time
from tenacity import retry, stop_after_attempt, wait_fixed

class PromptedLLMAgent(Agent):
    """
    Agent that uses a prompted LLM (OpenAI API) to select actions given text observations.
    The policy prompt describes the task; the LLM emits an action as JSON.

    Args:
        action_space: RL env action space (discrete).
        system_prompt: Task/system prompt for LLM.
        model: OpenAI model name (default: 'gpt-3.5-turbo').
        api_key: OpenAI API key (default: from env OPENAI_API_KEY).
        max_retries: Number of action extraction retries (default: 3).
        temperature: Sampling temperature for LLM (default: 0.0 for deterministic output).
    """
    def __init__(self, action_space, system_prompt=None, model="gpt-3.5-turbo", api_key=None, max_retries=3, temperature=0.0):
        self.action_space = action_space
        self.system_prompt = system_prompt or "You are an RL agent. Given a text observation, output a JSON action."
        self.model = model
        openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.max_retries = max_retries
        self.temperature = temperature

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(0.5))
    def _prompt_and_parse_action(self, observation):
        """
        Prompt LLM and extract action JSON. Retries on parse failure or invalid action.
        Returns:
            action (int) if valid, else raises exception to trigger retry.
        """
        prompt = f"Observation: {observation}\nOutput your action as a JSON: {{\"action\": <int>}}"
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=32,
            temperature=self.temperature
        )
        output = response["choices"][0]["message"]["content"].strip()
        parsed = safe_json_parse(output)
        if parsed and "action" in parsed:
            action = parsed["action"]
            # Validate action for discrete action space
            if hasattr(self.action_space, "n") and isinstance(action, int) and 0 <= action < self.action_space.n:
                return action
        # Raise to trigger retry
        raise ValueError(f"Failed to extract valid action from LLM output: {output}")

    def act(self, observation):
        """
        Given a text observation, prompt the LLM and parse its output as an action.
        Retries extraction up to max_retries. Falls back to random action on error.
        Returns:
            action (int): Action index (for discrete spaces)
        """
        try:
            return self._prompt_and_parse_action(observation)
        except Exception:
            # Fallback: random action
            return self.action_space.sample()
