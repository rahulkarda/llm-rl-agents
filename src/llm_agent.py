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
        system_prompt_template: Optional string template for system prompt (supports .format(**kwargs)).
        model: OpenAI model name (default: 'gpt-3.5-turbo').
        api_key: OpenAI API key (default: from env OPENAI_API_KEY).
        max_retries: Number of action extraction retries (default: 3).
        temperature: Sampling temperature for LLM (default: 0.2 for more diverse output).
        instrument: If True, record token usage and latency statistics.
    """
    def __init__(self, action_space, system_prompt=None, system_prompt_template=None, template_kwargs=None, model="gpt-3.5-turbo", api_key=None, max_retries=3, temperature=0.2, instrument=False):
        self.action_space = action_space
        self._base_system_prompt = system_prompt or "You are an RL agent. Given a text observation, output a JSON action."
        self.system_prompt_template = system_prompt_template
        self.template_kwargs = template_kwargs or {}
        self.model = model
        openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.max_retries = max_retries
        self.temperature = temperature
        self.instrument = instrument
        self.token_stats = []  # List of dicts: {"prompt_tokens", "completion_tokens", "total_tokens", "latency_sec"}

    def get_system_prompt(self):
        """
        Return the system prompt, applying template if specified.
        """
        if self.system_prompt_template:
            try:
                return self.system_prompt_template.format(**self.template_kwargs)
            except Exception:
                # fallback to base system prompt
                return self._base_system_prompt
        else:
            return self._base_system_prompt

    def set_system_prompt_template(self, template, template_kwargs=None):
        """
        Set system prompt template and optional keyword arguments for formatting.
        Args:
            template: String template (e.g. "You are playing {env_name}.")
            template_kwargs: Dict for template variables
        """
        self.system_prompt_template = template
        self.template_kwargs = template_kwargs or {}

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(0.5))
    def _prompt_and_parse_action(self, observation):
        """
        Prompt LLM and extract action JSON. Retries on parse failure or invalid action.
        Returns:
            action (int) if valid, else raises exception to trigger retry.
        """
        prompt = f"Observation: {observation}\nOutput your action as a JSON: {{\"action\": <int>}}"
        start_time = time.time() if self.instrument else None
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.get_system_prompt()},
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
                if self.instrument:
                    latency = time.time() - start_time
                    usage = response.get("usage", {})
                    self.token_stats.append({
                        "prompt_tokens": usage.get("prompt_tokens", None),
                        "completion_tokens": usage.get("completion_tokens", None),
                        "total_tokens": usage.get("total_tokens", None),
                        "latency_sec": latency
                    })
                return action
        # Raise to trigger retry
        if self.instrument:
            latency = time.time() - start_time
            self.token_stats.append({"prompt_tokens": None, "completion_tokens": None, "total_tokens": None, "latency_sec": latency})
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

    def get_token_stats(self):
        """
        Return list of token usage and latency stats per call.
        Each item is a dict: {'prompt_tokens', 'completion_tokens', 'total_tokens', 'latency_sec'}
        """
        return self.token_stats

    def reset_stats(self):
        """
        Clear token and latency stats.
        """
        self.token_stats = []
