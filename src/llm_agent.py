import os
import random
import json
import time
from typing import Any, Dict, Optional, Callable
from tenacity import retry, stop_after_attempt, wait_fixed

from agent import Agent

try:
    import openai
except ImportError:
    openai = None

class PromptedLLMAgent(Agent):
    """
    Agent that wraps an LLM as a policy. Prompts the model with observation, expects action in JSON.
    Falls back to random action if parsing fails. Tracks token/latency stats.

    Args:
        action_space: gym action space
        system_prompt: string for LLM system prompt
        model: OpenAI model name (default: gpt-4-0613)
        temperature: sampling temperature for LLM
        template: observation->prompt formatter (optional)
        template_kwargs: dict for prompt template formatting
    """
    def __init__(self,
                 action_space,
                 system_prompt: str = "You are an RL agent.",
                 model: str = "gpt-4-0613",
                 temperature: float = 0.7,
                 template: Optional[Callable[[Any], str]] = None,
                 template_kwargs: Optional[Dict[str, Any]] = None):
        self.action_space = action_space
        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature
        self.template = template
        self.template_kwargs = template_kwargs or {}
        self.stats = {
            'total_tokens': 0,
            'total_calls': 0,
            'total_latency': 0.0,
            'parse_failures': 0,
            'fallback_random': 0,
        }
        self.last_prompt = None
        self.last_response = None
        self.last_action = None
        self._rng = random.Random()

        if openai is None:
            raise ImportError("openai package not available. Install openai for PromptedLLMAgent.")
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY environment variable required for OpenAI LLM access.")

    def reset(self):
        self.last_prompt = None
        self.last_response = None
        self.last_action = None
        self.stats = {
            'total_tokens': 0,
            'total_calls': 0,
            'total_latency': 0.0,
            'parse_failures': 0,
            'fallback_random': 0,
        }

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(0.5))
    def _call_llm(self, prompt: str) -> str:
        """
        Call OpenAI chat completion, return response text.
        """
        start = time.time()
        resp = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=64
        )
        latency = time.time() - start
        self.stats['total_latency'] += latency
        self.stats['total_calls'] += 1
        self.stats['total_tokens'] += resp['usage']['total_tokens'] if 'usage' in resp else 0
        return resp['choices'][0]['message']['content']

    def _format_prompt(self, observation: Any) -> str:
        """
        Format prompt from observation using template if provided.
        """
        if self.template:
            return self.template(observation, **self.template_kwargs)
        if isinstance(observation, str):
            return observation
        # Fallback: dump dict or list as JSON string
        return json.dumps(observation)

    def _parse_action(self, llm_output: str) -> Any:
        """
        Try to extract action from LLM output (expects JSON like {"action": 0}).
        Returns parsed action or None if parsing fails.
        """
        try:
            obj = json.loads(llm_output)
            if 'action' in obj:
                return obj['action']
        except Exception:
            pass
        return None

    def act(self, observation: Any) -> Any:
        prompt = self._format_prompt(observation)
        self.last_prompt = prompt
        try:
            resp = self._call_llm(prompt)
            self.last_response = resp
            action = self._parse_action(resp)
            if action is None:
                self.stats['parse_failures'] += 1
                action = self.action_space.sample()
                self.stats['fallback_random'] += 1
            else:
                # Validate action
                if hasattr(self.action_space, 'contains') and not self.action_space.contains(action):
                    action = self.action_space.sample()
                    self.stats['fallback_random'] += 1
            self.last_action = action
            return action
        except Exception:
            self.stats['parse_failures'] += 1
            self.stats['fallback_random'] += 1
            self.last_response = None
            action = self.action_space.sample()
            self.last_action = action
            return action
