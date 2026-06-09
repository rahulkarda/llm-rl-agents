"""
Utility functions for nested dict manipulation and pretty-printing.

Functions:
- flatten_dict: Flattens nested dicts using dotted keys.
- dict_to_str: Pretty-prints (possibly nested) dicts for logging/debugging.
- safe_json_parse: Robustly parse JSON, returning None on failure.
- get_env_name: Extract environment name from gym env or spec.
- is_discrete_space: Check if gym action space is discrete.
- hash_dict: Produce a stable hash for a dict (for caching/debugging).
- deep_copy_dict: Return a deep copy of a dict (for safe mutation).

Usage examples:
    # Flatten a nested dict
    d = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}
    flat = flatten_dict(d)
    # flat: {'a': 1, 'b.c': 2, 'b.d.e': 3}

    # Pretty-print a dict
    s = dict_to_str(d)
    print(s)
    # Output:
    # a: 1
    # b:
    #   c: 2
    #   d:
    #     e: 3

    # Safe JSON parse from LLM output
    obj = safe_json_parse('{"action": 0}')  # returns dict or None

    # Get env name
    import gymnasium as gym
    env = gym.make('CartPole-v1')
    name = get_env_name(env)  # 'CartPole-v1'

    # Check action space type
    from gymnasium.spaces import Discrete, Box
    is_discrete = is_discrete_space(env.action_space)
    # True for Discrete, False for Box

    # Hash a dict
    d = {'foo': 1, 'bar': 2}
    h = hash_dict(d)  # returns an int hash

    # Deep copy a dict
    d2 = deep_copy_dict(d)  # returns a new dict (deep copy)

Notes:
- These utilities are used for agent logging, episode trace formatting, and robust action extraction.
- flatten_dict is useful for flattening nested info dicts for logging or CSV export.
- dict_to_str helps with readable debug output, especially for deeply nested transitions.
- hash_dict enables stable dict hashing for caching or trace comparison.
- deep_copy_dict is useful for safe mutation of dicts, e.g. when storing transitions.
"""
def flatten_dict(d, parent_key='', sep='.'): 
    """Flatten a nested dictionary, joining keys with sep."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def dict_to_str(d, indent=0):
    """
    Pretty-print a (potentially nested) dict as an indented string for logging/debugging.
    Args:
        d: dict to print
        indent: current indentation level (internal use)
    Returns:
        str
    """
    lines = []
    ind = '  ' * indent
    if not isinstance(d, dict):
        return repr(d)
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(f"{ind}{k}:")
            lines.append(dict_to_str(v, indent=indent+1))
        else:
            lines.append(f"{ind}{k}: {repr(v)}")
    return '\n'.join(lines)


import json

def safe_json_parse(s):
    """
    Parse a string as JSON, returning the parsed object or None on failure.
    Useful for extracting tool-use actions from LLM outputs.
    Args:
        s: str, JSON to parse
    Returns:
        obj or None
    """
    try:
        return json.loads(s)
    except Exception:
        return None


def get_env_name(env_or_spec):
    """
    Extract the canonical environment name from a gymnasium env or spec.
    Args:
        env_or_spec: gymnasium.Env instance or gymnasium.env.spec
    Returns:
        str: environment name (e.g., 'CartPole-v1') or None if not found
    """
    if hasattr(env_or_spec, 'spec') and env_or_spec.spec is not None:
        return getattr(env_or_spec.spec, 'id', None)
    elif hasattr(env_or_spec, 'id'):
        return env_or_spec.id
    else:
        return None


def is_discrete_space(space):
    """
    Returns True if the gymnasium action space is Discrete, else False.
    Useful for agent logic that branches by action space type.
    Args:
        space: gymnasium.Space instance
    Returns:
        bool
    """
    # Avoid direct import to reduce dependency risk, check type by class name
    return getattr(space, '__class__', None).__name__ == 'Discrete' or hasattr(space, 'n')


def hash_dict(d):
    """
    Return a stable integer hash for a dict (including nested dicts).
    Useful for caching, episode trace deduplication, or quick comparison.
    Args:
        d: dict (possibly nested)
    Returns:
        int: hash value
    """
    try:
        # Use json.dumps with sort_keys for stable serialization
        s = json.dumps(d, sort_keys=True, separators=(',', ':'))
        return hash(s)
    except Exception:
        return hash(str(d))


def deep_copy_dict(d):
    """
    Return a deep copy of a dict (including nested dicts/lists).
    Uses json serialization for simplicity; suitable for dicts with JSON-serializable contents.
    Args:
        d: dict to copy
    Returns:
        dict: deep copy
    """
    try:
        return json.loads(json.dumps(d))
    except Exception:
        import copy
        return copy.deepcopy(d)
