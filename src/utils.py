"""
Utility functions for nested dict manipulation, gym env introspection, and agent episode trace formatting.

Functions:
- flatten_dict: Flattens nested dicts using dotted keys.
- dict_to_str: Pretty-prints (possibly nested) dicts for logging/debugging.
- safe_json_parse: Robustly parse JSON, returning None on failure.
- get_env_name: Extract environment name from gym env or spec.
- is_discrete_space: Check if gym action space is discrete.
- hash_dict: Produce a stable hash for a dict (for caching/debugging).
- deep_copy_dict: Return a deep copy of a dict (for safe mutation).
- pad_list: Pad or truncate a list to a target length.
- dict_diff: Compute the difference between two dicts (added, removed, changed keys).
- filter_dict: Return a new dict containing only specified keys from input dict.

Typical scenarios:
- Agent logging: Use dict_to_str and flatten_dict to produce readable or flat logs of env info, episode traces, or transition dicts.
- LLM action parsing: Use safe_json_parse to robustly extract actions from LLM outputs that may be malformed or partial.
- Gym env introspection: Use get_env_name and is_discrete_space to branch logic or annotate traces based on env specifics.
- Trace hashing and diffing: Use hash_dict and dict_diff to compare episode traces or cache results for debugging and evaluation.
- Buffer management: Use pad_list to align episode steps, action lists, or other sequences for analysis.
- Safe mutation: Use deep_copy_dict to avoid accidental mutation of dicts when storing transitions or infos.
- Selective dict extraction: Use filter_dict to keep only relevant keys from info or transition dicts.

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

    # Pad a list
    x = [1, 2]
    padded = pad_list(x, 4, pad_value=0)  # [1, 2, 0, 0]
    truncated = pad_list([1,2,3,4,5], 3)  # [1,2,3]

    # Dict diff
    d1 = {'a': 1, 'b': 2}
    d2 = {'a': 1, 'b': 3, 'c': 4}
    diff = dict_diff(d1, d2)
    # diff: {'added': {'c': 4}, 'removed': {}, 'changed': {'b': (2, 3)}}

    # Filter dict keys
    d = {'a': 1, 'b': 2, 'c': 3}
    filtered = filter_dict(d, ['a', 'c'])
    # filtered: {'a': 1, 'c': 3}

Notes:
- flatten_dict is useful for flattening nested info dicts for logging or CSV export.
- dict_to_str helps with readable debug output, especially for deeply nested transitions.
- hash_dict enables stable dict hashing for caching or trace comparison.
- deep_copy_dict is useful for safe mutation of dicts, e.g. when storing transitions.
- pad_list is useful for aligning sequence lengths (e.g., episode steps, action lists).
- dict_diff is useful for trace comparison, debugging, and change tracking.
- filter_dict is useful for extracting only relevant keys from info dicts or transitions.
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
    Return True if gymnasium action space is Discrete.
    Args:
        space: gymnasium.Space instance
    Returns:
        bool
    """
    # Discrete spaces have 'n' attribute
    return hasattr(space, 'n')


def hash_dict(d):
    """
    Compute a stable hash for a dict using JSON serialization.
    Args:
        d: dict
    Returns:
        int hash
    """
    try:
        s = json.dumps(d, sort_keys=True)
        return hash(s)
    except Exception:
        return hash(str(d))


def deep_copy_dict(d):
    """
    Return a deep copy of a dict (via JSON serialization).
    Args:
        d: dict
    Returns:
        dict (deep copy)
    """
    try:
        return json.loads(json.dumps(d))
    except Exception:
        import copy
        return copy.deepcopy(d)


def pad_list(lst, target_len, pad_value=None):
    """
    Pad or truncate a list to a target length.
    Args:
        lst: list
        target_len: int
        pad_value: value to use for padding (default None)
    Returns:
        list with length == target_len
    """
    n = len(lst)
    if n > target_len:
        return lst[:target_len]
    elif n < target_len:
        return lst + [pad_value] * (target_len - n)
    else:
        return lst


def dict_diff(d1, d2):
    """
    Compute difference between two dicts:
      - 'added': keys in d2 not in d1
      - 'removed': keys in d1 not in d2
      - 'changed': keys present in both but with different values (tuple of old, new)
    Args:
        d1: dict (original)
        d2: dict (updated)
    Returns:
        dict: {'added': ..., 'removed': ..., 'changed': ...}
    """
    added = {k: d2[k] for k in d2.keys() if k not in d1}
    removed = {k: d1[k] for k in d1.keys() if k not in d2}
    changed = {k: (d1[k], d2[k]) for k in d1.keys() & d2.keys() if d1[k] != d2[k]}
    return {'added': added, 'removed': removed, 'changed': changed}


def filter_dict(d, keys):
    """
    Return a new dict containing only the specified keys from d.
    Args:
        d: dict to filter
        keys: iterable of keys to include
    Returns:
        dict containing only keys in 'keys', if present in d
    """
    return {k: d[k] for k in keys if k in d}
