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
- partition_dict: Split a dict into two dicts based on a set of keys (added).
- deep_merge_dicts: Recursively merge two nested dicts (added).

Typical scenarios:
- Agent logging: Use dict_to_str and flatten_dict to produce readable or flat logs of env info, episode traces, or transition dicts.
- LLM action parsing: Use safe_json_parse to robustly extract actions from LLM outputs that may be malformed or partial.
- Gym env introspection: Use get_env_name and is_discrete_space to branch logic or annotate traces based on env specifics.
- Trace hashing and diffing: Use hash_dict and dict_diff to compare episode traces or cache results for debugging and evaluation.
- Buffer management: Use pad_list to align episode steps, action lists, or other sequences for analysis.
- Safe mutation: Use deep_copy_dict to avoid accidental mutation of dicts when storing transitions or infos.
- Selective dict extraction: Use filter_dict to keep only relevant keys from info or transition dicts.
- Dict splitting: Use partition_dict to separate fields for agent vs env, or for logging selective info.
- Deep dict merging: Use deep_merge_dicts to combine nested dicts for config, info, or state overlays (added).

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

    # Partition dict keys
    d = {'a': 1, 'b': 2, 'c': 3}
    left, right = partition_dict(d, ['a', 'c'])
    # left: {'a': 1, 'c': 3}, right: {'b': 2}

    # Deep merge dicts
    d1 = {'a': 1, 'b': {'c': 2, 'd': 4}}
    d2 = {'b': {'c': 3, 'e': 5}, 'f': 6}
    merged = deep_merge_dicts(d1, d2)
    # merged: {'a': 1, 'b': {'c': 3, 'd': 4, 'e': 5}, 'f': 6}

Notes:
- flatten_dict is useful for flattening nested info dicts for logging or CSV export.
- dict_to_str helps with readable debug output, especially for deeply nested transitions.
- hash_dict enables stable dict hashing for caching or trace comparison.
- deep_copy_dict is useful for safe mutation of dicts, e.g. when storing transitions.
- pad_list is useful for aligning sequence lengths (e.g., episode steps, action lists).
- dict_diff is useful for trace comparison, debugging, and change tracking.
- filter_dict is useful for extracting only relevant keys from info dicts or transitions.
- partition_dict is useful for separating agent-related vs env-related fields or logging selective info.
- deep_merge_dicts is useful for merging config overlays or combining nested info dicts.
"""
import json
import copy
import hashlib

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
    Pretty-print a (potentially nested) dict for logging/debugging.
    Each nested level is indented by 2 spaces. Returns a string.
    Args:
        d: dict or value to print
        indent: current indentation level (int)
    Returns:
        str
    """
    spaces = '  ' * indent
    if not isinstance(d, dict):
        return f"{spaces}{d}"
    if not d:
        return f"{spaces}{{}}"
    lines = []
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(f"{spaces}{k}:")
            lines.append(dict_to_str(v, indent + 1))
        else:
            lines.append(f"{spaces}{k}: {v}")
    return '\n'.join(lines)


def safe_json_parse(s):
    """Parse a JSON string robustly, returning None on failure."""
    try:
        return json.loads(s)
    except Exception:
        return None


def get_env_name(env):
    """Extract environment name from gym env or spec."""
    try:
        if hasattr(env, 'spec') and env.spec is not None:
            return env.spec.id
        if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'spec') and env.unwrapped.spec is not None:
            return env.unwrapped.spec.id
    except Exception:
        pass
    return str(type(env))


def is_discrete_space(space):
    """Check if a gym action space is discrete."""
    return hasattr(space, 'n')


def hash_dict(d):
    """Produce a stable integer hash for a dict."""
    s = json.dumps(d, sort_keys=True, default=str)
    return int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16)


def deep_copy_dict(d):
    """Deep copy a dict for safe mutation."""
    return copy.deepcopy(d)


def pad_list(lst, target_len, pad_value=None):
    """Pad or truncate a list to target_len."""
    if len(lst) >= target_len:
        return lst[:target_len]
    return lst + [pad_value] * (target_len - len(lst))


def dict_diff(d1, d2):
    """Compute difference between two dicts: added, removed, changed keys."""
    added = {k: d2[k] for k in d2 if k not in d1}
    removed = {k: d1[k] for k in d1 if k not in d2}
    changed = {k: (d1[k], d2[k]) for k in d1 if k in d2 and d1[k] != d2[k]}
    return {'added': added, 'removed': removed, 'changed': changed}


def filter_dict(d, keys):
    """Return new dict with only specified keys from d."""
    return {k: d[k] for k in keys if k in d}


def partition_dict(d, keys):
    """
    Partition dict d into two dicts: those with keys in keys, and those without.
    Returns (included, excluded).
    """
    included = {k: d[k] for k in keys if k in d}
    excluded = {k: v for k, v in d.items() if k not in keys}
    return included, excluded


def deep_merge_dicts(d1, d2):
    """
    Recursively merge two dicts. Values from d2 overwrite d1.
    For nested dicts, merge recursively. Does not mutate inputs.
    Args:
        d1: dict (base)
        d2: dict (overlay)
    Returns:
        dict (merged)
    """
    result = deep_copy_dict(d1)
    for k, v in d2.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge_dicts(result[k], v)
        else:
            result[k] = deep_copy_dict(v) if isinstance(v, dict) else v
    return result
