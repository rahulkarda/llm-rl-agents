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


def safe_json_parse(s):
    """
    Robustly parse a JSON string, returning None on failure.
    Args:
        s: JSON string
    Returns:
        parsed object (dict/list) or None
    """
    try:
        return json.loads(s)
    except Exception:
        return None


def get_env_name(env):
    """
    Extract environment name from gym env or spec.
    Args:
        env: gym.Env or gym.EnvSpec
    Returns:
        str name
    """
    if hasattr(env, 'spec') and env.spec is not None:
        return env.spec.id
    elif hasattr(env, 'name'):
        return env.name
    elif hasattr(env, 'id'):
        return env.id
    else:
        return str(env)


def is_discrete_space(space):
    """
    Check if gym action space is discrete.
    Args:
        space: gym.Space
    Returns:
        bool
    """
    from gymnasium.spaces import Discrete
    return isinstance(space, Discrete)


def hash_dict(d):
    """
    Produce a stable hash for a dict (for caching/debugging).
    Args:
        d: dict
    Returns:
        int hash
    """
    s = json.dumps(d, sort_keys=True, default=str)
    return int(hashlib.md5(s.encode()).hexdigest()[:12], 16)


def deep_copy_dict(d):
    """
    Return a deep copy of a dict (for safe mutation).
    Args:
        d: dict
    Returns:
        dict (deep copy)
    """
    return copy.deepcopy(d)


def pad_list(x, target_len, pad_value=None):
    """
    Pad or truncate a list to a target length.
    Args:
        x: list
        target_len: desired length
        pad_value: value to pad with
    Returns:
        padded/truncated list
    """
    if len(x) >= target_len:
        return x[:target_len]
    else:
        return x + [pad_value] * (target_len - len(x))


def dict_diff(d1, d2):
    """
    Compute the difference between two dicts (added, removed, changed keys).
    Args:
        d1: dict (original)
        d2: dict (updated)
    Returns:
        dict: {'added': ..., 'removed': ..., 'changed': ...}
    """
    added = {k: d2[k] for k in d2 if k not in d1}
    removed = {k: d1[k] for k in d1 if k not in d2}
    changed = {k: (d1[k], d2[k]) for k in d1 if k in d2 and d1[k] != d2[k]}
    return {'added': added, 'removed': removed, 'changed': changed}


def filter_dict(d, keys):
    """
    Return a new dict containing only specified keys from input dict.
    Args:
        d: input dict
        keys: iterable of keys to keep
    Returns:
        dict
    """
    return {k: d[k] for k in keys if k in d}


def partition_dict(d, left_keys):
    """
    Split a dict into two dicts based on a set of keys.
    Args:
        d: dict to partition
        left_keys: iterable of keys for left dict
    Returns:
        (left, right): tuple of dicts
            left: contains keys in left_keys (if present)
            right: contains remaining keys
    """
    left = {k: d[k] for k in left_keys if k in d}
    right = {k: v for k, v in d.items() if k not in left_keys}
    return left, right


def deep_merge_dicts(d1, d2):
    """
    Recursively merge two dictionaries. Values from d2 overwrite d1.
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
