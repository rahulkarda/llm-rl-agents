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
- list_chunk: Split a list into chunks of given size (added).

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
- Sequence batching: Use list_chunk to break lists into fixed-size batches for processing or logging (added).

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
    # Handles malformed JSON gracefully:
    obj2 = safe_json_parse('{"action": 0')  # returns None

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
    # Mutating d2 won't affect d

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

    # Chunk list
    x = [1,2,3,4,5,6]
    chunks = list(list_chunk(x, 2))
    # chunks: [[1,2],[3,4],[5,6]]

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
- list_chunk is useful for batching or splitting long lists for processing or logging.
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
    Pretty-print a (potentially nested) dict for debug/logging.
    Args:
        d: dict (possibly nested)
        indent: int (current indentation)
    Returns:
        str: multi-line formatted string
    """
    if not isinstance(d, dict):
        return str(d)
    if len(d) == 0:
        return '{}'
    out = []
    for k, v in d.items():
        prefix = '  ' * indent
        if isinstance(v, dict):
            out.append(f"{prefix}{k}:")
            out.append(dict_to_str(v, indent=indent+1))
        else:
            out.append(f"{prefix}{k}: {v}")
    return '\n'.join(out)


def safe_json_parse(s):
    """
    Robustly parse JSON string, returning None on failure.
    Args:
        s: str
    Returns:
        dict or list or None
    """
    try:
        return json.loads(s)
    except Exception:
        return None


def get_env_name(env):
    """
    Extract environment name from gym env or spec.
    Args:
        env: gym env or env.spec
    Returns:
        str name
    """
    if hasattr(env, 'spec') and env.spec is not None:
        return env.spec.id
    elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'spec'):
        return env.unwrapped.spec.id
    elif hasattr(env, 'id'):
        return env.id
    return 'unknown-env'


def is_discrete_space(space):
    """
    Check if gym action space is discrete.
    Args:
        space: gymnasium.Space
    Returns:
        bool
    """
    return hasattr(space, 'n')


def hash_dict(d):
    """
    Produce a stable hash for a dict (for caching/debugging).
    Args:
        d: dict
    Returns:
        int hash
    """
    s = json.dumps(d, sort_keys=True, separators=(',', ':'))
    return int(hashlib.sha256(s.encode('utf-8')).hexdigest()[:16], 16)


def deep_copy_dict(d):
    """
    Return a deep copy of a dict (for safe mutation).
    Args:
        d: dict
    Returns:
        dict (deep copy)
    """
    return copy.deepcopy(d)


def pad_list(lst, target_len, pad_value=None):
    """
    Pad or truncate a list to target_len.
    Args:
        lst: list
        target_len: int
        pad_value: value for padding (default None)
    Returns:
        new list
    """
    if len(lst) > target_len:
        return lst[:target_len]
    return lst + [pad_value] * (target_len - len(lst))


def dict_diff(d1, d2):
    """
    Compute dict difference: added, removed, changed keys.
    Args:
        d1: dict
        d2: dict
    Returns:
        dict with keys: 'added', 'removed', 'changed'
    """
    added = {k: d2[k] for k in d2 if k not in d1}
    removed = {k: d1[k] for k in d1 if k not in d2}
    changed = {k: (d1[k], d2[k]) for k in d1 if k in d2 and d1[k] != d2[k]}
    return {'added': added, 'removed': removed, 'changed': changed}


def filter_dict(d, keys):
    """
    Return new dict containing only specified keys from d.
    Args:
        d: dict
        keys: iterable of keys
    Returns:
        dict
    """
    return {k: d[k] for k in keys if k in d}


def partition_dict(d, keys):
    """
    Split a dict into two dicts: left contains keys in 'keys', right the rest.
    Args:
        d: dict
        keys: iterable of keys
    Returns:
        (left, right)
    """
    keys_set = set(keys)
    left = {k: v for k, v in d.items() if k in keys_set}
    right = {k: v for k, v in d.items() if k not in keys_set}
    return left, right


def deep_merge_dicts(d1, d2):
    """
    Recursively merge two nested dicts. Values from d2 overwrite d1.
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


def list_chunk(lst, chunk_size):
    """
    Split a list into chunks of length chunk_size.
    Args:
        lst: list
        chunk_size: int
    Returns:
        generator of list chunks
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]
