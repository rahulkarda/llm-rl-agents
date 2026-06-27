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
- compute_episode_cost: Compute total episode cost from trace (added).
- dict_values_to_list: Convert values of dict for specified keys to a list (added).
- flatten_dict_keys: Extract all flat (dotted) keys from a nested dict (added).

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
- Episode cost tracking: Use compute_episode_cost to sum API or action costs for an episode trace (added).
- Dict values to list: Use dict_values_to_list to extract values for selected keys into a list for easy batching or export (added).
- Flatten dict keys: Use flatten_dict_keys to enumerate all keys for CSV header or extraction (added).

Usage examples:
    # Flatten a nested dict
    d = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}
    flat = flatten_dict(d)
    # flat: {'a': 1, 'b.c': 2, 'b.d.e': 3}

    # Extract flat keys
    keys = flatten_dict_keys(d)
    # keys: ['a', 'b.c', 'b.d.e']

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

    # Compute episode cost
    transitions = [
        {"cost": 0.01, "info": {"api_cost": 0.02}},
        {"info": {"api_cost": 0.03}},
        {"cost": 0.04}
    ]
    total = compute_episode_cost(transitions)
    # total: 0.1

    # Dict values to list
    d = {'a': 1, 'b': 2, 'c': 3}
    vals = dict_values_to_list(d, ['a', 'c'])
    # vals: [1, 3]

    # Extract flat keys
    d = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}
    keys = flatten_dict_keys(d)
    # keys: ['a', 'b.c', 'b.d.e']

Notes:
- flatten_dict is useful for flattening nested info dicts for logging or CSV export.
- flatten_dict_keys helps enumerate columns for CSV export or analysis.
- dict_to_str helps with readable debug output.
- safe_json_parse avoids exceptions on malformed LLM outputs.
- get_env_name and is_discrete_space allow env introspection for agent logic.
- hash_dict produces stable hashes for dicts, useful in caching, deduplication.
- deep_copy_dict returns a true deep copy (not just a reference).
- pad_list aligns lists for batching or fixed-length buffers.
- dict_diff helps with debugging changes in info dicts or configs.
- filter_dict and partition_dict select/split fields for agent logging or trace export.
- deep_merge_dicts allows recursively merging configs or infos.
- list_chunk splits lists for batching.
- compute_episode_cost sums API or action costs from episode transitions.
- dict_values_to_list batches values from dicts for model input or export.
- flatten_dict_keys extracts all flat keys for CSV or header mapping.
"""
import json
import copy
import hashlib


def flatten_dict(d, parent_key='', sep='.'):
    """
    Flatten a nested dict using dotted keys.
    Args:
        d: input dict
        parent_key: prefix for keys
        sep: separator
    Returns:
        flat dict
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def flatten_dict_keys(d, parent_key='', sep='.'):
    """
    Extract all flat (dotted) keys from a nested dict.
    Args:
        d: input dict
        parent_key: prefix for keys
        sep: separator
    Returns:
        list of keys (strings)
    """
    keys = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            keys.extend(flatten_dict_keys(v, new_key, sep=sep))
        else:
            keys.append(new_key)
    return keys


def dict_to_str(d, indent=0):
    """
    Pretty-print a nested dict for logging/debugging.
    Args:
        d: dict
        indent: starting indentation
    Returns:
        string representation
    """
    lines = []
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(' ' * indent + f"{k}:")
            lines.append(dict_to_str(v, indent=indent+2))
        else:
            lines.append(' ' * indent + f"{k}: {v}")
    return '\n'.join(lines)


def safe_json_parse(s):
    """
    Robustly parse JSON, returning None on failure.
    Args:
        s: string
    Returns:
        dict/object, or None on error
    """
    try:
        return json.loads(s)
    except Exception:
        return None


def get_env_name(env):
    """
    Extract environment name from gym env or spec.
    Args:
        env: gym env
    Returns:
        str
    """
    if hasattr(env, 'spec') and env.spec is not None:
        return env.spec.id
    elif hasattr(env, 'name'):
        return env.name
    else:
        return str(type(env).__name__)


def is_discrete_space(space):
    """
    Check if gym action space is discrete.
    Args:
        space: gym space
    Returns:
        True if Discrete, False otherwise
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
    # Sort keys for deterministic hash
    s = json.dumps(d, sort_keys=True)
    h = hashlib.md5(s.encode('utf-8')).hexdigest()
    return int(h[:16], 16)  # truncate to 64-bit


def deep_copy_dict(d):
    """
    Return a deep copy of a dict (for safe mutation).
    Args:
        d: dict
    Returns:
        new dict
    """
    return copy.deepcopy(d)


def pad_list(lst, target_len, pad_value=None):
    """
    Pad or truncate a list to target length.
    Args:
        lst: input list
        target_len: desired length
        pad_value: value to pad
    Returns:
        list
    """
    if len(lst) > target_len:
        return lst[:target_len]
    else:
        return lst + [pad_value] * (target_len - len(lst))


def dict_diff(d1, d2):
    """
    Compute the difference between two dicts (added, removed, changed keys).
    Args:
        d1: original dict
        d2: new dict
    Returns:
        dict with keys: 'added', 'removed', 'changed'
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
        keys: list/iterable of keys
    Returns:
        filtered dict
    """
    return {k: d[k] for k in keys if k in d}


def partition_dict(d, keys):
    """
    Split a dict into two dicts based on a set of keys.
    Args:
        d: input dict
        keys: list/iterable of keys
    Returns:
        (dict with keys in keys, dict with the rest)
    """
    keys_set = set(keys)
    left = {k: v for k, v in d.items() if k in keys_set}
    right = {k: v for k, v in d.items() if k not in keys_set}
    return left, right


def deep_merge_dicts(d1, d2):
    """
    Recursively merge two nested dicts.
    Args:
        d1: base dict
        d2: overlay dict
    Returns:
        merged dict
    """
    result = copy.deepcopy(d1)
    for k, v in d2.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge_dicts(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def list_chunk(lst, chunk_size):
    """
    Split a list into chunks of given size.
    Args:
        lst: input list
        chunk_size: int
    Yields:
        list chunks
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i+chunk_size]


def compute_episode_cost(transitions, cost_keys=("cost", "api_cost")):
    """
    Compute total episode cost from trace.
    Args:
        transitions: list of transition dicts
        cost_keys: keys to look for in each dict or info dict
    Returns:
        sum of costs (float)
    """
    total = 0.0
    for t in transitions:
        for key in cost_keys:
            if key in t and isinstance(t[key], (int, float)):
                total += t[key]
            elif "info" in t and isinstance(t["info"], dict) and key in t["info"] and isinstance(t["info"][key], (int, float)):
                total += t["info"][key]
    return total


def dict_values_to_list(d, keys):
    """
    Convert values of dict for specified keys to a list, in order.
    Useful for batching values from dicts for model input or export.
    Args:
        d: input dict
        keys: iterable of keys to extract
    Returns:
        list of values (None for missing keys)
    """
    return [d.get(k) for k in keys]
