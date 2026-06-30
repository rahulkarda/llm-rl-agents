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
- dict_key_exists: Check if a dotted/nested key exists in dict (added).

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
- dict_key_exists: For quick nested key presence checks in complex info dicts (added).

Usage examples:
    # Flatten a nested dict
    d = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}
    flat = flatten_dict(d)
    # flat: {'a': 1, 'b.c': 2, 'b.d.e': 3}

    # Extract flat keys
    keys = flatten_dict_keys(d)
    # keys: ['a', 'b.c', 'b.d.e']

    # Check nested key exists
    exists = dict_key_exists(d, 'b.d.e')  # True
    missing = dict_key_exists(d, 'b.x.y') # False

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
    d = {"a": 1, "b": 2, "c": 3}
    keys = ["a", "c"]
    values = dict_values_to_list(d, keys)
    # values: [1, 3]

    # Flatten dict keys
    d = {"foo": {"bar": 1}, "baz": {"qux": 2}}
    flat_keys = flatten_dict_keys(d)
    # flat_keys: ['foo.bar', 'baz.qux']
"""
import json
import copy
from collections.abc import Mapping


def flatten_dict(d, parent_key="", sep="."):
    """
    Flatten a nested dict using dotted keys.
    Example: {'a': 1, 'b': {'c': 2}} -> {'a': 1, 'b.c': 2}
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def flatten_dict_keys(d, parent_key="", sep="."):
    """
    Extract all flat keys from a nested dict using dotted notation, sorted for deterministic order.
    """
    keys = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            keys.extend(flatten_dict_keys(v, new_key, sep=sep))
        else:
            keys.append(new_key)
    return sorted(keys)


def dict_key_exists(d, dotted_key, sep="."):
    """
    Check if a dotted/nested key exists in a dict.
    Example:
        d = {'a': 1, 'b': {'c': 2}}
        dict_key_exists(d, 'b.c') -> True
        dict_key_exists(d, 'b.x') -> False
    Args:
        d: dict (may be nested)
        dotted_key: str, e.g. 'foo.bar.baz'
        sep: separator (default '.')
    Returns:
        True if key exists, False otherwise
    """
    keys = dotted_key.split(sep)
    curr = d
    for k in keys:
        if isinstance(curr, Mapping) and k in curr:
            curr = curr[k]
        else:
            return False
    return True


def dict_to_str(d, indent=0):
    """
    Pretty-print a dict (possibly nested) for logging/debug.
    """
    lines = []
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append("  " * indent + f"{k}:")
            lines.append(dict_to_str(v, indent + 1))
        else:
            lines.append("  " * indent + f"{k}: {v}")
    return "\n".join(lines)


def safe_json_parse(txt):
    """
    Robustly parse JSON, returning None on failure.
    """
    try:
        return json.loads(txt)
    except Exception:
        return None


def get_env_name(env):
    """
    Extract env name from gym env or spec.
    """
    if hasattr(env, "spec") and hasattr(env.spec, "id"):
        return env.spec.id
    elif hasattr(env, "name"):
        return env.name
    return str(env)


def is_discrete_space(space):
    """
    Check if gym action space is discrete.
    """
    from gymnasium.spaces import Discrete
    return isinstance(space, Discrete) or getattr(space, "n", None) is not None


def hash_dict(d):
    """
    Produce a stable hash for a dict (for caching/debugging).
    """
    return hash(json.dumps(d, sort_keys=True, separators=(",", ":")))


def deep_copy_dict(d):
    """
    Return a deep copy of a dict (for safe mutation).
    """
    return copy.deepcopy(d)


def pad_list(lst, target_len, pad_value=None):
    """
    Pad or truncate a list to target_len.
    """
    if len(lst) >= target_len:
        return lst[:target_len]
    return lst + [pad_value] * (target_len - len(lst))


def dict_diff(d1, d2):
    """
    Compute difference between two dicts: added, removed, changed keys.
    """
    added = {k: d2[k] for k in d2 if k not in d1}
    removed = {k: d1[k] for k in d1 if k not in d2}
    changed = {k: (d1[k], d2[k]) for k in d1 if k in d2 and d1[k] != d2[k]}
    return {"added": added, "removed": removed, "changed": changed}


def filter_dict(d, keys):
    """
    Return new dict with only specified keys from d.
    """
    return {k: d[k] for k in keys if k in d}


def partition_dict(d, left_keys):
    """
    Split dict d into two dicts based on left_keys.
    """
    left = {k: d[k] for k in left_keys if k in d}
    right = {k: d[k] for k in d if k not in left_keys}
    return left, right


def deep_merge_dicts(a, b):
    """
    Recursively merge two dicts: b overlays a.
    """
    if not isinstance(a, dict):
        return b
    result = dict(a)
    for k, v in b.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge_dicts(result[k], v)
        else:
            result[k] = v
    return result


def list_chunk(lst, n):
    """
    Split list into chunks of size n.
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def compute_episode_cost(transitions, cost_keys=("cost", "api_cost")):
    """
    Compute total episode cost from trace transitions.
    Sums up cost fields for each transition, handling top-level or info dict.
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
