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
- get_nested_value: Extract value from nested dict by dotted key (added).
- extract_keys_from_dict: Extract values for a list of dotted keys from nested dicts (added).

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
- get_nested_value: For extracting values from nested dicts using dotted keys (added).
- extract_keys_from_dict: For extracting multiple values from nested dicts using dotted keys (added).

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

    # Get value from nested dict
    val = get_nested_value(d, 'b.d.e')    # 3
    missing_val = get_nested_value(d, 'b.x.y')  # None

    # Extract values for multiple keys
    vals = extract_keys_from_dict(d, ['a', 'b.c', 'b.d.e', 'b.x.y'])
    # vals: [1, 2, 3, None]

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
    left, right = partition_dict(d, ['a', 'b'])
    # left: {'a': 1, 'b': 2}, right: {'c': 3}

    # Deep merge dicts
    d1 = {'a': 1, 'b': {'c': 2}}
    d2 = {'b': {'d': 3}}
    merged = deep_merge_dicts(d1, d2)
    # merged: {'a': 1, 'b': {'c': 2, 'd': 3}}

    # Chunk a list
    x = [1,2,3,4,5]
    chunks = list_chunk(x, 2)
    # chunks: [[1,2],[3,4],[5]]

    # Compute episode cost
    trace = [{'info': {'cost': 1.0}}, {'info': {'cost': 0.5}}]
    cost = compute_episode_cost(trace)
    # cost: 1.5

    # Dict values to list
    d = {'a': 1, 'b': 2, 'c': 3}
    vals = dict_values_to_list(d, ['a', 'c'])
    # vals: [1, 3]

    # Flatten dict keys
    d = {'a': 1, 'b': {'c': 2}}
    keys = flatten_dict_keys(d)
    # keys: ['a', 'b.c']

    # dict_key_exists
    exists = dict_key_exists(d, 'b.c')
    # True

    # get_nested_value
    val = get_nested_value(d, 'b.c')
    # 2

    # extract_keys_from_dict
    vals = extract_keys_from_dict(d, ['a', 'b.c', 'b.d'])
    # [1, 2, None]
"""
import json
import copy
import hashlib


def flatten_dict(d, parent_key="", sep="."):
    items = {}
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def dict_to_str(d, indent=0):
    lines = []
    for k, v in d.items():
        prefix = " " * indent
        if isinstance(v, dict):
            lines.append(f"{prefix}{k}:")
            lines.append(dict_to_str(v, indent=indent+2))
        else:
            lines.append(f"{prefix}{k}: {v}")
    return "\n".join(lines)


def safe_json_parse(s):
    try:
        return json.loads(s)
    except Exception:
        return None


def get_env_name(env):
    if hasattr(env, "unwrapped"):
        env = env.unwrapped
    if hasattr(env, "spec") and getattr(env, "spec", None) is not None:
        return env.spec.id
    return getattr(env, "__class__", type(env)).__name__


def is_discrete_space(space):
    return hasattr(space, "n") and isinstance(space.n, int)


def hash_dict(d):
    s = json.dumps(d, sort_keys=True, default=str)
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def deep_copy_dict(d):
    return copy.deepcopy(d)


def pad_list(lst, length, pad_value=None):
    if len(lst) >= length:
        return lst[:length]
    else:
        return lst + [pad_value] * (length - len(lst))


def dict_diff(a, b):
    added = {k: b[k] for k in b if k not in a}
    removed = {k: a[k] for k in a if k not in b}
    changed = {k: (a[k], b[k]) for k in a if k in b and a[k] != b[k]}
    return {"added": added, "removed": removed, "changed": changed}


def filter_dict(d, keys):
    return {k: d[k] for k in keys if k in d}


def partition_dict(d, keys):
    left = {k: d[k] for k in keys if k in d}
    right = {k: d[k] for k in d if k not in keys}
    return left, right


def deep_merge_dicts(a, b):
    res = copy.deepcopy(a)
    for k in b:
        if k in res and isinstance(res[k], dict) and isinstance(b[k], dict):
            res[k] = deep_merge_dicts(res[k], b[k])
        else:
            res[k] = copy.deepcopy(b[k])
    return res


def list_chunk(lst, chunk_size):
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def compute_episode_cost(trace, cost_keys=("cost", "api_cost")):
    total = 0.0
    for t in trace:
        for k in cost_keys:
            # Look in t and t["info"]
            v = t.get(k, None)
            if v is not None:
                total += float(v)
            elif "info" in t and isinstance(t["info"], dict) and k in t["info"]:
                total += float(t["info"][k])
    return total


def dict_values_to_list(d, keys):
    return [d.get(k, None) for k in keys]


def flatten_dict_keys(d, parent_key="", sep="."):
    keys = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            keys.extend(flatten_dict_keys(v, new_key, sep=sep))
        else:
            keys.append(new_key)
    return sorted(keys)


def dict_key_exists(d, dotted_key, sep="."):
    keys = dotted_key.split(sep)
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return False
    return True


def get_nested_value(d, dotted_key, sep="."):
    """
    Extract value from nested dict by dotted key.
    Example: get_nested_value({'a': {'b': 1}}, 'a.b') -> 1
    Returns None if any key is missing.
    """
    keys = dotted_key.split(sep)
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return None
    return cur


def extract_keys_from_dict(d, dotted_keys, sep="."):
    """
    Extract values for a list of dotted keys from nested dict.
    Returns a list of values, None if key is missing.
    Example:
        d = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}
        extract_keys_from_dict(d, ['a', 'b.c', 'b.d.e', 'b.x.y'])
        -> [1, 2, 3, None]
    """
    return [get_nested_value(d, k, sep=sep) for k in dotted_keys]
