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
- get_all_nested_keys: Recursively enumerate all nested keys in a dict as lists of path elements (added).

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
- get_all_nested_keys: For enumerating all nested dict keys as lists of key paths (added).

Usage examples:
    # Flatten a nested dict
    d = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}
    flat = flatten_dict(d)
    # flat: {'a': 1, 'b.c': 2, 'b.d.e': 3}

    # Extract flat keys
    keys = flatten_dict_keys(d)
    # keys: ['a', 'b.c', 'b.d.e']

    # Enumerate all nested key paths
    paths = get_all_nested_keys(d)
    # paths: [['a'], ['b', 'c'], ['b', 'd', 'e']]

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
    # diff: {'added': ['c'], 'removed': [], 'changed': ['b']}

    # Filter dict
    f = filter_dict(d2, ['a', 'c'])  # {'a': 1, 'c': 4}

    # Partition dict
    d = {'a': 1, 'b': 2, 'c': 3}
    left, right = partition_dict(d, {'a', 'c'})
    # left: {'a': 1, 'c': 3}, right: {'b': 2}

    # Deep merge dicts
    d1 = {'a': {'x': 1}, 'b': 2}
    d2 = {'a': {'y': 3}, 'b': 4}
    merged = deep_merge_dicts(d1, d2)
    # merged: {'a': {'x': 1, 'y': 3}, 'b': 4}

    # List chunking
    xs = [1,2,3,4,5]
    chunks = list_chunk(xs, 2)
    # chunks: [[1,2],[3,4],[5]]

    # Compute episode cost
    trace = [
        {'info': {'cost': 0.3}},
        {'info': {'cost': 0.4}},
    ]
    total_cost = compute_episode_cost(trace)  # 0.7

    # Dict values to list
    d = {'a': 1, 'b': 2, 'c': 3}
    vals = dict_values_to_list(d, ['a', 'b'])  # [1, 2]

    # Flatten dict keys
    d = {'x': 1, 'y': {'z': 2}}
    keys = flatten_dict_keys(d)  # ['x', 'y.z']

    # Enumerate all nested key paths
    d = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}
    paths = get_all_nested_keys(d)  # [['a'], ['b', 'c'], ['b', 'd', 'e']]

"""
import json
import copy

# ... [existing functions truncated for brevity]

def flatten_dict(d, parent_key="", sep="."):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def dict_to_str(d, indent=0):
    lines = []
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(" " * indent + f"{k}:")
            lines.append(dict_to_str(v, indent + 2))
        else:
            lines.append(" " * indent + f"{k}: {v}")
    return "\n".join(lines)


def safe_json_parse(s):
    try:
        return json.loads(s)
    except Exception:
        return None


def get_env_name(env):
    if hasattr(env, "spec") and env.spec is not None:
        return env.spec.id
    if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "spec") and env.unwrapped.spec:
        return env.unwrapped.spec.id
    return str(type(env))


def is_discrete_space(space):
    return hasattr(space, "n")


def hash_dict(d):
    return hash(json.dumps(d, sort_keys=True))


def deep_copy_dict(d):
    return copy.deepcopy(d)


def pad_list(lst, length, pad_value=None):
    if len(lst) >= length:
        return lst[:length]
    return lst + [pad_value] * (length - len(lst))


def dict_diff(d1, d2):
    keys1 = set(d1.keys())
    keys2 = set(d2.keys())
    added = list(keys2 - keys1)
    removed = list(keys1 - keys2)
    changed = [k for k in keys1 & keys2 if d1[k] != d2[k]]
    return {"added": added, "removed": removed, "changed": changed}


def filter_dict(d, keys):
    return {k: d[k] for k in keys if k in d}


def partition_dict(d, key_set):
    left = {k: v for k, v in d.items() if k in key_set}
    right = {k: v for k, v in d.items() if k not in key_set}
    return left, right


def deep_merge_dicts(d1, d2):
    out = deep_copy_dict(d1)
    for k, v in d2.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge_dicts(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def list_chunk(lst, size):
    return [lst[i:i+size] for i in range(0, len(lst), size)]


def compute_episode_cost(trace, cost_keys=("cost", "api_cost")):
    total = 0.0
    for t in trace:
        for k in cost_keys:
            if k in t:
                try:
                    total += float(t[k])
                except Exception:
                    pass
            elif "info" in t and k in t["info"]:
                try:
                    total += float(t["info"][k])
                except Exception:
                    pass
    return total


def dict_values_to_list(d, keys):
    return [d[k] if k in d else None for k in keys]


def flatten_dict_keys(d, parent_key="", sep="."):
    keys = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
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


def get_all_nested_keys(d, parent_path=None):
    """
    Recursively enumerate all nested keys in a dict as lists of path elements.
    Example:
        d = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}
        get_all_nested_keys(d)
        -> [['a'], ['b', 'c'], ['b', 'd', 'e']]
    """
    if parent_path is None:
        parent_path = []
    keys = []
    for k, v in d.items():
        cur_path = parent_path + [k]
        if isinstance(v, dict):
            keys.extend(get_all_nested_keys(v, cur_path))
        else:
            keys.append(cur_path)
    return keys
