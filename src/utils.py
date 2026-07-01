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
    # diff: {'added': ['c'], 'removed': [], 'changed': ['b']}

    # Filter dict
    filtered = filter_dict(d2, ['a', 'c'])  # {'a': 1, 'c': 4}

    # Partition dict
    d3 = {'foo': 1, 'bar': 2, 'baz': 3}
    left, right = partition_dict(d3, ['foo', 'baz'])  # left={'foo': 1, 'baz': 3}, right={'bar': 2}

    # Deep merge dicts
    d4 = {'a': {'b': 1}, 'c': 2}
    d5 = {'a': {'d': 3}, 'c': 4}
    merged = deep_merge_dicts(d4, d5)
    # merged: {'a': {'b': 1, 'd': 3}, 'c': 4}

    # Chunk list
    chunks = list_chunk([1,2,3,4,5], chunk_size=2)  # [[1,2],[3,4],[5]]

    # Compute episode cost
    trace = [
        {'info': {'cost': 1}},
        {'info': {'cost': 2}},
        {'info': {'cost': 3}},
    ]
    total_cost = compute_episode_cost(trace, cost_keys=('cost',))  # 6

    # Dict values to list
    d6 = {'a': 1, 'b': 2, 'c': 3}
    vals = dict_values_to_list(d6, ['a', 'c'])  # [1, 3]

    # Flatten dict keys
    keys = flatten_dict_keys({'x': {'y': 2}, 'z': 3})  # ['x.y', 'z']

    # dict_key_exists
    exists = dict_key_exists({'a': {'b': 1}}, 'a.b')  # True

    # get_nested_value
    d7 = {'a': {'b': {'c': 5}}}
    v = get_nested_value(d7, 'a.b.c')  # 5

    # extract_keys_from_dict
    d8 = {'a': 1, 'b': {'c': 2}}
    vals = extract_keys_from_dict(d8, ['a', 'b.c'])  # [1, 2]
"""
import json
import copy


def flatten_dict(d, parent_key="", sep="."):
    """
    Flatten nested dict using dotted keys.
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


def dict_to_str(d, indent=0):
    """
    Pretty-print (possibly nested) dict for logging/debugging.
    """
    lines = []
    for k, v in d.items():
        prefix = " " * indent
        if isinstance(v, dict):
            lines.append(f"{prefix}{k}:")
            lines.append(dict_to_str(v, indent=indent + 2))
        else:
            lines.append(f"{prefix}{k}: {v}")
    return "\n".join(lines)


def safe_json_parse(s):
    """
    Robustly parse JSON from string, returning None if malformed.
    """
    try:
        return json.loads(s)
    except Exception:
        return None


def get_env_name(env):
    """
    Extract env name from gym env or spec.
    """
    try:
        if hasattr(env, 'spec') and env.spec is not None:
            return env.spec.id
        elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'spec') and env.unwrapped.spec is not None:
            return env.unwrapped.spec.id
        else:
            return str(type(env).__name__)
    except Exception:
        return str(type(env).__name__)


def is_discrete_space(space):
    """
    Check if action space is discrete (gymnasium.spaces.Discrete).
    """
    return hasattr(space, 'n')


def hash_dict(d):
    """
    Produce a stable hash for a dict (for caching/debugging).
    """
    try:
        return hash(json.dumps(d, sort_keys=True))
    except Exception:
        return hash(str(d))


def deep_copy_dict(d):
    """
    Return a deep copy of a dict.
    """
    return copy.deepcopy(d)


def pad_list(x, target_len, pad_value=None):
    """
    Pad or truncate a list x to target_len.
    """
    if len(x) >= target_len:
        return x[:target_len]
    else:
        return x + [pad_value] * (target_len - len(x))


def dict_diff(a, b):
    """
    Compute difference between two dicts. Returns dict with keys:
      - 'added': keys in b but not in a
      - 'removed': keys in a but not in b
      - 'changed': keys present in both but with different values
    """
    a_keys = set(a.keys())
    b_keys = set(b.keys())
    added = b_keys - a_keys
    removed = a_keys - b_keys
    changed = set(k for k in a_keys & b_keys if a[k] != b[k])
    return {'added': list(added), 'removed': list(removed), 'changed': list(changed)}


def filter_dict(d, keep_keys):
    """
    Return a new dict containing only specified keys from input dict.
    """
    return {k: v for k, v in d.items() if k in keep_keys}


def partition_dict(d, left_keys):
    """
    Split a dict into two dicts based on a set of keys.
    Returns (left_dict, right_dict):
      - left_dict contains keys in left_keys
      - right_dict contains remaining keys
    Example:
        d = {'foo': 1, 'bar': 2, 'baz': 3}
        left, right = partition_dict(d, ['foo', 'baz'])
        # left: {'foo': 1, 'baz': 3}, right: {'bar': 2}
    """
    left = {k: v for k, v in d.items() if k in left_keys}
    right = {k: v for k, v in d.items() if k not in left_keys}
    return left, right


def deep_merge_dicts(a, b):
    """
    Recursively merge two nested dicts.
    a: base dict
    b: overlay dict
    Values from b override a. Dicts are merged recursively.
    """
    result = deep_copy_dict(a)
    for k, v in b.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge_dicts(result[k], v)
        else:
            result[k] = v
    return result


def list_chunk(lst, chunk_size):
    """
    Split a list into chunks of given size.
    """
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def compute_episode_cost(trace, cost_keys=("cost", "api_cost")):
    """
    Compute total episode cost from trace.
    Sums values for cost_keys in each transition's info dict.
    """
    total = 0.0
    for t in trace:
        info = t.get("info", {})
        for k in cost_keys:
            if k in info and isinstance(info[k], (int, float)):
                total += info[k]
    return total


def dict_values_to_list(d, keys):
    """
    Convert values of dict for specified keys to a list.
    """
    return [d[k] if k in d else None for k in keys]


def flatten_dict_keys(d, parent_key="", sep="."):
    """
    Extract all flat (dotted) keys from nested dict.
    Returns sorted list of keys for deterministic order.
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
    Check if a dotted/nested key exists in dict.
    Example:
        d = {'a': {'b': 1}}
        dict_key_exists(d, 'a.b') -> True
    """
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
    Example:
        d = {'a': {'b': {'c': 5}}}
        get_nested_value(d, 'a.b.c') -> 5
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
