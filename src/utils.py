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
- random_action: Sample a random action from a gym action space (added).
- deep_get: Safely extract a value from a nested dict given a list of keys (added).
- episode_reward_summary: Compute total, mean, and step-wise rewards from an episode trace (added).

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
- random_action: For quick sampling of a valid action from any gym action space (added).
- deep_get: For robust nested dict value extraction using explicit key path (added).
- episode_reward_summary: For summarizing rewards from episode traces (added).

Concrete usage examples:
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

    # Partition dict
    d = {'a': 1, 'b': 2, 'c': 3}
    left, right = partition_dict(d, {'a', 'c'})
    # left: {'a': 1, 'c': 3}, right: {'b': 2}

    # Deep merge dicts
    d1 = {'a': 1, 'b': {'c': 2}}
    d2 = {'b': {'d': 3}, '...
"""
import json
import hashlib
import copy


def flatten_dict(d, parent_key='', sep='.'):
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


def dict_to_str(d, indent=0):
    """
    Pretty-print a (possibly nested) dict for logging/debugging.
    """
    pad = '  ' * indent
    s = ''
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict):
                s += f"{pad}{k}:\n{dict_to_str(v, indent=indent+1)}"
            else:
                s += f"{pad}{k}: {v}\n"
    else:
        s += f"{pad}{d}\n"
    return s


def safe_json_parse(s):
    """
    Robustly parse JSON, returning None on failure.
    """
    try:
        return json.loads(s)
    except Exception:
        return None


def get_env_name(env):
    """
    Extract environment name (e.g. 'CartPole-v1') from gym env or spec.
    """
    if hasattr(env, 'spec') and env.spec is not None:
        return env.spec.id
    if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'spec'):
        return env.unwrapped.spec.id
    return str(type(env))


def is_discrete_space(space):
    """
    Check if gym action space is discrete.
    """
    return hasattr(space, 'n')


def hash_dict(d):
    """
    Produce a stable hash for a dict (for caching/debugging).
    """
    s = json.dumps(d, sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()


def deep_copy_dict(d):
    """
    Return a deep copy of a dict (for safe mutation).
    """
    return copy.deepcopy(d)


def pad_list(lst, length, fill=None):
    """
    Pad or truncate a list to a target length.
    """
    if len(lst) >= length:
        return lst[:length]
    return lst + [fill] * (length - len(lst))


def dict_diff(a, b):
    """
    Compute the difference between two dicts (added, removed, changed keys).
    Returns dict with keys 'added', 'removed', 'changed'.
    """
    a_keys = set(a.keys())
    b_keys = set(b.keys())
    added = b_keys - a_keys
    removed = a_keys - b_keys
    changed = {k for k in a_keys & b_keys if a[k] != b[k]}
    return {'added': added, 'removed': removed, 'changed': changed}


def filter_dict(d, keys):
    """
    Return a new dict containing only specified keys from input dict.
    """
    return {k: d[k] for k in keys if k in d}


def partition_dict(d, key_set):
    """
    Split a dict into two dicts based on a set of keys.
    Returns (left, right).
    """
    left = {k: v for k, v in d.items() if k in key_set}
    right = {k: v for k, v in d.items() if k not in key_set}
    return left, right


def deep_merge_dicts(a, b):
    """
    Recursively merge two nested dicts.
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
    Compute total episode cost from transition trace.
    Looks for keys in each transition or transition['info'].
    """
    total = 0.0
    for t in trace:
        for k in cost_keys:
            if k in t:
                total += t[k]
            elif 'info' in t and isinstance(t['info'], dict) and k in t['info']:
                total += t['info'][k]
    return total


def dict_values_to_list(d, keys):
    """
    Convert values of dict for specified keys to a list.
    """
    return [d.get(k, None) for k in keys]


def flatten_dict_keys(d, parent_key='', sep='.'):  # returns list of dotted keys
    """
    Extract all flat (dotted) keys from a nested dict.
    Example: {'b': {'c': 2, 'd': {'e': 3}}} -> ['b.c', 'b.d.e']
    """
    keys = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            keys.extend(flatten_dict_keys(v, new_key, sep=sep))
        else:
            keys.append(new_key)
    return keys


def dict_key_exists(d, dotted_key):
    """
    Check if a dotted/nested key exists in dict.
    Example: dict_key_exists({'a': {'b': 2}}, 'a.b') -> True
    """
    keys = dotted_key.split('.')
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return False
    return True


def get_nested_value(d, dotted_key):
    """
    Extract value from nested dict by dotted key.
    Example: get_nested_value({'a': {'b': 2}}, 'a.b') -> 2
    Returns None if any intermediate key is missing.
    """
    keys = dotted_key.split('.')
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return None
    return cur


def extract_keys_from_dict(d, dotted_keys):
    """
    Extract values for a list of dotted keys from nested dicts.
    Returns list of values (or None for missing).
    """
    return [get_nested_value(d, k) for k in dotted_keys]


def get_all_nested_keys(d, prefix=None):
    """
    Recursively enumerate all nested keys in a dict as lists of path elements.
    Example: {'a': {'b': 2}, 'c': 3} -> [['a', 'b'], ['c']]
    """
    keys = []
    for k, v in d.items():
        path = prefix + [k] if prefix else [k]
        if isinstance(v, dict):
            keys.extend(get_all_nested_keys(v, path))
        else:
            keys.append(path)
    return keys


def random_action(action_space):
    """
    Sample a random action from a gym action space.
    """
    return action_space.sample()


def deep_get(d, keys):
    """
    Safely extract a value from a nested dict given a list of keys.
    Returns None if any key is missing.
    """
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return None
    return cur


def episode_reward_summary(trace):
    """
    Compute total, mean, and step-wise rewards from an episode trace.
    Example:
        trace = [
            {'observation': 'foo', 'action': 1, 'reward': 0.5},
            {'observation': 'bar', 'action': 2, 'reward': 1.0},
        ]
        summary = episode_reward_summary(trace)
        # summary: {'total_reward': 1.5, 'mean_reward': 0.75, 'step_rewards': [0.5, 1.0]}
    """
    rewards = [t.get('reward', 0.0) for t in trace]
    total = sum(rewards)
    mean = total / len(rewards) if rewards else 0.0
    return {'total_reward': total, 'mean_reward': mean, 'step_rewards': rewards}
