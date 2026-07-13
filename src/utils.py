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
- nested_dict_update: Update nested dict values in-place given a mapping of dotted keys to values (added).

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
- nested_dict_update: For in-place update of nested dicts from dotted-key mapping (added).

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
    d2 = {'b': {'d': 3}, 'f': 4}
    merged = deep_merge_dicts(d1, d2)
    # merged: {'a': 1, 'b': {'c': 2, 'd': 3}, 'f': 4}

    # Chunk list
    chunks = list_chunk([1,2,3,4,5], 2)
    # chunks: [[1,2],[3,4],[5]]

    # Compute episode cost
    trace = [
        {'info': {'cost': 1.2}},
        {'info': {'cost': 1.8}},
        {'info': {'cost': 0.5}},
    ]
    total = compute_episode_cost(trace, cost_keys=("cost",))
    # total: 3.5

    # Extract values for keys
    vals = dict_values_to_list({'a': 1, 'b': 2}, ['a','b'])
    # vals: [1,2]

    # Update nested dict in-place
    d = {'a': {'b': {'c': 1}}, 'x': 2}
    nested_dict_update(d, {'a.b.c': 99, 'x': 5})
    # d now: {'a': {'b': {'c': 99}}, 'x': 5}
"""
import json
import copy
import hashlib

# --- flatten_dict ---
def flatten_dict(d, parent_key='', sep='.'):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

# --- dict_to_str ---
def dict_to_str(d, indent=0):
    s = ''
    ind = '  ' * indent
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict):
                s += f"{ind}{k}:\n"
                s += dict_to_str(v, indent=indent+1)
            else:
                s += f"{ind}{k}: {v}\n"
    else:
        s += f"{ind}{d}\n"
    return s

# --- safe_json_parse ---
def safe_json_parse(s):
    try:
        return json.loads(s)
    except Exception:
        return None

# --- get_env_name ---
def get_env_name(env):
    if hasattr(env, 'spec') and env.spec is not None:
        return getattr(env.spec, 'id', str(env.spec))
    elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'spec'):
        return getattr(env.unwrapped.spec, 'id', str(env.unwrapped.spec))
    return str(env)

# --- is_discrete_space ---
def is_discrete_space(space):
    return hasattr(space, 'n')

# --- hash_dict ---
def hash_dict(d):
    flat = flatten_dict(d)
    s = json.dumps(flat, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()

# --- deep_copy_dict ---
def deep_copy_dict(d):
    return copy.deepcopy(d)

# --- pad_list ---
def pad_list(lst, length, padding=None):
    if len(lst) > length:
        return lst[:length]
    return lst + [padding]*(length-len(lst))

# --- dict_diff ---
def dict_diff(a, b):
    a_keys = set(a.keys())
    b_keys = set(b.keys())
    added = b_keys - a_keys
    removed = a_keys - b_keys
    changed = {k for k in a_keys & b_keys if a[k] != b[k]}
    return {'added': list(added), 'removed': list(removed), 'changed': list(changed)}

# --- filter_dict ---
def filter_dict(d, keys):
    return {k: d[k] for k in keys if k in d}

# --- partition_dict ---
def partition_dict(d, keys):
    left = {k: d[k] for k in keys if k in d}
    right = {k: d[k] for k in d if k not in keys}
    return left, right

# --- deep_merge_dicts ---
def deep_merge_dicts(a, b):
    result = copy.deepcopy(a)
    for k, v in b.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge_dicts(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result

# --- list_chunk ---
def list_chunk(lst, chunk_size):
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

# --- compute_episode_cost ---
def compute_episode_cost(trace, cost_keys=("cost", "api_cost")):
    total = 0.0
    for t in trace:
        info = t.get('info', t)
        for key in cost_keys:
            if isinstance(info, dict) and key in info:
                try:
                    total += float(info[key])
                except Exception:
                    pass
    return total

# --- dict_values_to_list ---
def dict_values_to_list(d, keys):
    return [d.get(k, None) for k in keys]

# --- flatten_dict_keys ---
def flatten_dict_keys(d, sep='.'):  # Returns dotted keys
    return list(flatten_dict(d, sep=sep).keys())

# --- dict_key_exists ---
def dict_key_exists(d, dotted_key, sep='.'):  # Checks nested key
    keys = dotted_key.split(sep)
    curr = d
    for k in keys:
        if isinstance(curr, dict) and k in curr:
            curr = curr[k]
        else:
            return False
    return True

# --- get_nested_value ---
def get_nested_value(d, dotted_key, sep='.'):  # Returns value or None
    keys = dotted_key.split(sep)
    curr = d
    for k in keys:
        if isinstance(curr, dict) and k in curr:
            curr = curr[k]
        else:
            return None
    return curr

# --- extract_keys_from_dict ---
def extract_keys_from_dict(d, dotted_keys, sep='.'):  # Returns list of values
    return [get_nested_value(d, k, sep=sep) for k in dotted_keys]

# --- get_all_nested_keys ---
def get_all_nested_keys(d):
    def _walk(obj, path):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield from _walk(v, path + [k])
        else:
            yield path
    return list(_walk(d, []))

# --- random_action ---
def random_action(space):
    return space.sample()

# --- deep_get ---
def deep_get(d, keys):  # keys: list[str]
    curr = d
    for k in keys:
        if isinstance(curr, dict) and k in curr:
            curr = curr[k]
        else:
            return None
    return curr

# --- episode_reward_summary ---
def episode_reward_summary(trace):
    """
    Compute reward summary for episode trace (list of dicts).
    Returns dict: {'total_reward', 'mean_reward', 'step_rewards'}
    """
    rewards = [t.get('reward', 0.0) for t in trace]
    total = sum(rewards)
    mean = total / len(rewards) if rewards else 0.0
    return {'total_reward': total, 'mean_reward': mean, 'step_rewards': rewards}

# --- nested_dict_update ---
def nested_dict_update(d, updates, sep='.'):
    """
    Update nested dict 'd' in-place given mapping 'updates' of dotted keys to values.
    Args:
        d: dict to update
        updates: dict, mapping dotted keys (e.g. 'a.b.c') to values
        sep: separator for key paths (default '.')
    Example:
        d = {'a': {'b': {'c': 1}}, 'x': 2}
        nested_dict_update(d, {'a.b.c': 99, 'x': 5})
        # d now: {'a': {'b': {'c': 99}}, 'x': 5}
    """
    for dotted_key, value in updates.items():
        keys = dotted_key.split(sep)
        curr = d
        for k in keys[:-1]:
            if k not in curr or not isinstance(curr[k], dict):
                curr[k] = {}
            curr = curr[k]
        curr[keys[-1]] = value
