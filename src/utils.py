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
    orig = {'x': 1, 'y': {'z': 2}}
    copy = deep_copy_dict(orig)

    # Pad/truncate list
    x = [1, 2, 3]
    padded = pad_list(x, 5, pad_val=0)
    # [1, 2, 3, 0, 0]

    # Dict diff
    a = {'foo': 1, 'bar': 2}
    b = {'foo': 1, 'baz': 3}
    diff = dict_diff(a, b)

    # Filter dict
    d = {'a': 1, 'b': 2, 'c': 3}
    filtered = filter_dict(d, ['a', 'c'])
    # {'a': 1, 'c': 3}

    # Partition dict
    d = {'x': 1, 'y': 2, 'z': 3}
    d1, d2 = partition_dict(d, {'x', 'z'})
    # d1: {'x': 1, 'z': 3}, d2: {'y': 2}

    # Deep merge dicts
    a = {'foo': {'bar': 1}, 'baz': 2}
    b = {'foo': {'baz': 3}}
    merged = deep_merge_dicts(a, b)

    # Chunk a list
    x = list(range(10))
    chunks = list_chunk(x, 3)
    # [[0,1,2], [3,4,5], [6,7,8], [9]]

    # Compute episode cost
    trace = [
        {"info": {"api_cost": 1.5}},
        {"info": {"api_cost": 2.1}},
    ]
    total = compute_episode_cost(trace, key="api_cost")
    # 3.6

    # Dict values to list
    d = {"a": 1, "b": 2, "c": 3}
    vals = dict_values_to_list(d, keys=["a", "c"])
    # [1, 3]

    # Flatten dict keys
    d = {'a': 1, 'b': {'c': 2}}
    keys = flatten_dict_keys(d)
    # ['a', 'b.c']

    # Check nested key exists
    exists = dict_key_exists(d, 'b.c')  # True

    # Get nested value
    val = get_nested_value(d, 'b.c')  # 2

    # Extract keys from dict
    vals = extract_keys_from_dict(d, ['a', 'b.c', 'b.x'])
    # [1, 2, None]

    # Get all nested keys
    paths = get_all_nested_keys(d)
    # [['a'], ['b', 'c']]

    # Sample a random action
    import gymnasium as gym
    env = gym.make('CartPole-v1')
    action = random_action(env.action_space)

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
    s = ""
    pad = "  " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            s += f"{pad}{k}:\n" + dict_to_str(v, indent + 1)
        else:
            s += f"{pad}{k}: {v}\n"
    return s


def safe_json_parse(s):
    try:
        return json.loads(s)
    except Exception:
        return None


def get_env_name(env):
    if hasattr(env, "spec") and env.spec is not None:
        return env.spec.id
    elif hasattr(env, "unwrapped") and hasattr(env.unwrapped, "spec") and env.unwrapped.spec is not None:
        return env.unwrapped.spec.id
    else:
        return str(type(env).__name__)


def is_discrete_space(action_space):
    return hasattr(action_space, "n")


def hash_dict(d):
    s = json.dumps(d, sort_keys=True, default=str)
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def deep_copy_dict(d):
    return copy.deepcopy(d)


def pad_list(l, target_len, pad_val=None):
    if len(l) >= target_len:
        return l[:target_len]
    else:
        return l + [pad_val] * (target_len - len(l))


def dict_diff(a, b):
    diff = {"added": [], "removed": [], "changed": []}
    a_keys = set(a.keys())
    b_keys = set(b.keys())
    diff["added"] = list(b_keys - a_keys)
    diff["removed"] = list(a_keys - b_keys)
    diff["changed"] = [k for k in a_keys & b_keys if a[k] != b[k]]
    return diff


def filter_dict(d, keys):
    return {k: d[k] for k in keys if k in d}


def partition_dict(d, key_set):
    d1 = {k: v for k, v in d.items() if k in key_set}
    d2 = {k: v for k, v in d.items() if k not in key_set}
    return d1, d2


def deep_merge_dicts(a, b):
    result = copy.deepcopy(a)
    for k, v in b.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge_dicts(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def list_chunk(l, chunk_size):
    return [l[i:i + chunk_size] for i in range(0, len(l), chunk_size)]


def compute_episode_cost(trace, key="cost"):
    total = 0.0
    for step in trace:
        # Accept cost at top-level or under 'info'
        val = step.get(key, None)
        if val is None and "info" in step:
            val = step["info"].get(key, None)
        if isinstance(val, (int, float)):
            total += val
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


def dict_key_exists(d, dotted_key):
    keys = dotted_key.split(".")
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return False
    return True


def get_nested_value(d, dotted_key):
    keys = dotted_key.split(".")
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return None
    return cur


def extract_keys_from_dict(d, dotted_keys):
    return [get_nested_value(d, k) for k in dotted_keys]


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


def random_action(action_space):
    """
    Sample a random action from a gym action space.
    Handles Discrete, Box, MultiDiscrete, MultiBinary, etc.
    Args:
        action_space: gymnasium.spaces.Space instance
    Returns:
        action: valid action sampled from the space
    """
    return action_space.sample()
