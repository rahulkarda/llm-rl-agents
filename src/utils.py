"""
Utility functions for nested dict manipulation and pretty-printing.
- flatten_dict: flattens nested dicts using dotted keys.
- dict_to_str: pretty-prints (possibly nested) dicts for logging/debugging.
- safe_json_parse: robustly parse JSON, returning None on failure.
- get_env_name: extract environment name from gym env or spec.
"""
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


import json

def safe_json_parse(s):
    """
    Parse a string as JSON, returning the parsed object or None on failure.
    Useful for extracting tool-use actions from LLM outputs.
    Args:
        s: str, JSON to parse
    Returns:
        obj or None
    """
    try:
        return json.loads(s)
    except Exception:
        return None


def get_env_name(env_or_spec):
    """
    Extract the canonical environment name from a gymnasium env or spec.
    Args:
        env_or_spec: gymnasium.Env instance or gymnasium.env.spec
    Returns:
        str: environment name (e.g., 'CartPole-v1') or None if not found
    """
    if hasattr(env_or_spec, 'spec') and env_or_spec.spec is not None:
        return getattr(env_or_spec.spec, 'id', None)
    elif hasattr(env_or_spec, 'id'):
        return env_or_spec.id
    else:
        return None
