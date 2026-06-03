"""
Utility functions for nested dict manipulation and pretty-printing.
- flatten_dict: flattens nested dicts using dotted keys.
- dict_to_str: pretty-prints (possibly nested) dicts for logging/debugging.
- safe_json_parse: robustly parse JSON, returning None on failure.
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
