from utils import flatten_dict, dict_to_str, safe_json_parse

def test_flatten_dict_basic():
    d = {
        'a': 1,
        'b': {
            'c': 2,
            'd': {
                'e': 3
            }
        }
    }
    flat = flatten_dict(d)
    assert flat == {'a': 1, 'b.c': 2, 'b.d.e': 3}, f"Unexpected flatten result: {flat}"

def test_dict_to_str_example():
    d = {
        'x': 7,
        'y': {
            'z': 8,
            'w': {'u': 9}
        }
    }
    s = dict_to_str(d)
    # Should contain indented keys and values
    assert 'x:' in s
    assert 'y:' in s
    assert 'z:' in s and '8' in s
    assert 'w:' in s and 'u:' in s
    assert '9' in s

def test_safe_json_parse():
    good = '{"action": 5}'
    bad = '{action: 5}'  # Invalid JSON
    assert safe_json_parse(good) == {"action": 5}, f"Failed to parse valid JSON: {good}"
    assert safe_json_parse(bad) is None, f"Should return None for invalid JSON: {bad}"

if __name__ == "__main__":
    test_flatten_dict_basic()
    print("flatten_dict basic test passed.")
    test_dict_to_str_example()
    print("dict_to_str example test passed.")
    test_safe_json_parse()
    print("safe_json_parse test passed.")
