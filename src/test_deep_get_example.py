from utils import deep_get

if __name__ == "__main__":
    d = {
        'a': {'b': {'c': 5}},
        'x': {'y': 42},
        'empty': {},
        'none': None,
    }
    v1 = deep_get(d, ['a', 'b', 'c'])  # should be 5
    v2 = deep_get(d, ['x', 'y'])       # should be 42
    v3 = deep_get(d, ['a', 'b', 'z'])  # should be None
    v4 = deep_get(d, ['empty', 'foo']) # should be None
    v5 = deep_get(d, ['none', 'foo'])  # should be None
    print("deep_get(d, ['a', 'b', 'c']):", v1)
    print("deep_get(d, ['x', 'y']):", v2)
    print("deep_get(d, ['a', 'b', 'z']):", v3)
    print("deep_get(d, ['empty', 'foo']):", v4)
    print("deep_get(d, ['none', 'foo']):", v5)
    assert v1 == 5
    assert v2 == 42
    assert v3 is None
    assert v4 is None
    assert v5 is None
    print("deep_get tests passed.")
