from utils import deep_get

if __name__ == "__main__":
    d = {'a': {'b': {'c': 5}}, 'x': {'y': 42}}
    val1 = deep_get(d, ['a', 'b', 'c'])
    val2 = deep_get(d, ['x', 'y'])
    val3 = deep_get(d, ['a', 'b', 'z'])
    print("d:", d)
    print("deep_get(['a', 'b', 'c']):", val1)  # Expected: 5
    print("deep_get(['x', 'y']):", val2)      # Expected: 42
    print("deep_get(['a', 'b', 'z']):", val3) # Expected: None
