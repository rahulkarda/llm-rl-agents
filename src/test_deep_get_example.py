from utils import deep_get

def test_deep_get_basic():
    d = {'a': {'b': {'c': 5}}, 'x': {'y': 42}}
    assert deep_get(d, ['a', 'b', 'c']) == 5
    assert deep_get(d, ['x', 'y']) == 42
    assert deep_get(d, ['a', 'b', 'z']) is None
    assert deep_get(d, ['a', 'b']) == {'c': 5}
    assert deep_get(d, ['a', 'b', 'c', 'd']) is None

def test_deep_get_edge_cases():
    d2 = {'foo': 3, 'bar': {'baz': {'qux': 7}}}
    assert deep_get(d2, ['foo']) == 3
    assert deep_get(d2, ['bar', 'baz', 'qux']) == 7
    assert deep_get(d2, ['bar', 'baz', 'missing']) is None
    assert deep_get(d2, ['bar', 'baz']) == {'qux': 7}
    assert deep_get(d2, ['bar', 'baz', 'qux', 'extra']) is None

def test_deep_get_non_dict_mid():
    d3 = {'a': {'b': 2}}
    assert deep_get(d3, ['a', 'b', 'c']) is None
    assert deep_get(d3, ['a', 'b']) == 2

def run_all():
    test_deep_get_basic()
    test_deep_get_edge_cases()
    test_deep_get_non_dict_mid()
    print("deep_get utility tests passed.")

if __name__ == "__main__":
    run_all()
