from utils import flatten_dict_keys

def test_flatten_dict_keys_example():
    d = {
        'a': 1,
        'b': {
            'c': 2,
            'd': {
                'e': 3,
                'f': 4
            }
        },
        'g': 5
    }
    keys = flatten_dict_keys(d)
    print("Flattened keys:", keys)
    expected = ['a', 'b.c', 'b.d.e', 'b.d.f', 'g']
    assert set(keys) == set(expected), f"Expected {expected}, got {keys}"

if __name__ == "__main__":
    test_flatten_dict_keys_example()
    print("flatten_dict_keys example test passed.")
