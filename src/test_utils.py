from utils import flatten_dict

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

if __name__ == "__main__":
    test_flatten_dict_basic()
    print("flatten_dict basic test passed.")
