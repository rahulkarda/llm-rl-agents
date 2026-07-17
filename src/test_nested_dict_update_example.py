from utils import nested_dict_update

if __name__ == "__main__":
    d = {'a': {'b': {'c': 1}}, 'x': 2}
    print("Original dict:", d)
    updates = {'a.b.c': 99, 'x': 5}
    nested_dict_update(d, updates)
    print("After nested_dict_update:", d)
    # Expected: {'a': {'b': {'c': 99}}, 'x': 5}
