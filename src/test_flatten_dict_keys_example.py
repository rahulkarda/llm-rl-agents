from utils import flatten_dict_keys

if __name__ == "__main__":
    d = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}
    keys = flatten_dict_keys(d)
    print("d:", d)
    print("flatten_dict_keys:", keys)  # Expected: ['a', 'b.c', 'b.d.e']
