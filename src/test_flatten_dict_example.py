from utils import flatten_dict

if __name__ == "__main__":
    d = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}
    flat = flatten_dict(d)
    print("Original dict:", d)
    print("Flattened dict:", flat)
    # Expected: {'a': 1, 'b.c': 2, 'b.d.e': 3}
