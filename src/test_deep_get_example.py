from utils import deep_get

if __name__ == "__main__":
    d = {'a': {'b': {'c': 5}}, 'x': {'y': 42}}
    val1 = deep_get(d, ['a', 'b', 'c'])   # 5
    val2 = deep_get(d, ['x', 'y'])        # 42
    val3 = deep_get(d, ['a', 'b', 'z'])   # None
    val4 = deep_get(d, ['a', 'b'])        # {'c': 5}
    val5 = deep_get(d, ['a', 'b', 'c', 'd']) # None
    print("val1:", val1)
    print("val2:", val2)
    print("val3:", val3)
    print("val4:", val4)
    print("val5:", val5)
