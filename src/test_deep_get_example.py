from utils import deep_get

def example_deep_get():
    d = {
        'a': {'b': {'c': 5}},
        'x': {'y': 42},
        'foo': {'bar': 'baz'},
    }
    val1 = deep_get(d, ['a', 'b', 'c'])   # 5
    val2 = deep_get(d, ['x', 'y'])        # 42
    val3 = deep_get(d, ['a', 'b', 'z'])   # None
    val4 = deep_get(d, ['foo', 'bar'])    # 'baz'
    val5 = deep_get(d, ['a', 'b'])        # {'c': 5}
    val6 = deep_get(d, ['a', 'b', 'c', 'd']) # None
    print("val1:", val1)
    print("val2:", val2)
    print("val3:", val3)
    print("val4:", val4)
    print("val5:", val5)
    print("val6:", val6)

if __name__ == "__main__":
    example_deep_get()
