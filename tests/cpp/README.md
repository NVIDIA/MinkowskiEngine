# C++ Backend Unit Tests

## Installation

```
rm -rf build; CXX=g++; python setup.py install <test_name>
```

## Individual Test

```
python -m unittest <test_name>
python -m unittest <test_name> --nodebug # to remove deub flags
```

e.g.

```
python -m unittest coordinate_map_key_test
```
