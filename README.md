locus
=====

[![](https://dev.azure.com/lycantropos/locus/_apis/build/status/lycantropos.locus?branchName=master)](https://dev.azure.com/lycantropos/locus/_build/latest?definitionId=25&branchName=master "Azure Pipelines")
[![](https://readthedocs.org/projects/locus/badge/?version=latest)](https://locus.readthedocs.io/en/latest "Documentation")
[![](https://codecov.io/gh/lycantropos/locus/branch/master/graph/badge.svg)](https://codecov.io/gh/lycantropos/locus "Codecov")
[![](https://img.shields.io/github/license/lycantropos/locus.svg)](https://github.com/lycantropos/locus/blob/master/LICENSE "License")
[![](https://badge.fury.io/py/locus.svg)](https://badge.fury.io/py/locus "PyPI")

In what follows `python` is an alias for `python3.5` or `pypy3.5`
or any later version (`python3.6`, `pypy3.6` and so on).

Installation
------------

Install the latest `pip` & `setuptools` packages versions
```bash
python -m pip install --upgrade pip setuptools
```

### User

Download and install the latest stable version from `PyPI` repository:
```bash
python -m pip install --upgrade locus
```

### Developer

Download the latest version from `GitHub` repository
```bash
git clone https://github.com/lycantropos/locus.git
cd locus
```

Install dependencies
```bash
python -m pip install -r requirements.txt
```

Install
```bash
python setup.py install
```

Usage
-----
```python
>>> from ground.geometries import to_point_cls, to_segment_cls
>>> Point, Segment = to_point_cls(), to_segment_cls()
>>> from locus import kd
>>> points = list(map(Point, range(-10, 11), range(0, 20)))
>>> kd_tree = kd.Tree(points)
>>> kd_tree.nearest_index(Point(0, 0)) == 5
True
>>> kd_tree.nearest_point(Point(0, 0)) == Point(-5, 5)
True
>>> kd_tree.n_nearest_indices(2, Point(0, 0)) == [6, 5]
True
>>> kd_tree.n_nearest_points(2, Point(0, 0)) == [Point(-4, 6), Point(-5, 5)]
True
>>> kd_tree.find_ball_indices(Point(0, 3), 5) == [6, 7]
True
>>> kd_tree.find_ball_points(Point(0, 3), 5) == [Point(-4, 6), Point(-3, 7)]
True
>>> kd_tree.find_interval_indices(((-1, 1), (0, 10))) == [9, 10]
True
>>> kd_tree.find_interval_points(((-1, 1), (0, 10))) == [Point(-1, 9), Point(0, 10)]
True
>>> from locus import r
>>> intervals = list(zip(zip(range(-10, 11), range(0, 20)), 
...                      zip(range(-20, 0), range(-10, 11))))
>>> r_tree = r.Tree(intervals)
>>> r_tree.nearest_index(Point(0, 0)) == 10
True
>>> r_tree.nearest_interval(Point(0, 0))
((0, 10), (-10, 0))
>>> r_tree.n_nearest_indices(2, Point(0, 0)) == [10, 11]
True
>>> r_tree.n_nearest_intervals(2, Point(0, 0)) == [((0, 10), (-10, 0)), ((1, 11), (-9, 1))]
True
>>> r_tree.find_subsets_indices(((0, 10), (-10, 10))) == [10]
True
>>> r_tree.find_subsets(((0, 10), (-10, 10))) == [((0, 10), (-10, 0))]
True
>>> r_tree.find_supersets_indices(((0, 10), (-10, 0))) == [10]
True
>>> r_tree.find_supersets(((0, 10), (-10, 0))) == [((0, 10), (-10, 0))]
True

```

Development
-----------

### Bumping version

#### Preparation

Install
[bump2version](https://github.com/c4urself/bump2version#installation).

#### Pre-release

Choose which version number category to bump following [semver
specification](http://semver.org/).

Test bumping version
```bash
bump2version --dry-run --verbose $CATEGORY
```

where `$CATEGORY` is the target version number category name, possible
values are `patch`/`minor`/`major`.

Bump version
```bash
bump2version --verbose $CATEGORY
```

This will set version to `major.minor.patch-alpha`. 

#### Release

Test bumping version
```bash
bump2version --dry-run --verbose release
```

Bump version
```bash
bump2version --verbose release
```

This will set version to `major.minor.patch`.

### Running tests

Install dependencies
```bash
python -m pip install -r requirements-tests.txt
```

Plain
```bash
pytest
```

Inside `Docker` container:
- with `CPython`
  ```bash
  docker-compose --file docker-compose.cpython.yml up
  ```
- with `PyPy`
  ```bash
  docker-compose --file docker-compose.pypy.yml up
  ```

`Bash` script (e.g. can be used in `Git` hooks):
- with `CPython`
  ```bash
  ./run-tests.sh
  ```
  or
  ```bash
  ./run-tests.sh cpython
  ```

- with `PyPy`
  ```bash
  ./run-tests.sh pypy
  ```

`PowerShell` script (e.g. can be used in `Git` hooks):
- with `CPython`
  ```powershell
  .\run-tests.ps1
  ```
  or
  ```powershell
  .\run-tests.ps1 cpython
  ```
- with `PyPy`
  ```powershell
  .\run-tests.ps1 pypy
  ```
