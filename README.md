locus
=====

[![](https://travis-ci.com/lycantropos/locus.svg?branch=master)](https://travis-ci.com/lycantropos/locus "Travis CI")
[![](https://dev.azure.com/lycantropos/locus/_apis/build/status/lycantropos.locus?branchName=master)](https://dev.azure.com/lycantropos/locus/_build/latest?definitionId=25&branchName=master "Azure Pipelines")
[![](https://readthedocs.org/projects/locus/badge/?version=latest)](https://locus.readthedocs.io/en/latest "Documentation")
[![](https://codecov.io/gh/lycantropos/locus/branch/master/graph/badge.svg)](https://codecov.io/gh/lycantropos/locus "Codecov")
[![](https://img.shields.io/github/license/lycantropos/locus.svg)](https://github.com/lycantropos/locus/blob/master/LICENSE "License")
[![](https://badge.fury.io/py/locus.svg)](https://badge.fury.io/py/locus "PyPI")

In what follows
- `python` is an alias for `python3.5` or any later
version (`python3.6` and so on),
- `pypy` is an alias for `pypy3.5` or any later
version (`pypy3.6` and so on).

Installation
------------

Install the latest `pip` & `setuptools` packages versions:
- with `CPython`
  ```bash
  python -m pip install --upgrade pip setuptools
  ```
- with `PyPy`
  ```bash
  pypy -m pip install --upgrade pip setuptools
  ```

### User

Download and install the latest stable version from `PyPI` repository:
- with `CPython`
  ```bash
  python -m pip install --upgrade locus
  ```
- with `PyPy`
  ```bash
  pypy -m pip install --upgrade locus
  ```

### Developer

Download the latest version from `GitHub` repository
```bash
git clone https://github.com/lycantropos/locus.git
cd locus
```

Install dependencies:
- with `CPython`
  ```bash
  python -m pip install --force-reinstall -r requirements.txt
  ```
- with `PyPy`
  ```bash
  pypy -m pip install --force-reinstall -r requirements.txt
  ```

Install:
- with `CPython`
  ```bash
  python setup.py install
  ```
- with `PyPy`
  ```bash
  pypy setup.py install
  ```

Usage
-----
```python
>>> from locus import kd
>>> points = list(zip(range(-10, 11), range(0, 20)))
>>> kd_tree = kd.Tree(points)
>>> kd_tree.nearest_index((0, 0))
5
>>> kd_tree.nearest_point((0, 0))
(-5, 5)
>>> kd_tree.n_nearest_indices(2, (0, 0))
[6, 5]
>>> kd_tree.n_nearest_points(2, (0, 0))
[(-4, 6), (-5, 5)]
>>> kd_tree.find_ball_indices((0, 3), 5)
[6, 7]
>>> kd_tree.find_ball_points((0, 3), 5)
[(-4, 6), (-3, 7)]
>>> kd_tree.find_interval_indices(((-1, 1), (0, 10)))
[9, 10]
>>> kd_tree.find_interval_points(((-1, 1), (0, 10)))
[(-1, 9), (0, 10)]
>>> from locus import r
>>> intervals = list(zip(zip(range(-10, 11), range(0, 20)), 
...                      zip(range(-20, 0), range(-10, 11))))
>>> r_tree = r.Tree(intervals)
>>> r_tree.nearest_index((0, 0))
10
>>> r_tree.nearest_interval((0, 0))
((0, 10), (-10, 0))
>>> r_tree.n_nearest_indices(2, (0, 0))
[10, 11]
>>> r_tree.n_nearest_intervals(2, (0, 0))
[((0, 10), (-10, 0)), ((1, 11), (-9, 1))]
>>> r_tree.find_interval_indices(((0, 10), (-10, 10)))
[10]
>>> r_tree.find_interval_intervals(((0, 10), (-10, 10)))
[((0, 10), (-10, 0))]

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

Install dependencies:
- with `CPython`
  ```bash
  python -m pip install --force-reinstall -r requirements-tests.txt
  ```
- with `PyPy`
  ```bash
  pypy -m pip install --force-reinstall -r requirements-tests.txt
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
