# Efficient domain-adapted continual pretraining for (almost) any language

## Development

**tldr**: use your branch, tests and logging are good, NumPy style for docstrings, use poetry, isort, black and ruff.
Name stuff adequately, use PEP and all the programming knowledge you have.
---

### Python

Most packages are stable at python version 3.10, so we use it as default as noted in 
`pyproject.toml`. If you have any reasoning against that, please let us know and we'd discuss that.

### Git
- Each person has their own branch for development, please use those rather than pushing everything to `main`

- Naming of the branches should be so that it would be possible to identify
who is the owner of a branch (for example, `jd-dev` where "jd" stands for John Doe).

- Avoid using `--amend` and `force` stuff (only use when necessary) as those might affect the work of other people.
- Write readable commit messages.
- In general, the `main` branch should only contain code that is stable and "production-ready"
(so no random exceptions would show up if we were to run it).
- Avoid storing and versioning some redundant files like `123123.py` or `test.ipynb`. Only stage and version
the stuff that is relevant to the project.

- Data goes to `data` (make dirs inside for different data sources, for example), notebooks go to `notebooks`,
development code goes to the `efficient_multilingual_continual_pretraining` source.


### Tests
Tests are good (we suggest using `pytest`) but not necessary and should be kept near the code being tested. For example, you could have:

```
.
└── efficient_multilingual_continual_pretraining/
    └── models/
        ├── tests/
        │   └── test_my_model_1.py
        ├── my_model_1.py
        └── __init__.py
```

### Logging
Logging is also good and is often helpful. `loguru` provides straightforward logging out-of-the-box.
Please store logs in the `logs` folder which is excluded from versioning in .gitignore
and do not version those to avoid clutter.

### Poetry
We use [`poetry` to manage the dependencies](https://python-poetry.org/). The stuff you should know:

- `poetry install` installs the current dependencies (including `dev` ones).
- `poetry add <package_name>` works similarly to `pip install`. 
- The installed packages can be seen in `pyproject.toml`.
- `poetry` also generates `poetry.lock` file. Be sure to version it as well.

### Codestyle

We also suggest using [**File watchers**](https://medium.com/compendium/automatically-run-black-in-pycharm-on-windows-d2eab855a918)
to keep consistent codestyle. The recommended watchers are `isort`, `black` and `ruff`. You could use the `watchers.xml` found
[here](https://drive.google.com/file/d/1ycj9xTUWl4bfDnEbvlBcunvW8QBcbjvX/view?usp=sharing).
Replace the paths to programs to the ones you have.

Talking about docstrings, we suggest using the [NumPy style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html). 


#### Known watcher issues

1. `isort` aims to clear the unused imports. If you are making an import in the
`init.py` file, the import most probably will be unused. To solve this, use the 
`__all__` [(stackoverflow)](https://stackoverflow.com/questions/44834/what-does-all-mean-in-python):
