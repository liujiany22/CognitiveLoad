"""Auto-import every loader module so that subclasses register themselves."""

import importlib
import pathlib
import pkgutil

_pkg_dir = pathlib.Path(__file__).parent
for _, module_name, _ in pkgutil.iter_modules([str(_pkg_dir)]):
    if not module_name.startswith("_"):
        importlib.import_module(f".{module_name}", package=__name__)
