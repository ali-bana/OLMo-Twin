"""Shared helpers for the OLMo-Twin code.

The package currently holds one module, :mod:`olmo_twin.paths`, which resolves
the data root (environment variable ``OLMO_TWIN_DATA``, default
``<repository root>/data``) and defines the directory layout that all scripts
use. Scripts import it as::

    from olmo_twin import paths

Use the package from a checkout of the repository: either run the scripts
without installing anything (each script adds the repository root to
``sys.path`` itself) or install the repository in editable mode
(``pip install -e .``). A non-editable install is not supported, because the
repository root is derived from the location of the package.
"""

__all__ = ["paths"]
