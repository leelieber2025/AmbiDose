"""Package version."""

from __future__ import annotations

__all__ = ["__version__", "version", "version_tuple"]

__version__ = "0.3.1"
version = __version__

_parts: list[int | str] = []
for _seg in __version__.split("."):
    try:
        _parts.append(int(_seg))
    except ValueError:
        _parts.append(_seg)
version_tuple: tuple[int | str, ...] = tuple(_parts)
