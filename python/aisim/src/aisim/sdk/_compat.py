# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for the legacy aisim.sdk compatibility surface."""

from __future__ import annotations

import importlib
import sys
from collections.abc import MutableMapping
from types import ModuleType

_FACADE_LOCAL_ATTRIBUTES = frozenset(
    {
        "__all__",
        "__builtins__",
        "__cached__",
        "__dir__",
        "__doc__",
        "__file__",
        "__getattr__",
        "__loader__",
        "__name__",
        "__package__",
        "__path__",
        "__spec__",
        "_canonical_module",
        "_export_public_package",
    }
)


class _CanonicalPackageFacade(ModuleType):
    """Keep package metadata local while forwarding API mutations."""

    @property
    def __dict__(self) -> dict[str, object]:
        """Expose delegated attributes to tools that inspect before patching.

        Patching tools may save the original value from this mapping.
        Hiding canonical names can make their undo delete the real attribute.
        Build a fresh view so canonical updates remain visible, while import
        metadata continues to describe this compatibility package.
        Mutating this snapshot does not update either module; use attribute
        assignment to change delegated names.
        """
        namespace = super().__dict__
        canonical = namespace.get("_canonical_module")
        if canonical is None:
            return namespace
        return {**vars(canonical), **namespace}

    def __setattr__(self, name: str, value: object) -> None:
        canonical = super().__dict__.get("_canonical_module")
        if canonical is None or name in _FACADE_LOCAL_ATTRIBUTES:
            super().__setattr__(name, value)
            return
        setattr(canonical, name, value)

    def __delattr__(self, name: str) -> None:
        canonical = super().__dict__.get("_canonical_module")
        if canonical is None or name in _FACADE_LOCAL_ATTRIBUTES:
            super().__delattr__(name)
            return
        delattr(canonical, name)


def alias_module(alias_name: str, canonical_name: str) -> ModuleType:
    """Make an import path resolve to the canonical module object.

    Re-exporting names would create two distinct module objects. That can split
    module-level caches and private state. Replacing the wrapper entry in
    sys.modules makes both import paths share all implementation state.
    """
    module = importlib.import_module(canonical_name)
    sys.modules[alias_name] = module
    return module


def export_public_package(
    canonical_name: str,
    namespace: MutableMapping[str, object],
) -> tuple[ModuleType, list[str]]:
    """Delegate an API while keeping the legacy package search path.

    Models and operations retain real compatibility packages so their legacy
    child-module wrappers remain importable. Reads, writes, and deletes of API
    attributes are delegated to the canonical package so tools such as
    ``unittest.mock.patch`` behave the same through either import path.
    """
    module = importlib.import_module(canonical_name)
    public_names = list(getattr(module, "__all__", ()))
    namespace["_canonical_module"] = module
    facade = sys.modules[str(namespace["__name__"])]
    facade.__class__ = _CanonicalPackageFacade
    return module, public_names
