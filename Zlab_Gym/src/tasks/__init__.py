"""Zlab task registry for Z01 tracking, velocity, and soccer tasks."""

import importlib

_TASK_GROUPS = {
    "tracking": (
        "src.tasks.tracking.config.z01_std_add_head",
    ),
    "velocity": (
        "src.tasks.velocity.config.z01_std_add_head",
    ),
    "soccer": ("src.tasks.soccer",),
}

for _group_modules in _TASK_GROUPS.values():
    for module_name in _group_modules:
        importlib.import_module(module_name)
