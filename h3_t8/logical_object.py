"""Resolve a selected object before Core physically installs object patches."""

import comfy.utils


def logical_object(model, name):
    """Resolve descendants of selected or backed-up parent object patches."""
    parts = name.split(".")
    for patches in (
        getattr(model, "object_patches", {}),
        getattr(model, "object_patches_backup", {}),
    ):
        for count in range(len(parts), 0, -1):
            parent = ".".join(parts[:count])
            if parent in patches:
                return (
                    comfy.utils.get_attr(patches[parent], ".".join(parts[count:]))
                    if count < len(parts)
                    else patches[parent]
                )
    return model.get_model_object(name)
