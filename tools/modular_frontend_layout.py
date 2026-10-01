"""Layout helpers for opt-in split graphs with tall editable widgets."""


def spread_frontend_columns(workflow: dict, *, gap: int = 48) -> None:
    """Preserve column order while keeping node controls from overlapping."""
    columns: dict[float, list[dict]] = {}
    for node in workflow["nodes"]:
        columns.setdefault(node["pos"][0], []).append(node)
    for column in columns.values():
        bottom = None
        for node in sorted(column, key=lambda item: (item["pos"][1], item["id"])):
            top = node["pos"][1] if bottom is None else max(node["pos"][1], bottom)
            node["pos"][1] = top
            bottom = top + node["size"][1] + gap
