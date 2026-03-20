"""
Formatting helpers for UI and logging text.
"""


def format_value(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.3f}".rstrip("0").rstrip(".")