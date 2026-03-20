#!/usr/bin/env python3
from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class AxisControl(ttk.LabelFrame):
    def __init__(self, master: tk.Misc, title: str, minimum: int, maximum: int, initial: int) -> None:
        super().__init__(master, text=title, padding=12)
        self.minimum = minimum
        self.maximum = maximum

        self.value_var = tk.IntVar(value=initial)
        self.entry_var = tk.StringVar(value=str(initial))
        self.index_var = tk.StringVar(value=f"index={initial}")

        self.columnconfigure(0, weight=1)

        self.scale = ttk.Scale(
            self,
            from_=minimum,
            to=maximum,
            orient="horizontal",
            command=self._on_scale,
        )
        self.scale.set(initial)
        self.scale.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 10))

        ttk.Label(self, textvariable=self.index_var, width=12).grid(row=1, column=0, sticky="w")
        self.entry = ttk.Entry(self, textvariable=self.entry_var, width=12)
        self.entry.grid(row=1, column=1, sticky="ew", padx=8)
        ttk.Button(self, text="Set", command=self._apply_entry).grid(row=1, column=2, sticky="e")

    def _on_scale(self, value: str) -> None:
        index = int(round(float(value)))
        index = max(self.minimum, min(self.maximum, index))
        self.value_var.set(index)
        self.entry_var.set(str(index))
        self.index_var.set(f"index={index}")

    def _apply_entry(self) -> None:
        try:
            index = int(round(float(self.entry_var.get().strip())))
        except ValueError:
            return
        index = max(self.minimum, min(self.maximum, index))
        self.scale.set(index)
        self._on_scale(str(index))

    def value(self) -> int:
        return int(self.value_var.get())


class TkSmokeApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Tkinter Smoke Test")
        self.geometry("960x640+80+80")

        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=16)
        left.grid(row=0, column=0, sticky="nsew")
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)

        ttk.Label(left, text="Tkinter Prototype", font=("Helvetica", 20, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 12)
        )

        self.info = tk.Text(left, wrap="word", height=20)
        self.info.grid(row=1, column=0, sticky="nsew")
        self.info.insert(
            "1.0",
            "If this window appears and the controls respond, tkinter is working.\n\n"
            "Use the controls on the right and click Refresh to update this panel.",
        )
        self.info.configure(state="disabled")

        right = ttk.Frame(self, padding=16)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)

        self.crossline = AxisControl(right, "Crossline", 0, 235, 118)
        self.crossline.grid(row=0, column=0, sticky="ew", pady=(0, 12))

        self.inline = AxisControl(right, "Inline", 0, 295, 148)
        self.inline.grid(row=1, column=0, sticky="ew", pady=(0, 12))

        self.sample = AxisControl(right, "Sample", 0, 250, 125)
        self.sample.grid(row=2, column=0, sticky="ew", pady=(0, 12))

        ttk.Button(right, text="Refresh", command=self.refresh_info).grid(row=3, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(right, text="Quit", command=self.destroy).grid(row=4, column=0, sticky="ew", pady=(8, 0))

    def refresh_info(self) -> None:
        text = (
            "Tkinter control test\n\n"
            f"Crossline: {self.crossline.value()}\n"
            f"Inline:    {self.inline.value()}\n"
            f"Sample:    {self.sample.value()}\n\n"
            "If slider dragging and direct input both work, the environment is ready."
        )
        self.info.configure(state="normal")
        self.info.delete("1.0", "end")
        self.info.insert("1.0", text)
        self.info.configure(state="disabled")


def main() -> int:
    app = TkSmokeApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
