"""Scrollable control column shared by the PyAbel Studio workspaces."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Optional


class ScrollablePanel(tk.Frame):
    """A vertically scrolling column of GUI controls.

    Place controls in :attr:`content`. The column scrolls with its scrollbar
    and with the mouse wheel or trackpad anywhere over it, so every control
    stays reachable however short the window is. Text and list widgets keep
    scrolling their own contents, and the wheel scrolls the column instead of
    changing a combobox value it happens to pass over.

    ``width`` fixes the column width; by default the column is as wide as its
    controls. With ``fill_height`` the controls are stretched to the visible
    height while they fit, so widgets that expand inside :attr:`content`
    still use the spare room, and the column scrolls only once the window is
    shorter than the controls. Inside a themed (ttk) container the column's
    frames are themed too, so it keeps the background of its surroundings.
    """

    PIXELS_PER_UNIT = 20
    # Windows, and every platform from Tk 8.7 on, report one wheel detent as a
    # delta of 120. Three units per detent matches Tk's own list widgets.
    _DELTA_PER_UNIT = 40
    # Widgets with their own scrollable contents keep the wheel to themselves.
    _SELF_SCROLLING = frozenset({"Text", "Listbox", "Canvas", "Treeview"})

    def __init__(
        self,
        parent: tk.Misc,
        *,
        width: Optional[int] = None,
        fill_height: bool = False,
        **frame_options,
    ):
        super().__init__(parent, **frame_options)
        themed = isinstance(parent, ttk.Widget)
        if themed:
            background = ttk.Style(self).lookup("TFrame", "background")
            self.configure(background=background)
            frame_type = ttk.Frame
        else:
            background = self.cget("background")
            frame_type = tk.Frame
        self._auto_width = width is None
        self._fill_height = fill_height
        self._wheel_remainder = 0
        self._touchpad_remainder = 0
        self._wheel_tag = f"ScrollablePanel{id(self)}"
        # Tk 8.6 on macOS reports wheel and trackpad deltas in scroll lines.
        self._wheel_in_lines = (
            self.tk.call("tk", "windowingsystem") == "aqua"
            and tk.TkVersion < 8.7
        )

        self.canvas = tk.Canvas(
            self,
            highlightthickness=0,
            borderwidth=0,
            background=background,
            yscrollincrement=self.PIXELS_PER_UNIT,
        )
        if width is not None:
            self.canvas.configure(width=width)
        self.scrollbar = ttk.Scrollbar(
            self, orient=tk.VERTICAL, command=self.canvas.yview
        )
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # The body is always at least as tall as the visible canvas when
        # ``fill_height`` is set, by giving its only row a minimum size. Its
        # requested size still follows the controls, so it is reported through
        # <Configure> whenever controls are shown, hidden, or resized.
        self._body = frame_type(self.canvas)
        self._body.grid_columnconfigure(0, weight=1)
        self._body.grid_rowconfigure(0, weight=1)
        self.content = frame_type(self._body)
        self.content.grid(row=0, column=0, sticky="nsew")
        self._window = self.canvas.create_window(
            (0, 0), window=self._body, anchor="nw"
        )

        self._body.bind("<Configure>", self._on_body_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self._bind_wheel_events()
        self.after_idle(self.refresh)

    def refresh(self) -> None:
        """Resynchronise scroll limits and wheel bindings with the content."""
        if not self.winfo_exists():
            return
        if self._auto_width:
            wanted = self.content.winfo_reqwidth()
            if self.canvas.winfo_reqwidth() != wanted:
                self.canvas.configure(width=wanted)
        bounds = self.canvas.bbox(self._window)
        if bounds is not None:
            self.canvas.configure(scrollregion=bounds)
        self._tag_for_wheel(self)

    def can_scroll(self) -> bool:
        """Return whether the controls are taller than the visible column."""
        top, bottom = self.canvas.yview()
        return top > 0.0 or bottom < 1.0

    def scroll_to_end(self) -> None:
        """Reveal the bottom of the controls."""
        self.update_idletasks()
        self.refresh()
        self.canvas.yview_moveto(1.0)

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self._window, width=max(1, event.width))
        if self._fill_height:
            self._body.grid_rowconfigure(0, minsize=max(1, event.height))
        self.refresh()

    def _on_body_configure(self, event: tk.Event) -> None:
        self.refresh()

    # Mouse wheel and trackpad scrolling
    def _bind_wheel_events(self) -> None:
        self.bind_class(self._wheel_tag, "<MouseWheel>", self._on_mousewheel)
        # Shift+wheel and sideways trackpad swipes are horizontal scrolling.
        self.bind_class(
            self._wheel_tag, "<Shift-MouseWheel>", lambda event: "break"
        )
        if self.tk.call("tk", "windowingsystem") == "x11":
            self.bind_class(
                self._wheel_tag, "<Button-4>", lambda event: self._scroll(-3)
            )
            self.bind_class(
                self._wheel_tag, "<Button-5>", lambda event: self._scroll(3)
            )
        try:  # Tk 9 delivers precise trackpad motion as its own event type.
            self.bind_class(
                self._wheel_tag, "<TouchpadScroll>", self._on_touchpad_scroll
            )
        except tk.TclError:
            pass

    def _tag_for_wheel(self, widget: tk.Misc) -> None:
        """Route wheel events over ``widget`` and its children to the panel."""
        own_scrolling = widget.winfo_class() in self._SELF_SCROLLING
        if widget is self.canvas or not own_scrolling:
            tags = widget.bindtags()
            if self._wheel_tag not in tags:
                # Ahead of the class bindings, so "break" stops a combobox
                # from also stepping through its values.
                widget.bindtags(tags[:1] + (self._wheel_tag,) + tags[1:])
        for child in widget.winfo_children():
            if not isinstance(child, tk.Toplevel):  # pop-ups scroll alone
                self._tag_for_wheel(child)

    def _on_mousewheel(self, event: tk.Event) -> str:
        delta = int(getattr(event, "delta", 0))
        if self._wheel_in_lines:
            return self._scroll(-delta)
        self._wheel_remainder += delta
        steps = int(self._wheel_remainder / self._DELTA_PER_UNIT)
        self._wheel_remainder -= steps * self._DELTA_PER_UNIT
        return self._scroll(-steps)

    def _on_touchpad_scroll(self, event: tk.Event) -> str:
        # The low 16 bits carry the signed vertical motion in pixels.
        delta_y = int(event.delta) & 0xFFFF
        if delta_y >= 0x8000:
            delta_y -= 0x10000
        self._touchpad_remainder += delta_y
        steps = int(self._touchpad_remainder / self.PIXELS_PER_UNIT)
        self._touchpad_remainder -= steps * self.PIXELS_PER_UNIT
        return self._scroll(-steps)

    def _scroll(self, units: int) -> str:
        if units and self.can_scroll():
            self.canvas.yview_scroll(int(units), "units")
        return "break"
