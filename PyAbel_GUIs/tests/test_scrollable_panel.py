import tkinter as tk
from tkinter import ttk

import pytest

from PyAbel_GUIs.scrollable_panel import ScrollablePanel


@pytest.fixture
def root():
    try:
        window = tk.Tk()
    except tk.TclError as exc:  # no display, e.g. a headless CI runner
        pytest.skip(f"Tk is unavailable: {exc}")
    window.geometry("320x240")
    yield window
    window.destroy()


def wheel_down(widget):
    """Send a downward wheel detent to ``widget``."""
    widget.event_generate("<MouseWheel>", delta=-120)
    widget.update()


def fill(panel, rows):
    for row in range(rows):
        ttk.Entry(panel.content).grid(row=row, column=0, pady=4)


def test_tall_controls_scroll_and_short_ones_do_not(root):
    panel = ScrollablePanel(root)
    panel.pack(fill=tk.BOTH, expand=True)
    fill(panel, 2)
    root.update()
    assert not panel.can_scroll()

    fill(panel, 40)
    root.update()
    assert panel.can_scroll()
    panel.scroll_to_end()
    assert panel.canvas.yview()[1] == pytest.approx(1.0)


def test_column_is_as_wide_as_its_controls_unless_fixed(root):
    auto = ScrollablePanel(root)
    auto.pack(side=tk.LEFT, fill=tk.Y)
    ttk.Label(auto.content, text="x" * 30).pack()
    fixed = ScrollablePanel(root, width=50)
    fixed.pack(side=tk.LEFT, fill=tk.Y)
    ttk.Label(fixed.content, text="x" * 30).pack()
    root.update()
    assert auto.canvas.winfo_width() == auto.content.winfo_reqwidth()
    assert fixed.canvas.winfo_width() == 50


def test_fill_height_stretches_short_content_to_the_window(root):
    panel = ScrollablePanel(root, fill_height=True)
    panel.pack(fill=tk.BOTH, expand=True)
    panel.content.grid_rowconfigure(0, weight=1)
    log = tk.Text(panel.content, height=2)
    log.grid(row=0, column=0, sticky="nsew")
    root.update()
    assert panel.content.winfo_height() >= panel.canvas.winfo_height()
    assert log.winfo_height() > log.winfo_reqheight()
    assert not panel.can_scroll()

    # Once the controls no longer fit, they keep their size and scroll.
    fill(panel, 40)
    root.update()
    assert panel.content.winfo_height() == panel.content.winfo_reqheight()
    assert panel.can_scroll()


def test_wheel_over_any_control_scrolls_the_column(root):
    panel = ScrollablePanel(root)
    panel.pack(fill=tk.BOTH, expand=True)
    fill(panel, 40)
    root.update()
    wheel_down(panel.content.winfo_children()[3])
    assert panel.canvas.yview()[0] > 0


def test_wheel_scrolls_the_column_instead_of_changing_a_combobox(root):
    panel = ScrollablePanel(root)
    panel.pack(fill=tk.BOTH, expand=True)
    combo = ttk.Combobox(panel.content, values=("a", "b", "c"))
    combo.set("a")
    combo.grid(row=0, column=0)
    for row in range(1, 40):
        ttk.Entry(panel.content).grid(row=row, column=0, pady=4)
    root.update()
    wheel_down(combo)
    assert combo.get() == "a"
    assert panel.canvas.yview()[0] > 0


def test_text_widgets_keep_their_own_wheel_scrolling(root):
    panel = ScrollablePanel(root)
    panel.pack(fill=tk.BOTH, expand=True)
    log = tk.Text(panel.content, height=3)
    log.insert("1.0", "\n".join(str(line) for line in range(100)))
    log.grid(row=0, column=0)
    fill(panel, 1)
    root.update()
    assert panel._wheel_tag not in log.bindtags()
    assert panel._wheel_tag in panel.content.winfo_children()[1].bindtags()


def test_controls_added_later_also_scroll_the_column(root):
    panel = ScrollablePanel(root)
    panel.pack(fill=tk.BOTH, expand=True)
    fill(panel, 40)
    root.update()
    late = ttk.Entry(panel.content)
    late.grid(row=99, column=0)
    root.update()
    assert panel._wheel_tag in late.bindtags()


def test_themed_parent_gets_themed_frames(root):
    plain = ScrollablePanel(root)
    themed = ScrollablePanel(ttk.Frame(root))
    assert plain.content.winfo_class() == "Frame"
    assert themed.content.winfo_class() == "TFrame"


def test_popups_inside_the_column_keep_their_own_wheel(root):
    panel = ScrollablePanel(root)
    panel.pack(fill=tk.BOTH, expand=True)
    popup = tk.Toplevel(panel.content)
    entry = ttk.Entry(popup)
    entry.pack()
    root.update()
    panel.refresh()
    assert panel._wheel_tag not in popup.bindtags()
    assert panel._wheel_tag not in entry.bindtags()
