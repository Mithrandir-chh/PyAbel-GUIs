import numpy as np
import abel
from scipy.ndimage import shift
from scipy.signal import find_peaks, peak_prominences
import matplotlib; matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from six.moves import tkinter as tk
from six.moves import tkinter_ttk as ttk
from six.moves import tkinter_tkfiledialog as filedialog

Abel_methods = ['basex', 'direct', 'hansenlaw', 'linbasex', 'onion_bordas',
                'onion_peeling', 'rbasex', 'three_point', 'two_point']

center_methods = ['com', 'convolution', 'gaussian', 'slice']

root = tk.Tk()
root.wm_title("Simple GUI PyAbel")

f = Figure(figsize=(5, 4), dpi=100)
a = f.add_subplot(111)


IM = None
AIM = None
raw_IM = None
centered_IM = None
speed_distribution = None
radial_coords = None



def update_plot(image=None):
    global IM, canvas, vmin_slider, vmax_slider, vmin_entry, vmax_entry
    if image is None:
        image = IM
    if image is not None:
        f.clf()
        a = f.add_subplot(111)
        vmin = float(vmin_entry.get())
        vmax = float(vmax_entry.get())
        im = a.imshow(image, vmin=vmin, vmax=vmax)
        f.colorbar(im)
        canvas.draw()


def update_slider_range(data):
    vmin = np.min(data)
    vmax = np.max(data)
    range_size = vmax - vmin

    if range_size <= 2:
        resolution = 0.05
    else:
        resolution = (vmax - vmin) / 100

    vmin_slider.config(from_=vmin, to=vmax, resolution=resolution)
    vmax_slider.config(from_=vmin, to=vmax, resolution=resolution)
    vmin_slider.set(vmin)
    vmax_slider.set(vmax)
    vmin_entry.delete(0, tk.END)
    vmin_entry.insert(0, f"{vmin:.2f}")
    vmax_entry.delete(0, tk.END)
    vmax_entry.insert(0, f"{vmax:.2f}")


def slider_changed(value):
    vmin_entry.delete(0, tk.END)
    vmin_entry.insert(0, f"{vmin_slider.get():.2f}")
    vmax_entry.delete(0, tk.END)
    vmax_entry.insert(0, f"{vmax_slider.get():.2f}")
    update_plot()


def entry_changed(*args):
    try:
        vmin = float(vmin_entry.get())
        vmax = float(vmax_entry.get())
        vmin_slider.set(vmin)
        vmax_slider.set(vmax)
        update_plot()
    except ValueError:
        pass  # Ignore invalid input

def _display():
    global IM, raw_IM, canvas, text
    text.insert(tk.END, "raw image\n")
    IM = raw_IM.copy()
    update_slider_range(IM)
    update_plot()

def _getfilename():
    global IM, raw_IM, centered_IM, text
    fn = filedialog.askopenfilename()
    text.delete(1.0, tk.END)
    text.insert(tk.END, f"reading image file {fn}\n")
    canvas.draw()

    if ".txt" in fn or ".dat" in fn:
        IM = np.loadtxt(fn)
    else:
        IM = imread(fn)

    if IM.shape[0] % 2 == 0:
        text.insert(tk.END, "make image odd size")
        IM = shift(IM, (-0.5, -0.5))[:-1, :-1]
    if len(IM.shape) == 3:
        IM = np.mean(IM, axis=2)

    raw_IM = IM.copy()
    centered_IM = None
    update_slider_range(IM)
    _display()

def _center():
    global cent, IM, raw_IM, centered_IM, text
    method = cent.get()
    text.delete(1.0, tk.END)
    text.insert(tk.END, f"centering image using abel.tools.center.center_image(method={method})\n")
    canvas.draw()
    centered_IM = abel.tools.center.center_image(raw_IM, method=method, odd_size=True)
    IM = centered_IM.copy()
    update_slider_range(IM)
    update_plot()

def _display_centered():
    global IM, raw_IM, centered_IM, canvas, text
    if centered_IM is not None:
        text.insert(tk.END, "centered raw image\n")
        IM = centered_IM.copy()
        update_slider_range(IM)
        update_plot()
    else:
        text.insert(tk.END, "Please center the image first\n")

def P2(x):
    return (3*x*x-1)/2

def PAD(theta, beta, amp):
    return amp*(1 + beta*P2(np.cos(theta)))

def _transform():
    global IM, AIM, raw_IM, centered_IM, canvas, transform, text
    method = transform.get()
    text.delete(1.0, tk.END)
    text.insert(tk.END, f"inverse Abel transform: {method}\n")
    if "basex" in method:
        text.insert(tk.END, "  first time calculation of the basis functions may take a while ...\n")
    if "direct" in method:
        text.insert(tk.END, "   calculation is slowed if Cython unavailable ...\n")
    canvas.draw()

    image_to_transform = centered_IM if centered_IM is not None else raw_IM

    AIM = abel.Transform(image_to_transform, method=method, direction="inverse", symmetry_axis=None)
    IM = AIM.transform
    update_slider_range(IM)
    update_plot()


def find_peak_ranges(x, y, threshold=10, rel_height=0.8, min_width=50):
    # Find peaks
    peaks, _ = find_peaks(y, height=threshold)

    # Calculate prominences
    prominences, _, _ = peak_prominences(y, peaks)

    peak_ranges = []
    for peak, prominence in zip(peaks, prominences):
        # Define a threshold based on the peak height and prominence
        height_threshold = y[peak] - prominence * rel_height

        # Find left boundary
        left = peak
        while left > 0 and y[left] > height_threshold:
            left -= 1

        # Find right boundary
        right = peak
        while right < len(y) - 1 and y[right] > height_threshold:
            right += 1

        # Ensure minimum width
        if right - left < min_width:
            padding = (min_width - (right - left)) // 2
            left = max(0, left - padding)
            right = min(len(y) - 1, right + padding)

        peak_ranges.append((x[left], x[right]))

    return peak_ranges


def _speed():
    global IM, AIM, canvas, transform, text, speed_distribution, radial_coords
    _transform()
    text.insert(tk.END, "speed distribution\n")
    canvas.draw()

    if transform.get() == 'linbasex':
        radial, speed = AIM.radial, AIM.Beta[0]
    elif transform.get() == 'rbasex':
        radial, speed, _ = AIM.distr.rIbeta()
    else:
        radial, speed = abel.tools.vmi.angular_integration_3D(AIM.transform)

    speed_distribution = speed / speed[50:].max()
    radial_coords = radial

    f.clf()
    a = f.add_subplot(111)
    a.plot(radial, speed_distribution)
    a.set_xlabel("Radial coordinate (pixels)")
    a.set_ylabel("Speed distribution (arb. units)")
    a.axis(xmax=500, ymin=-0.05)

    peak_ranges = find_peak_ranges(radial, speed_distribution, threshold=0.1, rel_height=0.5, min_width=5)

    for i, (left, right) in enumerate(peak_ranges):
        a.axvspan(left, right, alpha=0.2, color=f'C{i}')
        a.annotate(f"Peak {i + 1}", ((left + right) / 2, a.get_ylim()[1]), ha='center')

    canvas.draw()

    text.insert(tk.END, f"Prominent peaks found at ranges: {peak_ranges}\n")


def _quit():
    root.quit()
    root.destroy()


# Tk Block
tk.Button(master=root, text='Load image file', command=_getfilename).pack(anchor=tk.W)
#############################################################################################
frame_for_center1 = tk.Frame(root)
frame_for_center1.pack(anchor=tk.N, expand=True, padx=(230,0) )
tk.Button(master=frame_for_center1, text='center image', command=_center).pack(side=tk.LEFT)

cent = ttk.Combobox(master=frame_for_center1, values=center_methods, state="readonly",
                    width=11, height=len(center_methods))
cent.current(3)
cent.pack(side=tk.LEFT, padx=(110,0))
#############################################################################################
tk.Button(master=root, text='raw image', command=_display).pack(anchor=tk.W)

frame_for_center2 = tk.Frame(root)
frame_for_center2.pack(anchor=tk.N, expand=True, padx=(220,0) )
tk.Button(master=frame_for_center2, text='inverse Abel transform', command=_transform).pack(side=tk.LEFT)

transform = ttk.Combobox(master=frame_for_center2, values=Abel_methods, state="readonly",
                         width=13, height=len(Abel_methods))
transform.current(2)
transform.pack(side=tk.LEFT, padx=(75,0))
#############################################################################################
tk.Button(master=root, text='centered raw image', command=_display_centered).pack(anchor=tk.W)
#############################################################################################
tk.Button(master=root, text='speed distribution', command=_speed).pack(anchor=tk.N)


tk.Button(master=root, text='Quit', command=_quit).pack(anchor=tk.SW)


vmin_frame = tk.Frame(root)
vmin_frame.pack(fill=tk.X)
tk.Label(vmin_frame, text="vmin:").pack(side=tk.LEFT)
vmin_entry = tk.Entry(vmin_frame, width=10)
vmin_entry.pack(side=tk.LEFT)
vmin_slider = tk.Scale(vmin_frame, from_=0, to=255, orient=tk.HORIZONTAL, command=slider_changed)
vmin_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

vmax_frame = tk.Frame(root)
vmax_frame.pack(fill=tk.X)
tk.Label(vmax_frame, text="vmax:").pack(side=tk.LEFT)
vmax_entry = tk.Entry(vmax_frame, width=10)
vmax_entry.pack(side=tk.LEFT)
vmax_slider = tk.Scale(vmax_frame, from_=0, to=255, orient=tk.HORIZONTAL, command=slider_changed)
vmax_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

vmin_entry.bind('<Return>', entry_changed)
vmax_entry.bind('<Return>', entry_changed)


canvas = FigureCanvasTkAgg(f, master=root)
a.annotate("load image file", (0.5, 0.6), horizontalalignment="center")
a.annotate("e.g. data/O2-ANU1024.txt.bz2", (0.5, 0.5), horizontalalignment="center")
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

toolbar = NavigationToolbar2Tk(canvas, root)
toolbar.update()
canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

text = tk.Text(master=root, height=4, fg="blue")
text.pack(fill=tk.X)
text.insert(tk.END, "To start load an image data file using the `Load image file' button\n")

tk.mainloop()