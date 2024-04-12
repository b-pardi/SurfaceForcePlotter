"""
Author: Brandon Pardi
Created: 1/16/2024
"""

import tkinter as tk
from tkinter import filedialog
from io import StringIO
import sys
import os
import re
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector

'''To Do
- setup span selector
- couple 4 subplots of span selector together
'''

# global vars
FILE = ""
COLUMN_HEADERS = ("X_Value","Voltage_0","Voltage_1","Voltage_2","Voltage_3")
IS_COUPLED = False
WILL_PLOT_FFT = False


def window():
    root = tk.Tk()
    
    # file browse button
    data_label_var = tk.StringVar()
    file_btn = tk.Button(root, text="Browse for data file", command=lambda: get_file(data_label_var), width=15)
    file_btn.pack(padx=32,pady=24)

    # file name label
    data_label_var.set("File not selected")
    data_label = tk.Label(root, textvariable=data_label_var)
    data_label.pack(pady=(0,8))

    # opt to couple/decouple
    is_coupled_var = tk.IntVar()
    is_coupled_check = tk.Checkbutton(root, text="Couple subplot selections", variable=is_coupled_var, offvalue=0, onvalue=1, command=lambda: set_couple_var(is_coupled_var))
    is_coupled_check.pack()

    # fourier transform plotting
    plot_fft_var = tk.IntVar()
    plot_fft_check = tk.Checkbutton(root, text="Plot Fourier transform of selection", variable=plot_fft_var, onvalue=1, offvalue=0, command=lambda: set_plot_fft_var(plot_fft_var))
    plot_fft_check.pack()

    # friction loop plotting
    friction_loop_frame = tk.Frame()
    friction_loop_label = tk.Label(friction_loop_frame, text="Friction loop plotting")
    friction_loop_label.grid(row=0, column=0, columnspan=4, pady=12)
    friction_loop_cycle_start_label = tk.Label(friction_loop_frame, text="First cycle: ")
    friction_loop_cycle_start_label.grid(row=1, column=0)
    friction_loop_cycle_start_entry = tk.Entry(friction_loop_frame, width=5)
    friction_loop_cycle_start_entry.grid(row=1, column=1, padx=(0,4))
    friction_loop_cycle_end_label = tk.Label(friction_loop_frame, text="Last cycle: ")
    friction_loop_cycle_end_label.grid(row=1, column=2, padx=(4,0))
    friction_loop_cycle_end_entry = tk.Entry(friction_loop_frame, width=5)
    friction_loop_cycle_end_entry.grid(row=1, column=3)
    plot_friction_loop_btn = tk.Button(friction_loop_frame, text="Plot friction loop", width=15, command=lambda: plot_friction_loop((int(friction_loop_cycle_start_entry.get()), int(friction_loop_cycle_end_entry.get()))))
    plot_friction_loop_btn.grid(row=2, column=0, columnspan=4, pady=8)
    friction_loop_frame.pack()

    # unit conversion

    # submit button
    submit_btn = tk.Button(root, text="Submit", command=main, width=15)
    submit_btn.pack(padx=32, pady=(32,12))

    # clear output file for when switching datafiles
    clear_btn = tk.Button(root, text="Clear output file", command=clear_output_file, width=15)
    clear_btn.pack(padx=32, pady=(0,12))

    # exit button
    exit_btn = tk.Button(root, text="Exit", command=sys.exit, width=15)
    exit_btn.pack(padx=32, pady=(0,12))
    
    root.mainloop()

def set_couple_var(is_coupled_var):
    global IS_COUPLED
    IS_COUPLED = True if is_coupled_var.get() == 1 else False

def set_plot_fft_var(plot_fft_var):
    global WILL_PLOT_FFT
    WILL_PLOT_FFT = True if plot_fft_var.get() == 1 else False


def clear_output_file():
    df = pd.read_csv("output/sensor_stats.csv")
    df['xmin'] = np.nan
    df['xmax'] = np.nan

    for col in COLUMN_HEADERS:
        if col != 'X_Value':
            df[f"{col}_mean"] = np.nan
            df[f"{col}_stddev"] = np.nan

    dfT = df.T # transpose for readability
    dfT.to_csv("output/sensor_stats_T.csv", float_format="%.8E", index=True)
    df.to_csv("output/sensor_stats.csv", float_format="%.8E", index=False)

def get_file(label_var):
    fp = filedialog.askopenfilename(initialdir=os.path.join(os.getcwd(), 'data'),
                                    title='Browse for data file',
                                    filetypes=[("Labview file", "*.lvm")])
    
    if fp:
        label_var.set(os.path.basename(fp))

    global FILE
    FILE = fp


def read_lvm(fp):
    with open(fp, 'rb') as file:
        raw_file = file.read()

    raw_file = raw_file.replace(b'\r\n', b'\n') # replace CRLF with newline

    # use regex to find start of data file
    data_begin_pattern = rb'\*\*\*End_of_Header\*\*\*.*?(\n.*?)(?=\*\*\*End_of_Header|\Z)'
    matches = re.finditer(data_begin_pattern, raw_file, re.DOTALL)
    next(matches) # ***END OF HEADER*** occurs twice before data happens
    target = next(matches).group(1).decode('utf-8')

    df = pd.read_csv(StringIO(target)) # make dataframe of cleaned values
    print(df.head())
    df.to_csv("output/parsed_labview.csv", index=False)
    
    return df


# plot formatting
def generate_int_plot():
    # declare plot and subplots
    span_plot = plt.figure()
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    span_plot.set_figwidth(16)
    span_plot.set_figheight(10)
    ax = span_plot.add_subplot(1,1,1) # the 'big' subplot for shared axis
    s1_ax = span_plot.add_subplot(2,2,1) # sensor 1 data
    s2_ax = span_plot.add_subplot(2,2,2) # sensor 2 data
    s3_ax = span_plot.add_subplot(2,2,3) # sensor 3 data
    s4_ax = span_plot.add_subplot(2,2,4) # sensor 4 data

    # formatting and labels
    ax.set_title("Click and drag to select range", fontsize=20, fontfamily='Arial', weight='bold', pad=40)
    s1_ax.set_title("\nSensor 1", fontsize=18, fontfamily='Arial')
    s2_ax.set_title("\nSensor 2", fontsize=18, fontfamily='Arial')
    s3_ax.set_title("\nSensor 3", fontsize=18, fontfamily='Arial')
    s4_ax.set_title("\nSensor 4", fontsize=18, fontfamily='Arial')
    ax.set_xlabel("Time, (s)", fontsize=24, fontfamily='Arial', labelpad=20)
    ax.set_ylabel("Voltage, (V)", fontsize=24, fontfamily='Arial', labelpad=20)

    plt.sca(s1_ax)
    plt.xticks(fontsize=12, fontfamily='Arial')
    plt.yticks(fontsize=12, fontfamily='Arial')
    plt.sca(s2_ax)
    plt.xticks(fontsize=12, fontfamily='Arial')
    plt.yticks(fontsize=12, fontfamily='Arial')
    plt.sca(s3_ax)
    plt.xticks(fontsize=12, fontfamily='Arial')
    plt.yticks(fontsize=12, fontfamily='Arial')
    plt.sca(s4_ax)
    plt.xticks(fontsize=12, fontfamily='Arial')
    plt.yticks(fontsize=12, fontfamily='Arial')

    # Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

    return span_plot, ax, s1_ax, s2_ax, s3_ax, s4_ax


def get_sr(df):
    avg_sample_interval = np.mean(np.diff(df[COLUMN_HEADERS[0]]))
    return 1 / avg_sample_interval

def find_closest(df, col, target):
    idx = np.abs(df[col] - target).idxmin()
    return idx

def plot_friction_loop(cycle_range):
    global FILE
    df = read_lvm(FILE)
    xdata = df[COLUMN_HEADERS[0]]

    cycle_start, cycle_end = cycle_range
    n_cycles, n_cycles_prior, cycle_points = find_num_cycles(df, COLUMN_HEADERS[1])
    cycle_points = cycle_points[cycle_start-1:cycle_end]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for i, (t0,tf) in enumerate(cycle_points):
        idx0, idxf = find_closest(df, COLUMN_HEADERS[0], t0), find_closest(df, COLUMN_HEADERS[0], tf)
        df_seg = df.iloc[idx0:idxf]
        ax.plot(df_seg[COLUMN_HEADERS[1]], df_seg[COLUMN_HEADERS[2]], '.', markersize=2, color=f'C{i}', label=f"cycle: {i+cycle_start}")

    if n_cycles <= 3:
        legend = ax.legend(loc='best', framealpha=0.3)
    else: # put legend outside plot if more than 3 datasets for readability
        legend = ax.legend(loc='upper left', bbox_to_anchor=(1, 1), framealpha=0.3)

    plt.xlabel("Drive voltage (V)")
    plt.ylabel("Friction voltage (V)")
    plt.title(f"Friction Loop for cycles: {cycle_start} - {cycle_end}")
    plt.grid(True)
    fig.tight_layout()
    fig.savefig(f"figures/friction_loop_cycles{cycle_start}-{cycle_end}.png", bbox_extra_artists=(legend,), dpi=400)
    plt.close(fig)

    print(n_cycles, n_cycles_prior, cycle_points)



def plot_fft(freqs, mag, dom_freq, sensor_num):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(freqs, mag, color='red', label=f"Dominant frequency sensor {sensor_num}: {dom_freq:.4f} Hz")
    ax.legend()
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude normalized")
    plt.title("Fourier Transform of Selected Signal Data")
    plt.grid(True)
    fig.tight_layout()
    fig.savefig(f"figures/sensor{sensor_num}_selected_signal_fft.png")
    plt.close(fig)

def find_xrange_from_cycle_range():
    pass

def gaussian_parabolic_interpolation(freqs, magnitudes, est_max_freq_index):
    """
    Perform Gaussian interpolation to estimate a more accurate peak frequency.

    This method assumes the peak of the spectrum resembles a Gaussian shape.
    It applies a logarithmic transformation to turn the Gaussian peak into a parabolic shape,
    which can then be fitted using a parabolic interpolation formula.

    The Gaussian interpolation uses the natural logarithm of three points around the peak magnitude
    to fit a parabola and then estimates the vertex of this parabola to find a refined peak frequency.
    
    Args:
        freqs (numpy.ndarray): Array of frequency bins.
        magnitudes (numpy.ndarray): Array of magnitude values from FFT.
        index (int): Index of the detected peak in magnitudes.

    Returns:
        float: Interpolated frequency at the peak.
    """
    if 0 < est_max_freq_index < len(magnitudes) - 1:
        # logarithmic transformation of magnitudes at index and its neighbors
        y1 = np.log(magnitudes[est_max_freq_index - 1])
        y2 = np.log(magnitudes[est_max_freq_index])
        y3 = np.log(magnitudes[est_max_freq_index + 1])

        # Parabolic interpolation to find the precise peak location
        # p calculates the offset from the central frequency based on the vertex of the parabola
        p = 0.5 * (y1 - y3) / (y1 - 2 * y2 + y3)

        # Calculate the actual frequency by adjusting from the central frequency
        # This step adjusts the frequency at the maximum magnitude index by the interpolated offset
        return freqs[est_max_freq_index] + p * (freqs[1] - freqs[0])
    else:
        # Return the frequency directly if the peak is at the boundaries
        return freqs[est_max_freq_index]
    
def find_num_cycles(df, column_name, col_xranges=None):
    """
    Analyze the signal to find the number of cycles within specified range using FFT.

    This function applies Gaussian interpolation to the FFT results to obtain a more accurate
    estimation of the dominant frequency, especially useful when dealing with sharp spectral peaks.

    Args:
        df (DataFrame): The dataframe containing signal data.
        column_name (str): The name of the column containing signal data.
        col_xranges (dict, optional): A dictionary specifying the min and max x-ranges for analysis.

    Returns:
        tuple: A tuple containing the number of cycles, cycles prior to the range, and cycle points.
    """
    if col_xranges is not None:
        xmin, xmax = col_xranges[column_name]
    else:  # if col_xrange not specified, use whole range
        xmin, xmax = 0, df[COLUMN_HEADERS[0]].iloc[-1]
    xmin = max(0, xmin)
    xmax = min(df[COLUMN_HEADERS[0]].iloc[-1], xmax)
    range_df = df[(df[COLUMN_HEADERS[0]] >= xmin) & (df[COLUMN_HEADERS[0]] <= xmax)]

    signal_fft = np.fft.fft(range_df[column_name])
    N = len(range_df[column_name])
    T = np.mean(np.diff(df[COLUMN_HEADERS[0]]))

    freqs = np.fft.fftfreq(N, T)[:N // 2]
    magnitude = np.abs(signal_fft)[:N // 2]
    magnitude_norm = magnitude * (1 / N)
    max_magnitude_idx = np.argmax(magnitude_norm)

    # Interpolate to find a more accurate peak frequency
    dom_freq = gaussian_parabolic_interpolation(freqs, magnitude_norm, max_magnitude_idx)

    selection_duration = xmax - xmin
    n_cycles = dom_freq * selection_duration
    n_cycles_prior = dom_freq * xmin
    period = 1 / dom_freq
    cycle_points = []
    current_x = xmin
    while current_x + 0.5 * period <= xmax:
        cycle_points.append((current_x, current_x + period))
        current_x += period

    print(f"***Selection contains approximately {n_cycles} cycles between {xmin}s and {xmax}s")

    global WILL_PLOT_FFT
    if WILL_PLOT_FFT:
        sensor_num = int(column_name[-1]) + 1
        plot_fft(freqs, magnitude_norm, dom_freq, sensor_num)

    return n_cycles, n_cycles_prior, cycle_points


def statistically_analyze_selected_range(df, imin, imax, cur_col, col_xranges):
    global IS_COUPLED
    
    if IS_COUPLED: # analyze all columns if couple
        range_df = df.iloc[imin:imax+1].copy() # slice of df containing selected data
        stats_dict = {
            "xmin": range_df[COLUMN_HEADERS[0]].values[0],
            "xmax": range_df[COLUMN_HEADERS[0]].values[-1]
        }
        
        # create new dataframe of statistical data analyzed from selected range
        for col in COLUMN_HEADERS:
            if col != 'X_Value':
                stats_dict[f"{col}_mean"] = np.average(range_df[col].values)
                stats_dict[f"{col}_stddev"] = np.median(range_df[col].values)
        
        stats_df = pd.DataFrame(stats_dict, index=[0])

    else: # if not coupled set each col's data to its respective col_xrange
        stats_dict = {}
        for col in COLUMN_HEADERS:
            if col != 'X_Value':
                xmin, xmax = col_xranges[col]
                if xmin is None:
                    stats_dict[f"{col}_xmin"] = np.nan
                    stats_dict[f"{col}_xmax"] = np.nan
                    stats_dict[f"{col}_mean"] = np.nan
                    stats_dict[f"{col}_stddev"] = np.nan
                else:
                    cur_col_df = df[(df[COLUMN_HEADERS[0]] >= xmin) & (df[COLUMN_HEADERS[0]] <= xmax)]
                    print(cur_col_df)
                    stats_dict[f"{col}_xmin"] = cur_col_df[COLUMN_HEADERS[0]].values[0]
                    stats_dict[f"{col}_xmax"] = cur_col_df[COLUMN_HEADERS[0]].values[-1]
                    stats_dict[f"{col}_mean"] = np.average(cur_col_df[col].values)
                    stats_dict[f"{col}_stddev"] = np.median(cur_col_df[col].values)

        stats_df = pd.DataFrame(stats_dict, index=[0])  

    stats_df_T = stats_df.T # transpose for readability
    stats_df_T.to_csv("output/sensor_stats_T.csv", float_format="%.8E", index=True)
    stats_df.to_csv("output/sensor_stats.csv", float_format="%.8E", index=False)
    
    print("STATS SAVED TO 'output/sensor_stats.csv'")


def reset_ax(ax, df):
    init_xmin = df[COLUMN_HEADERS[0]].min()
    init_xmax = df[COLUMN_HEADERS[0]].max()
    ax.set_xlim(init_xmin, init_xmax)
    # Remove all annotations, legends, and vertical lines
    ax.legend_ = None
    [t.remove() for t in reversed(ax.texts)]
    [l.remove() for l in reversed(ax.lines) if l.get_linestyle() == '--']
    # Redraw the canvas
    ax.figure.canvas.draw()


def on_key(event, sensor_axes, df, spans, col_xranges):
    global IS_COUPLED
    xmin = df[COLUMN_HEADERS[0]].min()
    xmax = df[COLUMN_HEADERS[0]].max()

    if event.key == 'escape':
        for ax in sensor_axes:
            reset_ax(ax, df)
        if IS_COUPLED:
            for span in spans:
                span.extents = (xmin, xmax)
        for key in col_xranges.keys():
            col_xranges[key] = (None, None)


def int_plot(df):
    global IS_COUPLED
    span_plot, ax, s1_ax, s2_ax, s3_ax, s4_ax = generate_int_plot()
    sensor_axes = [s1_ax, s2_ax, s3_ax, s4_ax]
    spans = []
    col_xranges = {column: (None, None) for column in COLUMN_HEADERS[1:]} # x ranges of all span selectors
    x_data = df[COLUMN_HEADERS[0]]

    # initial plotting of data
    for i, ax in enumerate(sensor_axes):
        ax.plot(df[COLUMN_HEADERS[0]], df[COLUMN_HEADERS[i+1]], '.', markersize=1)

    def make_onselect(span_selector, column_name):
        def onselect(xmin, xmax):
            imin, imax = np.searchsorted(x_data, (xmin, xmax))
            imax = min(len(x_data) - 1, imax)
            col_xranges[column_name] = (xmin, xmax)
            if IS_COUPLED:
                for col in col_xranges.keys():
                    col_xranges[col] = (xmin, xmax)
                    ax = sensor_axes[COLUMN_HEADERS.index(col) - 1]  # Get the correct axis based on column_name
                    ax.set_xlim(xmin, xmax)
                for span in spans:
                    if span != span_selector:
                        span.extents = (xmin, xmax)
            else:
                ax = sensor_axes[COLUMN_HEADERS.index(column_name) - 1]  # Get the correct axis based on column_name
                ax.set_xlim(xmin, xmax)

            # selection analysis
            statistically_analyze_selected_range(df, imin, imax, column_name, col_xranges)
            
            for col in list(col_xranges.keys())[:2]: # only first 2 columns have periodicity
                if col_xranges[col][0]:
                    n_cycles, n_cycles_prior, cycle_points = find_num_cycles(df, col, col_xranges)
                    # update legend with the number of cycles
                    ax = sensor_axes[COLUMN_HEADERS.index(col) - 1]  # Get the correct axis based on column_name
                    ax.legend([f"n cycles: {n_cycles:.2f}"], loc='best')  # Add legend to the subplot
            
                    # labelling cycles
                    [t.remove() for t in ax.texts] # remove old cycle texts
                    [l.remove() for l in ax.lines if l.get_linestyle() == '--'] # remove old lines
                    for i, (start_x, end_x) in enumerate(cycle_points):
                        start_idx = (np.abs(x_data - start_x)).argmin()
                        end_idx = (np.abs(x_data - end_x)).argmin()
                        mid_x = (start_x + end_x) / 2
                        mid_y = df[column_name][(start_idx + end_idx) // 2]
                        ax.annotate(f'Cycle {i+int(np.round(n_cycles_prior))+1}', xy=(mid_x, mid_y), textcoords='offset points', xytext=(0,10), ha='center')
                        
                        # Draw vertical dashed lines at the start of cycles
                        ax.axvline(x=x_data[start_idx], color='grey', linestyle='--', linewidth=1)
                        ax.axvline(x=x_data[end_idx], color='grey', linestyle='--', linewidth=1)
            
            plt.draw()
        return onselect

    # Create span selector objects for all 4 subplot axes
    for i, ax in enumerate(sensor_axes):
        column_name = COLUMN_HEADERS[i+1]
        selector = SpanSelector(
            ax,
            make_onselect(None, column_name),  # Pass the column name here
            'horizontal',
            useblit=True,
            interactive=True,
            props=dict(alpha=0.1, facecolor='gray')
        )
        spans.append(selector)
        # Update the onselect function with the current selector and column name
        selector.onselect = make_onselect(selector, column_name)

    # bind esc key press event to the on_key function with additional arguments
    span_plot.canvas.mpl_connect(
        'key_press_event', 
        lambda event: on_key(event, sensor_axes, df, spans, col_xranges)
    )

    plt.show()


def main():
    global FILE
    df = read_lvm(FILE)

    int_plot(df)


if __name__ == '__main__':
    window()

