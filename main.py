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

def window():
    root = tk.Tk()
    
    # file browse button
    data_label_var = tk.StringVar()
    file_btn = tk.Button(root, text="Browse for data file", command=lambda: get_file(data_label_var))
    file_btn.pack(padx=32,pady=24)

    # file name label
    data_label_var.set("File not selected")
    data_label = tk.Label(root, textvariable=data_label_var)
    data_label.pack(pady=(0,8))

    # opt to couple/decouple
    is_coupled_var = tk.IntVar()
    is_coupled_check = tk.Checkbutton(root, text="Couple subplot selections", variable=is_coupled_var, offvalue=0, onvalue=1, command=lambda: set_couple_var(is_coupled_var))
    is_coupled_check.pack()

    # submit button
    submit_btn = tk.Button(root, text="Submit", command=main)
    submit_btn.pack(padx=32, pady=12)

    # clear output file for when switching datafiles
    clear_btn = tk.Button(root, text="Clear output file", command=clear_output_file)
    clear_btn.pack(padx=32, pady=(0,12))

    # exit button
    exit_btn = tk.Button(root, text="Exit", command=sys.exit)
    exit_btn.pack(padx=32, pady=(0,12))
    
    root.mainloop()

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

def set_couple_var(is_coupled_var):
    global IS_COUPLED
    IS_COUPLED = True if is_coupled_var.get() == 1 else False
    print(IS_COUPLED)

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
            print(col_xranges)
            if IS_COUPLED:
                for col in col_xranges.keys():
                    col_xranges[col] = (xmin, xmax)
                for span in spans:
                    if span != span_selector:
                        span.extents = (xmin, xmax)
            
            statistically_analyze_selected_range(df, imin, imax, column_name, col_xranges)

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
            props=dict(alpha=0.3, facecolor='gray')
        )
        spans.append(selector)
        # Update the onselect function with the current selector and column name
        selector.onselect = make_onselect(selector, column_name)

    plt.show()


def main():
    global FILE
    df = read_lvm(FILE)

    int_plot(df)


if __name__ == '__main__':
    window()

