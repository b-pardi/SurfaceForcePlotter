"""
Author: Brandon Pardi
Created: 1/16/2024
"""

import tkinter as tk
from tkinter import filedialog
from io import StringIO
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

FILE = ""

COLUMN_HEADERS = ("X_Value","Voltage_0","Voltage_1","Voltage_2","Voltage_3")

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

    # submit button
    submit_btn = tk.Button(root, text="Submit", command=main)
    submit_btn.pack(padx=32, pady=12)
    
    root.mainloop()


def get_file(label_var):
    fp = filedialog.askopenfilename(initialdir=os.path.join(os.getcwd(), 'data'),
                                    title='Browse for data file',
                                    filetypes=[("Labview file", "*.lvm")])
    
    if fp:
        label_var.set(os.path.basename(fp))

    global FILE
    FILE = fp


# plot formatting
def format_int_plot():
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


def int_plot(df):
    span_plot, ax, time_ax, s1_ax, s2_ax, s3_ax = format_int_plot()

    


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


def main():
    global FILE
    df = read_lvm(FILE)

    int_plot(df)


if __name__ == '__main__':
    window()

