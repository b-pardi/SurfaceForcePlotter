## WIP

- zoom in selections of plot
- ability to cancel selections (zoom back out) by hitting escape
- friction loop plot
    - friction voltage (sensor 2) vs position voltage (sensor 1)
    - choose range of cycles to include in friction loop plot
- entries for scalar values to change units


# Changelog

4/10
- added function to find the number of cycles in a given selection
    - finds the fft of the signal in the selection
    - multiplies the dominant frequency to the duration of the selection for n_cycles
- added a legend that shows the number of cycles in the selection
- modified cycle counter to also grab the start and end time points of each cycle
- added cycle labels to selection
- added dashed horizontal lines to indicate cycle ranges

4/4
- added ability to couple/decouple selections
    - checkbox in ui sets a global var IS_COUPLED
    - can be changed during file analysis without needing to close and resubmit
    - adjusted onselect in int plot to make onselect functions for widgets which allows the script to read which box was just selected
    - when coupled data analysis and plot updating is business as usual
    - when decoupled only one span selector changes, all others are left unchanged
        - the data from the changed selector is what is analyzed
    - onselect passes a dict of every columns xmin and xmax from the span selectors
        - this allows the stats from other columns to be kept while analyzing the selected columns
    - when uncoupled, each columns xmin/xmax are shown
    - when coupled one xmin/xmax is shown that applies to all columns
- added a clear output file button for use when changing data files
    - this is due to the nature of the coupling feature reading the previous output data to update only one
- added an exit button

2/6
- updated readme to reflect solution to error with spyder

1/16 (start of project)
- made tkinter ui to prompt for file selection and submit button
- gross regex code to read raw .lvm files into csv
- created interactive plot for viewing 4 panels of data from file all over time
- interactive plot formatting
- added span selectors to int plot for all 4 panels
- coupled selectors together so changing one updates the others
- statistics calculations of selected ata
- save lower/upper bounds of x axis (time) selection, and stat calculations above