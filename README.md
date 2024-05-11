# Instructions
- run 'install_packages.py' before any other software execution
- run main.py
- small tkinter window open to prompt data file selection and some analysis options
    - 
- after selecting data file, click submit
- 2x2 interactive plot window opens
- all 4 panels are interactable, and coupled. So a selection in one will update in the others
- selection of data saves statistical calculations to files in 'output' folder

# Surface Force Plotter Description
This software is designed to provide advanced data analysis and visualization tools for Surface Force Apparatus (SFA) experimental data. It facilitates the detailed examination of sensor outputs through interactive plots, allowing users to dynamically select data ranges and analyze underlying patterns such as friction loops and periodic signals. Key features include:
---
1. Interactive Plotting:
Users can interactively select ranges on plots to focus on specific sections of the data.
The software supports both coupled and individual column analysis, adapting the analysis scope based on user configuration.

2. Statistical and Cycle Analysis:
Upon selection in the interactive plots, the software performs statistical analysis, calculating means and standard deviations for selected ranges.
It identifies the number of cycles within a selected range using fast Fourier transform (FFT) analysis enhanced by Gaussian parabolic interpolation for precise frequency determination.

3. Friction Loop Visualization:
For studies involving tribological (frictional) measurements, the software can plot friction loops, highlighting the relationship between drive and friction voltages across specified cycle ranges.
These plots help in understanding the material behaviors under repeated loading and unloading cycles, crucial for tribological assessments.

---

# How to Use

## Preparation
- First run 'install_packages.py'
    - if you get an error with this script in Spyder IDE,
    - in a command prompt, (not anaconda terminal) type 'where python' on windows, or in a mac terminal type 'which python'
    - copy and paste the full path that it prints out
        - on windows it should look something like: 'C:\<some path stuff>\Python\Python310\python.exe'
        - in spider, go to tools > preferences > python interpreter
        - select 'Use the following Python interpreter:'
        - paste in the path you copied earlier from the terminal
        - click apply and ok, and restart spyder for changes to take effect

## Software Execution
- Run 'main.py' upon completion of above
- Click button to select video file
- Click Submit
- Interactive plot opens where selections can be made

### Interactive Plot Details
- Selections can be made in any window
- Upon making a selection, cycles appear and are labelled and counted in the plot, as well as the plot zooms in to fill the range selected
    - Hit ESC at any time to reset the zoom on the axes
    - If box is checked to plot fourier transform, that will also happen on select
- The average and std dev of the y values within the selected x range are record and saved in the 'output' folder


### UI Notes
- Checkboxes in main UI can be altered with interactive plot still open with no need to close and reopen it
- User has option to have selections in each subplot be separate or all coupled together
- If plot Fourier Transfrom box is checked, the Fast Fourier transform (FFT) of the selection will be plotted and saved in the figures folder
    - This is already being found to find the number of cycles, but the box just saves a plot of it

### Output Data / Figures Notes
- For FFT:
    - Only first 2 sensors will have a Fourier Transform done
    - This is due to the fact that at least currently, The first 2 sensors are the only ones with cyclical data
    - Plots are saved as 'sensor{1 or 2}_selected_signal_fft.png'
        - where sensor number is determined on where selection is made
- For friction loop
    - Once user has identified the cycles in the interactive plot they would like to see the friction loop for, they simply need to enter the first and last cycle numbers into the appropriate UI text boxes and click the button to plot
    - Plots saved as 'friction_loop_cycles{c0}-{cf}.png'
        - c0 being the first cycle inputted and cf the final