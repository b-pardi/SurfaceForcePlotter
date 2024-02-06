- run 'install_packages.py' before any other software execution
- run main.py
- small tkinter window open to prompt data file selection and code execution
- after selecting data file, click submit
- 2x2 interactive plot window opens
- all 4 panels are interactable, and coupled. So a selection in one will update in the others
- selection of data saves statistical calculations to files in 'output' folder

# Budget Labview
### a more apt description will go here

# How to Use

### Preparation
- First run 'install_packages.py'
    - if you get an error with this script in Spyder IDE,
    - in a command prompt, (not anaconda terminal) type 'where python' on windows, or in a mac terminal type 'which python'
    - copy and paste the full path that it prints out
        - on windows it should look something like: 'C:\<some path stuff>\Python\Python310\python.exe'
        - in spider, go to tools > preferences > python interpreter
        - select 'Use the following Python interpreter:'
        - paste in the path you copied earlier from the terminal
        - click apply and ok, and restart spyder for changes to take effect

### Software Execution
- Run 'main.py' upon completion of above
- Click button to select video file
- Click Submit
- Interactive plot opens where selections can be made
    - selections can be made in any window, and they are all coupled together
- Outputs are saved to the folder 'output'
    - 2 versions of the same data are there
    - the difference is the file with '_T' at the end is transposed for potentially easier readability