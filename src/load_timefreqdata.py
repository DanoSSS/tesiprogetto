import pandas as pd
import matplotlib.pyplot

timevalue_disps = []
fftvalue_disps = []
dir = "D:/ExportFiles/"

for v in range(7):
    for i in range(7):
        for j in range(6):
            for k in range(4):
                inputfile_time = f"{dir}vertdisptime_{v}_{i}_{j}_{k}.txt"
                df = pd.read_csv(inputfile_time, delimiter='\s+', header=None, names=['vbase','b_length','b_height','air_gap','time','vertdisp7','vertdisp9'])
                timevalue_disps.append(df.iloc[:,[5,6,7]])
                inputfile_ff = f"{dir}fftabsdisp_{v}_{i}_{j}_{k}.txt"
                df = pd.read_csv(inputfile_ff, delimiter="\s+", header=None, names=['vbase','b_length','b_height','air_gap','freq','absdisp'])
                fftvalue_disps.append(df.iloc[:,[5,6]])
