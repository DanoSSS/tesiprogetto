import pandas as pd
import matplotlib.pyplot as plt


eigenvalue_freqs = []
pullin_voltages = []

dir = "D:/ExportFiles/"
for i in range(7):
    for j in range(6):
        for k in range(4):
            inputfile_eig = f"{dir}eigenvaluefreq_{i}_{j}_{k}.txt"
            df = pd.read_csv(inputfile_eig, delimiter = '\s+', header = None, names = ['b_length','b_height','air_gap','freq','realfreq'])
            eigenvalue_freqs.append(df.iloc[0,[0,1,2,4]].to_dict())
            inputfile_pul = f"{dir}pullinvoltage_{i}_{j}_{k}.txt"
            df = pd.read_csv(inputfile_pul, delimiter = '\s+', header = None, names = ['b_length','b_height','air_gap','vrel','pullinvoltage'])
            pullin_voltages.append(df.iloc[0,[0,1,2,4]].to_dict())

