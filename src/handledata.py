import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

def standardize_spaces_keep_newlines(filepath):
    """
    Replaces multiple consecutive spaces (but not newlines) with a single space
    in a given file, preserving the original line structure.

    Args:
        filepath (str): The path to the text file.
    """
    try:
        # Read all lines from the file
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines() # Reads lines into a list, keeping newline characters

        modified_lines = []
        for line in lines:
            stripped_line = line.strip() 
            
            processed_line = re.sub(r' +', ' ', stripped_line)
            
            
            modified_lines.append(processed_line + '\n')

        # Write the modified content back to the file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(modified_lines) # writelines expects a list of strings with newlines

        print(f"Successfully standardized spaces while preserving newlines in '{filepath}'.")

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


tablefile = "D:/ExportFiles/vd_0_0_0_1.txt"
#standardize_spaces_keep_newlines(tablefile)
df = pd.read_csv(tablefile, delimiter=' ', header=None, names=['voltage_type','b_length','b_height','air_gap','time','v'])
series = pd.Series(df["v"].values, index=df["time"])
series.describe()
#plt.figure(figsize=(12,6))
#plt.plot(series)
