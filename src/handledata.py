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
            # Use re.sub to replace one or more spaces with a single space
            # but specifically target spaces, not all whitespace (\s)
            # You might want to use '\t+' for tabs or '[ ]+' for only literal spaces
            # r' +' specifically targets one or more literal spaces.
            # If your "spaces" could also be tabs that you want to condense, use r'[ \t]+'
            # If you want to condense ALL whitespace characters except newlines,
            # you'd need a more complex approach or process after splitting/stripping.
            
            # For your specific case (multiple spaces between numbers),
            # r' +' is the most direct and generally safe approach.
            
            # First, strip leading/trailing whitespace if desired (often useful)
            stripped_line = line.strip() 
            
            # Then, replace internal multiple spaces with a single space
            # This handles cases like "  1   2.0   3  " -> "1 2.0 3"
            processed_line = re.sub(r' +', ' ', stripped_line)
            
            # If the original line had leading/trailing spaces, .strip() removes them.
            # If you want to preserve a single leading/trailing space if it existed,
            # but still condense internal spaces, this approach needs adjustment.
            # Given your example "1           3.25E-5", stripping is usually desired.
            
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
standardize_spaces_keep_newlines(tablefile)
df = pd.read_csv(tablefile, delimiter=' ', header=None, names=['voltage_type','b_length','b_height','air_gap','time','v'])
print(df.head())
#plt.figure(figsize=(12,6))
#plt.plot(df['time']['v'])
