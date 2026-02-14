import sys
import os


# Adds the parent directory (project root) to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd # Library for data manipulation [cite: 407]
import glob # Library to find files matching a pattern
from model_spinchain00 import NNN, jxy, ising, nrealise
from src.spectrum import frac

def merge_results():
    # Find all CSV files in your results folder that look like L results
    # This assumes you saved your runs as 'results_L10.csv', 'results_L14.csv', etc.
    all_files = glob.glob(f"results/data/results_L*_J2_{NNN}_J{jxy}_Nr_{nrealise}__DELTA_{ising}_frac_{frac}.csv")
    
    # List to store dataframes
    df_list = []
    
    for filename in all_files:
        # Read each individual file
        df = pd.read_csv(filename)
        df_list.append(df)
        print(f"Loaded {filename}")
        
    # Combine all data into one master dataframe
    master_df = pd.concat(df_list, axis=0, ignore_index=True)
    master_df.sort_values(["L", "W"], inplace=True)
    
    # Save the combined data for the plotter to use
    master_df.to_csv(f"results/data/fss_master_data_J2_{NNN}_J{jxy}_Nr_{nrealise}__DELTA_{ising}_frac_{frac}.csv", index=False)
    print("Master file created at results/fss_master_data.csv")

if __name__ == "__main__":
    merge_results()