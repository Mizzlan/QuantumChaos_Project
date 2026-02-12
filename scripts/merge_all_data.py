import pandas as pd
import glob
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def merge_production_files():
    # 1. Grab all CSVs in the data folder
    path = "data/"
    all_files = glob.glob(os.path.join(path, "*.csv"))
    
    # 2. Concatenate them
    li = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    
    # 3. Save as a "Master" file for the Plotter
    master_path = "data/master_fss_data_L[8,10,12,14]_J2[0.0,0.5]_NE50.csv"
    frame.to_csv(master_path, index=False)
    print(f"Merged {len(all_files)} files into {master_path}")

if __name__ == "__main__":
    merge_production_files()