import os
import pandas as pd
from argparse import ArgumentParser
import glob
import re

def merge_fluxes(main_folder):
    # Create a dictionary to store DataFrames for each site
    site_data = {}

    # Iterate through subfolders of the main folder
    for root, dirs, files in os.walk(main_folder):
        for directory in dirs:
            # Check if the subfolder name matches the expected pattern
            if directory.startswith("output_"):
                site = directory.split("_")[1]  # Extract the site from the folder name
                year = directory.split("_")[2]

                # Find subfolder that ends with "site_year_0"
                subfolder_pattern = os.path.join(root, directory, f"*{site}_{year}_0")
                matching_subfolders = glob.glob(subfolder_pattern)

                if matching_subfolders:
                    selected_subfolder = matching_subfolders[0]
                    fluxes_path = os.path.join(selected_subfolder, "fluxes.csv")
                    fluxes_df = pd.read_csv(fluxes_path, parse_dates=['DateTime'])

                    # If the site is not in the dictionary, add it
                    if site not in site_data:
                        site_data[site] = fluxes_df
                    else:
                        # Concatenate the DataFrames if the site already exists in the dictionary
                        site_data[site] = pd.concat([site_data[site], fluxes_df], ignore_index=True)

    # Save the merged DataFrames to CSV files
    for site, df in site_data.items():
        # Sort the DataFrame by the DateTime column
        df = df.sort_values(by='DateTime')
        
        # drop all columns that are called RECO_res_{i}
        for col in df.columns:
            match = re.match(r'NEE_(\d+)', col)
            if match:
                # Extract the integer suffix
                suffix = match.group(1)
                # Construct the new column name
                new_col_name = f'NEE_di_{suffix}'
                # Rename the column
                df.rename(columns={col: new_col_name}, inplace=True)
            
            match = re.match(r'RECO_res_(\d+)', col)
            if match:
                df.drop(columns=[col], inplace=True)


        # Save the merged DataFrame to a CSV file
        output_filename = f"{site}_fluxes.csv"
        output_path = os.path.join(main_folder, output_filename)
        df.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("-folder", type=str, help="folder to be merged")
    args = parser.parse_args()
    
    main_folder = os.path.expanduser(f"~/Repositories/dml-4-fluxes/results/{args.folder}")
    merge_fluxes(main_folder)