import os
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import nneddyproc.datasets.relevant_variables as relevant_variables
from nneddyproc.datasets.preprocessing import load_data, unwrap_time, standardize_column_names
from math import sqrt
from itertools import product
import numpy as np
from pathlib import Path

def calculate_metrics(true_values, predicted_values):
    r2 = r2_score(true_values, predicted_values)
    rmse = sqrt(mean_squared_error(true_values, predicted_values))
    bias = sum(predicted_values - true_values) / len(true_values)
    return r2, rmse, bias

def process_row(site, start_year, end_year, data_folder, result_folder):
    results = []
    for year in range(start_year, end_year + 1):

        # Load predictions
        predicted_folder = Path(result_folder).joinpath(f"output_{site}_{year}").expanduser()
        matching_directories = list(predicted_folder.glob(f"*{site}_{year}_0"))
        predicted_file = matching_directories[0].joinpath("fluxes.csv")
        predicted = pd.read_csv(predicted_file)
        predicted['DateTime'] = pd.to_datetime(predicted['DateTime'])
        predicted = predicted.set_index('DateTime')
        predicted['DateTime'] = predicted.index
        
        
        # Load ground truths
        data = load_data(site, path=data_folder + f"/FLX_{site}")
        data = unwrap_time(data)
        data['DateTime'] = pd.to_datetime(data['DateTime'])
        data = data.set_index('DateTime')
        data['DateTime'] = data.index
        data = standardize_column_names(data)
        data['NEE_NT'] = -data['GPP_NT'] + data['RECO_NT']
        data['NEE_DT'] = -data['GPP_DT'] + data['RECO_DT']
        data = data[["DateTime", "NIGHT", "Year", "NEE_DT", "NEE_NT", "GPP_DT", "GPP_NT", "RECO_DT", "RECO_NT"]]
        data = data[data['Year'] == year]
        
        # Select columns for computation
        result = dict()
        result['Site'] = site
        result['Year'] = year
        columns = ["NEE", "GPP", "RECO"]
        
        ### filter in different ways
        # Q: compare only on trained data
        # N: compare only on Night data
        # D: compare only on day data
        for column in columns:
            for suffix, quality, partial in product(["_DT", "_NT"], ["","_Q"], ["","_D", "_N"]):
                if partial == "_D":
                    partial_mask = data["NIGHT"] == 0
                elif partial == "_N":
                    partial_mask = data["NIGHT"] == 1
                else:
                    partial_mask = True
                if quality == "_Q":
                    quality_mask = predicted['NEE_QC'] == 0
                else:
                    quality_mask = True

                # isnan mask
                nan_mask = ~np.isnan(predicted[f"{column}_0"]) & ~np.isnan(data[column + suffix])

                true_values = data[column + suffix][partial_mask & quality_mask & nan_mask]
                predicted_values = predicted[f"{column}_0"][partial_mask & quality_mask & nan_mask]

                r2, rmse, bias = calculate_metrics(true_values, predicted_values)
                result["r2_"+column+suffix+quality+partial] = r2
                result["rmse_"+column+suffix+quality+partial] = rmse
                result["bias_"+column+suffix+quality+partial] = bias
        results.append(result)
    return results

def main(input_file, data_folder, result_folder):
    # Read input file with site, start-year, and end-year
    sites_data = pd.read_csv(Path(input_file).expanduser())

    all_results = []
    for index, row in sites_data.iterrows():
        print(f"Processing row {index}")
        site = row[0]
        start_year = row[1]
        end_year = row[2]

        results = process_row(site, start_year, end_year, data_folder, result_folder)
        all_results.extend(results)

    # Create DataFrame from the results
    results_df = pd.DataFrame(all_results)

    # Save results to CSV
    results_df.to_csv(Path(result_folder).joinpath("results.csv"), index=False)
    summary_stats = []
    for metric, column, suffix, quality, partial in product(["r2_", "rmse_", "bias_"], ["NEE", "GPP", "RECO"], ["_DT", "_NT"], ["","_Q"], ["","_D", "_N"]):
        # compute median, 25% and 75% quantile for each of these columns in results_df
        summary_stats.append({"Column": metric+column+suffix+quality+partial,
                            "median": results_df[metric+column+suffix+quality+partial].median(),
                            "25 quantile": results_df[metric+column+suffix+quality+partial].quantile([0.25])[0.25],
                            "75 quantile": results_df[metric+column+suffix+quality+partial].quantile([0.75])[0.75]}
                            )

    # Calculate summary statistics
    summary_stats = pd.DataFrame(summary_stats)
    # Save summary statistics to CSV
    summary_stats.to_csv(Path(result_folder).joinpath("summary_stats.csv"))

if __name__ == "__main__":
    # Example usage:
    input_file = "~/Repositories/dml-4-fluxes/experiments/dry_sites.txt"
    data_folder = "~/Repositories/data/Dry_sites"
    result_folder = "~/Repositories/dml-4-fluxes/results/ngoc_dry_nn_default_single"

    main(input_file, data_folder, result_folder)