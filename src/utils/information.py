import pandas as pd
import os
import json
 
from datetime import datetime


def update_report (report,experiment_id,reports_dir):
    """
     Updates the experiment report by appending new data to both a JSONL file and an Excel report.

    This function performs the following actions:
    1. Appends the `report` dictionary as a new line in a JSONL file.
    2. Updates an Excel report by appending the new `report` data to the existing sheet or creating
       a new sheet if the file doesn't exist.

    Parameters:
    ----------
    report : dict
        Report data to log.
    experiment_id : str
        Unique ID for the experiment, used to name the JSONL file.
    reports_dir : str
        Directory where report files are stored.
    """
    filename_jsonl = os.path.join(reports_dir, f"{experiment_id}_report.jsonl")
    with open(filename_jsonl, 'a') as jsonl_file:
        json.dump(report, jsonl_file)
        jsonl_file.write('\n')

    report_df = pd.DataFrame([report])
    filename_report = os.path.join(reports_dir, "report.xlsx")

    try:
        existing_report = pd.read_excel(filename_report, sheet_name='Report')
        updated_report = pd.concat([existing_report, report_df], ignore_index=True)
    except FileNotFoundError:
        updated_report = report_df


    updated_report.to_excel(filename_report, index=False, sheet_name='Report')