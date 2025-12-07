import argparse
import pandas as pd
import sys
import os

def parse_gem5_stats(file_path, output_csv_path):
    data = []
    
    if not os.path.exists(file_path):
        print(f"Error: Input file '{file_path}' not found.")
        sys.exit(1)
        
    print(f"Processing file: {file_path}...")
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                # Look for the specific marker in the log lines
                if "GPU_DVFS_STATS:" in line:
                    try:
                        # Extract the part after the marker
                        # Example line: ... info: GPU_DVFS_STATS: CU: 39, clock: ...
                        content = line.split("GPU_DVFS_STATS:")[1].strip()
                        
                        # Split by comma to get fields
                        parts = content.split(', ')
                        
                        row = {}
                        for part in parts:
                            if ':' in part:
                                key, val = part.split(': ')
                                row[key.strip()] = val.strip()
                        
                        if row:
                            data.append(row)
                    except IndexError:
                        continue # Skip malformed lines
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        sys.exit(1)

    if not data:
        print("No 'GPU_DVFS_STATS' lines found in the file.")
        return

    df = pd.DataFrame(data)
    
    # Convert columns to numeric, automatically handling '-nan'
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Sort by CU and then by clock so consecutive dumps appear vertically for each CU
    if 'CU' in df.columns and 'clock' in df.columns:
        df = df.sort_values(by=['CU', 'clock'])
    
    # Save to CSV
    try:
        df.to_csv(output_csv_path, index=False)
        print(f"Successfully saved parsed stats to: {output_csv_path}")
    except Exception as e:
        print(f"Error writing to output file: {e}")

if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(
        description="Extract GPU DVFS stats from a gem5 log file into a CSV."
    )
    
    # Add arguments for input and output files
    parser.add_argument("input_file", help="Path to the gem5 stats text file (input)")
    parser.add_argument("output_file", help="Path where the CSV file will be saved (output)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the function
    parse_gem5_stats(args.input_file, args.output_file)
