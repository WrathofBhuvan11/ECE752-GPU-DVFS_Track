import re
import csv
import argparse
import sys
import os

def natural_sort_key(s):
    """
    Splits string into a list of integers and strings to allow 
    for natural sorting (e.g., CUs2 comes before CUs10).
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def parse_files(input_files, output_file):
    # Regex pattern to match the data lines
    pattern = re.compile(r"system\.cpu1\.(CUs\d+\.wavefronts\d+)\.schCycles\s+(\d+)")
    
    # String pattern that marks the start of a new stats dump
    dump_start_marker = "---------- Begin Simulation Statistics ----------"

    # Data Structure: 
    # { 'CUs0.wavefronts0': { ('file1.txt', 1): 55375, ('file1.txt', 2): 60000 } }
    data_map = {}
    
    # Sets to keep track of all unique identifiers and columns found
    all_identifiers = set()
    all_columns = set() # Stores tuples like ('file1.txt', 1)

    print(f"Processing {len(input_files)} file(s)...")

    for filepath in input_files:
        filename = os.path.basename(filepath)
        current_dump_index = 0
        
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    # Check if this line marks the start of a new dump
                    if dump_start_marker in line:
                        current_dump_index += 1
                        continue
                    
                    match = pattern.search(line)
                    if match:
                        identifier = match.group(1)
                        value = match.group(2)
                        
                        # We only track data if we are inside a dump (index > 0)
                        # If your file has data before the first "Begin...", 
                        # it will be stored as Dump 0.
                        col_key = (filename, current_dump_index)

                        if identifier not in data_map:
                            data_map[identifier] = {}
                        
                        # Store the value. If the line repeats within the SAME dump,
                        # it overwrites (keeping the last instance for that specific dump).
                        data_map[identifier][col_key] = value
                        
                        all_identifiers.add(identifier)
                        all_columns.add(col_key)
                        
        except FileNotFoundError:
            print(f"Error: File '{filepath}' not found.")
            continue

    # --- SORTING ---

    # 1. Sort Rows (Identifiers) Naturally
    sorted_identifiers = sorted(list(all_identifiers), key=natural_sort_key)

    # 2. Sort Columns (Files and Dumps)
    # We want columns ordered by the File input order, then by Dump Index.
    # Create a map of filename -> index (0, 1, 2...) based on user input order
    file_order_map = {os.path.basename(f): i for i, f in enumerate(input_files)}
    
    def column_sort_key(key_tuple):
        fname, dump_idx = key_tuple
        # Return tuple: (File Order Index, Dump Number)
        return (file_order_map.get(fname, 999), dump_idx)
    
    sorted_columns = sorted(list(all_columns), key=column_sort_key)

    # --- WRITING CSV ---
    try:
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Generate Header Row
            # Format: Unit, File1 (Dump 1), File1 (Dump 2), File2 (Dump 1)...
            header = ["Unit"]
            for fname, dump_idx in sorted_columns:
                header.append(f"{fname} (Dump {dump_idx})")
            
            writer.writerow(header)

            # Generate Data Rows
            for ident in sorted_identifiers:
                row = [ident]
                for col_key in sorted_columns:
                    # Get value or empty string if this dump didn't have this CU
                    val = data_map.get(ident, {}).get(col_key, "")
                    row.append(val)
                writer.writerow(row)

        print(f"Success! Data extracted to '{output_file}'")
        
    except IOError as e:
        print(f"Error writing to CSV: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract schCycles from simulation logs with multiple dumps.")
    parser.add_argument('files', metavar='F', type=str, nargs='+',
                        help='list of files to process')
    parser.add_argument('-o', '--output', type=str, default='output.csv',
                        help='Output CSV filename (default: output.csv)')

    args = parser.parse_args()

    parse_files(args.files, args.output)