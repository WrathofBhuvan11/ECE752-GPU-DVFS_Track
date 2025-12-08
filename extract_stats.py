import re
import csv
import sys

def extract_stats(input_file, output_file):
    # Regex to identify the relevant lines
    line_pattern = re.compile(r"GPU_DVFS_STATS:\s+(.*)")
    
    # Regex to find key-value pairs
    kv_pattern = re.compile(r"(\w+):\s*([-\d\.e\+nan]+)")

    data_rows = []
    headers = set()

    print(f"Processing {input_file}...")

    try:
        with open(input_file, 'r') as f_in:
            for line in f_in:
                match = line_pattern.search(line)
                if match:
                    content_str = match.group(1)
                    pairs = kv_pattern.findall(content_str)
                    
                    if pairs:
                        row_dict = {}
                        for key, value in pairs:
                            row_dict[key] = value
                            headers.add(key)
                        data_rows.append(row_dict)

        if not data_rows:
            print("Warning: No 'GPU_DVFS_STATS' lines found.")
            return

        # --- NEW: SORTING LOGIC ---
        # Sort by 'CU' column numerically.
        # We use a helper lambda to convert the CU string to an int.
        # If 'CU' is missing for some reason, it defaults to -1.
        data_rows.sort(key=lambda x: int(x.get('CU', -1)))
        # --------------------------

        # Organize headers (CU first, then others)
        sorted_headers = sorted(list(headers))
        priority_cols = ['CU', 'perfLevel', 'clock', 'Cycles', 'IPC']
        for col in reversed(priority_cols):
            if col in sorted_headers:
                sorted_headers.insert(0, sorted_headers.pop(sorted_headers.index(col)))

        # Write to CSV
        with open(output_file, 'w', newline='') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=sorted_headers)
            writer.writeheader()
            writer.writerows(data_rows)
            
        print(f"Successfully extracted and sorted {len(data_rows)} rows to {output_file}")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 extract_stats_sorted.py <input_txt_file> <output_csv_file>")
    else:
        extract_stats(sys.argv[1], sys.argv[2])