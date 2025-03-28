import csv
import os

def append_csv(source_file, target_file):
    """
    Appends the contents of source_file CSV to target_file CSV.
    
    Args:
        source_file (str): Path to the source CSV file
        target_file (str): Path to the target CSV file
    """
    try:
        # Check if files exist
        if not os.path.exists(source_file):
            raise FileNotFoundError(f"Source file {source_file} not found")
        
        # Read source file
        with open(source_file, 'r', newline='', encoding='utf-8') as src:
            reader = csv.reader(src)
            source_rows = list(reader)
        
        # Check if target file exists
        target_exists = os.path.exists(target_file)
        
        # Write to target file
        with open(target_file, 'a', newline='', encoding='utf-8') as tgt:
            writer = csv.writer(tgt)
            
            # If target doesn't exist, write header from source
            if not target_exists and source_rows:
                writer.writerow(source_rows[0])
                start_row = 1
            else:
                start_row = 0
            
            # Write remaining rows
            for row in source_rows[start_row:]:
                writer.writerow(row)
        
        print(f"Successfully appended {len(source_rows)-start_row} rows from {source_file} to {target_file}")
    
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    # Get file paths from user
    source_path = input("Enter path to source CSV file: ").strip()
    target_path = input("Enter path to target CSV file: ").strip()
    
    append_csv(source_path, target_path)