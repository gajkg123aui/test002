import pandas as pd
import csv
import sys

def sort_newspaper_data(input_csv_path, output_csv_path):
    """
    Reads OCR data from a CSV, sorts it according to Japanese newspaper
    reading order (based on a 90-degree rotated image), and writes the
    sorted data to a new CSV file.
    """
    try:
        print("--- Starting sort_newspaper_data ---")
        print(f"Input file: {input_csv_path}")
        print(f"Output file: {output_csv_path}")

        # Read the data using pandas, specifying tab delimiter
        df = pd.read_csv(input_csv_path, sep='\\t', header=None, engine='python', quoting=csv.QUOTE_NONE)
        print("--- CSV file read successfully. DataFrame head: ---")
        print(df.head())

        # Assign column names for clarity
        df.columns = [
            'text', 'line_in_block', 'word_in_line', 'x0', 'y0', 'x1', 'y1',
            'mx', 'my', 'w', 'h', 'confidence', 'rotate', 'direction',
            'line_x0', 'line_y0', 'line_x1', 'line_y1', 'g_x0', 'g_y0', 'g_x1', 'g_y1',
            'gno', 'space_x', 'space_y'
        ]

        # Convert necessary columns to numeric types
        numeric_cols = ['x0', 'y0', 'x1', 'y1', 'mx', 'my', 'w', 'h', 'line_x0', 'line_y0', 'line_x1', 'line_y1', 'gno', 'line_in_block', 'word_in_line']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows where essential numeric data is missing
        df.dropna(subset=['my', 'mx', 'h'], inplace=True)
        print("--- Data types converted and NA values dropped. ---")


        # --- Column Grouping (based on 90-degree rotated image) ---
        df = df.sort_values(by='my')

        avg_char_height = df['h'].mean()
        y_threshold = avg_char_height * 2.5
        print(f"Calculated y_threshold for column grouping: {y_threshold}")

        df['col_id'] = (df['my'].diff().abs() > y_threshold).cumsum()
        print("--- Column IDs assigned. ---")

        # --- Sorting Logic ---
        df = df.sort_values(
            by=['col_id', 'mx'],
            ascending=[True, True]
        )
        print("--- DataFrame sorted successfully. ---")

        # Write the sorted data to the output file
        sorted_list = df.values.tolist()
        print(f"--- Writing {len(sorted_list)} rows to output file... ---")
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(sorted_list)

        print(f"--- Successfully sorted and saved to '{output_csv_path}' ---")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_csv_path}", file=sys.stderr)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python ocr_sorter.py <input_csv_path> <output_csv_path>", file=sys.stderr)
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        sort_newspaper_data(input_file, output_file)
