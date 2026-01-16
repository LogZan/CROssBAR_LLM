import pandas as pd
import os

# Define file paths
input_file = "/GenSIvePFS/users/clzeng/workspace/CROssBAR_LLM/BioMaze/TrueFalse-00000-of-00001.parquet"
output_file = "/GenSIvePFS/users/clzeng/workspace/CROssBAR_LLM/BioMaze_Filtered_Normal_Natural.csv"

print(f"Reading file: {input_file}")

try:
    # Read the parquet file
    df = pd.read_parquet(input_file)
    
    # Print available columns to confirm
    print(f"Columns found: {df.columns.tolist()}")
    
    # Filter conditions
    # Inquiry Type = Normal
    # Extra Condition = Natural
    
    filtered_df = df[
        (df['Inquiry Type'] == 'Normal') & 
        (df['Extra Condition'] == 'Natural')
    ]
    
    print(f"Total rows: {len(df)}")
    print(f"Filtered rows: {len(filtered_df)}")
    
    # Export to CSV
    filtered_df.to_csv(output_file, index=False)
    print(f"Exported filtered data to: {output_file}")

except Exception as e:
    print(f"An error occurred: {e}")
