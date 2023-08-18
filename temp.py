import pandas as pd

# Read the CSV file
df = pd.read_csv('merged_file.csv')

# Group by the 'index' column and concatenate instructor names
combined_instructors = df.groupby('index')['instructor_name'].apply(', '.join).reset_index()

# Merge the combined instructors back to the original DataFrame
df_combined = df.drop('instructor_name', axis=1).drop_duplicates(subset=['index']).merge(combined_instructors, on='index')

# Write the combined DataFrame to a new CSV file
df_combined.to_csv('combined_file.csv', index=False)
