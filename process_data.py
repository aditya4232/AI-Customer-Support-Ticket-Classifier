import pandas as pd
import numpy as np

# Read the excel file
df = pd.read_excel('data/telecom_cc.xlsx')

# Clean columns
df.columns = df.columns.str.strip()
print("Columns:", df.columns)

# Prepare dataframe
df_final = pd.DataFrame()
df_final['ticket_id'] = ['TKT' + str(i).zfill(6) for i in range(1, len(df) + 1)]
df_final['ticket_text'] = df['Question']
df_final['industry'] = 'Telecom'
df_final['category'] = df['Category']
df_final['priority'] = df['Priority']

# Save to CSV
df_final.to_csv('data/ticket_database.csv', index=False)
print("Saved data/ticket_database.csv")
