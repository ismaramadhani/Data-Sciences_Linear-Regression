import pandas as pd

def load_data():    
    CSV_URL = "https://docs.google.com/spreadsheets/d/14N_7TYa72pCxbJE5m3W9RwoZVGKqOP4r/export?format=csv&gid=970731575"
    data = pd.read_csv(CSV_URL)
    
    # Remove columns that start with 'ln' (log-transformed columns)
    data = data.loc[:, ~data.columns.str.startswith('ln')]
    
    return data
