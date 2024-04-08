# takes around 15-20 min for 14GB processed_data.zip

import os
import zipfile
from tqdm import tqdm

zip_file = "data/processed_data.zip"
destination = "data"

with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    file_list = zip_ref.namelist()
    total_files = len(file_list)
    
    progress_bar = tqdm(total=total_files, desc="Extracting")
    
    for file_name in file_list:
        zip_ref.extract(file_name, destination)
        progress_bar.update(1)
        
    progress_bar.close()