from pathlib import Path
import re
import shutil
import os

# Path to the parent folder
folder_path = Path("/home/dasb/workspace/QTT_SEG/qtt")

# Regular expression to match UUID-like folder names (32-character hex strings)
subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]

# List and filter only folders that match the UUID pattern
# uuid_folders = [f.name for f in folder_path.iterdir() if f.is_dir() and uuid_pattern.match(f.name)]

# Print the matching folders
for folder_name in subfolders:
    shutil.rmtree(folder_name)  
    print(f"Deleted: {folder_name}")
