import os
import shutil
import kagglehub
from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset
import rarfile

def delete_items(starting_with="kos", directory="."):
    """
    Deletes folders and files in the specified directory that start with the given prefix.
    
    :param starting_with: Prefix of the folders and files to delete (default is "kos").
    :param directory: Path to the directory where the folders and files are located (default is current directory).
    """
    try:
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if item.startswith(starting_with):
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f"Deleted folder: {item_path}")
                elif os.path.isfile(item_path):
                    os.remove(item_path)
                    print(f"Deleted file: {item_path}")
    except Exception as e:
        print(f"Error: {e}")




if __name__ == "__main__":

    path = kagglehub.dataset_download("fakhrealam9537/leaf-disease-segmentation-dataset")

    print("Path to dataset files:", path)