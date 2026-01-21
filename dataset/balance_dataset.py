import os
import random

def balance_folders(target_folder, reference_folder):
    """
    Deletes files from target_folder until it has the same number of files as reference_folder.
    """
    # 1. Validate paths
    if not os.path.exists(target_folder) or not os.path.exists(reference_folder):
        print(f"Error: One or both folders do not exist.\nTarget: {target_folder}\nReference: {reference_folder}")
        return

    # 2. Get list of files (filtering out subdirectories)
    target_files = [f for f in os.listdir(target_folder) if os.path.isfile(os.path.join(target_folder, f))]
    reference_files = [f for f in os.listdir(reference_folder) if os.path.isfile(os.path.join(reference_folder, f))]

    count_target = len(target_files)
    count_ref = len(reference_files)

    print(f"Files in '{target_folder}': {count_target}")
    print(f"Files in '{reference_folder}': {count_ref}")

    # 3. Check if deletion is necessary
    if count_target <= count_ref:
        print("Target folder has fewer or equal samples. No deletion needed.")
        return

    num_to_delete = count_target - count_ref
    print(f"Deleting {num_to_delete} files from '{target_folder}'...")

    # 4. Randomly select files to delete
    files_to_delete = random.sample(target_files, num_to_delete)

    # 5. Perform deletion
    for file_name in files_to_delete:
        file_path = os.path.join(target_folder, file_name)
        try:
            os.remove(file_path)
        except OSError as e:
            print(f"Error deleting {file_path}: {e}")

    print("Deletion complete.")
    
    # Verify final counts
    final_count = len([f for f in os.listdir(target_folder) if os.path.isfile(os.path.join(target_folder, f))])
    print(f"New count in '{target_folder}': {final_count}")

if __name__ == "__main__":
    # Define your directories relative to the script location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    REAL_DIR = os.path.join(base_dir, 'raw', 'real')
    FAKE_DIR = os.path.join(base_dir, 'raw', 'fake')
    
    balance_folders(REAL_DIR, FAKE_DIR)
