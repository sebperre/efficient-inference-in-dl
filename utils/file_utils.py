import os
import sys
import datetime
import argparse
import time
import functools

FILE = None

def format_time(time):
    """
    Formats time in hours, minutes and seconds.
    """
    hours = int(time // 3600)
    minutes = int((time % 3600) // 60)
    seconds = int(time % 60)

    return f"{hours}h {minutes}m {seconds}s"

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, description=None, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time

        if FILE is None or description is None:
            raise Exception("No file provided")

        FILE.write(f"[Timer] {description}: Took {format_time(elapsed_time)}.\n")

        return result

    return wrapper

def write_file(folder_name, dataset, model, epochs, subset_size = None, acc_sac = None, batch_size = None, model_classifier_epochs = None):
    """
    Creates the log directories if they don't exist.
    """
    global FILE
    # LOG_PATH = "/home/sebperre/programming-projects/efficient-inference-in-dl/logs"
    LOG_PATH = "/home/sebastien/programming-projects/efficient-inference-in-dl/logs"
    sub_folder_name = datetime.datetime.now().strftime("%m-%d %H:%M")
    os.makedirs(LOG_PATH, exist_ok=True)
    os.makedirs(f"{LOG_PATH}/{folder_name}_{os.path.basename(sys.argv[0])[:-3]}", exist_ok=True)
    path = f"{LOG_PATH}/{folder_name}_{os.path.basename(sys.argv[0])[:-3]}/{sub_folder_name}"
    os.makedirs(path, exist_ok=True)
    f = open(f"{path}/log.txt", "w")
    FILE = f
    stamp_file(folder_name, dataset, model, epochs, subset_size, acc_sac, batch_size, model_classifier_epochs)
    return f, path

def print_write(text):
    """
    Prints and writes to file.
    """
    if FILE is None:
        raise Exception("No file provided")
    print(text)
    FILE.write(f"{text}\n")

def get_args(epoch: bool = False, 
             subset: bool = False, 
             acc_sac: bool = False, 
             batch_size: bool = False,
             model_classifier_epoch: bool = False, 
             epoch_default: int = 50, 
             subset_default: int = 1000, 
             acc_sac_default: float = 0.1, 
             batch_size_default: int = 20,
             model_classifier_epoch_default: int = 15):
    """
    Parse command-line arguments and return them.
    """
    parser = argparse.ArgumentParser(description="Process training parameters.")

    if epoch:
        parser.add_argument("--epochs", type=int, default=epoch_default, help="Number of epochs for training (default: 10)")
    if subset:
        parser.add_argument("--subset", type=int, default=subset_default, help="Size of the training subset (default: 100)")
    if acc_sac:
        parser.add_argument("--acc_sac", type=float, default=acc_sac_default, help="Accuracy Sacriface for the combined model (default: 0.1)")
    if batch_size:
        parser.add_argument("--batch_size", type=int, default=batch_size_default, help="Batch Size for training (default: 20)")
    if model_classifier_epoch:
        parser.add_argument("--model_classifier_epochs", type=int, default=model_classifier_epoch_default, help="Number of epochs for training the model classifier (default: 15)")

    return parser.parse_args()

def stamp_file(folder_name, dataset, model, epochs, subset_size, acc_sac, batch_size, model_classifier_epochs):
    FILE.write("===========FILE STAMP=============\n")
    FILE.write(f"{folder_name} {os.path.basename(sys.argv[0])}\n")
    FILE.write(f"Time Started: {datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S")}\n")
    FILE.write(f"Dataset: {dataset}\n")
    FILE.write(f"Model: {model}\n")
    FILE.write(f"Epochs: {epochs}\n")
    if subset_size is not None:
        FILE.write(f"Subset Size: {subset_size}\n")
    if acc_sac is not None:
        FILE.write(f"Accuracy Sacrifice: {acc_sac}\n")
    if batch_size is not None:
        FILE.write(f"Batch Size: {batch_size}")
    if model_classifier_epochs is not None:
        FILE.write(f"Model Classifier Epochs: {model_classifier_epochs}\n")
    FILE.write("==================================\n\n")

if __name__ == "__main__":
    print("This is a utils file and shouldn't be run directly.")
