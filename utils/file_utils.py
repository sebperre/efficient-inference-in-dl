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

def write_file(folder_name):
    """
    Creates the log directories if they don't exist.
    """
    global FILE
    LOG_PATH = "/home/sebperre/programming-projects/efficient-inference-in-dl/logs"
    os.makedirs(LOG_PATH, exist_ok=True)
    os.makedirs(f"{LOG_PATH}/{folder_name}_{os.path.basename(sys.argv[0])[:-3]}", exist_ok=True)
    f = open(f"{LOG_PATH}/{folder_name}_{os.path.basename(sys.argv[0])[:-3]}/{datetime.datetime.now().strftime("%m-%d %H:%M")}.txt", "w")
    f.write(f"{folder_name} {os.path.basename(sys.argv[0])}: Ran at {datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S")}\n")
    FILE = f
    return f

def print_write(text):
    """
    Prints and writes to file.
    """
    if FILE is None:
        raise Exception("No file provided")
    print(text)
    FILE.write(f"{text}\n")

def get_args(epoch: bool = False, subset: bool = False, acc_sac: bool = False, batch_size: bool = False, epoch_default: int = 50, subset_default: int = 1000, acc_sac_default: float = 0.1, batch_size_default: int = 20):
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
        parser.add_argument("--batch_size", type=int, default=batch_size, help="Batch Size for training (default: 20)")

    return parser.parse_args()

if __name__ == "__main__":
    print("This is a utils file and shouldn't be run directly.")
