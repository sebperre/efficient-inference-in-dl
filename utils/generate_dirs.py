import os
import sys

def create_log_dir_and_folder():
    """
    Creates the log directories if they don't exist
    """
    os.makedirs("../logs", exist_ok=True)
    os.makedirs(f"../logs/{os.path.basename(sys.argv[0])}", exist_ok=True)
    f = open("../logs/{}")


def write_log_file():
    """
    Creates the log directories if they don't exist
    """
    f = open("../logs/ImageNetTime.txt", "w")
    return f

def print_write(text, f):
    """
    Prints and writes to file
    """
    print(text)
    f.write(text)

if __name__ == "__main__":
    print("This is a utils function and shouldn't be run directly.")
