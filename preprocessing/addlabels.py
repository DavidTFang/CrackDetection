import os
import coloredlogs
import logging
import argparse
import pathlib

def add_labels(input, output, name, logger):
    logger.info("Attempting to add labels")

    # Create the labels directory
    labels_dir = os.path.join(output, name, "labels")

    os.makedirs(labels_dir, exist_ok=True)

    for filename in os.listdir(input):
        f = os.path.join(input, filename)

        # checking if it is a file
        if os.path.isfile(f):
            # Get the base name (filename with extension)
            base_name = os.path.basename(f)

            # Split the base name into filename and extension
            filename_without_ext, _ = os.path.splitext(base_name)

            # Construct the path for the corresponding text file
            txt_file_path = os.path.join(labels_dir, f"{filename_without_ext}.txt")
            labels_file_path = os.path.join({output}, "labels")
            existing_files_path = os.path.join(labels_file_path, f"{filename_without_ext}.txt")

            if not os.path.isfile(existing_files_path):
                with open(txt_file_path, 'w') as f:
                    pass
    
    logger.info("Added labels")

def main():
    LOG_LEVEL = logging.INFO

    parser = argparse.ArgumentParser(
        description = "Splits the input and output data into train/test/val",
        formatter_class = argparse.RawDescriptionHelpFormatter,)
    
    parser.add_argument("--labels", "-la", type = lambda p:
        pathlib.Path(p).absolute(),
        required = False,
        help = "the path to the label directory for the dataset to be split",)
    
    parser.add_argument("--output", "-o", type = lambda p:
        pathlib.Path(p).absolute(),
        required = False,
        default = "output",
        help = "the path to the directory for the output of the split dataset",)
    
    parser.add_argument("--verbose", "-v", action = "store_true",
                        help = f"turn on debugging message({LOG_LEVEL})",)
    
    # creating args object and parsing arguments
    args = parser.parse_args()

    coloredlogs.install(level = LOG_LEVEL)
    LOG_LEVEL = logging.DEBUG if args.verbose else LOG_LEVEL
    logger = logging.getLogger()

    input = os.path.join(args.output, "images")
    add_labels(input, args.output, "train", logger)

main()