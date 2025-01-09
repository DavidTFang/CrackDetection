import splitfolders
import argparse
import pathlib
import coloredlogs
import logging
from splitfolders import ratio
import os

def copy_labels(labels, input, name, out):
    # iterate over files in that directory
    os.mkdir(f"{out}/{name}/labels")

    for filename in os.listdir(input):
        f = os.path.join(input, filename)
        # checking if it is a file
        if os.path.isfile(f):
            # Get the base name (filename with extension)
            base_name = os.path.basename(f)

            # Split the base name into filename and extension
            filename_without_ext, _ = os.path.splitext(base_name)

            # Join the directory path with the filename without extension
            os.system(f"cp {labels}/{filename_without_ext}.txt {out}/{name}/labels")

def run(images, out, logger):
    logger.info("Attempting to split dataset")
    ratio(images, output=out,
                   seed = 1337, ratio = (0.8, 0.1, 0.1), group_prefix = None, move = False)
    logger.info("Split dataset")

def main():
    LOG_LEVEL = logging.INFO

    parser = argparse.ArgumentParser(
        description = "Splits the input and output data into train/test/val",
        formatter_class = argparse.RawDescriptionHelpFormatter,)

    parser.add_argument("--images", "-i", type = lambda p:
        pathlib.Path(p).absolute(),
        required = True,
        help = "the path to the image directory for the dataset to be split",)
    
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
    run(args.images, args.output, logger)

    train_input = f'{args.output}/train/images'
    val_input = f'{args.output}/val/images'
    test_input = f'{args.output}/test/images'
    copy_labels(args.labels, train_input, "train", args.output)

main()