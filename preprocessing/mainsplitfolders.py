import splitfolders
import argparse
import pathlib
import coloredlogs
import logging

def run(input, out, logger):
    logger.info("attempting to split dataset")
    splitfolders.ratio(input, output=out,
                   seed=1337, ratio=(0.8, 0.1, 0.1), group_prefix=None, move=False)
    logger.info("split dataset")

def main():
    LOG_LEVEL = logging.INFO

    parser = argparse.ArgumentParser(
        description="Splits the input and output data into train/test/val",
        
    formatter_class=argparse.RawDescriptionHelpFormatter,)

    parser.add_argument("--input", "-i", type=lambda p:
    pathlib.Path(p).absolute(),
        required=True,
        help="the path to the directory for the dataset to be split",)
    
    parser.add_argument("--output", "-o", type=lambda p:
    pathlib.Path(p).absolute(),
        required=False,
        default="output",
        help="the path to the directory for the output of the split dataset",)
    
    parser.add_argument("--verbose", "-v", action="store_true",
                        help=f"turn on debugging message({LOG_LEVEL})",)
    
    # creating args object and parsing arguments
    # par
    args = parser.parse_args()
    LEVEL = logging.DEBUG if args.verbose else LOG_LEVEL
    logger = logging.getLogger()
    coloredlogs.install(level=LEVEL)
    run(args.input, args.output, logger)

main()