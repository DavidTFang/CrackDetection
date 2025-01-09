import argparse
import pathlib
import coloredlogs
import logging
import yaml

from ultralytics import YOLO

def run(newModel, pretrainedModel, data, epochs, imagesize, hyp, logger):
    # Load a pre-trained YOLO model (you can choose n, s, m, l, or x versions)
    logger.info("Building a new YOLO Model")
    model = YOLO(newModel)
    # model = YOLO("yolo11m-seg.yaml")

    logger.info("Loading a pre-trained YOLO Image Segmentation Model")
    model = YOLO(pretrainedModel)
    # model = YOLO("yolo11m-seg.pt")

    logger.info("Building from YAML to transfer weights")
    model = YOLO(newModel).load(pretrainedModel)
    # model = YOLO("yolo11m-seg.yaml").load("yolo11m-seg.pt")

    print(f"Data path: {data}")
    print(f"Hyperparameters file: {hyp}")

    # Start training on your custom dataset
    logger.info("Starts training the model on custom dataset")

    model.train(data = data, epochs = epochs, imgsz = imagesize, device = [0], cfg = hyp)

    logger.info("Training for model is complete")

def main():
    LOG_LEVEL = logging.info

    parser = argparse.ArgumentParser(
        description = "Trains the model",
        formatter_class = argparse.RawDescriptionHelpFormatter,)
    
    parser.add_argument("--newModel", "-nm", type = str, default = "yolo11m-seg.yaml")

    parser.add_argument("--pretrainedModel", "-ptm", type = str, default = "yolo11m-seg.pt")
    
    parser.add_argument("--data", "-d", type = str,
                        default = "/mnt/f/SegmentationData/data.yaml",
                        help = "the path to the data that the model will be trained on")
    
    parser.add_argument("--epochs", "-ep", type = int, default = 100)

    parser.add_argument("--imagesize","-imgsz", type = int, default = 640)

    parser.add_argument("--hyp", "-hy", type = str, 
                        default = "/mnt/c/Users/light/Documents/GitHub/CrackDetection/hyper.yaml",
                        help = "the path to the custom hyperparameters for the model")

    parser.add_argument("--verbose", "-v", action = "store_true",
                        help = f"turn on debugging message({LOG_LEVEL})")

    # Creating args objects and parsing arguments
    args = parser.parse_args()
    LEVEL = logging.DEBUG if args.verbose else LOG_LEVEL
    logger = logging.getLogger()
    coloredlogs.install(level='LEVEL')

    # print("args.data", args.data)
    run(args.newModel, args.pretrainedModel, args.data, args.epochs, args.imagesize, args.hyp, logger)

main()
