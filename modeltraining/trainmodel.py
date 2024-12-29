import argparse
import pathlib
import coloredlogs
import logging

from ultralytics import YOLO

def run(newModel, pretrainedModel, data, epochs, imagesize, logger):
    # Load a pre-trained YOLO model (you can choose n, s, m, l, or x versions)
    logger.info("Building a new YOLO Model")
    # model = YOLO(newModel)
    model = YOLO("yolo11n-seg.yaml")

    logger.info("Loading a pre-trained YOLO Image Segmentation Model")
    # model = YOLO(pretrainedModel)
    model = YOLO("yolo11n-seg.pt")

    logger.info("Building from YAML to transfer weights")
    # model = YOLO(newModel).load(pretrainedModel)
    model = YOLO("yolo11n-seg.yaml").load("yolo11n-seg.pt")

    # Start training on your custom dataset
    logger.info("Starts training the model on custom dataset")
    # model.train(data = data, epochs = epochs, imgsz = imagesize, device = [0])#, task = "segment")
    results = model.train(data = "job4segdataset/data.yaml", epochs = 100, imgsz = 640, device = [0])

    logger.info("Training for model is complete")

def main():
    LOG_LEVEL = logging.info

    parser = argparse.ArgumentParser(
        description = "Trains the model",
        formatter_class = argparse.RawDescriptionHelpFormatter,)
    
    parser.add_argument("--newModel", "-nm", type = str, default = "yolo11n-seg.yaml")

    parser.add_argument("--pretrainedModel", "-ptm", type = str, default = "yolo11n-seg.pt")
    
    parser.add_argument("--data", "-d", type = str,
                        default = "segsample/data.yaml",
                        help = "the path to the data that the model will be trained on")
    
    parser.add_argument("--epochs", "-ep", type = int, default = 100)

    parser.add_argument("--imagesize","-imgsz", type = int, default = 640)

    parser.add_argument("--verbose", "-v", action = "store_true",
                        help = f"turn on debugging message({LOG_LEVEL})")

    # creating args objects and parsing arguments
    args = parser.parse_args()
    LEVEL = logging.DEBUG if args.verbose else LOG_LEVEL
    logger = logging.getLogger()
    coloredlogs.install(level='LEVEL')

    # print("args.data", args.data)
    run(args.newModel, args.pretrainedModel, args.data, args.epochs, args.imagesize, logger)

main()

### to read data.yaml file for debugging
# file = open("sample/data.yaml")
# for line in file: 
#     print("line", line)

# line 5 has 1 flag, line 8 has 3 flags (data, epochs, imgsz)     (like parse.addargument stuff)
# line 5 has default "yolo11n.pt" value but give option to override
# makes it easier to add configs for argparse later and for making it easier to connect with remote machine

#################################################
# from ultralytics import YOLO

# # Load a pre-trained YOLO model (you can choose n, s, m, l, or x versions)
# model = YOLO("yolo11n.pt")

# # Start training on your custom dataset
# model.train(data="sample/data.yaml", epochs=100, imgsz=640)
#################################################
