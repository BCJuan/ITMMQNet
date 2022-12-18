"""Script to execute the data organization
"""
import argparse
import logging
from hdrml.data_org.datasets import (
    TestDataset,
    FuntDataset,
    HDRPlusDataset,
    WardDataset,
    PFSToolsDataset,
    DatasetCollection,
    ValidationDataset
)
from hdrml.data_org.tfrecord_builders import TFRecordBuilder, OnlyHDRTFRecordBuilder


def parse():
    """Parse args from cmd call

    Returns:
        arguments: arguments object with arguments as properties
    """
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--organize",
        dest="organize",
        default=False,
        action="store_true",
        help="boolean for organizing the hdr eye dataset",
    )
    argparser.add_argument(
        "--build",
        dest="build",
        default=False,
        action="store_true",
        help="boolean for organizing the game dataset",
    )
    argparser.add_argument(
        "--root",
        dest="root",
        type=str,
        default="./data",
        help="root for storing all data files"
    )
    arguments, _ = argparser.parse_known_args()
    return arguments


if __name__ == "__main__":

    args = parse()
    if args.organize:
        pfsToolsDataset = PFSToolsDataset("pfstools", args.root)
        pfsToolsDataset.organize()

        wardDataset = WardDataset("ward", args.root)
        wardDataset.organize()

        HDRPlusDataset.download()
        hdrplusdataset = HDRPlusDataset("hdrplus", args.root)
        hdrplusdataset.organize()

        hdreye = TestDataset("hdreye_test", args.root)
        hdreye.organize("../HDR-Eye.zip")

        FuntDataset.download()
        funt = FuntDataset("funt", args.root)
        funt.organize("../HDR.zip")

        hdrreal = TestDataset("hdrreal")
        hdrreal.organize("../HDR-Real.zip")

        validation = ValidationDataset("validation", "hdrplus", args.root)
        validation.organize()

        logging.warning("You need to run first funt, ward, pfstools and hdrplus.")
        ds_list = [HDRPlusDataset("hdrplus", root=args.root),
                   FuntDataset("funt", root=args.root),
                   WardDataset("ward", root=args.root),
                   PFSToolsDataset("pfstools", root=args.root)]

        dsCollection = DatasetCollection("train", ds_list, root=args.root)
        dsCollection.organize()

    if args.build:
        tBuilder = OnlyHDRTFRecordBuilder("train", args.root, n_machines=2)
        tBuilder.organize()

        vbuilder = OnlyHDRTFRecordBuilder("validation", args.root, n_machines=2)
        vbuilder.organize()

        hdreye = TFRecordBuilder("hdreye_test", args.root, n_machines=2)
        hdreye.organize()

        hdrreal = TFRecordBuilder("hdrreal", args.root, n_machines=2)
        hdrreal.organize()
