"""
Marimba pipeline for the CSIRO ANACC Zeiss Axio microscopes
"""
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

import cv2
import czifile
from ifdo.models import ImageData

from marimba.core.pipeline import BasePipeline
from marimba.core.utils.config import load_config

__author__ = "Chris Jackett"
__copyright__ = "Copyright 2023, Environment, CSIRO"
__credits__ = []
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Chris Jackett"
__email__ = "chris.jackett@csiro.au"
__status__ = "Development"


def valid_filename(filename: str) -> bool:
    """
    Args:
        filename: A string representing the filename to be checked.

    Returns:
        A boolean value indicating whether the filename structure conforms with ANACC standard.
    """
    # Check filename structure conforms with ANACC standard
    return len(filename.split("_")) == 8


class ZeissAxioObserver(BasePipeline):
    """The `ZeissAxioObserver` class is a subclass of `BasePipeline` that represents an observer for a Zeiss Axio microscope. It provides methods for importing data, processing source files
    *, and extracting image frames.

    Attributes:
        None

    Methods:
        - `get_pipeline_config_schema() -> dict`: Returns the pipeline configuration schema.
        - `get_collection_config_schema() -> dict`: Returns the collection configuration schema.
        - `_import(data_dir: Path, source_paths: List[Path], config: Dict[str, Any], **kwargs: dict) -> None`: Imports data from multiple source paths into a specified data directory.
        - `import_from_source_path(source_path: Path, data_dir: Path, config: Dict[str, Any]) -> None`: Imports data from a source path into a data directory.
        - `process_source_file(source_file: Path, data_dir: Path, config: Dict[str, Any]) -> None`: Processes a source file.
        - `directory_path_from_filename(data_dir: Path, filename: str) -> Path`: Constructs a new directory path from a filename.
        - `extract_image_from_file(source_file: Path, new_mlai_directory_path: Path) -> None`: Extracts an image from a source file and saves it to a specified directory.
        - `already_extracted(new_mlai_directory_path, file_name) -> bool`: Checks if a CZI file has already been extracted.
        - `construct_new_directory_paths(source_path, iso_timestamp, strain_id, magnification_factor, contrast_id, biological_stain_id) -> Path`: Constructs new directory paths for MLAI
    * images.
        - `extract_frames(image, file_name, new_mlai_directory_path) -> None`: Extracts frames from an image and saves them as JPG files in the MLAI archive directory.
        - `write_image_to_disk(file_path: str, image, location: str) -> None`: Writes an image to disk.
    """

    @staticmethod
    def get_pipeline_config_schema() -> dict:
        return {
            "project_pi": "Chris Jackett",
            "platform_id": "ZAO",
        }

    @staticmethod
    def get_collection_config_schema() -> dict:
        return {
            "data_collector": "Chris Jackett",
            "collection_year": 2021,
        }

    def _import(self, data_dir: Path, source_paths: List[Path], config: Dict[str, Any], **kwargs: dict):
        """
        Imports data from multiple source paths into a specified data directory.

        Args:
            data_dir (Path): The directory to import the data into.
            source_paths (List[Path]): A list of source paths where the data is located.
            config (Dict[str, Any]): A configuration dictionary for the import process.
            **kwargs (dict): Additional keyword arguments for future use.

        """

        self.logger.info(f"Importing data from {source_paths=} to {data_dir}")
        for source_path in source_paths:
            self.import_from_source_path(source_path, data_dir, config)

    def import_from_source_path(self, source_path: Path, data_dir: Path, config: Dict[str, Any]):
        """
        Args:
            source_path (Path): The path of the source directory to import from.
            data_dir (Path): The path of the data directory to store imported files.

        """
        if not source_path.is_dir():
            return

        for source_file in source_path.glob("**/*"):
            self.process_source_file(source_file, data_dir, config)

    def process_source_file(
        self,
        source_file: Path,
        data_dir: Path,
        config: Dict[str, Any],
    ):
        """
        Process the source file.

        Args:
            source_file (Path): The path to the source file.
            data_dir (Path): The path to the data directory.
            config (Dict[str, Any]): The configuration information.

        """
        if source_file.is_file() and source_file.suffix.lower() == ".czi" and f'_{config.get("collection_year")}' in source_file.name:
            self.logger.info(f"{source_file}")
            self.logger.info("-----------------------------------------------------------------------------------------------")
            self.logger.info(f"Processing CZI file: {source_file.name}...")

            if not valid_filename(source_file.name):
                return

            new_mlai_directory_path = self.directory_path_from_filename(data_dir, source_file.name)
            new_mlai_image_directory_path = new_mlai_directory_path / "images"
            new_mlai_data_directory_path = new_mlai_directory_path / "data"

            if self.already_extracted(new_mlai_directory_path, source_file.name):
                self.logger.debug(f"Imported {source_file.resolve().absolute()} -> {data_dir}")
                return

            self.logger.info(f"Reading CZI file: {source_file}...")

            # Try to read CZI file and extract image frames
            try:
                image = czifile.imread(str(source_file))
                # Extract CZI images and video frames

                if len(image.shape) == 5:
                    # Remove the seventh filename identifier from the filename by splitting on underscore
                    file_name_parts = source_file.stem.split("_")
                    file_name = "_".join(file_name_parts[:6] + file_name_parts[7:])

                    self.extract_frames(image, file_name, new_mlai_image_directory_path)
                    self.extract_data(source_file, file_name, new_mlai_data_directory_path)

            except Exception as e:
                self.logger.error(f"Error extracting file {source_file.name}")
                self.logger.error(e)

    def extract_data(self, source_file, file_name, new_mlai_data_directory_path):
        self.logger.info(f"Extracting data...")

        new_mlai_data_directory_path.mkdir(parents=True, exist_ok=True)
        with czifile.CziFile(source_file) as czi:
            # Get CZI file metadata as dictionary
            metadata = czi.metadata(raw=False)

            new_mlai_file_path = new_mlai_data_directory_path / (file_name + ".JSON")

            self.write_data_to_disk(metadata, new_mlai_file_path)

    def directory_path_from_filename(self, data_dir: Path, filename: str) -> Path:
        """
        Args:
            data_dir (Path): The root directory where the new directory paths will be created.
            filename (str): The name of the file from which to extract attributes.

        Returns:
            Path: The newly constructed directory path.

        Raises:
            ValueError: If the filename is not in the expected format or if any of the extracted attributes are invalid.
        """
        # Extract filename attributes
        (
            strain_id,
            imaging_system_id,
            magnification_factor,
            contrast_id,
            channel_id,
            biological_stain_id,
            object_id,
            iso_timestamp,
        ) = filename.split("_")
        # Construct new directory paths
        return self.construct_new_directory_paths(data_dir, iso_timestamp, strain_id, magnification_factor, contrast_id, biological_stain_id)

    def already_extracted(self, new_mlai_directory_path, file_name):
        """
        Check if the CZI file has previously been extracted int JPGs

        Note: This assumes that if the first video frame already exists, then the entire video sequence has previously been extracted

        :param new_anacc_file_path:
        :param new_mlai_directory_path:
        :param file_name:
        """

        # Define new MLAI image and video frame paths
        new_mlai_image_path = os.path.join(new_mlai_directory_path, file_name) + ".JPG"
        new_mlai_video_frame_path = os.path.join(new_mlai_directory_path, file_name) + "_001.JPG"

        if os.path.isfile(new_mlai_image_path) or os.path.isfile(new_mlai_video_frame_path):
            return True
        else:
            return False

    from pathlib import Path

    def construct_new_directory_paths(self, source_path, iso_timestamp, strain_id, magnification_factor, contrast_id, biological_stain_id):
        """
        Args:
            source_path: The path of the source directory where the file is located.
            iso_timestamp: The ISO timestamp of the file.
            strain_id: The strain ID of the file.
            magnification_factor: The magnification factor of the file.
            contrast_id: The contrast ID of the file.
            biological_stain_id: The biological stain ID of the file.
        Returns:
            The constructed new directory path for MLAI images.
        """
        # Copy to ANACC image archive
        self.logger.debug("Calculating new ANACC and MLAI image archive paths...")
        # Construct new MLAI file path and check directory path exists, creating new directories if necessary
        split_iso_timestamp = iso_timestamp.split("T")[0]
        year = split_iso_timestamp[0:4]
        month = split_iso_timestamp[4:6]
        day = split_iso_timestamp[6:8]
        new_mlai_directory_path = Path(source_path) / year / month / day / strain_id / magnification_factor / contrast_id / biological_stain_id

        return new_mlai_directory_path

    def extract_frames(self, image, file_name, new_mlai_image_directory_path):
        """
        Extracts frames from an image and saves them as JPG files in the MLAI archive directory.

        Args:
            image: The image from which to extract frames.
            file_name: The name of the original file.
            new_mlai_image_directory_path: The path to the MLAI archive directory where the extracted frames will be saved.
        """
        self.logger.info(f"Extracting frames...")

        # # If CZI file has stacked images, fetch number of images
        # if len(image.shape) == 4:
        #     # Squeeze empty image dimensions
        #     single_image = image.squeeze()
        #
        #     # Write new JPG image to MLAI archive
        #     new_mlai_file_path = os.path.join(new_mlai_image_directory_path, new_file_name) + ".JPG"
        #     self.write_image_to_disk(new_mlai_file_path, single_image, "MLAI")

        # if len(image.shape) == 5:
        new_mlai_image_directory_path.mkdir(parents=True, exist_ok=True)
        number_of_stacked_images = image.shape[0]

        for i in range(number_of_stacked_images):
            # Squeeze empty image dimensions
            stacked_image = image[i].squeeze()

            # new_mlai_file_path = os.path.join(new_mlai_image_directory_path, new_file_name) + f"_{i + 1:03d}.JPG"
            new_mlai_file_path = Path(new_mlai_image_directory_path) / (file_name + f"_{i + 1:03d}.JPG")

            # Write new JPG image to MLAI archive
            self.write_image_to_disk(new_mlai_file_path, stacked_image, "MLAI")

    def write_image_to_disk(self, file_path: Path, image, location: str):
        """
        Args:
            file_path (str): The file path where the image will be written to.
            image: The input image that will be written to disk.
            location (str): The location/destination of the image.

        """
        self.logger.debug(f"Writing new {location} JPG file: {file_path}")

        # Normalise CZI image
        normalised_image = cv2.normalize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_16U)

        # Write JPG to disk
        if cv2.imwrite(str(file_path), normalised_image, [cv2.IMWRITE_JPEG_QUALITY, 90]):
            self.logger.debug(f"Completed writing JPG file: {file_path}")
        else:
            self.logger.error(f"Could not write JPG image: {file_path}")

    def write_data_to_disk(self, file_path: str, data: Dict):
        """
        Write data to a JSON file on disk.
        Args:
            file_path (str): The file path where the image will be written to.
            data: The input dictionary that will be written to disk.
        """

        self.logger.debug(f"Writing new data to JSON file: {file_path}")

        # Write dictionary to JSON file
        try:
            with open(file_path, "w") as json_file:
                json.dump(data, json_file)
            self.logger.debug(f"Completed writing data to JSON file: {file_path}")
        except Exception as e:
            self.logger.error(f"Could not write data to JSON file: {file_path}")
            self.logger.error(e)

    def _process(self, data_dir: Path, config: Dict[str, Any], **kwargs: dict):
        """
        Implementation of the Marimba process command for the Zeiss Axio Observer
        """

        # Loop through each deployment subdirectory in the instrument work directory
        for deployment in os.scandir(self.work_path):
            # Get deployment name and config path
            deployment_name = deployment.path.split("/")[-1]
            deployment_config_path = Path(deployment.path) / Path(deployment_name + ".yml")

            # Check if deployment metadata file exists and skip deployment if not present
            if not deployment_config_path.is_file():
                self.logger.warning(
                    f'SKIPPING DEPLOYMENT - Cannot find deployment metadata file "{deployment_name}.yml" in deployment directory at path: "{deployment.path}"'
                )
                continue
            else:
                # TODO: Need to validate deployment metadata file here and load deployment config
                self.logger.info(f'Found valid Marimba deployment with "{deployment_name}.yml" at path: "{deployment.path}"')
                deployment_config = load_config(deployment_config_path)

                # Loop through each file in the deployment directory
                for file in os.scandir(deployment.path):
                    # Define regex to match any of the filetypes to be renamed
                    extensions_pattern = f'({"|".join(re.escape(extension) for extension in self.filetypes)})$'
                    file_path = file.path

                    # Match case-insensitive regex expression in file name
                    if re.search(extensions_pattern, file_path, re.IGNORECASE):
                        # Extract the CZI files
                        # self.extract()
                        print(f"Extracting...{file_path}")
                        self.extract(file_path)

    def _compose(self, data_dirs: List[Path], configs: List[Dict[str, Any]], **kwargs: dict) -> Dict[Path, Tuple[Path, List[ImageData]]]:
        data_mapping = {}

        return data_mapping
