"""
Marimba pipeline for the CSIRO ANACC Zeiss Axio microscopes
"""
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
from uuid import uuid4

import cv2
import czifile
import numpy as np
from PIL import Image
from ifdo.models import (
    ImageData,
    ImageAcquisition,
    ImageQuality,
    ImageDeployment,
    ImageIllumination,
    ImagePixelMagnitude,
    ImageMarineZone,
    ImageSpectralResolution,
    ImageCaptureMode,
    ImageFaunaAttraction,
    ImagePI,
)

from marimba.core.pipeline import BasePipeline

__author__ = "Chris Jackett"
__copyright__ = "Copyright 2023, Environment, CSIRO"
__credits__ = []
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Chris Jackett"
__email__ = "chris.jackett@csiro.au"
__status__ = "Development"

from marimba.lib import image


def is_valid_filename(filename: str) -> bool:
    """
    Args:
        filename (str): The filename to be checked.

    Returns:
        bool: True if the filename structure conforms with ANACC standard, False otherwise.

    """
    # Check filename structure conforms with ANACC standard
    return len(filename.split("_")) == 8


class ZeissAxioObserver(BasePipeline):
    """
    ZeissAxioObserver class.
    This class is a pipeline implementation for importing and processing data from Zeiss Axio Observer microscopy systems.

    Attributes:
        None

    Methods:
        get_pipeline_config_schema() -> dict:
            Returns the schema for the pipeline configuration.

        get_collection_config_schema() -> dict:
            Returns the schema for the collection configuration.

        _import(data_dir: Path, source_paths: List[Path], config: Dict[str, Any], **kwargs: dict):
            Imports data from source paths and saves it to the data directory.

        process_source_file(source_file: Path, data_dir: Path, config: Dict[str, Any]):
            Processes a source file and extracts images and videos from a CZI file.

        get_output_dir_from_filename(data_dir: Path, filename: str) -> Path:
            Generates the output directory path based on the filename attributes.

        construct_new_paths(data_dir, magnification_factor, contrast_id, biological_stain_id, strain_id, iso_timestamp) -> str:
            Constructs the output path based on the given attributes.

        extract_images(image, output_image_name, output_image_dir):
            Extracts individual images from a stacked image array and saves them to disk.

        write_image_to_disk(output_image_path: Path, image):
            Writes an image to disk in JPG format.

        extract_video(image, output_video_name, output_video_dir, video_frame_rate):
            Extracts a video from an image array and saves it to disk.

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
        Args:
            data_dir (Path): The directory where the imported data will be saved.
            source_paths (List[Path]): A list of paths to the source directories or files to import.
            config (Dict[str, Any]): A dictionary containing configuration options for the import process.
            **kwargs (dict): Additional keyword arguments.

        """

        self.logger.info(f"Importing data from {source_paths=} to {data_dir}")
        for source_path in source_paths:
            if not source_path.is_dir():
                return

            for source_file in source_path.glob("**/*"):
                self.process_source_file(source_file, data_dir, config)

    def process_source_file(self, source_file: Path, data_dir: Path, config: Dict[str, Any]):
        """
        Processes a source file and extracts images and videos from a CZI file.

        Args:
            source_file (Path): The path to the source file.
            data_dir (Path): The directory where the output data will be stored.
            config (Dict[str, Any]): A dictionary containing the configuration parameters.

        """
        is_czi_file = source_file.suffix.lower() == ".czi"
        contains_collection_year = f'_{config.get("collection_year")}' in source_file.name
        contains_platform_id = f'_{self.config.get("platform_id")}' in source_file.name

        if source_file.is_file() and is_czi_file and contains_collection_year and contains_platform_id:
            self.logger.info(f"Processing file: {source_file.name}...")

            if not is_valid_filename(source_file.name):
                return

            output_base_dir = self.get_output_dir_from_filename(data_dir, source_file.stem)
            output_image_dir = output_base_dir / "images"
            output_video_dir = output_base_dir / "video"
            output_data_dir = output_base_dir / "data"

            self.logger.info(f"Reading CZI file: {source_file}...")

            # Try to read CZI file and extract image frames
            try:
                image = czifile.imread(str(source_file))
                # Extract CZI images and video frames

                # Check that the CZI file is a video
                if len(image.shape) == 5:
                    # Remove the seventh filename identifier from the filename by splitting on underscore
                    # file_name_parts = source_file.stem.split("_")
                    # output_file_name = "_".join(file_name_parts[:6] + file_name_parts[7:])
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
                    ) = source_file.stem.split("_")
                    # Construct new directory paths
                    output_file_name = "_".join([imaging_system_id, magnification_factor, contrast_id, biological_stain_id, strain_id, iso_timestamp])

                    self.extract_images(image, output_file_name, output_image_dir)
                    video_frame_rate = self.extract_metadata(source_file, output_file_name, output_data_dir)
                    self.extract_video(image, output_file_name, output_video_dir, video_frame_rate)

            except Exception as e:
                self.logger.error(f"Error extracting file {source_file.name}")
                self.logger.error(e)

    def get_output_dir_from_filename(self, data_dir: Path, filename: str) -> Path:
        """
        Args:
            data_dir (Path): The root directory where the output directory will be created.
            filename (str): The filename from which the attributes will be extracted.

        Returns:
            Path: The generated output directory path.

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
        return data_dir / magnification_factor / contrast_id / biological_stain_id / strain_id / iso_timestamp
        # return self.construct_new_paths(data_dir, iso_timestamp, strain_id, magnification_factor, contrast_id, biological_stain_id)

    def construct_new_paths(self, data_dir, magnification_factor, contrast_id, biological_stain_id, strain_id, iso_timestamp):
        """
        Args:
            data_dir (str): The directory where the new output directory structure will be built.
            magnification_factor (str): The magnification factor of the image.
            contrast_id (str): The contrast ID of the image.
            biological_stain_id (str): The ID of the biological stain used in the image.
            strain_id (str): The ID of the strain of the subject in the image.
            iso_timestamp (str): The ISO timestamp of the image.

        Returns:
            str: The constructed output path.

        """
        # Copy to ANACC image archive
        self.logger.debug("Building new output directory structure...")
        # Construct new MLAI file path and check directory path exists, creating new directories if necessary
        # split_iso_timestamp = iso_timestamp.split("T")[0]
        # year = split_iso_timestamp[0:4]
        # month = split_iso_timestamp[4:6]
        # day = split_iso_timestamp[6:8]
        # output_path = data_dir / year / month / day / strain_id / magnification_factor / contrast_id / biological_stain_id
        output_path = data_dir / magnification_factor / contrast_id / biological_stain_id / strain_id / iso_timestamp

        return output_path

    def extract_images(self, image, output_image_name, output_image_dir):
        """
        Extracts individual images from a stacked image array and saves them to disk in a specified directory.

        Args:
            image (np.ndarray): A stacked image array.
            output_image_name (str): The base name for the output images.
            output_image_dir (pathlib.Path): The directory where the output images will be saved.
        """

        self.logger.info(f"Extracting images...")
        output_file_path = output_image_dir / f"{output_image_name}_001.JPG"
        if output_file_path.is_file():
            self.logger.warning(f"File {output_file_path.resolve().absolute()} already imported")
            return

        output_image_dir.mkdir(parents=True, exist_ok=True)
        number_of_stacked_images = image.shape[0]

        for i in range(number_of_stacked_images):
            # Squeeze empty image dimensions
            stacked_image = image[i].squeeze()

            output_image_path = output_image_dir / (output_image_name + f"_{i + 1:03d}.JPG")

            # Write new JPG image to MLAI archive
            self.write_image_to_disk(output_image_path, stacked_image)

    def write_image_to_disk(self, output_image_path: Path, image):
        """
        Writes an image to disk in JPG format.

        Args:
            output_image_path (Path): The path and filename of the output image file.
            image: The image to be written to disk.

        """
        self.logger.debug(f"Writing new JPG file: {output_image_path}")

        # Normalise CZI image
        normalised_image = cv2.normalize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_16U)

        # Write JPG to disk
        if cv2.imwrite(str(output_image_path), normalised_image, [cv2.IMWRITE_JPEG_QUALITY, 90]):
            self.logger.debug(f"Completed writing JPG file: {output_image_path}")
        else:
            self.logger.error(f"Could not write JPG image: {output_image_path}")

    def extract_video(self, image, output_video_name, output_video_dir, video_frame_rate):
        """
        Args:
            image: A numpy array representing the image data to extract the video from.
            output_video_name: A string representing the name of the output video file.
            output_video_dir: A string representing the directory to save the output video file.
            video_frame_rate: A float representing the frame rate of the output video.

        """
        self.logger.info(f"Extracting video...")
        output_file_path = output_video_dir / f"{output_video_name}.MP4"
        if output_file_path.is_file():
            self.logger.warning(f"File {output_file_path.resolve().absolute()} already imported")
            return

        output_video_dir.mkdir(parents=True, exist_ok=True)
        number_of_stacked_images = image.shape[0]
        output_video_path = output_video_dir / (output_video_name + ".MP4")  # Define path outside the loop

        try:
            # Initialize video writer
            out = cv2.VideoWriter(str(output_video_path), cv2.VideoWriter_fourcc(*"mp4v"), video_frame_rate, (image.shape[3], image.shape[2]))

            for i in range(number_of_stacked_images):
                # print(i)
                # Squeeze empty image dimensions
                stacked_image = image[i].squeeze()
                # Normalise CZI image
                normalised_image = cv2.normalize(
                    cv2.cvtColor(stacked_image, cv2.COLOR_BGR2RGB), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_16U
                )

                # Write the image to video file
                out.write(normalised_image)

            # Don't forget to release the video writer
            out.release()

            self.logger.info(f"Completed writing video to file: {output_video_path}")
        except Exception as e:
            self.logger.error(f"Unable to extract video due to error: {str(e)}")

    def extract_metadata(self, source_file, output_metadata_name, output_data_dir) -> float:
        """
        Extracts metadata from a given source file and writes it to a JSON file.

        Args:
            source_file: The path to the source file.
            output_metadata_name: The name of the output metadata file.
            output_data_dir: The directory where the output metadata file will be stored.

        Returns:
            The frame rate value extracted from the metadata.

        """
        self.logger.info(f"Extracting data...")
        output_file_path = output_data_dir / f"{output_metadata_name}.JSON"
        if output_file_path.is_file():
            self.logger.warning(f"File {output_file_path.resolve().absolute()} already imported")
            return

        output_data_dir.mkdir(parents=True, exist_ok=True)
        with czifile.CziFile(source_file) as czi:
            # Get CZI file metadata as dictionary
            metadata = czi.metadata(raw=False)

            output_metadata_path = output_data_dir / (output_metadata_name + ".JSON")

            self.write_metadata_to_disk(output_metadata_path, metadata)

            parameters = metadata["ImageDocument"]["Metadata"]["HardwareSetting"]["ParameterCollection"]
            if len(parameters) > 1 and float(parameters[1]["FrameRate"]["value"]) != 0.0:
                frame_rate = float(parameters[1]["FrameRate"]["value"])
                self.logger.info(f"Frame rate extracted from second parameter index is: {frame_rate}")
            else:
                frame_rate = float(parameters[0]["FrameRate"]["value"])
                self.logger.info(f"Frame rate extracted from first parameter index is: {frame_rate}")
            return frame_rate

    def write_metadata_to_disk(self, output_metadata_path: Path, data: Dict):
        """
        Write data to a JSON file on disk only if the file does not exist.
        Args:
            output_metadata_path (str): The file path where the data will be written to.
            data: The input dictionary that will be written to disk.
        """

        self.logger.debug(f"Writing new data to JSON file: {output_metadata_path}")
        # Write dictionary to JSON file
        try:
            with open(output_metadata_path, "w") as json_file:
                json.dump(data, json_file, indent=4, sort_keys=True)
            self.logger.debug(f"Completed writing data to JSON file: {output_metadata_path}")
        except Exception as e:
            self.logger.error(f"Could not write data to JSON file: {output_metadata_path}")
            self.logger.error(e)

    def _process(self, data_dir: Path, config: Dict[str, Any], **kwargs: dict):
        """
        Implementation of the Marimba process command for the Zeiss Axio Observer.

        Args:
            data_dir (Path): The directory where the data is stored.
            config (Dict[str, Any]): The configuration for the method.
            **kwargs (dict): Additional keyword arguments.

        Returns:
            None

        """

        all_images = data_dir.glob("**/*.JPG")

        # Initialize an empty set to hold unique parent directories
        unique_parent_dirs = set()

        # Iterate over the images generator
        for image_path in all_images:
            # Add the parent directory of each image to the set
            unique_parent_dirs.add(image_path.parent.parent)

        # Convert the set to a list to get a list of unique parent directories
        unique_parent_dirs_list = list(unique_parent_dirs)

        for base_image_sequence_dir in unique_parent_dirs_list:
            image_files = list(base_image_sequence_dir.glob("images/*.JPG"))
            image_files.sort()

            thumb_list = []
            output_thumbnails_directory = base_image_sequence_dir / "thumbnails"
            output_thumbnails_directory.mkdir(exist_ok=True)

            for jpg in image_files:
                output_filename = jpg.stem + "_THUMB" + jpg.suffix
                output_path = output_thumbnails_directory / output_filename
                if not output_path.exists():
                    self.logger.info(f"Generating thumbnail image: {output_path}")
                    image.resize_fit(jpg, 300, 300, output_path)
                    thumb_list.append(output_path)

            # TODO: Finalise name of this file - ask Carlie...
            thumbnail_overview_path = base_image_sequence_dir / "OVERVIEW.JPG"
            if not thumbnail_overview_path.exists():
                self.logger.info(f"Creating thumbnail overview image: {str(thumbnail_overview_path)}")
                image.create_grid_image(thumb_list, thumbnail_overview_path)

    def _compose(self, data_dirs: List[Path], configs: List[Dict[str, Any]], **kwargs: dict) -> Dict[Path, Tuple[Path, List[ImageData]]]:
        """
        Implementation of the Marimba package command for the Zeiss Axio Observer.

        Args:
            data_dirs (List[Path]): List of data directories to process.
            configs (List[Dict[str, Any]]): List of configurations for each data directory.
            **kwargs (dict): Additional keyword arguments.

        Returns:
            Dict[Path, Tuple[Path, List[ImageData]]]: Data mapping containing file paths, output file paths, and image data.

        """
        data_mapping = {}

        for data_dir, config in zip(data_dirs, configs):
            # List all files in the root directory recursively
            all_files = list(data_dir.glob("**/*"))

            # Split the files using list comprehensions
            jpg_files = [file for file in all_files if file.suffix.lower() == ".jpg"]
            ancillary_files = [file for file in all_files if file.suffix.lower() != ".jpg"]

            # Add ancillary files to data mapping
            for file_path in ancillary_files:
                if file_path.is_file():
                    output_file_path = file_path.relative_to(data_dir)
                    data_mapping[file_path] = output_file_path, None

            # Process and add jpg files to data mapping
            for file_path in jpg_files:
                if "_THUMB" not in file_path.name and "overview" not in file_path.name:
                    output_file_path = file_path.relative_to(data_dir)

                    # TODO: This information should live in the collection.yml config then this can roll through that list
                    # Set the image creators
                    image_creators = [
                        ImagePI(name="Chris Jackett", orcid="0000-0003-1132-1558"),
                        ImagePI(name="Ian Jameson", orcid=""),
                        ImagePI(name="Carlie Devine", orcid=""),
                        ImagePI(name="Emily", orcid=""),
                        ImagePI(name="CSIRO", orcid=""),
                    ]

                    # img = self.open_image(file_path)
                    # image_entropy = self.calculate_shannon_entropy(img)
                    # image_average_color = self.calculate_average_image_color(img)

                    # TODO: Don't sort iFDO in core Marimba

                    image_data_list = [
                        ImageData(
                            # iFDO core (required)
                            # TODO: Get image_datetime from the JSON file (AcquisitionDateAndTime)
                            image_datetime=datetime.strptime(Path(file_path).stem.split("_")[5], "%Y%m%dT%H%M%SZ"),
                            image_latitude=-42.88742265404429,
                            image_longitude=147.3387391318042,
                            image_altitude=None,
                            image_coordinate_reference_system="EPSG:4326",
                            image_coordinate_uncertainty_meters=None,
                            # image_context: Optional[str] = None
                            # image_project=row["survey_id"],
                            # image_event=f'{row["survey_id"]}_{row["deployment_number"]}',
                            image_platform=self.config.get("platform_id"),
                            # image_sensor=row["camera_name"],
                            image_uuid=str(uuid4()),
                            # image_hash_sha256=image_hash_sha256,
                            image_pi=ImagePI(name="Chris Jackett", orcid="0000-0003-1132-1558"),
                            image_creators=image_creators,
                            image_license="CC BY 4.0",
                            image_copyright="CSIRO",
                            # image_abstract=self.config.get("abstract"),
                            #
                            # # iFDO capture (optional)
                            image_acquisition=ImageAcquisition.PHOTO,
                            image_quality=ImageQuality.PRODUCT,
                            image_deployment=ImageDeployment.STATIONARY,
                            # image_navigation=ImageNavigation.RECONSTRUCTED,
                            # image_scale_reference=ImageScaleReference.NONE,
                            image_illumination=ImageIllumination.ARTIFICIAL_LIGHT,
                            image_pixel_mag=ImagePixelMagnitude.UM,
                            image_marine_zone=ImageMarineZone.LABORATORY,
                            image_spectral_resolution=ImageSpectralResolution.RGB,
                            image_capture_mode=ImageCaptureMode.MANUAL,
                            image_fauna_attraction=ImageFaunaAttraction.NONE,
                            # image_area_square_meter: Optional[float] = None
                            # image_meters_above_ground: Optional[float] = None
                            # image_acquisition_settings: Optional[dict] = None
                            # image_camera_yaw_degrees: Optional[float] = None
                            # image_camera_pitch_degrees: Optional[float] = None
                            # image_camera_roll_degrees: Optional[float] = None
                            # image_overlap_fraction=0,
                            image_datetime_format="%Y-%m-%d %H:%M:%S.%f",
                            # image_camera_pose: Optional[CameraPose] = None
                            # image_camera_housing_viewport: Optional[CameraHousingViewport] = None
                            # image_flatport_parameters: Optional[FlatportParameters] = None
                            # image_domeport_parameters: Optional[DomeportParameters] = None
                            # image_camera_calibration_model: Optional[CameraCalibrationModel] = None
                            # image_photometric_calibration: Optional[PhotometricCalibration] = None
                            # image_objective: Optional[str] = None
                            image_target_environment="Benthic habitat",
                            # image_target_timescale: Optional[str] = None
                            # image_spatial_constraints: Optional[str] = None
                            # image_temporal_constraints: Optional[str] = None
                            # image_time_synchronization: Optional[str] = None
                            image_item_identification_scheme="<imaging_system_id>_<magnification_factor>_<contrast_id>_<biological_stain_id>_<strain_id>_<iso_timestamp>_<image_id>.<ext>",
                            image_curation_protocol="Processed with Marimba"
                            #
                            # # iFDO content (optional)
                            # image_entropy=image_entropy,
                            # image_particle_count: Optional[int] = None
                            # image_average_color=image_average_color,
                            # image_mpeg7_colorlayout: Optional[List[float]] = None
                            # image_mpeg7_colorstatistics: Optional[List[float]] = None
                            # image_mpeg7_colorstructure: Optional[List[float]] = None
                            # image_mpeg7_dominantcolor: Optional[List[float]] = None
                            # image_mpeg7_edgehistogram: Optional[List[float]] = None
                            # image_mpeg7_homogenoustexture: Optional[List[float]] = None
                            # image_mpeg7_stablecolor: Optional[List[float]] = None
                            # image_annotation_labels: Optional[List[ImageAnnotationLabel]] = None
                            # image_annotation_creators: Optional[List[ImageAnnotationCreator]] = None
                            # image_annotations: Optional[List[ImageAnnotation]] = None
                        )
                    ]

                    data_mapping[file_path] = output_file_path, image_data_list

        return data_mapping

    def open_image(self, image_path):
        """
        Opens an image from the specified image file path.

        Args:
            image_path: A string representing the file path of the image to be opened.

        Returns:
            An instance of the `PIL.Image` class representing the opened image.

            Returns `None` if there was an error while loading the image.
        """
        try:
            img = Image.open(image_path)
            return img
        except IOError:
            print("Error: Unable to load image.")
            return None

    def calculate_shannon_entropy(self, img):
        """
        Calculates the Shannon entropy of an image.

        Args:
            img: The image to calculate entropy for. It should be a PIL.Image object.

        Returns:
            The Shannon entropy value of the image as a float. If the image is None, None is returned.
        """
        if img is None:
            return None

        # Convert to grayscale
        gray_img = img.convert("L")

        # Calculate the histogram
        histogram = np.array(gray_img.histogram(), dtype=np.float32)

        # Normalize the histogram to get probabilities
        probabilities = histogram / histogram.sum()

        # Filter out zero probabilities
        probabilities = probabilities[probabilities > 0]

        # Calculate Shannon entropy
        entropy = -np.sum(probabilities * np.log2(probabilities))

        return entropy

    def calculate_average_image_color(self, img):
        """
        Calculates the average color of an image.

        Args:
            img: The input image to calculate the average color from.

        Returns:
            A list of integers representing the average color of the image in RGB format.
            Each element in the list corresponds to the average intensity of the Red, Green, and Blue channels, respectively.

            Note: If the input image is None, None will be returned.
        """
        if img is None:
            return None

        # Convert the image to numpy array
        np_img = np.array(img)

        # Calculate the average color for each channel
        average_color = np.mean(np_img, axis=(0, 1))

        return list(map(int, average_color))

    def to_deg(self, value, loc):
        """
        Converts a given value in degrees, minutes, and seconds to decimal degrees.

        Args:
            value (float): The value in degrees, minutes, and seconds to be converted.
            loc (tuple): A tuple containing two values representing the positive and negative symbols for the location.

        Returns:
            tuple: A tuple containing four sub-tuples representing the converted decimal degrees.
                - The first sub-tuple contains the degrees value and a denominator of 1.
                - The second sub-tuple contains the minutes value and a denominator of 1.
                - The third sub-tuple contains the seconds value and a denominator of 1.
                - The fourth sub-tuple contains the location symbol.

        Note:
            The calculation of decimal degrees is based on the formula: decimal_degrees = degrees + (minutes/60) + (seconds/3600).

        Example:
            >>> obj = SomeClass()
            >>> obj.to_deg(45.75, ('N', 'S'))
            ((45, 1), (45, 1), (0, 1), 'N')

        """
        if value < 0:
            loc_value = loc[1]
        else:
            loc_value = loc[0]

        # TODO: Check why not decimal seconds
        abs_value = abs(value)
        deg = int(abs_value)
        min = int((abs_value - deg) * 60)
        sec = int((abs_value - deg - min / 60) * 3600)
        return (deg, 1), (min, 1), (sec, 1), loc_value
