"""Marimba Pipeline for the CSIRO ANACC Zeiss Axio microscopes."""  # noqa: N999

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import czifile
import numpy as np
from ifdo.models import (
    ImageAcquisition,
    ImageCaptureMode,
    ImageData,
    ImageDeployment,
    ImageFaunaAttraction,
    ImageIllumination,
    ImageMarineZone,
    ImagePI,
    ImagePixelMagnitude,
    ImageQuality,
    ImageSpectralResolution,
)
from marimba.core.pipeline import BasePipeline
from marimba.lib import image
from marimba.lib.concurrency import multithreaded_generate_image_thumbnails
from marimba.lib.decorators import multithreaded
from marimba.main import __version__
from numpy.typing import NDArray

EXPECTED_FILENAME_PARTS = 8
# strain_id, imaging_system_id, magnification_factor, contrast_id,
# channel_id, biological_stain_id, object_id, iso_timestamp

def is_valid_filename(filename: str) -> bool:
    """
    Validate if a given filename follows the expected format.

    This function checks if the provided filename adheres to a specific format by verifying that it contains
    exactly 8 parts separated by underscores.

    Args:
        filename (str): The filename to be validated.

    Returns:
        bool: True if the filename is valid (contains exactly 8 parts separated by underscores), False otherwise.
    """
    return len(filename.split("_")) == EXPECTED_FILENAME_PARTS


class ZeissAxioPipeline(BasePipeline):
    """
    Implements a pipeline for importing and processing data from Zeiss Axio Observer microscopy systems.

    This class extends BasePipeline to provide functionality for handling Zeiss Axio Observer microscopy data. It
    includes methods for importing, processing, and packaging data, as well as utilities for extracting images, videos,
    and metadata from CZI files.

    Attributes:
        VIDEO_DIMENSION_COUNT (int): The expected number of dimensions for video data in CZI files.

    Methods:
        get_pipeline_config_schema(): Returns the schema for the pipeline configuration.
        get_collection_config_schema(): Returns the schema for the collection configuration.
        _import(data_dir, source_path, config, **kwargs): Imports data from source paths to the data directory.
        process_source_file(source_file, data_dir, config): Processes a source file and extracts images and videos.
        get_output_dir_from_filename(data_dir, filename): Generates the output directory path based on filename
        attributes.
        czi_already_processed(output_image_name, output_base_dir): Checks if a CZI file has already been processed.
        extract_images(image, output_image_name, output_image_dir): Extracts and saves individual images from a stack.
        write_image_to_disk(output_image_path, image): Writes an image to disk in JPG format.
        extract_video(image, output_video_name, output_video_dir, video_frame_rate): Extracts and saves a video from
        images.
        extract_metadata(source_file, output_metadata_name, output_data_dir): Extracts metadata from a source file.
        write_metadata_to_disk(output_metadata_path, data): Writes metadata to a JSON file on disk.
        _process(data_dir, config, **kwargs): Processes data in the specified directory.
        _package(data_dir, config, **kwargs): Packages processed data for further use or distribution.
    """
    VIDEO_DIMENSION_COUNT = 5  # Number of dimensions in a CZI video file (time, size_c, size_z, size_y, size_x)

    @staticmethod
    def get_pipeline_config_schema() -> dict[str, str]:
        """
        Get the pipeline configuration schema for the PLAOS pipeline.

        Returns:
            dict: Configuration parameters for the pipeline
        """
        return {
            "project_pi": "Chris Jackett",
            "platform_id": "ZAO",
        }

    @staticmethod
    def get_collection_config_schema() -> dict[str, str]:
        """
        Get the collection configuration schema for the PLAOS pipeline.

        Returns:
            dict: Configuration parameters for the collection
        """
        return {
            "data_collector": "Chris Jackett",
            "collection_year": "2021",
        }

    def _import(
            self,
            data_dir: Path,
            source_path: Path,
            config: dict[str, Any],
            **kwargs: dict[str, Any],
    ) -> None:
        """
        Import data from source directories or files to a specified Marimba Collection.

        This function imports data from the provided source path to the Marimba Collection. It processes all files
        within the source directory recursively, using multithreading for improved performance. The function uses a
        configuration dictionary to customize the import process.

        Args:
            data_dir (Path): The directory where the imported data will be saved.
            source_path (Path): Path to the source directories or files to import.
            config (Dict[str, Any]): A dictionary containing configuration options for the import process.
            **kwargs (dict): Additional keyword arguments.

        Raises:
            FileNotFoundError: If the source_path does not exist.
            PermissionError: If there are insufficient permissions to read the source files or write to the data
            directory.
            ValueError: If the config dictionary contains invalid or incompatible options.
        """
        self.logger.debug(f"Importing data from {source_path} to {data_dir}")
        if not source_path.is_dir():
            return

        files_to_process = [source_file for source_file in source_path.glob("**/*") if source_file.is_file()]

        # Dynamically apply the multithreaded decorator
        @multithreaded()  # type: ignore[misc]
        def process_file(
                self: ZeissAxioPipeline,
                thread_num: str,  # noqa: ARG001
                item: Path,
                data_dir: Path,
                config: dict[str, Any],
        ) -> None:
            self.process_source_file(item, data_dir, config)

        # Call the decorated function
        process_file(
            self,
            items=files_to_process,
            data_dir=data_dir,
            config=config,
        )  # type: ignore[call-arg]

    def process_source_file(
            self,
            source_file: Path,
            data_dir: Path,
            config: dict[str, Any],
    ) -> None:
        """
        Processes a source file and extracts images and videos from a CZI file.

        Args:
            source_file (Path): The path to the source file.
            data_dir (Path): The directory where the output data will be stored.
            config (Dict[str, Any]): A dictionary containing the configuration parameters.

        """
        # Validate that self.config exists
        if self.config is None:
            raise ValueError("Pipeline configuration is missing")

        # Get platform_id from config and validate it
        platform_id = self.config.get("platform_id")
        if not isinstance(platform_id, str):
            raise TypeError("platform_id must be provided in the pipeline config and must be a string")

        contains_platform_id = f'_{self.config.get("platform_id")}' in source_file.name
        contains_collection_year = f'_{config.get("collection_year")}' in source_file.name
        is_czi_file = source_file.suffix.lower() == ".czi"

        if source_file.is_file() and is_czi_file and contains_collection_year and contains_platform_id:
            self.logger.debug(f"Processing file: {source_file.name}...")

            if not is_valid_filename(source_file.name):
                return

            output_base_dir = self.get_output_dir_from_filename(data_dir, source_file.stem)
            output_image_dir = output_base_dir / "images"
            output_video_dir = output_base_dir / "videos"
            output_data_dir = output_base_dir / "data"

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
            output_file_name = (
                f"{imaging_system_id}_"
                f"{magnification_factor}_"
                f"{contrast_id}_"
                f"{biological_stain_id}_"
                f"{strain_id}_"
                f"{iso_timestamp}"
            )

            # if not self.czi_already_processed(output_file_name, output_base_dir):
            self.logger.debug(f"Reading CZI file: {source_file}...")

            # Try to read CZI file and extract image frames
            try:
                image = czifile.imread(str(source_file))

                # Check that the CZI file is a video
                if len(image.shape) == self.VIDEO_DIMENSION_COUNT:
                    self.extract_images(image, output_file_name, output_image_dir)
                    video_frame_rate = self.extract_metadata(source_file, output_file_name, output_data_dir)
                    self.extract_video(image, output_file_name, output_video_dir, video_frame_rate)

            except Exception as e:
                self.logger.exception(f"Error extracting file {source_file.name}")
                self.logger.exception(e)

    def get_output_dir_from_filename(self, data_dir: Path, filename: str) -> Path:
        """
        Construct output directory path from filename components.

        This function parses a filename to extract various components and constructs a hierarchical directory structure
        based on these components. The resulting path is a combination of the provided data directory and subdirectories
        created from the filename's parsed elements.

        Args:
            data_dir (Path): The base directory path where the output directory will be created.
            filename (str): The filename containing components separated by underscores.

        Returns:
            Path: A Path object representing the constructed output directory.

        Raises:
            ValueError: If the filename does not contain the expected number of components (8) when split by
            underscores.
        """
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

    def czi_already_processed(
            self,
            output_image_name: str,
            output_base_dir: Path,
    ) -> bool:
        """
        Check if a CZI file has already been processed.

        This function determines whether a CZI file has been previously processed by checking for the existence of
        corresponding output files (image, video, and data) in their respective directories. It uses the provided
        output image name and base directory to construct the expected file paths.

        Args:
            output_image_name (str): The name of the output image file without extension.
            output_base_dir (Path): The base directory where processed files are stored.

        Returns:
            bool: True if the CZI file has already been processed (all output files exist), False otherwise.
        """
        output_image_path = output_base_dir / "images" / f"{output_image_name}_001.JPG"
        output_video_path = output_base_dir / "videos" / f"{output_image_name}.MP4"
        output_data_path = output_base_dir / "data" / f"{output_image_name}.JSON"

        if output_image_path.is_file() and output_video_path.is_file() and output_data_path.is_file():
            self.logger.warning(f"CZI file for {output_image_name} has already been imported")
            return True
        return False

    def extract_images(
            self,
            image: NDArray[np.uint16],
            output_image_name: str,
            output_image_dir: Path,
    ) -> None:
        """
        Extracts individual images from a stacked image array and saves them to disk in a specified directory.

        Args:
            image (np.ndarray): A stacked image array.
            output_image_name (str): The base name for the output images.
            output_image_dir (pathlib.Path): The directory where the output images will be saved.
        """
        self.logger.debug("Extracting images...")

        output_image_dir.mkdir(parents=True, exist_ok=True)
        number_of_stacked_images = image.shape[0]

        for i in range(number_of_stacked_images):
            # Squeeze empty image dimensions
            stacked_image = image[i].squeeze()

            output_image_path = output_image_dir / (output_image_name + f"_{i + 1:03d}.JPG")

            # Write new JPG image to MLAI archive
            self.write_image_to_disk(output_image_path, stacked_image)

    def write_image_to_disk(
            self,
            output_image_path: Path,
            image: NDArray[np.uint16],
    ) -> None:
        """
        Writes an image to disk in JPG format.

        Args:
            output_image_path (Path): The path and filename of the output image file.
            image: The image to be written to disk.

        """
        self.logger.debug(f"Writing new JPG file: {output_image_path}")

        # Convert to RGB and normalize
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Create output array with same type as input
        dst = np.empty_like(rgb_image)
        normalized_image = cv2.normalize(
            src=rgb_image,
            dst=dst,
            alpha=0.0,
            beta=255.0,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_16U,
        )

        # Write JPG to disk
        if cv2.imwrite(str(output_image_path), normalized_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90]):
            self.logger.debug(f"Completed writing JPG file: {output_image_path}")
        else:
            self.logger.exception(f"Could not write JPG image: {output_image_path}")

    def extract_video(
            self,
            image: NDArray[np.uint16],
            output_video_name: str,
            output_video_dir: Path,
            video_frame_rate: float,
    ) -> None:
        """
        Extract video from stacked images and save it to a file.

        This function takes a stack of images, processes them, and creates a video file. It normalizes each image,
        converts it to the appropriate color space, and writes it to the output video file. The function handles the
        creation of the output directory and logs the process, including any errors that may occur.

        Args:
            image (numpy.ndarray): A 4D array of stacked images with shape (num_frames, height, width, channels).
            output_video_name (str): The name of the output video file (without extension).
            output_video_dir (pathlib.Path): The directory where the output video will be saved.
            video_frame_rate (float): The frame rate of the output video.

        Returns:
            None

        Raises:
            Exception: If there's an error during video extraction or writing process.
        """
        self.logger.debug("Extracting video...")

        output_video_dir.mkdir(parents=True, exist_ok=True)
        number_of_stacked_images = image.shape[0]
        output_video_path = output_video_dir / (output_video_name + ".MP4")  # Define path outside the loop

        try:
            # Initialize video writer
            out = cv2.VideoWriter(
                str(output_video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore[attr-defined]
                int(video_frame_rate),
                (image.shape[3], image.shape[2]),
            )

            for i in range(number_of_stacked_images):
                stacked_image = image[i].squeeze()
                rgb_image = cv2.cvtColor(stacked_image, cv2.COLOR_BGR2RGB)

                # Create output array with same type as input
                dst = np.empty_like(rgb_image)
                normalized_image = cv2.normalize(
                    src=rgb_image,
                    dst=dst,
                    alpha=0.0,
                    beta=255.0,
                    norm_type=cv2.NORM_MINMAX,
                    dtype=cv2.CV_16U,
                )

                out.write(normalized_image)

            # Don't forget to release the video writer
            out.release()

            self.logger.debug(f"Completed writing video to file: {output_video_path}")
        except Exception as e:
            self.logger.exception(f"Unable to extract video due to error: {e!s}")

    def extract_metadata(
            self,
            source_file: Path,
            output_metadata_name: str,
            output_data_dir: Path,
    ) -> float:
        """
        Extracts metadata from a given source file and writes it to a JSON file.

        Args:
            source_file: The path to the source file.
            output_metadata_name: The name of the output metadata file.
            output_data_dir: The directory where the output metadata file will be stored.

        Returns:
            The frame rate value extracted from the metadata.

        """
        self.logger.debug("Extracting data...")
        output_data_dir.mkdir(parents=True, exist_ok=True)

        with czifile.CziFile(source_file) as czi:
            metadata = czi.metadata(raw=False)
            output_metadata_path = output_data_dir / f"{output_metadata_name}.JSON"
            self.write_metadata_to_disk(output_metadata_path, metadata)

            parameters = (
                metadata.get("ImageDocument", {})
                .get("Metadata", {})
                .get("HardwareSetting", {})
                .get("ParameterCollection", [])
            )

            # Initialize default frame rate value
            frame_rate = 10.0

            # The ZAP metadata provides a parameters dictionary, whereas the ZAO provides a list
            if isinstance(parameters, dict):
                try:
                    temp_frame_rate = float(parameters.get("FrameRate", {}).get("value", 0))
                    if temp_frame_rate > 0:
                        frame_rate = temp_frame_rate
                except ValueError:
                    pass
            # If parameters is a list
            elif isinstance(parameters, list):
                # Search for a positive frame rate value in parameters
                for param in parameters:
                    try:
                        temp_frame_rate = float(param.get("FrameRate", {}).get("value", 0))
                        if temp_frame_rate > 0:
                            frame_rate = temp_frame_rate
                            break
                    except ValueError:
                        # In case of conversion error, ignore the current value and continue
                        continue

            self.logger.debug(f"Extracted frame rate is: {frame_rate}")
            return frame_rate

    def write_metadata_to_disk(self, output_metadata_path: Path, data: dict[str, Any]) -> None:
        """
        Write data to a JSON file on disk only if the file does not exist.

        Args:
            output_metadata_path (str): The file path where the data will be written to.
            data: The input dictionary that will be written to disk.
        """
        self.logger.debug(f"Writing new data to JSON file: {output_metadata_path}")
        # Write dictionary to JSON file
        try:
            with Path.open(output_metadata_path, "w") as json_file:
                json.dump(data, json_file, indent=4, sort_keys=True)
            self.logger.debug(f"Completed writing data to JSON file: {output_metadata_path}")
        except Exception as e:
            self.logger.exception(f"Could not write data to JSON file: {output_metadata_path}")
            self.logger.exception(e)

    # ruff: noqa: ARG002
    def _process(
            self,
            data_dir: Path,
            config: dict[str, Any],
            **kwargs: dict[str, Any],
    ) -> None:
        """
        Implementation of the Marimba process command for the Zeiss Axio Observer.

        Args:
            data_dir (Path): The directory where the data is stored.
            config (Dict[str, Any]): The configuration for the method.
            **kwargs (dict): Additional keyword arguments.

        Returns:
            None

        """
        self.logger.debug(f"Processing data in {data_dir}...")

        all_images = data_dir.glob("**/images/*.JPG")

        # Initialize an empty set to hold unique parent directories
        unique_parent_dirs = set()

        # Iterate over the images generator
        for image_path in all_images:
            # Add the parent directory of each image to the set
            unique_parent_dirs.add(image_path.parent.parent)

        # Convert the set to a list to get a list of unique parent directories
        unique_parent_dirs_list = list(unique_parent_dirs)

        for base_image_sequence_dir in unique_parent_dirs_list:
            image_list = list(base_image_sequence_dir.glob("images/*.JPG"))

            # Generate thumbnails using multithreading
            thumbnail_list = multithreaded_generate_image_thumbnails(
                self,
                image_list=image_list,
                output_directory=base_image_sequence_dir / "thumbnails",
            )

            # Create an overview image from the generated thumbnails
            thumbnail_overview_path = base_image_sequence_dir / "OVERVIEW.JPG"
            image.create_grid_image(thumbnail_list, thumbnail_overview_path)

    # ruff: noqa: ARG002
    def _package(
            self,
            data_dir: Path,
            config: dict[str, Any],
            **kwargs: dict[str, Any],
    ) -> dict[Path, tuple[Path, ImageData | None, dict[str, Any] | None]]:
        """
        Implementation of the Marimba package command for the Zeiss Axio Observer.

        Args:
            data_dir (Path): Data directory to process.
            config (Dict[str, Any]): Configuration for each data directory.
            **kwargs (dict): Additional keyword arguments.

        Returns:
            Dict[Path, Tuple[Path, List[ImageData]]]: Data mapping containing file paths, output file paths, and image
            data.

        """
        data_mapping: dict[Path, tuple[Path, list[ImageData] | None, dict[str, Any] | None]] = {}

        # List all files in the root directory recursively
        all_files = list(data_dir.glob("**/*"))

        # Split the files using list comprehensions
        jpg_files = [file for file in all_files if file.suffix.lower() == ".jpg"]
        ancillary_files = [file for file in all_files if file.suffix.lower() != ".jpg"]

        # Add ancillary files to data mapping
        self.logger.debug("Adding ancillary files to data mapping")
        for file_path in ancillary_files:
            if file_path.is_file():
                output_file_path = file_path.relative_to(data_dir)
                data_mapping[file_path] = output_file_path, None, None

        # Process and add jpg files to data mapping
        self.logger.debug("Processing and adding jpg files to data mapping")
        for file_path in jpg_files:
            output_file_path = file_path.relative_to(data_dir)
            if file_path.parent.name == "images":
                # Set the image pi and creators
                image_pi = ImagePI(name="Chris Jackett", orcid="0000-0003-1132-1558")
                image_creators = [
                    image_pi,
                    ImagePI(name="Ian Jameson", orcid=""),
                    ImagePI(name="Carlie Devine", orcid=""),
                    ImagePI(name="Emily Gumina", orcid=""),
                    ImagePI(name="CSIRO", orcid=""),
                ]

                # Validate that self.config exists
                if self.config is None:
                    raise ValueError("Pipeline configuration is missing")

                # Get platform_id from config and validate it
                platform_id = self.config.get("platform_id")
                if not isinstance(platform_id, str):
                    raise TypeError("platform_id must be provided in the pipeline config and must be a string")

                # ruff: noqa: ERA001
                image_data_list = ImageData(
                    # iFDO core
                    # TODO(<cjackett>): Get image_datetime from the JSON file (AcquisitionDateAndTime)
                    image_datetime=datetime.strptime(Path(file_path).stem.split("_")[5], "%Y%m%dT%H%M%SZ")
                    .replace(tzinfo=timezone.utc),
                    image_latitude=-42.88742265404429,
                    image_longitude=147.3387391318042,
                    image_altitude=None,
                    image_coordinate_reference_system="EPSG:4326",
                    image_coordinate_uncertainty_meters=None,
                    # image_context: Optional[str] = None
                    # image_project=row["survey_id"],
                    # image_event=f'{row["survey_id"]}_{row["deployment_number"]}',
                    image_platform=platform_id,
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
                    image_curation_protocol=f"Processed with Marimba v{__version__}",
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

                data_mapping[file_path] = output_file_path, [image_data_list], None

            else:
                data_mapping[file_path] = output_file_path, None, None

        return data_mapping
