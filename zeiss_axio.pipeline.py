"""Marimba Pipeline for the CSIRO ANACC Zeiss Axio microscopes."""  # noqa: INP001

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
    ImageContext,
    ImageCreator,
    ImageData,
    ImageDeployment,
    ImageFaunaAttraction,
    ImageIllumination,
    ImageLicense,
    ImageMarineZone,
    ImageNavigation,
    ImagePI,
    ImagePixelMagnitude,
    ImageQuality,
    ImageSpectralResolution,
)
from numpy.typing import NDArray

from marimba.core.pipeline import BasePipeline
from marimba.core.schemas.base import BaseMetadata
from marimba.core.schemas.ifdo import iFDOMetadata
from marimba.core.utils.paths import format_path_for_logging
from marimba.lib import image
from marimba.lib.concurrency import multithreaded_generate_image_thumbnails
from marimba.lib.decorators import multithreaded
from marimba.main import __version__

# Number of underscore-separated parts in microscopy filenames.
# Required format: strain_id, imaging_system_id, magnification_factor, contrast_id,
# channel_id, biological_stain_id, object_id, iso_timestamp
EXPECTED_FILENAME_PARTS = 8


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
    Marimba Pipeline for the CSIRO ANACC Zeiss Axio microscopes.

    This class extends BasePipeline to provide functionality for handling Zeiss Axio Observer microscopy data. It
    includes methods for importing, processing, and packaging data, as well as utilities for extracting images, videos,
    and metadata from CZI files.

    Attributes:
        VIDEO_DIMENSION_COUNT (int): The expected number of dimensions for video data in CZI files.

    Methods:
        get_pipeline_config_schema(): Get the pipeline configuration schema.
        get_collection_config_schema(): Get the collection configuration schema.
        _import(data_dir, source_path, config, **kwargs): Import data from source paths to the data directory.
        process_source_file(source_file, data_dir, config): Process a source file and extract images and videos.
        get_output_dir_from_filename(data_dir, filename): Generate output directory path based on filename attributes.
        extract_images(image, output_image_name, output_image_dir): Extract and save individual images from a stack.
        write_image_to_disk(output_image_path, image): Write an image to disk in JPG format.
        extract_video(image, output_video_name, output_video_dir, video_frame_rate): Extract and save a video from
            images.
        extract_metadata(source_file, output_metadata_name, output_data_dir): Extract metadata from a source file.
        write_metadata_to_disk(output_metadata_path, data): Write metadata to a JSON file on disk.
        _process(data_dir, config, **kwargs): Process data in the specified directory.
        _package(data_dir, config, **kwargs): Package processed data for further use or distribution.
    """
    VIDEO_DIMENSION_COUNT = 5  # Number of dimensions in a CZI video file (time, size_c, size_z, size_y, size_x)

    def __init__(
        self,
        root_path: str | Path,
        config: dict[str, Any] | None = None,
        *,
        dry_run: bool = False,
    ) -> None:
        """
        Initialize a new Pipeline instance.

        Args:
            root_path (str | Path): Base directory path where the pipeline will store its data and configuration files.
            config (dict[str, Any] | None, optional): Pipeline configuration dictionary. If None, default configuration
             will be used. Defaults to None.
            dry_run (bool, optional): If True, prevents any filesystem modifications. Useful for validation and testing.
             Defaults to False.
        """
        super().__init__(
            root_path,
            config,
            dry_run=dry_run,
            metadata_class=iFDOMetadata,
        )

    @staticmethod
    def get_pipeline_config_schema() -> dict[str, Any]:
        """
        Get the pipeline configuration schema for the PLAOS pipeline.

        Returns:
            dict: Configuration parameters for the pipeline
        """
        return {
            "platform_id": "ZAO",
            "image_sensor": "AxioCam HR R3",
        }

    @staticmethod
    def get_collection_config_schema() -> dict[str, Any]:
        """
        Get the collection configuration schema for the PLAOS pipeline.

        Returns:
            dict: Configuration parameters for the collection
        """
        return {
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
        if not source_path.is_dir():
            return

        files_to_process = [source_file for source_file in source_path.glob("**/*") if source_file.is_file()]

        # Dynamically apply the multithreaded decorator
        @multithreaded(max_workers=6)  # type: ignore[misc]
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
            if not is_valid_filename(source_file.name):
                return

            # Extract strain_id from filename
            strain_id = source_file.stem.split("_")[0]

            # Skip processing if strain_id is "MSA"
            if strain_id == "MSA":
                self.logger.debug(f"Skipping MSA strain file {source_file.name}")
                return

            output_base_dir = self.get_output_dir_from_filename(data_dir, source_file.stem)
            output_data_dir = output_base_dir / "data"
            output_image_dir = output_base_dir / "images"
            output_video_dir = output_base_dir / "videos"

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

            # Try to read CZI file and extract image frames
            try:
                image = czifile.imread(str(source_file))

                # Check that the CZI file is a video
                if len(image.shape) == self.VIDEO_DIMENSION_COUNT:
                    self.logger.debug(
                        f"Started extracting images from CZI file "
                        f"{format_path_for_logging(source_file, Path(self._root_path).parents[2])}",
                    )
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
        try:
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
                self.logger.debug(
                    f"Created image {format_path_for_logging(output_image_path, Path(self._root_path).parents[2])}",
                )
            else:
                self.logger.exception(
                    f"Failed to create {format_path_for_logging(output_image_path, Path(self._root_path).parents[2])}",
                )
        except Exception as e:
            self.logger.exception(
                f"Error creating {format_path_for_logging(output_image_path, Path(self._root_path).parents[2])}: {e}",
            )

    def extract_video(
        self,
        image: NDArray[np.uint16],
        output_video_name: str,
        output_video_dir: Path,
        video_frame_rate: float,
    ) -> None:
        """
        Extract video from stacked images and save it to a file.

        This function takes a stack of images, processes them, and creates a video file.
        It properly handles color conversion and normalization to maintain image quality.

        Args:
            image (numpy.ndarray): A 4D array of stacked images with shape (num_frames, height, width, channels).
            output_video_name (str): The name of the output video file (without extension).
            output_video_dir (pathlib.Path): The directory where the output video will be saved.
            video_frame_rate (float): The frame rate of the output video.
        """
        output_video_dir.mkdir(parents=True, exist_ok=True)
        number_of_stacked_images = image.shape[0]
        output_video_path = output_video_dir / (output_video_name + ".MP4")

        try:
            # Get the correct dimensions from the input image
            height = image.shape[2]  # Original height
            width = image.shape[3]  # Original width

            # Initialize video writer with mp4v codec
            out = cv2.VideoWriter(
                str(output_video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                int(video_frame_rate),
                (width, height),
            )

            for i in range(number_of_stacked_images):
                # Extract and squeeze the frame to remove singleton dimensions
                frame = image[i].squeeze()

                # Convert from uint16 to uint8 with proper scaling
                frame_normalized = cv2.normalize(
                    frame,
                    None,
                    alpha=0.0,
                    beta=255.0,
                    norm_type=cv2.NORM_MINMAX,
                    dtype=cv2.CV_8U,
                )

                # Convert from RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame_normalized, cv2.COLOR_RGB2BGR)

                # Write the frame
                out.write(frame_bgr)

            # Release the video writer
            out.release()
            self.logger.debug(
                f"Created video {format_path_for_logging(output_video_path, Path(self._root_path).parents[2])}",
            )

        except Exception as e:
            self.logger.exception(f"Error during video extraction: {e}")
            # Make sure to release the writer even if there's an error
            if "out" in locals():
                out.release()
            raise

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

            self.logger.debug(f"Frame rate extracted from CZI metadata is {frame_rate} fps")
            return frame_rate

    def write_metadata_to_disk(self, output_metadata_path: Path, data: dict[str, Any]) -> None:
        """
        Write data to a JSON file on disk only if the file does not exist.

        Args:
            output_metadata_path (str): The file path where the data will be written to.
            data: The input dictionary that will be written to disk.
        """
        try:
            with Path.open(output_metadata_path, "w") as json_file:
                json.dump(data, json_file, indent=4, sort_keys=True)
            self.logger.debug(
                f"Created metadata {format_path_for_logging(output_metadata_path, Path(self._root_path).parents[2])}",
            )
        except Exception as e:
            self.logger.exception(
                f"Error creating metadata "
                f"{format_path_for_logging(output_metadata_path, Path(self._root_path).parents[2])}: {e}",
            )

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

            # Create overview image name from the first image's identifiers
            first_image = image_list[0]
            identifiers = first_image.stem.rsplit("_", 1)[0].split("_")
            overview_name = (
                f"{identifiers[0]}_"
                f"{identifiers[1]}_"
                f"{identifiers[2]}_"
                f"{identifiers[3]}_"
                f"{identifiers[4]}_"
                f"{identifiers[5]}_"
                f"OVERVIEW.JPG"
            )

            # Generate the overview image
            overview_path = base_image_sequence_dir / overview_name
            image.create_grid_image(thumbnail_list, overview_path)
            self.logger.debug(
                f"Created overview image {format_path_for_logging(overview_path, Path(self._root_path).parents[2])}",
            )

    # ruff: noqa: ARG002
    def _package(
        self,
        data_dir: Path,
        config: dict[str, Any],
        **kwargs: dict[str, Any],
    ) -> dict[Path, tuple[Path, list[BaseMetadata] | None, dict[str, Any] | None]]:
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
        # Initialise an empty dictionary to store file mappings
        data_mapping: dict[Path, tuple[Path, list[BaseMetadata] | None, dict[str, Any] | None]] = {}

        # List all files in the root directory recursively
        all_files = list(data_dir.glob("**/*"))

        # Split the files using list comprehensions
        media_files = [file for file in all_files if file.suffix.lower() in {".jpg", ".mp4"}]
        ancillary_files = [file for file in all_files if file.suffix.lower() not in {".jpg", ".mp4"}]

        # Add ancillary files to data mapping
        for file_path in ancillary_files:
            if file_path.is_file():
                output_file_path = file_path.relative_to(data_dir)
                data_mapping[file_path] = output_file_path, None, None
        if len(ancillary_files):
            self.logger.debug(f"Added {len(ancillary_files)} ancillary files to data mapping")

        # Process and add jpg files to data mapping
        for file_path in media_files:
            output_file_path = file_path.relative_to(data_dir)
            parent_dir_name = file_path.parent.name

            # Only process files in images or videos directories
            if parent_dir_name not in {"images", "videos"}:
                data_mapping[file_path] = output_file_path, None, None
                continue

            # Set the image pi and creators
            image_pi = ImagePI(name="Christopher Jackett", uri="https://orcid.org/0000-0003-1132-1558")
            image_creators = [
                ImageCreator(name="Christopher Jackett", uri="https://orcid.org/0000-0003-1132-1558"),
                ImageCreator(name="Ian Jameson", uri="https://orcid.org/0000-0002-1365-9723"),
                ImageCreator(name="Carlie Devine", uri="https://orcid.org/0000-0003-1397-7446"),
                ImageCreator(name="Ros Watson", uri="https://orcid.org/0009-0005-9604-3658"),
                ImageCreator(name="Peter H. Thrall", uri="https://orcid.org/0000-0003-1670-4240"),
                ImageCreator(name="CSIRO", uri="https://www.csiro.au"),
            ]

            # Add Emily to creators if collection year is 2023
            collection_year = config.get("collection_year")
            if collection_year == "2023":
                # Insert Emily at second position
                image_creators.insert(
                    1,
                    ImageCreator(name="Emily Gumina", uri="https://orcid.org/0009-0004-0169-9770")
                )

            # Validate that self.config exists
            if self.config is None:
                raise ValueError("Pipeline configuration is missing")

            # Get platform_id from config and validate it
            filename_split = Path(file_path).stem.split("_")
            image_datetime = datetime.strptime(
                filename_split[5], "%Y%m%dT%H%M%SZ",
            ).replace(tzinfo=timezone.utc)
            platform_id = self.config.get("platform_id")
            if not isinstance(platform_id, str):
                raise TypeError("platform_id must be provided in the pipeline config and must be a string")
            image_platform = ImageContext(name=platform_id)

            # Create ImageContext and ImageLicense objects
            image_context = ImageContext(name=(
                "The CSIRO Machine Learning and Artificial Intelligence Future Science Platform (MLAI FSP) algae "
                "detection project was established to develop automated methods for identifying phytoplankton "
                "species in mixed communities using deep learning techniques. The project aimed to improve the "
                "speed and accuracy of phytoplankton identification in images to support Australian aquaculture, "
                "fisheries, and environmental management, with applications ranging from monitoring ecosystem health "
                "to identifying Harmful Algal Bloom (HAB) species."
                ),
                uri="https://research.csiro.au/mlai-fsp",
            )
            image_project = ImageContext(name=(
                "The CSIRO Australian Phytoplankton Microscopy Dataset (CAPMD) - A comprehensive microscopy imaging "
                "campaign of living cultures of phytoplankton species from the Australian National Algae Culture "
                "Collection (ANACC), designed to create a high-quality training dataset for machine learning "
                "applications."
                ),
                uri="https://www.csiro.au/anacc",
            )
            image_event = ImageContext(name=file_path.stem)
            image_sensor = ImageContext(name=self.config.get("image_sensor"))
            image_license = ImageLicense(name="CC BY-NC 4.0", uri="https://creativecommons.org/licenses/by-nc/4.0")
            image_abstract = (
                "The CSIRO Australian Phytoplankton Microscopy Dataset (CAPMD) is a comprehensive collection of "
                "high-quality microscopy images documenting the morphological diversity of phytoplankton species from "
                "the Australian National Algae Culture Collection (ANACC). Images were acquired using ZEISS Axio "
                "Observer and Axio Plan microscopes under standardised laboratory conditions at the CSIRO Battery "
                "Point site in Hobart, Tasmania. Live specimens were prepared in suspension and systematically imaged "
                "by capturing multiple short videos (50-200 frames) using a focal rolling technique that involved "
                "maneuvering the objective lens through different focal planes of the sample to capture key taxonomic "
                "features and cellular structures. The imaging protocol implemented multiple modalities including "
                "bright field, differential interference contrast, and phase contrast microscopy at magnifications "
                "ranging from 100x to 1000x. Specimens were selectively treated with Tylose to immobilise highly "
                "motile cells while maintaining their structural integrity, or Lugol's (Iodine) solution as a fixing "
                "agent to capture fixed cell imagery, which is a standard approach in phytoplankton identification. "
                "The imaging protocol was specifically designed to support the development of automated phytoplankton "
                "identification systems using machine learning techniques, with careful attention given to image "
                "quality, consistency, and comprehensive coverage of taxonomically significant features across "
                "different imaging conditions. This systematic imaging campaign successfully balanced the need for "
                "high-quality, representative samples with efficient, high-volume image capture, creating a robust "
                "dataset for advancing automated phytoplankton identification methods."
            )

            # ruff: noqa: ERA001
            image_data = ImageData(
                # iFDO core
                image_datetime=image_datetime,
                image_latitude=-42.88742265404429,
                image_longitude=147.3387391318042,
                image_altitude_meters=None,
                image_coordinate_reference_system="EPSG:4326",
                image_coordinate_uncertainty_meters=None,
                image_context=image_context,
                image_project=image_project,
                image_event=image_event,
                image_platform=image_platform,
                image_sensor=image_sensor,
                image_uuid=str(uuid4()),
                image_pi=image_pi,
                image_creators=image_creators,
                image_license=image_license,
                image_copyright="CSIRO",
                image_abstract=image_abstract,

                # iFDO capture (optional)
                image_acquisition=ImageAcquisition.PHOTO,
                image_quality=ImageQuality.PRODUCT,
                image_deployment=ImageDeployment.STATIONARY,
                image_navigation=ImageNavigation.RECONSTRUCTED,
                # image_scale_reference=None,
                image_illumination=ImageIllumination.ARTIFICIAL_LIGHT,
                image_pixel_magnitude=ImagePixelMagnitude.UM,
                image_marine_zone=ImageMarineZone.LABORATORY,
                image_spectral_resolution=ImageSpectralResolution.RGB,
                image_capture_mode=ImageCaptureMode.MANUAL,
                image_fauna_attraction=ImageFaunaAttraction.NONE,
                # image_area_square_meter=None,
                # image_meters_above_ground=None,
                # image_acquisition_settings=None,
                # image_camera_yaw_degrees=None,
                # image_camera_pitch_degrees=None,
                # image_camera_roll_degrees=None,
                # image_overlap_fraction=0,
                image_datetime_format="%Y-%m-%d %H:%M:%S.%f",
                # image_camera_pose=None,
                # image_camera_housing_viewport=None,
                # image_flatport_parameters=None,
                # image_domeport_parameters=None,
                # image_camera_calibration_model=None,
                # image_photometric_calibration=None,
                # image_objective=None,
                # image_target_environment=None,
                # image_target_timescale=None,
                # image_spatial_constraints=None,
                # image_temporal_constraints=None,
                # image_time_synchronization=None,
                image_item_identification_scheme="<imaging_system_id>_<magnification_factor>_<contrast_id>_<biological_stain_id>_<strain_id>_<iso_timestamp>_<image_id>.<ext>",
                image_curation_protocol=f"Processed with Marimba v{__version__}",

                # iFDO content (optional)
                # image_entropy=0.0,
                # image_particle_count=None,
                # image_average_color=[0, 0, 0],
                # image_mpeg7_colorlayout=None,
                # image_mpeg7_colorstatistics=None,
                # image_mpeg7_colorstructure=None,
                # image_mpeg7_dominantcolor=None,
                # image_mpeg7_edgehistogram=None,
                # image_mpeg7_homogenoustexture=None,
                # image_mpeg7_stablecolor=None,
                # image_annotation_labels=None,
                # image_annotation_creators=None,
                # image_annotations=None,
            )

            metadata = self._metadata_class(image_data)
            data_mapping[file_path] = output_file_path, [metadata], None

        if len(media_files):
            self.logger.debug(f"Added {len(media_files)} media files to data mapping")

        return data_mapping
