# Zeiss Axio Pipeline

A Marimba Pipeline for automated processing of microscopy imagery collections from the CSIRO Australian National Algae 
Culture Collection (ANACC). The Pipeline specializes in extracting and processing phytoplankton microscopy videos 
captured using Zeiss Axio Observer and Axio Plan microscopes while preserving comprehensive experimental metadata.


## Overview

The Zeiss Axio Pipeline is designed to process phytoplankton microscopy data collected during CSIRO ANACC imaging 
campaigns between 2021 and 2023. It handles data from both the Zeiss Axio Observer (ZAO) and Zeiss Axio Plan (ZAP) 
microscopes, which capture short high-resolution video clips of phytoplankton specimens under various magnification and 
contrast settings.

Key capabilities include:

- Automated extraction of individual frames from microscopy video clips
- Generation of MP4 videos from extracted JPG images
- Creation of overview images for visual validation
- Integration of comprehensive metadata including magnification, contrast mode, and timestamps
- Preservation of original CZI metadata in JSON format
- Generation of FAIR-compliant datasets with embedded metadata


## Requirements

The Zeiss Axio Pipeline is built on the [Marimba](https://github.com/csiro-fair/marimba) framework which includes most 
necessary dependencies. Additional requirements include:
- czifile


## Installation

Create a new Marimba project and add the Zeiss Axio Pipeline:

```bash
marimba new project my-microscopy-project
cd my-microscopy-project

# Install ZAO pipeline
marimba new pipeline ZAO https://github.com/csiro-fair/zeiss-axio-pipeline.git \
--config '{"platform_id": "ZAO", "image_sensor": "AxioCam HR R3"}'

# Install ZAP pipeline
marimba new pipeline ZAP https://github.com/csiro-fair/zeiss-axio-pipeline.git \
--config '{"platform_id": "ZAP", "image_sensor": "Axiocam 506"}'
```


## Configuration

### Pipeline Configuration
The Pipeline requires:
- `platform_id`: Microscope identifier (either "ZAO" or "ZAP")
- `image_sensor`: Camera model ("AxioCam HR R3" for ZAO or "Axiocam 506" for ZAP)

### Collection Configuration
Each Collection requires:
- `collection_year`: Year of data collection (e.g., "2021", "2022", "2023")


## Usage

### Importing

Import collections with year-specific configurations:

```bash
marimba import 2021 /path/to/source/images \
--config '{"collection_year": "2021"}' \
--max-workers=1
```

For a multi-year imaging campaign, each year can be imported separately:

```bash
# Import multiple years
marimba import 2021 /path/to/2021/images --config '{"collection_year": "2021"}'
marimba import 2022 /path/to/2022/images --config '{"collection_year": "2022"}'
marimba import 2023 /path/to/2023/images --config '{"collection_year": "2023"}'
```

During importing, the Zeiss Axio Pipeline:
1. Creates a hierarchical directory structure by magnification, contrast mode, stain, strain, and timestamp
2. Extracts individual frames from CZI videos
3. Extracts CZI metadata into JSON format
4. Generates MP4 videos from frame sequences

### Source Data Structure

The Pipeline expects CZI files with standardized naming:

```
<strain_id>_<imaging_system_id>_<magnification_factor>_<contrast_id>_<channel_id>_<biological_stain_id>_<object_id>_<iso_timestamp>.czi
```

Example: `CS1197_ZAO_X400_BF_NA_TYL_001_20210805T061330Z.czi`

### Processing

```bash
marimba process
```

During processing, the Zeiss Axio Pipeline:
1. Creates thumbnail images
2. Assembles overview grids for quality control

### Packaging

```bash
marimba package CAPMD_2021 \
--collection-name 2021 \
--operation link \
--version 1.0 \
--contact-name "Keiko Abe" \
--contact-email "keiko.abe@email.com" \
--zoom 17
```

The `--operation link` flag creates hard links instead of copying files, optimizing storage for large datasets.


## Processed Data Structure

```
CAPMD_2021/                                         # Root dataset directory
├── data/                                           # Directory containing all processed data
│   └── [ZAO/ZAP]/                                  # Platform-specific data directory
│       └── [X100|X200|X400|X630]/                  # Magnification directories
│           └── [BF|DIC|PC]/                        # Contrast setting directories
│               └── [IDN|NA|TYL]/                   # Stain directories
│                   └── [CS*]/                      # Strain directories
│                       └── [TIMESTAMP]/            # Timestamp-based directories
│                           ├── data/               # JSON metadata files
│                           ├── images/             # Extracted image frames
│                           ├── thumbnails/         # Image thumbnails
│                           ├── videos/             # Generated MP4 files
│                           └── *_OVERVIEW.JPG      # Grid overview image
├── logs/                                           # Directory containing all processing logs
│   ├── pipelines/                                  # Pipeline-specific logs
│   │   ├── ZAO.log                                 # Logs from ZAO Pipeline
│   │   └── ZAP.log                                 # Logs from ZAP Pipeline
│   ├── dataset.log                                 # Dataset packaging logs
│   └── project.log                                 # Overall project processing logs
├── pipelines/                                      # Directory containing pipeline code
│   ├── ZAO/                                        # ZAO Pipeline directory
│   │   ├── repo/                                   # Pipeline source code repository
│   │   │   ├── LICENSE                             # Pipeline license file
│   │   │   ├── README.md                           # Pipeline README file
│   │   │   ├── requirements.txt                    # Pipeline dependencies
│   │   │   └── zeiss_axio.pipeline.py              # Pipeline implementation
│   │   └── pipeline.yml                            # Pipeline configuration
│   └── ZAP/                                        # ZAP Pipeline directory
├── ifdo.yml                                        # Dataset-level iFDO metadata file
├── manifest.txt                                    # File manifest with SHA256 hashes
├── map.png                                         # Spatial visualization of dataset
├── strain_list.csv                                 # List of imaged strains and associated metadata
└── summary.md                                      # Dataset summary and statistics
```


## Metadata

The Zeiss Axio Pipeline captures comprehensive metadata including:

### Technical Metadata
- Microscope configuration
- Camera settings
- Image acquisition parameters
- Processing parameters
- Quality metrics

### Sample Metadata
- Strain identifiers
- Magnification levels
- Contrast modes
- Staining information
- Collection timestamps
- Facility location coordinates

### File-Specific Data
- Frame extraction details
- Video generation parameters
- Overview image composition
- Directory structure context
- Processing history

All metadata is standardized using the iFDO schema (v2.1.0) and embedded in both image EXIF tags and dataset-level files.


## Contributors

The Zeiss Axio Pipeline was developed by:
- Christopher Jackett (CSIRO)
- Ian Jameson (CSIRO)
- Carlie Devine (CSIRO)
- Ros Watson (CSIRO)
- Peter Thrall (CSIRO)
- Emily Gumina (CSIRO)


## License

The Zeiss Axio Pipeline is distributed under the [CSIRO BSD/MIT](LICENSE) license.


## Contact

For inquiries related to this repository, please contact:

- **Christopher Jackett**  
  *Software Engineer, CSIRO*  
  Email: [chris.jackett@csiro.au](mailto:chris.jackett@csiro.au)
