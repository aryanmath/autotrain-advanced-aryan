# Main Repo (AutoTrain Advanced) - Complete Deep-Dive Map

**Repository:** [https://github.com/huggingface/autotrain-advanced/tree/main](https://github.com/huggingface/autotrain-advanced/tree/main)

---

## Table of Contents

1. [High-Level Architecture Overview](#1-high-level-architecture-overview)
2. [Core Files (src/autotrain/)](#2-core-files-srcautotrain)
3. [App Layer (src/autotrain/app/)](#3-app-layer-srcautotrainapp)
4. [Trainers Layer (src/autotrain/trainers/)](#4-trainers-layer-srcautotraintrainers)
5. [Preprocessor Layer (src/autotrain/preprocessor/)](#5-preprocessor-layer-srcautotrainpreprocessor)
6. [CLI Layer (src/autotrain/cli/)](#6-cli-layer-srcautotraincli)
7. [Backends Layer (src/autotrain/backends/)](#7-backends-layer-srcautotrainbackends)
8. [Tools Layer (src/autotrain/tools/)](#8-tools-layer-srcautotraintools)
9. [Tests Layer (src/autotrain/tests/)](#9-tests-layer-srcautotraintests)
10. [Configuration Files](#10-configuration-files)
11. [End-to-End Workflow](#11-end-to-end-workflow)
12. [Integration Points](#12-integration-points)
13. [Extension Patterns](#13-extension-patterns)

---

## 1. High-Level Architecture Overview

### Architecture Pattern
AutoTrain Advanced follows a **layered architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Web UI (App)  │  │   CLI Commands  │  │   API Calls  │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Business Logic Layer                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Project Mgmt  │  │   Dataset Mgmt  │  │   Param Mgmt │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Task-Specific Layer                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Trainers      │  │  Preprocessors  │  │   Utils      │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Infrastructure Layer                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Backends      │  │   Logging       │  │   Config     │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow
1. **User Input** → Web UI/CLI/API
2. **Parameter Validation** → App layer validates and processes parameters
3. **Dataset Processing** → Preprocessor layer handles data preparation
4. **Training Orchestration** → Project layer manages training execution
5. **Backend Execution** → Backend layer runs on specified hardware
6. **Model Output** → Results returned to user

---

## 2. Core Files (src/autotrain/)

### 2.1 `__init__.py` (69 lines)
**Role:** Package initialization and public API exports

**Key Functions:**
- `__version__`: Package version string
- `logger`: Centralized logging instance
- Public imports for main classes and functions

**Integration Points:**
- Imported by all other modules
- Defines what's publicly available from the package

**Extension Pattern:**
- Add new public exports here when creating new modules

### 2.2 `config.py` (5 lines)
**Role:** Configuration constants and settings

**Key Constants:**
- `ALLOW_REMOTE_CODE`: Flag for allowing remote code execution
- `AUTOTRAIN_BACKEND`: Default backend configuration

**Integration Points:**
- Used throughout the codebase for configuration checks
- Imported by trainers and preprocessors

### 2.3 `dataset.py` (829 lines)
**Role:** Dataset handling and preprocessing orchestration

**Key Classes:**

#### `AutoTrainDataset` (Main class)
**Purpose:** Generic dataset handler for all text/tabular tasks
**Key Methods:**
- `__post_init__()`: Validates and initializes dataset
- `_preprocess_data()`: Loads CSV/JSONL files into DataFrames
- `prepare()`: Routes to appropriate preprocessor based on task
- `num_samples`: Property returning total dataset size

**Integration Points:**
- Used by `project.py` for dataset preparation
- Routes to task-specific preprocessors
- Handles both local and Hugging Face Hub datasets

#### `AutoTrainImageClassificationDataset`
**Purpose:** Specialized handler for image classification datasets
**Key Methods:**
- `prepare()`: Extracts zip files, validates image structure, calls `ImageClassificationPreprocessor`

#### `AutoTrainObjectDetectionDataset`
**Purpose:** Specialized handler for object detection datasets
**Key Methods:**
- `prepare()`: Extracts zip files, validates annotation structure, calls `ObjectDetectionPreprocessor`

#### `AutoTrainImageRegressionDataset`
**Purpose:** Specialized handler for image regression datasets
**Key Methods:**
- `prepare()`: Extracts zip files, validates metadata structure, calls `ImageRegressionPreprocessor`

#### `AutoTrainVLMDataset`
**Purpose:** Specialized handler for vision-language model datasets
**Key Methods:**
- `prepare()`: Extracts zip files, validates image-text pairs, calls `VLMPreprocessor`

**Utility Functions:**
- `remove_non_image_files()`: Cleans extracted folders of non-image files

**Extension Pattern:**
- Add new dataset classes for new data types
- Follow the pattern: validate → extract → preprocess → return path

### 2.4 `help.py` (82 lines)
**Role:** Help text and documentation for CLI commands

**Key Functions:**
- `get_app_help()`: Returns help text for web UI elements
- `get_cli_help()`: Returns help text for CLI commands

**Integration Points:**
- Used by web UI for tooltips and help dialogs
- Used by CLI for command help

### 2.5 `logger.py` (64 lines)
**Role:** Centralized logging configuration

**Key Functions:**
- `setup_logger()`: Configures logging with file and console handlers
- `logger`: Global logger instance

**Integration Points:**
- Imported by all modules for consistent logging
- Used for training progress, errors, and debugging

### 2.6 `logging.py` (66 lines)
**Role:** Additional logging utilities and monitoring

**Key Functions:**
- `monitor`: Decorator for monitoring function execution
- Logging formatters and handlers

**Integration Points:**
- Used by trainers for monitoring training progress
- Used by preprocessors for monitoring data processing

### 2.7 `params.py` (767 lines)
**Role:** Parameter definitions and validation for all tasks

**Key Classes:**

#### `AppParams` (Main class)
**Purpose:** Parameter processing and validation for web UI
**Key Methods:**
- `munge()`: Routes to task-specific parameter processing
- `_munge_common_params()`: Processes common parameters for all tasks
- Task-specific methods: `_munge_params_img_clf()`, `_munge_params_text_clf()`, etc.

**Integration Points:**
- Used by `ui_routes.py` for parameter processing
- Routes to task-specific parameter classes
- Handles both local and hub dataset parameters

**Parameter Schemas:**
- `PARAMS`: Dictionary mapping task names to default parameter schemas
- `HIDDEN_PARAMS`: List of parameters not shown in UI

**Extension Pattern:**
- Add new task parameters to `PARAMS` dictionary
- Add new `_munge_params_*()` method for task-specific processing

### 2.8 `parser.py` (289 lines)
**Role:** Command-line argument parsing for CLI commands

**Key Functions:**
- `parse_args()`: Parses command-line arguments
- Task-specific argument parsers
- Validation and default value setting

**Integration Points:**
- Used by all CLI commands
- Provides consistent argument handling across tasks

### 2.9 `project.py` (635 lines)
**Role:** Main training orchestration and project management

**Key Classes:**

#### `AutoTrainProject` (Main class)
**Purpose:** Orchestrates the entire training process
**Key Methods:**
- `__post_init__()`: Validates backend configuration
- `_process_params_data()`: Routes to task-specific data processing
- `create()`: Creates and starts training job

**Data Processing Functions:**
- `tabular_munge_data()`: Processes tabular datasets
- `llm_munge_data()`: Processes language model datasets
- `seq2seq_munge_data()`: Processes sequence-to-sequence datasets
- `text_clf_munge_data()`: Processes text classification datasets
- `text_reg_munge_data()`: Processes text regression datasets
- `token_clf_munge_data()`: Processes token classification datasets
- `img_clf_munge_data()`: Processes image classification datasets
- `img_obj_detect_munge_data()`: Processes object detection datasets
- `sent_transformers_munge_data()`: Processes sentence transformer datasets
- `img_reg_munge_data()`: Processes image regression datasets
- `vlm_munge_data()`: Processes vision-language model datasets
- `ext_qa_munge_data()`: Processes extractive QA datasets
- `asr_munge_data()`: Processes automatic speech recognition datasets

**Integration Points:**
- Used by web UI and CLI for starting training jobs
- Routes to appropriate backend runners
- Manages parameter processing and dataset preparation

**Extension Pattern:**
- Add new `*_munge_data()` function for new tasks
- Update `_process_params_data()` to handle new parameter types

### 2.10 `tasks.py` (38 lines)
**Role:** Task registry and mapping

**Key Constants:**
- `NLP_TASKS`: Dictionary mapping NLP task names to IDs
- `VISION_TASKS`: Dictionary mapping vision task names to IDs
- `TABULAR_TASKS`: Dictionary mapping tabular task names to IDs
- `TASKS`: Combined dictionary of all tasks

**Integration Points:**
- Used by UI for task selection
- Used by parameter processing for task validation
- Used by dataset processing for task routing

**Extension Pattern:**
- Add new tasks to appropriate category dictionary
- Ensure task names match those used in other parts of the system

### 2.11 `utils.py` (192 lines)
**Role:** Utility functions used across the codebase

**Key Functions:**
- File and path utilities
- Data validation functions
- Format conversion utilities
- Error handling helpers

**Integration Points:**
- Imported by multiple modules
- Provides common functionality

### 2.12 `version.py` (2 lines)
**Role:** Package version information

**Key Constants:**
- `__version__`: Current package version

**Integration Points:**
- Used by `__init__.py` for version export
- Used by logging and error reporting

### 2.13 `client.py` (303 lines)
**Role:** Client-side utilities for API interaction

**Key Functions:**
- API client functions
- Authentication helpers
- Request/response handling

**Integration Points:**
- Used by CLI commands for API interaction
- Used by web UI for backend communication

### 2.14 `commands.py` (561 lines)
**Role:** CLI command definitions and execution

**Key Classes:**
- Base command classes
- Task-specific command implementations
- Command execution logic

**Integration Points:**
- Used by CLI entry points
- Provides command-line interface for all tasks

---

## 3. App Layer (src/autotrain/app/)

### 3.1 `__init__.py` (2 lines)
**Role:** App package initialization

### 3.2 `app.py` (44 lines)
**Role:** FastAPI application setup and configuration

**Key Functions:**
- `forward_to_ui()`: Root endpoint that redirects to UI
- App configuration and middleware setup
- Static file and template mounting

**Integration Points:**
- Entry point for web application
- Mounts UI routes and API routes
- Configures OAuth if running in Hugging Face Space

### 3.3 `api_routes.py` (812 lines)
**Role:** API endpoints for programmatic access

**Key Endpoints:**
- `/api/projects`: Project management
- `/api/datasets`: Dataset operations
- `/api/models`: Model operations
- `/api/training`: Training job management

**Integration Points:**
- Used by external applications and scripts
- Provides RESTful API for AutoTrain functionality
- Shares logic with UI routes

### 3.4 `colab.py` (437 lines)
**Role:** Google Colab integration utilities

**Key Functions:**
- Colab-specific setup and configuration
- Widget creation and management
- Colab environment detection and adaptation

**Integration Points:**
- Used when running AutoTrain in Google Colab
- Provides Colab-specific UI components

### 3.5 `db.py` (107 lines)
**Role:** Database operations for project tracking

**Key Classes:**

#### `AutoTrainDB`
**Purpose:** SQLite database for storing project information
**Key Methods:**
- `create_tables()`: Creates database schema
- `add_project()`: Adds new project to database
- `get_projects()`: Retrieves project list
- `update_project()`: Updates project status

**Integration Points:**
- Used by UI routes for project persistence
- Used by training API for job tracking

### 3.6 `models.py` (437 lines)
**Role:** Model fetching and management

**Key Functions:**
- `fetch_models()`: Fetches available models from Hugging Face Hub
- Model categorization and filtering
- Model metadata handling

**Integration Points:**
- Used by UI for model selection dropdowns
- Used by parameter processing for model validation

### 3.7 `oauth.py` (173 lines)
**Role:** OAuth authentication for Hugging Face Spaces

**Key Functions:**
- `attach_oauth()`: Configures OAuth for the app
- Authentication flow management
- Token validation and refresh

**Integration Points:**
- Used when running in Hugging Face Spaces
- Provides secure authentication for web UI

### 3.8 `params.py` (791 lines)
**Role:** App-specific parameter handling

**Key Functions:**
- `get_task_params()`: Fetches parameters for specific tasks
- Parameter validation and processing
- UI parameter rendering

**Integration Points:**
- Used by UI routes for parameter management
- Used by frontend JavaScript for dynamic UI updates

### 3.9 `training_api.py` (110 lines)
**Role:** Training job management API

**Key Functions:**
- Training job creation and monitoring
- Job status updates
- Log streaming

**Integration Points:**
- Used by web UI for training management
- Used by frontend for real-time updates

### 3.10 `ui_routes.py` (1078 lines)
**Role:** Web UI backend routes and logic

**Key Endpoints:**

#### Authentication
- `/ui/`: Main UI page with authentication
- `/ui/logout`: Logout endpoint

#### Parameter Management
- `/ui/params/{task}/{param_type}`: Fetches parameters for task
- `/ui/model_choices/{task}`: Fetches available models for task

#### Project Management
- `/ui/create_project`: Creates new training project
- `/ui/help/{element_id}`: Fetches help text

#### Training Management
- `/ui/accelerators`: Fetches available hardware
- `/ui/is_model_training`: Checks if training is running
- `/ui/logs`: Fetches training logs
- `/ui/stop_training`: Stops running training

#### LiFE App Integration (ASR-specific)
- `/ui/life_app_projects`: Fetches LiFE App projects
- `/ui/project_selected`: Handles project selection
- `/ui/script_selected`: Handles script selection

**Key Functions:**
- `user_authentication()`: Authentication dependency
- `handle_form()`: Main project creation logic
- `graceful_exit()`: Cleanup on shutdown

**Integration Points:**
- Main backend for web UI
- Handles file uploads and form processing
- Routes to appropriate dataset and parameter processors
- Manages training job lifecycle

### 3.11 `utils.py` (193 lines)
**Role:** App-specific utility functions

**Key Functions:**
- `get_running_jobs()`: Gets currently running training jobs
- `get_user_and_orgs()`: Fetches user and organization information
- `kill_process_by_pid()`: Kills processes by PID
- `token_verification()`: Verifies authentication tokens

**Integration Points:**
- Used by UI routes for job management
- Used by authentication system

### 3.12 `static/` Directory
**Role:** Static assets for web UI

#### `scripts/` Directory
- `fetch_data_and_update_models.js`: Model fetching and UI updates
- `listeners.js`: Main UI event handling
- `utils.js`: Frontend utility functions
- `poll.js`: Polling for updates
- `logs.js`: Log display and management

#### `templates/` Directory
- `index.html`: Main UI template
- `error.html`: Error page template
- `duplicate.html`: Duplicate project page
- `login.html`: Login page template

---

## 4. Trainers Layer (src/autotrain/trainers/)

### 4.1 `common.py` (547 lines)
**Role:** Common training utilities and base classes

**Key Functions:**
- Training loop utilities
- Model loading and saving
- Evaluation functions
- Logging and monitoring

**Integration Points:**
- Used by all task-specific trainers
- Provides shared training functionality

### 4.2 `image_classification/` Directory
**Role:** Image classification training implementation

#### `__main__.py` (242 lines)
**Purpose:** Main training entry point
**Key Functions:**
- `train()`: Main training function
- `parse_args()`: Argument parsing
- Dataset loading and validation
- Model configuration and training loop

**Integration Points:**
- Called by backend runners
- Uses `ImageClassificationParams`
- Uses `ImageClassificationDataset`

#### `dataset.py` (46 lines)
**Purpose:** Dataset class for image classification
**Key Methods:**
- `__getitem__()`: Returns image and label tensors
- `__len__()`: Returns dataset size

**Integration Points:**
- Used by training loop
- Handles image preprocessing and augmentation

#### `params.py` (41 lines)
**Purpose:** Parameter definition for image classification
**Key Attributes:**
- Model configuration parameters
- Training hyperparameters
- Dataset configuration

**Integration Points:**
- Used by training function
- Validated by parameter processing

#### `utils.py` (168 lines)
**Purpose:** Image classification utilities
**Key Functions:**
- `process_data()`: Data preprocessing pipeline
- Image augmentation functions
- Evaluation utilities

**Integration Points:**
- Used by training function
- Handles data transformation

### 4.3 `image_regression/` Directory
**Role:** Image regression training implementation

**Structure:** Similar to `image_classification/` but for regression tasks

### 4.4 `object_detection/` Directory
**Role:** Object detection training implementation

**Structure:** Similar to `image_classification/` but for object detection tasks

### 4.5 `text_classification/` Directory
**Role:** Text classification training implementation

**Structure:** Similar to `image_classification/` but for text classification tasks

### 4.6 `text_regression/` Directory
**Role:** Text regression training implementation

**Structure:** Similar to `text_classification/` but for regression tasks

### 4.7 `token_classification/` Directory
**Role:** Token classification training implementation

**Structure:** Similar to `text_classification/` but for token-level tasks

### 4.8 `seq2seq/` Directory
**Role:** Sequence-to-sequence training implementation

**Structure:** Similar to `text_classification/` but for seq2seq tasks

### 4.9 `sent_transformers/` Directory
**Role:** Sentence transformers training implementation

**Structure:** Similar to `text_classification/` but for sentence transformer tasks

### 4.10 `tabular/` Directory
**Role:** Tabular data training implementation

**Structure:** Similar to `text_classification/` but for tabular tasks

### 4.11 `clm/` Directory
**Role:** Causal language model training implementation

**Structure:** Similar to `text_classification/` but for language modeling tasks

### 4.12 `vlm/` Directory
**Role:** Vision-language model training implementation

**Structure:** Similar to `image_classification/` but for vision-language tasks

### 4.13 `extractive_question_answering/` Directory
**Role:** Extractive QA training implementation

**Structure:** Similar to `text_classification/` but for QA tasks

### 4.14 `generic/` Directory
**Role:** Generic training implementation

**Structure:** For custom training scripts and generic tasks

### 4.15 `automatic_speech_recognition/` Directory
**Role:** ASR training implementation

**Structure:** Similar to `text_classification/` but for speech recognition tasks

---

## 5. Preprocessor Layer (src/autotrain/preprocessor/)

### 5.1 `base.py` (57 lines)
**Role:** Base classes for preprocessors

**Key Classes:**
- Base preprocessor class
- Common preprocessing utilities

**Integration Points:**
- Inherited by all task-specific preprocessors
- Provides common preprocessing functionality

### 5.2 `vision.py` (566 lines)
**Role:** Image preprocessing for vision tasks

**Key Classes:**

#### `ImageClassificationPreprocessor`
**Purpose:** Preprocesses image classification datasets
**Key Methods:**
- `__post_init__()`: Validates image folder structure
- `split()`: Splits data into train/validation sets
- `prepare()`: Prepares dataset for training

#### `ImageRegressionPreprocessor`
**Purpose:** Preprocesses image regression datasets
**Key Methods:**
- `_process_metadata()`: Processes metadata.jsonl files
- `prepare()`: Prepares dataset for training

#### `ObjectDetectionPreprocessor`
**Purpose:** Preprocesses object detection datasets
**Key Methods:**
- `_process_metadata()`: Processes annotation files
- `prepare()`: Prepares dataset for training

**Integration Points:**
- Used by `dataset.py` for image dataset preparation
- Creates Hugging Face datasets
- Handles local and hub dataset saving

### 5.3 `text.py` (829 lines)
**Role:** Text preprocessing for NLP tasks

**Key Classes:**

#### `TextBinaryClassificationPreprocessor`
#### `TextMultiClassClassificationPreprocessor`
#### `TextSingleColumnRegressionPreprocessor`
#### `TextTokenClassificationPreprocessor`
#### `Seq2SeqPreprocessor`
#### `LLMPreprocessor`
#### `SentenceTransformersPreprocessor`
#### `TextExtractiveQuestionAnsweringPreprocessor`

**Key Methods:**
- `__post_init__()`: Validates text data
- `split()`: Splits data into train/validation sets
- `prepare()`: Prepares dataset for training

**Integration Points:**
- Used by `dataset.py` for text dataset preparation
- Creates Hugging Face datasets
- Handles text tokenization and preprocessing

### 5.4 `tabular.py` (274 lines)
**Role:** Tabular data preprocessing

**Key Classes:**

#### `TabularBinaryClassificationPreprocessor`
#### `TabularMultiClassClassificationPreprocessor`
#### `TabularMultiLabelClassificationPreprocessor`
#### `TabularSingleColumnRegressionPreprocessor`
#### `TabularMultiColumnRegressionPreprocessor`

**Key Methods:**
- `__post_init__()`: Validates tabular data
- `split()`: Splits data into train/validation sets
- `prepare()`: Prepares dataset for training

**Integration Points:**
- Used by `dataset.py` for tabular dataset preparation
- Handles data imputation and scaling
- Creates Hugging Face datasets

### 5.5 `vlm.py` (225 lines)
**Role:** Vision-language model preprocessing

**Key Classes:**

#### `VLMPreprocessor`
**Purpose:** Preprocesses vision-language datasets
**Key Methods:**
- `_process_metadata()`: Processes image-text metadata
- `prepare()`: Prepares dataset for training

**Integration Points:**
- Used by `dataset.py` for VLM dataset preparation
- Handles image-text pair processing
- Creates Hugging Face datasets

### 5.6 `automatic_speech_recognition.py` (195 lines)
**Role:** ASR preprocessing

**Key Classes:**

#### `AutomaticSpeechRecognitionPreprocessor`
**Purpose:** Preprocesses speech recognition datasets
**Key Methods:**
- `__post_init__()`: Validates audio data
- `prepare()`: Prepares dataset for training

**Integration Points:**
- Used by `dataset.py` for ASR dataset preparation
- Handles audio file processing
- Creates Hugging Face datasets

---

## 6. CLI Layer (src/autotrain/cli/)

### 6.1 `__init__.py` (14 lines)
**Role:** CLI package initialization

### 6.2 `autotrain.py` (83 lines)
**Role:** Main CLI entry point

**Key Functions:**
- `main()`: Main CLI function
- Command routing and execution

**Integration Points:**
- Entry point for command-line usage
- Routes to task-specific commands

### 6.3 `run.py` (35 lines)
**Role:** Generic run command

**Key Functions:**
- Generic training execution
- Parameter handling

### 6.4 `run_api.py` (71 lines)
**Role:** API server command

**Key Functions:**
- Starts API server
- Server configuration

### 6.5 `run_app.py` (177 lines)
**Role:** Web app command

**Key Functions:**
- Starts web application
- App configuration

### 6.6 Task-Specific Commands
**Role:** CLI commands for each task type

**Files:**
- `run_image_classification.py`
- `run_image_regression.py`
- `run_object_detection.py`
- `run_text_classification.py`
- `run_text_regression.py`
- `run_token_classification.py`
- `run_seq2seq.py`
- `run_sent_tranformers.py`
- `run_tabular.py`
- `run_llm.py`
- `run_vlm.py`
- `run_extractive_qa.py`
- `run_automatic_speech_recognition.py`

**Structure:** Each follows the same pattern:
- Command class inheriting from base command
- Parameter validation
- Project creation and execution

### 6.7 `utils.py` (189 lines)
**Role:** CLI utility functions

**Key Functions:**
- Command-line argument processing
- Parameter validation
- Error handling

**Integration Points:**
- Used by all CLI commands
- Provides common CLI functionality

### 6.8 `run_tools.py` (100 lines)
**Role:** Tools command

**Key Functions:**
- Tool execution utilities
- Tool parameter handling

### 6.9 `run_spacerunner.py` (144 lines)
**Role:** Space runner command

**Key Functions:**
- Hugging Face Space execution
- Space configuration

---

## 7. Backends Layer (src/autotrain/backends/)

### 7.1 `base.py` (238 lines)
**Role:** Base backend classes and utilities

**Key Classes:**

#### `BaseRunner`
**Purpose:** Base class for all backend runners
**Key Methods:**
- `create()`: Abstract method for job creation
- `validate()`: Validates backend configuration
- `setup()`: Sets up backend environment

**Integration Points:**
- Inherited by all backend implementations
- Provides common backend functionality

### 7.2 `local.py` (223 lines)
**Role:** Local backend implementation

**Key Classes:**

#### `LocalRunner`
**Purpose:** Runs training jobs locally
**Key Methods:**
- `create()`: Creates and starts local training job
- `setup()`: Sets up local environment
- `monitor()`: Monitors training progress

**Integration Points:**
- Used by `project.py` for local training
- Manages local process execution

### 7.3 `spaces.py` (94 lines)
**Role:** Hugging Face Spaces backend implementation

**Key Classes:**

#### `SpaceRunner`
**Purpose:** Runs training jobs on Hugging Face Spaces
**Key Methods:**
- `create()`: Creates Space and starts training
- `setup()`: Sets up Space configuration
- `monitor()`: Monitors Space status

**Integration Points:**
- Used by `project.py` for Space training
- Manages Space creation and monitoring

### 7.4 `endpoints.py` (87 lines)
**Role:** Hugging Face Endpoints backend implementation

**Key Classes:**

#### `EndpointsRunner`
**Purpose:** Runs training jobs on Hugging Face Endpoints
**Key Methods:**
- `create()`: Creates endpoint and starts training
- `setup()`: Sets up endpoint configuration
- `monitor()`: Monitors endpoint status

**Integration Points:**
- Used by `project.py` for Endpoint training
- Manages endpoint creation and monitoring

### 7.5 `ngc.py` (108 lines)
**Role:** NVIDIA NGC backend implementation

**Key Classes:**

#### `NGCRunner`
**Purpose:** Runs training jobs on NVIDIA NGC
**Key Methods:**
- `create()`: Creates NGC job and starts training
- `setup()`: Sets up NGC configuration
- `monitor()`: Monitors NGC job status

**Integration Points:**
- Used by `project.py` for NGC training
- Manages NGC job creation and monitoring

### 7.6 `nvcf.py` (204 lines)
**Role:** NVIDIA Cloud Functions backend implementation

**Key Classes:**

#### `NVCFRunner`
**Purpose:** Runs training jobs on NVIDIA Cloud Functions
**Key Methods:**
- `create()`: Creates NVCF job and starts training
- `setup()`: Sets up NVCF configuration
- `monitor()`: Monitors NVCF job status

**Integration Points:**
- Used by `project.py` for NVCF training
- Manages NVCF job creation and monitoring

---

## 8. Tools Layer (src/autotrain/tools/)

### 8.1 `__init__.py` (2 lines)
**Role:** Tools package initialization

### 8.2 `convert_to_kohya.py`
**Role:** Kohya SS conversion utility

**Key Functions:**
- Converts AutoTrain models to Kohya SS format
- Model format conversion utilities

### 8.3 `merge_adapter.py`
**Role:** Adapter merging utility

**Key Functions:**
- Merges LoRA adapters with base models
- Model merging utilities

---

## 9. Tests Layer (src/autotrain/tests/)

### 9.1 `__init__.py`
**Role:** Tests package initialization

### 9.2 `test_cli.py`
**Role:** CLI command tests

**Key Tests:**
- Command execution tests
- Parameter validation tests
- Error handling tests

### 9.3 `test_dummy.py`
**Role:** Dummy tests

**Key Tests:**
- Basic functionality tests
- Smoke tests

---

## 10. Configuration Files

### 10.1 `setup.py`
**Role:** Package setup and installation

**Key Sections:**
- Package metadata
- Dependencies
- Data files (static assets, templates)
- Entry points

### 10.2 `requirements.txt`
**Role:** Python dependencies

**Key Dependencies:**
- Core ML libraries (torch, transformers, datasets)
- Web framework (fastapi)
- Utilities (pandas, numpy, etc.)

### 10.3 `configs/` Directory
**Role:** Configuration templates for each task

**Structure:**
- `image_classification/`: Image classification configs
- `text_classification/`: Text classification configs
- `llm_finetuning/`: LLM fine-tuning configs
- etc.

### 10.4 `docs/` Directory
**Role:** Documentation

**Structure:**
- `source/tasks/`: Task-specific documentation
- Configuration guides
- Usage examples

---

## 11. End-to-End Workflow

### 11.1 Web UI Workflow

1. **User Access**
   - User visits web UI (`/ui/`)
   - Authentication via OAuth (if in Space)

2. **Task Selection**
   - User selects task from dropdown
   - Frontend (`listeners.js`) fetches parameters via `/ui/params/{task}/{param_type}`
   - UI updates with task-specific parameters

3. **Dataset Upload**
   - User selects dataset source (local/hub)
   - For local: Uploads zip file via form
   - For hub: Enters dataset name and splits
   - Frontend validates and prepares data

4. **Parameter Configuration**
   - User configures training parameters
   - Frontend updates parameter JSON
   - Validation occurs on both frontend and backend

5. **Training Start**
   - User clicks "Start Training"
   - Frontend sends data to `/ui/create_project`
   - Backend processes parameters and dataset
   - Training job is created and started

6. **Training Monitoring**
   - Frontend polls `/ui/logs` for training progress
   - Real-time log updates
   - Training status monitoring

### 11.2 CLI Workflow

1. **Command Execution**
   - User runs CLI command (e.g., `autotrain image-classification`)
   - CLI parses arguments and validates parameters

2. **Project Creation**
   - CLI creates `AutoTrainProject` instance
   - Parameters are processed and validated
   - Dataset is prepared if needed

3. **Backend Execution**
   - Appropriate backend runner is selected
   - Training job is created and started
   - Progress is monitored and logged

### 11.3 API Workflow

1. **API Call**
   - External application makes API call
   - Authentication and authorization
   - Parameter validation

2. **Job Creation**
   - API creates training job
   - Returns job ID and status
   - Job monitoring endpoints available

3. **Result Retrieval**
   - API provides endpoints for job status
   - Model download endpoints
   - Log retrieval endpoints

---

## 12. Integration Points

### 12.1 Data Flow Integration

```
User Input → UI/CLI/API → Parameter Processing → Dataset Preparation → Training Execution → Results
     ↓              ↓              ↓                    ↓                    ↓              ↓
  Frontend    Backend Routes   App Params        Preprocessors        Trainers      Backend Runners
```

### 12.2 Task Integration

```
Task Selection → Parameter Fetch → Dataset Processing → Training Execution
      ↓              ↓                    ↓                    ↓
   tasks.py      params.py          preprocessors/        trainers/
```

### 12.3 Backend Integration

```
Project Creation → Backend Selection → Job Execution → Monitoring
      ↓                ↓                ↓              ↓
   project.py      backends/        Local/Space/    Logging/
```

---

## 13. Extension Patterns

### 13.1 Adding New Tasks

1. **Register Task**
   - Add task to `tasks.py`
   - Add task to `params.py` PARAMS dictionary
   - Add task to `app/params.py` munge methods

2. **Create Preprocessor**
   - Create new preprocessor in `preprocessor/`
   - Follow existing preprocessor pattern
   - Add to `dataset.py` routing

3. **Create Trainer**
   - Create new trainer in `trainers/`
   - Follow existing trainer pattern
   - Add to `project.py` munge methods

4. **Update UI**
   - Add task to frontend task dropdown
   - Add task-specific UI logic if needed
   - Update parameter schemas

5. **Add CLI Command**
   - Create new CLI command in `cli/`
   - Follow existing command pattern
   - Add to command routing

### 13.2 Adding New Backends

1. **Create Backend Runner**
   - Create new runner in `backends/`
   - Inherit from `BaseRunner`
   - Implement required methods

2. **Update Backend Selection**
   - Add backend to `base.py` AVAILABLE_HARDWARE
   - Update `project.py` backend routing

3. **Add Configuration**
   - Add backend-specific configuration
   - Update parameter validation

### 13.3 Code Organization Principles

1. **Separation of Concerns**
   - UI logic in `app/`
   - Business logic in core files
   - Task-specific logic in task folders

2. **Common Logic Sharing**
   - Shared utilities in common files
   - Task-specific logic isolated
   - Clear integration points

3. **Consistent Patterns**
   - Follow existing patterns for new code
   - Use same naming conventions
   - Maintain code hygiene

4. **Error Handling**
   - Consistent error handling across layers
   - Proper logging and debugging
   - User-friendly error messages

---

## 14. Key Design Patterns

### 14.1 Factory Pattern
- Task-specific objects created based on task type
- Parameter classes instantiated dynamically
- Preprocessors selected based on task

### 14.2 Strategy Pattern
- Different backends implement same interface
- Different trainers for different tasks
- Different preprocessors for different data types

### 14.3 Observer Pattern
- Training progress monitoring
- Log streaming
- Real-time UI updates

### 14.4 Template Method Pattern
- Base classes define common workflow
- Task-specific classes implement details
- Consistent behavior across tasks

---

This comprehensive map provides a complete understanding of the AutoTrain Advanced codebase structure, workflow, and extension patterns. Every file, function, and integration point is documented for reference when implementing new features or tasks. 