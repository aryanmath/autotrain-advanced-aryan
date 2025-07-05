# AutoTrain Advanced Repository - Complete ASR Implementation Analysis

## Table of Contents
1. [Repository Overview](#repository-overview)
2. [File Structure Analysis](#file-structure-analysis)
3. [ASR Task Implementation](#asr-task-implementation)
4. [UI/Frontend Workflow](#uifrontend-workflow)
5. [Backend Workflow](#backend-workflow)
6. [Training Pipeline](#training-pipeline)
7. [Dataset Handling](#dataset-handling)
8. [Configuration Management](#configuration-management)
9. [Common vs Task-Specific Code](#common-vs-task-specific-code)
10. [Complete Workflow from UI to Training](#complete-workflow-from-ui-to-training)

## Repository Overview

This repository is a fork of the official Hugging Face AutoTrain Advanced repository with the addition of Automatic Speech Recognition (ASR) task support. The implementation follows the exact same patterns and structure as other tasks in the main repository.

### Key Components:
- **Frontend**: FastAPI-based web UI with JavaScript for dynamic interactions
- **Backend**: Python-based training pipeline with modular architecture
- **ASR Task**: Complete implementation following main repo patterns
- **LiFE App Integration**: Special dataset source for ASR tasks only

## File Structure Analysis

### Core Directory Structure
```
src/autotrain/
├── app/                    # Web UI and API routes
├── backends/              # Training backends (local, cloud, etc.)
├── cli/                   # Command-line interface
├── preprocessor/          # Data preprocessing modules
├── trainers/              # Task-specific training modules
├── __init__.py           # Main package initialization
├── tasks.py              # Task definitions and IDs
├── project.py            # Project creation and management
├── dataset.py            # Dataset handling classes
└── params.py             # Parameter management
```

### ASR-Specific Files
```
src/autotrain/trainers/automatic_speech_recognition/
├── __main__.py           # Main training entry point
├── dataset.py            # ASR dataset class
├── params.py             # ASR parameters
├── utils.py              # ASR utilities
└── __init__.py

src/autotrain/preprocessor/
└── automatic_speech_recognition.py  # ASR data preprocessing

src/autotrain/cli/
└── run_automatic_speech_recognition.py  # CLI command for ASR

configs/automatic_speech_recognition/
├── local_dataset.yml     # Local dataset configuration
└── hub_dataset.yml       # Hub dataset configuration
```

## ASR Task Implementation

### 1. Task Definition (`src/autotrain/tasks.py`)
```python
NLP_TASKS = {
    # ... other tasks ...
    "ASR": 32,  # ASR task ID
}
```

**Purpose**: Defines ASR as task ID 32 in the NLP tasks category.

### 2. ASR Parameters (`src/autotrain/trainers/automatic_speech_recognition/params.py`)

**Class**: `AutomaticSpeechRecognitionParams`
**Inherits from**: `AutoTrainParams`

**Key Parameters**:
- `audio_column`: Column name for audio data (default: "audio")
- `text_column`: Column name for transcription (default: "transcription")
- `max_duration`: Maximum audio duration in seconds (default: 30.0)
- `sampling_rate`: Audio sampling rate (default: 16000)
- `max_seq_length`: Maximum text sequence length (default: 128)

**Validation Logic**:
- Validates model exists in available models
- Checks data path exists for local datasets
- Validates required columns in CSV files
- Validates parameter ranges and types

### 3. ASR Dataset Class (`src/autotrain/trainers/automatic_speech_recognition/dataset.py`)

**Class**: `AutomaticSpeechRecognitionDataset`

**Key Features**:
- Universal dataset that works with all ASR models (Whisper, Wav2Vec2, Hubert, etc.)
- Automatic model type detection
- Dynamic padding for different audio lengths
- Robust error handling for audio processing

**Model Type Detection**:
```python
def detect_model_type(model):
    model_class = type(model).__name__
    if 'Whisper' in model_class or 'Seq2Seq' in model_class:
        return "seq2seq"
    elif 'CTC' in model_class or 'Wav2Vec' in model_class:
        return "ctc"
    else:
        return "generic"
```

**Audio Processing**:
- Loads audio using librosa
- Handles different sampling rates
- Truncates audio if exceeds max_duration
- Processes audio based on model type (seq2seq vs ctc)

**Text Processing**:
- Safe tokenization that works with all processors
- Handles different tokenizer types
- Fallback mechanisms for tokenization failures

### 4. ASR Training Main (`src/autotrain/trainers/automatic_speech_recognition/__main__.py`)

**Main Functions**:

#### `load_data(params, is_validation=False)`
- Loads datasets from local directory or HuggingFace Hub
- Handles CSV files with audio paths and transcriptions
- Validates audio file existence
- Creates train/validation splits

#### `load_model_and_processor(params)`
- Loads model and processor with fallback mechanisms
- Tries different processor types (AutoProcessor, Wav2Vec2Processor, WhisperProcessor)
- Tries different model types (Seq2Seq, CTC, Generic)
- Moves model to appropriate device (CUDA/CPU)

#### `train(config)`
- Main training function with monitoring
- Creates training and validation datasets
- Initializes Trainer with ASR-specific settings
- Handles model saving and upload to Hub

**Training Arguments**:
- Uses dynamic padding collator
- Sets up evaluation strategy
- Configures mixed precision training
- Sets up callbacks for logging and early stopping

### 5. ASR Preprocessor (`src/autotrain/preprocessor/automatic_speech_recognition.py`)

**Class**: `AutomaticSpeechRecognitionPreprocessor`

**Key Functions**:
- `split()`: Splits data into train/validation sets
- `prepare_columns()`: Renames columns to standard format
- `prepare()`: Main preprocessing pipeline
- `_process_data()`: Processes audio and text data
- `_process_audio()`: Loads and processes audio files

## UI/Frontend Workflow

### 1. Main UI Template (`src/autotrain/app/templates/index.html`)

**ASR-Specific JavaScript**:
```javascript
case 'ASR':
    fields = ['audio', 'transcription'];
    fieldNames = ['audio', 'transcription'];
    // Show LiFE App dataset source option only for ASR
    document.getElementById("dataset_source").querySelectorAll("option").forEach(opt => {
        if (opt.value === "life_app") {
            opt.style.display = "";
        }
    });
    break;
```

**Key Features**:
- Dynamic column mapping for ASR task
- Shows LiFE App dataset source only for ASR
- Hides LiFE App option for other tasks

### 2. Frontend Scripts (`src/autotrain/app/static/scripts/listeners.js`)

**Dataset Source Handling**:
```javascript
function handleDataSource() {
    const taskValue = document.getElementById('task').value;
    
    // Show/hide LiFE App option based on task
    if (taskValue === "ASR") {
        lifeAppOption.style.display = "";
    } else {
        lifeAppOption.style.display = "none";
    }
}
```

**LiFE App Integration**:
- `loadLifeAppProjects()`: Fetches available projects
- `loadScriptsForProjects()`: Fetches scripts for selected projects
- `loadDatasetsForScript()`: Fetches datasets for selected script

### 3. UI Routes (`src/autotrain/app/ui_routes.py`)

**Key Endpoints**:

#### `/ui/create_project` (POST)
Handles form submission for creating projects:

**ASR-Specific Logic**:
```python
if data_source == "life_app":
    if task != "ASR":
        raise HTTPException(
            status_code=400,
            detail="LiFE app datasets can only be used with Automatic Speech Recognition tasks"
        )
    
    # Process LiFE App dataset
    dataset_path = os.path.join(BASE_DIR, "static", "dataset.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset_json = json.load(f)
    
    # Convert base64 audio to files
    df = pd.DataFrame(dataset_json)
    audio_dir = os.path.join("life_app_data", "audio")
    # ... audio processing logic
```

**ZIP+CSV+Audio Support**:
```python
if task == "ASR" and file_extension.lower() == "zip":
    # Extract zip to temp dir
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(io.BytesIO(data_files_training[0].file.read()), 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    # Find CSV and audio folder
    csv_path = None
    audio_dir = None
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_path = os.path.join(root, file)
        for d in dirs:
            if d.lower() == 'audio':
                audio_dir = os.path.join(root, d)
    
    # Update CSV audio column to full paths
    df = pd.read_csv(csv_path)
    def get_audio_path(x):
        p = os.path.join(audio_dir, str(x))
        if not os.path.exists(p):
            raise HTTPException(status_code=400, detail=f"Audio file {x} not found")
        return p
    df['audio'] = df['audio'].apply(get_audio_path)
```

#### `/ui/life_app_projects` (GET)
Returns available LiFE App projects.

#### `/ui/life_app_scripts` (GET)
Returns scripts for selected projects.

#### `/ui/life_app_dataset` (GET)
Returns dataset information.

#### `/ui/project_selected` (POST)
Handles project selection and returns available scripts.

#### `/ui/script_selected` (POST)
Handles script selection and returns available datasets.

## Backend Workflow

### 1. Project Creation (`src/autotrain/project.py`)

**ASR Munge Data Function**:
```python
def asr_munge_data(params, local):
    # Handle CSV/JSONL files
    exts = ["csv", "jsonl"]
    ext_to_use = None
    for ext in exts:
        path = f"{params.data_path}/{params.train_split}.{ext}"
        if os.path.exists(path):
            ext_to_use = ext
            break
    
    # Create AutoTrainDataset for ASR
    dset = AutoTrainDataset(
        train_data=[train_data_path],
        task="ASR",
        token=params.token,
        project_name=params.project_name,
        username=params.username,
        column_mapping={"audio": params.audio_column, "text": params.text_column},
        valid_data=[valid_data_path] if valid_data_path is not None else None,
        percent_valid=None,
        local=local,
        ext=ext_to_use,
    )
    
    # Update parameters
    params.data_path = dset.prepare()
    params.valid_split = "validation"
    params.audio_column = "audio"
    params.text_column = "transcription"
    
    # LiFE App support
    if hasattr(params, "data_path") and params.data_path == "life_app_data":
        params.text_column = "transcription"
        params.audio_column = "audio"
        params.data_path = "life_app_data"
    
    return params
```

### 2. Dataset Handling (`src/autotrain/dataset.py`)

**ASR Dataset Preparation**:
```python
elif self.task in ["ASR", "ASR"]:
    audio_column = self.column_mapping["audio"]
    text_column = self.column_mapping.get("text") or self.column_mapping.get("transcription")
    
    if text_column is None:
        raise ValueError("Column mapping must include either 'text' or 'transcription'")
    
    preprocessor = AutomaticSpeechRecognitionPreprocessor(
        train_data=self.train_df,
        token=self.token,
        project_name=self.project_name,
        username=self.username,
        column_mapping=self.column_mapping,
        valid_data=self.valid_df,
    )
    return preprocessor.prepare()
```

### 3. Local Backend (`src/autotrain/backends/local.py`)

**ASR-Specific Training**:
```python
if isinstance(self.params, AutomaticSpeechRecognitionParams):
    # Create training config
    config_path = f"{self.params.project_name}/training_config.json"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    config_dict = self.params.dict()
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    
    # Run ASR training command
    command = f"{sys.executable} -m autotrain.trainers.automatic_speech_recognition.__main__ --training_config \"{config_path}\""
    
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=open("asr.log", "w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        env=env,
        cwd=WORKSPACE_ROOT
    )
    
    # Add job to database
    DB = AutoTrainDB("autotrain.db")
    DB.add_job(process.pid)
    
    self.job_id = str(process.pid)
```

### 4. CLI Commands (`src/autotrain/cli/run_automatic_speech_recognition.py`)

**ASR Command Factory**:
```python
def run_ASR_command_factory(args):
    return RunAutoTrainAutomaticSpeechRecognitionCommand(args)

class RunAutoTrainAutomaticSpeechRecognitionCommand(AutoTrainCLI):
    @staticmethod
    def register_subcommand(parser: argparse._SubParsersAction):
        arg_list = get_field_info(AutomaticSpeechRecognitionParams)
        # Add command-specific arguments
        arg_list = [
            {"arg": "--train", "help": "Command to train the model", "action": "store_true"},
            {"arg": "--deploy", "help": "Command to deploy the model", "action": "store_true"},
            {"arg": "--inference", "help": "Command to run inference", "action": "store_true"},
            {"arg": "--backend", "help": "Backend to use for training", "default": "local"},
        ] + arg_list
        
        run_parser = parser.add_parser("ASR", description="✨ Run AutoTrain Automatic Speech Recognition")
        # ... argument registration
```

## Training Pipeline

### 1. Training Configuration

**Training Arguments**:
- `output_dir`: Model output directory
- `per_device_train_batch_size`: Batch size per device
- `gradient_accumulation_steps`: Gradient accumulation
- `learning_rate`: Learning rate
- `num_train_epochs`: Number of training epochs
- `save_strategy`: Model saving strategy
- `evaluation_strategy`: Evaluation strategy
- `fp16/bf16`: Mixed precision training
- `gradient_checkpointing`: Memory optimization
- `early_stopping_patience`: Early stopping

### 2. Model Loading

**Processor Loading**:
```python
try:
    processor = AutoProcessor.from_pretrained(
        params.model,
        token=params.token if params.token else None,
        trust_remote_code=ALLOW_REMOTE_CODE,
    )
except Exception as e:
    # Fallback to Wav2Vec2Processor
    try:
        from transformers import Wav2Vec2Processor
        processor = Wav2Vec2Processor.from_pretrained(...)
    except Exception as e2:
        # Fallback to WhisperProcessor
        try:
            from transformers import WhisperProcessor
            processor = WhisperProcessor.from_pretrained(...)
        except Exception as e3:
            raise ValueError(f"Could not load processor with any known type")
```

**Model Loading**:
```python
try:
    # Try Seq2Seq model first
    model = AutoModelForSpeechSeq2Seq.from_pretrained(...)
except Exception as e:
    try:
        # Try CTC model
        model = AutoModelForCTC.from_pretrained(...)
    except Exception as e:
        try:
            # Try generic model
            model = AutoModel.from_pretrained(...)
        except Exception as e:
            raise ValueError(f"Could not load model as any supported type")
```

### 3. Dataset Creation

**Training Dataset**:
```python
train_dataset = AutomaticSpeechRecognitionDataset(
    data=dataset,
    processor=processor,
    model=model,
    audio_column=config.audio_column,
    text_column=config.text_column,
    max_duration=config.max_duration,
    sampling_rate=config.sampling_rate,
)
```

**Validation Dataset**:
```python
if valid_dataset is not None:
    valid_dataset_obj = AutomaticSpeechRecognitionDataset(
        data=valid_dataset,
        processor=processor,
        model=model,
        audio_column=config.audio_column,
        text_column=config.text_column,
        max_duration=config.max_duration,
        sampling_rate=config.sampling_rate,
    )
```

### 4. Training Execution

**Trainer Setup**:
```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset_obj if valid_dataset is not None else None,
    callbacks=callbacks,
    data_collator=dynamic_padding_collator,
    compute_metrics=compute_metrics if valid_dataset is not None else None,
)
```

**Training**:
```python
trainer.train()
```

### 5. Model Saving

**Save Model and Processor**:
```python
# Save final model and processor
trainer.save_model(config.project_name)
processor.save_pretrained(config.project_name)

# Save tokenizer if available
if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "save_pretrained"):
    processor.tokenizer.save_pretrained(config.project_name)
```

**Create Model Card**:
```python
model_card = utils.create_model_card(config, trainer)
with open(f"{config.project_name}/README.md", "w") as f:
    f.write(model_card)
```

**Upload to Hub**:
```python
if config.push_to_hub:
    api = HfApi(token=config.token)
    api.create_repo(
        repo_id=f"{config.username}/{config.project_name}", 
        repo_type="model", 
        private=True, 
        exist_ok=True
    )
    api.upload_folder(
        folder_path=config.project_name, 
        repo_id=f"{config.username}/{config.project_name}", 
        repo_type="model"
    )
```

## Dataset Handling

### 1. Local Dataset Support

**ZIP+CSV+Audio Structure**:
```
dataset.zip
├── data.csv          # CSV with audio filenames and transcriptions
└── audio/            # Folder containing audio files
    ├── audio1.wav
    ├── audio2.wav
    └── ...
```

**CSV Format**:
```csv
audio,transcription
audio1.wav,"Hello world"
audio2.wav,"How are you"
```

### 2. HuggingFace Hub Dataset Support

**Dataset Loading**:
```python
if params.using_hub_dataset:
    dataset = load_dataset(
        params.data_path,
        split=params.valid_split if is_validation else params.train_split,
        use_auth_token=params.token if params.token else None
    )
```

### 3. LiFE App Dataset Support

**Dataset Processing**:
```python
# Load JSON dataset
with open(dataset_path, "r", encoding="utf-8") as f:
    dataset_json = json.load(f)

df = pd.DataFrame(dataset_json)

# Convert base64 audio to files
audio_dir = os.path.join("life_app_data", "audio")
os.makedirs(audio_dir, exist_ok=True)
audio_paths = []

for idx, row in df.iterrows():
    audio_bytes = row["audio"]
    try:
        audio_data = base64.b64decode(audio_bytes)
    except Exception:
        audio_data = audio_bytes.encode("latin1")
    
    audio_path = os.path.join(audio_dir, f"audio_{idx}.wav")
    with open(audio_path, "wb") as af:
        af.write(audio_data)
    audio_paths.append(audio_path)

df["audio"] = audio_paths

# Create train/validation split
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
split_idx = int(0.8 * len(df))
train_df = df.iloc[:split_idx]
valid_df = df.iloc[split_idx:]

# Create HuggingFace datasets
train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(valid_df)
dataset_dict = DatasetDict({
    "train": train_dataset,
    "validation": valid_dataset
})

# Save to disk
output_dir = os.path.join(project_name, "autotrain-data")
os.makedirs(output_dir, exist_ok=True)
dataset_dict.save_to_disk(output_dir)
```

## Configuration Management

### 1. Local Dataset Config (`configs/automatic_speech_recognition/local_dataset.yml`)

```yaml
task: ASR
base_model: facebook/wav2vec2-base-960h
project_name: autotrain-asr-local
log: tensorboard
backend: local

data:
  path: data/ 
  train_split: train
  valid_split: valid
  column_mapping:
    audio_column: audio 
    text_column: transcription 

params:
  max_duration: 30.0
  sampling_rate: 16000
  max_seq_length: 128
  epochs: 3
  batch_size: 8
  lr: 3e-4
  optimizer: adamw_torch
  scheduler: linear
  gradient_accumulation: 1
  mixed_precision: fp16
  weight_decay: 0.01
  warmup_ratio: 0.1
  early_stopping_patience: 3
  early_stopping_threshold: 0.01

hub:
  username: ${HF_USERNAME}
  token: ${HF_TOKEN}
  push_to_hub: true 
```

### 2. Hub Dataset Config (`configs/automatic_speech_recognition/hub_dataset.yml`)

```yaml
task: ASR
base_model: facebook/wav2vec2-base-960h
project_name: autotrain-asr-hub
log: tensorboard
backend: local

data:
  path: common_voice 
  train_split: train 
  valid_split: validation 
  column_mapping:
    audio_column: audio 
    text_column: transcription 
  using_hub_dataset: true

# ... same params as local
```

## Common vs Task-Specific Code

### Common Code (Shared Across All Tasks)

1. **App Structure** (`src/autotrain/app/`)
   - `app.py`: FastAPI application setup
   - `api_routes.py`: Common API endpoints
   - `ui_routes.py`: UI routing (with task-specific handling)
   - `models.py`: Database models
   - `db.py`: Database operations
   - `oauth.py`: Authentication

2. **Backend Infrastructure** (`src/autotrain/backends/`)
   - `base.py`: Base backend class
   - `local.py`: Local training backend
   - `spaces.py`: HuggingFace Spaces backend
   - `endpoints.py`: Endpoints backend

3. **Project Management** (`src/autotrain/project.py`)
   - `AutoTrainProject`: Main project class
   - Task-specific munge functions
   - Backend selection logic

4. **Dataset Infrastructure** (`src/autotrain/dataset.py`)
   - `AutoTrainDataset`: Base dataset class
   - Task-specific dataset classes
   - Common preprocessing logic

5. **CLI Infrastructure** (`src/autotrain/cli/`)
   - `run.py`: Base CLI class
   - `utils.py`: CLI utilities
   - Task-specific command files

6. **Common Utilities**
   - `logger.py`: Logging setup
   - `utils.py`: Common utilities
   - `params.py`: Base parameter classes

### ASR-Specific Code

1. **ASR Trainer** (`src/autotrain/trainers/automatic_speech_recognition/`)
   - `__main__.py`: Training entry point
   - `dataset.py`: ASR dataset class
   - `params.py`: ASR parameters
   - `utils.py`: ASR utilities

2. **ASR Preprocessor** (`src/autotrain/preprocessor/automatic_speech_recognition.py`)
   - Audio and text preprocessing
   - Dataset splitting and preparation

3. **ASR CLI** (`src/autotrain/cli/run_automatic_speech_recognition.py`)
   - ASR-specific command-line interface

4. **ASR Configs** (`configs/automatic_speech_recognition/`)
   - Local and hub dataset configurations

5. **UI ASR Logic** (`src/autotrain/app/ui_routes.py`)
   - LiFE App integration
   - ZIP+CSV+audio handling
   - ASR-specific form processing

## Complete Workflow from UI to Training

### Step 1: User Interface Interaction

1. **User selects ASR task** in dropdown
   - JavaScript shows ASR-specific column mappings
   - LiFE App dataset source option becomes visible

2. **User selects dataset source**:
   - **Local**: Upload ZIP file with CSV and audio folder
   - **HuggingFace Hub**: Enter dataset name and splits
   - **LiFE App**: Select project and script (ASR only)

3. **User configures parameters**:
   - Basic/Full parameter modes
   - Model selection
   - Hardware selection
   - Training parameters

4. **User submits form** (`/ui/create_project`)

### Step 2: Form Processing (`src/autotrain/app/ui_routes.py`)

1. **Validate form data**:
   - Check task is ASR for LiFE App
   - Validate required fields
   - Check for running jobs

2. **Process dataset based on source**:

   **Local ZIP Processing**:
   ```python
   if task == "ASR" and file_extension.lower() == "zip":
       # Extract ZIP
       # Find CSV and audio folder
       # Update CSV with full audio paths
       # Validate file existence
   ```

   **LiFE App Processing**:
   ```python
   if data_source == "life_app":
       # Load JSON dataset
       # Convert base64 audio to files
       # Create train/validation split
       # Save as HuggingFace dataset
   ```

3. **Create AutoTrainProject**:
   ```python
   app_params = AppParams(...)
   params = app_params.munge()
   project = AutoTrainProject(params=params, backend=hardware)
   job_id = project.create()
   ```

### Step 3: Project Creation (`src/autotrain/project.py`)

1. **Process parameters**:
   ```python
   if self.process:
       processed_params = self._process_params_data()
       if isinstance(processed_params, AutomaticSpeechRecognitionParams):
           self.params = processed_params
   ```

2. **ASR munge data**:
   ```python
   def asr_munge_data(params, local):
       # Handle CSV/JSONL files
       # Create AutoTrainDataset
       # Update parameters
       # Handle LiFE App data
   ```

3. **Create backend runner**:
   ```python
   if self.backend.startswith("local"):
       runner = LocalRunner(params=self.params, backend=self.backend)
       return runner.create()
   ```

### Step 4: Local Backend Execution (`src/autotrain/backends/local.py`)

1. **ASR-specific training setup**:
   ```python
   if isinstance(self.params, AutomaticSpeechRecognitionParams):
       # Create training config file
       config_path = f"{self.params.project_name}/training_config.json"
       config_dict = self.params.dict()
       with open(config_path, "w") as f:
           json.dump(config_dict, f, indent=2)
   ```

2. **Launch training process**:
   ```python
   command = f"{sys.executable} -m autotrain.trainers.automatic_speech_recognition.__main__ --training_config \"{config_path}\""
   process = subprocess.Popen(command, ...)
   ```

3. **Add job to database**:
   ```python
   DB = AutoTrainDB("autotrain.db")
   DB.add_job(process.pid)
   ```

### Step 5: ASR Training Execution (`src/autotrain/trainers/automatic_speech_recognition/__main__.py`)

1. **Load training configuration**:
   ```python
   with open(args.training_config, "r") as f:
       config = json.load(f)
   train(config)
   ```

2. **Load dataset**:
   ```python
   def load_data(params, is_validation=False):
       if params.using_hub_dataset:
           dataset = load_dataset(...)
       else:
           # Load from local directory
           # Handle CSV files
           # Validate audio files
           # Create HuggingFace dataset
   ```

3. **Load model and processor**:
   ```python
   def load_model_and_processor(params):
       # Try different processor types
       # Try different model types
       # Move to device
   ```

4. **Create datasets**:
   ```python
   train_dataset = AutomaticSpeechRecognitionDataset(
       data=dataset,
       processor=processor,
       model=model,
       audio_column=config.audio_column,
       text_column=config.text_column,
       max_duration=config.max_duration,
       sampling_rate=config.sampling_rate,
   )
   ```

5. **Setup training**:
   ```python
   training_args = TrainingArguments(...)
   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=train_dataset,
       eval_dataset=valid_dataset_obj,
       callbacks=callbacks,
       data_collator=dynamic_padding_collator,
       compute_metrics=compute_metrics,
   )
   ```

6. **Execute training**:
   ```python
   trainer.train()
   ```

7. **Save model**:
   ```python
   trainer.save_model(config.project_name)
   processor.save_pretrained(config.project_name)
   ```

8. **Upload to Hub** (if enabled):
   ```python
   if config.push_to_hub:
       api = HfApi(token=config.token)
       api.create_repo(...)
       api.upload_folder(...)
   ```

### Step 6: Monitoring and Logging

1. **Training logs** written to `asr.log`
2. **Database tracking** of job status
3. **UI monitoring** via `/ui/logs` endpoint
4. **Job management** via `/ui/stop_training` endpoint

## Key Integration Points

### 1. Task ID Mapping
- ASR task ID: 32
- Mapped in `TASK_ID_TO_PARAMS_CLASS` in `src/autotrain/backends/local.py`

### 2. Parameter Class Integration
- `AutomaticSpeechRecognitionParams` integrated in `AutoTrainProject`
- Used in `_process_params_data()` method

### 3. Dataset Class Integration
- `AutomaticSpeechRecognitionDataset` used in training
- `AutomaticSpeechRecognitionPreprocessor` used in dataset preparation

### 4. CLI Integration
- `run_automatic_speech_recognition.py` registered in main CLI
- Follows same pattern as other task commands

### 5. UI Integration
- ASR task option in task dropdown
- ASR-specific column mappings
- LiFE App dataset source (ASR only)
- ZIP+CSV+audio file handling

### 6. Backend Integration
- ASR-specific handling in `LocalRunner`
- Same backend infrastructure as other tasks

## Summary

The ASR implementation in this repository follows the exact same patterns and structure as other tasks in the main AutoTrain Advanced repository. The implementation includes:

1. **Complete ASR training pipeline** with support for multiple model types
2. **Three dataset sources**: Local (ZIP+CSV+audio), HuggingFace Hub, and LiFE App
3. **Universal dataset class** that works with all ASR models
4. **Robust error handling** and fallback mechanisms
5. **Full UI integration** with dynamic column mapping and LiFE App support
6. **Standard AutoTrain workflow** from UI to training completion

The code maintains perfect compatibility with the main repository's architecture while adding ASR-specific functionality where needed. All common logic (UI, backend, params, config, logs, dataset handling) is shared, with only ASR-specific logic in ASR-specific files or blocks. 