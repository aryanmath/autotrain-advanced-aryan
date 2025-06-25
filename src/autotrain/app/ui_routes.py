import json
import os
import signal
import sys
import time
from typing import List, Optional, Dict, Any
from datetime import datetime
import subprocess
import sqlite3

import torch
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from huggingface_hub import repo_exists
from nvitop import Device

from autotrain import __version__, logger
from autotrain.app.db import AutoTrainDB
from autotrain.app.models import (
    fetch_models,
)
from autotrain.app.params import AppParams, get_task_params
from autotrain.app.utils import get_running_jobs, get_user_and_orgs, kill_process_by_pid, token_verification
from autotrain.dataset import (
    AutoTrainDataset,
    AutoTrainImageClassificationDataset,
    AutoTrainImageRegressionDataset,
    AutoTrainObjectDetectionDataset,
    AutoTrainVLMDataset,
)
from autotrain.help import get_app_help
from autotrain.project import AutoTrainProject


logger.info("Starting AutoTrain...")
HF_TOKEN = os.environ.get("HF_TOKEN", None)
IS_RUNNING_IN_SPACE = "SPACE_ID" in os.environ
ENABLE_NGC = int(os.environ.get("ENABLE_NGC", 0))
ENABLE_NVCF = int(os.environ.get("ENABLE_NVCF", 0))
AUTOTRAIN_LOCAL = int(os.environ.get("AUTOTRAIN_LOCAL", 1))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB = AutoTrainDB("autotrain.db")
MODEL_CHOICE = fetch_models()

ui_router = APIRouter()
templates_path = os.path.join(BASE_DIR, "templates")
templates = Jinja2Templates(directory=templates_path)

UI_PARAMS = {
    "mixed_precision": {
        "type": "dropdown",
        "label": "Mixed precision",
        "options": ["fp16", "bf16", "none"],
    },
    "optimizer": {
        "type": "dropdown",
        "label": "Optimizer",
        "options": ["adamw_torch", "adamw", "adam", "sgd"],
    },
    "scheduler": {
        "type": "dropdown",
        "label": "Scheduler",
        "options": ["linear", "cosine", "cosine_warmup", "constant"],
    },
    "eval_strategy": {
        "type": "dropdown",
        "label": "Evaluation strategy",
        "options": ["epoch", "steps"],
    },
    "logging_steps": {
        "type": "number",
        "label": "Logging steps",
    },
    "save_total_limit": {
        "type": "number",
        "label": "Save total limit",
    },
    "auto_find_batch_size": {
        "type": "dropdown",
        "label": "Auto find batch size",
        "options": [True, False],
    },
    "warmup_ratio": {
        "type": "number",
        "label": "Warmup proportion",
    },
    "max_grad_norm": {
        "type": "number",
        "label": "Max grad norm",
    },
    "weight_decay": {
        "type": "number",
        "label": "Weight decay",
    },
    "epochs": {
        "type": "number",
        "label": "Epochs",
    },
    "batch_size": {
        "type": "number",
        "label": "Batch size",
    },
    "lr": {
        "type": "number",
        "label": "Learning rate",
    },
    "seed": {
        "type": "number",
        "label": "Seed",
    },
    "gradient_accumulation": {
        "type": "number",
        "label": "Gradient accumulation",
    },
    "block_size": {
        "type": "number",
        "label": "Block size",
    },
    "model_max_length": {
        "type": "number",
        "label": "Model max length",
    },
    "add_eos_token": {
        "type": "dropdown",
        "label": "Add EOS token",
        "options": [True, False],
    },
    "disable_gradient_checkpointing": {
        "type": "dropdown",
        "label": "Disable GC",
        "options": [True, False],
    },
    "use_flash_attention_2": {
        "type": "dropdown",
        "label": "Use flash attention",
        "options": [True, False],
    },
    "log": {
        "type": "dropdown",
        "label": "Logging",
        "options": ["tensorboard", "none"],
    },
    "quantization": {
        "type": "dropdown",
        "label": "Quantization",
        "options": ["int4", "int8", "none"],
    },
    "target_modules": {
        "type": "string",
        "label": "Target modules",
    },
    "merge_adapter": {
        "type": "dropdown",
        "label": "Merge adapter",
        "options": [True, False],
    },
    "peft": {
        "type": "dropdown",
        "label": "PEFT/LoRA",
        "options": [True, False],
    },
    "lora_r": {
        "type": "number",
        "label": "Lora r",
    },
    "lora_alpha": {
        "type": "number",
        "label": "Lora alpha",
    },
    "lora_dropout": {
        "type": "number",
        "label": "Lora dropout",
    },
    "model_ref": {
        "type": "string",
        "label": "Reference model",
    },
    "dpo_beta": {
        "type": "number",
        "label": "DPO beta",
    },
    "max_prompt_length": {
        "type": "number",
        "label": "Prompt length",
    },
    "max_completion_length": {
        "type": "number",
        "label": "Completion length",
    },
    "chat_template": {
        "type": "dropdown",
        "label": "Chat template",
        "options": ["none", "zephyr", "chatml", "tokenizer"],
    },
    "padding": {
        "type": "dropdown",
        "label": "Padding side",
        "options": ["right", "left", "none"],
    },
    "max_seq_length": {
        "type": "number",
        "label": "Max sequence length",
    },
    "early_stopping_patience": {
        "type": "number",
        "label": "Early stopping patience",
    },
    "early_stopping_threshold": {
        "type": "number",
        "label": "Early stopping threshold",
    },
    "max_target_length": {
        "type": "number",
        "label": "Max target length",
    },
    "categorical_columns": {
        "type": "string",
        "label": "Categorical columns",
    },
    "numerical_columns": {
        "type": "string",
        "label": "Numerical columns",
    },
    "num_trials": {
        "type": "number",
        "label": "Number of trials",
    },
    "time_limit": {
        "type": "number",
        "label": "Time limit",
    },
    "categorical_imputer": {
        "type": "dropdown",
        "label": "Categorical imputer",
        "options": ["most_frequent", "none"],
    },
    "numerical_imputer": {
        "type": "dropdown",
        "label": "Numerical imputer",
        "options": ["mean", "median", "none"],
    },
    "numeric_scaler": {
        "type": "dropdown",
        "label": "Numeric scaler",
        "options": ["standard", "minmax", "maxabs", "robust", "none"],
    },
    "vae_model": {
        "type": "string",
        "label": "VAE model",
    },
    "prompt": {
        "type": "string",
        "label": "Prompt",
    },
    "resolution": {
        "type": "number",
        "label": "Resolution",
    },
    "num_steps": {
        "type": "number",
        "label": "Number of steps",
    },
    "checkpointing_steps": {
        "type": "number",
        "label": "Checkpointing steps",
    },
    "use_8bit_adam": {
        "type": "dropdown",
        "label": "Use 8-bit Adam",
        "options": [True, False],
    },
    "xformers": {
        "type": "dropdown",
        "label": "xFormers",
        "options": [True, False],
    },
    "image_square_size": {
        "type": "number",
        "label": "Image square size",
    },
    "unsloth": {
        "type": "dropdown",
        "label": "Unsloth",
        "options": [True, False],
    },
    "max_doc_stride": {
        "type": "number",
        "label": "Max doc stride",
    },
    "distributed_backend": {
        "type": "dropdown",
        "label": "Distributed backend",
        "options": ["ddp", "deepspeed"],
    },
    "audio_column": {
        "type": "string",
        "label": "Audio column",
    },
    "text_column": {
        "type": "string",
        "label": "Transcription column",
    },
    "max_duration": {
        "type": "number",
        "label": "Max audio duration (seconds)",
    },
    "sampling_rate": {
        "type": "number",
        "label": "Sampling rate (Hz)",
    },
}


def graceful_exit(signum, frame):
    """
    Handles the SIGTERM signal to perform cleanup and exit the program gracefully.

    Args:
        signum (int): The signal number.
        frame (FrameType): The current stack frame (or None).

    Logs:
        Logs the receipt of the SIGTERM signal and the initiation of cleanup.

    Exits:
        Exits the program with status code 0.
    """
    logger.info("SIGTERM received. Performing cleanup...")
    sys.exit(0)


signal.signal(signal.SIGTERM, graceful_exit)


logger.info("AutoTrain started successfully")


async def user_authentication(request: Request):
    """
    Authenticates the user based on the HF_TOKEN environment variable.

    Args:
        request (Request): The incoming HTTP request object.

    Returns:
        str: The authenticated token string.

    Raises:
        HTTPException: If the token is invalid or expired or verification fails.
    """
    if HF_TOKEN is None:
        logger.error("HF_TOKEN environment variable is not set.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="HF_TOKEN environment variable is not set.",
        )
        
    try:
        get_user_and_orgs(user_token=HF_TOKEN)
        return HF_TOKEN
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token: HF_TOKEN",
        )


@ui_router.get("/", response_class=HTMLResponse)
async def load_index(request: Request, token: str = Depends(user_authentication)):
    """
    This function is used to load the index page
    :return: HTMLResponse
    """
    if os.environ.get("SPACE_ID") == "autotrain-projects/autotrain-advanced":
        return templates.TemplateResponse("duplicate.html", {"request": request})
    try:
        _users = get_user_and_orgs(user_token=token)
    except Exception as e:
        logger.error(f"Failed to get user and orgs after authentication: {e}")
        return templates.TemplateResponse("login.html", {"request": request})
    context = {
        "request": request,
        "valid_users": _users,
        "enable_ngc": ENABLE_NGC,
        "enable_nvcf": ENABLE_NVCF,
        "enable_local": AUTOTRAIN_LOCAL,
        "version": __version__,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return templates.TemplateResponse("index.html", context)


@ui_router.get("/logout", response_class=HTMLResponse)
async def oauth_logout(request: Request, authenticated: bool = Depends(user_authentication)):
    """
    This function is used to logout the oauth user
    :return: HTMLResponse
    """
    request.session.pop("oauth_info", None)
    return RedirectResponse("/")


@ui_router.get("/params/{task}/{param_type}", response_class=JSONResponse)
async def fetch_params(task: str, param_type: str, authenticated: bool = Depends(user_authentication)):
    """
    This function is used to fetch the parameters for a given task
    :param task: str
    :param param_type: str (basic, full)
    :return: JSONResponse
    """
    logger.info(f"Task: {task}")
    task_params = get_task_params(task, param_type)
    if len(task_params) == 0:
        return {"error": "Task not found"}
    ui_params = {}
    for param in task_params:
        if param in UI_PARAMS:
            ui_params[param] = UI_PARAMS[param]
            ui_params[param]["default"] = task_params[param]
        else:
            logger.info(f"Param {param} not found in UI_PARAMS")

    ui_params = dict(sorted(ui_params.items(), key=lambda x: (x[1]["type"], x[1]["label"])))
    return ui_params


@ui_router.get("/model_choices/{task}", response_class=JSONResponse)
async def fetch_model_choices(
    task: str,
    custom_models: str = Query(None),
    authenticated: bool = Depends(user_authentication),
):
    """
    This function is used to fetch the model choices for a given task
    :param task: str
    :param custom_models: str (optional, comma separated list of custom models, query parameter)
    :return: JSONResponse
    """
    resp = []

    if custom_models is not None:
        custom_models = custom_models.split(",")
        for custom_model in custom_models:
            custom_model = custom_model.strip()
            resp.append({"id": custom_model, "name": custom_model})

    if os.environ.get("AUTOTRAIN_CUSTOM_MODELS", None) is not None:
        custom_models = os.environ.get("AUTOTRAIN_CUSTOM_MODELS")
        custom_models = custom_models.split(",")
        for custom_model in custom_models:
            custom_model = custom_model.strip()
            resp.append({"id": custom_model, "name": custom_model})

    if task == "text-classification":
        hub_models = MODEL_CHOICE["text-classification"]
    elif task.startswith("llm"):
        hub_models = MODEL_CHOICE["llm"]
    elif task.startswith("st:"):
        hub_models = MODEL_CHOICE["sentence-transformers"]
    elif task == "image-classification":
        hub_models = MODEL_CHOICE["image-classification"]
    elif task == "seq2seq":
        hub_models = MODEL_CHOICE["seq2seq"]
    elif task == "tabular:classification":
        hub_models = MODEL_CHOICE["tabular-classification"]
    elif task == "tabular:regression":
        hub_models = MODEL_CHOICE["tabular-regression"]
    elif task == "token-classification":
        hub_models = MODEL_CHOICE["token-classification"]
    elif task == "text-regression":
        hub_models = MODEL_CHOICE["text-regression"]
    elif task == "image-object-detection":
        hub_models = MODEL_CHOICE["image-object-detection"]
    elif task == "image-regression":
        hub_models = MODEL_CHOICE["image-regression"]
    elif task.startswith("vlm:"):
        hub_models = MODEL_CHOICE["vlm"]
    elif task == "extractive-qa":
        hub_models = MODEL_CHOICE["extractive-qa"]
    elif task == "ASR":
        hub_models = MODEL_CHOICE["ASR"]
    else:
        raise NotImplementedError

    for hub_model in hub_models:
        resp.append({"id": hub_model, "name": hub_model})
    return resp


@ui_router.post("/create_project", response_class=JSONResponse)
async def handle_form(
    request: Request,
    project_name: str = Form(...),
    task: str = Form(...),
    base_model: str = Form(...),
    hardware: str = Form(...),
    params: str = Form(...),
    autotrain_user: str = Form(...),
    column_mapping: str = Form('{"default": "value"}'),
    data_files_training: List[UploadFile] = File(None),
    data_files_valid: List[UploadFile] = File(None),
    hub_dataset: str = Form(""),
    train_split: str = Form(""),
    valid_split: str = Form(""),
    token: str = Depends(user_authentication),
    life_app_project: str = Form(""),
    life_app_script: str = Form(""),
):
    """
    Handle form submission for creating and managing AutoTrain projects.
    """
    logger.info("---------- handle_form (first definition) started ----------")
    logger.info(f"Incoming form data: {await request.form()}") # Log all incoming form data

    train_split = train_split.strip()
    if len(train_split) == 0:
        train_split = None

    valid_split = valid_split.strip()
    if len(valid_split) == 0:
        valid_split = None

    print(f"[DEBUG] hardware value from UI: {hardware}")
    logger.info(f"[DEBUG] hardware value from UI: {hardware}")

    if hardware == "local-ui":
        running_jobs = get_running_jobs(DB)
        if running_jobs:
            raise HTTPException(
                status_code=409, detail="Another job is already running. Please wait for it to finish."
            )

    if repo_exists(f"{autotrain_user}/{project_name}", token=token):
        raise HTTPException(
            status_code=409,
            detail=f"Project {project_name} already exists. Please choose a different name.",
        )

    params = json.loads(params)
    # convert "null" to None
    for key in params:
        if params[key] == "null":
            params[key] = None
    column_mapping = json.loads(column_mapping)

    training_files = [f.file for f in data_files_training if f.filename != ""] if data_files_training else []
    validation_files = [f.file for f in data_files_valid if f.filename != ""] if data_files_valid else []

    form = await request.form()
    data_source = form.get("dataset_source", "local")
    selected_project = form.get("life_app_project")
    selected_script = form.get("life_app_script")

    logger.info(f"LiFE App Data Source: {data_source}, Project: {selected_project}, Script: {selected_script}")

    # LiFE App dataset handling
    if data_source == "life_app":
        # TEMPORARY BYPASS: Set dummy values for project and script to bypass validation
        selected_project = "autotrain_dummy_project"
        selected_script = "autotrain_dummy_script"
        logger.info(f"TEMPORARY BYPASS ACTIVE: Project set to '{selected_project}', Script set to '{selected_script}'")

        if task != "ASR":
            raise HTTPException(
                status_code=400,
                detail="LiFE app datasets can only be used with Automatic Speech Recognition tasks"
            )
        if not selected_project or not selected_script:
            raise HTTPException(
                status_code=400,
                detail="Please select both a project and a script from LiFE app"
            )
        # Load dataset from dataset.json
        dataset_path = os.path.join(BASE_DIR, "static", "dataset.json")
        logger.info(f"Checking LiFE App dataset path: {dataset_path}, Exists: {os.path.exists(dataset_path)}")
        if not os.path.exists(dataset_path):
            raise HTTPException(
                status_code=400,
                detail="LiFE app dataset file not found"
            )
        import pandas as pd
        import base64
        from datasets import Dataset

        # Read dataset.json
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset_json = json.load(f)
        # Filter by selected project if needed (currently only one dataset)
        # For now, use all rows
        df = pd.DataFrame(dataset_json)
        # Save audio bytes to files
        audio_dir = os.path.join("life_app_data", "audio")
        os.makedirs(audio_dir, exist_ok=True)
        audio_paths = []
        for idx, row in df.iterrows():
            audio_bytes = row["audio"]
            # decode base64 if needed, else treat as bytes string
            try:
                audio_data = base64.b64decode(audio_bytes)
            except Exception:
                audio_data = audio_bytes.encode("latin1")
            audio_path = os.path.join(audio_dir, f"audio_{idx}.wav")
            with open(audio_path, "wb") as af:
                af.write(audio_data)
            audio_paths.append(audio_path)
        df["audio"] = audio_paths
        # Save processed CSV for reference
        processed_csv = os.path.join("life_app_data", "processed_dataset.csv")
        df.to_csv(processed_csv, index=False)
        # Save as HuggingFace Dataset
        hf_dataset = Dataset.from_pandas(df)
        # Save as arrow for training
        arrow_path = os.path.join("life_app_data", "dataset.arrow")
        hf_dataset.save_to_disk("life_app_data")
        # Set data_path to "life_app_data"
        data_path = "life_app_data"
        # Set splits to None (single dataset)
        train_split = None
        valid_split = None

        # Check if params is already a dict to avoid TypeError from redundant json.loads
        if not isinstance(params, dict):
            params = json.loads(params)

        # Update params for ASR
        params["audio_column"] = "audio"
        params["text_column"] = "transcription"
        params["data_path"] = data_path
        params["life_app_project"] = selected_project
        params["life_app_script"] = selected_script
        params["using_hub_dataset"] = False
        params["train_split"] = None
        params["valid_split"] = None
        # Set column mapping
        column_mapping = {"audio": "audio", "transcription": "transcription"}
        app_params = AppParams(
            job_params_json=json.dumps(params),
            token=token,
            project_name=project_name,
            username=autotrain_user,
            task=task,
            data_path=data_path,
            base_model=base_model,
            column_mapping=column_mapping,
            using_hub_dataset=False,
            train_split=None,
            valid_split=None,
        )
        params = app_params.munge()
        project = AutoTrainProject(params=params, backend=hardware)
        job_id = project.create()
        monitor_url = ""
        if hardware == "local-ui":
            DB.add_job(job_id)
            monitor_url = "Monitor your job locally / in logs"
        elif hardware.startswith("ep-"):
            monitor_url = f"https://ui.endpoints.huggingface.co/{autotrain_user}/endpoints/{job_id}"
        elif hardware.startswith("spaces-"):
            monitor_url = f"https://hf.co/spaces/{job_id}"
        else:
            monitor_url = f"Success! Monitor your job in logs. Job ID: {job_id}"
        return {"success": "true", "monitor_url": monitor_url}

    if len(training_files) > 0 and len(hub_dataset) > 0:
        raise HTTPException(
            status_code=400, detail="Please either upload a dataset or choose a dataset from the Hugging Face Hub."
        )
    elif len(training_files) == 0 and len(hub_dataset) == 0:
        raise HTTPException(
            status_code=400, detail="Please upload a dataset or choose a dataset from the Hugging Face Hub."
        )
    elif len(hub_dataset) > 0:
        if not train_split:
            raise HTTPException(status_code=400, detail="Please enter a training split.")
        data_path = hub_dataset
    else:
        # Handle local dataset upload
        file_extension = os.path.splitext(data_files_training[0].filename)[1]
        file_extension = file_extension[1:] if file_extension.startswith(".") else file_extension
        if task == "image-classification":
            dset = AutoTrainImageClassificationDataset(
                train_data=training_files[0],
                token=token,
                project_name=project_name,
                username=autotrain_user,
                valid_data=validation_files[0] if validation_files else None,
                percent_valid=None,
                local=hardware.lower() == "local-ui",
            )
        elif task == "image-regression":
            dset = AutoTrainImageRegressionDataset(
                train_data=training_files[0],
                token=token,
                project_name=project_name,
                username=autotrain_user,
                valid_data=validation_files[0] if validation_files else None,
                percent_valid=None,
                local=hardware.lower() == "local-ui",
            )
        elif task == "image-object-detection":
            dset = AutoTrainObjectDetectionDataset(
                train_data=training_files[0],
                token=token,
                project_name=project_name,
                username=autotrain_user,
                valid_data=validation_files[0] if validation_files else None,
                percent_valid=None,
                local=hardware.lower() == "local-ui",
            )
        elif task.startswith("vlm:"):
            dset = AutoTrainVLMDataset(
                train_data=training_files[0],
                token=token,
                project_name=project_name,
                username=autotrain_user,
                column_mapping=column_mapping,
                valid_data=validation_files[0] if validation_files else None,
                percent_valid=None,
                local=hardware.lower() == "local-ui",
            )
        else:
            if task.startswith("llm"):
                dset_task = "lm_training"
            elif task.startswith("st:"):
                dset_task = "sentence_transformers"
            elif task == "text-classification":
                dset_task = "text_multi_class_classification"
            elif task == "text-regression":
                dset_task = "text_single_column_regression"
            elif task == "seq2seq":
                dset_task = "seq2seq"
            elif task.startswith("tabular"):
                if "," in column_mapping["label"]:
                    column_mapping["label"] = column_mapping["label"].split(",")
                else:
                    column_mapping["label"] = [column_mapping["label"]]
                column_mapping["label"] = [col.strip() for col in column_mapping["label"]]
                subtask = task.split(":")[-1].lower()
                if len(column_mapping["label"]) > 1 and subtask == "classification":
                    dset_task = "tabular_multi_label_classification"
                elif len(column_mapping["label"]) == 1 and subtask == "classification":
                    dset_task = "tabular_multi_class_classification"
                elif len(column_mapping["label"]) > 1 and subtask == "regression":
                    dset_task = "tabular_multi_column_regression"
                elif len(column_mapping["label"]) == 1 and subtask == "regression":
                    dset_task = "tabular_single_column_regression"
                else:
                    raise NotImplementedError
            elif task == "token-classification":
                dset_task = "text_token_classification"
            elif task == "extractive-qa":
                dset_task = "text_extractive_question_answering"
            elif task == "ASR":
                dset_task = "ASR"
            else:
                raise NotImplementedError
            logger.info(f"Task: {dset_task}")
            logger.info(f"Column mapping: {column_mapping}")
            dset_args = dict(
                train_data=training_files,
                task=dset_task,
                token=token,
                project_name=project_name,
                username=autotrain_user,
                column_mapping=column_mapping,
                valid_data=validation_files,
                percent_valid=None,
                local=hardware.lower() == "local-ui",
                ext=file_extension,
            )
            if task in ("text-classification", "token-classification", "st:pair_class"):
                dset_args["convert_to_class_label"] = True
            dset = AutoTrainDataset(**dset_args)
        data_path = dset.prepare()

    app_params = AppParams(
        job_params_json=json.dumps(params),
        token=token,
        project_name=project_name,
        username=autotrain_user,
        task=task,
        data_path=data_path,
        base_model=base_model,
        column_mapping=column_mapping,
        using_hub_dataset=len(hub_dataset) > 0,
        train_split=None if len(hub_dataset) == 0 else train_split,
        valid_split=None if len(hub_dataset) == 0 else valid_split,
    )
    params = app_params.munge()
    project = AutoTrainProject(params=params, backend=hardware)
    job_id = project.create()
    monitor_url = ""
    
    # Handle ASR training specifically for local-ui
    if task == "ASR" and hardware == "local-ui":
        try:
            logger.info("ASR training detected - creating config file and starting training...")
            
            # Create config file
            config_path = f"{project_name}/training_config.json"
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            # Prepare config data
            config_data = {
                "model": base_model,
                "data_path": data_path,
                "audio_column": params.get("audio_column", "audio"),
                "text_column": params.get("text_column", "transcription"),
                "max_duration": params.get("max_duration", 30.0),
                "sampling_rate": params.get("sampling_rate", 16000),
                "batch_size": params.get("batch_size", 8),
                "epochs": params.get("epochs", 3),
                "lr": params.get("lr", 3e-5),
                "output_dir": f"{project_name}/output",
                "project_name": project_name,
                "username": autotrain_user,
                "token": token,
                "using_hub_dataset": len(hub_dataset) > 0,
                "train_split": train_split,
                "valid_split": valid_split,
                "mixed_precision": params.get("mixed_precision", "no"),
                "optimizer": params.get("optimizer", "adamw_torch"),
                "scheduler": params.get("scheduler", "linear"),
                "gradient_accumulation": params.get("gradient_accumulation", 1),
                "weight_decay": params.get("weight_decay", 0.01),
                "warmup_ratio": params.get("warmup_ratio", 0.1),
                "max_grad_norm": params.get("max_grad_norm", 1.0),
                "early_stopping_patience": params.get("early_stopping_patience", 3),
                "early_stopping_threshold": params.get("early_stopping_threshold", 0.01),
                "eval_strategy": params.get("eval_strategy", "epoch"),
                "save_total_limit": params.get("save_total_limit", 1),
                "auto_find_batch_size": params.get("auto_find_batch_size", False),
                "logging_steps": params.get("logging_steps", 10),
                "push_to_hub": False,
                "hub_model_id": None,
                "log": "tensorboard",
                "max_seq_length": 128,
                "seed": 42
            }
            
            # Save config file
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Config file created: {config_path}")
            
            # Run ASR training command
            WORKSPACE_ROOT = os.path.abspath(".")
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONPATH"] = os.path.join(WORKSPACE_ROOT, "src")
            
            # Use absolute path for config file
            abs_config_path = os.path.abspath(config_path)
            
            # Create command
            command = [
                sys.executable,
                "-m", 
                "autotrain.trainers.automatic_speech_recognition.__main__",
                "--training_config",
                abs_config_path
            ]
            
            logger.info(f"Running ASR command: {' '.join(command)}")
            logger.info(f"Current working directory: {os.getcwd()}")
            logger.info(f"Config path: {abs_config_path}, exists: {os.path.exists(abs_config_path)}")
            
            # Start subprocess
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1,
                env=env,
                cwd=WORKSPACE_ROOT
            )
            
            # Get process ID
            pid = process.pid
            
            # Add job to database
            try:
                DB.add_job(pid)
                logger.info(f"Added ASR job with PID {pid} to database")
            except sqlite3.IntegrityError:
                try:
                    kill_process_by_pid(pid)
                except:
                    pass
                DB.remove_job(pid)
                DB.add_job(pid)
            
            logger.info(f"ASR training started successfully with PID: {pid}")
            monitor_url = f"ASR training started with PID: {pid}. Check logs for progress."
            
            # Start background monitoring
            def monitor_asr_process():
                logger.info("Starting ASR process monitoring...")
                try:
                    while True:
                        if process.poll() is not None:
                            logger.info(f"ASR process finished with return code: {process.returncode}")
                            break
                        
                        stdout_line = process.stdout.readline()
                        if stdout_line:
                            logger.info(f"[ASR STDOUT] {stdout_line.strip()}")
                        
                        stderr_line = process.stderr.readline()
                        if stderr_line:
                            logger.error(f"[ASR STDERR] {stderr_line.strip()}")
                        
                        time.sleep(0.1)
                        
                except Exception as e:
                    logger.error(f"Error in ASR process monitoring: {str(e)}")
                finally:
                    logger.info("ASR process monitoring finished")

            import threading
            monitor_thread = threading.Thread(target=monitor_asr_process, daemon=True)
            monitor_thread.start()
            
        except Exception as e:
            logger.error(f"Error starting ASR training: {str(e)}")
            monitor_url = f"Error starting ASR training: {str(e)}"
    else:
        # Original logic for other tasks
        if hardware == "local-ui":
            DB.add_job(job_id)
            monitor_url = "Monitor your job locally / in logs"
        elif hardware.startswith("ep-"):
            monitor_url = f"https://ui.endpoints.huggingface.co/{autotrain_user}/endpoints/{job_id}"
        elif hardware.startswith("spaces-"):
            monitor_url = f"https://hf.co/spaces/{job_id}"
        else:
            monitor_url = f"Success! Monitor your job in logs. Job ID: {job_id}"

    return {"success": "true", "monitor_url": monitor_url}


@ui_router.get("/help/{element_id}", response_class=JSONResponse)
async def fetch_help(element_id: str, authenticated: bool = Depends(user_authentication)):
    """
    This function is used to fetch the help text for a given element
    :param element_id: str
    :return: JSONResponse
    """
    msg = get_app_help(element_id)
    return {"message": msg}


@ui_router.get("/accelerators", response_class=JSONResponse)
async def available_accelerators(authenticated: bool = Depends(user_authentication)):
    """
    This function is used to fetch the number of available accelerators
    :return: JSONResponse
    """
    if AUTOTRAIN_LOCAL == 0:
        return {"accelerators": "Not available in cloud mode."}
    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()
    if cuda_available:
        num_gpus = torch.cuda.device_count()
    elif mps_available:
        num_gpus = 1
    else:
        num_gpus = 0
    return {"accelerators": num_gpus}


@ui_router.get("/is_model_training", response_class=JSONResponse)
async def is_model_training(authenticated: bool = Depends(user_authentication)):
    """
    This function is used to fetch the number of running jobs
    :return: JSONResponse
    """
    if AUTOTRAIN_LOCAL == 0:
        return {"model_training": "Not available in cloud mode."}
    running_jobs = get_running_jobs(DB)
    if running_jobs:
        return {"model_training": True, "pids": running_jobs}
    return {"model_training": False, "pids": []}


@ui_router.get("/logs", response_class=JSONResponse)
async def fetch_logs(authenticated: bool = Depends(user_authentication)):
    """
    This function is used to fetch the logs
    :return: JSONResponse
    """
    if not AUTOTRAIN_LOCAL:
        return {"logs": "Logs are only available in local mode."}
    log_file = "autotrain.log"
    with open(log_file, "r", encoding="utf-8") as f:
        logs = f.read()
    if len(str(logs).strip()) == 0:
        logs = "No logs available."

    logs = logs.split("\n")
    logs = logs[::-1]
    # remove lines containing /is_model_training & /accelerators
    logs = [log for log in logs if "/ui/" not in log and "/static/" not in log and "nvidia-ml-py" not in log]

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        devices = Device.all()
        device_logs = []
        for device in devices:
            device_logs.append(
                f"Device {device.index}: {device.name()} - {device.memory_used_human()}/{device.memory_total_human()}"
            )
        device_logs.append("-----------------")
        logs = device_logs + logs
    return {"logs": logs}


@ui_router.get("/stop_training", response_class=JSONResponse)
async def stop_training(authenticated: bool = Depends(user_authentication)):
    """
    This function is used to stop the training
    :return: JSONResponse
    """
    running_jobs = get_running_jobs(DB)
    if running_jobs:
        for _pid in running_jobs:
            try:
                kill_process_by_pid(_pid)
            except Exception:
                logger.info(f"Process {_pid} is already completed. Skipping...")
        return {"success": True}
    return {"success": False}


@ui_router.get("/life_app_projects", response_class=JSONResponse)
async def get_life_app_projects(authenticated: bool = Depends(user_authentication)):
    """
    Returns the list of projects from the local JSON file for LiFE App integration.
    """
    project_list_path = os.path.join(BASE_DIR, "static", "projectList.json")
    if not os.path.exists(project_list_path):
        return JSONResponse(content={"projects": []})
    with open(project_list_path, "r", encoding="utf-8") as f:
        projects = json.load(f)
    return {"projects": projects}

@ui_router.get("/life_app_scripts", response_class=JSONResponse)
async def get_life_app_scripts(authenticated: bool = Depends(user_authentication)):
    """
    Returns the list of scripts from the local JSON file for LiFE App integration.
    """
    script_list_path = os.path.join(BASE_DIR, "static", "scriptList.json")
    if not os.path.exists(script_list_path):
        return JSONResponse(content={"scripts": []})
    with open(script_list_path, "r", encoding="utf-8") as f:
        scripts = json.load(f)
    return {"scripts": scripts}

@ui_router.get("/life_app_dataset", response_class=JSONResponse)
async def get_life_app_dataset(authenticated: bool = Depends(user_authentication)):
    """
    Returns the dataset from the local JSON file for LiFE App integration.
    """
    dataset_path = os.path.join(BASE_DIR, "static", "dataset.json")
    if not os.path.exists(dataset_path):
        return JSONResponse(content={"dataset": []})
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return {"dataset": dataset}

@ui_router.post("/project_selected", response_class=JSONResponse)
async def handle_project_selection(request: Request, authenticated: bool = Depends(user_authentication)):
    """
    Handle project selection and return corresponding scripts based on project-script mapping.
    """
    try:
        # Request se selected projects get karna
        data = await request.json()
        selected_projects = data.get('projects', [])
        
        # Log selected projects
        logger.info(f"Projects selected: {selected_projects}")
        
        # Load project-script mapping
        mapping_path = os.path.join(BASE_DIR, "static", "project_script_mapping.json")
        if not os.path.exists(mapping_path):
            logger.error("Project-script mapping file not found")
            return JSONResponse(content={"scripts": []})
            
        with open(mapping_path, "r", encoding="utf-8") as f:
            project_script_mapping = json.load(f)
        
        # Get scripts for selected projects
        available_scripts = set()
        for project in selected_projects:
            if project in project_script_mapping:
                available_scripts.update(project_script_mapping[project])
        
        # Convert set to list for JSON response
        scripts = list(available_scripts)
        
        # Log available scripts
        logger.info(f"Available scripts for projects {selected_projects}: {scripts}")
        
        # Return scripts
        return JSONResponse(content={
            "status": "success",
            "projects": selected_projects,
            "scripts": scripts
        })
    except Exception as e:
        logger.error(f"Error in project selection: {str(e)}")
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )

@ui_router.post("/script_selected", response_class=JSONResponse)
async def handle_script_selection(request: Request, authenticated: bool = Depends(user_authentication)):
    """
    Handle script selection and return corresponding datasets based on script-dataset mapping.
    """
    try:
        data = await request.json()
        selected_script = data.get('script', '')
        logger.info(f"Script selected (received from frontend): {selected_script}")
        print(f"[BACKEND] Script selected (received from frontend): {selected_script}")

        mapping_path = os.path.join(BASE_DIR, "static", "script_dataset_mapping.json")
        if not os.path.exists(mapping_path):
            logger.error("Script-dataset mapping file not found")
            return JSONResponse(content={"datasets": []})

        with open(mapping_path, "r", encoding="utf-8") as f:
            script_dataset_mapping = json.load(f)

        datasets = script_dataset_mapping.get(selected_script, [])
        logger.info(f"Datasets fetched for script {selected_script}: {datasets}")
        print(f"[BACKEND] Datasets fetched for script {selected_script}: {datasets}")

        return JSONResponse(content={
            "status": "success",
            "script": selected_script,
            "datasets": datasets
        })
    except Exception as e:
        logger.error(f"Error in script selection: {str(e)}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

