# import json
# import os
# import subprocess

# from autotrain.commands import launch_command
# from autotrain.trainers.clm.params import LLMTrainingParams
# from autotrain.trainers.extractive_question_answering.params import ExtractiveQuestionAnsweringParams
# from autotrain.trainers.generic.params import GenericParams
# from autotrain.trainers.image_classification.params import ImageClassificationParams
# from autotrain.trainers.image_regression.params import ImageRegressionParams
# from autotrain.trainers.object_detection.params import ObjectDetectionParams
# from autotrain.trainers.sent_transformers.params import SentenceTransformersParams
# from autotrain.trainers.seq2seq.params import Seq2SeqParams
# from autotrain.trainers.tabular.params import TabularParams
# from autotrain.trainers.text_classification.params import TextClassificationParams
# from autotrain.trainers.text_regression.params import TextRegressionParams
# from autotrain.trainers.token_classification.params import TokenClassificationParams
# from autotrain.trainers.vlm.params import VLMTrainingParams
# from autotrain.trainers.asr.params import ASRParams  # Add this import

# ALLOW_REMOTE_CODE = os.environ.get("ALLOW_REMOTE_CODE", "true").lower() == "true"

# def run_training(params, task_id, local=False, wait=False):
#     """
#     Run the training process based on the provided parameters and task ID.

#     Args:
#         params (str): JSON string of the parameters required for training.
#         task_id (int): Identifier for the type of task to be performed.
#         local (bool, optional): Flag to indicate if the training should be run locally. Defaults to False.
#         wait (bool, optional): Flag to indicate if the function should wait for the process to complete. Defaults to False.

#     Returns:
#         int: Process ID of the launched training process.

#     Raises:
#         NotImplementedError: If the task_id does not match any of the predefined tasks.
#     """
#     params = json.loads(params)
#     if isinstance(params, str):
#         params = json.loads(params)
#     if task_id == 9:
#         params = LLMTrainingParams(**params)
#     elif task_id == 28:
#         params = Seq2SeqParams(**params)
#     elif task_id in (1, 2):
#         params = TextClassificationParams(**params)
#     elif task_id in (13, 14, 15, 16, 26):
#         params = TabularParams(**params)
#     elif task_id == 27:
#         params = GenericParams(**params)
#     elif task_id == 18:
#         params = ImageClassificationParams(**params)
#     elif task_id == 4:
#         params = TokenClassificationParams(**params)
#     elif task_id == 10:
#         params = TextRegressionParams(**params)
#     elif task_id == 29:
#         params = ObjectDetectionParams(**params)
#     elif task_id == 30:
#         params = SentenceTransformersParams(**params)
#     elif task_id == 24:
#         params = ImageRegressionParams(**params)
#     elif task_id == 31:
#         params = VLMTrainingParams(**params)
#     elif task_id == 5:
#         params = ExtractiveQuestionAnsweringParams(**params)
#     elif task_id == 32:  # Add this condition for ASR
#         params = ASRParams(**params)
#     else:
#         raise NotImplementedError

#     params.save(output_dir=params.project_name)
#     cmd = launch_command(params=params)
#     cmd = [str(c) for c in cmd]
#     env = os.environ.copy()
#     process = subprocess.Popen(cmd, env=env)
#     if wait:
#         process.wait()
#     return process.pid



# # import json
# # import os
# # import subprocess

# # from autotrain.commands import launch_command
# # from autotrain.trainers.clm.params import LLMTrainingParams
# # from autotrain.trainers.extractive_question_answering.params import ExtractiveQuestionAnsweringParams
# # from autotrain.trainers.generic.params import GenericParams
# # from autotrain.trainers.image_classification.params import ImageClassificationParams
# # from autotrain.trainers.image_regression.params import ImageRegressionParams
# # from autotrain.trainers.object_detection.params import ObjectDetectionParams
# # from autotrain.trainers.sent_transformers.params import SentenceTransformersParams
# # from autotrain.trainers.seq2seq.params import Seq2SeqParams
# # from autotrain.trainers.tabular.params import TabularParams
# # from autotrain.trainers.text_classification.params import TextClassificationParams
# # from autotrain.trainers.text_regression.params import TextRegressionParams
# # from autotrain.trainers.token_classification.params import TokenClassificationParams
# # from autotrain.trainers.vlm.params import VLMTrainingParams


# # ALLOW_REMOTE_CODE = os.environ.get("ALLOW_REMOTE_CODE", "true").lower() == "true"


# # def run_training(params, task_id, local=False, wait=False):
# #     """
# #     Run the training process based on the provided parameters and task ID.

# #     Args:
# #         params (str): JSON string of the parameters required for training.
# #         task_id (int): Identifier for the type of task to be performed.
# #         local (bool, optional): Flag to indicate if the training should be run locally. Defaults to False.
# #         wait (bool, optional): Flag to indicate if the function should wait for the process to complete. Defaults to False.

# #     Returns:
# #         int: Process ID of the launched training process.

# #     Raises:
# #         NotImplementedError: If the task_id does not match any of the predefined tasks.
# #     """
# #     params = json.loads(params)
# #     if isinstance(params, str):
# #         params = json.loads(params)
# #     if task_id == 9:
# #         params = LLMTrainingParams(**params)
# #     elif task_id == 28:
# #         params = Seq2SeqParams(**params)
# #     elif task_id in (1, 2):
# #         params = TextClassificationParams(**params)
# #     elif task_id in (13, 14, 15, 16, 26):
# #         params = TabularParams(**params)
# #     elif task_id == 27:
# #         params = GenericParams(**params)
# #     elif task_id == 18:
# #         params = ImageClassificationParams(**params)
# #     elif task_id == 4:
# #         params = TokenClassificationParams(**params)
# #     elif task_id == 10:
# #         params = TextRegressionParams(**params)
# #     elif task_id == 29:
# #         params = ObjectDetectionParams(**params)
# #     elif task_id == 30:
# #         params = SentenceTransformersParams(**params)
# #     elif task_id == 24:
# #         params = ImageRegressionParams(**params)
# #     elif task_id == 31:
# #         params = VLMTrainingParams(**params)
# #     elif task_id == 5:
# #         params = ExtractiveQuestionAnsweringParams(**params)
# #     else:
# #         raise NotImplementedError

# #     params.save(output_dir=params.project_name)
# #     cmd = launch_command(params=params)
# #     cmd = [str(c) for c in cmd]
# #     env = os.environ.copy()
# #     process = subprocess.Popen(cmd, env=env)
# #     if wait:
# #         process.wait()
# #     return process.pid

import os
import signal
import sys

import psutil
import requests

from autotrain import config, logger
from autotrain.trainers.asr.__main__ import train as asr_train  # Import ASR training function

def run_training(params, task_id, local=False, wait=False):
    """
    Run the training process for the specified task.

    Args:
        params (dict): Training parameters.
        task_id (str): The task identifier (e.g., "asr", "text_classification").
        local (bool): Whether to run the training locally.
        wait (bool): Whether to wait for the training to complete.

    Returns:
        int or None: The process ID if running locally and not waiting, otherwise None.
    """
    if task_id == "asr":
        # Convert params to dict if necessary
        if not isinstance(params, dict):
            params = params.dict() if hasattr(params, "dict") else vars(params)
        asr_train(params)
        if local and not wait:
            return os.getpid()  # Return PID for local training
        return None
    else:
        raise NotImplementedError(f"Task {task_id} is not implemented.")

def graceful_exit(signum, frame):
    logger.info("SIGTERM received. Performing cleanup...")
    sys.exit(0)

signal.signal(signal.SIGTERM, graceful_exit)

def get_running_jobs(db):
    """
    Retrieves and manages running jobs from the database.

    Args:
        db: A database object that provides methods to get and delete running jobs.

    Returns:
        list: An updated list of running jobs from the database.
    """
    running_jobs = db.get_running_jobs()
    if running_jobs:
        for _pid in running_jobs:
            proc_status = get_process体の_status(_pid)
            proc_status = proc_status.strip().lower()
            if proc_status in ("completed", "error", "zombie"):
                logger.info(f"Killing PID: {_pid}")
                try:
                    kill_process_by_pid(_pid)
                except Exception as e:
                    logger.info(f"Error while killing process: {e}")
                    logger.info(f"Process {_pid} is already completed. Skipping...")
                db.delete_job(_pid)
    running_jobs = db.get_running_jobs()
    return running_jobs

def get_process_status(pid):
    """
    Retrieve the status of a process given its PID.

    Args:
        pid (int): The process ID of the process to check.

    Returns:
        str: The status of the process. If the process does not exist, returns "Completed".
    """
    try:
        process = psutil.Process(pid)
        proc_status = process.status()
        return proc_status
    except psutil.NoSuchProcess:
        logger.info(f"No process found with PID: {pid}")
        return "Completed"

def kill_process_by_pid(pid):
    """
    Kill a process by its PID (Process ID).

    Args:
        pid (int): The Process ID of the process to be terminated.
    """
    try:
        os.kill(pid, signal.SIGTERM)
        logger.info(f"Sent SIGTERM to process with PID {pid}")
    except ProcessLookupError:
        logger.error(f"No process found with PID {pid}")
    except Exception as e:
        logger.error(f"Failed to send SIGTERM to process with PID {pid}: {e}")

def token_verification(token):
    """
    Verifies the provided token with the Hugging Face API and retrieves user information.

    Args:
        token (str): The token to be verified.

    Returns:
        dict: A dictionary containing user information.
    """
    if token.startswith("hf_oauth"):
        _api_url = config.HF_API + "/oauth/userinfo"
        _err_msg = "/oauth/userinfo"
    else:
        _api_url = config.HF_API + "/api/whoami-v2"
        _err_msg = "/api/whoami-v2"
    headers = {}
    cookies = {}
    if token.startswith("hf_"):
        headers["Authorization"] = f"Bearer {token}"
    else:
        cookies = {"token": token}
    try:
        response = requests.get(_api_url, headers=headers, cookies=cookies, timeout=3)
    except (requests.Timeout, ConnectionError) as err:
        logger.error(f"Failed to request {_err_msg} - {repr(err)}")
        raise Exception(f"Hugging Face Hub ({_err_msg}) is unreachable, please try again later.")
    if response.status_code != 200:
        logger.error(f"Failed to request {_err_msg} - {response.status_code}")
        raise Exception(f"Invalid token ({_err_msg}). Please login with a write token.")
    resp = response.json()
    user_info = {}
    if token.startswith("hf_oauth"):
        user_info["id"] = resp["sub"]
        user_info["name"] = resp["preferred_username"]
        user_info["orgs"] = [resp["orgs"][k]["preferred_username"] for k in range(len(resp["orgs"]))]
    else:
        user_info["id"] = resp["id"]
        user_info["name"] = resp["name"]
        user_info["orgs"] = [resp["orgs"][k]["name"] for k in range(len(resp["orgs"]))]
    return user_info

def get_user_and_orgs(user_token):
    """
    Retrieve the username and organizations associated with the provided user token.

    Args:
        user_token (str): The token used to authenticate the user.

    Returns:
        list: A list containing the username followed by the organizations.
    """
    if user_token is None or len(user_token) == 0:
        raise Exception("Invalid token. Please login with a write token.")
    user_info = token_verification(token=user_token)
    username = user_info["name"]
    orgs = user_info["orgs"]
    who_is_training = [username] + orgs
    return who_is_training
