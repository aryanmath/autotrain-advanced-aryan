from autotrain import logger
from autotrain.backends.base import BaseBackend
from autotrain.utils import run_training
import json
import os
import subprocess
import threading
import time
import sys

# Patch sys.stdout and sys.stderr to utf-8 for Unicode logs
import io
if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if not isinstance(sys.stderr, io.TextIOWrapper) or sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Import all parameter classes
from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.extractive_question_answering.params import ExtractiveQuestionAnsweringParams
from autotrain.trainers.generic.params import GenericParams
from autotrain.trainers.image_classification.params import ImageClassificationParams
from autotrain.trainers.image_regression.params import ImageRegressionParams
from autotrain.trainers.object_detection.params import ObjectDetectionParams
from autotrain.trainers.sent_transformers.params import SentenceTransformersParams
from autotrain.trainers.seq2seq.params import Seq2SeqParams
from autotrain.trainers.tabular.params import TabularParams
from autotrain.trainers.text_classification.params import TextClassificationParams
from autotrain.trainers.text_regression.params import TextRegressionParams
from autotrain.trainers.token_classification.params import TokenClassificationParams
from autotrain.trainers.vlm.params import VLMTrainingParams
from autotrain.trainers.automatic_speech_recognition.params import AutomaticSpeechRecognitionParams


# Mapping of task_id to parameter class
TASK_ID_TO_PARAMS_CLASS = {
    9: LLMTrainingParams,
    2: TextClassificationParams,
    26: TabularParams,
    27: GenericParams,
    28: Seq2SeqParams,
    18: ImageClassificationParams,
    4: TokenClassificationParams,
    10: TextRegressionParams,
    29: ObjectDetectionParams,
    30: SentenceTransformersParams,
    24: ImageRegressionParams,
    31: VLMTrainingParams,
    5: ExtractiveQuestionAnsweringParams,
    32: AutomaticSpeechRecognitionParams,
}


class LocalRunner(BaseBackend):
    """
    LocalRunner is a class that inherits from BaseBackend and is responsible for managing local training tasks.

    Methods:
        create():
            Starts the local training process by retrieving parameters and task ID from environment variables.
            Logs the start of the training process.
            Runs the training with the specified parameters and task ID.
            If the `wait` attribute is False, logs the training process ID (PID).
            Returns the training process ID (PID).
    """

    def _setup(self):
        """Setup the training environment."""
        logger.info("Setting up local training environment...")
        # No special setup needed for local training
        pass

    def _prepare(self):
        """Prepare the training environment."""
        logger.info("Preparing local training environment...")
        # No special preparation needed for local training
        pass

    def _validate(self):
        """Validate the training configuration."""
        logger.info("Validating local training configuration...")
        # No special validation needed for local training
        pass

    def _monitor(self):
        """Monitor the training process."""
        logger.info("Monitoring local training process...")
        # No special monitoring needed for local training
        pass

    def create(self):
        """Create a new training job."""
        self._setup()
        self._prepare()
        self._validate()
        self._create()
        self._monitor()
        return self.job_id

    def _create(self):
        """Create the training job."""
        logger.info("Starting local training...")
        
        # Handle ASR training differently
        if isinstance(self.params, AutomaticSpeechRecognitionParams):
            # Save params to a temporary config file
            config_path = f"{self.params.project_name}/training_config.json"
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            # Convert params to dict and save
            config_dict = self.params.dict()
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2)
            
            # Construct the training command
            WORKSPACE_ROOT = os.path.abspath(".")  
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            command = f"{sys.executable} -m autotrain.trainers.automatic_speech_recognition.__main__ --training_config \"{config_path}\""

            print(f"[DEBUG] Subprocess command: {command}")
            logger.info(f"Running ASR command: {command}")
            logger.info(f"Current working directory: {os.getcwd()}")
            logger.info(f"Config path: {config_path}, exists: {os.path.exists(config_path)}")
            logger.info(f"Python executable: {sys.executable}")
            logger.info(f"Environment PATH: {os.environ.get('PATH')}")
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                env=env,
                cwd=WORKSPACE_ROOT
            )
            
            def tee_stream(stream, log_file_path):
                with open(log_file_path, "a", encoding="utf-8") as log_file:
                    for line in iter(stream.readline, ''):
                        print(line, end='')
                        log_file.write(line)
                        log_file.flush()
                stream.close()

            stdout_thread = threading.Thread(target=tee_stream, args=(process.stdout, "asr_stdout.log"))
            stderr_thread = threading.Thread(target=tee_stream, args=(process.stderr, "asr_stderr.log"))
            stdout_thread.start()
            stderr_thread.start()

            # Register the ASR job PID in the database immediately after starting the process
            from autotrain.app.db import AutoTrainDB
            DB = AutoTrainDB("autotrain.db")
            DB.add_job(process.pid)
            
            self.job_id = str(process.pid)
            logger.info(f"ASR Training started with PID: {self.job_id}")

            # # Start a background thread to monitor the process
            # def monitor_process():
            #     while True:
            #         stdout_line = process.stdout.readline()
            #         if stdout_line:
            #             logger.info(f"[ASR STDOUT] {stdout_line.strip()}")
            #         stderr_line = process.stderr.readline()
            #         if stderr_line:
            #             logger.error(f"[ASR STDERR] {stderr_line.strip()}")
            #         if process.poll() is not None:
            #             break

            # monitor_thread = threading.Thread(target=monitor_process, daemon=True)
            # monitor_thread.start()
                
            return
            
        # Original code for other tasks
        params_json = self.env_vars["PARAMS"]
        task_id = int(self.env_vars["TASK_ID"])
        
        # Deserialize the JSON string into the correct parameter object
        params_dict = json.loads(params_json)
        params_class = TASK_ID_TO_PARAMS_CLASS.get(task_id)
        
        if params_class is None:
            raise ValueError(f"Unknown task ID: {task_id}")
            
        params_object = params_class(**params_dict)

        training_pid = run_training(params_object, task_id, wait=self.wait)
        if not self.wait:
            logger.info(f"Training PID: {training_pid}")
        return training_pid

    def monitor_process(self, process, pid, db):
        """
        Monitor a training process.
        
        Args:
            process: The process to monitor
            pid (int): Process ID
            db: Database connection
        """
        try:
            # Monitor process output
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    logger.info(f"ASR Trainer: {output.strip()}")
                    
            # Get return code
            return_code = process.poll()
            
            # Check if process failed
            if return_code != 0:
                logger.error("ASR Trainer failed")
                try:
                    if hasattr(db, 'delete_job'):
                        db.delete_job(pid)
                except Exception as e:
                    logger.error(f"Error deleting job from database: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error monitoring process: {str(e)}")
            try:
                if hasattr(db, 'delete_job'):
                    db.delete_job(pid)
            except Exception as e:
                logger.error(f"Error deleting job from database: {str(e)}")
                
        finally:
            # Cleanup
            try:
                process.stdout.close()
                process.stderr.close()
                if hasattr(db, 'conn'):
                    db.conn.close()
                if hasattr(db, 'close'):
                    db.close()
                if hasattr(db, '__exit__'):
                    db.__exit__(None, None, None)
                if hasattr(db, '__del__'):
                    db.__del__()
                if hasattr(db, '__enter__'):
                    db.__enter__()
            except:
                pass


# from autotrain import logger
# from autotrain.backends.base import BaseBackend
# from autotrain.utils import run_training


# # Import all parameter classes
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
# from autotrain.trainers.automatic_speech_recognition.params import AutomaticSpeechRecognitionParams


# # Mapping of task_id to parameter class
# TASK_ID_TO_PARAMS_CLASS = {
#     9: LLMTrainingParams,
#     2: TextClassificationParams,
#     26: TabularParams,
#     27: GenericParams,
#     28: Seq2SeqParams,
#     18: ImageClassificationParams,
#     4: TokenClassificationParams,
#     10: TextRegressionParams,
#     29: ObjectDetectionParams,
#     30: SentenceTransformersParams,
#     24: ImageRegressionParams,
#     31: VLMTrainingParams,
#     5: ExtractiveQuestionAnsweringParams,
#     32: AutomaticSpeechRecognitionParams,
# }

# class LocalRunner(BaseBackend):
#     """
#     LocalRunner is a class that inherits from BaseBackend and is responsible for managing local training tasks.

#     Methods:
#         create():
#             Starts the local training process by retrieving parameters and task ID from environment variables.
#             Logs the start of the training process.
#             Runs the training with the specified parameters and task ID.
#             If the `wait` attribute is False, logs the training process ID (PID).
#             Returns the training process ID (PID).
#     """

#     def create(self):
#         logger.info("Starting local training...")
#         params = self.env_vars["PARAMS"]
#         task_id = int(self.env_vars["TASK_ID"])
#         training_pid = run_training(params, task_id, local=True, wait=self.wait)
#         if not self.wait:
#             logger.info(f"Training PID: {training_pid}")
#         return training_pid