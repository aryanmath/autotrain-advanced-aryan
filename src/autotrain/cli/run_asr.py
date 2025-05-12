from argparse import ArgumentParser

from autotrain import logger
from autotrain.cli.utils import get_field_info
from autotrain.project import AutoTrainProject
from autotrain.trainers.asr.params import WhisperTrainingParams

from . import BaseAutoTrainCommand


def run_asr_command_factory(args):
    return RunAutoTrainASRCommand(args)


class RunAutoTrainASRCommand(BaseAutoTrainCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        arg_list = get_field_info(WhisperTrainingParams)
        arg_list = [
            {
                "arg": "--train",
                "help": "Command to train the model",
                "required": False,
                "action": "store_true",
            },
            {
                "arg": "--deploy",
                "help": "Command to deploy the model (limited availability)",
                "required": False,
                "action": "store_true",
            },
            {
                "arg": "--inference",
                "help": "Command to run inference (limited availability)",
                "required": False,
                "action": "store_true",
            },
            {
                "arg": "--backend",
                "help": "Backend",
                "required": False,
                "type": str,
                "default": "local",
            },
        ] + arg_list
        
        run_asr_parser = parser.add_parser(
            "asr", description="✨ Run AutoTrain Automatic Speech Recognition (Whisper)"
        )
        for arg in arg_list:
            if arg["arg"] == "--target_modules":
                run_asr_parser.add_argument(
                    arg["arg"],
                    help=arg["help"],
                    required=arg.get("required", False),
                    type=str,
                    nargs="+",
                    default=arg.get("default"),
                )
            elif "action" in arg:
                run_asr_parser.add_argument(
                    arg["arg"],
                    help=arg["help"],
                    required=arg.get("required", False),
                    action=arg["action"],
                    default=arg.get("default"),
                )
            else:
                run_asr_parser.add_argument(
                    arg["arg"],
                    help=arg["help"],
                    required=arg.get("required", False),
                    type=arg["type"],
                    default=arg.get("default"),
                )
        run_asr_parser.set_defaults(func=run_asr_command_factory)

    def __init__(self, args):
        self.args = args
        self.train = args.train
        self.deploy = args.deploy
        self.inference = args.inference
        self.backend = args.backend
        
        # Convert args to WhisperTrainingParams
        params_dict = vars(args)
        # Remove args that are not part of WhisperTrainingParams
        params_dict.pop("train", None)
        params_dict.pop("deploy", None)
        params_dict.pop("inference", None)
        params_dict.pop("backend", None)
        params_dict.pop("func", None)
        params_dict.pop("command", None)
        
        self.params = WhisperTrainingParams(**params_dict)

    def run(self):
        if not any([self.train, self.deploy, self.inference]):
            logger.warning(
                "No action specified. Please specify one of --train, --deploy, or --inference."
            )
            return
        
        if self.train:
            logger.info("Training ASR model...")
            project = AutoTrainProject(params=self.params, backend=self.backend)
            project.create()
        
        if self.deploy:
            logger.info("Deploying ASR model...")
            # Add deployment logic here
        
        if self.inference:
            logger.info("Running inference with ASR model...")



# from argparse import ArgumentParser

# from autotrain import logger
# from autotrain.cli.utils import get_field_info
# from autotrain.project import AutoTrainProject
# from autotrain.trainers.asr.params import ASRParams

# from . import BaseAutoTrainCommand


# def run_asr_command_factory(args):
#     return RunAutoTrainASRCommand(args)


# class RunAutoTrainASRCommand(BaseAutoTrainCommand):
#     @staticmethod
#     def register_subcommand(parser: ArgumentParser):
#         arg_list = get_field_info(WhisperTrainingParams)
#         arg_list = [
#             {
#                 "arg": "--train",
#                 "help": "Command to train the ASR model",
#                 "required": False,
#                 "action": "store_true",
#             },
#             {
#                 "arg": "--deploy",
#                 "help": "Command to deploy the ASR model (limited availability)",
#                 "required": False,
#                 "action": "store_true",
#             },
#             {
#                 "arg": "--inference",
#                 "help": "Command to run inference on the ASR model (limited availability)",
#                 "required": False,
#                 "action": "store_true",
#             },
#             {
#                 "arg": "--backend",
#                 "help": "Backend to use for training",
#                 "required": False,
#                 "type": str,
#                 "default": "local",
#             },
#         ] + arg_list

#         run_asr_parser = parser.add_parser("asr", description="✨ Run AutoTrain Automatic Speech Recognition (Whisper)")
#         for arg in arg_list:
#             names = [arg["arg"]] + arg.get("alias", [])
#             if "action" in arg:
#                 run_asr_parser.add_argument(
#                     *names,
#                     dest=arg["arg"].replace("--", "").replace("-", "_"),
#                     help=arg["help"],
#                     required=arg.get("required", False),
#                     action=arg.get("action"),
#                     default=arg.get("default"),
#                 )
#             else:
#                 run_asr_parser.add_argument(
#                     *names,
#                     dest=arg["arg"].replace("--", "").replace("-", "_"),
#                     help=arg["help"],
#                     required=arg.get("required", False),
#                     type=arg.get("type"),
#                     default=arg.get("default"),
#                     choices=arg.get("choices"),
#                 )
#         run_asr_parser.set_defaults(func=run_asr_command_factory)

#     def __init__(self, args):
#         self.args = args

#         store_true_arg_names = [
#             "train",
#             "deploy",
#             "inference",
#             "auto_find_batch_size",
#             "push_to_hub",
#         ]
#         for arg_name in store_true_arg_names:
#             if getattr(self.args, arg_name) is None:
#                 setattr(self.args, arg_name, False)

#         if self.args.train:
#             if self.args.project_name is None:
#                 raise ValueError("Project name must be specified")
#             if self.args.data_path is None:
#                 raise ValueError("Data path must be specified")
#             if self.args.model is None:
#                 raise ValueError("Model must be specified")
#             if self.args.push_to_hub:
#                 if self.args.username is None:
#                     raise ValueError("Username must be specified for push to hub")
#         else:
#             raise ValueError("Must specify --train, --deploy or --inference")

#         if self.args.backend.startswith("spaces") or self.args.backend.startswith("ep-"):
#             if not self.args.push_to_hub:
#                 raise ValueError("Push to hub must be specified for spaces backend")
#             if self.args.username is None:
#                 raise ValueError("Username must be specified for spaces backend")
#             if self.args.token is None:
#                 raise ValueError("Token must be specified for spaces backend")

#     def run(self):
#         logger.info("Running ASR Task")
#         if self.args.train:
#             params = ASRParams(**vars(self.args))
#             project = AutoTrainProject(params=params, backend=self.args.backend, process=True)
#             job_id = project.create()
#             logger.info(f"Job ID: {job_id}")