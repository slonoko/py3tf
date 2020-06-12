from azureml.core import Experiment, RunConfiguration, ScriptRunConfig, Workspace, Environment
from azureml.train.dnn import TensorFlow

ws = Workspace.from_config()

envs = Environment.list(workspace=ws)

for env in envs:
    if env.find("GPU")>-1:
        print("Name",env)
        print("packages", envs[env].python.conda_dependencies.serialize_to_string())