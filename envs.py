from azureml.core import Experiment, RunConfiguration, ScriptRunConfig, Workspace, Environment, Model
from azureml.train.dnn import TensorFlow

ws = Workspace.from_config()

envs = Environment.list(workspace=ws)

for env in envs:
    if env.find("sentiment")>-1:
        print("Name",env)
        print("packages", envs[env].python.conda_dependencies.serialize_to_string())

for model in Model.list(ws):
    # Get model name and auto-generated version
    print(model.name, 'version:', model.version)