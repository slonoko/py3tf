from azureml.core import Environment, Experiment, Workspace, Run, Model

run = Run.get_context()

run.publish_pipeline(name='sentiment-pipeline',
                     description='Model training pipeline',
                     version='1.0')
