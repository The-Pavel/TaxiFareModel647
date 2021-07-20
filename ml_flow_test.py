import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "[CN] [Shanghai] [Pavel] LinReg v1"

# Indicate mlflow to log to remote server
mlflow.set_tracking_uri("https://mlflow.lewagon.co/")

client = MlflowClient()

try:
    experiment_id = client.create_experiment(EXPERIMENT_NAME)
except BaseException:
    experiment_id = client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

yourname = 'Pavel'

if yourname is None:
    print("please define your name, il will be used as a parameter to log")


run = client.create_run(experiment_id)
client.log_metric(run.info.run_id, "rmse", 4.5)
client.log_param(run.info.run_id, "model", 'LinReg')
client.log_param(run.info.run_id, "student_name", yourname)
