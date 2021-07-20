# imports
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from memoized_property import memoized_property

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from TaxiFareModel.encoders import DistanceTransformer
from TaxiFareModel.encoders import TimeFeaturesEncoder
from TaxiFareModel.encoders import DistanceToCenter
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data

import mlflow
from mlflow.tracking import MlflowClient
MLFLOW_URI = "https://mlflow.lewagon.co/"
EXPERIMENT_NAME = "[CN] [Shanghai] [the-pavel] LinReg v2"


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.experiment_name = EXPERIMENT_NAME
        self.pipeline = None
        self.X = X
        self.y = y
        print('trainer initialized...')

    def set_pipeline(self):
        print('started pipeline creation...')
        if not self.pipeline:
            """defines the pipeline as a class attribute"""
            dist_pipe = Pipeline([
                ('dist_trans', DistanceTransformer()),
                ('stdscaler', StandardScaler())
            ])

            dist_to_center_pipe = Pipeline([
                ('dist_trans', DistanceToCenter()),
                ('stdscaler', StandardScaler())
            ])

            time_pipe = Pipeline([
                ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
                ('ohe', OneHotEncoder(handle_unknown='ignore'))
            ])

            preproc_pipe = ColumnTransformer([
                ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
                ('distance_to_center', dist_to_center_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
                ('time', time_pipe, ['pickup_datetime'])
            ], remainder="drop")

            pipe = Pipeline([
                ('preproc', preproc_pipe),
                ('linear_model', LinearRegression())
            ])
            self.pipeline = pipe
            return pipe
        return self.pipeline

    def holdout(self):
        print('started holdout...')
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2)
        return self.x_train, self.x_test, self.y_train, self.y_test

    def train(self):
        """set and train the pipeline"""
        print('started fitting process...')
        self.set_pipeline()
        self.holdout()
        self.pipeline.fit(self.x_train, self.y_train)
        print('finished fitting process...')
        return self.pipeline


    def evaluate(self, x=None, y=None, **kwargs):
        """evaluates the pipeline on df_test and return the RMSE"""
        print('evaluating pipe...')
        y_pred = self.pipeline.predict(self.x_test)
        rmse = compute_rmse(y_pred, self.y_test)
        print("RMSE -> ", rmse)
        self.mlflow_log_metric('rmse', rmse)
        for k, v in kwargs.items():
            self.mlflow_log_param(k, v)
        return rmse

    ### MLFLOW METHODS

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":
    # get clean data
    # set X and y
    x, y = get_data()
    # create a new trainer
    trainer = Trainer(x, y)
    # hold out (done in train)
    # train
    pipe = trainer.train()
    # evaluate
    trainer.evaluate(distance_to_center=True)
