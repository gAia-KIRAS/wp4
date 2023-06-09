import mlflow
from mlflow import MlflowClient
from pyinstrument import Profiler
from src.config.config import Config

from nci.nci_class import NCI


def print_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    run = mlflow.get_run(r.info.run_id)
    print("run_id: {}".format(run.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(run.data.params))
    print("metrics: {}".format(run.data.metrics))
    print("tags: {}".format(tags))


if __name__ == "__main__":
    config = Config()
    profiler = Profiler()

    if config.profiling_active:
        profiler.start()

    with mlflow.start_run() as run:
        mlflow.log_params(config.nci_conf)
        nci = NCI(config)
        nci.run()

    print_logged_info(run)

    if config.profiling_active:
        profiler.stop()
        print(profiler.output_text(unicode=True, color=True))
        if config.profiling_browser:
            profiler.open_in_browser()
