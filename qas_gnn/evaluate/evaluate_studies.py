import optuna
import os

from config import PathConfig

def get_params(study_name, storage):
    loaded_study = optuna.load_study(study_name=study_name, storage=storage)

    best_params = loaded_study.best_trial.params
    best_trial = loaded_study.best_trial.value
    return best_params, best_trial


if __name__ == '__main__':

    path_config = PathConfig()

    study_name = 'gcn_spearman_gs2'
    storage = f"sqlite:///{os.path.join(path_config.paths[f'optuna_studies'], study_name)}.db"
    best_trial, best_params = get_params(study_name, storage)
    print(f'best trial: {best_trial}')
    print(f'best params: {best_params}')