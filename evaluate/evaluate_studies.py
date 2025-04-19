# Copyright 2025 Fraunhofer Institute for Open Communication Systems FOKUS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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