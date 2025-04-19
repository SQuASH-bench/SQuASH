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


import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from benchmark.search_spaces.ls_a.ls_a_search_space_config import ls_a_conf


def visualize_pairplot_8d(X_train, y_train, save_plot_path: str = './pairplot.png'):
    df = pd.DataFrame(X_train, columns=[f'Feature {i + 1}' for i in range(X_train.shape[1])])
    df['Class'] = y_train
    df['Class'] = df['Class'].astype(str)

    sns.pairplot(df, hue='Class', palette={'-1': '#814255', '1': '#59858E'}, plot_kws={'alpha': 0.6})
    plt.suptitle('Pairwise Feature Plot (8D)', y=1.02)
    if save_plot_path is not None:
        plt.savefig(save_plot_path)
        print(f"[INFO] Pairplot saved as {save_plot_path}")
    else:
        plt.show()


def visualize(X_train, y_train, save_plot_path: str = './data.png'):
    """
          Visualizes linearly separable data with two features using a scatter plot.
          The data points are color-coded based on their class labels (-1 and 1).
          Additionally, a hypothetical decision boundary (hyperplane) is plotted.

          Parameters:
          - X (list): Input features of the dataset. Should be a 2D array or matrix
            where each row represents a sample and each column represents a feature.
          - y (list): Target labels corresponding to each sample in X. Should be a 1D
            array where each element is either -1 or 1, indicating the class label.
          - save_plot (bool): Optional. If specified, the plot will be saved to the
            file path specified by save_plot. If None, the plot will be displayed interactively.

          Returns:
          - None: This function does not return any value. It generates and displays a plot
            using matplotlib.

          Note:
          - The plot includes two scatter plots for positive (blue 'x') and negative (red 'o')
            class labels based on the provided data.
          - A straight line representing the decision boundary (hyperplane) is drawn assuming
            a hypothetical weight vector [1, 1] and a bias term of 0. The decision boundary
            equation used is y = -(w_true[0] * x) / w_true[1].

          """
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1],
                color='#814255', marker='o', label='Class -1')
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1],
                color='#59858E', marker='x', label='Class 1')

    # Plot the decision boundary (hyperplane)
    w_true = np.ones(2)
    x_values = np.linspace(-1, 1, 100)
    y_values = -(w_true[0] * x_values) / w_true[1]
    plt.plot(x_values, y_values, label='Decision Boundary', color='black')

    plt.xlabel('')
    plt.ylabel('')
    plt.legend()
    plt.title('Linearly Separable Data Visualization')
    plt.grid(True)
    if save_plot_path is not None:
        plt.savefig(save_plot_path)
        print(f"[INFO] Plot saved as {save_plot_path}")
    else:
        plt.show()


def generate_data(data_seed: int = ls_a_conf["seed_dataset_generation"], current_qas_seed=None):
    # Copyright 2024 Xanadu Quantum Technologies Inc.

    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at

    #     http://www.apache.org/licenses/LICENSE-2.0

    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.

    """Data generation procedure for 'linearly separable'.

    Args:
        :param data_seed(int) seed for the global data generation
        :param seeds_for_data_split (list) seeds for data splittings
        :param current_seed (int) current seed for the QAS method
        :param n_samples: (int) number of samples to generate
        :param n_features: (int) dimension of the data samples
        :param margin:(float)width between hyperplane and closest samples
        :param seed: (int) random
    """
    n_features = ls_a_conf['n_features']
    n_samples = ls_a_conf['n_samples']
    margin = ls_a_conf['margin']
    if data_seed is not None:
        np.random.seed(data_seed)
        random.seed(data_seed)

    w_true = np.ones(n_features)

    # hack: sample more data than we need randomly from a hypercube
    X = 2 * np.random.rand(2 * n_samples * 10, n_features) - 1

    # only retain data outside a margin
    X = [x for x in X if np.abs(np.dot(x, w_true)) > margin]
    train_data_size = int(n_samples * 0.8)
    X = X[:n_samples]
    X_train = np.array(X[:train_data_size])
    X_test = np.array(X[train_data_size:])
    print(f"[INFO] ls_a: Size of the train dataset:{len(X_train)}")
    print(f"[INFO] ls_a: Size of the test dataset:{len(X_test)}")

    y_train = [np.dot(x, w_true) for x in X_train]
    y_train = np.array([-1 if y_ > 0 else 1 for y_ in y_train])

    y_test = [np.dot(x, w_true) for x in X_test]
    y_test = np.array([-1 if y_ > 0 else 1 for y_ in y_test])

    #visualize(X_train, y_train, save_plot_path=None)
    #visualize_pairplot_8d(X_train, y_train, save_plot_path=None)
    if current_qas_seed:
        np.random.seed(current_qas_seed)
        random.seed(current_qas_seed)
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = generate_data()
