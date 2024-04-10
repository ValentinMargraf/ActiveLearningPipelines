import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize


def plot_all_combis():

    setting_enums = ["Small", "Medium", "Large"]
    metric_enums = ["Accuracy", "F1", "Precision", "Recall", "AUC", "Log Loss"]

    for setting_enum, setting in enumerate(["small", "medium", "large"]):
        for metric_enum, metric in enumerate(["test_accuracy", "test_f1", "test_precision", "test_recall",
                                              "test_auc", "test_log_loss"]):

            df = pd.read_csv(f"results/dataframes/{setting}_{metric}.csv")

            openmlids = df['openmlid'].unique()
            seeds = df['seed'].unique()
            test_split_seed = df['test_split_seed'].unique()
            train_split_seed = df['train_split_seed'].unique()
            learners = df['learner'].unique()
            sampling_strategies = df['sampling_strategy'].unique()

            winning_table = {}

            # initialize winning table
            for learner in learners:
                for sampling_strategy in sampling_strategies:
                    winning_table[(learner, sampling_strategy)] = 0

            for openmlid in openmlids:
                for seed in seeds:
                    for test_seed in test_split_seed:
                        for train_seed in train_split_seed:
                            filtered_df = df[(df['openmlid'] == openmlid) & (df['seed'] == seed) &
                                             (df['test_split_seed'] == test_seed) &
                                             (df['train_split_seed'] == train_seed)]
                            # get learner and sampling strategy
                            # check that filtered df is non empty
                            if filtered_df.empty:
                                continue
                            else:
                                learner = filtered_df['learner'].values[0]
                                sampling_strategy = filtered_df['sampling_strategy'].values[0]
                                winning_table[(learner, sampling_strategy)] += 1

            # create numpy array from dict
            data = np.zeros((len(learners), len(sampling_strategies)))
            for i, learner in enumerate(learners):
                for j, sampling_strategy in enumerate(sampling_strategies):
                    data[i, j] = winning_table[(learner, sampling_strategy)]
            # create plot
            norm = Normalize(vmin=data.min(), vmax=data.max() * 1.5)
            colors = plt.cm.viridis(norm(data))

            # Use the 'Reds' colormap for red tones
            colors = plt.cm.Reds(norm(data))

            # Erstellen des Plots
            fig, ax = plt.subplots(figsize=(2.5 * len(sampling_strategies), 1.5 * len(learners)))

            # Zeichnen der Tafel mit den Farben basierend auf den Werten
            for i in range(len(learners)):
                for j in range(len(sampling_strategies)):
                    ax.text(j + 0.5, i + 0.5, str(data[i, j]), ha='center', va='center', color='white', fontsize=25,
                            weight='bold')
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color=colors[i, j]))

            # Einstellungen für den Plot
            x_pos = np.arange(1, len(sampling_strategies) + 1.5, 1) - .5
            y_pos = np.arange(1, len(learners) + 1.5, 1) - .5
            x_pos[-1] -= .5
            y_pos[-1] -= .5
            ax.set_xticks(x_pos)
            ax.set_yticks(y_pos)

            sampling_names = list(sampling_strategies)
            learners_names = list(learners)
            sampling_names.append(" ")
            learners_names.append(" ")
            ax.set_xticklabels(sampling_names, fontsize=16)
            ax.set_yticklabels(learners_names, fontsize=16)
            ax.set_title("Setting: " + setting_enums[setting_enum] + ", Metric: " + metric_enums[metric_enum],
                         fontsize=20)
            # plt.show()
            fig.savefig(f"results/figures/{setting}_{metric}.pdf", facecolor='white', transparent=True)


def plot_fixed_learner():

    all_results = pd.read_csv("results/dataframes/all_results.csv")

    setting_enums = ["Small", "Medium", "Large"]
    metric_enums = ["Accuracy", "F1", "Precision", "Recall", "AUC", "Log Loss"]

    for setting_enum, setting in enumerate(["small", "medium", "large"]):
        for metric_enum, metric in enumerate(["test_accuracy", "test_f1", "test_precision",
                                              "test_recall", "test_auc", "test_log_loss"]):

            df = all_results[(all_results['setting_name'] == setting)]
            # kickout all columns except the specified metric

            df = df[['openml_id', 'test_split_seed', 'train_split_seed', 'seed', 'learner_name',
                     'sampling_strategy_name', metric]]

            openmlids = df['openml_id'].unique()
            seeds = df['seed'].unique()
            test_split_seed = df['test_split_seed'].unique()
            train_split_seed = df['train_split_seed'].unique()
            learners = df['learner_name'].unique()
            sampling_strategies = df['sampling_strategy_name'].unique()

            winning_table = {}

            # initialize winning table
            for learner in learners:
                for sampling_strategy in sampling_strategies:
                    winning_table[(learner, sampling_strategy)] = 0

            # fix learner
            for learner in learners:
                for openmlid in openmlids:
                    for seed in seeds:
                        for test_seed in test_split_seed:
                            for train_seed in train_split_seed:
                                filtered_df = df[(df['openml_id'] == openmlid) & (df['seed'] == seed) &
                                                 (df['test_split_seed'] == test_seed) &
                                                 (df['train_split_seed'] == train_seed)]
                                learner_filtered_df = filtered_df[filtered_df['learner_name'] == learner]
                                learner_filtered_df = learner_filtered_df.reset_index()
                                if learner_filtered_df.empty:
                                    continue
                                else:
                                    if metric != "test_log_loss":
                                        best = learner_filtered_df.loc[learner_filtered_df[metric].idxmax()]
                                        sampling_strategy = best['sampling_strategy_name']
                                    else:
                                        best = learner_filtered_df.loc[learner_filtered_df[metric].idxmin()]
                                        sampling_strategy = best['sampling_strategy_name']

                                    winning_table[(learner, sampling_strategy)] += 1

            # create numpy array from dict
            data = np.zeros((len(learners), len(sampling_strategies)))
            for i, learner in enumerate(learners):
                for j, sampling_strategy in enumerate(sampling_strategies):
                    data[i, j] = winning_table[(learner, sampling_strategy)]

            if len(learners) > 0:

                # create plot
                norm = Normalize(vmin=data.min(), vmax=data.max() * 1.5)
                colors = plt.cm.viridis(norm(data))

                # Use the 'Reds' colormap for red tones
                colors = plt.cm.Reds(norm(data))

                # Erstellen des Plots
                fig, ax = plt.subplots(figsize=(2.5 * len(sampling_strategies), 1.5 * len(learners)))

                # Zeichnen der Tafel mit den Farben basierend auf den Werten
                for i in range(len(learners)):
                    for j in range(len(sampling_strategies)):
                        ax.text(j + 0.5, i + 0.5, str(data[i, j]), ha='center', va='center', color='white', fontsize=25,
                                weight='bold')
                        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color=colors[i, j]))

                # Einstellungen für den Plot
                x_pos = np.arange(1, len(sampling_strategies) + 1.5, 1) - .5
                y_pos = np.arange(1, len(learners) + 1.5, 1) - .5
                x_pos[-1] -= .5
                y_pos[-1] -= .5
                ax.set_xticks(x_pos)
                ax.set_yticks(y_pos)

                sampling_names = list(sampling_strategies)
                learners_names = list(learners)
                sampling_names.append(" ")
                learners_names.append(" ")
                ax.set_xticklabels(sampling_names, fontsize=16)
                ax.set_yticklabels(learners_names, fontsize=16)
                ax.set_title("Setting: " + setting_enums[setting_enum] + ", Metric: " + metric_enums[metric_enum],
                             fontsize=20)
                # plt.show()
                fig.savefig(f"results/figures/fixed_learners_{setting}_{metric}.pdf", facecolor='white',
                            transparent=True)


def plot_fixed_sampling_strategy():

    all_results = pd.read_csv("results/dataframes/all_results.csv")

    setting_enums = ["Small", "Medium", "Large"]
    metric_enums = ["Accuracy", "F1", "Precision", "Recall", "AUC", "Log Loss"]

    for setting_enum, setting in enumerate(["small", "medium", "large"]):
        for metric_enum, metric in enumerate(["test_accuracy", "test_f1", "test_precision",
                                              "test_recall", "test_auc", "test_log_loss"]):

            df = all_results[(all_results['setting_name'] == setting)]
            # kickout all columns except the specified metric

            df = df[['openml_id', 'test_split_seed', 'train_split_seed', 'seed', 'learner_name',
                     'sampling_strategy_name', metric]]

            openmlids = df['openml_id'].unique()
            seeds = df['seed'].unique()
            test_split_seed = df['test_split_seed'].unique()
            train_split_seed = df['train_split_seed'].unique()
            learners = df['learner_name'].unique()
            sampling_strategies = df['sampling_strategy_name'].unique()

            winning_table = {}

            # initialize winning table
            for learner in learners:
                for sampling_strategy in sampling_strategies:
                    winning_table[(learner, sampling_strategy)] = 0

            # fix learner
            for sampling_strategy in sampling_strategies:
                for openmlid in openmlids:
                    for seed in seeds:
                        for test_seed in test_split_seed:
                            for train_seed in train_split_seed:
                                filtered_df = df[(df['openml_id'] == openmlid) & (df['seed'] == seed) &
                                                 (df['test_split_seed'] == test_seed) &
                                                 (df['train_split_seed'] == train_seed)]
                                sampling_filtered_df = filtered_df[filtered_df['sampling_strategy_name']
                                                                   == sampling_strategy]
                                sampling_filtered_df = sampling_filtered_df.reset_index()
                                if sampling_filtered_df.empty:
                                    continue
                                else:
                                    if metric != "test_log_loss":
                                        best = sampling_filtered_df.loc[sampling_filtered_df[metric].idxmax()]
                                        learner = best['learner_name']
                                    else:
                                        best = sampling_filtered_df.loc[sampling_filtered_df[metric].idxmin()]
                                        learner = best['learner_name']

                                    winning_table[(learner, sampling_strategy)] += 1

            # create numpy array from dict
            data = np.zeros((len(learners), len(sampling_strategies)))
            for i, learner in enumerate(learners):
                for j, sampling_strategy in enumerate(sampling_strategies):
                    data[i, j] = winning_table[(learner, sampling_strategy)]

            # create plot
            norm = Normalize(vmin=data.min(), vmax=data.max() * 1.5)
            colors = plt.cm.viridis(norm(data))

            # Use the 'Reds' colormap for red tones
            colors = plt.cm.Reds(norm(data))

            # Erstellen des Plots
            fig, ax = plt.subplots(figsize=(2.5 * len(sampling_strategies), 1.5 * len(learners)))

            # Zeichnen der Tafel mit den Farben basierend auf den Werten
            for i in range(len(learners)):
                for j in range(len(sampling_strategies)):
                    ax.text(j + 0.5, i + 0.5, str(data[i, j]), ha='center', va='center', color='white', fontsize=25,
                            weight='bold')
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color=colors[i, j]))

            # Einstellungen für den Plot
            x_pos = np.arange(1, len(sampling_strategies) + 1.5, 1) - .5
            y_pos = np.arange(1, len(learners) + 1.5, 1) - .5
            x_pos[-1] -= .5
            y_pos[-1] -= .5
            ax.set_xticks(x_pos)
            ax.set_yticks(y_pos)

            sampling_names = list(sampling_strategies)
            learners_names = list(learners)
            sampling_names.append(" ")
            learners_names.append(" ")
            ax.set_xticklabels(sampling_names, fontsize=16)
            ax.set_yticklabels(learners_names, fontsize=16)
            ax.set_title("Setting: " + setting_enums[setting_enum] + ", Metric: " + metric_enums[metric_enum],
                         fontsize=20)
            # plt.show()
            fig.savefig(f"results/figures/fixed_samplers_{setting}_{metric}.pdf", facecolor='white', transparent=True)


def main():

    plot_all_combis()
    plot_fixed_learner()
    plot_fixed_sampling_strategy()


if __name__ == '__main__':
    main()
