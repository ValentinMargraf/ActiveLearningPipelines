import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize


def plot_distributions(SAVEFIG=False, fix_learner=False, fix_sampling_strategy=False):

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

            if fix_learner:
                fixed_components = learners
            elif fix_sampling_strategy:
                fixed_components = sampling_strategies
            else:
                fixed_components = [None]
            for fixed in fixed_components:
                for openmlid in openmlids:
                    for seed in seeds:
                        for test_seed in test_split_seed:
                            for train_seed in train_split_seed:
                                filtered_df = df[(df['openml_id'] == openmlid) & (df['seed'] == seed) &
                                                 (df['test_split_seed'] == test_seed) &
                                                 (df['train_split_seed'] == train_seed)]
                                if fix_learner:
                                    filtered_df = filtered_df[filtered_df['learner_name'] == fixed]
                                if fix_sampling_strategy:
                                    filtered_df = filtered_df[filtered_df['sampling_strategy_name'] == fixed]

                                filtered_df = filtered_df.reset_index()
                                if filtered_df.empty:
                                    continue
                                else:
                                    if metric != "test_log_loss":
                                        # Find the maximum value of the metric
                                        max_value = filtered_df[metric].max()
                                        # Select rows where the metric equals the maximum value
                                        best_combis = filtered_df.loc[filtered_df[metric] == max_value]
                                        for _, best in best_combis.iterrows():
                                            learner = best['learner_name']
                                            sampling_strategy = best['sampling_strategy_name']
                                            winning_table[(learner, sampling_strategy)] += 1

                                    else:
                                        # Find the maximum value of the metric
                                        min_value = filtered_df[metric].min()
                                        # Select rows where the metric equals the maximum value
                                        best_combis = filtered_df.loc[filtered_df[metric] == min_value]
                                        for _, best in best_combis.iterrows():
                                            learner = best['learner_name']
                                            sampling_strategy = best['sampling_strategy_name']
                                            winning_table[(learner, sampling_strategy)] += 1

            # create numpy array from dict
            data = np.zeros((len(learners), len(sampling_strategies)))
            for i, learner in enumerate(learners):
                for j, sampling_strategy in enumerate(sampling_strategies):
                    data[i, j] = winning_table[(learner, sampling_strategy)]

            if len(data) > 0:

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
                        ax.text(j + 0.5, i + 0.5, str(data[i, j]), ha='center', va='center', color='white',
                                fontsize=25,
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
                print("s names", sampling_strategies)

                sampling_names.append(" ")
                learners_names.append(" ")
                ax.set_xticklabels(sampling_names, fontsize=16)
                ax.set_yticklabels(learners_names, fontsize=16)
                ax.set_title("Setting: " + setting_enums[setting_enum] + ", Metric: " + metric_enums[metric_enum],
                             fontsize=20)
                if SAVEFIG:
                    if fix_learner:
                        fig.savefig(f"results/figures/fixed_learners_{setting}_{metric}.pdf", facecolor='white',
                                    transparent=True)
                    elif fix_sampling_strategy:
                        fig.savefig(f"results/figures/fixed_sampling_strategies_{setting}_{metric}.pdf",
                                    facecolor='white', transparent=True)
                    else:
                        fig.savefig(f"results/figures/{setting}_{metric}.pdf", facecolor='white', transparent=True)


def main():

    SAVEFIG = True
    plot_distributions(SAVEFIG=SAVEFIG)
    plot_distributions(SAVEFIG=SAVEFIG, fix_learner=True)
    plot_distributions(SAVEFIG=SAVEFIG, fix_sampling_strategy=True)


if __name__ == '__main__':
    main()
