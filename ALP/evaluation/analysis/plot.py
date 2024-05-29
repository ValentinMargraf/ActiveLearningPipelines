import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_distributions(SAVEFIG=False, fix_learner=False, fix_sampling_strategy=False):

    all_results = pd.read_csv("results/dataframes/all_results.csv")

    setting_enums = ["Small", "Medium", "Large"]
    metric_enums = ["Accuracy", "F1", "Precision", "Recall", "AUC", "Log Loss"]

    for setting_enum, setting in enumerate(["small", "medium", "large"]):
        for metric_enum, metric in enumerate(
            ["test_accuracy", "test_f1", "test_precision", "test_recall", "test_auc", "test_log_loss"]
        ):

            df = all_results[(all_results["setting_name"] == setting)]
            # kickout all columns except the specified metric

            df = df[
                [
                    "openml_id",
                    "test_split_seed",
                    "train_split_seed",
                    "seed",
                    "learner_name",
                    "sampling_strategy_name",
                    metric,
                ]
            ]

            openmlids = df["openml_id"].unique()
            seeds = df["seed"].unique()
            test_split_seed = df["test_split_seed"].unique()
            train_split_seed = df["train_split_seed"].unique()
            learners = df["learner_name"].unique()
            sampling_strategies = df["sampling_strategy_name"].unique()

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
                                filtered_df = df[
                                    (df["openml_id"] == openmlid)
                                    & (df["seed"] == seed)
                                    & (df["test_split_seed"] == test_seed)
                                    & (df["train_split_seed"] == train_seed)
                                ]
                                if fix_learner:
                                    filtered_df = filtered_df[filtered_df["learner_name"] == fixed]
                                if fix_sampling_strategy:
                                    filtered_df = filtered_df[filtered_df["sampling_strategy_name"] == fixed]

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
                                            learner = best["learner_name"]
                                            sampling_strategy = best["sampling_strategy_name"]
                                            winning_table[(learner, sampling_strategy)] += 1

                                    else:
                                        # Find the maximum value of the metric
                                        min_value = filtered_df[metric].min()
                                        # Select rows where the metric equals the maximum value
                                        best_combis = filtered_df.loc[filtered_df[metric] == min_value]
                                        for _, best in best_combis.iterrows():
                                            learner = best["learner_name"]
                                            sampling_strategy = best["sampling_strategy_name"]
                                            winning_table[(learner, sampling_strategy)] += 1

            # create numpy array from dict
            data = np.zeros((len(learners), len(sampling_strategies)))
            for i, learner in enumerate(learners):
                for j, sampling_strategy in enumerate(sampling_strategies):
                    data[i, j] = winning_table[(learner, sampling_strategy)]


def main():

    SAVEFIG = True
    plot_distributions(SAVEFIG=SAVEFIG)
    plot_distributions(SAVEFIG=SAVEFIG, fix_learner=True)
    plot_distributions(SAVEFIG=SAVEFIG, fix_sampling_strategy=True)


if __name__ == "__main__":
    main()


class PerformanceProfile:
    def __init__(self, metric, df, path_to_save=None):
        self.metric = metric
        self.df = df
        self.df = pd.read_csv("results/dataframes/small_test_accuracy.csv")
        self.path_to_save = path_to_save

    def show(self):
        best_performance_dict = {}
        openmlids = self.df["openml_id"].unique()
        seeds = self.df["seed"].unique()
        test_split_seed = self.df["test_split_seed"].unique()
        train_split_seed = self.df["train_split_seed"].unique()
        # each combination of learner/sampling strategy is an approach
        plots = {}
        learners = self.df["learner_name"].unique()
        sampling_strategies = self.df["sampling_strategy_name"].unique()
        approaches = [(learner, sampling_strategy) for learner in learners for sampling_strategy in sampling_strategies]
        taus = np.linspace(0, 1000, 100)

        for approach in approaches:
            plots[approach] = []
            for tau in taus:
                value = 0
                for oid in openmlids:
                    for seed in seeds:
                        for test_seed in test_split_seed:
                            for train_seed in train_split_seed:
                                filtered_df = self.df[
                                    (self.df["openml_id"] == oid)
                                    & (self.df["seed"] == seed)
                                    & (self.df["test_split_seed"] == test_seed)
                                    & (self.df["train_split_seed"] == train_seed)
                                ]
                                learner = approach[0]
                                sampling_strategy = approach[1]
                                approach_perf = filtered_df[
                                    (filtered_df["learner_name"] == learner)
                                    & (filtered_df["sampling_strategy_name"] == sampling_strategy)
                                ][self.metric].iloc[0]
                                if self.metric != "test_logloss":
                                    approach_perf = 1 - approach_perf
                                    best_perf = 1 - filtered_df[self.metric].max()
                                    # approach_perf = approach_perf
                                    # best_perf = filtered_df[self.metric].max()
                                else:
                                    best_perf = filtered_df[self.metric].min()

                                if approach_perf <= tau * best_perf:
                                    value += 1
                value /= len(openmlids) * len(seeds) * len(test_split_seed) * len(train_split_seed)
                plots[approach].append(value)

        plt.figure(figsize=(10, 6))
        x = taus
        for i in range(len(approaches)):
            plt.plot(x, plots[approaches[i]], label=approaches[i])

        plt.xlabel("tau")
        # plt.ylabel('fraction of datasets')
        plt.title("Performance profile")
        plt.legend()

        plt.savefig("results/figures/performance_profile_" + self.metric + ".pdf", bbox_inches="tight")

        # TODO clean up here

        if self.filtered_learners is not None:
            merged_df = merged_df[merged_df["learner_name"].str.contains(self.filtered_learners)]
        if self.filtered_sampling_strategies is not None:
            merged_df = merged_df[merged_df["sampling_strategy_name"].str.contains(self.filtered_sampling_strategies)]

        print("end_df", end_df.columns)

        pre_openmlids = end_df["openml_id"].unique()

        print("we start with ", len(pre_openmlids))

        seeds = end_df["seed"].unique()
        test_split_seed = end_df["test_split_seed"].unique()
        train_split_seed = end_df["train_split_seed"].unique()

        learners = end_df["learner_name"].unique()
        sampling_strategies = end_df["sampling_strategy_name"].unique()

        setting_df = end_df[end_df["setting_name"].str.contains(self.setting)]

        if not self.fix_learner and not self.fix_sampling_strategy:
            # only those openmlids where i have at least one entry per combination of learner and sampling_strategy
            num_learners = len(learners)
            num_sampling_strategies = len(sampling_strategies)
            openmlids = []
            for oid in pre_openmlids:
                filtered_df = setting_df[setting_df["openml_id"] == oid]
                num_combis = len(filtered_df.groupby(["learner_name", "sampling_strategy_name"]).size())
                print(num_combis)
                if num_combis >= num_learners * num_sampling_strategies:
                    openmlids.append(oid)
            print("we stay with ", len(openmlids))
        else:
            openmlids = pre_openmlids

        metric_filtered_df = setting_df[
            [
                "openml_id",
                "test_split_seed",
                "train_split_seed",
                "seed",
                "learner_name",
                "sampling_strategy_name",
                self.metric,
            ]
        ]

        self.df = metric_filtered_df

        if self.csv_path is not None:
            self.csv_path = self.csv_path + self.setting + "_" + self.metric + ".csv"
            self.save_csv(metric_filtered_df)

        winning_table = {}

        # initialize winning table
        # sort the learner and sampling strategies

        learners = ["knn_3", "svm_rbf", "mlp", "rf_entropy", "catboost", "xgb", "tabnet", "tabpfn"]
        sampling_strategies = [
            "random",
            "margin",
            "least_confident",
            "entropy",
            "qbc_variance_ratio",
            "max_entropy",
            "power_margin",
            "bald",
            "power_bald",
            "core_set",
            "kmeans",
            "cluster_margin",
            "typ_cluster",
            "weighted_cluster",
            "falcun",
        ]

        for learner in learners[::-1]:
            for sampling_strategy in sampling_strategies:
                winning_table[(learner, sampling_strategy)] = 0

        if self.fix_learner:
            fixed_components = learners
        elif self.fix_sampling_strategy:
            fixed_components = sampling_strategies
        else:
            fixed_components = [None]
        for fixed in fixed_components:
            for openmlid in openmlids:

                # TODO eigentlich hier die panda dataframe mit aubc rein

                filtered_df = metric_filtered_df[(metric_filtered_df["openml_id"] == openmlid)]
                filtered_df = (
                    filtered_df.groupby(["learner_name", "sampling_strategy_name"])[self.metric].mean().reset_index()
                )

                # average over seeds!!!!
                # for seed in seeds:
                #    for test_seed in test_split_seed:
                #        for train_seed in train_split_seed:
                # filtered_df = metric_filtered_df[(metric_filtered_df['openml_id'] == openmlid) & (metric_filtered_df['seed'] == seed) &
                #                 (metric_filtered_df['test_split_seed'] == test_seed) &
                #                 (metric_filtered_df['train_split_seed'] == train_seed)]
                if self.fix_learner:
                    filtered_df = filtered_df[filtered_df["learner_name"] == fixed]
                if self.fix_sampling_strategy:
                    filtered_df = filtered_df[filtered_df["sampling_strategy_name"] == fixed]

                filtered_df = filtered_df.reset_index()
                if filtered_df.empty:
                    continue
                else:
                    if self.metric != "test_log_loss":
                        # Find the maximum value of the metric
                        max_value = filtered_df[self.metric].max()
                        # Select rows where the metric equals the maximum value
                        best_combis = filtered_df.loc[filtered_df[self.metric] == max_value]
                        # print("currently fixed" ,  fixed)
                        # print("num equ val", len(best_combis))

                        # TODO test auf statistical significance

                        # print("\n")
                        for _, best in best_combis.iterrows():
                            learner = best["learner_name"]
                            sampling_strategy = best["sampling_strategy_name"]

                            winning_table[(learner, sampling_strategy)] += 1

                    else:
                        # Find the maximum value of the metric
                        min_value = filtered_df[self.metric].min()
                        # Select rows where the metric equals the maximum value
                        best_combis = filtered_df.loc[filtered_df[self.metric] == min_value]

                        # TODO test auf statistical significance

                        for _, best in best_combis.iterrows():

                            learner = best["learner_name"]
                            sampling_strategy = best["sampling_strategy_name"]
                            winning_table[(learner, sampling_strategy)] += 1

        # create numpy array from dict
        res = np.zeros((len(learners), len(sampling_strategies)))
        for i, learner in enumerate(learners):
            for j, sampling_strategy in enumerate(sampling_strategies):
                res[i, j] = winning_table[(learner, sampling_strategy)]

        data = {}
        data["data"] = res
        data["learners"] = learners
        data["sampling_strategies"] = sampling_strategies
        data["setting_name"] = self.setting
        data["metric_name"] = self.metric
        return data
