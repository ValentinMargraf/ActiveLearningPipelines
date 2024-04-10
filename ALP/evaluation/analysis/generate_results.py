import pandas as pd
from py_experimenter.experimenter import PyExperimenter

exp_scenario_file = "../experimenter/config/exp_learner_sampler.yml"
db_config_file = "../experimenter/config/db_conf.yml"


def generate_csv():

    experimenter_scenarios = PyExperimenter(experiment_configuration_file_path=exp_scenario_file,
                                            database_credential_file_path=db_config_file)
    results = experimenter_scenarios.get_table()
    results = results.rename(columns={'ID': 'experiment_id'})
    accuracies = experimenter_scenarios.get_logtable('accuracy_log')

    merged_df = pd.merge(results, accuracies, on='experiment_id', how='inner')
    merged_df = merged_df[['ID', 'setting_name', 'openml_id', 'learner_name', 'sampling_strategy_name',
                           'test_split_seed', 'train_split_seed', 'seed', 'iteration', 'test_accuracy', 'test_f1',
                           'test_precision', 'test_recall', 'test_auc', 'test_log_loss']]

    merged_df.to_csv("results/dataframes/all_results.csv")
    max_iter = merged_df['iteration'].max()

    end_df = merged_df[merged_df['iteration'] == max_iter]

    openmlids = end_df['openml_id'].unique()

    seeds = end_df['seed'].unique()
    test_split_seed = end_df['test_split_seed'].unique()
    train_split_seed = end_df['train_split_seed'].unique()

    for setting in ["small", "medium", "large"]:
        setting_df = end_df[end_df['setting_name'].str.contains(setting)]

        for metric in ["test_accuracy", "test_f1", "test_precision", "test_recall", "test_auc", "test_log_loss"]:
            df = pd.DataFrame(columns=['openmlid', 'test_split_seed', 'train_split_seed', 'seed',
                                       'learner', 'sampling_strategy', metric])
            metric_filtered_df = setting_df[['openml_id', 'test_split_seed', 'train_split_seed', 'seed',
                                             'learner_name', 'sampling_strategy_name', metric]]

            for openmlid in openmlids:
                for seed in seeds:
                    for test_seed in test_split_seed:
                        for train_seed in train_split_seed:

                            id_filtered_df = metric_filtered_df[(metric_filtered_df['openml_id'] == openmlid) &
                                                                (metric_filtered_df['seed'] == seed) &
                                                                (metric_filtered_df['test_split_seed'] == test_seed) &
                                                                (metric_filtered_df['train_split_seed'] == train_seed)]
                            print(id_filtered_df)
                            # reset index

                            id_filtered_df = id_filtered_df.reset_index()
                            if id_filtered_df.empty:
                                continue
                            else:
                                if metric != "test_log_loss":
                                    best = id_filtered_df.loc[id_filtered_df[metric].idxmax()]
                                    df.loc[len(df)] = [best['openml_id'], best['test_split_seed'],
                                                       best['train_split_seed'], best['seed'], best['learner_name'],
                                                       best['sampling_strategy_name'], best[metric]]

                                else:
                                    best = id_filtered_df.loc[id_filtered_df[metric].idxmin()]
                                    df.loc[len(df)] = [best['openml_id'], best['test_split_seed'],
                                                       best['train_split_seed'], best['seed'], best['learner_name'],
                                                       best['sampling_strategy_name'], best[metric]]
            print("save df")
            df.to_csv(f"results/dataframes/{setting}_{metric}.csv")


def main():

    generate_csv()


if __name__ == "__main__":
    main()
