import numpy as np
from py_experimenter.experimenter import PyExperimenter, ResultProcessor

from ALP.benchmark.BenchmarkConnector import MySQLBenchmarkConnector
from ALP.evaluation.experimenter.LogTableObserver import LogTableObserver
from ALP.pipeline.ActiveLearningPipeline import ActiveLearningPipeline
from ALP.pipeline.Oracle import Oracle

exp_setting_file = "config/exp_setting_conf.yml"
exp_scenario_file = "config/exp_scenario_conf.yml"
exp_learner_sampler_file = "config/exp_learner_sampler_conf.yml"
db_config_file = "config/db_conf.yml"

setup_table = False

small_openml_ids = [3, 6, 8, 10, 11, 12, 14, 15, 16, 18, 20, 21, 22, 23, 26, 28, 29, 30, 31, 32, 36, 37, 39, 40, 41, 43,
                    44, 45, 46, 48, 49, 50, 53, 54, 59, 60, 61, 62, 151, 155, 161, 162, 164, 180, 181, 182, 183, 184,
                    187, 197, 209, 219, 223, 279, 285, 287, 292, 294, 300, 307, 312, 313, 329, 333, 334, 335, 336, 337,
                    338, 375, 377, 383, 384, 385, 386, 387, 388, 389, 391, 392, 394, 395, 397, 398, 400, 401, 444, 446,
                    448, 458, 461, 463, 464, 469, 475, 478, 679, 685, 694, 714, 715, 716, 717, 718, 719, 720, 721, 722,
                    723, 724, 725, 726, 727, 728, 730, 732, 733, 734, 735, 736, 737, 740, 741, 742, 743, 744, 745, 746,
                    747, 748, 749, 750, 751, 752, 753, 754, 756, 761, 762, 763, 766, 768, 769, 770, 771, 772, 773, 774,
                    775, 776, 778, 779, 782, 783, 784, 788, 789, 792, 793, 794, 795, 796, 797, 799, 801, 803, 805, 806,
                    807, 808, 811, 812, 813, 814, 816, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830,
                    832, 833, 834, 837, 838, 841, 843, 845, 846, 847, 849, 850, 851, 853, 855, 860, 863, 865, 866, 867,
                    868, 869, 870, 871, 872, 873, 875, 876, 877, 878, 879, 880, 881, 884, 885, 886, 888, 889, 890, 895,
                    896, 900, 901, 902, 903, 904, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 920,
                    921, 922, 923, 924, 925, 926, 931, 932, 933, 934, 935, 936, 937, 941, 943, 948, 949, 952, 953, 954,
                    955, 956, 958, 959, 962, 965, 969, 970, 971, 973, 974, 976, 978, 979, 980, 983, 987, 991, 994, 995,
                    996, 997, 1004, 1005, 1006, 1011, 1012, 1013, 1014, 1016, 1019, 1020, 1021, 1022, 1025, 1026, 1036,
                    1038, 1040, 1041, 1043, 1044, 1045, 1046, 1048, 1049, 1050, 1054, 1059, 1061, 1063, 1064, 1065,
                    1066, 1067, 1071, 1073, 1075, 1077, 1078, 1080, 1084, 1100, 1106, 1115, 1116, 1120, 1121, 1122,
                    1123, 1124, 1125, 1126, 1127, 1129, 1131, 1132, 1133, 1135, 1136, 1137, 1140, 1141, 1143, 1144,
                    1145, 1147, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1162,
                    1163, 1164, 1165, 1167, 1169, 1217, 1236, 1237, 1238, 1412, 1413, 1441, 1442, 1443, 1444, 1446,
                    1447, 1448, 1449, 1450, 1451, 1453, 1454, 1455, 1457, 1459, 1460, 1464, 1467, 1471, 1472, 1473,
                    1475, 1481, 1482, 1483, 1486, 1487, 1488, 1489, 1496, 1498, 1500, 1501, 1503, 1505, 1507, 1508,
                    1509, 1510, 1512, 1513, 1519, 1520, 1527, 1528, 1529, 1531, 1533, 1534, 1535, 1537, 1538, 1539,
                    1540, 1542, 1543, 1545, 1546, 1556, 1557, 1565, 1568, 1600, 2074, 2079, 3021, 3022, 3560, 3902,
                    3904, 3913, 3917, 4134, 4135, 4153, 4340, 4534, 4538, 7592, 9910, 9946, 9952, 9957, 9960, 9964,
                    9971, 9976, 9977, 9978, 9981, 9985, 10093, 10101, 14952, 14954, 14965, 14969, 14970, 40474, 40475,
                    40476, 40477, 40478, 125920, 125922, 146195, 146800, 146817, 146819, 146820, 146821, 146822, 146824,
                    146825, 167119, 167120, 167121, 167124, 167125, 167140, 167141]
medium_openml_ids = [3, 6, 11, 12, 14, 15, 16, 18, 20, 21, 22, 23, 26, 28, 29, 30, 31, 32, 36, 37, 44, 45, 46, 50, 54,
                     60, 151, 155, 161, 162, 180, 181, 182, 183, 184, 197, 209, 219, 223, 279, 287, 292, 294, 300, 307,
                     312, 313, 333, 334, 335, 375, 377, 383, 385, 386, 389, 391, 392, 394, 395, 398, 400, 401, 458, 469,
                     478, 679, 715, 717, 718, 720, 722, 723, 725, 727, 728, 734, 735, 737, 740, 741, 742, 743, 749, 750,
                     751, 752, 761, 766, 770, 772, 774, 779, 792, 795, 797, 799, 803, 805, 806, 807, 813, 816, 819, 821,
                     822, 823, 824, 825, 826, 827, 833, 837, 838, 841, 843, 845, 846, 847, 849, 853, 855, 866, 869, 870,
                     871, 872, 879, 881, 884, 886, 888, 896, 901, 903, 904, 910, 912, 913, 914, 917, 920, 923, 926, 931,
                     934, 936, 937, 943, 948, 949, 953, 954, 958, 959, 962, 970, 971, 976, 978, 979, 980, 983, 987, 991,
                     994, 995, 997, 1004, 1014, 1016, 1019, 1020, 1021, 1022, 1036, 1038, 1040, 1041, 1043, 1044, 1046,
                     1049, 1050, 1063, 1067, 1116, 1120, 1137, 1145, 1158, 1165, 1169, 1217, 1236, 1237, 1238, 1443,
                     1444, 1451, 1453, 1454, 1457, 1459, 1460, 1464, 1467, 1471, 1472, 1475, 1481, 1483, 1486, 1487,
                     1489, 1496, 1501, 1503, 1505, 1507, 1509, 1510, 1527, 1528, 1529, 1531, 1533, 1534, 1535, 1537,
                     1538, 1539, 1540, 1542, 1543, 1545, 1546, 1557, 1568, 2074, 2079, 3021, 3022, 3560, 3904, 3917,
                     4134, 4135, 4534, 4538, 7592, 9910, 9946, 9952, 9957, 9960, 9964, 9971, 9976, 9977, 9978, 9981,
                     9985, 10093, 10101, 14952, 14954, 14965, 14969, 14970, 40474, 40475, 40476, 40477, 40478, 125920,
                     125922, 146195, 146800, 146817, 146819, 146820, 146821, 146822, 146824, 146825, 167119, 167120,
                     167121, 167124, 167125, 167140, 167141]
large_openml_ids = [3, 6, 12, 14, 16, 18, 20, 21, 22, 26, 28, 30, 32, 36, 44, 45, 46, 60, 151, 155, 161, 162, 180, 182,
                    183, 184, 197, 209, 219, 279, 287, 294, 300, 312, 375, 389, 391, 395, 398, 720, 722, 725, 727, 728,
                    734, 735, 737, 752, 761, 772, 803, 807, 816, 819, 821, 822, 823, 833, 843, 846, 847, 871, 881, 901,
                    914, 923, 948, 953, 958, 959, 962, 971, 976, 978, 979, 980, 991, 995, 1019, 1020, 1021, 1022, 1036,
                    1038, 1040, 1041, 1043, 1044, 1046, 1050, 1067, 1116, 1120, 1169, 1217, 1236, 1237, 1238, 1457,
                    1459, 1460, 1471, 1475, 1481, 1483, 1486, 1487, 1489, 1496, 1501, 1503, 1505, 1507, 1509, 1527,
                    1528, 1529, 1531, 1533, 1534, 1535, 1537, 1538, 1539, 1540, 1557, 1568, 2074, 3021, 4134, 4135,
                    4534, 4538, 7592, 9910, 9952, 9960, 9964, 9976, 9977, 9978, 9985, 14952, 14965, 14969, 14970, 40474,
                    40475, 40476, 40477, 40478, 125922, 146195, 146817, 146820, 146821, 146822, 146824, 146825, 167119,
                    167120, 167121, 167124, 167125, 167140, 167141]


def run_experiment(parameters: dict, result_processor: ResultProcessor, custom_config: dict):
    connector: MySQLBenchmarkConnector = custom_config["dbbc"]

    OPENML_ID = int(parameters["openml_id"])
    SETTING_NAME = int(parameters["setting_name"])
    TEST_SPLIT_SEED = int(parameters["test_split_seed"])
    TRAIN_SPLIT_SEED = int(parameters["train_split_seed"])
    SEED = int(parameters["seed"])

    setting = connector.load_setting_by_name(SETTING_NAME)
    scenario = connector.load_or_create_scenario(openml_id=OPENML_ID, test_split_seed=TEST_SPLIT_SEED,
                                                 train_split_seed=TRAIN_SPLIT_SEED, seed=SEED,
                                                 setting_id=setting.get_setting_id())

    X_l, y_l, X_u, y_u, X_test, y_test = scenario.get_data_split()

    SAMPLING_STRATEGY = connector.load_sampling_strategy_by_name(parameters["sampling_strategy_name"])
    LEARNER = connector.load_learner_by_name(parameters["learner_name"])

    OBSERVER = [LogTableObserver()]

    ALP = ActiveLearningPipeline(learner=SAMPLING_STRATEGY, sampling_strategy=LEARNER, observer_list=OBSERVER,
                                 # init_budget=INIT_BUDGET,
                                 num_iterations=setting.get_number_of_iterations(),
                                 num_samples_per_iteration=setting.get_number_of_samples())

    oracle = Oracle(X_u, y_u)
    ALP.active_fit(X_l, y_l, X_u, oracle)


def main():
    run_setup = True

    experimenter = PyExperimenter(experiment_configuration_file_path=exp_learner_sampler_file,
                                  database_credential_file_path=db_config_file)

    db_name = experimenter.config.database_configuration.database_name
    db_credentials = experimenter.db_connector._get_database_credentials()

    dbbc = MySQLBenchmarkConnector(host=db_credentials["host"], user=db_credentials["user"],
                                   password=db_credentials["password"], database=db_name)

    if run_setup:
        from DefaultSetup import ensure_default_setup
        ensure_default_setup(dbbc=dbbc)

        setting_combinations = [{'setting_name': 'small', 'openml_id': oid} for oid in small_openml_ids]
        setting_combinations += [{'setting_name': 'medium', 'openml_id': oid} for oid in medium_openml_ids]
        setting_combinations += [{'setting_name': 'large-10', 'openml_id': oid} for oid in large_openml_ids]
        setting_combinations += [{'setting_name': 'large-20', 'openml_id': oid} for oid in large_openml_ids]

        experimenter.fill_table_from_combination(
            parameters={
                "learner_name": ["svm_lin", "svm_rbf", "rf_entropy", "rf_gini", "rf_entropy_large",
                                 "rf_gini_large", "knn_3", "knn_10", "log_reg", "multinomial_bayes",
                                 "etc_entropy", "etc_gini", "etc_entropy_large", "etc_gini_large",
                                 "naive_bayes", "mlp", "GBT_logloss", "GBT_exp", "GBT_logloss_large",
                                 "GBT_exp_large"],
                "sampling_strategy_name": ["random", "entropy", "margin", "least_confident", "mc_logloss",
                                           "mc_misclass", "discrim", "qbc_entropy", "qbc_kl", "bald", "power_margin",
                                           "random_margin", "min_margin", "expected_avg", "typ_cluster",
                                           "weighted_cluster", "random_margin"],
                "test_split_seed": np.arange(5),
                "train_split_seed": np.arange(5),
                "seed": np.arange(30)
            },
            fixed_parameter_combinations=setting_combinations)

    experimenter.execute(run_experiment, -1)


if __name__ == "__main__":
    main()
