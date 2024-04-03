from py_experimenter.experimenter import PyExperimenter, ResultProcessor
from ALP.benchmark.BenchmarkConnector import MySQLBenchmarkConnector
from ALP.pipeline.ActiveLearningPipeline import ActiveLearningPipeline
from ALP.pipeline.Oracle import Oracle
from ALP.evaluation.experimenter.LogTableObserver import LogTableObserver
from ALP.pipeline.SamplingStrategy import (
    BatchBaldSampling,
    DiscriminativeSampling,
    EntropySampling,
    ExpectedAveragePrecision,
    LeastConfidentSampling,
    MarginSampling,
    MinMarginSampling,
    MonteCarloEERLogLoss,
    MonteCarloEERMisclassification,
    PowerMarginSampling,
    QueryByCommitteeEntropySampling,
    QueryByCommitteeKLSampling,
    RandomMarginSampling,
    RandomSamplingStrategy,
    TypicalClusterSampling,
    WeightedClusterSampling,
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier

exp_setting_file = "config/exp_setting_conf.yml"
exp_scenario_file = "config/exp_scenario_conf.yml"
exp_learner_sampler_file = "config/exp_learner_sampler_conf.yml"
db_config_file = "config/db_conf.yml"

setup_table = False

small_openml_ids = [3, 6, 8, 10, 11, 12, 14, 15, 16, 18, 20, 21, 22, 23, 26, 28, 29, 30, 31, 32, 36, 37, 39, 40,
                           41, 43, 44, 45, 46, 48, 49, 50, 53, 54, 59, 60, 61, 62, 151, 155, 161, 162, 164, 180, 181, 182,
                           183, 184, 187, 197, 209, 219, 223, 279, 285, 287, 292, 294, 300, 307, 312, 313, 329, 333, 334, 335,
                           336, 337, 338, 375, 377, 383, 384, 385, 386, 387, 388, 389, 391, 392, 394, 395, 397, 398, 400, 401,
                           444, 446, 448, 458, 461, 463, 464, 469, 475, 478, 679, 685, 694, 714, 715, 716, 717, 718, 719, 720,
                           721, 722, 723, 724, 725, 726, 727, 728, 730, 732, 733, 734, 735, 736, 737, 740, 741, 742, 743, 744,
                           745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 756, 761, 762, 763, 766, 768, 769, 770, 771, 772,
                           773, 774, 775, 776, 778, 779, 782, 783, 784, 788, 789, 792, 793, 794, 795, 796, 797, 799, 801, 803,
                           805, 806, 807, 808, 811, 812, 813, 814, 816, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828,
                           829, 830, 832, 833, 834, 837, 838, 841, 843, 845, 846, 847, 849, 850, 851, 853, 855, 860, 863, 865,
                           866, 867, 868, 869, 870, 871, 872, 873, 875, 876, 877, 878, 879, 880, 881, 884, 885, 886, 888, 889,
                           890, 895, 896, 900, 901, 902, 903, 904, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917,
                           918, 920, 921, 922, 923, 924, 925, 926, 931, 932, 933, 934, 935, 936, 937, 941, 943, 948, 949, 952,
                           953, 954, 955, 956, 958, 959, 962, 965, 969, 970, 971, 973, 974, 976, 978, 979, 980, 983, 987, 991,
                           994, 995, 996, 997, 1004, 1005, 1006, 1011, 1012, 1013, 1014, 1016, 1019, 1020, 1021, 1022, 1025, 1026,
                           1036, 1038, 1040, 1041, 1043, 1044, 1045, 1046, 1048, 1049, 1050, 1054, 1059, 1061, 1063, 1064, 1065,
                           1066, 1067, 1071, 1073, 1075, 1077, 1078, 1080, 1084, 1100, 1106, 1115, 1116, 1120, 1121, 1122, 1123,
                           1124, 1125, 1126, 1127, 1129, 1131, 1132, 1133, 1135, 1136, 1137, 1140, 1141, 1143, 1144, 1145, 1147,
                           1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1162, 1163, 1164, 1165,
                           1167, 1169, 1217, 1236, 1237, 1238, 1412, 1413, 1441, 1442, 1443, 1444, 1446, 1447, 1448, 1449, 1450,
                           1451, 1453, 1454, 1455, 1457, 1459, 1460, 1464, 1467, 1471, 1472, 1473, 1475, 1481, 1482, 1483, 1486,
                           1487, 1488, 1489, 1496, 1498, 1500, 1501, 1503, 1505, 1507, 1508, 1509, 1510, 1512, 1513, 1519, 1520,
                           1527, 1528, 1529, 1531, 1533, 1534, 1535, 1537, 1538, 1539, 1540, 1542, 1543, 1545, 1546, 1556, 1557,
                           1565, 1568, 1600, 2074, 2079, 3021, 3022, 3560, 3902, 3904, 3913, 3917, 4134, 4135, 4153, 4340, 4534,
                           4538, 7592, 9910, 9946, 9952, 9957, 9960, 9964, 9971, 9976, 9977, 9978, 9981, 9985, 10093, 10101, 14952,
                           14954, 14965, 14969, 14970, 40474, 40475, 40476, 40477, 40478, 125920, 125922, 146195, 146800, 146817,
                           146819, 146820, 146821, 146822, 146824, 146825, 167119, 167120, 167121, 167124, 167125, 167140, 167141]
medium_openml_ids = [3, 6, 11, 12, 14, 15, 16, 18, 20, 21, 22, 23, 26, 28, 29, 30, 31, 32, 36, 37, 44, 45, 46, 50, 54, 60, 151, 155, 161, 162, 180, 181, 182, 183, 184, 197, 209, 219, 223, 279, 287, 292, 294, 300, 307, 312, 313, 333, 334, 335, 375, 377, 383, 385, 386, 389, 391, 392, 394, 395, 398, 400, 401, 458, 469, 478, 679, 715, 717, 718, 720, 722, 723, 725, 727, 728, 734, 735, 737, 740, 741, 742, 743, 749, 750, 751, 752, 761, 766, 770, 772, 774, 779, 792, 795, 797, 799, 803, 805, 806, 807, 813, 816, 819, 821, 822, 823, 824, 825, 826, 827, 833, 837, 838, 841, 843, 845, 846, 847, 849, 853, 855, 866, 869, 870, 871, 872, 879, 881, 884, 886, 888, 896, 901, 903, 904, 910, 912, 913, 914, 917, 920, 923, 926, 931, 934, 936, 937, 943, 948, 949, 953, 954, 958, 959, 962, 970, 971, 976, 978, 979, 980, 983, 987, 991, 994, 995, 997, 1004, 1014, 1016, 1019, 1020, 1021, 1022, 1036, 1038, 1040, 1041, 1043, 1044, 1046, 1049, 1050, 1063, 1067, 1116, 1120, 1137, 1145, 1158, 1165, 1169, 1217, 1236, 1237, 1238, 1443, 1444, 1451, 1453, 1454, 1457, 1459, 1460, 1464, 1467, 1471, 1472, 1475, 1481, 1483, 1486, 1487, 1489, 1496, 1501, 1503, 1505, 1507, 1509, 1510, 1527, 1528, 1529, 1531, 1533, 1534, 1535, 1537, 1538, 1539, 1540, 1542, 1543, 1545, 1546, 1557, 1568, 2074, 2079, 3021, 3022, 3560, 3904, 3917, 4134, 4135, 4534, 4538, 7592, 9910, 9946, 9952, 9957, 9960, 9964, 9971, 9976, 9977, 9978, 9981, 9985, 10093, 10101, 14952, 14954, 14965, 14969, 14970, 40474, 40475, 40476, 40477, 40478, 125920, 125922, 146195, 146800, 146817, 146819, 146820, 146821, 146822, 146824, 146825, 167119, 167120, 167121, 167124, 167125, 167140, 167141]
large_openml_ids =  [3, 6, 12, 14, 16, 18, 20, 21, 22, 26, 28, 30, 32, 36, 44, 45, 46, 60, 151, 155, 161, 162, 180, 182, 183, 184, 197, 209, 219, 279, 287, 294, 300, 312, 375, 389, 391, 395, 398, 720, 722, 725, 727, 728, 734, 735, 737, 752, 761, 772, 803, 807, 816, 819, 821, 822, 823, 833, 843, 846, 847, 871, 881, 901, 914, 923, 948, 953, 958, 959, 962, 971, 976, 978, 979, 980, 991, 995, 1019, 1020, 1021, 1022, 1036, 1038, 1040, 1041, 1043, 1044, 1046, 1050, 1067, 1116, 1120, 1169, 1217, 1236, 1237, 1238, 1457, 1459, 1460, 1471, 1475, 1481, 1483, 1486, 1487, 1489, 1496, 1501, 1503, 1505, 1507, 1509, 1527, 1528, 1529, 1531, 1533, 1534, 1535, 1537, 1538, 1539, 1540, 1557, 1568, 2074, 3021, 4134, 4135, 4534, 4538, 7592, 9910, 9952, 9960, 9964, 9976, 9977, 9978, 9985, 14952, 14965, 14969, 14970, 40474, 40475, 40476, 40477, 40478, 125922, 146195, 146817, 146820, 146821, 146822, 146824, 146825, 167119, 167120, 167121, 167124, 167125, 167140, 167141]

dict_sampling_strategies = {
        "random": RandomMarginSampling(42),
        "entropy": EntropySampling(42),
        "margin": MarginSampling(42),
        "least_confident": LeastConfidentSampling(42),
        "mc_logloss": MonteCarloEERLogLoss(42),
        "mc_misclass": MonteCarloEERMisclassification(42),
        "discrim": DiscriminativeSampling(42),
        "qbc_entropy": QueryByCommitteeEntropySampling(42, 10),
        "qbc_kl": QueryByCommitteeKLSampling(42, 10),
        "bald": BatchBaldSampling(42, 10),
        "power_margin": PowerMarginSampling(42),
        "random_margin": RandomMarginSampling(42),
        "min_margin": MinMarginSampling(42),
        "expected_avg": ExpectedAveragePrecision(42),
        "typ_cluster": TypicalClusterSampling(42),
        "weighted_cluster": WeightedClusterSampling(42)
    }
dict_learner = {
        "svm_lin": SVC(kernel='linear', probability=True),
        "svm_rbf": SVC(kernel='rbf', probability=True),
        "rf_entropy": RandomForestClassifier(n_estimators=100, max_depth=10, criterion='entropy'),
        "rf_gini": RandomForestClassifier(n_estimators=100, max_depth=10, criterion='gini'),
        "rf_entropy_large": RandomForestClassifier(n_estimators=250, max_depth=10, criterion='entropy'),
        "rf_gini_large": RandomForestClassifier(n_estimators=250, max_depth=10, criterion='gini'),
        "knn_3": KNeighborsClassifier(n_neighbors=3),
        "knn_10": KNeighborsClassifier(n_neighbors=10),
        "log_reg": LogisticRegression(),
        "multinomial_bayes": MultinomialNB(),
        "etc_entropy": ExtraTreesClassifier(n_estimators=100, max_depth=10, criterion='entropy'),
        "etc_gini": ExtraTreesClassifier(n_estimators=100, max_depth=10, criterion='gini'),
        "etc_entropy_large": ExtraTreesClassifier(n_estimators=250, max_depth=10, criterion='entropy'),
        "etc_gini_large": ExtraTreesClassifier(n_estimators=250, max_depth=10, criterion='gini'),
        "naive_bayes": GaussianNB(),
        "mlp": MLPClassifier(),
        "GBT_logloss": GradientBoostingClassifier(n_estimators=100),
        "GBT_exp": GradientBoostingClassifier(n_estimators=100, loss='exponential'),
        "GBT_logloss_large": GradientBoostingClassifier(n_estimators=250),
        "GBT_exp_large": GradientBoostingClassifier(n_estimators=250, loss='exponential')

    }


def setup_setting_ids(parameters: dict, result_processor: ResultProcessor, custom_config: dict):
    host = db_config_file["host"]
    user = db_config_file["user"]
    password = db_config_file["password"]
    database = db_config_file["database"]
    connector = MySQLBenchmarkConnector(host, user, password, database)

    SETTING_NAME = parameters["setting_name"]
    SETTING_TRAIN_SIZE = float(parameters["setting_labeled_train_size"])
    SETTING_TRAIN_TYPE = parameters["setting_train_type"]
    SETTING_TEST_SIZE = float(parameters["setting_test_size"])
    NUMBER_OF_IT = int(parameters["number_of_iterations"])
    NUMBER_OF_SAMPLES = int(parameters["number_of_samples"])

    al_setting = connector.load_or_create_setting(setting_name=SETTING_NAME,
                                      setting_labeled_train_size=SETTING_TRAIN_SIZE,
                                      setting_train_type=SETTING_TRAIN_TYPE, setting_test_size=SETTING_TEST_SIZE,
                                      number_of_iterations=NUMBER_OF_IT, number_of_samples=NUMBER_OF_SAMPLES)

def setup_learner_sampling_strategy(parameters: dict, result_processor: ResultProcessor, custom_config: dict):
    host = db_config_file["host"]
    user = db_config_file["user"]
    password = db_config_file["password"]
    database = db_config_file["database"]
    connector = MySQLBenchmarkConnector(host, user, password, database)



    name = parameters["sampling_strategy"]
    obj = dict_sampling_strategies[name]

    descr, obj = connector.load_or_create_sampling_strategy(name, obj)


    name = parameters["learner_name"]
    obj = dict_learner[name]

    descr, obj = connector.load_or_create_learner(name, obj)

def run_experiment(parameters: dict, result_processor: ResultProcessor, custom_config: dict):

    host = db_config_file["host"]
    user = db_config_file["user"]
    password = db_config_file["password"]
    database = db_config_file["database"]

    connector = MySQLBenchmarkConnector(host, user, password, database)

    SETTING_ID = int(parameters["setting_id"])
    OPENML_ID = int(parameters["openml_id"])
    TEST_SPLIT_SEED = int(parameters["test_split_seed"])
    TRAIN_SPLIT_SEED = int(parameters["train_split_seed"])
    SEED = int(parameters["seed"])

    setting = connector.load_setting(SETTING_ID)


    setting_name = setting.setting_name
    if setting_name == 'small' and OPENML_ID not in small_openml_ids or setting_name == 'medium' and OPENML_ID not in medium_openml_ids or setting_name == 'large' and OPENML_ID not in large_openml_ids:
        return

    scenario = connector.load_or_create_scenario(openmlid = OPENML_ID, test_split_seed = TEST_SPLIT_SEED, train_split_seed = TRAIN_SPLIT_SEED, seed = SEED, setting_id = SETTING_ID)
    X_l, y_l, X_u, y_u, X_test, y_test = scenario.get_data_split()


    SAMPLING_STRATEGY = connector.load_sampling_strategy_by_name(parameters["sampling_strategy"])
    LEARNER = connector.load_learner_by_name(parameters["learner"])


    OBSERVER = [LogTableObserver()]
    NUM_ITERATIONS = setting.number_of_iterations
    NUMBER_OF_SAMPLES = setting.number_of_samples

    ALP = ActiveLearningPipeline(learner=SAMPLING_STRATEGY, sampling_strategy=LEARNER, observer_list=OBSERVER,
                                 #init_budget=INIT_BUDGET,
                                 num_iterations=NUM_ITERATIONS, num_samples_per_iteration=NUMBER_OF_SAMPLES)

    oracle = Oracle(X_u, y_u)
    ALP.active_fit(X_l, y_l, X_u, oracle)


def main():
    experimenter_settings = PyExperimenter(experiment_configuration_file_path=exp_setting_file,
                                  database_credential_file_path=db_config_file)

    experimenter_learner_sampler = PyExperimenter(experiment_configuration_file_path=exp_learner_sampler_file,
                                  database_credential_file_path=db_config_file)

    experimenter_scenarios = PyExperimenter(experiment_configuration_file_path=exp_scenario_file,
                                  database_credential_file_path=db_config_file)



    if setup_table:
        setting_combinations = [{'setting_name': 'small', 'setting_labeled_train_size': 30,  'number_of_samples': 1}]
        setting_combinations += [{'setting_name': 'medium', 'setting_labeled_train_size': 100, 'number_of_samples': 5}]
        setting_combinations += [{'setting_name': 'large', 'setting_labeled_train_size': 300, 'number_of_samples': num_to_label} for num_to_label in [10, 20]]

        experimenter_settings.fill_table_from_combination(parameters={'setting_train_type': ['absolute'], 'setting_test_size': ['0.33'], 'number_of_iterations': ['20']},
                                                 fixed_parameter_combinations=setting_combinations)

    experimenter_settings.execute(setup_setting_ids, -1)

    if setup_table:
        experimenter_learner_sampler.fill_table_from_config()
    experimenter_learner_sampler.execute(setup_learner_sampling_strategy, -1)

    if setup_table:
        experimenter_scenarios.fill_table_from_config()
    experimenter_scenarios.execute(run_experiment, -1)


if __name__ == "__main__":
    main()