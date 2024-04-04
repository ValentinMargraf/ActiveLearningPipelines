from abc import ABC, abstractmethod

import openml


class BenchmarkSuite(ABC):

    def __init__(self, name="SALTBench-Suite"):
        self.name = name

    @abstractmethod
    def get_openml_dataset_ids(self):
        pass


class OpenMLBenchmarkSuite(BenchmarkSuite):

    def __init__(self, openml_benchmark_id, name="OpenML-BenchmarkSuite"):
        super(OpenMLBenchmarkSuite, self).__init__(name=name + "-" + str(openml_benchmark_id))
        self.openml_benchmark_id = openml_benchmark_id

        benchmark = openml.study.get_suite(openml_benchmark_id)
        tasks = openml.tasks.list_tasks(task_id=benchmark.tasks)

        self.openml_dataset_ids = list()
        for k, t in tasks.items():
            self.openml_dataset_ids.append(t['did'])

    def get_openml_dataset_ids(self):
        return self.openml_dataset_ids


class SALTBenchmarkSuiteLarge(BenchmarkSuite):
    def __init__(self):
        super().__init__(name="SALTBenchSuite-Large")

    def get_openml_dataset_ids(self):
        return [3, 6, 12, 14, 16, 18, 20, 21, 22, 26, 28, 30, 32, 36, 44, 45, 46, 60, 151, 155, 161, 162, 180, 182,
                183, 184, 197, 209, 219, 279, 287, 294, 300, 312, 375, 389, 391, 395, 398, 720, 722, 725, 727, 728,
                734, 735, 737, 752, 761, 772, 803, 807, 816, 819, 821, 822, 823, 833, 843, 846, 847, 871, 881, 901,
                914, 923, 948, 953, 958, 959, 962, 971, 976, 978, 979, 980, 991, 995, 1019, 1020, 1021, 1022, 1036,
                1038, 1040, 1041, 1043, 1044, 1046, 1050, 1067, 1116, 1120, 1169, 1217, 1236, 1237, 1238, 1457,
                1459, 1460, 1471, 1475, 1481, 1483, 1486, 1487, 1489, 1496, 1501, 1503, 1505, 1507, 1509, 1527,
                1528, 1529, 1531, 1533, 1534, 1535, 1537, 1538, 1539, 1540, 1557, 1568, 2074, 3021, 4134, 4135,
                4534, 4538, 7592, 9910, 9952, 9960, 9964, 9976, 9977, 9978, 9985, 14952, 14965, 14969, 14970, 40474,
                40475, 40476, 40477, 40478, 125922, 146195, 146817, 146820, 146821, 146822, 146824, 146825, 167119,
                167120, 167121, 167124, 167125, 167140, 167141]


class SALTBenchmarkSuiteMedium(BenchmarkSuite):
    def __init__(self):
        super().__init__(name="SALTBenchSuite-Medium")

    def get_openml_dataset_ids(self):
        large_ids = SALTBenchmarkSuiteLarge().get_openml_dataset_ids()
        medium_ids = large_ids + [11, 15, 23, 29, 31, 37, 50, 54, 181, 223, 292, 307, 313, 333, 334, 335, 377, 383, 385,
                                  386, 392, 394, 400, 401, 458, 469, 478, 679, 715, 717, 718, 723, 740, 741, 742, 743,
                                  749, 750, 751, 766, 770, 774, 779, 792, 795, 797, 799, 805, 806, 813, 824, 825, 826,
                                  827, 837, 838, 841, 845, 849, 853, 855, 866, 869, 870, 872, 879, 884, 886, 888, 896,
                                  903, 904, 910, 912, 913, 917, 920, 926, 931, 934, 936, 937, 943, 949, 954, 970, 983,
                                  987, 994, 997, 1004, 1014, 1016, 1049, 1063, 1137, 1145, 1158, 1165, 1443, 1444, 1451,
                                  1453, 1454, 1464, 1467, 1472, 1510, 1542, 1543, 1545, 1546, 2079, 3022, 3560, 3904,
                                  3917, 9946, 9957, 9971, 9981, 10093, 10101, 14954, 125920, 146800, 146819]
        medium_ids = sorted(medium_ids)
        return medium_ids


class SALTBenchmarkSuiteSmall(BenchmarkSuite):
    def __init__(self):
        super().__init__(name="SALTBenchSuite-Small")

    def get_openml_dataset_ids(self):
        medium_ids = SALTBenchmarkSuiteMedium().get_openml_dataset_ids()
        small_ids = medium_ids + [8, 10, 39, 40, 41, 43, 48, 49, 53, 59, 61, 62, 164, 187, 285, 329, 336, 337, 338, 384,
                                  387, 388, 397, 444, 446, 448, 461, 463, 464, 475, 685, 694, 714, 716, 719, 721, 724,
                                  726, 730, 732, 733, 736, 744, 745, 746, 747, 748, 753, 754, 756, 762, 763, 768, 769,
                                  771, 773, 775, 776, 778, 782, 783, 784, 788, 789, 793, 794, 796, 801, 808, 811, 812,
                                  814, 818, 820, 828, 829, 830, 832, 834, 850, 851, 860, 863, 865, 867, 868, 873, 875,
                                  876, 877, 878, 880, 885, 889, 890, 895, 900, 902, 906, 907, 908, 909, 911, 915, 916,
                                  918, 921, 922, 924, 925, 932, 933, 935, 941, 952, 955, 956, 965, 969, 973, 974, 996,
                                  1005, 1006, 1011, 1012, 1013, 1025, 1026, 1045, 1048, 1054, 1059, 1061, 1064, 1065,
                                  1066, 1071, 1073, 1075, 1077, 1078, 1080, 1084, 1100, 1106, 1115, 1121, 1122, 1123,
                                  1124, 1125, 1126, 1127, 1129, 1131, 1132, 1133, 1135, 1136, 1140, 1141, 1143, 1144,
                                  1147, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1159, 1160, 1162,
                                  1163, 1164, 1167, 1412, 1413, 1441, 1442, 1446, 1447, 1448, 1449, 1450, 1455, 1473,
                                  1482, 1488, 1498, 1500, 1508, 1512, 1513, 1519, 1520, 1556, 1565, 1600, 3902, 3913,
                                  4153, 4340]
        small_ids = sorted(small_ids)
        return small_ids
