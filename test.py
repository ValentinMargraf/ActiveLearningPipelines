from ALPScenario import ActiveLearningPipeline as ALP
import numpy as np
import sklearn


def check_ids():

    openmlids = [3, 6, 11, 12, 14, 15, 16, 18, 22, 23, 28, 29, 31, 32, 37, 43, 45, 49, 53, 219, 2074, 2079, 3021, 3022, 3481, 3549, 3560, 3573, 3902, 3903, 3904, 3913, 3917, 3918, 7592, 9910, 9946, 9952, 9957, 9960, 9964, 9971, 9976, 9977, 9978, 9981, 9985, 10093, 10101, 14952, 14954, 14965, 14969, 14970, 125920, 125922, 146195, 146800, 146817, 146819, 146820, 146821, 146822, 146824, 146825, 167119, 167120, 167121, 167124, 167125, 167140, 167141, 3, 6, 8, 10, 11, 12, 14, 15, 16, 18, 20, 21, 22, 23, 26, 28, 29, 30, 31, 32, 36, 37, 39, 40, 41, 43, 44, 46, 48, 49, 50, 53, 54, 59, 60, 61, 62, 151, 155, 161, 162, 164, 180, 181, 182, 183, 184, 187, 197, 209, 223, 279, 285, 287, 292, 294, 300, 307, 312, 313, 329, 333, 334, 335, 336, 337, 338, 375, 377, 383, 384, 385, 386, 387, 388, 389, 391, 392, 394, 395, 397, 398, 400, 401, 444, 446, 448, 458, 461, 463, 464, 469, 475, 478, 679, 685, 694, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 730, 732, 733, 734, 735, 736, 737, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 756, 761, 762, 763, 766, 768, 769, 770, 771, 772, 773, 774, 775, 776, 778, 779, 782, 783, 784, 788, 789, 792, 793, 794, 795, 796, 797, 799, 801, 803, 805, 806, 807, 808, 811, 812, 813, 814, 816, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 832, 833, 834, 837, 838, 841, 843, 845, 846, 847, 849, 850, 851, 853, 855, 860, 863, 865, 866, 867, 868, 869, 870, 871, 872, 873, 875, 876, 877, 878, 879, 880, 881, 884, 885, 886, 888, 889, 890, 895, 896, 900, 901, 902, 903, 904, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 920, 921, 922, 923, 924, 925, 926, 931, 932, 933, 934, 935, 936, 937, 941, 943, 948, 949, 952, 953, 954, 955, 956, 958, 959, 962, 965, 969, 970, 971, 973, 974, 976, 978, 979, 980, 983, 987, 991, 994, 995, 996, 997, 1004, 1005, 1006, 1011, 1012, 1013, 1014, 1016, 1019, 1020, 1021, 1022, 1025, 1026, 1036, 1038, 1040, 1041, 1043, 1044, 1045, 1046, 1048, 1049, 1050, 1054, 1059, 1061, 1063, 1064, 1065, 1066, 1067, 1071, 1073, 1075, 1077, 1078, 1080, 1084, 1100, 1106, 1115, 1116, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1129, 1131, 1132, 1133, 1135, 1136, 1137, 1140, 1141, 1143, 1144, 1145, 1147, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1162, 1163, 1164, 1165, 1167, 1169, 1217, 1236, 1237, 1238, 1412, 1413, 1441, 1442, 1443, 1444, 1446, 1447, 1448, 1449, 1450, 1451, 1453, 1454, 1455, 1457, 1459, 1460, 1464, 1467, 1471, 1472, 1473, 1475, 1481, 1482, 1483, 1486, 1487, 1488, 1489, 1496, 1498, 1500, 1501, 1503, 1505, 1507, 1508, 1509, 1510, 1512, 1513, 1519, 1520, 1527, 1528, 1529, 1531, 1533, 1534, 1535, 1537, 1538, 1539, 1540, 1542, 1543, 1545, 1546, 1556, 1557, 1565, 1568, 1600, 3560, 3902, 3904, 3913, 3917, 4134, 4135, 4153, 4340, 4534, 4538, 9976, 9977, 9978, 9981, 9985, 10093, 10101, 14952, 14954, 14965, 14969, 14970, 40474, 40475, 40476, 40477, 40478, 125920, 125922, 146195, 146800, 146817, 146819, 146821, 146822, 146824, 146825, 167119, 167120, 167124, 167125, 167140, 167141]

    num_to_labels = [1,10,20]
    kickout = [[],[],[]]
    minimal_sizes_of_dataset = [100,500,1500]

    for i, setting in enumerate(["small", "medium", "large"]):
        minimal_size_of_dataset = minimal_sizes_of_dataset[i]
        num_to_label = num_to_labels[i]
        for openmlid in openmlids:
            for seed in np.arange(30):
                print(f"Trying {openmlid} with seed {seed} and setting {setting}")
                ALPipe = ALP(1, "test", openmlid, seed, setting, 4, "random", "rf_entropy", "random", "rf_entropy", num_to_label, 10)
                try:
                    ALPipe.init(testrun=True)
                    if ALPipe.num_classes == 1 or ALPipe.num_X < minimal_size_of_dataset:
                        print(f"Failed for {openmlid} with seed {seed} and setting {setting}")
                        kickout[i].append(openmlid)
                        break
                except sklearn.utils._param_validation.InvalidParameterError as e:
                    print(f"Failed for {openmlid} with seed {seed} and setting {setting}: InvalidParameterError")
                    print(e)  # Print the exception message for debugging
                    kickout[i].append(openmlid)
                    break


                kickout_small = np.array(kickout[0])
                kickout_medium = np.array(kickout[1])
                kickout_large = np.array(kickout[2])

                np.save("setting_dataset_ids/kickout_small.npy", kickout_small)
                np.save("setting_dataset_ids/kickout_medium.npy", kickout_medium)
                np.save("setting_dataset_ids/kickout_large.npy", kickout_large)

    take_small = list(set(openmlids) - set(list(kickout_small)))
    take_medium = list(set(openmlids) - set(list(kickout_medium)))
    take_large = list(set(openmlids) - set(list(kickout_large)))

    np.save("setting_dataset_ids/take_small.npy", take_small)
    np.save("setting_dataset_ids/take_medium.npy", take_medium)
    np.save("setting_dataset_ids/take_large.npy", take_large)


def check_queries():
    ALPipe = ALP(1, "test", 3, 0, "small", 4, "random", "rf_entropy", "random", "rf_entropy", 10,
                 10)
    ALPipe.init(testrun=False)
    X_l = ALPipe.Xl
    y_l = ALPipe.yl
    X_u = ALPipe.Xu
    num_samples = 10
    learner = ALPipe.learner
    # from num_saples "uncovered" cluster (there where are no X_l) select the one with highest "typicality"
    pool_size = len(y_l)
    num_cluster = pool_size + num_samples
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import euclidean_distances
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kmeans = KMeans(n_clusters=num_cluster)
        X = np.concatenate((X_l, X_u))
        kmeans.fit(X)
        class_ids = kmeans.labels_
        labeled_cluster_classes = np.unique(kmeans.labels_[:pool_size])
        cluster_sizes = [len(np.argwhere(kmeans.labels_ == i)) for i in range(num_cluster)]
        ids_by_size = np.argsort(-np.array(cluster_sizes))
        ct = 0
        for idx in ids_by_size:
            if idx not in labeled_cluster_classes:
                instances = np.argwhere(kmeans.labels_ == class_ids[idx])
                if ct==num_samples:
                    return instances
                ct += 1

def main():
    #check_ids()
    check_queries()
# test query strategies


if __name__ == "__main__":
    main()