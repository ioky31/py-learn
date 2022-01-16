from my_stats import *


def stats_flow(WTdataset, ADdataset, significance_level=0.05):
    if normality_test(WTdataset, ADdataset, significance_level):
        if homogeneity_test_of_variance(WTdataset, ADdataset, significance_level):
            bool_, p = ttest_equal_var(WTdataset, ADdataset, significance_level)
            if bool_:
                return True, p, "双正态-齐性-T检验"
            else:
                return False, p, "双正态-齐性-T检验"
        else:
            bool_, p = ttest_inequal_var(WTdataset, ADdataset, significance_level)
            if bool_:
                return True, p, "双正态-非齐性-修正方差T检验"
            else:
                return False, p, "双正态-非齐性-修正方差T检验"
    else:
        bool_, p = non_parametric_test(WTdataset, ADdataset, significance_level)

        if bool_:
            return True, p, "非双正态-非参数检验"
        else:
            return False, p, "非双正态-非参数检验"

def stats_flow_corr(dataset, dataset_half, mode, significance_level=0.05):
    if normality_test_corr(dataset, dataset_half, mode, significance_level):
        if homogeneity_test_of_variance(dataset, dataset_half, significance_level):
            bool_, p = ttest_equal_var(dataset, dataset_half, significance_level)
            if bool_:
                return True, p, "双正态-齐性-T检验"
            else:
                return False, p, "双正态-齐性-T检验"
        else:
            bool_, p = ttest_inequal_var(dataset, dataset_half, significance_level)
            if bool_:
                return True, p, "双正态-非齐性-修正方差T检验"
            else:
                return False, p, "双正态-非齐性-修正方差T检验"
    else:
        bool_, p = non_parametric_test(dataset, dataset_half, significance_level)

        if bool_:
            return True, p, "非双正态-非参数检验"
        else:
            return False, p, "非双正态-非参数检验"
