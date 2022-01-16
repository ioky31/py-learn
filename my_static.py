from scipy import stats


def normality_test(WTdataset, ADdataset, significance_level=0.05):
    # W1, p1 = stats.shapiro(WTdataset)
    W1, p1 = stats.normaltest(WTdataset)
    # W2, p2 = stats.kstest(ADdataset, 'norm', args=(ADdataset.mean(), ADdataset.std()))
    W2, p2 = stats.normaltest(ADdataset)
    # W2, p2 = stats.shapiro(ADdataset)
    if (p1 > significance_level) & (p2 > significance_level):
        # 都是正态
        return True
    else:
        # 一组或两组不是正态
        return False

def normality_test_corr(dataset, dataset_half, mode, significance_level=0.05):
    if mode == "AD":
        W1, p1 = stats.normaltest(dataset)
        W2, p2 = stats.normaltest(dataset_half)
    elif mode == "WT":
        W1, p1 = stats.shapiro(dataset)
        W2, p2 = stats.shapiro(dataset_half)

    if (p1 > significance_level) & (p2 > significance_level):
        # 都是正态
        return True
    else:
        # 一组或两组不是正态
        return False

def homogeneity_test_of_variance(WTdataset, ADdataset, significance_level=0.05):
    # W, p = stats.f(WTdataset, ADdataset)
    W, p = stats.levene(WTdataset, ADdataset)

    if p > significance_level:
        # 方差相同
        return True
    else:
        return False


def ttest_equal_var(WTdataset, ADdataset, significance_level=0.05):
    W, p = stats.ttest_ind(WTdataset, ADdataset, equal_var=True)

    if p < significance_level:
        # 均值有显著性差异
        return True, p
    else:
        return False, p


def ttest_inequal_var(WTdataset, ADdataset, significance_level=0.05):
    W, p = stats.ttest_ind(WTdataset, ADdataset, equal_var=False)

    if p < significance_level:
        # 均值有显著性差异
        return True, p
    else:
        return False, p


def non_parametric_test(WTdataset, ADdataset, significance_level=0.05):
    W, p = stats.mannwhitneyu(WTdataset, ADdataset)

    if p < significance_level:
        # 均值有显著性差异
        return True, p
    else:
        return False, p
