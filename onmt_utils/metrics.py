import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.stats import entropy as H


def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """
    Kernel Density Estimation with Scikit-learn
        x_grid : points where we want an evaluation of the pdf
        x : experimental data points (LIKELIHOOD)
    """
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])

    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)


def jensen_shannon_divergence(prob_distributions, weights, logbase=2):
    """
    Jensen shannon divergence for probability distributions (NO pdf)
        prob_distributions : DISCRETE probability distributions to be
                             compared. Not necessarly normalized.
        weights : weights given to each distribution
    """

    # entropy of mixture
    wprobs = np.multiply(prob_distributions, weights[:, np.newaxis])
    mixture = wprobs.sum(axis=0)
    entropy_of_mixture = H(mixture, base=logbase)

    # weighted sum of entropies
    entropies = np.array([H(P_i, base=logbase) for P_i in prob_distributions])
    wentropies = weights * entropies
    sum_of_entropies = wentropies.sum()

    divergence = entropy_of_mixture - sum_of_entropies
    return divergence, entropy_of_mixture, sum_of_entropies


def CRE(cdf, x_grid):
    """
    Cumulative residual entropy from experimental data points
        cdf : cumulative residual distribution
        x_grid : points at which the 'step occurs'. Points where
                 the cdf is evaluated. Need to be uniformly spaced
        the logbase used is 2
    """

    # Function in the entropy integral
    cdf_func = [(1-cdf_elem) * np.log2(1-cdf_elem) if cdf_elem < 1 else 0 for cdf_elem in cdf]

    # Integral calculation
    cre = (x_grid[1]-x_grid[0]) * np.sum(cdf_func)
    return -cre


def cumulative_jensen_shannon_divergence(cdfs, weights, x_grid):

    """
    Cumulative Jensen shannon divergence (applicable both in discrete and continuous case)
        cdfs : cumulative distributions to be
               compared. Values should be in increasing order and <= 1
        weights : weights given to each distribution
    """
    # entropy of mixture
    wprobs = np.multiply(cdfs, weights[:, np.newaxis])
    mixture = wprobs.sum(axis=0)
    entropy_of_mixture = CRE(mixture, x_grid)

    # weighted sum of entropies
    entropies = np.array([CRE(CP_i, x_grid) for CP_i in cdfs])
    wentropies = weights * entropies
    sum_of_entropies = wentropies.sum()

    cjsd = entropy_of_mixture - sum_of_entropies
    return cjsd, entropy_of_mixture, sum_of_entropies
