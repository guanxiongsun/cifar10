import numpy as np
# from scipy.stats import multivariate_normal


# def likelihood_moment(x, ytk, moment):
#     x_moment = np.power(np.float32(x), moment) if moment > 0 else np.float32([1])
#     return x_moment * ytk
#
#
# def likelihood_statistics(samples, means, covs, weights):
#     gaussians, s0, s1, s2 = {}, {}, {}, {}
#     samples = zip(range(0, len(samples)), samples)
#
#     g = [multivariate_normal(mean=means[k], cov=covs[k]) for k in range(0, len(weights))]
#     for index, x in samples:
#         gaussians[index] = np.array([g_k.pdf(x) for g_k in g])
#
#     for k in range(0, len(weights)):
#         s0[k], s1[k], s2[k] = 0, 0, 0
#         for index, x in samples:
#             probabilities = np.multiply(gaussians[index], weights)
#             probabilities = probabilities / np.sum(probabilities)
#             s0[k] = s0[k] + likelihood_moment(x, probabilities[k], 0)
#             s1[k] = s1[k] + likelihood_moment(x, probabilities[k], 1)
#             s2[k] = s2[k] + likelihood_moment(x, probabilities[k], 2)
#
#     return s0, s1, s2
#
#
# def fisher_vector_weights(s0, s1, s2, means, covs, w, T):
#     return np.float32([((s0[k] - T * w[k]) / np.sqrt(w[k])) for k in range(0, len(w))])
#
#
# def fisher_vector_means(s0, s1, s2, means, sigma, w, T):
#     return np.float32([(s1[k] - means[k] * s0[k]) / (np.sqrt(w[k] * sigma[k])) for k in range(0, len(w))])
#
#
# def fisher_vector_sigma(s0, s1, s2, means, sigma, w, T):
#     return np.float32(
#         [(s2[k] - 2 * means[k] * s1[k] + (means[k] * means[k] - sigma[k]) * s0[k]) / (np.sqrt(2 * w[k]) * sigma[k]) for
#          k in range(0, len(w))])
#
#
# def normalize(fisher_vector):
#     v = np.sqrt(abs(fisher_vector)) * np.sign(fisher_vector)
#     return v / np.sqrt(np.dot(v, v))
#
#
# def fisher_vector(samples, means, covs, w):
#     s0, s1, s2 = likelihood_statistics(samples, means, covs, w)
#     T = samples.shape[0]
#     covs = np.float32([np.diagonal(covs[k]) for k in range(0, covs.shape[0])])
#     a = fisher_vector_weights(s0, s1, s2, means, covs, w, T)
#     b = fisher_vector_means(s0, s1, s2, means, covs, w, T)
#     c = fisher_vector_sigma(s0, s1, s2, means, covs, w, T)
#     fv = np.concatenate([np.concatenate(a), np.concatenate(b), np.concatenate(c)])
#     fv = normalize(fv)
#     return fv
#
#
# def get_fisher_vectors(xx_te, gmm):
#     return np.float32(fisher_vector(xx_te, gmm.means_, gmm.covariances_, gmm.weights_ ))


def fisher_features(xx, gmm):
    # features = [get_fisher_vectors(xx_te, gmm)]
    # return features
    xx = np.atleast_2d(xx)
    N = xx.shape[0]

    # Compute posterior probabilities.
    Q = gmm.predict_proba(xx)  # NxK

    # Compute the sufficient statistics of descriptors.
    Q_sum = np.sum(Q, 0)[:, np.newaxis] / N
    Q_xx = np.dot(Q.T, xx) / N
    Q_xx_2 = np.dot(Q.T, xx ** 2) / N

    # Compute derivatives with respect to mixing weights, means and variances.
    d_pi = Q_sum.squeeze() - gmm.weights_
    d_mu = Q_xx - Q_sum * gmm.means_
    d_sigma = (- Q_xx_2 - Q_sum * gmm.means_ ** 2 + Q_sum * gmm.covariances_ + 2 * Q_xx * gmm.means_)

    # Merge derivatives into a vector.
    return np.hstack((d_pi, d_mu.flatten(), d_sigma.flatten()))

# def main():
#     # Short demo.
#     K = 3
#     N = 15
#
#     xx, _ = make_classification(n_samples=N)
#     xx_tr, xx_te = xx[: -5], xx[-5:]
#
#     gmm = GaussianMixture(n_components=K)
#     gmm.fit(xx_tr)
#
#     fisher = fisher_features(xx_te, gmm)
#
#
# if __name__ == '__main__':
#     main()
