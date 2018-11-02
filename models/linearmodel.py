import numpy as np
from sympy import Matrix
from scipy.linalg import norm

def sample_with_basis(M):
    weight = np.random.normal(0, 1, len(M))
    v = weight.dot(M)
    v /= norm(v)
    return v

class LinearModel(object):
  def __init__(self, n_features, learning_rate,
               n_candidates=0, learning_rate_decay=1.0):
    self.n_features = n_features
    self.learning_rate = learning_rate
    self.n_models = n_candidates + 1
    self.weights = np.zeros((n_features, self.n_models))
    self.learning_rate_decay = learning_rate_decay

  def copy(self):
    copy = LinearModel(n_features = self.n_features,
                       learning_rate = self.learning_rate,
                       n_candidates = self.n_models-1)
    copy.weights = self.weights.copy()
    return copy

  def candidate_score(self, features):
    self._last_features = features
    return np.dot(features, self.weights).T

  def score(self, features):
    self._last_features = features
    return np.dot(features, self.weights[:,0:1])[:,0]

  def sample_candidates(self):
    assert self.n_models > 1
    vectors = np.random.randn(self.n_features, self.n_models-1)
    vector_norms = np.sum(vectors ** 2, axis=0) ** (1. / 2)
    vectors /= vector_norms[None, :]
    self.weights[:, 1:] = self.weights[:, 0, None] + vectors

  def update_to_mean_winners(self, winners):
    assert self.n_models > 1
    if len(winners) > 0:
      # print 'winners:', winners
      gradient = np.mean(self.weights[:, winners], axis=1) - self.weights[:, 0]
      self.weights[:, 0] += self.learning_rate * gradient
      self.learning_rate *= self.learning_rate_decay

  def update_to_documents(self, doc_ind, doc_weights):
    weighted_docs = self._last_features[doc_ind, :] * doc_weights[:, None]
    gradient = np.sum(weighted_docs, axis=0)
    self.weights[:, 0] += self.learning_rate * gradient
    self.learning_rate *= self.learning_rate_decay

  # sample candidate from null space for NSGD
  def sample_candidates_null_space(self, grads, features, withBasis=False):
    assert self.n_models > 1
    vectors = np.random.randn(self.n_features, self.n_models-1)
    vector_norms = np.sum(vectors ** 2, axis=0) ** (1. / 2)
    vectors /= vector_norms[None, :]
    self.weights[:, 1:] = self.weights[:, 0, None] + vectors

    N = Matrix(grads).nullspace() #  get null space of gradient matrix
    newN = np.array(N).astype(np.float64)
    for i in range(0, len(newN)):
        norm = np.linalg.norm(newN[i])
        if norm > 0:
            newN[i] = newN[i]/norm

    # sample vectors normally from the nullspace
    if withBasis:
    # sample with basis
        nsVecs = [sample_with_basis(newN) for i in range(2*self.n_models)]
    else:
    # Directly sample from null space
        nsVecs = [newN[randint(0, len(N) - 1)] for i in range(2*self.n_models)]

    # get average candidate document feature vector
    avgdocfeat = [sum(feat)/len(feat) for feat in zip(*features)]
    # sort vectors by dot product (decreasing absolute value)
    nsVecs = sorted(nsVecs, key=lambda vec: abs(np.dot(vec, avgdocfeat)), reverse=True)

    self.gs = np.array(nsVecs[:self.n_models-1])
    self.weights[:, 1:] = self.weights[:, 0, None] + self.gs.T
