import numpy as np
from sympy import Matrix
from scipy.linalg import norm

def sample_with_basis(M):
    weight = np.random.normal(0, 1, len(M))
    v = weight.dot(M)
    # print(v)
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
    # print(self.weights[:,1])
    # print("#*******************#")
    # print(self.weights)
    # print("####################")

  def update_to_mean_winners(self, winners, viewed_list=None, svd=None, project_norm=False, _lambda=None, lambda_intp=None):
    assert self.n_models > 1
    self.u_t = None
    self.g_t = None
    if len(winners) > 0:
      # print 'winners:', winners
      gradient = np.mean(self.weights[:, winners], axis=1) - self.weights[:, 0]
      self.u_t = gradient

      # added for projection
      lambda_gradient = 0
      if viewed_list is not None and len(viewed_list)>0:
        if lambda_intp: # add Linear Interpolation (project only partially)
          gradient = (1-lambda_intp)*gradient + (lambda_intp * self.project_to_viewed_doc(gradient,viewed_list,svd,project_norm))

        else:
          gradient = self.project_to_viewed_doc(gradient,viewed_list,svd,project_norm)
        self.g_t = gradient
        if _lambda: # add L2 Regularization (add back a portion of original weight to gradient)
          lambda_gradient =  _lambda * self.weights[:, 0]

      self.weights[:, 0] += self.learning_rate * gradient
      if lambda_gradient is not 0:
        self.weights[:, 0] -= lambda_gradient
      self.learning_rate *= self.learning_rate_decay
      # print(self.g_t)


  def update_to_documents(self, doc_ind, doc_weights, viewed_list=None, svd=None, project_norm=None, _lambda=None, lambda_intp=None):
    # print("svd: %s, project_norm: %s " %(svd,project_norm))
    weighted_docs = self._last_features[doc_ind, :] * doc_weights[:, None]

    # added for projection
    gradient = np.sum(weighted_docs, axis=0)
    # added for projection
    lambda_gradient = 0
    if viewed_list is not None and len(viewed_list)>0:
      if lambda_intp: # add Linear Interpolation (project only partially)
        gradient = (1-lambda_intp)*gradient + (lambda_intp * self.project_to_viewed_doc(gradient,viewed_list,svd,project_norm))

      else:
        gradient = self.project_to_viewed_doc(gradient,viewed_list,svd,project_norm)
      self.g_t = gradient
      if _lambda: # add L2 Regularization (add back a portion of original weight to gradient)
        lambda_gradient = self.weights[:, 0] * _lambda

    self.weights[:, 0] += self.learning_rate * gradient
    if lambda_gradient is not None:
      self.weights[:, 0] -= 0
    self.learning_rate *= self.learning_rate_decay


##############################################################
  def project_to_viewed_doc(self, winning_gradient, viewed_list, svd, normalize):
    # print("svd: %s, normalize: %s " %(svd,normalize))
    # Make projections to each of viewed document as basis vector
    gradient_proj = np.zeros(self.n_features)

    # viewed_list has each row as the basis, so it is the transpose of columnspace M
    basis_trans = np.matrix.transpose(np.asarray(viewed_list))
    if svd:
        # SVD decomposition, column of both u_ and vh_ is orthogonal basis of columnspace of input
        # Use u matrix for basis, as u_ is 'document-to-concept' simialrity
        # vh_ is 'feature-to-concept' similarity
        u_,s_,vh_ = np.linalg.svd(np.asarray(basis_trans), full_matrices=False)
        # transpose to row space
        basis_list = np.matrix.transpose(np.asarray(u_))

    else:
        # QR decomposition, column of q is orthogonal basis of columnspace of input
        q_,r_ = np.linalg.qr(np.asarray(basis_trans))
        # transpose to row space
        basis_list = np.matrix.transpose(np.asarray(q_))

    
    # proj_g onto x =  dot(x,g)/|x|^2  x
    for basis in basis_list:
        len_basis = np.sqrt(basis.dot(basis)) 
        # len_basis = np.sqrt(sum(k*k for k in basis)) # could take out np.sqrt and square in next line
        gradient_proj += np.dot(basis, winning_gradient) / (len_basis * len_basis) * basis

    if normalize :
      norm = np.linalg.norm(gradient_proj)
      if norm > 0:
        gradient_proj = gradient_proj / norm

    return gradient_proj
##############################################################
  # sample candidate from null space for NSGD
  def sample_candidates_null_space(self, grads, features, withBasis=False):
    assert self.n_models > 1
    # vectors = np.random.randn(self.n_features, self.n_models-1)
    # vector_norms = np.sum(vectors ** 2, axis=0) ** (1. / 2)
    # vectors /= vector_norms[None, :]
    # self.weights[:, 1:] = self.weights[:, 0, None] + vectors

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
