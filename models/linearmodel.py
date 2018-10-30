import numpy as np

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





  # def sample_candidates(self):
  #   assert self.n_models > 1
  #   vectors = np.random.randn(self.n_features, self.n_models-1)
  #   vector_norms = np.sum(vectors ** 2, axis=0) ** (1. / 2)
  #   vectors /= vector_norms[None, :]
  #   self.weights[:, 1:] = self.weights[:, 0, None] + vectors

  #   ### From lerot
  #   delta = 1.0
  #   for i in range(0, self.n_models-1):
  #       u = self.sample_unit_sphere(self.n_features)
  #       r = self.w + delta * u
  #       self.weights[:, 1:] = self.weights[:, 0, None] + delta * u
  #   ###


    

  def update_to_mean_winners(self, winners, viewed_list=None):
    assert self.n_models > 1
    if len(winners) > 0:
      # print 'winners:', winners
      gradient = np.mean(self.weights[:, winners], axis=1) - self.weights[:, 0]
      # added for projection
      if viewed_list:
        gradient = self.project_to_viewed_doc(gradient,viewed_list,True,True)

      self.weights[:, 0] += self.learning_rate * gradient
      self.learning_rate *= self.learning_rate_decay

  def update_to_documents(self, doc_ind, doc_weights, viewed_list=None):
    weighted_docs = self._last_features[doc_ind, :] * doc_weights[:, None]
    # added for projection
    gradient = np.sum(weighted_docs, axis=0)
    if viewed_list:
      gradient = self.project_to_viewed_doc(gradient,viewed_list,True,True)
    self.weights[:, 0] += self.learning_rate * gradient
    self.learning_rate *= self.learning_rate_decay

##############################################################
  def project_to_viewed_doc(self, winning_gradient, viewed_list, svd, normalize):
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


    norm = np.linalg.norm(gradient_proj)
    if normalize and norm > 0:
        gradient_proj = gradient_proj / norm

    return gradient_proj
##############################################################