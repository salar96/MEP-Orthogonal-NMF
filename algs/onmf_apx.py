import numpy as np
from utils import divide, normalize
from sklearn.cluster import KMeans


class ONMF_apx:
    name = "ONMF-apx"
    def func (X, k, double = False):
        m, n = np.shape(X)
        l = np.sum(X * X, axis = 0, keepdims = True)
        l_sqrt = np.sqrt(l)
        X_norm = divide (X, l_sqrt)

        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0).fit(X_norm.T, sample_weight = l[0])
        asgn = kmeans.labels_
        centers = kmeans.cluster_centers_.T
        if double:
            q = np.zeros(k)
            for j in range(n):
                q[asgn[j]] += l[0,j]
            for k1 in range(k):
                for k2 in range(k1+1,k):
                    if q[k1] > 1e-10 and q[k2] > 1e-10 and 1/2 <= ONMF_apx.cos(centers[:,k1],centers[:,k2]) <= np.sqrt(3)/2:
                        temp = min(q[k1], q[k2])
                        q[k1] -= temp
                        q[k2] -= temp
            num_group = 0
            group_members = [[] for k1 in range(k)]
            group_weights = [[] for k1 in range(k)]
            group_asgn = [0 for k1 in range(k)]
            for k1 in range(k):
                if q[k1] > 1e-10:
                    grouped = False
                    for k2 in range(num_group):
                        if ONMF_apx.cos(centers[:, k1], group_members[k2][0]) > np.sqrt(3)/2:
                            group_members[k2].append(centers[:,k1])
                            group_weights[k2].append(q[k1])
                            group_asgn[k1] = k2
                            grouped = True
                    if not grouped:
                        group_members[num_group].append(centers[:,k1])
                        group_weights[num_group].append(q[k1])
                        group_asgn[k1] = num_group
                        num_group += 1

            group_mean = np.zeros((m,k))
            group_total_weight_sqrt = np.zeros((k))
            for k2 in range(num_group):
                group_members[k2] = np.array(group_members[k2]).T
                group_weights[k2] = np.array(group_weights[k2]).reshape(1, -1)
                group_total_weight_sqrt[k2] = np.sqrt(np.sum(group_weights[k2]))
                group_mean[:,k2] = np.sum(group_members[k2]*group_weights[k2], axis = 1) / group_total_weight_sqrt[k2]
            A = np.zeros((m,k))
            for i in range(m):
                k2 = np.argmax(group_mean[i])
                A[i,k2] = group_mean[i,k2]/group_total_weight_sqrt[k2]
            W = np.zeros((k,n))
            for j in range(n):
                k2 = group_asgn[asgn[j]]
                norm = np.linalg.norm(A[:,k2])
                if norm > 0:
                    W[k2,j] = X[:,j].T @ A[:,k2] / norm / norm
        else:
            A = centers
            W = np.zeros((k,n))
            for j in range(n):
                k2 = asgn[j]
                norm = np.linalg.norm(A[:,k2])
                if norm > 0:
                    W[k2,j] = X[:,j].T @ A[:,k2] / norm / norm
        if double:
            return A, np.eye(k), W.T
        else:
            return A, W.T
    def cos(x,y):
        return x.T @ y / (np.linalg.norm(x) * np.linalg.norm(y))
