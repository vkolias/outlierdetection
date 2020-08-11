import numpy as np
import matplotlib.pyplot as plt 
from scipy.spatial import distance_matrix 
from sklearn.neighbors import LocalOutlierFactor as sklof


class LocalOutlierFactor(object):
    
    def __init__(self, X, min_pts):
        self.X = X
        self.min_pts = min_pts 
        self.dist_mat = distance_matrix(X, X)
        
    def k_distance(self, p):
        return np.sort(self.dist_mat[p])[self.min_pts]
    
    def k_distance_neighborhood(self, p):
        return np.argsort(self.dist_mat[p])[1:self.min_pts + 1]
    
    def reachability_distance(self, p, o):
        return max(self.dist_mat[p, o], self.k_distance(p))   
    
    def local_reachability_density(self, p):
        kn = self.k_distance_neighborhood(p)
        s = 0
        for k in kn:
            rd = self.reachability_distance(p, k)
            s += rd
        return 1 / (s / len(kn))
    
    def local_outlier_factor(self, p):
        kn = self.k_distance_neighborhood(p)
        s = 0
        for o in kn:
            olrd = self.local_reachability_density(o)
            plrd = self.local_reachability_density(p)
            s += olrd/plrd
        return s/len(kn) 
    
    def fit(self):
        lofs = []
        for i in np.arange(len(self.X)):
            lofs.append(self.local_outlier_factor(i))
        return np.array(lofs)
    


np.random.seed(42)

x_inliers = 0.3 * np.random.randn(100, 2)
x_inliers = np.r_[x_inliers + 2, x_inliers -2]
x_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
x = np.r_[x_inliers, x_outliers]

# my lof
lof = LocalOutlierFactor(x, 20)
lofs = lof.fit()
outliers = np.argsort(lofs)[-20:]
# sklearns lof
clf = sklof(n_neighbors=20)
y_pred = clf.fit_predict(x)
lofs_index = np.where(y_pred==-1)

lofs_values = x[lofs_index]
plt.scatter(x[:, 0], x[:, 1], color='k', s=3., label='Data points')
plt.scatter(x[outliers, 0], x[outliers, 1], marker='x', color='r', s=20, label='Outliers')
plt.scatter(lofs_values[:, 0], lofs_values[:, 1], marker='o', color='b', s=3, label='Outliers')
