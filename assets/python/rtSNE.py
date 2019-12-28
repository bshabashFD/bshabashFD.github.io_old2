from sklearn.manifold import TSNE
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor


class rtSNE(TSNE):
    '''
    repeatable t-distributed Stochastic Neighbor Embedding.
    
    This class is based on, and inherits from, sklearn.manifold.TSNE. 
    However, in addition it allows for a repeatable use of the embedding through an estimator of 
    choice (either Decision Tree or K-Nearest Neighbor).

    parameters:
    --------------
    n_components : int, optional (default: 2) (documentation taken from sklearn.manifold.TSNE)
       Dimension of the embedded space.
   
    perplexity : float, optional (default: 30) (documentation taken from sklearn.manifold.TSNE)
       The perplexity is related to the number of nearest neighbors that
       is used in other manifold learning algorithms. Larger datasets
       usually require a larger perplexity. Consider selecting a value
       between 5 and 50. The choice is not extremely critical since t-SNE
       is quite insensitive to this parameter.

    early_exaggeration : float, optional (default: 12.0) (documentation taken from sklearn.manifold.TSNE)
       Controls how tight natural clusters in the original space are in
       the embedded space and how much space will be between them. For
       larger values, the space between natural clusters will be larger
       in the embedded space. Again, the choice of this parameter is not
       very critical. If the cost function increases during initial
       optimization, the early exaggeration factor or the learning rate
       might be too high.

    learning_rate : float, optional (default: 200.0) (documentation taken from sklearn.manifold.TSNE)
       The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If
       the learning rate is too high, the data may look like a 'ball' with any
       point approximately equidistant from its nearest neighbours. If the
       learning rate is too low, most points may look compressed in a dense
       cloud with few outliers. If the cost function gets stuck in a bad local
       minimum increasing the learning rate may help.

    n_iter : int, optional (default: 1000) (documentation taken from sklearn.manifold.TSNE)
       Maximum number of iterations for the optimization. Should be at
       least 250.

    n_iter_without_progress : int, optional (default: 300) (documentation taken from sklearn.manifold.TSNE)
       Maximum number of iterations without progress before we abort the
       optimization, used after 250 initial iterations with early
       exaggeration. Note that progress is only checked every 50 iterations so
       this value is rounded to the next multiple of 50.
       

    min_grad_norm : float, optional (default: 1e-7) (documentation taken from sklearn.manifold.TSNE)
       If the gradient norm is below this threshold, the optimization will
       be stopped.

    metric : string or callable, optional (documentation taken from sklearn.manifold.TSNE)
       The metric to use when calculating distance between instances in a
       feature array. If metric is a string, it must be one of the options
       allowed by scipy.spatial.distance.pdist for its metric parameter, or
       a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
       If metric is "precomputed", X is assumed to be a distance matrix.
       Alternatively, if metric is a callable function, it is called on each
       pair of instances (rows) and the resulting value recorded. The callable
       should take two arrays from X as input and return a value indicating
       the distance between them. The default is "euclidean" which is
       interpreted as squared euclidean distance.

    init : string or numpy array, optional (default: "random") (documentation taken from sklearn.manifold.TSNE)
       Initialization of embedding. Possible options are 'random', 'pca',
       and a numpy array of shape (n_samples, n_components).
       PCA initialization cannot be used with precomputed distances and is
       usually more globally stable than random initialization.

    verbose : int, optional (default: 0) (documentation taken from sklearn.manifold.TSNE)
       Verbosity level.

    random_state : int, RandomState instance or None, optional (default: None) 
    (documentation taken from sklearn.manifold.TSNE)
       If int, random_state is the seed used by the random number generator;
       If RandomState instance, random_state is the random number generator;
       If None, the random number generator is the RandomState instance used
       by `np.random`.  Note that different initializations might result in
       different local minima of the cost function.

    method : string (default: 'barnes_hut') (documentation taken from sklearn.manifold.TSNE)
       By default the gradient calculation algorithm uses Barnes-Hut
       approximation running in O(NlogN) time. method='exact'
       will run on the slower, but exact, algorithm in O(N^2) time. The
       exact algorithm should be used when nearest-neighbor errors need
       to be better than 3%. However, the exact method cannot scale to
       millions of examples.

    angle : float (default: 0.5) (documentation taken from sklearn.manifold.TSNE)
       Only used if method='barnes_hut'
       This is the trade-off between speed and accuracy for Barnes-Hut T-SNE.
       'angle' is the angular size (referred to as theta in [3]) of a distant
       node as measured from a point. If this size is below 'angle' then it is
       used as a summary node of all points contained within it.
       This method is not very sensitive to changes in this parameter
       in the range of 0.2 - 0.8. Angle less than 0.2 has quickly increasing
       computation time and angle greater 0.8 has quickly increasing error.
       
    estimator : string, optional (default: 'DecisionTree')
       The estimator to be used to learn the repeatable embedding.
       Possible options are 'DecisionTree' or 'KNN'.
       Employing 'DecisionTree' will produce faster estimations of the embedded
       space, while employing 'KNN' will result in faster run times, but more
       accurate representations of the embedded space.
    
    max_depth : int or None, optional (default: None) (documentation taken from sklearn.tree.DecisionTreeRegressor)
       Only used if estimator='DecisionTree'
       The maximum depth of the tree. If None, then nodes are expanded until all 
       leaves are pure or until additional splits will result in leaves containing 
       less than min_samples_leaf data points
    
    min_samples_leaf : int, float, optional (default: 2) (documentation taken from sklearn.tree.DecisionTreeRegressor)
       Only used if estimator='DecisionTree'
       The minimum number of samples required to be at a leaf node. 
       A split point at any depth will only be considered if it leaves at least 
       min_samples_leaf training samples in each of the left and right branches. 
       This may have the effect of smoothing the model, especially in regression.
       
       * If int, then consider min_samples_leaf as the minimum number.
       * If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) 
         are the minimum number of samples for each node.
    
    n_neighbors : int, optional (default: 5) (documentation taken from sklearn.neighbors.KNeighborsRegressor)
       Only used if estimator='KNN'
       Number of neighbors to use by default for kneighbors queries
    
    weights : str or callable, optional (default: 'distance) (documentation taken from sklearn.neighbors.KNeighborsRegressor)
       Only used if estimator='KNN'
       weight function used in prediction. Possible values:

        * 'uniform' : uniform weights. All points in each neighborhood are weighted equally.
        * 'distance' : weight points by the inverse of their distance. in this case, closer neighbors of a 
                       query point will have a greater influence than neighbors which are further away.
        * [callable] : a user-defined function which accepts an array of distances, and returns an array of the same 
                       shape containing the weights.
    
    n_jobs : int or None, optional (default=1) (documentation taken from sklearn.neighbors.KNeighborsRegressor)
       Only used if estimator='KNN'
       The number of parallel jobs to run for neighbors search. None means 1 unless in a joblib.parallel_backend context. 
       -1 means using all processors. See Glossary for more details. Doesn’t affect fit method of estimator

    Attributes
    ----------
    embedding_ : array-like, shape (n_samples, n_components) (documentation taken from sklearn.manifold.TSNE)
       Stores the embedding vectors.

    kl_divergence_ : float (documentation taken from sklearn.manifold.TSNE)
       Kullback-Leibler divergence after optimization.

    n_iter_ : int (documentation taken from sklearn.manifold.TSNE)
       Number of iterations run.
       
    estimator_instance_ : object (DecisionTreeRegressor or KNeighborsRegressor)
       The estimator used to estimate the embedded space

    Examples
    --------

    >>> import numpy as np
    >>> import rtSNE
    >>> X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    >>> my_rtSNE = rtSNE()
    >>> my_rtSNE.fit(X)
    >>> X_embedded = my_rtSNE.transform(X)
    >>> X_embedded.shape
    (4, 2)
    
    >>> X2 = np.array([[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 0]])
    >>> X2_embedded = my_rtSNE.transform(X2)
    >>> X2_embedded.shape
    (4, 2)
    '''

    ###############################################################################################
    def __init__(self, 
                 n_components=2, 
                 perplexity=30.0, 
                 early_exaggeration=12.0, 
                 learning_rate=200.0, 
                 n_iter=1000, 
                 n_iter_without_progress=300, 
                 min_grad_norm=1e-07, 
                 metric='euclidean', 
                 init='random', 
                 verbose=0, 
                 random_state=None, 
                 method='barnes_hut', 
                 angle=0.5,
                 estimator='DecisionTree',
                 max_depth=None, 
                 min_samples_leaf=2,
                 n_neighbors=5, 
                 weights='distance', 
                 n_jobs=1):
        '''
        (documentation taken from sklearn.manifold.TSNE)
        Initialize self.  See help(type(self)) for accurate signature
        '''
        
        # Initialize superclass with current parameter values
        # The defaults were chosen to be the same as tSNE
        super().__init__(n_components, perplexity, early_exaggeration, learning_rate, n_iter, 
                                    n_iter_without_progress, min_grad_norm, metric, init, verbose, 
                                    random_state, method, angle)
        
        valid_estimators = ['DecisionTree', 'KNN']
        
        if (estimator not in valid_estimators):
            raise ValueError(f'estimator can only be one of {valid_estimators}, found "{estimator}" instead')
            
        
        self.estimator = estimator
        
        self.tree_max_depth = max_depth
        self.tree_min_samples_leaf = min_samples_leaf
        
        self.knn_n_neighbors = n_neighbors
        self.knn_weights = weights
        self.knn_n_jobs = n_jobs
        
        
    ###############################################################################################
    def fit(self, X, y=None):
        '''
        (documentation taken from sklearn.manifold.TSNE)
        
        Fit X into an embedded space. As well as learn an embedding function 
        for repeatability
       
        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. If the method
            is 'exact', X may be a sparse matrix of type 'csr', 'csc'
            or 'coo'.

        y : Ignored
        
        Returns:
        -------
        self : the fitted rtSNE estimator
        '''
        self.estimator_instance_ = None
        
        if (self.estimator == "DecisionTree"):
            self.estimator_instance_ = DecisionTreeRegressor(max_depth = self.tree_max_depth, 
                                                          min_samples_leaf = self.tree_min_samples_leaf)
        else:
            self.estimator_instance_ = KNeighborsRegressor(n_neighbors = self.knn_n_neighbors, 
                                                        weights=self.knn_weights, 
                                                        n_jobs=self.knn_n_jobs)
        
        
        
        super().fit_transform(X,y)
        
        X_reduced = self.embedding_
        
        
        self.estimator_instance_.fit(X, X_reduced)
        
        return self
        
    ###############################################################################################
    def transform(self, X, y=None):
        '''
        (documentation partially taken from sklearn.manifold.TSNE)
        return the transformed mapping learned. Notice this will produce 
        a different output than sklearn.manifold.TSNE.fit_transform since
        this will use the estimated mapping to perform
        the transformation.
        
        This means that sklearn.manifold.TSNE(random_state=1).fit_transform(X)
        and rtSNE(random_state=1).fit(X).transform(X) will likely return
        different results

        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
           If the metric is 'precomputed' X must be a square distance
           matrix. Otherwise it contains a sample per row.

        y : Ignored
        
        Returns
        -------
        X_new : array, shape (n_samples, n_components)
           An embedding of the training data in low-dimensional space.
        '''
        return self.estimator_instance_.predict(X)

    ###############################################################################################
    def fit_transform(self, X, y=None):
        '''
        (documentation partially taken from sklearn.manifold.TSNE)
        Fit X into an embedded space and return that transformed
        output. This method takes advantage of the fit_transform method
        of the superclass sklearn.manifold.TSNE
        
        This means that sklearn.manifold.TSNE(random_state=1).fit_transform(X)
        and rtSNE(random_state=1).fit_transform(X) will return the same
        results

        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
           If the metric is 'precomputed' X must be a square distance
           matrix. Otherwise it contains a sample per row.

        y : Ignored

        Returns
        -------
        X_new : array, shape (n_samples, n_components)
           Embedding of the training data in low-dimensional space.
        
        '''
        X_reduced =  super().fit_transform(X,y)
        self.estimator_instance_.fit(X, X_reduced)
        return X_reduced
    
