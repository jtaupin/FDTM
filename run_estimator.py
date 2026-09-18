from main import*

class Outputs:  # Class for storing outputs
    def __init__(self):
        self.distances = []
        self.paths = []
        self.durations = []

def run_estimator(metric, Xs:ndarray[float], knns:ndarray[int]=None, dtms:ndarray=None, precisions:ndarray[int]=None, avg:int=1, sources:ndarray[int]=0, targets:ndarray[int]=1, **metric_kwargs):
    '''
    Runs the estimation of the given metric for several sizes of point clouds and repeating `avg` times.
    Only one source and target for each point cloud.
    `metric` must be either `Euclidean`, `Fermat` or `FDTM` and indicates the metric estimated.
    If source or target are not specified, defaults to 0 and 1 respectively. If only one integer is provided, it is used for all repetitions. Else they must have same shape as `Xs`, that is (k, avg).
    `knns` is used to select edges in the graph. If not provided, all edges are used. If provided the shape must be (k,).
    `precisions` sets the number of points used to evaluate the integral of the DTM over edges, and has shape (k,). If not specified it defaults to 2.
    '''
    # Preliminaries
    same_source = isinstance(sources, int)
    if same_source : source = sources
    same_target = isinstance(targets, int)
    if same_target : target = targets
    if precisions is None : precisions = np.full(len(Xs), 2, dtype=int)

    outputs = Outputs()

    # Loop on sizes
    for k, X in tqdm(enumerate(Xs), total=len(Xs), dynamic_ncols=True):
        t, d = 0, []  # Store average runtime and list of distances
        # Loop on repetitions for averaging
        for a in tqdm(range(avg), leave=False, dynamic_ncols=True, disable=(avg==1)):
            if not same_source : source = sources[k, a]
            if not same_target : target = targets[k, a]
            
            t0 = time()
            
            if metric is FDTM : M = FDTM(X[a], dtms[k][a])
            else : M = metric(X[a], **metric_kwargs)
            
            if knns is not None : M.select_edges(knn=knns[k])
            
            M.make_edges(precision_segment=precisions[k])
            M.shortest_path(source=source)
            path = M.path((source, target))[0]
            dist = M.distances[source][target]
            d.append(dist)
            
            if a==0 : outputs.paths.append(path)  # Only store the path of the first repetition for plotting purposes
            
            t += (time()-t0) / avg
        
        outputs.distances.append(d)
        outputs.durations.append(t)

    # Convert to numpy array
    outputs.distances = np.array(outputs.distances)
    outputs.durations = np.array(outputs.durations)

    return outputs