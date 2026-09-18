from main import*

def dtm_circle_radius(r:ndarray[float], dtm_arg:DTM_arg):
    '''
    Computes the DTM of the uniform circle in polar coordinate. Only support `p=2`.
    '''
    assert dtm_arg.p == 2, "`dtm_circle_radius` only supports `p=2`."
    if dtm_arg.m == 0 : return np.abs(r-1)**dtm_arg.beta
    else: # Explicit formula
        pim = np.pi*dtm_arg.m
        return (1 + r**2 - 2*r * np.sin(pim)/pim)**(dtm_arg.beta/2)

def dtm_circle(x:ndarray[float], dtm_arg:DTM_arg):
    '''
    Computes the DTM at `x` with shape `(2,)` or `(n, 2)`.
    '''
    x = np.atleast_2d(x)
    return dtm_circle_radius(norm(x, axis=1), dtm_arg)


def circle_grid(N:int):
    '''
    Creates a point cloud of N points uniformly on the unit circle.
    '''
    angle = np.linspace(0, 2*np.pi, N, endpoint=False)
    return np.transpose(np.array([np.cos(angle), np.sin(angle)]))


class Grid:
    def __init__(self, Xs:ndarray[float, (...,...)], Ys:ndarray[float, (...,...)]):
        '''
        2d grid class for plotting purpose.
        `Xs` and `Ys` must have shape (Lx, Ly) which is the shape of the grid itself.
        Total points : Lx * Ly
        '''
        assert Xs.shape == Ys.shape
        self.Xs = Xs
        self.Ys = Ys
        self.bounding_box = [Xs.min(), Xs.max(), Ys.min(), Ys.max()]
        self.Lx, self.Ly = Xs.shape
        self.n = self.Lx * self.Ly
        self.points = np.array((Xs.ravel(), Ys.ravel())).T  # Shape (n, 2)

def square_grid(Lx:int, Ly:int=None, bounding_box:tuple|float=1):
    '''
    Square grid filling `bounding_box` with sides `Lx` and `Ly`.
    `bounding_box` is a tuple of the form `(xmin, ymin, xmax, ymax)` bounding the points. If only one value `x`, interpret it as `(-x, -x, x, x)`.
    If `Ly` is not provided, defaults to `Lx`.
    '''
    if Ly is None : Ly = Lx
    try : xmin, ymin, xmax, ymax = bounding_box
    except : xmin, ymin, xmax, ymax = -bounding_box, -bounding_box, bounding_box, bounding_box

    Xs, Ys = np.meshgrid(np.linspace(xmin, xmax, Lx), np.linspace(ymin, ymax, Ly))
    return Grid(Xs, Ys)


path_color_default = ['red', 'green', 'blue']

def plot_2d(M, grid:Grid, endpoints:tuple|ndarray[int]=None, dtm_grid:ndarray[float]=None,
                ax:plt.Axes=None, fig_size:float=5, cm=plt.cm.magma_r, path_color=path_color_default, imshow:bool=True,
                vmin:float=None, vmax:float=None, **kwargs):
        '''
        Plots path(s) from `endpoints` in a 2d representation (data must be 2d).
        The DTM surface is plotted over `grid`.
        `endpoints` is transformed into an array of shape `(n, 2)` of endpoints.
        If provided, `dtm_grid` are the precomputed values of the DTM on the grid.
        If provided, plots are done on `ax`.
        Colorbar ranges from `vmin` to `vmax`. Defaults to min & max of the data.
        '''
        if endpoints is not None : endpoints = np.atleast_2d(endpoints)

        if dtm_grid is None : dtm_grid = M.dtm.transform(grid.points).reshape((grid.Lx, grid.Ly))

        if not isinstance(path_color, list): path_color = [path_color]

        ax_is_None = ax is None
        if ax_is_None : _, ax =  plt.subplots(figsize=(fig_size, fig_size))

        if vmin is not None or vmax is not None : dtm_grid = np.clip(dtm_grid, vmin, vmax)
        
        # Plot surface
        if imshow:
            ax.imshow(dtm_grid, extent=grid.bounding_box, origin="lower", vmin=vmin, vmax=vmax, cmap=cm,
                      aspect="equal", interpolation='bilinear')
        else:
            ax.pcolormesh(grid.Xs, grid.Ys, dtm_grid, shading='auto', cmap=cm)

        # Plot paths
        if endpoints is not None:
            for k, (i, j) in enumerate(endpoints):
                ax.plot(M.X[M.paths[i][j]][:, 0], M.X[M.paths[i][j]][:, 1], color=path_color[k%len(path_color)], zorder=10, **kwargs)
        
        ax.set_axis_off()
        if ax_is_None : plt.show()