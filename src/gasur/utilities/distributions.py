"""Standardized implementations of common distribution objects.

These distributions are specific to RFS swarms and are often mixtures.
Other distributions can be found in GNCPy.
"""
import numpy as np
import numpy.linalg as la
import numpy.random as rnd
import scipy.stats as stats
from scipy.optimize import linear_sum_assignment

import warnings
import enum


class GaussianMixture:
    """Gaussian Mixture object.

    Attributes
    ----------
    means : list
        List of Gaussian means, each is a N x 1 numpy array
    covariances : list
        List of Gaussian covariances, each is a N x N numpy array
    weights : list
        List of Gaussian weights, no automatic normalization
    """

    def __init__(self, means=None, covariances=None, weights=None):
        if means is None:
            means = []
        if covariances is None:
            covariances = []
        if weights is None:
            weights = []
        self.means = means
        self.covariances = covariances
        self.weights = weights

    def sample(self, rng=None):
        """Draw a sample from the current mixture model.

        Parameters
        ----------
        rng : numpy random generator, optional
            Random number generator to use. If none is given then the numpy
            default is used. The default is None.

        Returns
        -------
        numpy array
            randomly sampled numpy array of the same shape as the mean.
        """
        if rng is None:
            rng = rnd.default_rng()
        mix_ind = rng.choice(np.arange(len(self.means), dtype=int),
                             p=self.weights)
        x = rng.multivariate_normal(self.means[mix_ind].flatten(),
                                    self.covariances[mix_ind])
        return x.reshape(self.means[mix_ind].shape)

    def pdf(self, x):
        """Multi-variate probability density function for this mixture.

        Returns
        -------
        float
            PDF value of the state `x`.
        """
        rv = stats.multivariate_normal
        flat_x = x.flatten()
        p = 0
        for m, s, w in zip(self.means, self.covariances, self.weights):
            p += w * rv.pdf(flat_x, mean=m.flatten(), cov=s)

        return p


class StudentsTMixture:
    """Students T mixture object.

    Attributes
    ----------
    means : list
        List of students T means, each is a N x 1 numpy array
    weights : list
        List of students T weights
    scalings : list
        List of scaling matrices, each is a numpy array
    dof : float
        Degrees of freedom for the Students T distribution
    """

    def __init__(self, means=None, scalings=None, weights=None, dof=3):
        if means is None:
            means = []
        if scalings is None:
            scalings = []
        if weights is None:
            weights = []
        self.means = means
        self.scalings = scalings
        self.weights = weights
        self.dof = dof

    @property
    def covariances(self):
        """List of covariance matrices, each element is a numpy array."""
        if self.dof <= 2:
            msg = 'Degrees of freedom is {} and must be > 2'
            raise RuntimeError(msg.format(self.dof))
        scale = self.dof / (self.dof - 2)
        return [scale * x for x in self.scalings]

    def pdf(self, x):
        """Multi-variate probability density function for this mixture.

        Parameters
        ----------
        x : N x 1 numpy array
            Value to evaluate the pdf at.

        Returns
        -------
        float
            PDF value of the state `x`.
        """
        rv = stats.multivariate_t
        flat_x = x.flatten()
        p = 0
        for m, s, w in zip(self.means, self.scalings, self.weights):
            p += w * rv.pdf(flat_x, loc=m.flatten(), shape=s, df=self.dof)

        return p

    def sample(self, rng=None):
        """Multi-variate probability density function for this mixture.

        Parameters
        ----------
        rng : numpy random generator, optional
            Random number generator to use. If none is given then the numpy
            default is used. The default is None.

        Returns
        -------
        float
            PDF value of the state `x`.
        """
        if rng is None:
            rng = rnd.default_rng()

        rv = stats.multivariate_t
        rv.random_state = rng
        mix_ind = rng.choice(np.arange(len(self.means), dtype=int),
                             p=self.weights)
        if isinstance(self.dof, list):
            df = self.dof[mix_ind]
        else:
            df = self.dof
        x = rv.rvs(loc=self.means[mix_ind].flatten(),
                   shape=self.scalings[mix_ind], df=df)

        return x.reshape(self.means[mix_ind].shape)


class OSPAMethod(enum.Enum):
    """Enumeration for distance methods used in the OSPA calculation."""

    MANHATTAN = enum.auto()
    r"""Calculate the Manhattan/taxicab/:math:`L_1` distance.

    Notes
    -----
    Uses the form

    .. math::
        d(x, y) = \Sigma_i \vert x_i - y_i \vert
    """

    EUCLIDEAN = enum.auto()
    r"""Calculate the euclidean distance between two points.

    Notes
    -----
    Uses the form :math:`d(x, y) = \sqrt{(x-y)^T(x-y)}`.
    """

    HELLINGER = enum.auto()
    r"""Calculate the hellinger distance between two probability distributions.

    Notes
    -----
    It is at most 1, and for Gaussian distributions it takes the form

    .. math::
        d_H(f,g) &= 1 - \sqrt{\frac{\sqrt{\det{\left[\Sigma_x \Sigma_y\right]}}}
                              {\det{\left[0.5\Sigma\right]}}} \exp{\epsilon} \\
        \epsilon &= \frac{1}{4}(x - y)^T\Sigma^{-1}(x - y) \\
        \Sigma &= \Sigma_x + \Sigma_y
    """

    MAHALANOBIS = enum.auto()
    r"""Calculate the Mahalanobis distance between a point and a distribution.

    Notes
    -----
    Uses the form :math:`d(x, y) = \sqrt{(x-y)^T\Sigma_y^{-1}(x-y)}`.
    """

    def __str__(self):
        """Return the enum name for strings."""
        return self.name


def calculate_ospa(est_mat, true_mat, c, p, use_empty=True, core_method=None,
                   true_cov_mat=None, est_cov_mat=None):
    """Calculates the OSPA distance between the truth at all timesteps.

    Notes
    -----
    This calculates the Optimal SubPattern Assignment metric for the
    extracted states and the supplied truth point distributions. The
    calculation is based on
    :cite:`Schuhmacher2008_AConsistentMetricforPerformanceEvaluationofMultiObjectFilters`
    with much of the math defined in
    :cite:`Schuhmacher2008_ANewMetricbetweenDistributionsofPointProcesses`.
    A value is calculated for each timestep available in the data. This can
    use different distance metrics as the core distance. The default follows
    the main paper where the euclidean distance is used. Other options
    include the Hellinger distance
    (see :cite:`Nagappa2011_IncorporatingTrackUncertaintyintotheOSPAMetric`),
    or the Mahalanobis distance.

    Parameters
    ----------
    est_mat : S x T x N numpy array
        Numpy array of state dimension by number of timesteps by number of objects
        for times/objects which do not exist use a value of np.nan for all state
        dimensions (i.e. if object 1 at timestep 2 does not exist then
        :code:`est_mat[:, 2, 1] = np.nan * np.ones(state_dim)`). This
        corresonds to estimated states.
    true_mat : S x T x N numpy array
        Numpy array of state dimension by number of timesteps by number of objects
        for times/objects which do not exist use a value of np.nan for all state
        dimensions (i.e. if object 1 at timestep 2 does not exist then
        :code:`true_mat[:, 2, 1] = np.nan * np.ones(state_dim)`). This
        corresonds to the true states.
    c : float
        Distance cutoff for considering a point properly assigned. This
        influences how cardinality errors are penalized. For :math:`p = 1`
        it is the penalty given false point estimate.
    p : int
        The power of the distance term. Higher values penalize outliers
        more.
    use_empty : bool, Optional
        Flag indicating if empty values should be set to 0 or nan. The default
        of True is fine for most cases.
    core_method : :class:`OSPAMethod`, Optional
        The main distance measure to use for the localization component.
        The default value of None implies :attr:`.OSPAMethod.EUCLIDEAN`.
    true_cov_mat : S x S x T x N numpy array, Optional
        Numpy array of state dimension by state dimension by number of timesteps
        by number of objects for times/objects which do not exist use a value
        of np.nan for all state dimensions (i.e. if object 1 at timestep 2 does
        not exist then
        :code:`true_cov_mat[:, :, 2, 1] = np.nan * np.ones((state_dim, state_dim))`).
        This corresonds to the true states, the object order must be consistent
        with the truth matrix, and is only needed for core methods
        :attr:`OSPAMethod.HELLINGER`. The default value is None.
    est_cov_mat : S x S x T x N numpy array, Optional
        Numpy array of state dimension by state dimension by number of timesteps
        by number of objects for times/objects which do not exist use a value
        of np.nan for all state dimensions (i.e. if object 1 at timestep 2 does
        not exist then
        :code:`est_cov_mat[:, :, 2, 1] = np.nan * np.ones((state_dim, state_dim))`).
        This corresonds to the estimated states, the object order must be consistent
        with the estimated matrix, and is only needed for core methods
        :attr:`OSPAMethod.MAHALANOBIS`. The default value is None.

    Returns
    -------
    ospa : numpy array
        OSPA values at each timestep.
    localization : numpy array
        Localization component of the OSPA value at each timestep.
    cardinality : numpy array
        Cardinality component of the OSPA value at each timestep.
    core_method : :class:`.OSPAMethod`
        Method to use as the core distance statistic.
    c : float
        Maximum distance value used.
    p : int
        Power of the distance term used.
    distances : Ne x Nt x T numpy array
        Numpy array of distances, rows are estimated objects columns are truth.
    e_exists : Ne x T numpy array
        Bools indicating if the estimated object exists at that time.
    t_exists : Nt x T numpy array
        Bools indicating if the true object exists at that time.
    """
    # error checking on optional input arguments
    if core_method is None:
        core_method = OSPAMethod.EUCLIDEAN

    elif core_method is OSPAMethod.MAHALANOBIS and est_cov_mat is None:
        msg = 'Must give estimated covariances to calculate {:s} OSPA. Using {:s} instead'
        warnings.warn(msg.format(core_method, OSPAMethod.EUCLIDEAN))
        core_method = OSPAMethod.EUCLIDEAN

    elif core_method is OSPAMethod.HELLINGER and true_cov_mat is None:
        msg = 'Must save covariances to calculate {:s} OSPA. Using {:s} instead'
        warnings.warn(msg.format(core_method, OSPAMethod.EUCLIDEAN))
        core_method = OSPAMethod.EUCLIDEAN

    if core_method is OSPAMethod.HELLINGER:
        c = np.min([1, c]).item()

    # setup data structuers
    t_exists = np.logical_not(np.isnan(true_mat[0, :, :])).T
    e_exists = np.logical_not(np.isnan(est_mat[0, :, :])).T

    # compute distance for all permutations
    num_timesteps = true_mat.shape[1]
    nt_objs = true_mat.shape[2]
    ne_objs = est_mat.shape[2]
    distances = np.nan * np.ones((ne_objs, nt_objs, num_timesteps))
    comb = np.array(np.meshgrid(np.arange(ne_objs, dtype=int),
                                np.arange(nt_objs, dtype=int))).T.reshape(-1, 2)
    e_inds = comb[:, 0]
    t_inds = comb[:, 1]
    shape = (ne_objs, nt_objs)

    localization = np.nan * np.ones(num_timesteps)
    cardinality = np.nan * np.ones(num_timesteps)

    for tt in range(num_timesteps):
        # use proper core method
        if core_method is OSPAMethod.EUCLIDEAN:
            distances[:, :, tt] = np.sqrt(np.sum((true_mat[:, tt, t_inds]
                                                  - est_mat[:, tt, e_inds])**2,
                                                 axis=0)).reshape(shape)

        elif core_method is OSPAMethod.MANHATTAN:
            distances[:, :, tt] = np.sum(np.abs(true_mat[:, tt, t_inds]
                                                - est_mat[:, tt, e_inds]),
                                         axis=0).reshape(shape)

        elif core_method is OSPAMethod.HELLINGER:
            for row, col in zip(e_inds, t_inds):
                if not (e_exists[row, tt] and t_exists[col, tt]):
                    continue

                _x = est_mat[:, tt, row]
                _cov_x = est_cov_mat[:, :, tt, row]
                _y = true_mat[:, tt, col]
                _cov_y = true_cov_mat[:, :, tt, col]

                _cov = _cov_x + _cov_y
                diff = (_x - _y).reshape((_cov.shape[0], 1))
                epsilon = -0.25 * diff.T @ la.inv(_cov) @ diff

                distances[row, col, tt] = 1 - np.sqrt(np.sqrt(la.det(_cov_x @ _cov_y))
                                                      / la.det(0.5 * _cov)) \
                    * np.exp(epsilon)

        elif core_method is OSPAMethod.MAHALANOBIS:
            for row, col in zip(e_inds, t_inds):
                if not (e_exists[row, tt] and t_exists[col, tt]):
                    continue
                _x = est_mat[:, tt, row]
                _cov = est_cov_mat[:, :, tt, row]
                _y = true_mat[:, tt, col]
                diff = (_x - _y).reshape((_cov.shape[0], 1))
                distances[row, col, tt] = np.sqrt(diff.T @ _cov @ diff)

        else:
            warnings.warn('OSPA method {} is not implemented. SKIPPING'.format(core_method))
            core_method = None
            break

        # check for mismatch
        one_exist = np.logical_xor(e_exists[:, [tt]], t_exists[:, [tt]].T)
        empty = np.logical_and(np.logical_not(e_exists[:, [tt]]),
                               np.logical_not(t_exists[:, [tt]]).T)

        distances[one_exist, tt] = c
        if use_empty:
            distances[empty, tt] = 0
        else:
            distances[empty, tt] = np.nan

        distances[:, :, tt] = np.minimum(distances[:, :, tt], c)

        m = np.sum(e_exists[:, tt])
        n = np.sum(t_exists[:, tt])
        if n.astype(int) == 0 and m.astype(int) == 0:
            localization[tt] = 0
            cardinality[tt] = 0
            continue

        if n.astype(int) == 0 or m.astype(int) == 0:
            localization[tt] = 0
            cardinality[tt] = c
            continue

        cont_sub = distances[0:m.astype(int), 0:n.astype(int), tt]**p
        row_ind, col_ind = linear_sum_assignment(cont_sub)
        cost = cont_sub[row_ind, col_ind].sum()

        inv_max_card = 1. / np.max([n, m])
        card_diff = np.abs(n - m)
        inv_p = 1. / p
        c_p = c**p
        localization[tt] = (inv_max_card * cost)**inv_p
        cardinality[tt] = (inv_max_card * c_p * card_diff)**inv_p

    ospa = localization + cardinality

    return (ospa, localization, cardinality, core_method, c, p,
            distances, e_exists, t_exists)


def calculate_ospa2(est_mat, true_mat, c, p, win_len,
                    core_method=OSPAMethod.MANHATTAN, true_cov_mat=None,
                    est_cov_mat=None):
    """Calculates the OSPA(2) distance between the truth at all timesteps.

    Notes
    -----
    This calculates the OSPA-on-OSPA, or OSPA(2) metric as defined by
    :cite:`Beard2017_OSPA2UsingtheOSPAMetrictoEvaluateMultiTargetTrackingPerformance`
    and further explained in :cite:`Beard2020_ASolutionforLargeScaleMultiObjectTracking`.
    It can be thought of as the time averaged per track error between the true
    and estimated tracks. The inner OSPA calculation can use any suitable OSPA
    distance metric from :func:`.calculate_ospa`

    Parameters
    ----------
    est_mat : S x T x N numpy array
        Numpy array of state dimension by number of timesteps by number of objects
        for times/objects which do not exist use a value of np.nan for all state
        dimensions (i.e. if object 1 at timestep 2 does not exist then
        :code:`est_mat[:, 2, 1] = np.nan * np.ones(state_dim)`). This
        corresonds to estimated states.
    true_mat : S x T x N numpy array
        Numpy array of state dimension by number of timesteps by number of objects
        for times/objects which do not exist use a value of np.nan for all state
        dimensions (i.e. if object 1 at timestep 2 does not exist then
        :code:`true_mat[:, 2, 1] = np.nan * np.ones(state_dim)`). This
        corresonds to the true states.
    c : float
        Distance cutoff for considering a point properly assigned. This
        influences how cardinality errors are penalized. For :math:`p = 1`
        it is the penalty given false point estimate.
    p : int
        The power of the distance term. Higher values penalize outliers
        more.
    win_len : int
        Number of timesteps to average the OSPA over.
    core_method : :class:`OSPAMethod`, Optional
        The main distance measure to use for the localization component of the
        inner OSPA calculation.
        The default value of None implies :attr:`.OSPAMethod.EUCLIDEAN`.
    true_cov_mat : S x S x T x N numpy array, Optional
        Numpy array of state dimension by state dimension by number of timesteps
        by number of objects for times/objects which do not exist use a value
        of np.nan for all state dimensions (i.e. if object 1 at timestep 2 does
        not exist then
        :code:`true_cov_mat[:, :, 2, 1] = np.nan * np.ones((state_dim, state_dim))`).
        This corresonds to the true states, the object order must be consistent
        with the truth matrix, and is only needed for core methods
        :attr:`OSPAMethod.HELLINGER`. The default value is None.
    est_cov_mat : S x S x T x N numpy array, Optional
        Numpy array of state dimension by state dimension by number of timesteps
        by number of objects for times/objects which do not exist use a value
        of np.nan for all state dimensions (i.e. if object 1 at timestep 2 does
        not exist then
        :code:`est_cov_mat[:, :, 2, 1] = np.nan * np.ones((state_dim, state_dim))`).
        This corresonds to the estimated states, the object order must be consistent
        with the estimated matrix, and is only needed for core methods
        :attr:`OSPAMethod.MAHALANOBIS`. The default value is None.

    Returns
    -------
    ospa2 : numpy array
        OSPA values at each timestep.
    localization : numpy array
        Localization component of the OSPA value at each timestep.
    cardinality : numpy array
        Cardinality component of the OSPA value at each timestep.
    core_method : :class:`.OSPAMethod`
        Method to use as the core distance statistic.
    c : float
        Maximum distance value used.
    p : int
        Power of the distance term used.
    win_len : int
        Window length used.
    """
    # Note p is redundant here so set = 1
    (core_method, c, _,
     distances, e_exists, t_exists) = calculate_ospa(est_mat, true_mat, c, 1,
                                                     use_empty=False,
                                                     core_method=core_method,
                                                     true_cov_mat=true_cov_mat,
                                                     est_cov_mat=est_cov_mat)[3:9]

    num_timesteps = distances.shape[2]
    inv_p = 1. / p
    c_p = c**p

    localization = np.nan * np.ones(num_timesteps)
    cardinality = np.nan * np.ones(num_timesteps)

    for tt in range(num_timesteps):
        win_idx = np.array([ii for ii in range(max(tt - win_len + 1, 0),
                                               tt + 1)],
                           dtype=int)

        # find matrix of time averaged OSPA between tracks
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore',
                                    message='Mean of empty slice')
            track_dist = np.nanmean(distances[:, :, win_idx], axis=2)

        track_dist[np.isnan(track_dist)] = 0

        valid_rows = np.any(e_exists[:, win_idx], axis=1)
        valid_cols = np.any(t_exists[:, win_idx], axis=1)
        m = np.sum(valid_rows)
        n = np.sum(valid_cols)

        if n.astype(int) <= 0 and m.astype(int) <= 0:
            localization[tt] = 0
            cardinality[tt] = 0
            continue

        if n.astype(int) <= 0 or m.astype(int) <= 0:
            cost = 0
        else:
            inds = np.logical_and(valid_rows.reshape((valid_rows.size, 1)),
                                  valid_cols.reshape((1, valid_cols.size)))
            track_dist = (track_dist[inds]**p).reshape((m.astype(int),
                                                        n.astype(int)))
            row_ind, col_ind = linear_sum_assignment(track_dist)
            cost = track_dist[row_ind, col_ind].sum()

        max_nm = np.max([n, m])
        localization[tt] = (cost / max_nm)**inv_p
        cardinality[tt] = (c_p * np.abs(m - n) / max_nm)**inv_p

    ospa2 = localization + cardinality

    return ospa2, localization, cardinality, core_method, c, p, win_len
