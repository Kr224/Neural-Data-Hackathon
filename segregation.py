#!/usr/bin/env python3

## test Usage:
##   python -c "from segregation import test; test()"
##   python -c "from segregation import test; test(thresh=0.05, debug=True)"

import numpy as np


def segregation(M, Ci, *varargin):
    """
    The degree to which edges are more dense within communities and more
    sparse between communities, which quantifies the segregation of a
    weighted network (Chan et al. 2014).

    Parameters:
    - M: weighted symmetrical matrix (n x n matrix)
    - Ci: community affiliation vector (n x 1 vector)
    - varargin: (Optional) 'diagzero' to convert diagonal to zero, 'negzero' to convert negative values to zero

    Returns:
    - S: System segregation calculated with W & B
    - W: Mean edge weight between nodes within the same community
    - B: Mean edge weight between nodes from different communities
    """
    # Check inputs
    if len(Ci) != M.shape[0]:
        raise ValueError('Length of label does not match with matrix dimension.')
    if len(varargin) > 2:
        raise ValueError('Too many optional inputs')
    nCi = np.unique(Ci)
    Wv = []
    Bv = []
    # Set diagonal/negatives to zero if specified
    if 'diagzero' in varargin:
        np.fill_diagonal(M, 0)
    if 'negzero' in varargin:
        M[M < 0] = 0
    for i in range(len(nCi)):  # loop through unique communities
        Wi = Ci == nCi[i]  # find index for within community edges
        Bi = Ci != nCi[i]  # find index for between community edges
        Wv_temp = M[Wi, :][:, Wi]  # extract within community edges
        Bv_temp = M[Wi, :][:, Bi]  # extract between community edges
        Wv.extend(Wv_temp[np.triu(np.ones_like(Wv_temp),k=1) == 1].flatten())
        Bv.extend(Bv_temp.flatten())
    W = np.mean(Wv)  # mean within community edges
    B = np.mean(Bv)  # mean between community edges
    S = (W - B) / W  # system segregation
    return S, W, B


def test(**kwargs):
    '''
    Test for the segregation function.
     - Loads example timeseries.csv & network_members.csv files.
     - Calculate correlation coefficients (cc_r) (default thresh=0.05)
     - Zscore correlation coefficients (cc_z)
     - Fill main diagonal with 0's
     - Apply a proportional threshold (cc_thr)
     - Calculate network segregation (S,W,B)

    Parameters:
     - varargin: (Optional)
         + thresh (float: 0>t<100): thresholds connectivity matrix by preserving a
            proportion p (0<p<1) of the strongest weights. All other weights, and all
            weights on main diagonal (self-self connections) are set to 0. (default=0.05)
         + debug (bool): controls debuging & plotting of correlation matrix before
            and after zscrore,thesholding. (default=False)
    '''
    if len(kwargs) > 2:
        raise ValueError('Too many optional inputs')
    thresh, debug = 0.05, False  ## set defaults
    for k, v in kwargs.items():
        if k == 'debug' and type(v) == type(True):
            debug = v
        elif k == 'thresh' and v > 0 and v < 100:
            thresh = v

    import bct
    from nilearn import plotting

    ## load example timeseries data
    ts = np.loadtxt('../data/test_ts/sub-001_ses-1_Schaefer200x7_196v_ts.csv', delimiter=',')

    ## load network assigments for each of 200-nodes
    Ci = np.loadtxt('../data/network_members.csv')

    if debug:
        print(' + ts.shape:', ts.shape, ts[0][:5])
        print(' + Ci.shape:', Ci.shape, type(Ci))

    ## compute correlation coefficients
    cc_r = np.corrcoef(ts, rowvar=False)
    if debug:
        print(' + cc_r.shape:', cc_r.shape, type(cc_r), cc_r[0][:5])
        plotting.plot_matrix(cc_r, title='correlation matrix', colorbar=True, vmax=0.8, vmin=-0.8)
        plotting.show()

    ## z-score correlations
    cc_z = 0.5 * (np.log(1. + cc_r) - np.log(1. - cc_r))

    ## fill main diagonal with 0's
    np.fill_diagonal(cc_z, 0)

    ## apply proportional threshold
    cc_thr = bct.threshold_proportional(cc_z, thresh)
    if debug:
        print(' + cc_thr.shape:', cc_thr.shape, type(cc_thr), cc_thr[0][:5])
        plotting.plot_matrix(cc_thr, title='correlations - zscored & thresh-prop(%.2f)' % (thresh), colorbar=True,
                             vmax=0.8, vmin=-0.8)
        plotting.show()

    ## calculate & report segragation:
    S, W, B = segregation(cc_thr, Ci)
    print(' ++ S:', S)
    print(' ++ W:', W)
    print(' ++ B:', B)
    return (S, W, B)

def calculateSS(thresh=0.05, show_plots=False, debug=False):
    import bct
    from nilearn import plotting
    from scipy.stats import ttest_ind
    ss = list()

    # Number of samples
    num_samples = 300

    # Loop through all samples
    for i in range(1, num_samples + 1):
        if(i == 33 or i == 35 or i == 117 or i == 284):
            continue
        filename = f'sub-{i:03d}_ses-1_Schaefer200x7_196v_ts.csv'
        # print(f'data/clean_healthy_ts/{filename}')

        # Load time series data from the current file
        data = np.loadtxt(f'../data/clean_healthy_ts/{filename}', delimiter=',')

        if debug:
            print(' + data.shape:', data.shape, data[0][:5])

        ## compute correlation coefficients
        cc_r = np.corrcoef(data, rowvar=False)
        if debug:
            print(' + cc_r.shape:', cc_r.shape, type(cc_r), cc_r[0][:5])

        ## z-score correlations
        cc_z = 0.5 * (np.log(1. + cc_r) - np.log(1. - cc_r))

        ## fill main diagonal with 0's
        np.fill_diagonal(cc_z, 0)

        if show_plots:
            plotting.plot_matrix(cc_z, colorbar=True, vmax=0.8, vmin=-0.8)
            plotting.show()

        ## apply proportional threshold
        cm = bct.threshold_proportional(cc_z, thresh)
        if debug:
            print(' + cm.shape:', cm.shape, type(cm), cm[0][:5])
        if show_plots:
            plotting.plot_matrix(cc_z, colorbar=True, vmax=0.8, vmin=-0.8)
            plotting.show()

        ## load network assigments for each node
        Ci = np.loadtxt('../data/network_members.csv')
        if debug:
            print(' + Ci.shape:', Ci.shape, type(Ci))

        ## calculate segragation:
        S, W, B = segregation(cm, Ci)
        # print(f'{i}: ++ S:', S)
        ss.append(S)

    return ss
