%% matlab code for testing segmentation.matlab on singe 196x400 timeseries

addpath('../tools/BCT_v20190303');
addpath('../tools/system-segregation-and-graph-tools/MATLAB');

thresh=0.05;

%% load timeseries & 200-node network membership info
ts = load('../data/test_ts/sub-001_ses-1_Schaefer200x7_196v_ts.csv');
M = load('../data/network_members.csv');

%% correlation matrix
cc_r = corr(ts);
cc_r(1,1:5)

%% plot correlation matrix
h = heatmap(cc_r,'MissingDataColor','w');

%% convert to zscore
cc_z = 0.5.*[log(1.+cc_r)-log(1.-cc_r)];

%% zero main diagonal
cc_z = triu(cc_z,1) + tril(cc_z,-1);

%% apply threshold -
cc_thr = threshold_proportional(cc_z, thresh);
h = heatmap(cc_thr,'MissingDataColor','w');

%% print first few timepoints of cc_thr
cc_thr(1,1:5)

%% calculate network-segregation
[S_wei, W_wei, B_wei] = segregation(cc_thr, M);

disp(['S_wei=', num2str(S_wei)])
disp(['W_wei=', num2str(W_wei)])
disp(['B_wei=', num2str(B_wei)])

