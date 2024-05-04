close all
clear

%% -----------------------------------------------------------------------------------------------
%% Setup
%% -----------------------------------------------------------------------------------------------

%% set the main working directory path
%PROJECT_DIR = '/Users/jhash1/Dropbox/Documents/Amystartup/Teaching/hackathon_CS';
PROJECT_DIR = '/Users/carl/work/NeuroDataHackathon_2023/ds003592/hackathon_final';

%% add two main toolboxes (libraries) to matlab path:
%%
%% 1. Brain Connectivity Toolbox (BCT)
%%    https://sites.google.com/site/bctnet/
%addpath /Users/jhash1/Dropbox/macros_oa/2019_03_03_BCT
addpath(fullfile(PROJECT_DIR, 'tools', 'BCT_v20190303'));

%% 2. System segregation code
%%    https://github.com/mychan24/system-segregation-and-graph-tools/blob/master/MATLAB/segregation.m
%addpath /Users/jhash1/Dropbox/macros_oa/system-segregation-and-graph-tools/MATLAB
addpath(fullfile(PROJECT_DIR, 'tools', 'system-segregation-and-graph-tools'));

%% load subject list
%subj_table = textread(fullfile(PROJECT_DIR, 'participants_with_data_296.csv'), '%s');
subj_table = readtable(fullfile(PROJECT_DIR, 'data', 'participants_with_data_296.csv'));
subj_list = subj_table.participant_id;
subj_list(1:5)
nSubj = length(subj_list)
nSubj=nSubj;

%% select range of thresholds
thresh = 0.05:0.05:0.5;
nThresh = length(thresh)

%load this file to get node member ship to its respective network.
% This will lod the vector M, which will be used in the system segregation
% steps below.
% If you want to study the Atlas and after running the analysis plot the
% networks, you can use the provided excel sheet labeled
%Schaefer2018... This excel sheet has description of each ROI and its co-ordinates.
% You can use the ROI co-ordinates (x,y,z) in MNI space to plot the network

% M = load(fullfile(PROJECT_DIR, 'schaefer200x7_node-to-network.txt'));

AtlasTable = readtable(fullfile(PROJECT_DIR, 'data', 'Schaefer2018_200Parcels_7Networks_order_FSLMNI152_1mm_Centroid_RAS.csv'));
M = AtlasTable.network_member;
disp(' + network_members 10:15:')
M(10:15)
disp(' ++++++++++ DEBUG EARLY EXIT ++++++++++')
return


%Run loop, this will calculate value for each subject and save it in a
%matrix %see matrices labeled as output

for  i=1:nSubj; % loop over subject
    
    disp(strcat('Subject #',num2str(i)));
    
    %run loop for each sparsity threshold. Networks are created by taking a
    %certain proportion of nodes, e.g. 0.05= top 5%, 0.1= top 10%
    %this keeps the number of connections constant in each subjects so
    %that any differences observed are due to topology of network and not sue to a difference in total number of connections
    %essential for graph analysis.
    
    for k=1:nThresh % loop across threshold
        
        %% Load the files for each subject.
        %% The raw data is in one text file per subject, each line being one TS.
        %%   - first dimension is number of time points (TRs)
        %%   - second dimension is number of ROIs. For Schaefer Atlas there should be 200 ROIS
        
        b1_all = load([PROJECT_DIR '/clean_healthy_ts/' subj_list{i} '_ses-1_Schaefer200x7_195v_ts.csv']);
        
        %plot(b1_all)
        
        %this function correlates each time series with all other remaining time
        %series, one by one, to generate a R value. The R value denotes the
        %strength of synchrony between the time series.
        
        cc_r=corr(b1_all); % Basic correlation matrix, non-thresholded
        
        %plot the matrix for the subject to see connectivity matrix
        %imagesc(cc_r)
        
        %for graph analysis, you can standardise the matrix, useful for
        %group comparisons
        
        % z-score the connectivity matrix
        cc_z = 0.5.*[log(1.+cc_r)-log(1.-cc_r)];
        
        % Deal with the diagonal, everytime series will be fully correlated
        % with itself at the diagonal, so remove that
        cc_z = triu(cc_z,1) + tril(cc_z,-1);
        
        cc_thr = threshold_proportional(cc_z,thresh(k));% runs on each network (i.e. with each threshold)
        ccall_thresh_sess1(i,k,:,:)=(cc_z);
        
        % for FC analysis use unthresholded otherwise it changes
        % distribution to non-parametric
        % if using thresholded for FC,test with non-parametric stats
        
        %% for FC analysis if you are planning to use connectivity and not graph metrics
        %% Deal with the diagonals
        % cc_basic = triu(cc_r,1) + tril(cc_r,-1);
        % ccall_noz_nothresh(i,:,:) = cc_basic;
        
        [S_wei, W_wei, B_wei] = segregation(cc_thr,M);
        
        SS_all(i,k) = S_wei;
        W_all(i,k) = W_wei;
        B_all(i,k) = B_wei;
        
    end
end

%% save segregation results