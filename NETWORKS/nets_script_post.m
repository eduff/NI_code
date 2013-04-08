function results = nets_process(input,outputfile)

%%% function results = nets_process(group_maps,ts_dir,goodnodes,output)
%%% FSLNets - simple network matrix estimation and applications
%%% Stephen Smith, FMRIB Analysis Group
%%% Copyright (C) 2012 University of Oxford
%%% See documentation at  www.fmrib.ox.ac.uk/fsl


%%% change the following paths according to your local setup
addpath /home/fs0/steve/NETWORKS/FSLNets                 % wherever you've put this package
addpath /home/fs0/steve/matlab/L1precision            % L1precision toolbox
addpath /home/fs0/eduff/code/FSL/NETWORKS                 % pairwise causality toolbox
addpath(sprintf('%s/etc/matlab',getenv('FSLDIR')))    % you don't need to edit this if FSL is setup already

%%% setup the names of the directories containing your group-ICA and dualreg outputs
% group_maps='~/FC/analysis/group_analysis/groupmelodic_clean_100.ica/melodic_IC';     % spatial maps 4D NIFTI file, e.g. from group-ICA
   %%% you must have already run the following (outside MATLAB), to create summary pictures of the maps in the NIFTI file:
   %%% slices_summary <group_maps> 4 $FSLDIR/data/standard/MNI152_T1_2mm <group_maps>.sum
% ts_dir='~/FC/analysis/group_analysis/dualreg/v_100_clean.dr';    % dual regression output directory, containing all subjects' timeseries
load(input)
cnt=1;

p_corrected={};
p_uncorrected={};

lda_percentages=zeros(8,7);

for a = {'0','0a','1','2','3','4','5'};

    eval(strcat('netmat=netmat',a{1},';'));  
    [grotH,grotP,grotCI,grotSTATS]=ttest(netmat);  netmat(:,abs(grotSTATS.tstat)<8)=0;
    [p_uncorrected{cnt},p_corrected{cnt}]=nets_glm(netmat,'15_paired.mat','15_paired.con',0);

    [lda_percentages(cnt,:)]=nets_lda(netmat,15);
    cnt=cnt+1;

end


%%% simple cross-subject multivariate discriminant analyses, for just two-group cases.
%%% arg1 is whichever netmat you want to test.
%%% arg2 is the size of first group of subjects; set to 0 if you have two groups with paired subjects.
%%% outputs are: FLD, FLDmean(no covar), T, Tmax, Tthresh4, T/std, linear-SVM


save(outputfile)

%%% create boxplots for the two groups for a network-matrix-element of interest (e.g., selected from GLM output)
%%% arg3 = matrix row number,    i.e. the first  component of interest (from the DD list)
%%% arg4 = matrix column number, i.e. the second component of interest (from the DD list)
%%% arg5 = size of the first group (set to -1 for paired groups)
% nets_boxplots(ts,netmat3,1,7,36);
%print('-depsc',sprintf('boxplot-%d-%d.eps',IC1,IC2));  % example syntax for printing to file


