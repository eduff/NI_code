clear all
cd '/home/fs0/madugula/scratch/FC'
  s={'t' 'v' 'vt' 'vtbw'}; 
  c={'clean' 'noclean'}; 

%for j=[1 2]; 
   % for i=1:4; 
       
      
 %        load(['covarscript/dr_r' s{i} '_covar_' c{j} '.mat'])
 %         nets_glmmod2(netmat0,'15_paired.mat','15_paired.con',1,['covarscript/' c{j} 'l.txt'],['covarscript/ims/r' s{i} '_' c{j} '_cov'],['r,' s{i} ' ' c{j}]);
 %   end
%end

%200 components
%clear all
for i=1:4; 
load(['covarscript/dr_r' s{i} '_covar_clean200.mat'])
nets_glmmod2(netmat0,'15_paired.mat','15_paired.con',1,'covarscript/cleanl200.txt',['~/scratch/FC/covar/r' s{i} '_clean200_cov_cor'],['r,' s{i} ' clean200_cor'],1);
end
    
