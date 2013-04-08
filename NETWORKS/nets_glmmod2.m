%
% nets_glm(netmat,design matrix,contrast_matrix,view_output,txtfile_labels_vector, saveop,title,correctedon);
% do cross-subject GLM on set of netmats, giving uncorrected and corrected 1-p values

function [p_uncorrected,p_corrected,grot] = nets_glm(netmat,des,con,view,labels,saveop,titl,corron); 
XXX=size(netmat,2);
TTT=size(netmat,1);
Nf=sqrt(XXX);
N=round(Nf);
ShowSquare=0;
if (N==Nf)
  grot=reshape(mean(netmat),N,N);  
  
  if sum(sum(abs(grot-grot')))<0.00000001    % is netmat square and symmetric
    ShowSquare=1;
  end
end

addpath '/home/fs0/madugula/Documents/MATLAB' %sasi
fname=tempname;
save_avw(reshape(netmat',XXX,1,1,TTT),fname,'f',[1 1 1 1]);
system(sprintf('randomise -i %s -o %s -d %s -t %s -x -n 2000',fname,fname,des,con));

% how many contrasts were run?
[grot,ncon]=system(sprintf('imglob %s_vox_corrp_tstat*.* | wc -w',fname));
ncon=str2num(ncon);

if view==1 %figure setting sasi
    
  h=figure; 
  set(gcf,'Units','normalized')
  set(gcf,'Position',[.1 .1 .5 .4])
end

%creating labels
label=dlmread(labels); %sasi
cnt='0'; 
nam={'MedVis' 'Occ' 'LatVis' 'DMN' 'Cereb' 'SM' 'Temp' 'Exec' 'RLat' 'LLat'}; 
nams={'MV' 'O' 'LV' 'DMN' 'C' 'SM' 'T' 'E' 'RL' 'LL'}; %because rotation is a pain--sasi
for i=1:numel(label); 
    if ~strcmp(num2str(label(i)),cnt)
    namlabel{i}=nam{label(i)}; 
    namslabel{i}=nams{label(i)};
    else
        namlabel{i}=' '; 
        namslabel{i}=' '; 
    end
    cnt=num2str(label(i)); 
end
%setting ax2 tick marks
cnt=1; 

for i=1:numel(namlabel); 
    if ~strcmp(namlabel{i},' ')
        tick(cnt)=i; 
        cnt=cnt+1;
    end
     
end


for i=1:ncon
  p_uncorrected(i,:)= read_avw(sprintf('%s_vox_p_tstat%d',fname,i));
  p_corrected(i,:)=   read_avw(sprintf('%s_vox_corrp_tstat%d',fname,i));
  [grot,FDRthresh]=system(sprintf('fdr -i %s_vox_p_tstat%d -q 0.05 --oneminusp | grep -v Probability',fname,i));
  FDRthresh=str2num(FDRthresh);

%prethreshold sasi
p_uncorrected(i,:)=p_uncorrected(i,:).*(p_uncorrected(i,:)>.95); 
%p_corrected(i,:)=p_corrected(i,:).*(p_corrected(i,:)>.95);
  
    
  sprintf('contrast %d, best values: uncorrected_p=%f FWE_corrected_p=%f. \nFDR-correction-threshold=%f (to be applied to uncorrected p-values)',i,1-max(p_uncorrected(i,:)),1-max(p_corrected(i,:)),FDRthresh)
end

if exist('corron','var')
    if corron==1; 
grot1=reshape(p_corrected(1,:),N,N);
grot2=reshape(p_corrected(2,:),N,N);
grot=-grot1+grot2; %task is positive (sasi) 
    end
else

grot1=reshape(p_uncorrected(1,:),N,N);
grot2=reshape(p_uncorrected(2,:),N,N);
grot=-grot1+grot2; %task is positive (sasi) 
end

  if view==1
    
    if ShowSquare==1
      
     fac=.5;
      %olormap(redblue) %sasi
      %imagesc(grot.*(triu(grot,1)>0.95) + tril(grot));  % delete non-significant entries above the diag
      imagesc(grot.*(abs(grot)>0.9)); 
      set(gca,'Xtick',[1:1:length(label)]-fac) %sasi
      set(gca,'XtickLabel',namslabel) %sasi
      %th=rotateticklabel(gca); %sasi
      set(gca,'Ytick',[1:1:length(label)]-fac) %sasi
      set(gca,'YtickLabel',namlabel) %sasi
      ax1=gca; 
      ax2=axes('xticklabel',{},'yticklabel',{},'Color','none'); 
      linkaxes([ax1 ax2]);
      set(ax2,'XTick',tick-fac)
      set(ax2,'YTick',length(label)-fliplr(tick)+1+fac)
     
      grid on
      %set(ax2,'Layer','Top')
      %set(ax1,'Layer','Top')
      %set(th,'Layer', 'Top')
      set(gca,'GridLineStyle','-')
      set(gca,'LineWidth',2)
      
      
      
    else
        disp('whaaat')
      plot(p_uncorrected(1,:)); %becomes meaningless 
    end
    if exist('titl','var') %user customizes title-sasi
    title(titl);
    end
  end


if exist('saveop','var')
    set(gcf,'PaperPositionMode','auto'); 
saveas(gcf,saveop,'jpg')
save(saveop,'grot')

end
