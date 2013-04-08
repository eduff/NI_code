%
% nets_hierarchy - create hierarchical clustering figure
% Steve Smith, 2012
%
% [hier_order,linkages] = nets_hierarchy(netmatL,netmatH,DD,sumpics)
%
%   netmatL is the net matrix shown below the diagonal, and drives the hierarchical clustering.
%           It is typically the Z-transformed full (normal) correlation, averaged across subjects.
%   netmatH is the net matrix shown above the diagonal, for example partial correlation.
%   DD is the list of good nodes (needed to map the current set of nodes back to originals), e.g., ts.DD
%   sumpics is the directory containing summary pictures for each component, e.g., sprintf('%s.sum',group_maps)
%   hier_order (output) is the index numbers of the good nodes, as reordered by the hierarchical clustering.

function [dpRSN,yyRSN] = nets_hierarchy(netmatL,netmatH,DD,sumpics);    %%%% hierarchical clustering figure

%grot=prctile(abs(netmatL(:)),99); sprintf('99th%% abs value below diagonal is %f',grot)
%netmatL=netmatL/grot;
%grot=prctile(abs(netmatH(:)),99); sprintf('99th%% abs value above diagonal is %f',grot)
%netmatH=netmatH/grot;

grot1=prctile(abs(netmatL(:)),99); sprintf('99th%% abs value below diagonal is %f',grot1)
grot2=prctile(abs(netmatH(:)),99); sprintf('99th%% abs value above diagonal is %f',grot2)
grot=grot1;
netmatL=netmatL/grot;
netmatH=netmatH/grot;

usenet=netmatL;
usenet(usenet<0)=0;   % zero negative entries.....seems to give nicer hierarchies

figure;  clear y;  N=size(netmatL,1);  gap=.5/(N+1);  
%H1=0.6; H2=0.8;   % old sizes for 3-slice summary pics
H1=0.73; H2=0.8;
grot=prctile(abs(usenet(:)),99); usenet=max(min(usenet/grot,1),-1)/2;
for J = 1:N, for I = 1:J-1,   y((I-1)*(N-I/2)+J-I) = 0.5 - usenet(I,J);  end; end;
  yyRSN=linkage(y,'ward');
subplot('position',[0 H2 1 1-H2]);
  [dh,dt,dpRSN]=dendrogram(yyRSN,0,'colorthreshold',.75);  set(dh,'LineWidth',3); set(gca,'ytick',[]);
subplot('position',[gap 0 1-2*gap H1-0.01]); i=dpRSN;
  grot=tril(netmatL(i,i)) + triu(netmatH(i,i)); grot=max(min(grot,0.95),-0.95);  grot(eye(length(grot))>0)=Inf;
  grotc=colormap;  grotc(end,:)=[.8 .8 .8];  colormap(grotc);  imagesc(grot,[-1 1]); axis off;
system(sprintf('slices_summary %s grot.png %s',sumpics,num2str(DD(dpRSN)-1)));
  grot=imread('grot.png');  subplot('position',[gap H1 1-2*gap H2-H1-0.035]); imagesc(grot); axis off;
set(gcf,'PaperPositionMode','auto','Position',[10 10 1000 500]); 

%print('-dpng',sprintf('hier.png'));

