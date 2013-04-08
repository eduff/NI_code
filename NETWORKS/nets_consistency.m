function [Znet,Tnet]=nets_consistency(netmat,gofigure,varargin);

Nsubgroup=1;
if nargin==3
  Nsubgroup=varargin{1};
end

Nf=sqrt(size(netmat,2));  N=round(Nf);  Nsub=size(netmat,1);

% one-group t-test
grot=netmat; DoF=Nsub-1;
if Nsubgroup>1
  clear grot;
  for i=1:Nsub/Nsubgroup
    grot(i,:)=mean(netmat((i-1)*Nsubgroup+1:i*Nsubgroup,:));
  end
  DoF=i-1;
end

[grotH,grotP,grotCI,grotSTATS]=ttest(grot);  Tnet=grotSTATS.tstat;  Tnet(isfinite(Tnet)==0)=0;

Znet=sign(Tnet).*(2^0.5).*erfinv(1-2.*(betainc(DoF./(DoF+abs(Tnet).^2),DoF/2,1/2)/2));
Znet(isinf(Znet)==1)=20*sign(Znet(isinf(Znet)==1));  % very large t values would otherwise be called infinite

Znetd=Znet;
if N==Nf      % is netmat square....
  Znet=reshape(Znet,N,N);
end

if gofigure>0
  figure;
  subplot(1,2,1);
  plot(Znetd);
  if N==Nf      % is netmat square....
    Znetd=reshape(Znetd,N,N);
    if sum(sum(abs(Znetd)-abs(Znetd')))<0.00000001    % .....and symmetric
      imagesc(Znetd,[-10 10]);
    end
  end
  title('one-group t-test');
  colorbar;

  % scatter plot of each session's netmat vs the mean netmat
  subplot(1,2,2); 
  grot=repmat(mean(netmat),Nsub,1);
  scatter(netmat(:),grot(:));
  title('scatter of each session''s netmat vs mean netmat');
end

