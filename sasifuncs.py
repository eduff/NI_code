from numpy import *
import os, sys
from scikits.learn.cross_val import KFold
import matplotlib.pyplot as plt
import colorsys

def validate(svmobj,numf,training,targets,labels):

	loo=KFold(len(target))
	results=[]
	for train,test in loo:
		svmobj.fit(training[train],targets[train])
		results.append(abs(svmobj.predict(training[test])-targets[test]))

	a=0
	for i in range(len(results)):
		a=sum(results[i]==0)+a
	a=a*1.
	print a
	total=a/(len(targets)*1.)
	return total

def normali(inmat):
	outmat=zeros(inmat.shape)
	for i in range(inmat.shape[0]):
		outmat[i,:]=inmat[i,:]*1./(inmat[i,:].max()*1.)
	return outmat		

def sub2ind(matsiz,ind,*args):
	if not (not args) and args[0]==1:
		if any([matsiz[a]<(max(ind[a])+1) for a in [0,1]]):
			return "Error"
		out=[]
		for i in range(len(ind[0])):
			for j in range(len(ind[1])):
				out.append(ind[1][i]*matsiz[0]+ind[0][j])
	else:		
		if not isinstance(ind[0],int):
			if len(ind[0]) != len(ind[1]):
				return "Error"
			out=[]
			for i in range(len(ind[0])):
				if any([matsiz[a]<(ind[a][i]+1) for a in [0,1]]):
					return "Error"
				out.append(ind[1][i]*matsiz[0]+ind[0][i])
		else:
			if any([(a<b) for a,b in zip(matsiz,ind)]):
				return "Error"
			out=[ind[1]*matsiz[0]+ind[0]]


	return array(out)		

def multibar(data,fig,stds=[],condlabels=[],sort=[],ylabel='',xlabels='',title='',showvals=[]):

    fig.clf()

    if len(data.shape)==3:
        data=mean(data,2)
        stds=var(d)

    (ncnds,npts)=data.shape
    rects=[]
    legels=[]

    if sort:
        order = argsort(-abs(mean(data,1)))
        data=data[order,:]
        condlabels=array(condlabels)[order]

    inds=arange(npts)
    width=1/(ncnds+1.)
    
    HSV_tuples = [(x*1.0/ncnds, 0.5, 0.5) for x in range(ncnds)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)

    ax = fig.add_subplot(111)

    for cnd in arange(ncnds):

        if stds == []:
            rects.append(ax.bar(inds+width*cnd, data[cnd,:], width, color=RGB_tuples[cnd]))
        else:
            rects.append(ax.bar(inds+width*cnd, data[cnd,:], width, color=RGB_tuples[cnd],yerr=stds[cnd,:]))

        legels.append(rects[-1][0])

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.xticks(inds+.4,xlabels, rotation=34)

    ax.legend( tuple(legels), tuple(condlabels),loc=4)

    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            print(height)
            ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%double(height),
                    ha='center', va='bottom')

    if showvals != []:
        for rect in rects:
            autolabel(rect)

