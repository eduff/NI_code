#!/usr/local/EPD/bin/python
"""Module docstring.
This serves as a long usage message.
"""

# Allow python embedding
# from IPython.Shell import IPShellEmbed
# ipshell = IPShellEmbed('=1')
# ipshell()

import sys,os
import argparse
import scikits.learn.cross_val as cv
from scikits.learn import svm
from numpy import *
import cPickle as pickle
import nibabel
from subprocess import call
import matplotlib.pyplot as plt
import scipy
from numpy import *
import scipy.io
from matplotlib.pylab import *
from sasifuncs import *
from itertools import *

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

def main():
    """ Template
    """

    sys.excepthook = info

    #parser = argparse.ArgumentParser(description='Template,')
    #parser.add_argument('input', metavar='input',type=str, nargs=1,help='input images')
    #parser.add_argument('masks', metavar='mask', type=str, nargs='+',
    #                   help='mask(s) images')
    #parser.add_argument('-o', dest='output',metavar='output',type=str, default=[], nargs=1,help='output')
    #args = parser.parse_args()
    #print (args.masks)
 
    classify()

def classify(data='All_10.dr',plot=[],fig=[],sort=1):

    #reading and parsing labels
    with open("/home/fs0/madugula/scratch/FC/covarscript/fgcutshncl2.txt") as f:
        label=f.read().splitlines()
        label=array(map(int,label))
        blk=concatenate(([1],diff(label)))
        blk[blk!=0]=1
        ind=nonzero(blk)[0]

        mats={}
        results=[]
        xlabels=[]

        tasks=['r','t','v','vt','vtbw']
        netmats=['1','0','0a','2','3','4','5']
        netmatnames=['Corr','Cov','Amp','ICOV','ICOV0.1','ICOV1','ICOV10']

        for nmm in arange(len(netmats)): 
            mm=netmats[nmm]
            mats[nmm]=[]
            results.append([])

            for i in range(len(tasks)):

                tmp_task=scipy.io.loadmat(tasks[i]+'_'+data+'/out.mat')
                tmp2=tmp_task['netmat'+mm]

                mats[nmm].append((tmp2))

            subnum=mats[nmm][0].shape[0]

            titles=[]
            matsub=[]
    
        # create new feature sets

        nmm+=1
        mats[nmm]=[]
        results.append([])

        # corr+amp

        shp=mats[0][0].shape 
        nsubs=shp[0]
        nels=shp[1] 
        size=mats[0][0].size 
        diagels=diag(reshape(arange(nels),(nels**.5,nels**.5)))

        for i in range(len(tasks)):
            corrs=mats[0][i]
            for ii in range(nsubs):
                corrs[ii,diagels]=mats[2][i][ii,:]
            mats[nmm].append(corrs)
        netmatnames.append('Corr+Amp')

        # corr+ICOVs
        for ii in r_[1,arange(3,7)]:

            nmm+=1
            mats[nmm]=[]
            results.append([])

            for i in range(len(tasks)):
                mats[nmm].append(c_[mats[0][i],mats[ii][i]])

            netmatnames.append('Corr + ' + netmatnames[ii])
    
        # now, prediction

        for nmm in arange(len(mats)): 

            subnum=mats[nmm][0].shape[0]

            titles=[]
            matsub=[]

            for i in arange(len(mats[nmm])-1)+1:
                matsub.append(mats[nmm][i]-mats[nmm][0])
                
             # print("vs rest, no sub ")
            for x,y in combinations((arange(len(mats[nmm]))),2):
                training=concatenate((mats[nmm][x],mats[nmm][y]))
                targets=concatenate((0*ones((subnum)),1*ones((subnum))))
                labels=concatenate([arange(subnum),arange(subnum)])
                vec=arange(mats[nmm][x].shape[0])
                # random.shuffle(vec)
                # training=training[vec]
                # targets=targets[vec]
                clf=svm.SVC(kernel='linear')
                lolo=cv.LeaveOneLabelOut(labels)

                results[nmm].append(mean(cv.cross_val_score(clf,training,targets,cv=lolo)))
                xlabels.append(tasks[x] + ' vs ' + tasks[y] +' No sub' )
                # results=validate(clf,K,training,targets)
                # print(task[x]+" versus "+task[y]+": "+str(results[nmm][-1]))

               
            # print("Ranking")
            for x in range(len(matsub)):

                training=concatenate([matsub[x],-matsub[x]])
                targets=concatenate((0*ones((subnum)),1*ones((subnum))))
                labels=concatenate([arange(subnum),arange(subnum)])
                clf=svm.SVC(kernel='linear')
                lolo=cv.LeaveOneLabelOut(labels)

                results[nmm].append(mean(cv.cross_val_score(clf,training,targets,cv=lolo)))
                xlabels.append(tasks[x+1] + ' vs r')
                # results=validate(clf,K,training,targets)
                # print(tasks[x]+"from Rest: "+str(results))

            # print("Subtractions")
            for x,y in combinations((arange(len(matsub))),2):
                training=concatenate((matsub[x],matsub[y]))
                targets=concatenate((0*ones((subnum)),1*ones((subnum))))
                labels=concatenate([arange(subnum),arange(subnum)])
                vec=arange(mats[nmm][x].shape[0])
                # random.shuffle(vec)
                # training=training[vec]
                # targets=targets[vec]
                clf=svm.SVC(kernel='linear')
                lolo=cv.LeaveOneLabelOut(labels)

                results[nmm].append(mean(cv.cross_val_score(clf,training,targets,cv=lolo)))
                xlabels.append(tasks[x+1] + ' vs ' + tasks[y+1])
                # results=validate(clf,K,training,targets)
                # print(task[x]+" versus "+task[y]+": "+str(results[nmm][-1]))

        if plot:
            if fig == []:
                fig=plt.figure()
                
            multibar(array(results),fig,sort=sort,xlabels=xlabels,condlabels=netmatnames,title='Rest Measures',ylabel='Accuracy')

        return results

def info(type, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
    # we are in interactive mode or we don't have a tty-like
    # device, so we call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        import traceback, pdb
        traceback.print_exception(type, value, tb)
        print
        # pdb.pm() # deprecated
        pdb.post_mortem(tb)


if __name__ == "__main__":
    sys.exit(main())    


