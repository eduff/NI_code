
from numpy import *
import sys,os
import getopt
import re,glob

def __init__(self, msg):
    self.msg = msg

def importfsf(design):
    """ Load a FEAT design file into a dict
        design = load('design.fsf')
        Evs, contrasts and related settings are stored as numpy arrays
        Stores descriptions of all settings in 'descriptions' entry
    """

    import re
    design = file(design)
    fmri={}
    descs={}
    text=[]
    convert=createaddlist

    for line in design.readlines():

        if line!=[]:

            if line[0]=='#':

                text.append(line)

            elif re.match('set',line) != None:
                # create info 
                dictel=re.search("\((.*)\)",line)

                if dictel!=None:

                    dictel=dictel.groups()[0]
                    # descs[dictel]=text
                    
                # extract data through multiple find/replaces

                line = re.sub('^set[ ]*','',line)
                txt =  re.sub('^set[ ]*','',line)
                
                # odd one out - level2orth
                line = re.sub('(^[^c][a-z]*)\(level2orth\) (.*)','convert(fmri,\'level2orth\',\\2)',line)
                txt = re.sub('(^[^c][a-z]*)\(level2orth\) (.*)','convert(descs,\'level2orth\',text)',txt)

                # 2-D

                line = re.sub('(^[^c][a-z]*)\(([^0-9]*)([0-9].*)[\._]([0-9]*)\)[ ]*([^ ].*)','convert(\\1,\'\\2\',\\5,\\3,\\4)',line)

                txt = re.sub('(^[^c][a-z]*)\(([^0-9]*)([0-9].*)[\._]([0-9]*)\)[ ]*([^ ].*)','convert(descs,\'\\2\',text,\\3,\\4)',txt)

                # 1-D, not fmri

                line = re.sub('(^[^c].*)\(([0-9].*)\)[ ]*([^ ].*)','convert(fmri,\'\\1\',\\3,\\2)',line)
                txt = re.sub('(^[^c].*)\(([0-9].*)\)[ ]*([^ ].*)','convert(descs,\'\\1\',text,\\2)',txt)

                line = re.sub('(^confound.*)\(([0-9].*)\)[ ]*([^ ].*)','convert(fmri,\'\\1\',\\3,\\2)',line)
                txt = re.sub('(^confound.*)\(([0-9].*)\)[ ]*([^ ].*)','convert(descs,\'\\1\',text,\\2)',txt)

                oldline=line
                # 1-D, fmri

                line = re.sub('(^[^c][a-z]*)\(([^0-9]*)([0-9][^)]*)\)[ ]*([^ ].*)','convert(\\1,\'\\2\',\\4,\\3)',line)
                txt = re.sub('(^[^c][a-z]*)\(([^0-9]*)([0-9][^)]*)\)[ ]*([^ ].*)','convert(descs,\'\\2\',text,\\3)',txt)
                 
                if re.search('convert',line):

                    eval(line) 
                    eval(txt)

                else:

                    # 0-D
                    # dictel = re.sub('^([^c#].*)\(([^\)]*)\)(.*)\n','\\2',line)
                    
                    value=re.sub('(.*\) *)(.*)\n','\\2',line)

                    # for number, make float

                    try:
                        value=float(value)

                        if mod(value,1)==0:

                            value=int(value)

                    except ValueError:

                        tmp=1

                    fmri[dictel] = value
                    descs[dictel] = text
                    fmri['descriptions']=descs
                 
                text=[]

    design.close()

    return(fmri)

def exportfsf(filename,design):
    ''' Export a FEAT design file dict to a text design file
        
    '''
    fmri=design.copy()

    out = file(filename,'w')
    descs = fmri.pop('descriptions')

    for a in fmri.keys():

        dataout = fmri[a]
        descout = descs[a]

        if type(dataout) == ndarray:

            # feat_files or highres_files

            elshape = dataout.shape
            
            if len(elshape)==1:
                (sx,sy) = (elshape[0],1)
                dataout = dataout.reshape(sx,1)
                descout = descout.reshape(sx,1)
            else:
                (sx,sy) = elshape 

            for x in arange(sx):
                  
                for y in arange(sy):

                    txt = dataout[x,y] 

                    if txt == None:
                        continue
                    elif type(txt) == str:
                        txt = "\"" + txt + "\""

                    txt=string_(txt)
                    if descout[x,y]!=None:
                           out.writelines(descout[x,y])
                    else:
                           out.writelines('')



                    if ( a == 'ortho') | ( a == 'ortho_txt' ):
                        oy = y-1
                    else:
                        oy = y

                    if (a == 'feat_files') | (a == 'highres_files') | (a == 'unwarp_files_mag') | (a == 'unwarp_files') | (a == 'confoundev_files'):

                        out.write('set ' + a + '(' + string_(x+1) + ') ' + txt + '\n')

                    elif sy == 1:

                        out.write('set fmri(' + a + string_(x+1) + ') ' + txt + '\n')

                    else:

                        out.write('set fmri(' + a + string_(x+1) + '.' + string_(oy+1) + ') ' +  txt + '\n')

                    out.writelines(['\n'])

        else:

            txt = dataout 

            if txt == None:
                continue

            txt=string_(txt)
            out.writelines(descout)
            out.write('set fmri(' + a + ') ' + string_(txt) + '\n')
            out.writelines(['\n'])

    out.close()

def subev(fmri,rem):
    """ Remove EV number rem
    """

    rem=rem-1

    noevs=fmri['evs_orig'] - 1 
    prev_derivs = sum(fmri['deriv_yn'][0:rem])


    if fmri['deriv_yn'][rem] != 1:

         nrevs=fmri['evs_real'] - 1

         remr = [rem+prev_derivs]

    else:
         nrevs=fmri['evs_real'] - 2

         remr = [rem+prev_derivs, rem+prev_derivs+1]
    
    # update all ev data elements

    oesd = setdiff1d(arange(fmri['evs_orig']),[rem]) 
    resd = setdiff1d(arange(fmri['evs_real']),remr) 


    fmri['ortho'] =  fmri['ortho'][ix_(oesd,r_[0,1+oesd])]

    if fmri.has_key('con_orig'):

        fmri['con_orig'] =  fmri['con_orig'][:,oesd]
        

    if fmri.has_key('evg'):
        fmri['evg'] = fmri['evg'][:,oesd]
        fmri['descriptions']['evg'] = fmri['descriptions']['evg'][:,oesd]

    fmri['con_real'] =  fmri['con_real'][:,resd]

    fmri['evs_orig'] = noevs 
    fmri['evs_real'] = nrevs 


    # remove empty contrasts
    rem_em_cons(fmri)

    subevdesc(fmri, rem)
    subevdesc(fmri['descriptions'], rem)

def rem_em_cons(fmri):
    """ remove empty contrasts (after removal)
    """

    b=0
    ncons = fmri['ncon_real']

    while b < ncons:
        toremove=[]

        if sum(abs(fmri['con_real'][b,:])) == 0:
            subcon(fmri,b+1)
            ncons = fmri['ncon_real']
        else:
            b+=1

def subevdesc(fmri,rem):
    """ remove info and descriptions associated with ev
    """
    noevs = len(fmri['shape'])

    sd = setdiff1d(arange(noevs),[rem]) 

    for a in ['evtitle','shape', 'convolve', 'convolve_phase', 'tempfilt_yn',  'deriv_yn', 'custom', 'gammasigma', 'gammadelay','skip','off','on','stop','phase']:
        if fmri.has_key(a):
            fmri[a]=fmri[a][sd]  

def addev(fmri,bev=-1):

    noevs=fmri['evs_orig'] + 1

    if fmri['deriv_yn'][bev] != 1:

         nrevs=fmri['evs_real'] + 1

    else:

         nrevs=fmri['evs_real'] + 2

    ncon_orig = fmri['ncon_orig']
    ncon_real = fmri['ncon_real']

    for a in ['evtitle','shape', 'convolve', 'convolve_phase', 'tempfilt_yn',  'deriv_yn', 'custom', 'gammasigma', 'gammadelay','skip','off','on','stop','phase']:
        if fmri.has_key(a):
            calistdesc(fmri,a,bev,noevs)

    createaddlist(fmri,'evs_orig',noevs)
    createaddlist(fmri,'evs_real',nrevs)

    calistdesc(fmri,'ortho', (bev,bev), noevs,noevs)
    fmri['ortho'][noevs-1,:] =  r_[fmri['ortho'][bev,:-1],0] 
    fmri['ortho'][:,noevs] =  r_[fmri['ortho'][:-1,bev],0] 
    fmri['descriptions']['ortho'][noevs-1,:] =  r_[fmri['descriptions']['ortho'][bev,:-1],fmri['descriptions']['ortho'][bev,0]]  
    fmri['descriptions']['ortho'][:,noevs] =  r_[fmri['descriptions']['ortho'][:-1,bev],fmri['descriptions']['ortho'][bev,0]]  

    calistdesc(fmri,'con_orig',(0,bev), ncon_orig,noevs)
    fmri['con_orig'][:,noevs-1] =  0
    fmri['descriptions']['con_orig'][:,noevs-1] =  fmri['descriptions']['con_orig'][:,bev] 

    calistdesc(fmri,'con_real',(0,bev), ncon_real,nrevs)

    fmri['con_real'][:,nrevs-1] =  0
    fmri['descriptions']['con_real'][:,nrevs-1] =  fmri['descriptions']['con_real'][:,bev] 

    if fmri['deriv_yn'][bev] == 1:
        fmri['con_real'][:,nrevs-2] =  0
        fmri['descriptions']['con_real'][:,nrevs-2] =  fmri['descriptions']['con_real'][:,bev] 

def addcon(fmri,bcon=-1):

    ncon_orig = fmri['ncon_orig']+1
    ncon_real = fmri['ncon_real']+1
        
    noevs=fmri['evs_orig']
    nrevs=fmri['evs_real']

    createaddlist(fmri,'ncon_orig',ncon_orig)
    createaddlist(fmri,'ncon_real',ncon_real)

    for a in ['conpic_real.', 'conname_real.','conname_orig.']:
        calistdesc(fmri,a,bcon,ncon_orig)

    calistdesc(fmri,'con_real', (bcon,bcon), ncon_real , noevs)
    fmri['con_real'][ncon_real-1,:] =  0
    fmri['con_real'][ncon_real-1,-1] =  1
    fmri['descriptions']['con_real'][ncon_real-1,:] =  fmri['descriptions']['con_real'][bcon,:] 

    calistdesc(fmri,'con_orig', (bcon,bcon), ncon_orig , noevs)
    fmri['con_orig'][ncon_orig-1,:] =  0
    fmri['con_orig'][ncon_orig-1,-1] =  1
    fmri['descriptions']['con_orig'][ncon_orig-1,:] =  fmri['descriptions']['con_orig'][bcon,:] 

    if fmri['con_mode']=='orig':
        calistdesc(fmri,'conmask', (bcon,bcon), ncon_orig, noevs)
        fmri['conmask'][ncon_orig-1,:] =  0
        fmri['conmask'][ncon_orig-1,-1] =  1
        fmri['descriptions']['conmask'][ncon_orig-1,:] =  fmri['descriptions']['conmask'][bcon,:] 
    else:
        calistdesc(fmri,'conmask', (bcon,bcon), ncon_real , noevs)
        fmri['conmask'][ncon_real-1,:] =  0
        fmri['conmask'][ncon_real-1,-1] =  1
        fmri['descriptions']['conmask'][ncon_real-1,:] =  fmri['descriptions']['conmask'][bcon,:] 

def subcon(fmri,rem):

    rem=rem-1

    nocons=fmri['ncon_orig'] - 1 
    nrcons=fmri['ncon_real'] - 1 

    nrevs=fmri['evs_real'] 
    noevs=fmri['evs_orig'] 
    oconsd = setdiff1d(arange(nocons+1),[rem]) 
    rconsd = setdiff1d(arange(nrcons+1),[rem]) 


    if fmri.has_key('con_orig'):
        fmri['ncon_orig'] = nocons
        fmri['con_orig'] =  fmri['con_orig'][oconsd-1,:]
        fmri['descriptions']['con_orig'] =  fmri['descriptions']['con_orig'][oconsd-1,:]

    fmri['ncon_real'] = nrcons

    fmri['con_real'] =  fmri['con_real'][rconsd,:]
    fmri['descriptions']['con_real'] =  fmri['descriptions']['con_real'][rconsd,:]
    fmri['conmask'] =  fmri['conmask'][ix_(rconsd,arange(noevs))]
    fmri['descriptions']['conmask'] =  fmri['descriptions']['conmask'][ix_(rconsd,arange(noevs))]


    subcondesc(fmri, rem)
    subcondesc(fmri['descriptions'], rem)

def subcondesc(fmri,rem):
   
    # todo fix con orig/real, conpic
    
    nocons = len(fmri['conpic_real.'])

    sd = setdiff1d(arange(nocons),[rem]) 

    # for a in ['conpic_real.','conpic_orig.','conname_real.', 'conname_orig.']:
    for a in ['conpic_real.','conname_real.']:

        if fmri.has_key(a):
                fmri[a]=fmri[a][sd]  

def calistdesc(fmri,inputterm,bev,x=[],y=-1):

    createaddlist(fmri,inputterm,fmri[inputterm][bev],x,y)
    createaddlist(fmri['descriptions'],inputterm,fmri['descriptions'][inputterm][bev],x,y)

def createaddlist(fmri,inputterm,el,x=[],y=-1):
    """ Test
    """

    if x==[]:
        # no list
        # eval('fmri[\'' + inputterm + '\']' + '.append(el)')
        fmri[inputterm]=el

    elif y == -1:

        if fmri.has_key(inputterm):

            oldsz  = fmri[inputterm].shape
            nwsz = (max(oldsz[0],x),)
            
            if oldsz != nwsz:
                
                # if list not the

                ll = empty(nwsz,dtype=list)

                ll[0:oldsz[0]]=fmri[inputterm]

                ll[x-1,]=el

                fmri[inputterm]=ll
                
                # eval('fmri[\'' + inputterm + '\']' + '.append(ll)')

            else:

                fmri[inputterm][x-1,]=el

        else:

            nwsz = (x,)
            ll = empty(nwsz,dtype=list)
            ll[x-1,]=el

            fmri[inputterm] = ll
 
    elif y >= 0:
        
        # 2d list
        if ( inputterm == 'ortho') | ( inputterm == 'ortho_txt' ):
            y+=1

        if fmri.has_key(inputterm):

            oldsz  = fmri[inputterm].shape
            nwsz = (max(oldsz[0],x),max(oldsz[1],y))
            
            if oldsz != nwsz:
                
                # if list not te
                ll = empty(nwsz,dtype=list)

                ll[0:oldsz[0],0:oldsz[1]]=fmri[inputterm]

                ll[x-1,y-1]=el

                fmri[inputterm]=ll
                
                # eval('fmri[\'' + inputterm + '\']' + '.append(ll)')

            else:

                fmri[inputterm][x-1,y-1]=el

        else:

            nwsz = (x,y)
            ll = empty(nwsz,dtype=list)
            ll[x-1,y-1]=el

            fmri[inputterm]=ll

def splitevents(fmri, EVno, outdir=[]):
    """ splits an EV modelling multiple events into a set of evs modelling individual events
    """

    import os.path

    EVel=EVno-1

    if fmri['shape'][EVel]==3:

        # split text file

        cnt = 0
        noevs = fmri['evs_orig']

        customevs= loadtxt(fmri['custom'][EVel])

        # not sure why this is necessary..
        if len(customevs.shape)==1:
            customevs = customevs.reshape((customevs.shape[0]/3,3))

        newevs=[]
        
        lcust=3

        for a in arange(customevs.shape[0]):

            (fileout,fileoutext) = os.path.splitext(fmri['custom'][EVel])
    
            (fileoutdir,fileoutname) = os.path.split(fileout)

            if outdir != []:
                fileoutdir=outdir
                
            fileoutnew = fileoutdir+'/'+fileoutname+'_'+string_(cnt)+fileoutext
            
            savetxt(fileoutnew,customevs[a,:].reshape((1,lcust)),delimiter='    ',fmt='%f')

            addev(fmri,EVno-1)

            fmri['custom'][cnt+noevs] = fileoutnew
            fmri['evtitle'][cnt+noevs] = fileoutname

            newevs.append(cnt+noevs)

            # add contrast
            # this is not working?
        
            # addcon(fmri,1)
            # nconso = fmri['ncon_orig']
            # nconsr = fmri['ncon_real']

            # fmri['conname_orig.'][nconso-1] = fileoutname
            # fmri['con_real'][nconsr-1,EVno-1] = 1
            # fmri['con_orig'][nconso-1,EVno-1] = 1
    
            cnt+=1

        subev(fmri,EVno)

        return fmri

        #ipshell = IPShellEmbed('=1')



