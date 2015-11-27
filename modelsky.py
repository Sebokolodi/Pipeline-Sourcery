#!/usr/bin/env python
import os,sys
import numpy as np
import random
from pyrap.tables import table
from optparse import OptionParser
from pyrap.measures import measures
dm = measures()


#creating a random sky model
def random_sky(msname, output, alpha=3, num_src=500, fov=1.0, pointsrc=True, extendedsrc=False,\
               mixsrc=False, spidx=False, freq0=None, fluxRange=[0,10]):
    """num_src=100 is default it is the number of sources on your model, fov is the
    field of view of the telescope== 1deg by default, if sources are point sources (pointsrc)
    then intensity is gaussian, one can set extended sources to true or mix of both extended 
    and point sources. Extended sources are modelled by ....."""

    phs_dir = table(msname+"/FIELD").getcol('PHASE_DIR')[0,0,:]
    ra0 = phs_dir[0]*12.0/np.pi #phase center right ascension
    dec0 = phs_dir[1]*180.0/np.pi #phase center declination 
    #using them as references and fov
    #making positions of the sources 

    ra_d,dec_d,I,emaj,emin = [],[],[],[],[]
    pa = np.zeros([num_src])
    if dec0>0:
        dec_d = np.random.uniform(dec0-fov,dec0+fov,num_src)
    else:
        dec_d = np.random.uniform(dec0+fov,dec0-fov,num_src)
    for i in range(num_src):	
        if pointsrc==True:
	    I.append(fluxRange[0]+np.random.pareto(a=alpha)*fluxRange[1]) #maximum amplitude is 3Jy
	if extendedsrc==True:
	    I.append(0) # zero just for now
	if mixsrc==True:
	    I.append(0)
        emaj.append(np.random.random()*0)
        emin.append( random.random()*0)	
    
        
    ra_d = np.random.uniform(ra0-fov,ra0+fov,num_src)

    if spidx:       
        freqs = [freq0]*num_src
        first_line=['#format:name','ra_d','dec_d','i','emaj_s','emin_s','pa_d',"spi","freq0"] #converting to ASCII format

    else:
        first_line=['#format:name','ra_d','dec_d','i','emaj_s','emin_s','pa_d'] #converting to ASCII format
        

    data=open(output,'w')
   
        
    for line in first_line:
    	data.write('%s '%line)
    for i in range(num_src):
        if spidx:
            g = random.gauss(-0.7,0.7/100.0) # mean of -0.7,sigma of 0.7/100.
    	    data.write('\nS%d %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n'%(i,ra_d[i],dec_d[i],I[i],emaj[i],emin[i],pa[i],g,freqs[i])) 
        else:
    	    data.write('\nS%d %.3f %.3f %.3f %.3f %.3f %.3f\n'%(i,ra_d[i],dec_d[i],I[i],emaj[i],emin[i],pa[i])) 
    data.close()
    
    return data

if __name__=='__main__':
  

    parser = OptionParser()
    parser.set_description('Creates a random sky model for a given field of view (fov) and\
                           the fluxes are modeled using a power-law. The model is centered about the MS file provided')

    parser.add_option('--ms',dest='MS',help='Input a Measurement Set')
    parser.add_option('--output',dest='output',type=str,
           help='input a name of a text file to write in the sky model value')
    parser.add_option('--fov',dest='fov',type=float,
           help='Input a field of view e.g if fov=2 it is read as  [-1,1], thus fov of 2 degs input as 1  ',default=1.0)
    parser.add_option('--f',dest='fluxrange',help='input flux range as [0,10]',default="0:10")
    parser.add_option('--n',dest='num_srcs',type=int,help='Input number of sources',default=200)
    parser.add_option('--spi',dest='spectral_index',help='Input spectral indices for sources',default=False)
    parser.add_option('--freq0',dest='freq0',type=float,help='Optional if spi=False, but provide it spi=True,\
                      which is the central observing frequency',default=None)


    opts,args = parser.parse_args(sys.argv[1:])
    msname = opts.MS
    fluxrange = opts.fluxrange.split(":")
    output = opts.output
    fov = opts.fov
    if opts.spectral_index:
         if not opts.freq0:
              raise ValueError("Please provide the central observing frequency or set --spi=False")
     
    random_sky(msname=msname,output=output,num_src=opts.num_srcs,fov=fov,\
              fluxRange=[float(fluxrange[0]),float(fluxrange[1])],spidx=opts.spectral_index,freq0=opts.freq0)#[float(fluxrange[1]),float(fluxrange[3])])

