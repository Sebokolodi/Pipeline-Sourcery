#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import numpy as np
import Tigger 
import os, sys
#import argparse
import pylab as plt
import pyfits
from optparse import OptionParser



deg = lambda r: r * (180.0/np.pi) # radians to degrees
rad = lambda d: d * (np.pi/180.0) # degrees to rad

def beam(image):
    hdu = pyfits.open(image)
    beam_size =  hdu[0].header['BMAJ']
    return beam_size #in degrees


# estimates the noise in an image
def estimate_noise(image):

    """estimates noise in an image """

    data = pyfits.open(image)[0].data
    negative = data[data<0]
    return np.concatenate([negative,-negative]).std()


def get_names(Gaul,initial_gaul=True):

    gaul = open(Gaul,'r')
    if initial_gaul:
        for i in range(5):
            useless = gaul.readline().split()
        names = gaul.readline().split()[1:]
    else:
        names = gaul.readline().split()[1:]

    return gaul, names


def do_crossmatching(icatalog, pcatalog, Gaul, names, beam_size=None):

    """  Crossmatch sources and save to the source finder LSM and Gaul fiel.

    icatalog: The initial/simulated model.
    pcatalog: The source finder model
    Gaul: The gaul file 
    """

   # changing the initial_gaul is important """"""
    model = Tigger.load(icatalog) # recentered model
    test_model = Tigger.load(pcatalog)    

    data = Gaul
       
    dtype = ['float']*len(names)
    gaul = np.genfromtxt(data, names=names, dtype=dtype)

    tolerance = rad(beam_size) # degrees to radians

    #comparing the positions of the sources in .gaul and the recenter file
    eo = np.zeros(len(gaul), dtype=int)

    for i,src in enumerate(test_model.sources):

        ra = src.pos.ra
        dec = src.pos.dec
	within = model.getSourcesNear(ra, dec, tolerance)
	length = len(within)

	if length > 0:
            src.setAttribute('real',True)
	    eo[i] = True
	else:
   	    eo[i] = False
            src.setAttribute('real',False)

    gaul_modified = open(icatalog.replace('.lsm.html','_tagged.gaul'), 'w')

    names = names + ['Tag']
    gaul_modified.write("%s"%[ x for x in names ])
    gaul_modified.write('\n')
    test_model.save(pcatalog)
    for dt,tag in zip(gaul,eo):
        gaul_modified.write('   '.join(map(str,dt))+' %d\n'%tag)
    return model, tolerance


if __name__=='__main__':
  

    parser = OptionParser()
    parser.set_description("Cross matching simulated sky models to source finder sources")
    parser.add_option('--im', dest='image', help='A restored image')
    parser.add_option('--g',  dest='gaul', help='A gaul file obtained the from source finder.')
    parser.add_option('--s', dest='pcatalog',help='Sourcer finder source model. Tigger format.')
    parser.add_option('--i', dest='icatalog',help='Initial source model.')
    opts,args = parser.parse_args(sys.argv[1:])

    image = opts.image
    gaul = opts.gaul
    pcat = opts.pcatalog    
    icat = opts.icatalog    
    beam_size = beam(image)
    noise = estimate_noise(image)
    g, names = get_names(Gaul=gaul)
    do_crossmatching(icatalog=icat, pcatalog=pcat, Gaul=gaul, names=names, beam_size=beam_size)
