#!/usr/bin/env python

import Tigger
import os
import pyfits
import numpy
from optparse import OptionParser
import sys


def compare_models(image, icatalog, pcatalog, R_th=None):

    """ Compares the initial simulated model to the model
     obtained from REDDSIT
    """
    model1 = Tigger.load(icatalog, verbose=False)
    model2 = Tigger.load(pcatalog, verbose=False)
    hdu = pyfits.open(image)
    hdr = hdu[0].header
    bmaj = hdr["BMAJ"]
  
    tolerance = numpy.deg2rad(bmaj)
    T, FP, FN, F = 0, 0, 0, 0

    for i, src in enumerate(model2.sources):
        ra_r = src.pos.ra #in radians
	dec_r = src.pos.dec
	Rel = src.rel
	if Rel > R_th:
	    within = model1.getSourcesNear(ra_r, dec_r, tolerance)
	    length = len(within)
	    if length > 0:
	        #src.setAttribute('real', True)
                src.setTag("t", True)
	        T += 1
	    else:
		#src.setAttribute('real', False)
                src.setTag("fp", True)
	        FP += 1
	else:		
	    within = model1.getSourcesNear(ra_r,dec_r,tolerance)
	    length = len(within)
	    if length > 0:
                src.setTag("fn", True) 
		FN += 1
          
	    else:
		F += 1
                src.setTag("f", True)

    output = pcatalog.replace('.lsm.html','.txt')
    summary_detections = open(output,'w')
    model2.save(pcatalog)
    summary_detections.write("\t\tSUMMARY\t\t\n")
    summary_detections.write('\nTotal number of detections = %d\n'%len(model2))
    summary_detections.write('True Source Detection = %d\n'%T)
    summary_detections.write('False Positive Detections = %d\n'%FP)
    summary_detections.write('False Negative Detections = %d\n'%FN)
    summary_detections.write('False Detection (artifacts) = %d\n'%F)

if __name__=='__main__':
  

    parser = OptionParser()
    parser.set_description('Compares initial model with the reddsit model.')
    parser.add_option('--im',dest='image',help='Input a restored image')
    parser.add_option('--pcat',dest='pcatalog',help='Input a pybdsm catalog')
    parser.add_option('--icat',dest='icatalog',help='Input a initial sky model catalog')
    parser.add_option('--r',dest='rel', type=float, help='Reliability threshold', default=0.0)
    opts, args = parser.parse_args(sys.argv[1:])

    pcata = opts.pcatalog.split(",") 
    icata = opts.icatalog.split(",") 
    for pcat, icat in zip(pcata, icata):
        compare_models(opts.image, icat, pcat, opts.rel)
    
