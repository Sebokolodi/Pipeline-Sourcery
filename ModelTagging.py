#!/usr/bin/env python

#Tags Tigger file: Specially designed for simulated data.

import Tigger
import numpy 
import pyfits
from optparse import OptionParser
import sys


def model_in_tigger(ilsm, plsm, tolerance=None):
    """Tags sources as real and artefacts
    Takes in initial LSM and sourcefinder LSM
    Tolerance in radians
    """
      
    imodel = Tigger.load(ilsm)
    pmodel = Tigger.load(plsm)
    for src in pmodel.sources:
        sources_within = imodel.getSourcesNear(src.pos.ra,
            src.pos.dec, tolerance)
        l = len(sources_within)
        if l > 0:
            src.setTag("tr", True)
        else:
            src.setTag("ar", True)
    pmodel.save(plsm)
    
 
def combine_sources(lsm, beam, outlsm):

   model = Tigger.load(lsm, verbose=False)
   tolerance = beam

   sources = model.sources

   for src in sources:
       source_max = []
 
       srcnear = model.getSourcesNear(src.pos.ra, src.pos.dec, tolerance)
       flux = 0
       err_flux = 0
       err_ra = 0
       err_dec = 0
       if srcnear == []:
           model.sources(src)
       else:
           for srs in srcnear:
               if len(srcnear) > 1:
                   srs_f = srs.flux.I
                   srs_ferr = srs.getTag("_pybdsm_E_Peak_flux")
                   flux +=  srs_f # adding the flux
                   err_flux += (srs_ferr)**2
                   source_max.append(srs_f)

           err_flux = (err_flux)**0.5
           if len(srcnear) > 1:
               ind = numpy.where(max(source_max))[0][0]
               srcs = srcnear.pop(ind)
               srcs.flux.I = flux
               srs.setAttribute("_pybdsm_E_Peak_flux", err_flux)
               for srcss in srcnear:
                   model.sources.remove(srcss)
   model.save(outlsm)

def reliability(lsm, tag="ar"):
    model = Tigger.load(lsm, verbose= False)
    sources = filter(lambda src: src.getTag("ar"), model.sources)
    rel = [src.rel for src in sources]
    return max(rel)


rad2deg = lambda r: r * 180.0/nump.pi
deg2rad = lambda d: d * numpy.pi/180.0

if __name__=='__main__':
  

    parser = OptionParser()
    parser.set_description('Tags simulated data. If true sources it gets tag 1 and etc.'
           ' Also combines sources within the beam size of the observation. To avoid multiple'
            ' sources assigned to a single component')
    parser.add_option('--im',dest='image',help='Input a restored image')
    parser.add_option('--pcat',dest='pcatalog',help='Input a pybdsm catalog')
    parser.add_option('--icat',dest='icatalog',help='Input a initial sky model catalog')
    parser.add_option('--do-cmb',dest='combine', action="store_false", default=True,
                     help='Combines sources. Default is True.') 
    parser.add_option('--high-rel', dest='high_rel', action="store_true", default=False,
                     help='Saves the highest reliability of the artefacts. Default is False')
   
    opts, args = parser.parse_args(sys.argv[1:])
    hdu = pyfits.open(opts.image)
    beam = deg2rad(hdu[0].header["BMAJ"])
    
    
    pcata = opts.pcatalog.split(",") 
    icata = opts.icatalog.split(",")

    if opts.high_rel:
        data = open(pcata[0].replace(".lsm.html", ".txt"), "w") 
    for pcat, icat in zip(pcata, icata):
        if opts.combine:
            outlsm = pcat.replace(".lsm.html", "_com.lsm.html")
            combine_sources(lsm=pcat, beam=beam, outlsm=outlsm)
            model_in_tigger(icat, outlsm, tolerance=beam)
            if opts.high_rel:
                r = reliability(outlsm)
                data.write("%.3f \n" %r)
        else:
            model_in_tigger(icat, pcat, tolerance=beam)
            if opts.high_rel:
                r = reliability(outlsm)
                data.write("%r \n" %r)
    if opts.high_rel:                    
        data.close()


