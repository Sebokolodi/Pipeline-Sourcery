import Pyxis
# once Pyxis is loaded, ms,mqt, im, lsm, std,stefcal become available
import ms # ms module
import im # imager module 
import mqt # meqtree-pipeliner wrap
import stefcal # self calibration module 
from Pyxis.ModSupport import * # I will make a note whenever I use something from here
import random
import math
from pyrap.tables import table
import pyfits
import lsm
import Tigger
import numpy
import lsm


define("MS_REDO",True,"Remake MS if it already exists")
 
from simms import simms

FITS_L_AXIS, FITS_M_AXIS = "-L", "M"


def create_empty_ms(msname='$MS',observatory='$OBSERVATORY',antennas='$ANTENNAS',
           synthesis='$SYNTHESIS', integration=5, freq0='$FREQ0',dfreq='$DFREQ',nchan=None,direction='$DIRECTION',**kw):
    """ Create empty MS """

    # First lets evaluate the variables within the "$" construct.
    msname,observatory,antennas,direction,synthesis,freq0,dfreq = \
		interpolate_locals('msname observatory antennas direction synthesis freq0 dfreq')
   
    pos_type = 'casa' if os.path.isdir(antennas) else 'ascii'

    if not os.path.exists(msname) or MS_REDO:
        info('Making empty MS ...')
	nchan = NCHAN if nchan is None else 1
        simms(msname=msname,tel=observatory,freq0=freq0,dfreq=dfreq,direction=direction,nchan=nchan,
              dtime=integration,synthesis=float(synthesis),pos=antennas,pos_type=pos_type,stokes="RR RL LR LL",**kw)
        v.MS = msname # update the super global variable MS

    # creates DESTDIR if it doesn't exist
    makedir(DESTDIR)
    ms.plot_uvcov(save=II('${OUTFILE}_uvcov.png'),ms=.1)

# The parameters parsed via the string argument will form part of the doc string for create_empty_ms()
# Run pyxis help[create_empty_ms] to see how this works
document_globals(create_empty_ms,"MS OBSERVATORY ANTENNAS")


	
def compute_vis_noise (sefd):
    """Computes nominal per-visibility noise"""
    #sefd = sefd or SEFD
    tab = ms.ms()
    spwtab = ms.ms(subtable="SPECTRAL_WINDOW")
    
    anttab = ms.table(tab.getkeyword('ANTENNA'))
    N = len(anttab.getcol('DISH_DIAMETER'))
    freq0 = spwtab.getcol("CHAN_FREQ")[ms.SPWID,0]
    wavelength = 300e+6/freq0
    bw = spwtab.getcol("CHAN_WIDTH")[ms.SPWID,0]
    dt = tab.getcol("EXPOSURE",0,1)[0]
    dtf = (tab.getcol("TIME",tab.nrows()-1,1)-tab.getcol("TIME",0,1))[0]

    # close tables properly, else the calls below will hang waiting for a lock...
    tab.close()
    spwtab.close()

    info(">>> $MS freq %.2f MHz (lambda=%.2fm), bandwidth %.2g kHz, %.2fs integrations, %.2fh synthesis"%(freq0*1e-6,wavelength,bw*1e-3,dt,dtf/3600))
    noise = sefd/math.sqrt(2*N*(N-1)*bw*dt)
    info(">>> SEFD of %.2f Jy gives per-visibility noise of %.2f mJy"%(sefd,noise*1000))

    return noise


def sky_model(msname="$MS",output="$LSM",fov="$FOV",fluxrange='0.001:1',\
              num_src="$NUM_SRC",spi="$SPI",f0="$FREQ0",spectral=False):

    msname,output,fov,fluxrange,num_src,spi,f0 = interpolate_locals('msname output fov fluxrange num_src spi f0')
    f0 = float(f0.replace("GHz","e9"))
    if spectral:        
        x.sh('./modelsky.py --ms $MS --output $output --fov $fov --f $fluxrange --n $num_src --spi $spi --freq0 $f0')
    else:
        x.sh('./modelsky.py --ms $MS --output $output --fov $fov --f $fluxrange --n $num_src')



def simsky(msname="$MS", lsmname="$LSM", column="$COLUMN",
           tdlconf="$TDLCONF", tdlsec="$SIMSEC", addnoise=True,
           noise=0, sefd=0, recenter=True, options={} ,args=[],**kw):
    """ 
    Simulates visibilities into a MS.
    msname : MS name
    lsmname : LSM name
    column : Column to simulate visibilities into
    tdlconf : Meqtrees TDL configuration profiles file (required to run MeqTrees pipeliner) 
    tdlsec : Section to execute in tdlconf
    noise : Visibility noise to add to simulation.
    args, kw : extra arguments to pass to the MeqTrees pipeliner
    """	
    msname,lsmname,column,tdlsec,tdlconf = interpolate_locals('msname lsmname column tdlsec tdlconf')

    # recenter LSM if required    
    if recenter:
        x.sh('tigger-convert --recenter=$DIRECTION $lsmname $RLSM -f')
        v.LSM = RLSM
    else:
        v.LSM = lsmname

    args = ["${ms.MS_TDL} ${lsm.LSM_TDL}"] + list(args)

    options["pybeams_fits.filename_pattern"] = BEAM_PATTERN_HOL
    options['ms_sel.output_column'] = column

    if addnoise:
        sefd = sefd or SEFD
        options['noise_stddev'] = noise or compute_vis_noise(sefd)
    options.update(kw) # extra keyword args get preference
    mqt.run(TURBO_SIM,job='_tdl_job_1_simulate_MS',config=tdlconf,section=tdlsec,options=options,args=args)

document_globals(simsky,"MS LSM COLUMN SIMSEC TDLCONF TURBO_SIM")



def calibrate(msname='$MS', lsmname='$LSM',
              column='$COLUMN', do_dE=False, args=[], options={}, **kw):
    """ Calibrate MS """
    
    msname,lsmname,column,tdlconf,tdlsec = interpolate_locals('msname lsmname '
            'column tdlconf tdlsec')
    
    v.MS = msname
    v.LSM = lsmname
    args = ["${ms.MS_TDL} ${lsm.LSM_TDL}"] + list(args)
    options.update(dict(diffgain_plot_prefix=None,gain_plot_prefix=None,ifrgain_plot_prefix=None))   
    options["pybeams_fits.filename_pattern"] = BEAM_PATTERN
 
    if do_dE:
        """ add dE opts into options dict"""
        options.update(dict(diffgains=True,stefcal_reset_all=True,diffgain_plot=True))
    stefcal.stefcal(msname, options=options, args=args, **kw)

document_globals(calibrate,"MS LSM COLUMN CALSEC TDLCONF")


        
def find_sources(image='${im.RESTORED_IMAGE}',psf_image="${im.PSF_IMAGE}", thresh_isl=None, 
                 thresh_pix=None, neg_isl=None, neg_pix=None, output='$PLSM', 
                 reliability=False, config=None, outdir="$DESTDIR", gaul="$GAUL", **kw):

    image, psf_image, output, gaul, outdir = interpolate_locals('image psf_image output gaul outdir')
    thresh_pix = thresh_pix or THRESHOLDS[0]
    thresh_isl = thresh_isl or THRESHOLDS[1]
    neg_thresh_pix = neg_pix  or NEG_THRESH[0]
    neg_thresh_isl = neg_isl or NEG_THRESH[1]   
 

    if reliability:    
        x.sh("sourcery -i $image -p $psf_image -od $outdir -nisl=$neg_thresh_isl\
              -npix=$neg_thresh_pix -pisl=$thresh_isl -ppix=$thresh_pix -ps=3 -ns=1\
               -apsf -alv")
    else:
        lsm.pybdsm_search(image=image, output=output, thresh_pix=thresh_pix, thresh_isl=thresh_isl, **kw)


    
def apply_primarybeam(msname="$MS", lsmname="$PLSM", direction="$DIRECTION",
                     beams="$BEAM_PATTERN", output="$PLSM"):
    msname, lsmname, direction, beams, output = interpolate_locals('msname lsmname direction beams output')

    x.sh("./tigger-convert --app-to-int --beam-freq  ${ms.SPW_CENTRE_MHZ} \
         --beam-clip 0.0001 --center=$direction --primary-beam='$beams' --fits-l-axis=$FITS_L_AXIS\
         --fits-m-axis=$FITS_M_AXIS --pa-from-ms $msname -f $lsmname $output")


def crossmatch(image="${im.RESTORED_IMAGE}", lsmname="$LSM", plsmname="$PLSM"):
    image, lsmname, plsmname = interpolate_locals("image lsmname plsmname")
    x.sh("./crossmatch.py --im $image --icat $lsmname --pcat $plsmname")



def matching(image="${im.RESTORED_IMAGE}", plsmname="$PLSM", ilsmname="$LSM", gaul="$GAUL"):
    image, plsmname, ilsmname, gaul = interpolate_locals("image plsmname ilsmname gaul")
    x.sh("./matchsources.py --im  $image --g $gaul --s $plsmname --i $ilsmname")


def remove_sources(lsmname="$LSM"):
    lsmname = interpolate_locals("lsmname")

    model = Tigger.load(lsmname)
    zeroflux = filter(lambda a: a.flux.I ==0, model.sources)
    for s in zeroflux:
        model.sources.remove(s)
    model.save(lsmname)
             
 
def image_noise(sigma):
    """ Estimates RMS in noise in naturally weighted image """
    tab = ms.ms()
    data = tab.getcol('DATA')
    sigma_vis = sigma
    N_vis = np.product(data.shape[:2])
    return sigma_vis/math.sqrt(N_vis)


def sim_cal():
    """ Simulate and Calibrate """
   
    # Make empty MS
    create_empty_ms(synthesis=SYNTHESIS)
    # Estimate image noise
    sigma_vis = compute_vis_noise(SEFD)
    sigma_im = image_noise(sigma_vis)
    
    # Create random Sky
    sky_model(spectral=False, fluxrange="%.6f:%.6f"%(10*sigma_im,1)) # fluxes between 2*noise to 10 Jy
    # remove sources with flux 0 in the initial sky model
    remove_sources(lsmname=LSM)

    # Simulalte sky into MS
    simsky(column='DATA', sefd=420)
    # calibrate 
    calibrate(output='CORR_DATA')
    opts = {}
    if im.IMAGER == "wsclean":
        opts["mgain"] = 0.1
    # imaging
    im.make_image(dirty=True, psf=True, residual=False, restore=True, column='CORRECTED_DATA',
                  restore_lsm=False, **opts)
    
     #Run sourcefinder and make plots
    find_sources(reliability=False)
 
    # calculates intrinsic fluxes using primary beam gain
    #apply_primarybeam()
    
    # crosss match the sources
    matching()
   


def per_sim_cal(prefix="applied_soucery", nms=10):
    global MSLSM_List

    mslist = [ prefix+"%03d.MS"%x for x in range(nms) ]
    lsmlist = [ prefix+"%03d.txt"%x for x in range(nms) ]
    # create unified MS-LSM list 
    MSLSM_List = [ "%s %s"%(x,y) for x,y in zip(mslist, lsmlist)]
	
    def _run(ms_lsm="$MSLSM"):
        v.MS, v.LSM = II(ms_lsm).split()
        sim_cal()

    pper("MSLSM",_run)

