import numpy as np

# Path definitions

mockdir = '/share/des/sv/BCC-SVA1-WL-v3.0/'
golddir = '/share/des/disc2/y1/gold_v102/'
pzdir = '/share/des/disc2/y1/photo_z/'
pztestdir = '/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/'
cosmosiscosmodir = '/home/troxel/cosmosis/cosmosis-des-library/wl/y1prep/'
cosmosisrootdir = '/home/troxel/cosmosis/'
wcsfile = '/share/des/disc2/y1/wcs/y1a1_wcs.fits.gz'
spointsfile = 'y1a1_special_field_points.fits'
y1sysmapdir = '/share/des/disc2/y1/sysmaps/'
svsysmapdir = '/share/des/sv/systematics_maps/'
redmagicdir = '/share/des/disc2/y1/redmagicv6.4.11/'
redmapperdir = '/share/des/disc2/y1/redmapperv6.4.11/'
redmagicdirnersc = '/scratch2/scratchdirs/troxel/redmagicv6.4.11/'
redmapperdirnersc = '/scratch2/scratchdirs/troxel/redmapperv6.4.11/'
y1blacklist = '/share/des/disc2/y1/blacklist-y1.txt'
coaddtiles = '/share/des/coadd_tiles.fits'
tapebumps = '/home/troxel/destest/tape_bumps.fits'
e2edir = '/home/troxel/des-shear-pipeline-code/end-to-end/end-to-end_code/'
i3_main_nersc_dir = '/project/projectdirs/des/wl/desdata/wlpipe/im3shape_y1a1_v1/nbc/main/'


# Cosmosis source command
cosmosissource = 'source my-source'

# Tests to compute

tests = ['all']

# Dictionaries

cfg = {
  
  'lbins':20,
  'hbins':500,
  'hexbins':5,
  'footbins':100,
  'sbins':2,
  'slop':0.1,
  'tbins':8,
  'cbins':5,
  'sep':np.array([1.,400.]),
  'num_patch':126,
  'num_reg':150,
  'bs':False,
  'wt':False,
  'pzrw':False,
  'proc':16,
  'tasks':1

}

ra_lim = {
  
  's82':(-45,10),
  'sptc':(50,105),
  'sptb':(0,50),
  'spta':(-65,0)

}

dec_lim = {
  
  's82':(-3,3),
  'sptc':(-70,-35),
  'sptb':(-70,-35),
  'spta':(-70,-35)

}

error_name = {
  
  0:r'Complete failure',
  1:r'Minimizer failed',
  2:r'e$<$1e-4',
  3:r'e$>$1',
  4:r'Radius$>$20 arcsec',
  5:r'Rgpprp$>$6',
  6:r'Neg. or NaN rgpprp',
  7:r'SNR$<$1',
  8:r'Chi2 per eff. pixel$>$3',
  9:r'Normed residual $<$ -20',
  10:r'Normed residual$>$20',
  11:r'Deltau$>$10 arcsec',
  12:r'Deltav$>$10 arcsec',
  13:r'Failed FWHM of PSF or gal',
  14:r'Sextractor flag in r-band$>$0x4'

}

info_name = {
  
  0:r'Gold footprint',
  1:r'Gold bad region',
  2:r'Modest class',
  3:r'Mask fraction$>$0.75',
  4:r'Evals$>$10000',
  5:r'Sextractor r-band flag$>$0',
  6:r'Sextractor r-band flag$>$1',
  7:r'Unmasked flux fraction$>$0.75',
  8:r'SNR$<$10',
  9:r'SNR$>$10000',
  10:r'Rgpprp$<$1.1',
  11:r'Rgpprp$>$3.5',
  12:r'Radius$>$5',
  13:r'Radius$<$0.1',
  14:r'Position offset>1 arcsec',
  15:r'Chi2 per pix$<$0.5',
  16:r'Chi2 per pix$>$1.5',
  17:r'Minimum residual$<$-0.2',
  18:r'Maximum residual$>$0.2',
  19:r'PSF FWHM$>$7',
  20:r'PSF FWHM$<$0',
  21:r'Error flag$>$0'

}

pz_binning = {
  
  'skynet':(.005,1.8,201),
  'ada':(0.025,2.475,50),
  'hwe':(0.,3.7037037037,741),
  'dnf':(.005,1.985,199),
  'bpz':(.01,2.5,250)

}

# Change lookup names at own risk - will need to modify any direct references throught the code, too.

ng_col_lookup = {
  
  'coadd':'coadd_objects_id',
  'ra':'ra',
  'dec':'dec',
  'e1':'exp_e_1',
  'e2':'exp_e_2',
  'psf1':'psfrec_e_1',
  'psf2':'psfrec_e_2',
  'psffwhm':'psfrec_t',
  's1':'exp_e_sens_1',
  's2':'exp_e_sens_2',
  'cov11':'exp_e_cov_1_1',
  'cov12':'exp_e_cov_1_2',
  'cov22':'exp_e_cov_2_2',
  'radius':'exp_t',
  'tsnr':'exp_t_s2n',
  'snr':'exp_s2n_w'
  
}

psf_col_lookup = {

  'ra':'ra',
  'dec':'dec',
  'e1':'e1',
  'e2':'e2',
  'ccd':'ccdnum',
  'col':'x',
  'row':'y',
  'psf_e1':'psfex_e1',
  'psf_e2':'psfex_e2',
  'psf_size':'psfex_size',
  'size':'size',
  'flag':'flag',
  'ccdnum':'ccdnum',
  'x_mike':'x',
  'y_mike':'y',
  'mag':'mag',
  'psfex_size':'psfex_size',
  'erin_e1':'erin_e1',
  'erin_e2':'erin_e2',
  'erin_size':'erin_size',
  'desdm_e1':'desdm_e1',
  'desdm_e2':'desdm_e2',
  'desdm_size':'desdm_size'
}

i3_col_lookup = {
  
  'coadd':'coadd_objects_id',
  'ra':'ra',
  'dec':'dec',
  'e1':'e1',
  'e2':'e2',
  'psf1':'mean_psf_e1_sky',
  'psf2':'mean_psf_e2_sky',
  'hsmpsf1':'mean_hsm_psf_e1_sky',
  'hsmpsf2':'mean_hsm_psf_e2_sky',
  'psffwhm':'mean_psf_fwhm',
  'm1':'m',
  'm2':'m2',
  'c1':'c1',
  'c2':'c2',
  'w':'weight',
  'rgp':'mean_rgpp_rp',
  'snr':'snr',
  'ra_off':'ra_as',
  'dec_off':'dec_as',
  'rad':'radius',
  'bamp':'bulge_a',
  'damp':'disc_a',
  'bflux':'bulge_flux',
  'dflux':'disc_flux',
  'bfrac':'is_bulge',
  'ratflux':'flux_ratio',
  'resmin':'min_residuals',
  'resmax':'max_residuals',
  'modmin':'model_min',
  'modmax':'model_max',
  'like':'likelihood',
  'evals':'levmar_like_evals',
  'iter':'levmar_iterations',
  'flux':'mean_flux',
  'cov11':'covmat_2_2',
  'cov22':'covmat_3_3',
  'cov12':'covmat_2_3',
  'nexp':'n_exposure',
  'stamp':'stamp_size',
  'flagi':'flags_i',
  'flagr':'flags_r',
  'info':'info_flag',
  'error':'error_flag',
  'chi2pix':'chi2_pixel',
  'nrej':'nreject',
  'tile':'tilename',
  'rsnr':'round_snr',
  'maskfrac':'mean_mask_fraction',
  'fluxfrac':'mean_unmasked_flux_frac',
  'modelmu':'mean_model_edge_mu',
  'modelsig':'mean_model_edge_sigma',
  'mu':'mean_edge_mu',
  'sig':'mean_edge_sigma',
  'g':'mag_auto_g',
  'r':'mag_auto_r',
  'i':'mag_auto_i',
  'z':'mag_auto_z',
  'zp':'desdm_zp',
  'gold_flag':'gold_flag',
  'gold_mask':'gold_mask',
  'ee1':'intrinsic_e1',
  'ee2':'intrinsic_e2',
  'g1':'true_g1',
  'g2':'true_g2',
  'cosid':'cosmos_ident',
  'czp':'cosmos_photoz',
  'time':'time',
  'flag':'flag',
  'ccdnum':'ccdnum',
  'x':'x_mike',
  'y':'y_mike',
  'mag':'mag'
}

i3_epoch_col_lookup = {
  
  'coadd':'coadd_objects_id',
  'xoff':'x',
  'yoff':'y',
  'row':'orig_row',
  'col':'orig_col',
  'e1':'e1',
  'e2':'e2',
  'psf1_sky':'psf_e1_sky',
  'psf2_sky':'psf_e2_sky',
  'psf1':'psf_e1',
  'psf2':'psf_e2',
  'psffwhm':'psf_fwhm',
  'hsmpsf1':'hsm_psf_e1',
  'hsmpsf2':'hsm_psf_e2',
  'hsmpsf1_sky':'hsm_psf_e1_sky',
  'hsmpsf2_sky':'hsm_psf_e2_sky',
  'hsm_psffwhm':'hsm_psf_rho4',
  'expnum':'expnum',
  'ccd':'ccd',
  'res':'residual_stdev',
  'back':'background'
}

truth_col_lookup = {

  'coadd':'DES_id',
  'cid':'cosmos_ident',
  'zp':'cosmos_photoz',
  'info':'flags',
  'nflag':'neg_pixel_flag',
  'pflag':'psf_flag',
  'e1':'true_g1',
  'e2':'true_g2',
  'ee1':'intrinsic_e1',
  'ee2':'intrinsic_e2',
  'hlr':'hlr',
  'ra':'ra',
  'dec':'dec',
  'r':'r_mag',
  'flux':'flux',
  'nexp':'nexp',
  'psf1':'mean_psf_e1',
  'psf2':'mean_psf_e2',
  'psffwhm':'mean_psf_fwhm',
  'wcs1':'mean_wcs_e1',
  'wcs2':'mean_wcs_e2',
  'wcss':'wcs_scale',
  'wcst':'wcs_theta',
  'pixoff':'sextractor_pixel_offset'
}


gold_col_lookup = {  

  'coadd':'coadd_objects_id',
  'ra':'ra',
  'dec':'dec',
  'sm':'spread_model_i',
  'i':'mag_auto_i'

}

buzzard_col_lookup = {
  

  'coadd':'ID',
  'ra':'RA',
  'dec':'DEC',
  'e1':'EPSILON1',
  'e2':'EPSILON2',
  'zp':'ANNZ'

}

gal_col_lookup = {

  'coadd':'COADD_OBJECTS_ID',
  'ra':'RA',
  'dec':'DEC',
  'zp':'ZREDMAGIC',
  'error':'REDMAGICFLAG',
  'e1':'e1',
  'e2':'e2',
  'c1':'c1',
  'c2':'c2',
  'm':'m',
  'w':'weight',
  'info':'info_flag',
  'lum':'ZLUM',
  'mabs':'MABS'

}

redmapper_col_lookup = {

  'coadd':'MEM_MATCH_ID',
  'ra':'RA',
  'dec':'DEC',
  'zp':'Z_LAMBDA',
  'zperr':'Z_LAMBDA_E',
  'rich':'LAMBDA_CHISQ',
  'lum':'ZLUM',
  'bcglum':'BCG_ZLUM'

}


log_val = {
  
  'ra':False,
  'dec':False,
  'e1':False,
  'e2':False,
  'psf1':False,
  'psf2':False,
  'psffwhm':False,
  'm':False,
  'c1':False,
  'c2':False,
  'w':False,
  'rgp':False,
  'snr':True,
  'ra_off':False,
  'dec_off':False,
  'rad':False,
  'bamp':False,
  'damp':False,
  'bflux':False,
  'dflux':False,
  'ratflux':False,
  'resmin':False,
  'resmax':False,
  'modmin':False,
  'modmax':False,
  'like':False,
  'nlike':True,
  'evals':True,
  'iter':True,
  'flux':True,
  'cov11':False,
  'cov22':False,
  'cov12':False,
  'nexp':False,
  'stamp':False,
  'flagi':False,
  'flagr':False,
  'info':False,
  'error':False,
  'chi2pix':False,
  'nrej':False,
  'tile':False,
  'rsnr':True,
  'maskfrac':False,
  'fluxfrac':False,
  'modelmu':False,
  'modelsig':False,
  'mu':False,
  'sig':False,
  'pos':False,
  'psfpos':False,
  'invfluxfrac':True,
  'airmass':False,
  'exptime':False,
  'fwhm':False,
  'maglimit':False,
  'skybrite':False,
  'skysigma':False,

  'box_size':False,
  'coadd_run':False,
  'flags':False,
  'fofid':False,
  'mask_frac':True,
  'mcal_flags':False,
  'mcal_g_1':False,
  'mcal_g_1m_1':False,
  'mcal_g_1m_2':False,
  'mcal_g_1m_psf_1':False,
  'mcal_g_1m_psf_2':False,
  'mcal_g_1p_1':False,
  'mcal_g_1p_2':False,
  'mcal_g_1p_psf_1':False,
  'mcal_g_1p_psf_2':False,
  'mcal_g_2':False,
  'mcal_g_2m_1':False,
  'mcal_g_2m_2':False,
  'mcal_g_2m_psf_1':False,
  'mcal_g_2m_psf_2':False,
  'mcal_g_2p_1':False,
  'mcal_g_2p_2':False,
  'mcal_g_2p_psf_1':False,
  'mcal_g_2p_psf_2':False,
  'mcal_gpsf_1':False,
  'mcal_gpsf_2':False,
  'mcal_pars_1':False,
  'mcal_pars_1m_1':False,
  'mcal_pars_1m_2':False,
  'mcal_pars_1m_3':False,
  'mcal_pars_1m_4':False,
  'mcal_pars_1m_5':True,
  'mcal_pars_1m_6':True,
  'mcal_pars_1m_psf_1':False,
  'mcal_pars_1m_psf_2':False,
  'mcal_pars_1m_psf_3':False,
  'mcal_pars_1m_psf_4':False,
  'mcal_pars_1m_psf_5':True,
  'mcal_pars_1m_psf_6':True,
  'mcal_pars_1p_1':False,
  'mcal_pars_1p_2':False,
  'mcal_pars_1p_3':False,
  'mcal_pars_1p_4':False,
  'mcal_pars_1p_5':True,
  'mcal_pars_1p_6':True,
  'mcal_pars_1p_psf_1':False,
  'mcal_pars_1p_psf_2':False,
  'mcal_pars_1p_psf_3':False,
  'mcal_pars_1p_psf_4':False,
  'mcal_pars_1p_psf_5':True,
  'mcal_pars_1p_psf_6':True,
  'mcal_pars_2':False,
  'mcal_pars_2m_1':False,
  'mcal_pars_2m_2':False,
  'mcal_pars_2m_3':False,
  'mcal_pars_2m_4':False,
  'mcal_pars_2m_5':True,
  'mcal_pars_2m_6':True,
  'mcal_pars_2m_psf_1':False,
  'mcal_pars_2m_psf_2':False,
  'mcal_pars_2m_psf_3':False,
  'mcal_pars_2m_psf_4':False,
  'mcal_pars_2m_psf_5':True,
  'mcal_pars_2m_psf_6':True,
  'mcal_pars_2p_1':False,
  'mcal_pars_2p_2':False,
  'mcal_pars_2p_3':False,
  'mcal_pars_2p_4':False,
  'mcal_pars_2p_5':True,
  'mcal_pars_2p_6':True,
  'mcal_pars_2p_psf_1':False,
  'mcal_pars_2p_psf_2':False,
  'mcal_pars_2p_psf_3':False,
  'mcal_pars_2p_psf_4':False,
  'mcal_pars_2p_psf_5':True,
  'mcal_pars_2p_psf_6':True,
  'mcal_s2n_r':True,
  'mcal_s2n_r_1m':True,
  'mcal_s2n_r_1m_psf':True,
  'mcal_s2n_r_1p':True,
  'mcal_s2n_r_1p_psf':True,
  'mcal_s2n_r_2m':True,
  'mcal_s2n_r_2m_psf':True,
  'mcal_s2n_r_2p':True,
  'mcal_s2n_r_2p_psf':True,
  'mcal_T':True,
  'mcal_T_1m':True,
  'mcal_T_1m_psf':True,
  'mcal_T_1p':True,
  'mcal_T_1p_psf':True,
  'mcal_T_2m':True,
  'mcal_T_2m_psf':True,
  'mcal_T_2p':True,
  'mcal_T_2p_psf':True,
  'mcal_T_err':True,
  'mcal_T_err_1m':True,
  'mcal_T_err_1m_psf':True,
  'mcal_T_err_1p':True,
  'mcal_T_err_1p_psf':True,
  'mcal_T_err_2m':True,
  'mcal_T_err_2m_psf':True,
  'mcal_T_err_2p':True,
  'mcal_T_err_2p_psf':True,
  'mcal_T_r':True,
  'mcal_T_r_1m':True,
  'mcal_T_r_1m_psf':True,
  'mcal_T_r_1p':True,
  'mcal_T_r_1p_psf':True,
  'mcal_T_r_2m':True,
  'mcal_T_r_2m_psf':True,
  'mcal_T_r_2p':True,
  'mcal_T_r_2p_psf':True,
  'mcal_Tpsf':False,
  'nimage_tot':False,
  'nimage_use':False,
  'number':False,
  'obj_flags':False,
  'psf_flags':False,
  'psf_flux':True,
  'psf_flux_err':True,
  'psf_flux_s2n':True,
  'psf_mag':False,
  'psfrec_g_1':False,
  'psfrec_g_2':False,
  'psfrec_T':False,
  'time_last_fit':True

}

lbl = {
  
  'coadd':'Coadd Objects ID',
  'ra':'RA',
  'dec':'Dec',
  'e1':r'$e_1$',
  'e2':r'$e_2$',
  'e':r'$e$',
  'psf1':r'PSF $e_1$',
  'psf2':r'PSF $e_2$',
  'psfe':r'PSF $|e|$',
  'psffwhm':'PSF FWHM',
  'hsmpsf1':r'HSM PSF $e_1$',
  'hsmpsf2':r'HSM PSF $e_2$',
  'hsmpsfe':r'HSM PSF $|e|$',
  'm':'m',
  'c1':r'$c_1$',
  'c2':r'$c_2$',
  'w':'Weight',
  'rgp':r'$R_{gp}/R_{p}$',
  'snr':'Signal-to-Noise',
  'ra_off':'RA offset',
  'dec_off':'Dec offset',
  'rad':'Radius',
  'bamp':'Bulge Amp.',
  'damp':'Disc Amp.',
  'bflux':'Bulge Flux',
  'dflux':'Disc Flux',
  'ratflux':'Flux Ratio',
  'resmin':'Min Model Res.',
  'resmax':'Max Model Res.',
  'modmin':'Model Min.',
  'modmax':'Model Max.',
  'like':'Likelihood',
  'nlike':'Neg. Likelihood',
  'evals':'Levmar Evals',
  'iter':'Iterations',
  'flux':'Flux',
  'cov11':'Cov 11',
  'cov22':'Cov 22',
  'cov12':'Cov 12',
  'nexp':'Num. Exposures',
  'stamp':'Stamp Size',
  'flagi':'Flags i',
  'flagr':'Flags r',
  'info':'Info Flag',
  'error':'Error Flag',
  'chi2pix':r'$\chi^{2}$ pix$^{-1}$',
  'nrej':'Num. Reject Exp.',
  'tile':'Tile Name',
  'rsnr':'Round SNR',
  'maskfrac':'Mask Frac.',
  'fluxfrac':'UnMask Flux Frac.',
  'modelmu':r'Mdoel Edge $\mu$',
  'modelsig':r'Model Edge $\sigma$',
  'mu':r'$Edge \mu$',
  'sig':r'$Edge \sigma$',
  'pos':'Pos. Angle',
  'psfpos':'PSF Pos. Angle',
  'dpsf':r'PSF $e_1-$ PSF $e_2$',
  'hsmpsfpos':'HSM PSF Pos. Angle',
  'hsmdpsf':r'HSM PSF $e_1-$ PSF $e_2$',
  'invfluxfrac':'Inv. Flux Frac.',
  'airmass':'Air Mass',
  'exptime':'Exp. Time',
  'fwhm':'FWHM',
  'maglimit':'Mag. Limit',
  'skybrite':'Sky Brightness',
  'skysigma':r'Sky $\sigma$',
  'g':r'$g$ magauto',
  'r':r'$r$ magauto',
  'i':r'$i$ magauto',
  'z':r'$z$ magauto',
  'gr':r'$g-r$ magauto',
  'ri':r'$r-i$ magauto',
  'iz':r'$i-z$ magauto',
  'pz':r'redshift',
  'zp':r'redshift',
  'zmean':r'$zmean$',
  'zpeak':r'$zpeak$',
  'bfrac':r'Bulge Frac',
  'time':r'time',
  'cov11':'cov11',
  'cov22':'cov22',
  'cov12':'cov12',
  'g1':r'$g_1$',
  'g2':r'$g_2$',
  'psf_resid1':r'psfresid$_1$',
  'psf_resid2':r'psfresid$_2$',
  'flag':'flag',
  'ccdnum':'ccdnum',
  'x':r'x_mike',
  'y':'y_mike',
  'mag':'mag',
  'psfex_size':r'psfex_size',
  'erin_e1':r'erin_e1',
  'erin_e2':r'erin_e2',
  'erin_size':r'erin_size',
  'desdm_e1':r'desdm_e1',
  'desdm_e2':r'desdm_e2',
  'desdm_size':r'desdm_size'
}

map_name_y1 = {
  
  'airmass':y1sysmapdir+'Y1A1_SPT_and_S82_IMAGE_SRC_band_r_nside4096_oversamp4_AIRMASS_coaddweights_mean.fits.gz',
  'exptime':y1sysmapdir+'Y1A1_SPT_and_S82_IMAGE_SRC_band_r_nside4096_oversamp4_EXPTIME__total.fits.gz',
  'fwhm':y1sysmapdir+'Y1A1_SPT_and_S82_IMAGE_SRC_band_r_nside4096_oversamp4_FWHM_coaddweights_mean.fits.gz',
  'maglimit':y1sysmapdir+'Y1A1_SPT_and_S82_IMAGE_SRC_band_r_nside4096_oversamp4_maglimit2__.fits.gz',
  'skybrite':y1sysmapdir+'Y1A1_SPT_and_S82_IMAGE_SRC_band_r_nside4096_oversamp4_SKYBRITE_coaddweights_mean.fits.gz',
  'skysigma':y1sysmapdir+'Y1A1_SPT_and_S82_IMAGE_SRC_band_r_nside4096_oversamp4_SKYSIGMA_coaddweights_mean.fits.gz'
  
}

map_name_sv = {

  'airmass':svsysmapdir+'SVA1_IMAGE_SRC_band_r_nside4096_oversamp4_AIRMASS_coaddweights_mean.fits.gz',
  'ebv':svsysmapdir+'Planck_EBV_2048r_Q.fits',
  'exptime':svsysmapdir+'SVA1_IMAGE_SRC_band_r_nside4096_oversamp4_EXPTIME__total.fits.gz',
  'fwhm':svsysmapdir+'SVA1_IMAGE_SRC_band_r_nside4096_oversamp4_FWHM_coaddweights_mean.fits.gz',
  'maglimit':svsysmapdir+'SVA1_IMAGE_SRC_band_r_nside4096_oversamp4_maglimit__.fits.gz',
  'skybrite':svsysmapdir+'SVA1_IMAGE_SRC_band_r_nside4096_oversamp4_SKYBRITE_coaddweights_mean.fits.gz',
  'skysigma':svsysmapdir+'SVA1_IMAGE_SRC_band_r_nside4096_oversamp4_SKYSIGMA_coaddweights_mean.fits.gz'
  
}

