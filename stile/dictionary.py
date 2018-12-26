"""
This is a small script to solve the problem casued by new
naming system of the data columns since S17A, which is different
from S16A.

If your code is based on S16A and want to read S17A and after, you
just need to import this dictionary and specify your code is for
S16A but want to read S17A data. If your code is already new for
S17A, and want to read S16A for some reason. This can also help you.

Example:

import dictionary as dict

"""
import numpy as np

s16_tags =np.array(['object_id','parent_id','ira','idec',\
        'iflux_kron','iflux_kron_err','iflux_kron_flags',\
        'imag_kron','imag_kron_err',\
        'imag_cmodel','imag_cmodel_err','iflux_cmodel_flags',\
        'iflux_cmodel','iflux_cmodel_err','merge_measurement_i',\
	'a_g','a_r','a_i','a_z','a_y','gmag_forced_kron',\
	'gmag_forced_kron_err','gflux_forced_kron_flags','rmag_forced_kron',\
	'rmag_forced_kron_err','rflux_forced_kron_flags','imag_forced_kron',\
	'imag_forced_kron_err','iflux_forced_kron_flags','zmag_forced_kron',\
	'zmag_forced_kron_err','zflux_forced_kron_flags','ymag_forced_kron',\
	'ymag_forced_kron_err','yflux_forced_kron_flags','gmag_forced_cmodel',\
	'gmag_forced_cmodel_err','gflux_forced_cmodel','gflux_forced_cmodel_err',\
	'gflux_forced_cmodel_flags','rmag_forced_cmodel','rmag_forced_cmodel_err',\
	'rflux_forced_cmodel','rflux_forced_cmodel_err','rflux_forced_cmodel_flags',\
	'imag_forced_cmodel','imag_forced_cmodel_err','iflux_forced_cmodel',\
	'iflux_forced_cmodel_err','iflux_forced_cmodel_flags','zmag_forced_cmodel',\
	'zmag_forced_cmodel_err','zflux_forced_cmodel','zflux_forced_cmodel_err',\
	'zflux_forced_cmodel_flags','ymag_forced_cmodel','ymag_forced_cmodel_err',\
	'yflux_forced_cmodel','yflux_forced_cmodel_err','yflux_forced_cmodel_flags',\
	'ishape_hsm_regauss_e1','ishape_hsm_regauss_e2','ishape_hsm_regauss_sigma',\
	'ishape_hsm_reguass_resolution','ishape_hsm_regauss_derived_rms_e',\
	'ishape_sdss_ixx','ishape_sdss_iyy',\
	'ishape_sdss_ixy','ishape_sdss_psf_ixx','ishape_sdss_psf_iyy',\
	'ishape_sdss_psf_ixy','tract','patch','merge_peak_g','merge_peak_r',\
	'merge_peak_i','merge_peak_z','merge_peak_y','gcountinputs',\
	'rcountinputs','icountinputs','zcountinputs','ycountinputs',\
	'iflags_pixel_bright_object_center','iflags_pixel_bright_object_any',\
	'iblendedness_flags','iblendedness_abs_flux',\
	'ishape_hsm_reguass_derived_shape_weight',\
	'ishape_hsm_regauss_derived_shear_bias_m',\
	'ishape_hsm_regauss_derived_shear_bias_c1',\
	'ishape_hsm_regauss_derived_shear_bias_c2'])

snw_tags =np.array(['object_id','parent_id','i_ra','i_dec',\
        'i_kronflux_flux','i_kronflux_fluxsigma','i_kronflux_flags',\
        'i_kronflux_mag','i_kronflux_magsigma',\
        'i_cmodel_mag','i_cmodel_magsigma','i_cmodel_flags',\
        'i_cmodel_flux','i_cmodel_fluxsigma','merge_measurement_i',\
	'a_g','a_r','a_i','a_z','a_y','g_kronflux_mag',\
	'g_kronflux_magsigma','g_kronflux_flags','r_kronflux_mag',\
	'r_kronflux_magsigma','r_kronflux_flags','i_kronflux_mag',\
	'i_kronflux_magsigma','i_kronflux_flags','z_kronflux_mag',\
	'z_kronflux_magsigma','z_kronflux_flags','y_kronflux_mag',\
	'y_kronflux_magsigma','y_kronflux_flags','g_cmodel_mag',\
	'g_cmodel_magsigma','g_cmodel_flux','g_cmodel_fluxsigma',\
	'g_cmodel_flags','r_cmodel_mag','r_cmodel_magsigma',\
	'r_cmodel_flux','r_cmodel_fluxsigma','r_cmodel_flags',\
	'i_cmodel_mag','i_cmodel_magsigma','i_cmodel_flux',\
	'i_cmodel_fluxsigma','i_cmodel_flags','z_cmodel_mag',\
	'z_cmodel_magsigma','z_cmodel_flux','z_cmodel_fluxsigma',\
	'z_cmodel_flags','y_cmodel_mag','y_cmodel_magsigma',\
	'y_cmodel_flux','y_cmodel_fluxsigma','y_cmodel_flags',\
	'i_hsmshaperegauss_e1','i_hsmshaperegauss_e2','i_hsmshaperegauss_sigma',\
	'i_hsmshapereguass_resolution','i_hsmshaperegauss_derived_rms_e',\
	'i_sdssshape_shape11','i_sdssshape_shape22',\
	'i_sdssshape_shape12','i_sdssshape_psf_shape11','i_sdssshape_psf_shape22',\
	'i_sdssshape_psf_shape12','tract','patch','merge_peak_g','merge_peak_r',\
	'merge_peak_i','merge_peak_z','merge_peak_y','g_inputcount_value',\
	'r_inputcount_value','i_inputcount_value','z_inputcount_value',\
	'y_inputcount_value',\
	'i_pixelflags_bright_objectcenter','i_pixelflag_bright_object',\
	'i_blendedness_flags','i_blendedness_abs_flux',\
	'i_hsmshapereguass_derived_shape_weight',\
	'i_hsmshaperegauss_derived_shear_bias_m',\
	'i_hsmshaperegauss_derived_shear_bias_c1',\
	'i_hsmshaperegauss_derived_shear_bias_c2'])


def _crossfind(code_for,data_version,colname):
   if code_for=='S16A' and data_version=='S17A' or data_version=='S18A':
      if isinstance(colname,str):
        ix      = s16_tags==colname
        newtags = snw_tags[ix][0]
      if isinstance(colname,list):
	nx      = len(colname)
	newtags = []
	for i in range(nx):
            ix  = s16_tags==colname[i]	
	    newtags.append(snw_tags[ix][0])
	newtags = np.array(newtags)
   if code_for=='S17A' or code_for=='S18A' and data_version=='S16A':
      if isinstance(colname,str): 
        ix      = snw_tags==colname
        newtags = s16_tags[ix][0]
      if isinstance(colname,list):
	nx      = len(colname)
	newtags = []
	for i in range(nx):
            ix  = snw_tags==colname[i]	
	    newtags.append(s16_tags[ix][0])
	newtags = np.array(newtags)

   return newtags

