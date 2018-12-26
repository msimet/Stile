SELECT
  meas.object_id
  , meas.parent_id
  , meas.ira
  , meas.idec

  -- photometry
  -- unforced kron magnitudes
  , meas.iflux_kron
  , meas.iflux_kron_err
  , meas.iflux_kron_flags
  , meas.imag_kron
  , meas.imag_kron_err

  -- unforced CModel magnitudes
  , meas.icmodel_mag             as imag_cmodel
  , meas.icmodel_mag_err         as imag_cmodel_err
  , meas.icmodel_flux_flags      as iflux_cmodel_flags
  , meas.icmodel_flux            as iflux_cmodel
  , meas.icmodel_flux_err        as iflux_cmodel_err

  -- forced measurement
  , forced.merge_measurement_i
  , forced.a_g
  , forced.a_r
  , forced.a_i
  , forced.a_z
  , forced.a_y

  -- forced Kron magnitudes
  , forced.gmag_kron        as gmag_forced_kron
  , forced.gmag_kron_err    as gmag_forced_kron_err
  , forced.gflux_kron_flags as gflux_forced_kron_flags
  , forced.rmag_kron  	    as rmag_forced_kron
  , forced.rmag_kron_err    as rmag_forced_kron_err
  , forced.rflux_kron_flags as rflux_forced_kron_flags
  , forced.imag_kron        as imag_forced_kron
  , forced.imag_kron_err    as imag_forced_kron_err
  , forced.iflux_kron_flags as iflux_forced_kron_flags
  , forced.zmag_kron        as zmag_forced_kron
  , forced.zmag_kron_err    as zmag_forced_kron_err
  , forced.zflux_kron_flags as zflux_forced_kron_flags
  , forced.ymag_kron        as ymag_forced_kron
  , forced.ymag_kron_err    as ymag_forced_kron_err
  , forced.yflux_kron_flags as yflux_forced_kron_flags

  -- forced CModel magnitudes and fluxes
  , forced.gcmodel_mag        as gmag_forced_cmodel
  , forced.gcmodel_mag_err    as gmag_forced_cmodel_err
  , forced.gcmodel_flux       as gflux_forced_cmodel
  , forced.gcmodel_flux_err   as gflux_forced_cmodel_err
  , forced.gcmodel_flux_flags as gflux_forced_cmodel_flags

  , forced.rcmodel_mag        as rmag_forced_cmodel
  , forced.rcmodel_mag_err    as rmag_forced_cmodel_err
  , forced.rcmodel_flux       as rflux_forced_cmodel
  , forced.rcmodel_flux_err   as rflux_forced_cmodel_err
  , forced.rcmodel_flux_flags as rflux_forced_cmodel_flags

  , forced.icmodel_mag        as imag_forced_cmodel
  , forced.icmodel_mag_err    as imag_forced_cmodel_err
  , forced.icmodel_flux       as iflux_forced_cmodel
  , forced.icmodel_flux_err   as iflux_forced_cmodel_err
  , forced.icmodel_flux_flags as iflux_forced_cmodel_flags

  , forced.zcmodel_mag        as zmag_forced_cmodel
  , forced.zcmodel_mag_err    as zmag_forced_cmodel_err
  , forced.zcmodel_flux       as zflux_forced_cmodel
  , forced.zcmodel_flux_err   as zflux_forced_cmodel_err
  , forced.zcmodel_flux_flags as zflux_forced_cmodel_flags

  , forced.ycmodel_mag        as ymag_forced_cmodel
  , forced.ycmodel_mag_err    as ymag_forced_cmodel_err
  , forced.ycmodel_flux       as yflux_forced_cmodel
  , forced.ycmodel_flux_err   as yflux_forced_cmodel_err
  , forced.ycmodel_flux_flags as yflux_forced_cmodel_flags

  -- shapes
  -- hsm regauss shapes
  , meas2.ishape_hsm_regauss_e1
  , meas2.ishape_hsm_regauss_e2
  , meas2.ishape_hsm_regauss_sigma
  , meas2.ishape_hsm_regauss_resolution
  -- derived hsm regauss quantitites
  -- weight assuming rms_e = 0.365
  , 1./(meas2.ishape_hsm_regauss_sigma*meas2.ishape_hsm_regauss_sigma + 0.365*0.365) as ishape_hsm_regauss_derived_weight 
  , meas2.ishape_hsm_regauss_sigma			as ishape_hsm_regauss_derived_sigma_e
  , CAST(0.365 as FLOAT)				as ishape_hsm_regauss_derived_rms_e
  , CAST(0. as FLOAT) 					as ishape_hsm_regauss_derived_bias_m
  , CAST(0. as FLOAT) 					as ishape_hsm_regauss_derived_bias_c1
  , CAST(0. as FLOAT) 					as ishape_hsm_regauss_derived_bias_c2

  -- sdss shapes (without PSF correction)
  , meas2.ishape_sdss_11				as ishape_sdss_ixx
  , meas2.ishape_sdss_22				as ishape_sdss_iyy
  , meas2.ishape_sdss_12				as ishape_sdss_ixy
  , meas2.ishape_sdss_psf_11				as ishape_sdss_psf_ixx
  , meas2.ishape_sdss_psf_22				as ishape_sdss_psf_iyy
  , meas2.ishape_sdss_psf_12             		as ishape_sdss_psf_ixy

  -- flags
  -- weak lensing cut
  , (meas.icmodel_flux/meas.icmodel_flux_err > 10) AND (meas2.ishape_hsm_regauss_resolution > 0.3) AND (meas2.ishape_hsm_regauss_e1*meas2.ishape_hsm_regauss_e1 +  meas2.ishape_hsm_regauss_e2*meas2.ishape_hsm_regauss_e2 < 4) AND (meas2.ishape_hsm_regauss_sigma > 0.) AND (meas2.ishape_hsm_regauss_sigma < 0.4) AND (meas.icmodel_mag - forced.a_i < 24.5) as weak_lensing_flag

  -- columns which can be used for selection
  , meas.tract
  , meas.patch
  , meas.merge_peak_g
  , meas.merge_peak_r
  , meas.merge_peak_i
  , meas.merge_peak_z
  , meas.merge_peak_y
  , meas.gcountinputs
  , meas.rcountinputs
  , meas.icountinputs
  , meas.zcountinputs
  , meas.ycountinputs
  , meas.iflags_pixel_bright_object_center
  , meas.iflags_pixel_bright_object_any
  , meas.iblendedness_flags
  , meas.iblendedness_abs_flux

  FROM
  s16a_wide2.meas as meas
  LEFT JOIN s16a_wide2.meas2 as meas2 using (object_id)
  LEFT JOIN s16a_wide2.forced as forced using (object_id)
  WHERE
  -- Please uncomment to get a field you want

  -- AEGIS
   s16a_wide2.search_aegis(meas.patch_id)           AND

   -- HECTOMAP
   -- s16a_wide2.search_hectomap(meas.patch_id)        AND

   -- GAMA09H
   -- s16a_wide2.search_gama09h(meas.patch_id)         AND

   -- WIDE12H
   -- s16a_wide2.search_wide12h(meas.patch_id)         AND

   -- GAMA15H
   -- s16a_wide2.search_gama15h(meas.patch_id)         AND

   -- VVDS
   -- s16a_wide2.search_vvds(meas.patch_id)            AND

   -- XMM
   -- s16a_wide2.search_xmm(meas.patch_id)             AND

    NOT meas.ideblend_skipped                  AND
    NOT meas.iflags_badcentroid                AND
    NOT meas2.icentroid_sdss_flags             AND
    NOT meas.iflags_pixel_edge                 AND
    NOT meas.iflags_pixel_interpolated_center  AND
    NOT meas.iflags_pixel_saturated_center     AND
    NOT meas.iflags_pixel_cr_center            AND
    NOT meas.iflags_pixel_bad                  AND
    NOT meas.iflags_pixel_suspect_center       AND
    NOT meas.iflags_pixel_clipped_any          AND
    meas.idetect_is_primary                    AND
    NOT meas2.ishape_hsm_regauss_flags         AND
    meas.iclassification_extendedness != 0     AND
    -- In postgres, all numbers are comparable including NaN
    meas2.ishape_hsm_regauss_sigma != 'NaN'    
    --AND forced.wl_fulldepth_fullcolor
    ORDER BY meas.object_id
