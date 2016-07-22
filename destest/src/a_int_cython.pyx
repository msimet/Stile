from cython_gsl cimport *

ctypedef double * double_ptr
ctypedef void * void_ptr

cdef double lm(double j, void * params) nogil:
  cdef double lmin = (<double_ptr> params)[1]
  cdef double lmax = (<double_ptr> params)[2]
  cdef double nell = (<double_ptr> params)[3]

  return lmin+j*(lmax-lmin)/nell

cdef double well(double ell, void * params) nogil:
  cdef double j = (<double_ptr> params)[0]
  cdef double lmin = (<double_ptr> params)[1]
  cdef double lmax = (<double_ptr> params)[2]
  cdef double nell = (<double_ptr> params)[3]

  cdef double sig = gsl_sf_log(lm(j+1.0,params)/lm(j,params))/2.0

  cdef double expow = gsl_sf_pow_int((gsl_sf_log(ell)-gsl_sf_log(lm(j+0.5,params)))/sig,2)/2.

  if expow > 200.:
    return 0.
  else:
    return gsl_sf_exp(-gsl_sf_pow_int((gsl_sf_log(ell)-gsl_sf_log(lm(j+0.5,params)))/sig,2)/2.)/sig/M_SQRT2/M_SQRTPI

cdef double tint0(double ell, void * params) nogil:
  cdef double j = (<double_ptr> params)[0]
  cdef double lmin = (<double_ptr> params)[1]
  cdef double lmax = (<double_ptr> params)[2]
  cdef double nell = (<double_ptr> params)[3]

  return 2.0*tmax*gsl_sf_bessel_J1(tmax*ell)-tmin*gsl_sf_bessel_J1(tmin*ell)/(tmax*tmax-tmin*tmin)

cdef double tint4(double ell, void * params) nogil:
  cdef double tmin = (<double_ptr> params)[4]
  cdef double tmax = (<double_ptr> params)[5]

  return 2.0*(((ell*ell*tmax-8.0/tmax)*gsl_sf_bessel_J1(tmax*ell)-(ell*ell*tmin-8.0/tmin)*gsl_sf_bessel_J1(tmin*ell)-8.0*ell*(gsl_sf_bessel_Jn(2,tmax*ell)-gsl_sf_bessel_Jn(2,tmin*ell)))/ell/ell)/(tmax*tmax-tmin*tmin)

cdef double A0int(double ell, void * params) nogil:
  cdef double tmin = (<double_ptr> params)[4]
  cdef double tmax = (<double_ptr> params)[5]

  return tint0(ell,params)*well(ell,params)

cdef double A4int(double ell, void * params) nogil:
  cdef double tmin = (<double_ptr> params)[4]
  cdef double tmax = (<double_ptr> params)[5]

  return tint4(ell,params)*well(ell,params)

def A0_integral(double j, double lmin, double lmax, double nell, double tmin, double tmax,double tol, double a, double b):
  cdef double alpha, result, error, expected
  cdef gsl_integration_workspace * W
  W = gsl_integration_workspace_alloc(2000)
  cdef gsl_function F
  cdef double params[6]
  cdef size_t neval

  params[0] = j
  params[1] = lmin
  params[2] = lmax
  params[3] = nell
  params[4] = tmin
  params[5] = tmax

  F.function = &A0int
  F.params = params

  gsl_integration_qag(&F, a, b, tol, tol, 2000, GSL_INTEG_GAUSS61, W, &result, &error)
  gsl_integration_workspace_free(W)

  return result

def A4_integral(double j, double lmin, double lmax, double nell, double tmin, double tmax,double tol, double a, double b):
  cdef double alpha, result, error, expected
  cdef gsl_integration_workspace * W
  W = gsl_integration_workspace_alloc(2000)
  cdef gsl_function F
  cdef double params[6]
  cdef size_t neval

  params[0] = j
  params[1] = lmin
  params[2] = lmax
  params[3] = nell
  params[4] = tmin
  params[5] = tmax

  F.function = &A4int
  F.params = params

  gsl_integration_qag(&F, a, b, tol, tol, 2000, GSL_INTEG_GAUSS61, W, &result, &error)
  gsl_integration_workspace_free(W)

  return result

def bpp_integral(double j, double lmin, double lmax, double nell, double tmin, double tmax,double tol, double a, double b):
  cdef double alpha, result, error, expected
  cdef gsl_integration_workspace * W
  W = gsl_integration_workspace_alloc(2000)
  cdef gsl_function F
  cdef double params[6]
  cdef size_t neval

  params[0] = j
  params[1] = lmin
  params[2] = lmax
  params[3] = nell
  params[4] = tmin
  params[5] = tmax

  F.function = &A0int
  F.params = params

  gsl_integration_qag(&F, a, b, tol, tol, 2000, GSL_INTEG_GAUSS61, W, &result, &error)
  gsl_integration_workspace_free(W)

  return result

def bpm_integral(double j, double lmin, double lmax, double nell, double tmin, double tmax,double tol, double a, double b):
  cdef double alpha, result, error, expected
  cdef gsl_integration_workspace * W
  W = gsl_integration_workspace_alloc(2000)
  cdef gsl_function F
  cdef double params[6]
  cdef size_t neval

  params[0] = j
  params[1] = lmin
  params[2] = lmax
  params[3] = nell
  params[4] = tmin
  params[5] = tmax

  F.function = &A4int
  F.params = params

  gsl_integration_qag(&F, a, b, tol, tol, 2000, GSL_INTEG_GAUSS61, W, &result, &error)
  gsl_integration_workspace_free(W)

  return result