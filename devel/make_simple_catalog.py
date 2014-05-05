import galsim
import numpy

use_noise = True
extent_degrees = 1. # Create galaxies within a box of this side length
n_galaxies_per_sq_arcmin = 20
z_min = 0.1
z_max = 2.0
z_powerlaw_slope = 2.0
z_lens = 0.2

def make_safe_shear(g):
    if g[0]>1:
        g[0] = 1
    if g[1]>1:
        g[1] = 1
    mag = numpy.sqrt(g[0]**2+g[1]**2)
    if mag>0.99999:
        g /= (mag+0.00001)
    return g

def main():
    z_offs = z_min**(z_powerlaw_slope+1)
    n_total_galaxies = int(extent_degrees**2*3600*n_galaxies_per_sq_arcmin)
    halo = galsim.NFWHalo(mass=1.E14, conc=4., redshift=z_lens)
    for i in range(n_total_galaxies):
        ra,dec = extent_degrees*numpy.random.rand(2)-0.5*extent_degrees
        z = ((z_powerlaw_slope+1)*numpy.random.random()+z_offs)**(1./(z_powerlaw_slope+1))
        if use_noise:
            g_int = make_safe_shear(numpy.random.normal(scale=0.35,size=2))
            g_int = galsim.Shear(g1=g_int[0], g2=g_int[1])
        else:
            g_int = galsim.Shear(g1=0,g2=0)
        if z>z_lens:
            g_induced = halo.getShear(galsim.PositionD(3600*ra,3600*dec),z)
#            g_induced = (min(g_induced[0],1),min(g_induced[0],1))
            g_induced = galsim.Shear(g1=g_induced[0],g2=g_induced[1])
            g_total = g_induced+g_int
        else:
            g_total = g_int
        print i, ra, dec, z, g_total.getG1(), g_total.getG2()

if __name__=='__main__':
    main()

