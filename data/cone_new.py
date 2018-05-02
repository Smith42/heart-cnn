from pyfits import *
import pydicom
import numpy as np
from numpy import max,sum
from scipy import ndimage

global healthy
healthy = False

global healthy_thresh
healthy_thresh = 0.1

global height
height= 20.

global noise
noise = (230.,32.)

global b_noise
b_noise = (0.,128.)

example_cube_path = "/data/jim/Heart/PACIFIC/SPECT4/RESTRECONFBPNONAC/1.3.6.1.4.1.5962.99.1.3001709873.1101971777.1428930884913.92.0.dcm"
in_image = pydicom.dcmread(example_cube_path).pixel_array
mask = np.where(np.equal(in_image,0),0,1)


def z(x,y,a=0.1,b=0.1,c=0.5,d=5,dx=0,dy=0):
    return ((c*c*x*x)/(a*a) + (c*c*y*y)/(b*b))**0.5 + d

def return_sim(p,mask=None,case='infarction'):
    a,b,c,d,sm = p
    dz,dx,dy = np.shape(in_image)

    xs,ys = np.mgrid[0:dx,0:dy]

    xs -= int(dx/2.0)
    ys -= int(dy/2.0)
    #zs-=(dz/2)

    #rest
    ddx_rest = np.random.uniform(-2,2)
    ddy_rest = np.random.uniform(-2,2)
    ddz_rest = 0 #random.uniform(-2,2)
    xs_rest=xs+int(ddx_rest)
    ys_rest=ys+int(ddy_rest)
    zz_rest = z(xs_rest,ys_rest,a=a,b=a/b,c=c,d=d+ddz_rest)

    #stress
    ddx_stress = np.random.uniform(-2,2)
    ddy_stress = np.random.uniform(-2,2)
    ddz_stress = 0 #random.uniform(-2,2)
    xs_stress=xs+int(ddx_stress)
    ys_stress=ys+int(ddy_stress)
    zz_stress = z(xs_stress,ys_stress,a=a,b=a/b,c=c,d=d+ddz_stress)

    h0 = 0
    rest_cube = []
    stress_cube = []

    d = 2

    for i in range(dz):
        v = max([255,np.random.normal(noise[0],noise[1])])
        if i > height:
            i = height
            # Bug was here -- no np.greater modifier
            layer_rest = np.where(np.greater(zz_rest, i-d) & np.less(zz_rest, i+d), v, 0.0)  #-h0), v, 0.0)
            layer_stress = np.where(np.greater(zz_stress, i-d) & np.less(zz_stress, i+d), v, 0.0)  #-h0), v, 0.0)
            h0 += 1
        else:
            layer_rest = np.where(np.greater(zz_rest, i-d) & np.less(zz_rest, i+d), v, 0.0)
            layer_stress = np.where(np.greater(zz_stress, i-d) & np.less(zz_stress, i+d), v, 0.0)

        rest_cube.append(layer_rest)
        stress_cube.append(layer_stress)


    rest_cube = np.array(rest_cube)
    stress_cube = np.array(stress_cube)

    shp = stress_cube.shape
    xs_o,ys_o,zs_o = np.mgrid[0:shp[0],0:shp[1],0:shp[2]]

    x_r = np.ravel(xs_o)
    y_r = np.ravel(ys_o)
    z_r = np.ravel(zs_o)

    cube_r = np.ravel(rest_cube)
    cube_copy = np.ones_like(rest_cube)
    indx = np.where(np.greater(cube_r,0) & np.greater(x_r,d) & np.less(x_r,height+d))[0]

    injected=False
    while not injected:
        j = np.random.choice(indx)
        x_defect = x_r[j]
        y_defect = y_r[j]
        z_defect = z_r[j]
        injected=True


    defect_rest = np.ones_like(cube_copy)
    defect_stress = np.ones_like(cube_copy)

    defect_rest[x_defect+int(ddx_rest),y_defect+int(ddy_rest),z_defect+int(d+ddz_rest)] = 0
    defect_stress[x_defect+int(ddx_stress),y_defect+int(ddy_stress),z_defect+int(d+ddz_stress)] = 0

    defect_rest = ndimage.filters.gaussian_filter(defect_rest,1)
    defect_rest = np.where(np.less(defect_rest,0.999),np.random.uniform(0.,0.5),1)

    defect_stress = ndimage.filters.gaussian_filter(defect_stress,1)
    defect_stress = np.where(np.less(defect_stress,0.999),np.random.uniform(0.,0.5),1)

    #writeto('test.fits',sm_cube_copy,clobber=True)
    #exit(0)

    if case=='infarction':
        #same in stress and rest
        cubenoise = np.random.normal(b_noise[0],b_noise[1],size=dx*dy*dz).reshape(rest_cube.shape)
        rest = rest_cube*defect_rest #apply defect
        rest += cubenoise  #add noise
        rest[rest<0] = 0. #make sure no neg
        rest = ndimage.filters.gaussian_filter(rest,sm)*mask #smooth

        stress = stress_cube*defect_stress #apply defect
        cubenoise = np.random.normal(b_noise[0],b_noise[1],size=dx*dy*dz).reshape(rest_cube.shape)
        stress += cubenoise
        stress[stress<0]=0.
        stress = ndimage.filters.gaussian_filter(stress,sm)*mask #smooth

    elif case=='ischaemia':
        #only on stress
        cubenoise = np.random.normal(b_noise[0],b_noise[1],size=dx*dy*dz).reshape(rest_cube.shape)
        rest = rest_cube #don't apply defect
        rest += cubenoise  #add noise
        rest[rest<0] = 0. #make sure no neg
        rest = ndimage.filters.gaussian_filter(rest,sm)*mask #smooth

        stress = stress_cube*defect_stress #apply defect
        cubenoise = np.random.normal(b_noise[0],b_noise[1],size=dx*dy*dz).reshape(rest_cube.shape)
        stress += cubenoise
        stress[stress<0]=0.
        stress = ndimage.filters.gaussian_filter(stress,sm)*mask #smooth

    elif case=='healthy':
        #not in either
        cubenoise = np.random.normal(b_noise[0],b_noise[1],size=dx*dy*dz).reshape(rest_cube.shape)
        rest = rest_cube #don't apply defect
        rest += cubenoise  #add noise
        rest[rest<0] = 0. #make sure no neg
        rest = ndimage.filters.gaussian_filter(rest,sm)*mask #smooth

        stress = stress_cube #don't apply defect
        cubenoise = np.random.normal(b_noise[0],b_noise[1],size=dx*dy*dz).reshape(rest_cube.shape)
        stress += cubenoise
        stress[stress<0]=0.
        stress = ndimage.filters.gaussian_filter(stress,sm)*mask #smooth

    elif case=='artefact':
        #normalises in stress
        cubenoise = np.random.normal(b_noise[0],b_noise[1],size=dx*dy*dz).reshape(rest_cube.shape)
        rest = rest_cube*defect_rest #apply defect
        rest += cubenoise  #add noise
        rest[rest<0] = 0. #make sure no neg
        rest = ndimage.filters.gaussian_filter(rest,sm)*mask #smooth

        stress = stress_cube #don't apply defect
        cubenoise = np.random.normal(b_noise[0],b_noise[1],size=dx*dy*dz).reshape(rest_cube.shape)
        stress += cubenoise
        stress[stress<0]=0.
        stress = ndimage.filters.gaussian_filter(stress,sm)*mask #smooth

    else: #mixed - not fully normalised at rest
        #not in either
        cubenoise = np.random.normal(b_noise[0],b_noise[1],size=dx*dy*dz).reshape(rest_cube.shape)
        rest = rest_cube*np.where(np.less(defect_rest,1),np.random.uniform(1.3,1.9)*defect_rest,1) #apply scaled defect
        rest += cubenoise  #add noise
        rest[rest<0] = 0. #make sure no neg
        rest = ndimage.filters.gaussian_filter(rest,sm)*mask #smooth

        stress = stress_cube*defect_stress #apply defect
        cubenoise = np.random.normal(b_noise[0],b_noise[1],size=dx*dy*dz).reshape(rest_cube.shape)
        stress += cubenoise
        stress[stress<0]=0.
        stress = ndimage.filters.gaussian_filter(stress,sm)*mask #smooth

    return (rest,stress,defect_rest,defect_stress)


best_fit = None
min_c = 1.e99

do = False

if do:
    for a in np.arange(0.01,0.5,0.01):
        for c in np.arange(0.01,1.0,0.01):
            for d in np.arange(1,20,1):
                for s in [1]: #arange(1.,3.1,0.1):
                    b=a
                    sm_cube = return_sim((a,b,c,d,s),mask=mask)
                    resid = sum((sm_cube-in_image)**2)
                    if resid<min_c:
                        print(a,b,c,c, d, s, resid)
                        min_c = resid
                        best_fit = sm_cube


p0 = (0.31, 1.0, 0.85, 7, 1,0,0)
dp0 = (0.1,0.1,0.1,3,2,2)

for case in ["infarction", "healthy"]:

    for i in range(1000):
        if i%1000==0: print(i, case)
        healthy_thresh = 0.9 #random.normal(0.5,0.1)
        #if healthy_thresh<0.3: healthy_thresh = 0.3
        #if healthy_thresh>0.9: healthy_thresh=0.9
        p = (np.random.uniform(p0[0]-dp0[0],p0[0]+dp0[0]),
             np.random.uniform(p0[1]-dp0[1],p0[1]+dp0[1]),
             np.random.uniform(p0[2]-dp0[2],p0[2]+dp0[2]),
             np.random.uniform(p0[3]-dp0[3],p0[3]+dp0[3]),
             1)
        rest,stress,defect_rest,defect_stress = return_sim(p,mask=mask,case=case)
        writeto('rest_%s_%05d.fits'%(case,i),rest,clobber=True)
        writeto('stress_%s_%05d.fits'%(case,i),stress,clobber=True)
        #writeto('defect_rest_%s_%04d.fits'%(case,i),defect_rest,clobber=True)
        #writeto('defect_stress_%s_%04d.fits'%(case,i),defect_stress,clobber=True)
