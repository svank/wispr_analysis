import numpy as np
import reproject


class RadialTransformer():
    def __init__(self, ref_pa, ref_y, dpa,
            ref_elongation, ref_x, delongation, wcs_in):
        self.ref_pa = ref_pa
        self.ref_elongation = ref_elongation
        self.ref_x = ref_x
        self.ref_y = ref_y
        self.dpa = dpa
        self.delongation = delongation
        self.wcs_in = wcs_in
        
        self.pa_of_ecliptic = 90 * np.pi / 180
    
    
    def __call__(self, pixel_out):
        pixel_in = np.empty_like(pixel_out)
        pa = (pixel_out[..., 1] - self.ref_y) * self.dpa + self.ref_pa
        pa *= np.pi / 180
        elongation = ((pixel_out[..., 0] - self.ref_x) * self.delongation
                + self.ref_elongation)
        hp_lon = elongation * np.cos(pa - self.pa_of_ecliptic)
        hp_lat = elongation * np.sin(pa - self.pa_of_ecliptic)
        
        input_x, input_y = self.wcs_in.all_world2pix(hp_lon, hp_lat, 0)
        
        pixel_in[..., 0] = input_x
        pixel_in[..., 1] = input_y
        return pixel_in


def reproject_to_radial(data, wcs):
    out_shape = list(data.shape)
    out_shape[1] -= 250
    out_shape[1] *= 2
    out_shape[0] += 350
    transformer = RadialTransformer(
            80, out_shape[0]//2, wcs.wcs.cdelt[1] * 1.5,
            13, 0, wcs.wcs.cdelt[0] * .75, wcs)
    reprojected = np.zeros(out_shape)
    reproject.adaptive.deforest.map_coordinates(data.astype(float),
            reprojected, transformer, out_of_range_nan=True,
            center_jacobian=False)
    return reprojected
