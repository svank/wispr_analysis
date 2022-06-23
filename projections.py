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
        
        self.pa_of_ecliptic = 90
    
    
    def __call__(self, pixel_out):
        pixel_out = pixel_out.astype(float, copy=False)
        pixel_in = np.empty_like(pixel_out)
        elongation, pa = self.all_pix2world(
                pixel_out[..., 0], pixel_out[..., 1])
        
        hp_lon, hp_lat = self.elongation_to_hp(elongation, pa)
        
        input_x, input_y = self.wcs_in.all_world2pix(hp_lon, hp_lat, 0)
        
        pixel_in[..., 0] = input_x
        pixel_in[..., 1] = input_y
        return pixel_in
    
    
    def hp_to_elongation(self, lon, lat):
        lon = lon * np.pi / 180
        lat = lat * np.pi / 180
        
        # Expressions from Snyder (1987)
        # https://pubs.er.usgs.gov/publication/pp1395
        # Eqn (5-3a)
        elongation = 2 * np.arcsin(np.sqrt(
            np.sin(lat/2)**2 + np.cos(lat) * np.sin(lon/2)**2
        ))
        # Eqn (5-4b)
        pa = np.arctan2(np.cos(lat) * np.sin(lon), np.sin(lat))
        
        elongation *= 180 / np.pi
        pa *= 180 / np.pi
        pa += (self.pa_of_ecliptic - 90)
        
        return elongation, pa
    
    
    def elongation_to_hp(self, elongation, pa):
        elongation = elongation * np.pi / 180
        pa = pa * np.pi / 180
        pa -= (self.pa_of_ecliptic - 90)
        
        # Expressions from Snyder (1987)
        # https://pubs.er.usgs.gov/publication/pp1395
        # Eqn (5-5)
        lat = np.arcsin(np.sin(elongation) * np.cos(pa))
        # Eqn (5-6)
        lon = np.arctan2(np.sin(elongation) * np.sin(pa), np.cos(elongation))
        
        lat *= 180 / np.pi
        lon *= 180 / np.pi
        return lon, lat
    
    
    def all_pix2world(self, x, y, origin=0):
        x = x - origin
        y = y - origin
        pa = (y - self.ref_y) * self.dpa + self.ref_pa
        elongation = ((x - self.ref_x) * self.delongation
                + self.ref_elongation)
        
        return elongation, pa
    
    
    def all_world2pix(self, elongation, pa, origin=0):
        x = (elongation - self.ref_elongation) / self.delongation + self.ref_x
        y = (pa - self.ref_pa) / self.dpa + self.ref_y
        
        x += origin
        y += origin
        
        return x, y


def reproject_to_radial(data, wcs, out_shape=None, dpa=None, delongation=None,
        ref_pa=100, ref_elongation=13):
    if out_shape is None:
        # Defaults that are decent for a full-res WISPR-I image
        out_shape = list(data.shape)
        out_shape[1] -= 250
        out_shape[1] *= 2
        out_shape[0] += 350
    if dpa is None:
        dpa = wcs.wcs.cdelt[1] * 1.5
    if delongation is None:
        delongation = wcs.wcs.cdelt[0] * .75
    transformer = RadialTransformer(
            ref_pa=ref_pa, ref_y=out_shape[0]//2, dpa=-dpa,
            ref_elongation=ref_elongation, ref_x=0, delongation=delongation,
            wcs_in=wcs)
    reprojected = np.zeros(out_shape)
    reproject.adaptive.deforest.map_coordinates(data.astype(float),
            reprojected, transformer, out_of_range_nan=True,
            center_jacobian=False)
    return reprojected, transformer
