import numpy as np
import pandas as pd
import xarray as xr

from common import longitude
import xarray_extras

PIXEL_AREA_0_1_DEG_URL = '/o3p/iagos/softio/EMISSIONS/GFASv1.2_0.1_pixel_areas.nc'
PIXEL_AREA_0_1_DEG_VAR = 'Pixel_area'
_pixel_area = None


def _assign_position_coords(fp_ds):
    sim_end = np.datetime64(pd.Timestamp(fp_ds.attrs['iedate'] + 'T' + fp_ds.attrs['ietime']))
    rel_time_start = sim_end + xr.where(fp_ds.RELSTART <= fp_ds.RELEND, fp_ds.RELSTART, fp_ds.RELEND).compute()
    rel_time_end = sim_end + xr.where(fp_ds.RELSTART > fp_ds.RELEND, fp_ds.RELSTART, fp_ds.RELEND).compute()
    fp_ds = fp_ds.assign_coords({
        'release_time': ('pointspec', rel_time_start + (rel_time_end - rel_time_start) / 2),
        'release_lon': ('pointspec', longitude.geodesic_longitude_midpoint(fp_ds['RELLNG1'], fp_ds['RELLNG2'])),
        'release_lat': ('pointspec', (fp_ds['RELLAT1'] + fp_ds['RELLAT2']) / 2),
        'release_pressure': ('pointspec', 100. * (fp_ds['RELZZ1'] + fp_ds['RELZZ2']) / 2),
        'release_npart': ('pointspec', fp_ds['RELPART']),
    })
    return fp_ds.set_index({'pointspec': 'release_time'}).rename({'pointspec': 'release_time'})


def _get_pixel_area():
    # prepare Pixel_area data
    global _pixel_area
    if _pixel_area is None:
        ds = xr.load_dataset(PIXEL_AREA_0_1_DEG_URL)
        _pixel_area = ds[PIXEL_AREA_0_1_DEG_VAR].astype('f8') * 1e6
        _pixel_area.attrs = dict(standard_name='pixel_area', units='m2')
    return _pixel_area


def open_fp_dataset(url, releases_position_coords=True, pixel_area=False, chunks='auto'):
    ds = xarray_extras.open_dataset_with_disk_chunks(url, chunks=chunks)
    if ds['spec001_mr'].dtype != 'f4':
        ds['spec001_mr'] = ds['spec001_mr'].astype('f4')
    if releases_position_coords:
        ds = _assign_position_coords(ds)
    if pixel_area:
        pixel_area_ds = _get_pixel_area().geo_regrid.regrid_lon_lat(target_resol_ds=ds, method='sum',
                                                                    longitude_circular=True, keep_attrs=True)
        ds = ds.assign(area=pixel_area_ds)
        if ds['area'].isnull().any():
            raise ValueError(f'coordinates of {pixel_area_ds} and {ds} do not match')
    # ds = ds.assign(res_time=(ds['spec001_mr'] / ds['area'])) #.astype('f4')) #.persist()
    # ds['res_time'].attrs.update(units='s km-2')
    return ds


def aggregate_in_time(ds, time_aggr='24h'):
    time_aggr = pd.Timedelta(time_aggr)
    t = ds.time
    dt = pd.Timedelta(abs(t[0] - t[1]).values)
    if time_aggr % dt != pd.Timedelta(0):
        raise ValueError(f'time_aggr={time_aggr} must be a multiple of dt={dt}')
    return ds.coarsen(dim={'time': time_aggr // dt}, boundary='trim', coord_func='min', keep_attrs=True)\
        .sum(keep_attrs=True)


def rolling_in_time(ds, time_aggr='10days'):
    time_aggr = pd.Timedelta(time_aggr)
    t = ds.time
    dt = pd.Timedelta(abs(t[0] - t[1]).values)
    if time_aggr % dt != pd.Timedelta(0):
        raise ValueError(f'time_aggr={time_aggr} must be a multiple of dt={dt}')
    if ds.chunks.get('time') is not None:
        ds = ds.chunk({'time': -1})
    return ds.rolling(dim={'time': time_aggr // dt}).sum(keep_attrs=True).dropna('time')
