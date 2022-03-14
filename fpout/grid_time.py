import warnings
import pkg_resources
import numpy as np
import pandas as pd
import xarray as xr

from common import longitude
import xarray_extras    # noqa


PIXEL_AREA_0_1_DEG_URL = pkg_resources.resource_filename('fpout', 'resources/pixel_areas.nc')
PIXEL_AREA_0_1_DEG_VAR = 'Pixel_area'
_pixel_area = None

AIR_DENSITY_URL = pkg_resources.resource_filename('fpout', 'resources/vertical_profile_of_air_density.csv')

HEIGHT_DIM = 'height'
TOP_HEIGHT = 'top_height'
BOTTOM_HEIGHT = 'bottom_height'
DELTA_HEIGHT = 'delta_height'
OROGRAPHY = 'orography'
RES_TIME = 'res_time'

# prepare mean air density
vertical_profile_of_air_density = pd.read_csv(AIR_DENSITY_URL, index_col='height_in_m', dtype=np.float32)['density_in_kg_per_m3']
vertical_profile_of_air_density = xr.DataArray.from_series(vertical_profile_of_air_density).rename({'height_in_m': 'height'})
vertical_profile_of_air_density['height'] = vertical_profile_of_air_density.height.astype('f4')
vertical_profile_of_air_density.name = 'air_density'
vertical_profile_of_air_density.attrs = dict(long_name='air_density', units='kg m-3')


def _assign_position_coords(ds, index_releases_by_time=False):
    sim_end = np.datetime64(pd.Timestamp(ds.attrs['iedate'] + 'T' + ds.attrs['ietime']))
    rel_time_start = sim_end + xr.where(ds.RELSTART <= ds.RELEND, ds.RELSTART, ds.RELEND).compute()
    rel_time_end = sim_end + xr.where(ds.RELSTART > ds.RELEND, ds.RELSTART, ds.RELEND).compute()
    ds = ds.assign_coords({
        'release_time': ('pointspec', (rel_time_start + (rel_time_end - rel_time_start) / 2).data),
        'release_lon': ('pointspec', longitude.geodesic_longitude_midpoint(ds['RELLNG1'], ds['RELLNG2']).data),
        'release_lat': ('pointspec', ((ds['RELLAT1'] + ds['RELLAT2']) / 2).data),
        'release_pressure': ('pointspec', (100. * (ds['RELZZ1'] + ds['RELZZ2']) / 2).data),
        'release_npart': ('pointspec', ds['RELPART'].data),
    })
    if index_releases_by_time:
        ds = ds.set_index({'pointspec': 'release_time'}).rename({'pointspec': 'release_time'})
    return ds


def _assign_extra_height_coords(ds):
    top_height = ds[HEIGHT_DIM].astype('f8')
    ds = ds.assign_coords({
        TOP_HEIGHT: top_height,
        BOTTOM_HEIGHT: (HEIGHT_DIM, np.concatenate(([0.], top_height.data[:-1]))),
    })
    ds = ds.assign_coords({
        DELTA_HEIGHT: ds[TOP_HEIGHT] - ds[BOTTOM_HEIGHT]
    })
    return ds


def _get_pixel_area():
    # TODO: manage it by caching
    # prepare Pixel_area data
    global _pixel_area
    if _pixel_area is None:
        ds = xr.load_dataset(PIXEL_AREA_0_1_DEG_URL)
        _pixel_area = ds[PIXEL_AREA_0_1_DEG_VAR].astype('f8') * 1e6
        _pixel_area.attrs = dict(standard_name='pixel_area', units='m2')
    return _pixel_area


def _rt_transform_by_air_density(rt, oro):
    mean_height = 0.5 * (rt[TOP_HEIGHT] + rt[BOTTOM_HEIGHT])
    density = vertical_profile_of_air_density.interp(coords={'height': mean_height + oro}).astype(np.float32)
    rt = rt * density
    rt.attrs['units'] = 's'
    return rt


def open_dataset(url, releases_position_coords=True, pixel_area=True, normalize_longitude=True, chunks='auto'):
    ds = _open_dataset(url, releases_position_coords=releases_position_coords, pixel_area=pixel_area, chunks=chunks)
    ds = _assign_extra_height_coords(ds)

    # set orography
    ds[OROGRAPHY] = ds['ORO'].astype('f8')
    ds = ds.set_coords(OROGRAPHY)

    # set residence time in [s]
    ind_source = int(ds.attrs['ind_source'])
    ind_receptor = int(ds.attrs['ind_receptor'])
    if ind_source == 1 and ind_receptor == 2:
        # must change residence time units from 's m3 kg-1' to 's' by multiplying by air density at output grid cell
        ds[RES_TIME] = _rt_transform_by_air_density(ds['spec001_mr'], ds[OROGRAPHY])
    elif ind_source == 2 and ind_receptor == 2:
        ds[RES_TIME] = ds['spec001_mr']

    # set pixel area
    if pixel_area:
        ds = ds.set_coords('area')

    # normalize longitude
    if normalize_longitude:
        ds = ds.geo.normalize_longitude(keep_attrs=True)

    return ds


def open_fp_dataset(url, releases_position_coords=True, pixel_area=False, chunks='auto'):
    warnings.warn('use open_dataset', FutureWarning)
    return _open_dataset(url, index_releases_by_time=True,
                         releases_position_coords=releases_position_coords, pixel_area=pixel_area, chunks=chunks)


def _open_dataset(url, index_releases_by_time=False, releases_position_coords=True, pixel_area=True, chunks='auto'):
    ds = xarray_extras.open_dataset_with_disk_chunks(url, chunks=chunks)
    if ds['spec001_mr'].dtype != 'f4':
        ds['spec001_mr'] = ds['spec001_mr'].astype('f4')
    if releases_position_coords:
        ds = _assign_position_coords(ds, index_releases_by_time=index_releases_by_time)
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
    return ds.coarsen(dim={'time': time_aggr // dt}, boundary='trim', coord_func='min').sum(keep_attrs=True)


def rolling_in_time(ds, time_aggr='10days'):
    time_aggr = pd.Timedelta(time_aggr)
    t = ds.time
    dt = pd.Timedelta(abs(t[0] - t[1]).values)
    if time_aggr % dt != pd.Timedelta(0):
        raise ValueError(f'time_aggr={time_aggr} must be a multiple of dt={dt}')
    if ds.chunks.get('time') is not None:
        ds = ds.chunk({'time': -1})
    return ds.rolling(dim={'time': time_aggr // dt}).sum(keep_attrs=True).dropna('time')
