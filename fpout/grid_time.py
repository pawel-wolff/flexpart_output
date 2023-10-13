import warnings
import pkg_resources
import numpy as np
import pandas as pd
import xarray as xr

from common import longitude
import xarray_extras    # noqa


PIXEL_AREA_005DEG_URL = pkg_resources.resource_filename('fpout', 'resources/pixel_areas_005deg.nc')
PIXEL_AREA_005DEG_VAR = 'Pixel_area'
_pixel_area = None

OROGRAPHY_BY_RESOL_ULR = {}
OROGRAPHY_AVAIL_RESOL = {'1': 1, '05': .5, '025': 0.25}  # keep in decreasing order
for resol in OROGRAPHY_AVAIL_RESOL:
    OROGRAPHY_BY_RESOL_ULR[resol] = pkg_resources.resource_filename('fpout', f'resources/orography_{resol}deg.nc')
_orography_by_resol = {}

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


def _assign_releases_position_coords(ds, index_releases_by_time=False):
    sim_end = np.datetime64(pd.Timestamp(ds.attrs['iedate'] + 'T' + ds.attrs['ietime']))
    rel_time_start = sim_end + xr.where(ds.RELSTART <= ds.RELEND, ds.RELSTART, ds.RELEND).compute()
    rel_time_end = sim_end + xr.where(ds.RELSTART > ds.RELEND, ds.RELSTART, ds.RELEND).compute()
    ds = ds.assign_coords({
        'release_time': ('pointspec', (rel_time_start + (rel_time_end - rel_time_start) / 2).data,
                         {'long_name': 'release time',
                          'description': 'time coordinate of the center of a release box'}),
        'release_lon': ('pointspec', longitude.geodesic_longitude_midpoint(ds['RELLNG1'], ds['RELLNG2']).data,
                        {'long_name': 'release longitude',
                         'units': 'degrees_est',
                         'description': 'longitude coordinate of the center of a release box'}),
        'release_lat': ('pointspec', ((ds['RELLAT1'] + ds['RELLAT2']) / 2).data,
                        {'long_name': 'release latitude',
                         'units': 'degrees_nord',
                         'description': 'longitude coordinate of the center of a release box'}),
        'release_pressure': ('pointspec', (100. * (ds['RELZZ1'] + ds['RELZZ2']) / 2).data,
                             {'long_name': 'release pressure',
                              'units': 'Pa',
                              'description': 'pressure coordinate of the center of a release box'}),
        'release_npart': ('pointspec', ds['RELPART'].data,
                          {'long_name': 'number of release particles',
                           'units': '1',
                           'description': 'number of particles released from a release box'}),
    })
    if index_releases_by_time:
        ds = ds.set_index({'pointspec': 'release_time'}).rename({'pointspec': 'release_time'})
    return ds


def _assign_extra_height_coords(ds):
    height = ds[HEIGHT_DIM]
    top_height = height.astype('f8').data
    top_height_attrs = dict(height.attrs)
    top_height_attrs.update({'long_name': 'top height above ground of a grid cell'})
    bottom_height = np.concatenate(([0.], top_height[:-1]))
    bottom_height_attrs = dict(height.attrs)
    bottom_height_attrs.update({'long_name': 'bottom height above ground of a grid cell'})
    delta_height_attrs = dict(height.attrs)
    delta_height_attrs.update({'long_name': 'height of a grid cell'})

    ds = ds.assign_coords({
        TOP_HEIGHT: (HEIGHT_DIM, top_height, top_height_attrs),
        BOTTOM_HEIGHT: (HEIGHT_DIM, bottom_height, bottom_height_attrs),
        DELTA_HEIGHT: (HEIGHT_DIM, top_height - bottom_height, delta_height_attrs),
    })
    return ds


def get_pixel_area():
    # TODO: manage it by caching
    global _pixel_area
    if _pixel_area is None:
        # prepare Pixel_area data
        ds = xr.load_dataset(PIXEL_AREA_005DEG_URL)
        _pixel_area = ds[PIXEL_AREA_005DEG_VAR]
    return _pixel_area


def _get_orography(resol):
    # TODO: manage it by caching
    # prepare orography for a given resolution
    global _orography_by_resol
    if resol not in _orography_by_resol:
        if resol not in OROGRAPHY_AVAIL_RESOL:
            raise ValueError(f'resol={resol} is not available; use one of {OROGRAPHY_AVAIL_RESOL}')
        _orography_by_resol[resol] = xr.load_dataset(OROGRAPHY_BY_RESOL_ULR[resol])['ORO']
    return _orography_by_resol[resol]


def _rt_transform_by_air_density(rt, oro):
    mean_height = 0.5 * (rt[TOP_HEIGHT] + rt[BOTTOM_HEIGHT])
    density = vertical_profile_of_air_density.interp(coords={'height': mean_height + oro}).astype(np.float32)
    rt = rt * density
    rt.attrs['units'] = 's'
    return rt


def open_dataset(
        url,
        assign_releases_position_coords=True,
        index_releases_by_time=False,
        pixel_area=True,
        normalize_longitude=True,
        generic_orography=True,
        res_time_in_sec=True,
        chunks='auto',
        max_chunk_size=None,
):
    """
    Open a dataset with FLEXPART output
    :param url: path to Flexpart output dataset; must be in netcdf format and conform the Flexpart v10 output
    :param assign_releases_position_coords: bool; if True, assign releases positions as auxiliary coordinates (
    :param pixel_area:
    :param normalize_longitude:
    :param generic_orography:
    :param chunks:
    :return: xarray.Dataset
    """
    ds = _open_dataset(
        url,
        assign_releases_position_coords=assign_releases_position_coords,
        index_releases_by_time=index_releases_by_time,
        pixel_area=pixel_area,
        chunks=chunks,
        max_chunk_size=max_chunk_size,
    )

    ds = _assign_extra_height_coords(ds)

    # normalize longitude
    if normalize_longitude:
        ds = ds.geo.normalize_longitude(keep_attrs=True)

    # set orography
    if generic_orography:
        ds_lon = ds[ds.geo.get_lon_label()]
        target_resol = float(abs(ds_lon[1] - ds_lon[0]))
        for resol_str, resol in OROGRAPHY_AVAIL_RESOL.items():
            if resol <= target_resol:
                break
        oro = _get_orography(resol_str).astype('f8')
        oro = oro.geo_regrid.regrid_lon_lat(target_resol_ds=ds, method='linear',
                                            longitude_circular=True, keep_attrs=True)
    else:
        oro = ds['ORO'].astype('f8')
    ds = ds.assign_coords({OROGRAPHY: oro})

    if res_time_in_sec:
        # set residence time in [s]
        ind_source = int(ds.attrs['ind_source'])
        ind_receptor = int(ds.attrs['ind_receptor'])
        if ind_source == 1 and ind_receptor == 2:
            # must change residence time units from 's m3 kg-1' to 's' by multiplying by air density at output grid cell
            ds[RES_TIME] = _rt_transform_by_air_density(ds['spec001_mr'], ds[OROGRAPHY])
        elif ind_source == 2 and ind_receptor == 2:
            ds[RES_TIME] = ds['spec001_mr']

    return ds


def open_fp_dataset(url, assign_releases_position_coords=True, pixel_area=False, chunks='auto', max_chunk_size=None):
    warnings.warn('use open_dataset', FutureWarning)
    return _open_dataset(
        url,
        assign_releases_position_coords=assign_releases_position_coords,
        index_releases_by_time=True,
        pixel_area=pixel_area,
        chunks=chunks,
        max_chunk_size=max_chunk_size
    )


def _open_dataset(
        url,
        assign_releases_position_coords=True,
        index_releases_by_time=False,
        pixel_area=True,
        chunks='auto',
        max_chunk_size=None
):
    ds = xarray_extras.open_dataset_with_disk_chunks(url, chunks=chunks, max_chunk_size=max_chunk_size, engine='h5netcdf')

    backward_run = int(ds.attrs['ldirect']) == -1

    ds['time'].attrs.update({
        'standard_name': 'time',
        'long_name': f'{"left" if backward_run else "right"} endpoint of a simulation time interval'
    })

    if ds['spec001_mr'].dtype != 'f4':
        ds['spec001_mr'] = ds['spec001_mr'].astype('f4')
    if assign_releases_position_coords:
        ds = _assign_releases_position_coords(ds, index_releases_by_time=index_releases_by_time)
    if pixel_area:
        pixel_area_ds = get_pixel_area().geo_regrid.regrid_lon_lat(target_resol_ds=ds, method='sum',
                                                                   longitude_circular=True, keep_attrs=True)
        ds = ds.assign_coords(area=pixel_area_ds)
        ds['area'].attrs.update({
            'standard_name': 'cell_area',
            'long_name': 'horizontal area of a grid cell'
        })
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
