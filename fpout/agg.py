import pathlib
import numpy as np
import xarray as xr

from .grid_time import open_dataset


def get_agg_footprint_data(fp_dir):
    url = pathlib.PurePath(fp_dir) / 'grid_time.nc'
    with open_dataset(url, max_chunk_size=1e8) as _ds:
        da = _ds['res_time'].squeeze('nageclass')
        t = da['release_time'].min().dt.ceil('D').values.astype('M8[ns]')
        da2 = da.sel(time=slice(t - np.timedelta64(10, 'D'), None))
        res_time = da2.sum(['height', 'time'])
        res_time_per_km2 = res_time / res_time['area'] * 1e6
        res_time_per_km2 = res_time_per_km2.astype('f4').compute()
        ds = xr.Dataset(data_vars={'res_time_per_km2': res_time_per_km2})
        return ds


def save_agg_footprint_data(fp_dir, dest):
    ds = get_agg_footprint_data(fp_dir)
    dest_dir = pathlib.Path(dest).parent
    dest_dir.mkdir(parent=True, exist_ok=True)
    tmp_dest = str(dest) + '.tmp'
    ds.to_netcdf(tmp_dest, encoding={'res_time_per_km2': {'zlib': True, 'complevel': 4}})
    pathlib.Path(tmp_dest).rename(dest)
