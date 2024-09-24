def normalize_longitude(arr, smallest_lon_coord=-180.):
    return (arr - smallest_lon_coord) % 360. + smallest_lon_coord


def geodesic_longitude_midpoint(lon1, lon2, p=0.5):
    """
    Calculates a geodesic longitude of (1-p) * lon1 + p * lon2. Allows vectorization.
    :param lon1: float or array-like of floats: longitude(s) of a first point(s)
    :param lon2: float or array-like of floats longitude(s) of a second point(s)
    :param p: float or array-like of floats in [0, 1]
    :return: float or array-like of floats: longitude(s) of a geodesic midpoint(s)
    """
    return normalize_longitude(lon1 + p * normalize_longitude(lon2 - lon1))
