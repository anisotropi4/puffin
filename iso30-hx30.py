#!/usr/bin/env python3
"""iso-hx30.py: create region detailed interpolated hexagon"""
import argparse
import warnings
from functools import partial

# os.environ["USE_PYGEOS"] = "0"
import geopandas as gp
import numpy as np
import pandas as pd
from pyogrio import read_dataframe, write_dataframe
from pyogrio.errors import DataLayerError, DataSourceError
from shapely import set_precision
from tobler.area_weighted import area_interpolate

from shared.util import (
    append_layer,
    density,
    get_grid,
    get_hexagon,
    get_hside,
    get_yx,
    log,
    simplify_outer,
)

pd.set_option("display.max_columns", None)
WGS84 = "EPSG:4326"
CRS = "EPSG:3034"

INPATH = "europa.gpkg"

set_precision_one = partial(set_precision, grid_size=1.0)


def simplify_boundary(this_gf):
    """simplify_boundary:

    args:
      this_gf: GeoDataFrame
    :rvalue GeoSeries

    returns:

    """
    r = this_gf["geometry"].apply(set_precision_one)
    r = r.simplify(100.0)
    return r


def get_exterior(this_gf):
    """get_exterior:

    args:
      this_gf:

    returns:

    """
    r = this_gf["geometry"].values[0]
    return gp.GeoSeries(r.geoms, crs=CRS)


def get_population2(this_hx, inpath):
    """get_poluation2:

    args:
      this_hx: param inpath:
      inpath:

    returns:

    """
    bbox = tuple(this_hx.total_bounds)
    column = "population"
    population = read_dataframe(inpath, layer=column, bbox=bbox)
    hexagon = gp.GeoSeries(this_hx, crs=CRS).to_frame("geometry")
    r = area_interpolate(
        population, hexagon, allocate_total=False, extensive_variables=[column]
    )
    return r[column].values


def get_euro_boundary():
    """get_euro_boundary:

    args:
      None

    returns:
      cached simplified GeoDataFrame European boundary

    """
    _r = read_dataframe(INPATH, "boundary")
    _r["geometry"] = _r["geometry"].apply(set_precision_one)

    def _get_euro_boundary():
        """ """
        return _r

    return _get_euro_boundary


def set_region_boundary(outpath, boundary, country):
    """set_region_boundary:

    args:
      outpath: output file path
      boundary: European GeoDataFrame
      country: ISO A3 country list

    returns:
      GeoDataFrame region boundary

    """
    layer = "region_boundary"
    try:
        r = read_dataframe(outpath, layer=layer)
        return r
    except (DataLayerError, DataSourceError):
        pass
    ix = boundary["a3_code"].isin(country)
    r = boundary.loc[ix]
    r.loc[:, "geometry"] = simplify_boundary(r)
    write_dataframe(r, outpath, layer=layer)
    return r


def set_region_outer(outpath, boundary, country):
    """set_region_outer:

    args:
      country: ISO A3 country list

    returns:


    """
    layer = "region_outer"
    try:
        r = read_dataframe(OUTPATH, layer=layer)
        return r
    except (DataLayerError, DataSourceError):
        pass
    r = set_region_boundary(outpath, boundary, country)
    r = gp.GeoSeries(simplify_outer(r), crs=CRS)
    write_dataframe(r.to_frame("geometry"), outpath, layer=layer)
    return r


def set_region_circle(outpath, boundary, country):
    """set_region_circle:

    args:
      boundary:
      country: ISO A3 country list

    returns:
       GeoSeries centre point and circle GeoSeries

    """
    layer = "region_circle"
    try:
        r = read_dataframe(outpath, layer=layer)
        point = gp.GeoSeries(r.centroid, crs=CRS)
        return point, r
    except (DataLayerError, DataSourceError):
        pass
    r = set_region_outer(outpath, boundary, country)
    r = r.minimum_bounding_circle()
    write_dataframe(r.to_frame("geometry"), outpath, layer=layer)
    point = gp.GeoSeries(r.centroid, crs=CRS)
    return point, r


def set_zone_circle(outpath, circle, this_buffer=200.0e3):
    """set_zone_circle:

    args:
      this_buffer: Default value = 200.0e3)

    returns:

    """
    layer = "zone_circle"
    try:
        r = read_dataframe(OUTPATH, layer=layer)
        point = gp.GeoSeries(r.centroid, crs=CRS)
        return point, r
    except (DataLayerError, DataSourceError):
        pass
    r = circle.buffer(this_buffer)
    write_dataframe(r.to_frame("geometry"), outpath, layer=layer)
    point = gp.GeoSeries(r.centroid, crs=CRS)
    return point, r


def get_zone_boundary(circle, boundary, this_buffer=0.0):
    """get_zone_boundary:

    args:
      this_buffer: Default value = 100.0e3)

    returns:

    """
    radius = circle.minimum_bounding_radius().values[0]
    radius += this_buffer
    centre = circle.centroid
    geometry = boundary["geometry"].to_crs(CRS)
    ix = geometry.distance(centre.values[0]) < radius
    return boundary[ix]


def set_zone_outer(outpath, zone_circle, this_buffer=0.0):
    """set_zone_outer:

    args:
      this_buffer: Default value = 0.0e3)

    returns:

    """
    layer = "zone_outer"
    try:
        r = read_dataframe(OUTPATH, layer=layer)
        return r
    except (DataLayerError, DataSourceError):
        pass
    circle = zone_circle.buffer(this_buffer)
    euro_boundary = get_euro_boundary()
    r = get_zone_boundary(circle, euro_boundary())
    r = simplify_outer(r)
    r = gp.GeoSeries(r, crs=CRS)
    r = r.clip(circle).to_frame("geometry")
    write_dataframe(r, outpath, layer=layer)
    return r


def base_hexagon(grid, length, circle):
    """base_hexagon:

    args:
      grid:
      length:
      circle:

    returns:

    """
    r = get_hexagon(grid)
    ix = r.length < 6.04 * length
    r = r.loc[ix]
    r = r.clip(circle).explode(index_parts=False).reset_index(drop=True)
    ix = r.type == "Polygon"
    r = r[ix].to_frame("geometry")
    # write_dataframe(r, "dump.gpkg", layer=f"clip-{key}")
    # r["population"] = r["geometry"].map(get_population_inpath)
    return r


def get_hexagon2(grid, length, circle):
    """get_hexagon2:

    args:
      grid:
      length:
      circle:

    returns:

    """
    r = base_hexagon(grid, length, circle)
    r["population"] = get_population2(r["geometry"], INPATH)
    return r


def ix_difference(this_hx, interpolate, length):
    """ix_difference:

    args:
      this_hx:
      interpolate:
      length:

    returns:

    """
    centroid_hx = this_hx.centroid.to_frame("geometry")
    centroid = interpolate.centroid.to_frame("geometry")
    ix = centroid_hx.sjoin_nearest(centroid, max_distance=length).index
    return centroid_hx.index.difference(ix)


def get_hexagon3(grid, length, circle):
    """get_hexagon3:

    args:
      grid:
      length:
      circle:

    returns:

    """
    boost = get_hside(1)
    base_grid = get_grid(1, circle.buffer(boost)).to_frame("geometry")
    r = gp.GeoDataFrame()
    count = base_grid.size
    for i, j in base_grid.iterrows():
        log(f"{str(i + 1).zfill(3)}\t{str(count).zfill(3)}")
        k = gp.GeoSeries(j.values, crs=CRS).to_frame("geometry")
        k = grid.to_frame("geometry").sjoin_nearest(k, max_distance=boost)
        if k.empty:
            continue
        s = base_hexagon(k["geometry"], length, circle)
        ix = s.index
        if not r.empty:
            ix = ix_difference(s, r, length)
            s = s.loc[ix]
        if s.empty:
            continue
        # write_dataframe(s, "dump.gpkg", layer="hexagon")
        s["population"] = get_population2(s["geometry"], INPATH)
        r = pd.concat([r, s])
        # write_dataframe(r, "dump.gpkg", layer="interpolate")
    return r


def set_hexagon(outpath, grid, length, circle, layer):
    """set_hexagon:

    args:
      this_grid:
      circle:
      grid:
      length:
      layer:

    returns:
      GeoDataFrame

    """
    step = np.ceil(np.log2(grid.size))
    if step < 19:
        r = get_hexagon2(grid, length, circle)
    else:
        r = get_hexagon3(grid, length, circle)
    ix = r["population"] > 0.0
    r = r[ix]
    r["density"] = density(r)
    r = r.drop_duplicates(subset="geometry").reset_index(drop=True)
    ix = r.centroid.map(get_yx).sort_values()
    r = r.loc[ix.index].reset_index(drop=True)
    write_dataframe(r, outpath, layer=layer)
    return r


def get_hex_grid(exterior, this_buffer, n, pivot, angle):
    """get_hex_grid:

    args:

    returns:
      None

    """
    count = exterior.size
    grid = gp.GeoSeries()
    for i, j in exterior.items():
        if i % 32 == 0:
            log(f"{str(i + 1).zfill(4)}\t{count}")
        geometry = gp.GeoSeries(j, crs=CRS)
        geometry = geometry.rotate(angle, origin=pivot)
        r = get_grid(n, geometry.buffer(this_buffer))
        r = r.rotate(-angle, origin=pivot)
        grid = pd.concat([r, grid])
    return grid.drop_duplicates().reset_index(drop=True)


def main(outpath, region):
    """1. get region boundary
    2. get region outer
    3. get region circle
    4. get buffered circle
    5. get zone boundary
    6. get zone outer
    7. get zone exterior


    returns:
      None

    """
    log(f"start\t{outpath}")
    warnings.simplefilter(action="ignore", category=FutureWarning)
    euro_boundary = get_euro_boundary()
    _, circle = set_region_circle(outpath, euro_boundary(), region)
    centre_point, zone_circle = set_zone_circle(outpath, circle)
    pivot = centre_point[0]
    exterior = get_exterior(set_zone_outer(outpath, zone_circle))
    for n in range(1, 9):
        length = get_hside(n)
        for m in ["00", "30"]:
            log(f"{str(n).zfill(2)}\thex world-pop")
            log(f"{str(n).zfill(2)}\t{m}")
            key = f"{str(n).zfill(2)}-{m}"
            if append_layer(OUTPATH, f"interpolate-{key}"):
                continue
            grid = get_hex_grid(exterior, 2.5 * length, n, pivot, int(m))
            log("interpolate")
            set_hexagon(outpath, grid, length, zone_circle, f"interpolate-{key}")
            log(f"wrote layer {key}")
    log("wrote hex world-pop")


if __name__ == "__main__":
    COUNTRY = {
        "ine": ["GBR", "IMN", "IRL"],
        "dach": ["AUT", "BEL", "CHE", "DEU", "LIE", "LUX"],
        "fra": ["FRA"],
        "ibe": ["ESP", "POR"],
    }
    parser = argparse.ArgumentParser(
        description="create detailed region interpolated hexagon"
    )
    parser.add_argument(
        "region",
        nargs="?",
        type=str,
        help="country option",
        default="ine",
        choices=list(COUNTRY.keys()),
    )
    args = parser.parse_args()
    OUTPATH = f"{args.region}.gpkg"
    REGION = COUNTRY[args.region]
    main(OUTPATH, REGION)
