#!/usr/bin/env python3
"""euro-ghs: project GHS population data"""

import datetime as dt
from functools import partial

# os.environ["USE_PYGEOS"] = "0"
import geopandas as gp
import numpy as np
import pandas as pd
from pyogrio import read_dataframe, write_dataframe
from shapely import set_precision

from shared.util import (
    append_layer,
    filter_hexagon,
    get_hex_id,
    get_hexagon,
    get_hx_centre,
    get_population,
    log,
    set_pivot,
)

pd.set_option("display.max_columns", None)
CRS = "EPSG:3034"
CHECKPOINT = False
# CRS = "EPSG:32630"

INPATH = "europa.gpkg"
GHDIR = "GHS_STAT_UCDB2015MT_GLOBE_R2019A"
GHPOINT = "GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2_short_pnt"
GHSHAPE = "GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2"
OUTPATH = "europa-ghs.gpkg"

set_precision_one = partial(set_precision, grid_size=1.0)
get_population_path = partial(get_population, inpath=INPATH)


START = dt.datetime.now()

def get_circle():
    """get_circle:

    returns:
      circle GeoDataFrame centre Poinr and Polygon
    """
    circle = read_dataframe(INPATH, layer="circle")
    centre = circle.centroid
    return centre, circle


def get_ghs(circle, filename):
    """get_ghs:
    
    input:
      circle:
      filename

    returns:
      GeoSeries

    """
    filepath = f"{GHDIR}/{filename}.gpkg"
    r = read_dataframe(filepath).to_crs(CRS)
    return r.clip(circle).reset_index(drop=True)

def get_grid(n, circle):
    """get_grid: return sufficient grid points to allow construction of hexagons"
    "in two orientations to cover circle. boost and offset parameters determined"
    "by trial and error"""
    hex_id = get_hex_id(n, circle)
    r = gp.GeoSeries(get_hx_centre(hex_id), crs=CRS)
    return r.apply(set_precision_one)


def density(this_gf, column="population"):
    """density: from GeoSeries 'this_gf' return km2 density of 'column'"""
    return 1.0e6 * this_gf[column] / this_gf.area


def get_base_grid(n, circle):
    """get_base_grid:

    input:
      n: h3 level
      circle: perimiter

    returns:
      GeoSeries hexagon point grid
    """
    circle_zone = circle.buffer(370.0e3)
    if n == 1:
        circle_zone = circle.buffer(600.0e3)
    return get_grid(n, circle_zone)


def get_oriented_hex(angle, base_grid, pivot, circle):
    """get_oriented_hx:

    input:
      angle:
      base_grid:
      privot:
      circle:

    returns:
      rotated GeoSeries hexagon grid clipped to circle boundary
    """
    r = get_hexagon(base_grid.rotate(float(angle), origin=pivot))
    return r.clip(circle).reset_index(drop=True)


def main():
    """main: collates GHS data for 'Europe'""

    returns:
      None
    """
    log("start GHS calculation")
    centre, circle = get_circle()
    write_dataframe(centre.to_frame("geometry"), OUTPATH, "centre")
    write_dataframe(circle, OUTPATH, "circle")
    ghs_point = get_ghs(circle, GHPOINT)
    ghs_shape = get_ghs(circle, GHSHAPE)
    outer = read_dataframe(INPATH, layer="outer")
    outer = outer.explode(index_parts=False).reset_index(drop=True)
    outer = outer.clip(circle).reset_index(drop=True)
    write_dataframe(outer, OUTPATH, "outer")
    geometry = outer["geometry"].values
    ix = ghs_point["geometry"].sindex.query(geometry, predicate="contains")
    ghs_point = ghs_point.loc[ix[1]].sort_index().reset_index(drop=True)
    write_dataframe(ghs_point, OUTPATH, "point")
    log("wrote GHS point")
    ix = ghs_shape["geometry"].centroid.sindex.query(geometry, predicate="contains")
    ghs_shape = ghs_shape.loc[ix[1]].sort_index().reset_index(drop=True)
    ghs_shape = ghs_shape.dropna(axis=1)
    ghs_shape["density"] = density(ghs_shape, "P15")
    write_dataframe(ghs_shape, OUTPATH, "shape")
    log("wrote GHS shape")
    ghs_centroid = ghs_shape.copy()
    ghs_centroid["geometry"] = ghs_shape.centroid
    write_dataframe(ghs_centroid, OUTPATH, "centroid")
    log("wrote GHS centroid")


if __name__ == "__main__":
    main()
