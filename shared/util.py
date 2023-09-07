#!/usr/bin/env python
"""util: shared utilities used in puffin"""

import datetime as dt
import os
import os.path
from functools import partial

import geopandas as gp
import h3
import numpy as np
import pandas as pd
import requests
from fiona import listlayers
from pyogrio import read_dataframe, write_dataframe
from pyogrio.errors import DataLayerError, DataSourceError
from shapely import set_precision, unary_union, voronoi_polygons
from shapely.geometry import MultiPoint
from tobler.area_weighted import area_interpolate
from tobler.util import h3fy

pd.set_option("display.max_columns", None)

BOUNDARY = ["RUS", "KAZ"]

WGS84 = "EPSG:4326"
CRS = "EPSG:3034"
START = dt.datetime.now()
URLP = (
    "https://data.worldpop.org/GIS/Population/Global_2021_2022_1km_UNadj/"
    "unconstrained/2022"
)
URLB = "https://data.worldpop.org/GIS/Mastergrid/Global_2000_2020/"
URLE = (
    "https://en.wikipedia.org/wiki/"
    "List_of_sovereign_states_and_dependent_territories_in_Europe"
)

ISO_A3 = (
    "ALA,ALB,AND,ARM,AUT,AZE,BEL,BGR,BIH,BLR,CHE,CYP,CZE,DEU,DNK,ESP,EST,FIN,FRA,FRO,GBR,GEO,GGY,"
    "GIB,GRC,HRV,HUN,IMN,IRL,ISL,ITA,JEY,KAZ,KOS,LIE,LTU,LUX,LVA,MCO,MDA,MKD,MLT,MNE,NLD,NOR,POL,"
    "PRT,ROU,RUS,SJM,SMR,SRB,SVK,SVN,SWE,TUR,UKR,VAT"
).split(",")


set_precision_one = partial(set_precision, grid_size=1.0)


def append_layer(filepath, layer):
    """append_layer: return True is layer is in filepath"""
    if os.path.isfile(filepath):
        return layer in listlayers(filepath)
    return False


def archive(filepath):
    """archive: create archive directory if doesn't exist and move file to directory"""
    filename = os.path.basename(filepath)
    try:
        os.mkdir("archive")
    except FileExistsError:
        pass
    outpath = f"archive/{filename}"
    if os.path.isfile(outpath):
        raise FileExistsError(outpath)
    os.rename(filepath, outpath)


def density(this_gf, column="population"):
    """density: from GeoSeries 'this_gf' return km2 density of 'column'

    args:
      this_gf: GeoDataFrame:
      column: str:  (Default value = "population")
      this_gf: GeoDataFrame:
      column: str:  (Default value = "population")

    returns:

    """
    return 1.0e6 * this_gf[column] / this_gf.area


def download_file(url, filename):
    """download_file:

    args:
      url:
      filename:

    returns:
      return boolean

    """
    filepath = f"data/{filename}"
    if os.path.isfile(filepath):
        return False
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(filepath, "wb") as fout:
            for n, chunk in enumerate(r.iter_content(chunk_size=16777216)):
                log(str(n + 1).zfill(4))
                fout.write(chunk)
    return True


def filter_hexagon(hexagon, outer):
    """filter_hexgon:

    :param hexagon:
    :param outer:

    """
    distance = np.max(hexagon.length / np.sqrt(3) / 3)
    _, ix = hexagon.sindex.nearest(outer, max_distance=distance)
    r = hexagon.loc[np.unique(ix)].to_frame("geometry")
    r["area"] = r.area
    r[["population", "density"]] = 0.0, 0.0
    return r


def log(this_string):
    """log:

    :param this_string: str:

    """
    now = dt.datetime.now() - START
    print(this_string + f"\t{now}")


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


def get_boundary(this_gf):
    """get_boundary:

    :param this_gf:

    """
    r = simplify_outer(this_gf)
    try:
        return gp.GeoSeries(r.geoms, crs=CRS)
    except AttributeError:
        return gp.GeoSeries(r, crs=CRS)


def get_box(this_gf, this_buffer=0.0):
    """get_box:

    :param this_gf: param this_buffer:  (Default value = 0.0)

    """
    r = [this_gf.bounds.iloc[:, :2].min(), this_gf.bounds.iloc[:, 2:].max()]
    r = pd.concat(r)
    r.iloc[:2] = r.min()
    r.iloc[2:] = r.max()
    r.iloc[:2] -= this_buffer
    r.iloc[2:] += this_buffer
    return r


def get_circle(inpath):
    """get_circle:

    returns:
      circle GeoDataFrame centre Point and Polygon
    """
    circle = read_dataframe(inpath, layer="circle")
    centre = circle.centroid
    return centre, circle


def get_grid(n, circle):
    """get_grid: return sufficient grid points to allow construction of hexagons"
    "in two orientations to cover circle

    args:
      n: hexagon resolution
      circle: boundary circle

    returns:
      point GeoSeries

    """
    hex_id = get_hex_id(n, circle)
    r = gp.GeoSeries(get_hx_centre(hex_id), crs=CRS)
    return r.apply(set_precision_one)


def get_hexagon(grid):
    """get_hexagon:

    :param grid:

    """
    r = voronoi_polygons(MultiPoint(grid.values))
    r = gp.GeoSeries(r.geoms, crs=CRS)
    ix = r.type == "Polygon"
    return r[ix].apply(set_precision_one)


def get_hex_id(n, this_boundary):
    """get_hex_id:

    :param n: param this_buffer:

    """
    return h3fy(this_boundary.to_crs(WGS84), n, return_geoms=False)


def get_hside(resolution):
    """get_hside:

    args:
      h3 hexagon resolution

    returns:
      hexagon side length

    """
    # derived from avg_area_table.ipynb at https://github.com/uber/h3-py-notebooks
    hareas = (
        "4.357449416078392e12,6.097884417941342e11,8.680178039899734e10,"
        "1.239343465508818e10,1.770347654491310e09,2.529038581819453e08,"
        "3.612906216441251e07,5.161293359717200e06,7.373275975944190e05,"
        "1.053325134272069e05,1.504750190766437e04,2.149643129451882e03,"
        "3.070918756316065e02,4.387026794728303e01,6.267181135324324e00,"
        "8.953115907605805e-01"
    ).split(",")
    area = float(hareas[resolution])
    return round(np.sqrt(2.0 * area / 3.0 / np.sqrt(3.0)), 1)


def get_hx_centre(hex_id):
    """

    :param hex_id:

    """
    r = hex_id.map(h3.h3_to_geo)
    r = np.stack(r.map(np.asarray))
    return (
        gp.GeoSeries(gp.points_from_xy(*r[:, [1, 0]].T), crs=WGS84).to_crs(CRS).values
    )


def get_hx_centre2(hex_id):
    """

    :param hex_id:

    """
    r = hex_id.map(h3.h3_to_geo).apply(pd.Series)
    return gp.points_from_xy(*r.iloc[:, [1, 0]].values.T, crs=WGS84).to_crs(CRS)


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
    r = get_hexagon(base_grid.rotate(angle, origin=pivot))
    r = r.clip(circle).explode(index_parts=False)
    ix = r.type == "Polygon"
    return r[ix].reset_index(drop=True)


def get_population(this_hx, inpath):
    """

    :param this_hx:
    :param inpath:

    """
    bbox = tuple(this_hx.bounds)
    column = "population"
    population = read_dataframe(inpath, layer=column, bbox=bbox)
    hexagon = gp.GeoSeries(this_hx, crs=CRS).to_frame("geometry")
    r = area_interpolate(
        population, hexagon, allocate_total=False, extensive_variables=[column]
    )
    return r[column].values[0]


def get_xy(p):
    """get_xy:

    input:
      p: Point at coordinate (x, y)

    returns:
      x, y coordinates tuple
    """
    r = p.centroid
    return (r.x, r.y)


def get_yx(p):
    """get_yx:

    input:
      p: Point at coordinate (x, y)

    returns:
      y, x coordinates tuple
    """
    r = p.centroid
    return (r.y, r.x)


def set_pivot(outpath, centre_point, grid):
    """

    :param centre_point: param grid:
    :param grid:

    """
    try:
        r = read_dataframe(outpath, layer="pivot")
        return r["geometry"].values[0]
    except (DataLayerError, DataSourceError):
        pass
    ix = centre_point.to_frame("geometry")
    ix = ix.sjoin_nearest(grid.to_frame("geometry"))
    ix = ix["index_right"].values[0]
    r = grid.loc[ix]
    r = gp.GeoSeries(r, crs=CRS).to_frame("geometry")
    write_dataframe(r, outpath, "pivot")
    return grid.loc[ix]


def simplify_outer(this_gf):
    """

    :param this_gf: GeoDataFrame
    :rvalue MultiPolygon or Polygon

    """
    r = this_gf["geometry"].apply(set_precision_one)
    r = unary_union(r.values)
    r = r.simplify(100.0)
    return r
