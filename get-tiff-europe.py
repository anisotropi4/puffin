#!/usr/bin/env python

import os.path
import warnings
from functools import partial
from itertools import islice

import geopandas as gp
import numpy as np
import pandas as pd
import rasterio as rio
import rasterio.features as rif
import rasterio.windows as riw
import requests
import shapely.geometry as sg
from bs4 import BeautifulSoup
from pyogrio import read_dataframe, write_dataframe
from shapely import set_precision, unary_union
from shapely.geometry import Point, Polygon

from shared.util import append_layer, log

pd.set_option("display.max_columns", None)

BOUNDARY = ["RUS", "KAZ"]

WGS84 = "EPSG:4326"
CRS = "EPSG:3034"
URLP = (
    "https://data.worldpop.org/GIS/Population/Global_2021_2022_1km_UNadj/"
    "unconstrained/2022"
)
URLB = "https://data.worldpop.org/GIS/Mastergrid/Global_2000_2020/"
URLE = (
    "https://en.wikipedia.org/wiki/"
    "List_of_sovereign_states_and_dependent_territories_in_Europe"
)

OUTPATH = "europa.gpkg"

set_precision_one = partial(set_precision, grid_size=1.0)


def get_ne_10m_countries():
    """get_countries:

    args:
       None

    returns:
       None

    """
    url = (
        "https://github.com/nvkelso/natural-earth-vector/blob/master"
        "/geojson/ne_10m_admin_0_countries.geojson?raw=true"
    )
    filename = "ne_10m_admin_0_countries.geojson"
    download_file(url, filename)


def get_ne_10m(iso_a3):
    """get_ne_10m:

    args:
      iso_a3: ISO A3 country-code

    returns:

    """
    get_ne_10m_countries()
    filename = "data/ne_10m_admin_0_countries.geojson"
    r = gp.read_file(filename)
    r = r.replace("Unrecognized", pd.NA).dropna(axis=1)
    r = r[r["ADM0_A3"].isin(iso_a3)].sort_values("ADM0_A3")
    return r.reset_index(drop=True)


def get_iso_a3():
    """get_iso_a3:

    args:
      None

    returns:
      pandas DataSeries
    """
    url = "https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3"
    r = requests.get(url, timeout=30)
    soup = BeautifulSoup(r.content, "html.parser")
    ds = pd.Series(name="name").rename_axis("ISO_A3")
    for span in soup.find_all("span", {"class": "monospaced"}):
        link = span.find_parent()
        v = link.find("a").get_text()
        k = span.get_text()
        s = link.find("sup")
        if s is not None:
            break
        ds.loc[k] = v
    return ds


def get_europe():
    """get_europe: grovel Wikipedia for European country names

    args:
      None

    returns:
      pandas Series

    """
    r = requests.get(URLE, timeout=30)
    soup = BeautifulSoup(r.content, "html.parser")
    td_cd = soup.find("td", {"class": "navbox-list-with-group"})
    return pd.Series([a.get_text() for a in td_cd.find_all("a")], name="name")


def get_all_europe():
    """get_all_europe: grovel Wikipedia for other European country names

    args:
      None

    returns:
      pandas Series

    """
    r = requests.get(URLE, timeout=30)
    soup = BeautifulSoup(r.content, "html.parser")
    table = soup.find_all("table", {"class": "wikitable sortable"})[0]
    tr_cd = table.find_all("tr")[1:]
    return pd.Series(
        [i.find_all("a", string=True)[0].get_text() for i in tr_cd], name="name"
    )


def get_europe_wiki_iso_a3():
    """get_europe_wiki_iso_a3:

    args:
      None

    returns:
      pandas Series

    """
    filepath = "data/europe-codes.csv"
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        tables = pd.read_html(URLE)
        df = tables[1].iloc[:, 2].str.rsplit(" ", n=1, expand=True)
        df.columns = ["name", "iso_a3"]
        df.to_csv("data/europe-codes.csv", index=False)
    return df["iso_a3"]


def get_europe_iso_a3():
    """get_europe_iso_a3:

    args:
      None

    returns:
      pandas Series

    """
    r = pd.concat([get_europe_wiki_iso_a3(), get_europe_xlsx_iso_a3()])
    return r.drop_duplicates().sort_values().reset_index(drop=True)


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


def get_worldpop_xlsx():
    """get_worldpop_xlsx:

    args:
      None

    returns:
      pandas DataFrame

    """
    uri = "https://www.worldpop.org/resources/docs/national_boundaries"
    filename = "global-national-level-input-population-data-summary-rev11-ajt.xlsx"
    url = f"{uri}/{filename}"
    download_file(url, filename)
    tab = "Input Population Data Summary"
    filepath = f"data/{filename}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.read_excel(filepath, sheet_name=tab, header=None)
    df.columns = df.loc[1]
    return df[2:].reset_index(drop=True)


def get_europe_xlsx_iso_a3():
    """get_europe_xlsx_iso_a3:"""
    r = get_worldpop_xlsx()
    ix = r["Continent"] == "Europe"
    r = r.loc[ix, "ISOAlpha"].rename("iso_a3")
    return r.reset_index(drop=True)


def read_file(filepath, crs=WGS84):
    """read_file:

    args:
      filepath:
      crs:  (default value = WGS84)

    returns:
      GeoDataFrame

    """
    columns = ["ISO_A3", "longitude", "latitude", "population", "geometry"]
    data = pd.read_csv(f"{filepath}", header=None, names=columns)
    geometry = gp.GeoSeries.from_wkt(data["geometry"])
    return gp.GeoDataFrame(data=data, geometry=geometry).set_crs(crs)


def get_simple_rectangle(p):
    """get_simple_rectangle:

    args:
      p: rectangle as offset

    returns:
      numpy array
    """
    x1, y1 = np.asarray(p) - np.asarray([0.5, 0.5])
    x2, y2 = np.asarray(p) + np.asarray([0.5, 0.5])
    return np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])


def get_rectangle(this_array):
    """get_rectangle:

    args:
      this_array:

    returns:

    """
    return Polygon(this_array.reshape(4, 2))


def scale_points(points, dataset):
    """scale_points:

    args:
      points:
      dataset:

    returns:
      numpy array

    """
    rows, columns = dataset.shape
    x1, y1, x2, y2 = dataset.bounds
    x_value = x1 + points[:, 0] * (x2 - x1) / columns
    y_value = y2 - points[:, 1] * (y2 - y1) / rows
    return np.stack([x_value, y_value], axis=1)


def scale_rectangle(this_rectangle, dataset):
    """scale_rectangle:

    args:
      this_rectangle:
      dataset:

    returns:

    """
    r = scale_points(this_rectangle.reshape(-1, 2), dataset)
    return r.reshape(-1, 4, 2)


def get_block_shape(v):
    """get_block_shape:

    args:
      v:

    returns:

    """
    return sg.shape(v[0]), v[1]


def get_raster_frame(filepath):
    """get_raster_frame:

    args:
      filepath: raster filepath

    returns:
      pandas DataFrame

    """
    dataset = rio.open(filepath)
    crs = dataset.crs
    data = dataset.read(1)
    rows, columns = np.shape(data)
    x_value, y_value = np.meshgrid(range(columns), range(rows))
    x_value, y_value = x_value.reshape(-1), y_value.reshape(-1)
    ix = data.reshape(-1) > 0.0
    points = np.stack([x_value[ix], y_value[ix]], axis=1)
    r = np.apply_along_axis(get_simple_rectangle, 1, points)
    r = scale_rectangle(r, dataset)
    r = np.apply_along_axis(get_rectangle, 1, r.reshape(-1, 8))
    r = gp.GeoSeries(r, crs=crs).to_frame("geometry")
    r["data"] = data.reshape(-1)[ix]
    return r


def get_circle():
    def _set_circle():
        centre = gp.GeoSeries(Point(4500000, 3550000), crs=CRS)
        r = gp.GeoSeries(centre.buffer(_radius), crs=CRS)
        if not append_layer(OUTPATH, "circle"):
            write_dataframe(r.to_frame("geometry"), OUTPATH, layer="circle")
            write_dataframe(centre.to_frame("geometry"), OUTPATH, layer="centre")
        return r.to_crs(WGS84)

    _radius = 3200.0e3
    _circle = _set_circle()
    return _circle.values


def get_european_boundary():
    """get_euro_boundary:"""

    def _get_outer_boundary():
        """get_europe_boundary:

        return:
          None

        """
        r = read_dataframe(OUTPATH, layer="outer")
        return r.to_crs(WGS84)

    def _get_euro_boundary(this_gf):
        """_get_euro_boundary:

        args:
          this_gf: GeoDataFrame

        returns:

        """
        index = this_gf.index
        geometry = this_gf["geometry"].to_crs(CRS)
        ix = _outer.sindex.query(geometry)[0, :]
        r = this_gf.loc[index[ix]]
        if r.empty:
            return r
        ix = geometry[index[ix]].distance(_centre) < _radius
        return r.loc[ix]

    _circle = get_circle().to_crs(CRS)
    _radius = _circle.minimum_bounding_radius()[0]
    _centre = _circle.centroid[0]
    _outer = gp.GeoSeries(_circle)

    return _get_euro_boundary


def buffer_layer(this_gf, outfile, layer, crs):
    """buffer_layer:

    args:
      this_gf:
      outfile:
      layer:
      crs:

    returns:
      None

    """
    append = append_layer(outfile, layer)
    write_dataframe(this_gf.to_crs(crs), outfile, layer=layer, append=append)


def get_block(shape, get_boundary, layer, outfile, crs):
    """get_block:

    args:
      shape:
      layer:
      outfile:
      crs:
      n:

    returns:
      pandas DataFrame

    """
    m = 0
    r = []
    while block := map(get_block_shape, islice(shape, 1048576)):
        s = gp.GeoDataFrame(block, columns=["geometry", "data"], crs=crs)
        if s.empty:
            break
        if layer:
            s = get_boundary(s)
            if s.empty:
                log(f"\t{str(m).zfill(4)}\tx")
                break
            log(f"\t{str(m).zfill(4)}\t+")
        m = m + 1
        if layer == "population":
            buffer_layer(s, outfile, layer, crs)
            continue
        r.append(s)
    if len(r) == 0:
        return gp.GeoDataFrame(r, columns=["geometry", "data"], crs=crs)
    return pd.concat(r, copy=False)


def get_raster_shape(dataset, window):
    """get_raster_shape:

    args:
      dataset:
      window:

    returns:
      rif shape

    """

    data = dataset.read(1, window=window)
    mask = (data != 9999) & (data > 0.0)
    transform = dataset.window_transform(window)
    return rif.shapes(data, mask=mask, transform=transform)


def get_large_raster_frame(filepath, outfile=None, layer=False):
    """get_large_raster_frame:

    args:
      filepath:
      outfile:  (Default value = None)
      layer:  (Default value = False)

    returns:
      pandas DataFrame

    """
    dataset = rio.open(filepath)
    crs = dataset.crs
    # transform = dataset.transform
    get_boundary = get_european_boundary()
    r = None
    if layer is not None:
        circle_window = get_circle()
        circle_window = rif.geometry_window(dataset, circle_window)
    for n, (_, window) in enumerate(dataset.block_windows(1)):
        log(f"{str(n).zfill(6)}\t")
        if layer:
            try:
                riw.intersection(window, circle_window)
            except rio.errors.WindowError:
                continue
        # data = dataset.read(1, window=window)
        # mask = (data != 9999) & (data > 0.0)
        # transform = dataset.window_transform(window)
        # shape = rif.shapes(data, mask=mask, transform=transform)
        shape = get_raster_shape(dataset, window)
        s = get_block(shape, get_boundary, layer, outfile, crs)
        if not s.empty:
            r = pd.concat([r, s], copy=False)
    if layer == "population":
        return None
    return r.reset_index(drop=True)


def download_boundary(code):
    """download_boundary:

    args:
      code:

    returns:
      filename string

    """
    uri = f"{URLB}/{code}/L0/{code.lower()}_level0_100m_2000_2020.tif"
    filename = uri.split("/")[-1]
    if download_file(uri, filename):
        log(f"download {filename}")
    return filename


def set_boundary(outfile, code):
    """set_boundary:

    args:
      outfile:
      code:

    returns:
       None

    """
    layer = "boundary"
    if append_layer(outfile, layer):
        return
    filename = download_boundary(code)
    filepath = f"data/{filename}"
    if code in BOUNDARY:
        r = get_large_raster_frame(filepath, outfile, layer)
    else:
        r = get_large_raster_frame(filepath)
    geometry = (
        r["geometry"].to_crs(CRS).buffer(2.0, cap_style="square", join_style="bevel")
    )
    geometry = unary_union(geometry.values)
    try:
        geometry = geometry.geoms
    except AttributeError:
        pass
    r = gp.GeoSeries(geometry, crs=CRS).to_frame("geometry")
    r["code"] = code
    write_dataframe(r, outfile, layer="boundary")


def set_outer_boundary(outpath, iso_a3):
    """set_outer_boundary:

    args:
      outpath:
      iso_a3:

    returns:

    """
    layer = "outer"
    if append_layer(outpath, layer):
        return read_dataframe(outpath, layer=layer)
    layer = "boundary"
    if not append_layer(outpath, layer):
        append = False
        for a3_code in iso_a3:
            log(f"{a3_code}\tboundary")
            inpath = f"output/{a3_code}-population-r.gpkg"
            boundary = read_dataframe(inpath, layer=layer)
            boundary["geometry"] = boundary["geometry"].apply(set_precision_one)
            boundary["a3_code"] = a3_code
            write_dataframe(boundary, outpath, layer, append=append)
            append = True
    log("outer boundary")
    layer = "outer"
    if not append_layer(outpath, layer):
        r = read_dataframe(outpath, "boundary")
        outer = unary_union(r["geometry"].values)
        outer = outer.simplify(100.0)
        outer = gp.GeoSeries(outer, crs=CRS)
        outer = outer.explode(index_parts=False).reset_index(drop=True)
        ix = outer.area > 1.0e6
        outer = unary_union(outer.loc[ix].values)
        outer = gp.GeoSeries(outer, crs=CRS).to_frame("geometry")
        write_dataframe(outer, outpath, layer=layer)
    log("wrote boundary")
    return outer


def download_population(code):
    """download_population:

    args:
      code:

    returns:
      filename string

    """
    uri = f"{URLP}/{code}/{code.lower()}_ppp_2022_1km_UNadj.tif"
    filename = uri.split("/")[-1]
    if code == "VAT":
        filename = "vat_ppp_2020_1km_Aggregated_UNadj.tif"
        uri = f"{URLP}/{code}/{filename}".replace("2022", "2020")
        uri = uri.replace("2021", "2000").replace("unconstrained/", "")
    try:
        if download_file(uri, filename):
            log(f"download {filename}")
    except requests.exceptions.HTTPError:
        return None
    return filename


def get_population(code, outfile):
    """get_population:

    args:
      code:
      outfile:

    returns:
      panda DataFrame

    """
    filename = download_population(code)
    if filename is None:
        return gp.GeoDataFrame()
    filepath = f"data/{filename}"
    layer = "population"
    if append_layer(outfile, "population"):
        return read_dataframe(outfile, "population")
    if code in BOUNDARY:
        get_large_raster_frame(filepath, outfile, layer)
        r = read_dataframe(outfile, layer=layer)
    else:
        r = get_large_raster_frame(filepath)
    count = np.ceil(r.shape[0] / 262144)
    r["code"] = code
    for ix in np.array_split(r.index, count):
        geometry = r.loc[ix, "geometry"]
        r.loc[ix, "geometry"] = geometry.to_crs(CRS)
    r = r.set_crs(CRS, allow_override=True)
    r = r.rename(columns={"data": "population"})
    r["density"] = r["population"] * 1.0e6 / r.area
    return r


def set_population(outpath, r, code):
    """set_population:

    args:
      outpath:
      r:
      code:

    returns:
      None

    """
    layer = "population"
    append = append_layer(outpath, layer)
    log(f"simplify population {code}")
    count = int(np.ceil(r.shape[0] / 262144))
    for i, ix in enumerate(np.array_split(r.index, count)):
        log(f"{code}\t{str(i+1).zfill(2)} of {str(count).zfill(2)}\t")
        geometry = r.loc[ix, "geometry"]
        r.loc[ix, "geometry"] = geometry.apply(set_precision_one)
    log(f"write population {code}")
    write_dataframe(r, outpath, layer, append=append)
    log(f"wrote population {code}")


def reset_population(outpath, iso_a3):
    """reset_population:

    args:
      outpath:
      iso_a3:

    returns:
       None

    """
    try:
        os.remove(outpath)
    except FileNotFoundError:
        pass
    for code in iso_a3:
        inpath = f"output/{code}-population-r.gpkg"
        append = append_layer(inpath, "population")
        if append:
            r = read_dataframe(inpath, layer="population")
            set_population(outpath, r, code)


def main():
    """main: download europe boundary and population data

    args:
      None

    returns:
      None
    """
    # country = get_iso_a3()
    # europe = country.loc[europe_code].reset_index()
    # europe["short_name"] = get_europe().sort_values()
    europe_code = get_europe_iso_a3().sort_values().reset_index(drop=True)
    europe_ne = get_ne_10m(europe_code).set_index("ADM0_A3")
    europe_xlsx = get_worldpop_xlsx().set_index("ISOAlpha")
    log("start\tpopulation")
    get_european_boundary()
    try:
        os.mkdir("output")
    except FileExistsError:
        pass
    for code in europe_code.sort_values():
        if code in europe_ne["SOVEREIGNT"]:
            name = europe_ne.loc[code, "SOVEREIGNT"]
        else:
            name = europe_xlsx.loc[code, "Country or Territory Name"]
        log(f"{code}\t{name}")
        outfile = f"output/{code}-population-r.gpkg"
        log(f"{code}\tboundary")
        set_boundary(outfile, code)
        log(f"{code}\tpopulation")
        population = get_population(code, outfile)
        if not population.empty:
            write_dataframe(population, outfile, layer="population")
            set_population(OUTPATH, population, code)
    set_outer_boundary(OUTPATH, europe_code)

    log("end\tpopulation")


if __name__ == "__main__":
    main()
