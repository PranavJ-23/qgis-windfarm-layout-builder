# -*- coding: utf-8 -*-
from qgis.PyQt import QtWidgets, QtCore
from qgis.core import QgsProject, QgsFeature
from ui_wind_farm_dialog import Ui_WFDialog
import asyncio, random
import numpy as np
import pandas as pd
import xarray as xr
import httpx
from io import BytesIO
from shapely.geometry import Point
from shapely.affinity import scale, rotate, translate
from shapely.ops import unary_union
import rtree
import geopandas as gpd
import plotly.express as px
import plotly.io as pio
from qgis.PyQt.QtCore import QThread, pyqtSignal

# -------------------- Wind Data Functions -------------------- #
async def fetch_wind_stats(client, lat, lon):
    try:
        MESO_TEMPLATE = "https://wps.neweuropeanwindatlas.eu/api/mesoscale-atlas/v1/get-data-point?latitude={lat}&longitude={lon}&height=75&variable=wind_speed_mean"
        MICRO_TEMPLATE = "https://wps.neweuropeanwindatlas.eu/api/microscale-atlas/v1/get-data-point?latitude={lat}&longitude={lon}&height=100&variable=weib_A_combined&variable=weib_k_combined"
        TS_WS_TEMPLATE = "https://wps.neweuropeanwindatlas.eu/api/mesoscale-ts/v1/get-data-point?latitude={lat}&longitude={lon}&height=75&variable=WS&dt_start=2000-01-01T00:00:00&dt_stop=2022-12-31T23:30:00"
        TS_WD_TEMPLATE = "https://wps.neweuropeanwindatlas.eu/api/mesoscale-ts/v1/get-data-point?latitude={lat}&longitude={lon}&height=75&variable=WD&dt_start=2000-01-01T00:00:00&dt_stop=2022-12-31T23:30:00"

        urls = [MESO_TEMPLATE, MICRO_TEMPLATE, TS_WS_TEMPLATE, TS_WD_TEMPLATE]
        urls = [url.format(lat=lat, lon=lon) for url in urls]

        resp_meso, resp_micro, resp_ts_ws, resp_ts_wd = await asyncio.gather(*(client.get(u) for u in urls))
        for r in [resp_meso, resp_micro, resp_ts_ws, resp_ts_wd]:
            r.raise_for_status()

        ds_meso = xr.open_dataset(BytesIO(resp_meso.content))
        mean_ws = float(ds_meso['wind_speed_mean'].values)

        ds_micro = xr.open_dataset(BytesIO(resp_micro.content))
        A = float(ds_micro['weib_A_combined'].values)
        k = float(ds_micro['weib_k_combined'].values)

        ds_ts_ws = xr.open_dataset(BytesIO(resp_ts_ws.content))
        ds_ts_wd = xr.open_dataset(BytesIO(resp_ts_wd.content))
        wind_speeds = ds_ts_ws['WS'][:, 0].values
        wind_dirs = ds_ts_wd['WD'][:, 0].values

        return {
            'latitude': lat, 'longitude': lon, 'mean_ws': mean_ws,
            'weibull_A': A, 'weibull_k': k,
            'time_series_ws': wind_speeds, 'time_series_wd': wind_dirs
        }
    except Exception as e:
        return {"error": str(e), "latitude": lat, "longitude": lon}

async def fetch_all_coords_list(coords):
    async with httpx.AsyncClient(timeout=240) as client:
        tasks = [fetch_wind_stats(client, lat, lon) for lat, lon in coords]
        results = await asyncio.gather(*tasks)
        return results

def dominant_wind_direction(wind_dirs):
    wind_dirs_rad = np.deg2rad(wind_dirs)
    sin_sum = np.sin(wind_dirs_rad).mean()
    cos_sum = np.cos(wind_dirs_rad).mean()
    angle_rad = np.arctan2(sin_sum, cos_sum)
    return np.rad2deg(angle_rad) % 360

def plot_wind_rose(item):
    wind_speeds = item['time_series_ws']
    wind_dirs = item['time_series_wd']
    valid = ~np.isnan(wind_speeds) & ~np.isnan(wind_dirs)
    wind_speeds = wind_speeds[valid]
    wind_dirs = wind_dirs[valid]

    wind_data = pd.DataFrame({'wind_speed': wind_speeds, 'wind_dir': wind_dirs})
    wind_data['dir_bin'] = (wind_data['wind_dir'] // 30 * 30).astype(int)
    speed_bins = [0,2,4,6,8,10,15,25]
    speed_labels = ["0-2","2-4","4-6","6-8","8-10","10-15","15-25"]
    wind_data['speed_bin'] = pd.cut(wind_data['wind_speed'], bins=speed_bins, labels=speed_labels)

    rose_data = wind_data.groupby(['dir_bin','speed_bin']).size().reset_index(name='frequency')
    total_count = rose_data['frequency'].sum()
    rose_data['percentage'] = (rose_data['frequency']/total_count)*100
    rose_data['hover_text'] = rose_data['percentage'].map(lambda x: f"{x:.2f}%")

    pio.renderers.default = "browser"
    fig = px.bar_polar(
        rose_data, r="percentage", theta="dir_bin", color="speed_bin",
        template="plotly_dark", color_discrete_sequence=px.colors.sequential.Plasma,
        title="Wind rose", custom_data=["hover_text"]
    )
    fig.update_traces(hovertemplate="%{customdata}")
    fig.show()

# -------------------- Ellipse Packing -------------------- #
def create_rotated_ellipse(center_x, center_y, width, height, angle_deg):
    ellipse = Point(0,0).buffer(1)
    ellipse = scale(ellipse, width/2, height/2)
    ellipse = rotate(ellipse, angle_deg, origin=(0,0))
    ellipse = translate(ellipse, xoff=center_x, yoff=center_y)
    return ellipse

def generate_hex_grid(poly, spacing):
    minx, miny, maxx, maxy = poly.bounds
    dx = spacing*np.sqrt(3)
    dy = spacing*1.5
    points=[]
    y = miny
    row = 0
    while y<maxy:
        x = minx + (spacing*np.sqrt(3)/2 if row%2 else 0)
        while x<maxx:
            pt = Point(x,y)
            if poly.contains(pt):
                points.append(pt)
            x += dx
        y += dy
        row += 1
    return points

def get_smart_edge_points(poly, hex_points, edge_dist=5):
    poly_clean = poly.buffer(0)
    inner_buffer = poly_clean.buffer(-edge_dist)
    edge_zone = poly_clean.difference(inner_buffer)
    return [p for p in hex_points if edge_zone.contains(p)]

def pack_ellipses_from_points(poly, points, width, height, angle_deg, progress_callback=None):
    ellipses=[]
    idx=rtree.index.Index()
    for i, pt in enumerate(points):
        candidate = create_rotated_ellipse(pt.x, pt.y, width, height, angle_deg)
        cand_bounds = candidate.bounds
        possible_overlaps = list(idx.intersection(cand_bounds))
        overlap = any(candidate.intersects(ellipses[j]) for j in possible_overlaps)
        if not overlap:
            ellipses.append(candidate)
            idx.insert(len(ellipses)-1, cand_bounds)
        if progress_callback and i % max(1,len(points)//50)==0:
            progress_callback.emit(int(i/len(points)*100))
    return ellipses

def multiple_iterations_edge_start_smart(poly, width, height, angle_deg, spacing, n_trials=10, edge_buffer=None, progress_callback=None):
    poly_union = unary_union(poly)
    hex_points = generate_hex_grid(poly_union, spacing)
    if edge_buffer is None:
        edge_buffer = spacing
    edge_points = get_smart_edge_points(poly_union, hex_points, edge_dist=edge_buffer)
    best_ellipses=[]
    best_count=0
    for i in range(n_trials):
        if progress_callback:
            progress_callback.emit(int(i/n_trials*100))
        start_point = random.choice(edge_points)
        start_idx = hex_points.index(start_point)
        trial_points = hex_points[start_idx:] + hex_points[:start_idx]
        ellipses = pack_ellipses_from_points(poly_union, trial_points, width, height, angle_deg, progress_callback)
        if len(ellipses) > best_count:
            best_count = len(ellipses)
            best_ellipses = ellipses
    if progress_callback:
        progress_callback.emit(100)
    return best_ellipses

# -------------------- Worker Thread -------------------- #
class PackEllipseThread(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(list)
    def __init__(self, poly, width, height, angle_deg, spacing, n_trials):
        super().__init__()
        self.poly = poly
        self.width = width
        self.height = height
        self.angle_deg = angle_deg
        self.spacing = spacing
        self.n_trials = n_trials
    def run(self):
        ellipses = multiple_iterations_edge_start_smart(
            self.poly, self.width, self.height, self.angle_deg,
            spacing=self.spacing, n_trials=self.n_trials,
            progress_callback=self.progress_signal
        )
        self.finished_signal.emit(ellipses)

# -------------------- Plugin Dialog -------------------- #
class WindFarmDialog(QtWidgets.QDialog, Ui_WFDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.results=[]
        self.main_dir=None
        self.poly=None
        self.coords=[]

        # Connect buttons
        self.init_wind_btn.clicked.connect(self.on_init_wind_data)
        self.genrate_layout_btn.clicked.connect(self.on_generate_layout)
        self.plot_wind_btn.clicked.connect(self.on_plot_wind_rose)

    def on_init_wind_data(self):
        # Fetch coordinates
        if self.use_point_layer_chk.isChecked():
            layer = self.coord_layer_combo.currentLayer()
            if not layer:
                QtWidgets.QMessageBox.warning(self, "No Layer", "Please select a point layer")
                return

            from qgis.core import QgsCoordinateTransform, QgsProject, QgsCoordinateReferenceSystem

            # Transform layer CRS to WGS84
            crs_src = layer.crs()
            crs_dest = QgsCoordinateReferenceSystem("EPSG:4326")  # WGS84
            transform = QgsCoordinateTransform(crs_src, crs_dest, QgsProject.instance().transformContext())

            # Only a single feature expected
            feature = next(layer.getFeatures(), None)
            if not feature:
                QtWidgets.QMessageBox.warning(self, "No Feature", "Layer has no features")
                return

            geom = feature.geometry()
            if geom.isEmpty():
                QtWidgets.QMessageBox.warning(self, "No Geometry", "Point geometry is empty")
                return

            pt = geom.asPoint()
            pt_wgs84 = transform.transform(pt)

            # Store as (latitude, longitude)
            self.coords = [(pt_wgs84.y(), pt_wgs84.x())]

        else:
            # Manual lat/lon input (unchanged)
            lat_text = self.lat_input.text()
            lon_text = self.lon_input.text()
            if not lat_text or not lon_text:
                QtWidgets.QMessageBox.warning(self, "No Input", "Enter lat/lon or select point layer")
                return
            self.coords = [(float(lat_text), float(lon_text))]

        # --- Fetch wind data synchronously for now ---
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.results = loop.run_until_complete(fetch_all_coords_list(self.coords))
        if "error" in self.results[0]:
            QtWidgets.QMessageBox.critical(self, "Error", self.results[0]["error"])
            return
        self.main_dir = dominant_wind_direction(self.results[0]['time_series_wd'])
        QtWidgets.QMessageBox.information(self, "Finished", "Wind data obtained and dominant wind direction calculated.")

    def on_generate_layout(self):
        # Load polygon
        layer = self.construction_layer_combo.currentLayer()
        if not layer:
            QtWidgets.QMessageBox.warning(self, "No Layer", "Please select a polygon layer")
            return
        self.poly = unary_union([f.geometry().asPolygon() for f in layer.getFeatures() if f.geometry().isGeosValid()])

        try:
            r = float(self.rotor_diameter_input.text())/2
        except:
            QtWidgets.QMessageBox.warning(self, "Error", "Invalid rotor diameter")
            return
        spacing = float(self.grid_spacing_input.text() or 10)
        width = r*4
        height = r*3
        n_trials = self.itterations_slider.value()

        # Start ellipse packing in background thread
        self.thread = PackEllipseThread(self.poly, width, height, self.main_dir, spacing, n_trials)
        self.thread.progress_signal.connect(self.progress_bar.setValue)
        self.thread.finished_signal.connect(self.on_layout_finished)
        self.thread.start()

    def on_layout_finished(self, ellipses):
        if not ellipses:
            QtWidgets.QMessageBox.warning(self, "Layout Failed", "No valid layout generated")
            return
        # Save turbine centers
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Turbine Layout", "", "Shapefiles (*.shp)")
        if not path:
            return
        centers = [e.centroid for e in ellipses]
        gdf_points = gpd.GeoDataFrame(geometry=centers, crs="EPSG:4326")
        gdf_points.to_file(path)
        msg = f"Turbine centers saved to {path}"

        # Optionally save ellipses
        if self.export_wakes_chk.isChecked():
            ellipse_path = path.replace(".shp", "_wakes.shp")
            gdf_ellipses = gpd.GeoDataFrame(geometry=ellipses, crs="EPSG:4326")
            gdf_ellipses.to_file(ellipse_path)
            msg += f"\nTurbine wakes saved to {ellipse_path}"
        QtWidgets.QMessageBox.information(self, "Saved", msg)

    def on_plot_wind_rose(self):
        if not self.results:
            QtWidgets.QMessageBox.warning(self, "No Data", "Please fetch wind data first")
            return
        plot_wind_rose(self.results[0])
