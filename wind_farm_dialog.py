# -*- coding: utf-8 -*-
from qgis.PyQt import QtWidgets, QtCore, QtGui
from qgis.core import (
    QgsProject, QgsFeature, QgsVectorLayer, QgsGeometry, QgsFeatureRequest,
    QgsVectorFileWriter, QgsCoordinateReferenceSystem, QgsCoordinateTransform,
    QgsSpatialIndex, QgsPointXY, QgsFields, QgsField, QgsMarkerSymbol, QgsDiagramRenderer, QgsWkbTypes
)
from qgis.PyQt.QtGui import QColor, QImage, QPainter, QPen, QBrush, QFont, QPixmap
from qgis.PyQt.QtCore import QVariant, QThread, pyqtSignal, QSize, Qt, QPointF
from ui_wind_farm_dialog import Ui_WFDialog
import math
import random
import asyncio
import httpx
import xarray as xr
from io import BytesIO
import numpy as np

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
    sin_sum = sum(math.sin(math.radians(d)) for d in wind_dirs)/len(wind_dirs)
    cos_sum = sum(math.cos(math.radians(d)) for d in wind_dirs)/len(wind_dirs)
    angle_rad = math.atan2(sin_sum, cos_sum)
    return math.degrees(angle_rad) % 360

def bin_wind_data(wind_speeds, wind_dirs, n_dir_bins=12, speed_bins=[0,2,4,6,8,10,15,25]):
    dir_bin_width = 360 / n_dir_bins
    dir_bins = (wind_dirs // dir_bin_width).astype(int)
    
    n_speed_bins = len(speed_bins) - 1
    rose_data = np.zeros((n_dir_bins, n_speed_bins))
    
    for i in range(len(wind_speeds)):
        dir_bin = dir_bins[i]
        if not (0 <= dir_bin < n_dir_bins):
            continue
            
        speed = wind_speeds[i]
        speed_bin = -1
        for j in range(n_speed_bins):
            if speed_bins[j] < speed <= speed_bins[j+1]:
                speed_bin = j
                break
        
        if speed_bin != -1:
            rose_data[dir_bin, speed_bin] += 1
            
    total_count = np.sum(rose_data)
    if total_count > 0:
        rose_data = (rose_data / total_count) * 100
        
    return rose_data

def create_wedge_polygon(center, inner_radius, outer_radius, start_angle_deg, end_angle_deg, n_points=10):
    poly = QtGui.QPolygonF()
    
    # Outer arc
    for i in range(n_points + 1):
        angle_deg = start_angle_deg + (end_angle_deg - start_angle_deg) * i / n_points
        angle_rad = math.radians(angle_deg)
        poly.append(QPointF(center.x() + outer_radius * math.cos(angle_rad), 
                            center.y() + outer_radius * math.sin(angle_rad)))
        
    # Inner arc (in reverse)
    for i in range(n_points + 1):
        angle_deg = end_angle_deg - (end_angle_deg - start_angle_deg) * i / n_points
        angle_rad = math.radians(angle_deg)
        poly.append(QPointF(center.x() + inner_radius * math.cos(angle_rad), 
                            center.y() + inner_radius * math.sin(angle_rad)))
        
    return poly

def create_wind_rose_image(rose_data, speed_labels, size=600):
    image = QImage(QSize(size, size), QImage.Format_ARGB32)
    image.fill(Qt.white)
    
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing)
    
    center = QPointF(size / 2, size / 2)
    max_radius = size / 2 * 0.8
    
    n_dir_bins, n_speed_bins = rose_data.shape
    dir_bin_width_deg = 360 / n_dir_bins
    
    max_percentage_sum = np.max(np.sum(rose_data, axis=1))
    if max_percentage_sum == 0: max_percentage_sum = 1

    colors = ['#440154', '#482878', '#3e4989', '#31688e', '#26828e', '#1f9e89', '#35b779', '#6ece58', '#b5de2b', '#fde725']
    
    for dir_bin in range(n_dir_bins):
        angle_center_deg = dir_bin * dir_bin_width_deg - 90
        angle_start_deg = angle_center_deg - dir_bin_width_deg / 2
        angle_end_deg = angle_center_deg + dir_bin_width_deg / 2
        
        current_radius = 0
        for speed_bin in range(n_speed_bins):
            percentage = rose_data[dir_bin, speed_bin]
            if percentage == 0:
                continue
            
            outer_radius = current_radius + (percentage / max_percentage_sum) * max_radius
            
            wedge_poly = create_wedge_polygon(center, current_radius, outer_radius, angle_start_deg, angle_end_deg)
            
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(colors[speed_bin % len(colors)]))
            painter.drawPolygon(wedge_poly)
            
            current_radius = outer_radius

    painter.setFont(QFont("Arial", 10))
    painter.setPen(QColor("black"))
    legend_x = 10
    legend_y = 10
    for i, label in enumerate(speed_labels):
        painter.setBrush(QColor(colors[i % len(colors)]))
        painter.setPen(Qt.NoPen)
        painter.drawRect(legend_x, legend_y, 10, 10)
        
        painter.setPen(QColor("black"))
        painter.drawText(legend_x + 15, legend_y + 10, label)
        legend_y += 15

    painter.end()
    return image

class ImageDialog(QtWidgets.QDialog):
    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Wind Rose")
        layout = QtWidgets.QVBoxLayout(self)
        
        self.label = QtWidgets.QLabel()
        self.label.setPixmap(QPixmap.fromImage(image))
        layout.addWidget(self.label)
        
        self.save_button = QtWidgets.QPushButton("Save as JPG")
        self.save_button.clicked.connect(self.save_image)
        layout.addWidget(self.save_button)
        
        self.image = image

    def save_image(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Image", "", "JPEG Image (*.jpg)")
        if path:
            self.image.save(path, "JPG", 95)

# -------------------- Ellipse Packing (PyQGIS) -------------------- #
def create_rotated_ellipse(center_x, center_y, width, height, angle_deg, n_points=36):
    points = []
    angle_rad = math.radians(angle_deg)
    for i in range(n_points):
        theta = 2*math.pi*i/n_points
        x = (width/2)*math.cos(theta)
        y = (height/2)*math.sin(theta)
        xr = x*math.cos(angle_rad) - y*math.sin(angle_rad)
        yr = x*math.sin(angle_rad) + y*math.cos(angle_rad)
        points.append(QgsPointXY(center_x + xr, center_y + yr))
    points.append(points[0])
    return QgsGeometry.fromPolygonXY([points])

def generate_hex_grid(poly, spacing):
    bbox = poly.boundingBox()
    minx, miny, maxx, maxy = bbox.xMinimum(), bbox.yMinimum(), bbox.xMaximum(), bbox.yMaximum()
    dx = spacing*math.sqrt(3)
    dy = spacing*1.5
    points=[]
    y = miny
    row = 0
    while y < maxy:
        x = minx + (spacing*math.sqrt(3)/2 if row%2 else 0)
        while x < maxx:
            pt = QgsPointXY(x, y)
            if poly.contains(QgsGeometry.fromPointXY(pt)):
                points.append(pt)
            x += dx
        y += dy
        row += 1
    return points

def get_smart_edge_points(poly, hex_points, edge_dist=5):
    inner_buffer = poly.buffer(-edge_dist, 5)
    edge_zone = poly.difference(inner_buffer)
    return [p for p in hex_points if edge_zone.contains(QgsGeometry.fromPointXY(p))]

def pack_ellipses_from_points(poly, points, width, height, angle_deg):
    ellipses = []
    idx = QgsSpatialIndex()
    for pt in points:
        candidate = create_rotated_ellipse(pt.x(), pt.y(), width, height, angle_deg)
        
        possible_overlaps_ids = idx.intersects(candidate.boundingBox())
        
        overlap = False
        for i in possible_overlaps_ids:
            if candidate.intersects(ellipses[i]):
                overlap = True
                break
        
        if not overlap:
            feature = QgsFeature()
            feature.setId(len(ellipses))
            feature.setGeometry(candidate)
            idx.insertFeature(feature)
            ellipses.append(candidate)
            
    return ellipses

def multiple_iterations_edge_start_smart(poly, width, height, angle_deg, spacing, n_trials=10, edge_buffer=None, progress_callback=None):
    if edge_buffer is None:
        edge_buffer = spacing
    hex_points = generate_hex_grid(poly, spacing)
    
    best_ellipses = []
    best_count = 0

    strategies = ['random', 'bottom_up', 'top_down', 'left_right', 'right_left']
    
    num_trials_per_strategy = n_trials // len(strategies)
    if num_trials_per_strategy == 0:
        num_trials_per_strategy = 1

    total_trials = num_trials_per_strategy * len(strategies)
    completed_trials = 0

    for strategy in strategies:
        for i in range(num_trials_per_strategy):
            if strategy == 'random':
                random.shuffle(hex_points)
                trial_points = hex_points
            elif strategy == 'bottom_up':
                trial_points = sorted(hex_points, key=lambda p: p.y())
            elif strategy == 'top_down':
                trial_points = sorted(hex_points, key=lambda p: p.y(), reverse=True)
            elif strategy == 'left_right':
                trial_points = sorted(hex_points, key=lambda p: p.x())
            elif strategy == 'right_left':
                trial_points = sorted(hex_points, key=lambda p: p.x(), reverse=True)

            ellipses = pack_ellipses_from_points(poly, trial_points, width, height, angle_deg)
            if len(ellipses) > best_count:
                best_count = len(ellipses)
                best_ellipses = ellipses

            completed_trials += 1
            if progress_callback:
                progress_callback.emit(int(completed_trials / total_trials * 100))

    if progress_callback:
        progress_callback.emit(100)
        
    print(f"Generated {len(best_ellipses)} ellipses")
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
        ellipses = multiple_iterations_edge_start_smart(self.poly, self.width, self.height, self.angle_deg,
                                                        spacing=self.spacing, n_trials=self.n_trials,
                                                        progress_callback=self.progress_signal)
        self.finished_signal.emit(ellipses)

# -------------------- Plugin Dialog -------------------- #
class WindFarmDialog(QtWidgets.QDialog, Ui_WFDialog):
    def __init__(self, iface=None, parent=None):
        super().__init__(parent)
        self.iface = iface
        self.setupUi(self)
        self.results=[]
        self.main_dir=0
        self.poly=None
        self.coords=[]
        self.init_wind_btn.clicked.connect(self.on_init_wind_data)
        self.genrate_layout_btn.clicked.connect(self.on_generate_layout)
        self.plot_wind_btn.clicked.connect(self.on_plot_wind_rose)

    def on_init_wind_data(self):
        if self.use_point_layer_chk.isChecked():
            layer = self.coord_layer_combo.currentLayer()
            if not layer:
                QtWidgets.QMessageBox.warning(self, "No Layer", "Please select a point layer")
                return
            crs_src = layer.crs()
            crs_dest = QgsCoordinateReferenceSystem("EPSG:4326")
            transform = QgsCoordinateTransform(crs_src, crs_dest, QgsProject.instance().transformContext())
            feature = next(layer.getFeatures(), None)
            pt = feature.geometry().asPoint()
            pt_wgs84 = transform.transform(pt)
            self.coords = [(pt_wgs84.y(), pt_wgs84.x())]
        else:
            lat_text = self.lat_input.text()
            lon_text = self.lon_input.text()
            if not lat_text or not lon_text:
                QtWidgets.QMessageBox.warning(self, "No Input", "Enter lat/lon or select point layer")
                return
            self.coords = [(float(lat_text), float(lon_text))]
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.results = loop.run_until_complete(fetch_all_coords_list(self.coords))

        if "error" in self.results[0]:
            QtWidgets.QMessageBox.warning(self, "API Error", f"Could not fetch wind data: {self.results[0]['error']}. Using default wind direction of {self.main_dir} degrees.")
            return
        
        self.main_dir = dominant_wind_direction(self.results[0]['time_series_wd'])
        QtWidgets.QMessageBox.information(self, "Finished", "Wind data obtained and dominant wind direction calculated.")

    def on_generate_layout(self):
        layer = self.construction_layer_combo.currentLayer()
        if not layer:
            QtWidgets.QMessageBox.warning(self, "No Layer", "Please select a polygon layer")
            return
        features = [f for f in layer.getFeatures() if f.geometry() and f.geometry().isGeosValid()]
        geom_union = QgsGeometry.unaryUnion([f.geometry() for f in features])
        self.poly = geom_union
        try:
            r = float(self.rotor_diameter_input.text()) / 2
            spacing_text = self.grid_spacing_input.text()
            spacing = float(spacing_text) if spacing_text else 10.0
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Error", "Invalid numeric input.")
            return
        width = r*4
        height = r*3
        n_trials = self.itterations_slider.value()
        self.thread = PackEllipseThread(self.poly, width, height, self.main_dir, spacing, n_trials)
        self.thread.progress_signal.connect(self.progress_bar.setValue)
        self.thread.finished_signal.connect(self.on_layout_finished)
        self.thread.start()

    def on_layout_finished(self, ellipses):
        if not ellipses:
            QtWidgets.QMessageBox.warning(self, "Layout Failed", "No valid layout generated")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Turbine Layout", "", "Shapefiles (*.shp)")
        if not path:
            return
        layer = self.construction_layer_combo.currentLayer()
        crs = layer.crs()
        fields = QgsFields()
        fields.append(QgsField("id", QVariant.Int))
        
        writer = QgsVectorFileWriter(path, "UTF-8", fields, QgsWkbTypes.Point, crs, "ESRI Shapefile")
        if writer.hasError() != QgsVectorFileWriter.NoError:
            QtWidgets.QMessageBox.critical(self, "Error", f"Cannot write to {path}: {writer.errorMessage()}")
            return
            
        for i, ellipse in enumerate(ellipses):
            feat = QgsFeature()
            feat.setGeometry(ellipse.centroid())
            feat.setAttributes([i])
            writer.addFeature(feat)
        del writer
        
        msg = f"Turbine centers saved to {path}"

        if self.export_wakes_chk.isChecked():
            ellipse_path = path.replace(".shp", "_wakes.shp")
            writer = QgsVectorFileWriter(ellipse_path, "UTF-8", fields, QgsWkbTypes.Polygon, crs, "ESRI Shapefile")
            if writer.hasError() != QgsVectorFileWriter.NoError:
                QtWidgets.QMessageBox.critical(self, "Error", f"Cannot write to {ellipse_path}: {writer.errorMessage()}")
                return
                
            for i, ellipse in enumerate(ellipses):
                feat = QgsFeature()
                feat.setGeometry(ellipse)
                feat.setAttributes([i])
                writer.addFeature(feat)
            del writer
            msg += f"\nTurbine wakes saved to {ellipse_path}"

        QtWidgets.QMessageBox.information(self, "Saved", msg)

    def on_plot_wind_rose(self):
        if not self.results:
            QtWidgets.QMessageBox.warning(self, "No Data", "Please fetch wind data first")
            return
            
        wind_speeds = self.results[0]['time_series_ws']
        wind_dirs = self.results[0]['time_series_wd']
        valid = ~np.isnan(wind_speeds) & ~np.isnan(wind_dirs)
        wind_speeds = wind_speeds[valid]
        wind_dirs = wind_dirs[valid]

        n_dir_bins = 12
        speed_bins = [0, 2, 4, 6, 8, 10, 15, 25]
        speed_labels = ["0-2", "2-4", "4-6", "6-8", "8-10", "10-15", "15-25"]
        
        rose_data = bin_wind_data(wind_speeds, wind_dirs, n_dir_bins, speed_bins)
        
        image = create_wind_rose_image(rose_data, speed_labels)
        
        self.image_dialog = ImageDialog(image, self)
        self.image_dialog.show()