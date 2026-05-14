import json
import os
import subprocess
import warnings
from math import atan2, cos, radians, sin, sqrt
from typing import List, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmium
from deap import algorithms, base, creator, tools
from shapely import STRtree
from shapely.geometry import LineString, Point, mapping, shape
from shapely.ops import linemerge, nearest_points, polygonize
from shapely.prepared import prep

warnings.filterwarnings("ignore")

BIKE_SPEED_KMH = 20
TIME_15MIN_METERS = (BIKE_SPEED_KMH / 3.6) * 15 * 60  # ~5000 метров

BASE_N_STORES = 7
BASE_AREA_KM2 = 152.0  # Площадь Твери в км²


class CityDarkstoreOptimizer:
    def __init__(self, pbf_path: str, city_name: str, n_stores=None, density_candidates: bool = False, bike_infra: bool = False):
        self.pbf_path = pbf_path
        self.original_pbf_path = pbf_path
        self.city_name = city_name
        self.n_stores = n_stores
        self.density_candidates = density_candidates
        self.bike_infra = bike_infra
        self.residential_buildings = []
        self.candidate_locations = []
        self.road_network = None
        self.city_polygon = None
        self.city_bbox = None
        self._prepared_polygon = None
        # Велоинфраструктура
        self.bike_geometries = []
        self.transport_geometries = []
        self.bike_tree = None
        self.transport_tree = None
        self.infra_distances = None

    # ------------------------------------------------------------------
    # Подготовка: полигон + вырезание города в отдельный PBF
    # ------------------------------------------------------------------

    def _extract_city_boundary(self):
        """Извлекает полигон границы города из OSM PBF файла."""
        print(f"Поиск границы города {self.city_name}...")

        # Проход 1: найти relation с границей города, собрать ID outer-way
        class RelationFinder(osmium.SimpleHandler):
            def __init__(self, city_name):
                osmium.SimpleHandler.__init__(self)
                self.city_name = city_name
                self.way_ids = set()
                self.found = False
                self.matched_name = ""

            def relation(self, r):
                if self.found:
                    return
                name = r.tags.get("name", "")
                if self.city_name.lower() in name.lower() and r.tags.get("boundary") == "administrative" and r.tags.get("admin_level") in ("4", "6", "8"):
                    self.found = True
                    self.matched_name = name
                    for member in r.members:
                        if member.role == "outer" and member.type == "w":
                            self.way_ids.add(member.ref)

        finder = RelationFinder(self.city_name)
        finder.apply_file(self.pbf_path)

        if not finder.found:
            raise ValueError(f"Город '{self.city_name}' не найден в {self.pbf_path}. Проверьте точное название в OSM (например, 'городской округ ...').")

        print(f"  Совпадение: '{finder.matched_name}'")
        print(f"  Найдена граница: {len(finder.way_ids)} участков")

        # Проход 2: собрать координаты для boundary-way
        class WayCollector(osmium.SimpleHandler):
            def __init__(self, target_way_ids):
                osmium.SimpleHandler.__init__(self)
                self.target_way_ids = target_way_ids
                self.ways = {}

            def way(self, w):
                if w.id in self.target_way_ids:
                    coords = [(n.lon, n.lat) for n in w.nodes]
                    if len(coords) >= 2:
                        self.ways[w.id] = coords

        collector = WayCollector(finder.way_ids)
        collector.apply_file(self.pbf_path, locations=True)

        # Построить полигон из way-координат
        lines = [LineString(coords) for coords in collector.ways.values()]
        merged = linemerge(lines)
        polygons = list(polygonize(merged))

        if not polygons:
            raise ValueError(f"Не удалось построить полигон для '{self.city_name}'")

        self.city_polygon = max(polygons, key=lambda p: p.area)
        self._prepared_polygon = prep(self.city_polygon)

        bounds = self.city_polygon.bounds  # (minx, miny, maxx, maxy)
        self.city_bbox = (bounds[0], bounds[2], bounds[1], bounds[3])  # (min_lon, max_lon, min_lat, max_lat)

        print(f"  Полигон построен, BBOX: {self.city_bbox}")

    def _prepare_city_pbf(self):
        """Вырезает область города в отдельный PBF-файл для быстрой обработки."""
        city_pbf_dir = "data/cities"
        os.makedirs(city_pbf_dir, exist_ok=True)

        safe_name = self.city_name.lower().replace(" ", "_")
        city_pbf = os.path.join(city_pbf_dir, f"{safe_name}.osm.pbf")
        boundary_geojson = os.path.join(city_pbf_dir, f"{safe_name}_boundary.geojson")

        # Кэш: если оба файла есть — используем
        if os.path.exists(city_pbf) and os.path.exists(boundary_geojson):
            print(f"Кэш: используется {city_pbf}")
            with open(boundary_geojson, "r") as f:
                geojson = json.load(f)
            # GeoJSON может быть FeatureCollection — достаём геометрию
            if geojson.get("type") == "FeatureCollection":
                geometry = geojson["features"][0]["geometry"]
            elif geojson.get("type") == "Feature":
                geometry = geojson["geometry"]
            else:
                geometry = geojson
            self.city_polygon = shape(geometry)
            self._prepared_polygon = prep(self.city_polygon)
            bounds = self.city_polygon.bounds
            self.city_bbox = (bounds[0], bounds[2], bounds[1], bounds[3])
            self.pbf_path = city_pbf
            return

        # Извлечь полигон границы из оригинального PBF
        self._extract_city_boundary()

        # Сохранить полигон в GeoJSON
        geojson_data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"city": self.city_name},
                    "geometry": mapping(self.city_polygon),
                }
            ],
        }
        with open(boundary_geojson, "w") as f:
            json.dump(geojson_data, f)

        # Вырезать город из PBF через osmium extract
        print(f"Вырезание области города из PBF...")
        result = subprocess.run(
            ["osmium", "extract", "-p", boundary_geojson, self.original_pbf_path, "-o", city_pbf],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            # Fallback: bbox вместо полигона
            bbox_str = f"{self.city_bbox[0]},{self.city_bbox[2]},{self.city_bbox[1]},{self.city_bbox[3]}"
            print(f"  Полигон не подошёл, пробуем bbox: {bbox_str}")
            result = subprocess.run(
                ["osmium", "extract", "-b", bbox_str, self.original_pbf_path, "-o", city_pbf],
                capture_output=True,
                text=True,
            )

        if result.returncode != 0:
            raise RuntimeError(f"osmium extract не удался: {result.stderr}\nУстановите osmium-tool: brew install osmium-tool (macOS) / apt install osmium-tool (Ubuntu)")

        file_size_mb = os.path.getsize(city_pbf) / (1024 * 1024)
        print(f"Город сохранён в: {city_pbf} ({file_size_mb:.1f} МБ)")
        self.pbf_path = city_pbf

    # ------------------------------------------------------------------
    # Извлечение данных из PBF
    # ------------------------------------------------------------------

    def extract_city_data(self):
        print(f"Извлечение данных {self.city_name} из OSM...")
        optimizer = self
        city_bbox = self.city_bbox
        prepared_polygon = self._prepared_polygon

        class DataHandler(osmium.SimpleHandler):
            """Обрабатывает ways: жилые здания и коммерческие локации."""

            def __init__(self, buildings, candidates, opt, bbox, prep_poly):
                osmium.SimpleHandler.__init__(self)
                self.buildings = buildings
                self.candidates = candidates
                self.opt = opt
                self.bbox = bbox
                self.prep_poly = prep_poly

            def _in_city(self, lon, lat):
                """Быстрая проверка: точка внутри полигона города."""
                if not (self.bbox[0] <= lon <= self.bbox[1] and self.bbox[2] <= lat <= self.bbox[3]):
                    return False
                return self.prep_poly.contains(Point(lon, lat))

            def way(self, w):
                tags = w.tags
                if tags is None:
                    return

                # Жилые здания
                if "building" in tags and tags.get("building") in [
                    "residential",
                    "apartments",
                    "house",
                    "detached",
                    "semidetached_house",
                    "terrace",
                    "dormitory",
                    "yes",
                ]:
                    coords = [(n.lon, n.lat) for n in w.nodes]
                    if len(coords) < 2:
                        return
                    cx = sum(c[0] for c in coords) / len(coords)
                    cy = sum(c[1] for c in coords) / len(coords)
                    if self._in_city(cx, cy):
                        self.buildings.append(
                            {
                                "id": w.id,
                                "lon": cx,
                                "lat": cy,
                                "population": self.opt.estimate_population(tags),
                            }
                        )

                # Коммерческие кандидатные локации
                if tags.get("landuse") in ["commercial", "retail"] or tags.get("building") == "commercial":
                    coords = [(n.lon, n.lat) for n in w.nodes]
                    if len(coords) < 2:
                        return
                    cx = sum(c[0] for c in coords) / len(coords)
                    cy = sum(c[1] for c in coords) / len(coords)
                    if self._in_city(cx, cy):
                        self.candidates.append((cx, cy))

        data_handler = DataHandler(
            self.residential_buildings,
            self.candidate_locations,
            optimizer,
            city_bbox,
            prepared_polygon,
        )
        data_handler.apply_file(self.pbf_path, locations=True)

        print(f"Найдено {len(self.residential_buildings)} жилых домов")
        print(f"{len(self.candidate_locations)} кандидатных локаций")

        self._add_candidates()

    def estimate_population(self, tags) -> int:
        building_type = tags.get("building", "")
        try:
            levels = int(tags.get("building:levels", 1))
        except (ValueError, TypeError):
            levels = 1

        if building_type == "yes":
            return min(20 * levels, 200)
        return min(50 * levels, 500)

    def _add_candidates(self):
        """Добавляет кандидатные локации: OSM-коммерция + сетка/плотность."""
        if self.city_bbox is None or self._prepared_polygon is None:
            return
        if self.density_candidates:
            self._add_density_candidates()
        else:
            self._add_grid_candidates()

    def _add_grid_candidates(self):
        """Равномерная сетка кандидатов по полигону города."""
        lons = np.arange(self.city_bbox[0], self.city_bbox[1], 0.005)
        lats = np.arange(self.city_bbox[2], self.city_bbox[3], 0.005)
        for lon in lons[::5]:
            for lat in lats[::5]:
                if self._prepared_polygon.contains(Point(lon, lat)):
                    self.candidate_locations.append((lon, lat))

    def _add_density_candidates(self, n_target=500):
        """Генерация кандидатов пропорционально плотности населения.
        Больше кандидатов в плотных районах, меньше — в пустых."""
        if len(self.residential_buildings) == 0:
            return

        coords = np.array([(b["lon"], b["lat"]) for b in self.residential_buildings])
        pops = np.array([b["population"] for b in self.residential_buildings])
        weights = pops / pops.sum()

        candidates = []
        batch_size = n_target * 3
        while len(candidates) < n_target:
            sampled = np.random.choice(len(coords), size=batch_size, replace=True, p=weights)
            jitter_lon = np.random.normal(0, 0.002, size=batch_size)
            jitter_lat = np.random.normal(0, 0.002, size=batch_size)
            for i in range(batch_size):
                lon = float(coords[sampled[i], 0] + jitter_lon[i])
                lat = float(coords[sampled[i], 1] + jitter_lat[i])
                if self._prepared_polygon.contains(Point(lon, lat)):
                    candidates.append((lon, lat))
                    if len(candidates) >= n_target:
                        break

        self.candidate_locations.extend(candidates[:n_target])
        print(f"  density-кандидаты: {len(candidates[:n_target])} точек")

    def build_road_network(self):
        print("Построение дорожного графа...")

        class RoadHandler(osmium.SimpleHandler):
            def __init__(self, G):
                osmium.SimpleHandler.__init__(self)
                self.G = G
                self.nodes = {}
                self.node_id = 0

            def way(self, w):
                if w.tags.get("highway") in [
                    "primary",
                    "secondary",
                    "tertiary",
                    "residential",
                    "living_street",
                    "cycleway",
                ]:
                    coords = [(n.lon, n.lat) for n in w.nodes]
                    if len(coords) > 1:
                        for lon, lat in coords:
                            if (lon, lat) not in self.nodes:
                                self.nodes[(lon, lat)] = self.node_id
                                self.node_id += 1
                        for i in range(len(coords) - 1):
                            u = self.nodes[coords[i]]
                            v = self.nodes[coords[i + 1]]
                            dist = CityDarkstoreOptimizer.haversine(coords[i], coords[i + 1])
                            self.G.add_edge(u, v, weight=dist)

        self.road_network = nx.Graph()
        road_handler = RoadHandler(self.road_network)
        road_handler.apply_file(self.pbf_path, locations=True)
        print(f"Граф дорог: {self.road_network.number_of_nodes()} узлов")

    def _extract_bike_infrastructure(self):
        """Извлечение велодорожек и транспортных дорог из PBF."""
        print("Извлечение велоинфраструктуры...")

        class BikeInfraHandler(osmium.SimpleHandler):
            def __init__(self):
                osmium.SimpleHandler.__init__(self)
                self.bike_ways = []
                self.transport_ways = []

            def way(self, w):
                tags = w.tags
                if tags is None:
                    return

                highway = tags.get("highway", "")
                cycleway = tags.get("cycleway", "")
                bicycle = tags.get("bicycle", "")

                coords = [(n.lon, n.lat) for n in w.nodes]
                if len(coords) < 2:
                    return

                # Велоинфраструктура (высокий приоритет)
                is_bike = highway == "cycleway" or cycleway in ("lane", "track", "shared_lane") or bicycle in ("designated", "yes") or (highway == "path" and bicycle in ("designated", "yes"))

                if is_bike:
                    self.bike_ways.append(LineString(coords))
                elif highway in ("primary", "secondary", "tertiary"):
                    self.transport_ways.append(LineString(coords))

        handler = BikeInfraHandler()
        handler.apply_file(self.pbf_path, locations=True)

        self.bike_geometries = handler.bike_ways
        self.transport_geometries = handler.transport_ways

        self.bike_tree = STRtree(self.bike_geometries) if self.bike_geometries else None
        self.transport_tree = STRtree(self.transport_geometries) if self.transport_geometries else None

        print(f"  Велодорожки: {len(self.bike_geometries)}")
        print(f"  Транспортные дороги: {len(self.transport_geometries)}")

    def _compute_infra_distances(self):
        """Расстояние от каждого кандидата до ближайшей велодорожки / транспортной дороги."""
        n = len(self.candidate_pts)
        if n == 0 or (self.bike_tree is None and self.transport_tree is None):
            self.infra_distances = np.zeros(max(n, 0))
            return

        # Усреднённый перевод градусов в метры на данной широте
        avg_lat = (self.city_bbox[2] + self.city_bbox[3]) / 2
        m_per_deg = (111320 * cos(radians(avg_lat)) + 110540) / 2

        distances = np.full(n, 10000.0)  # по умолчанию 10 км

        for i in range(n):
            pt = Point(float(self.candidate_pts[i][0]), float(self.candidate_pts[i][1]))

            # Расстояние до велодорожки (наивысший приоритет)
            if self.bike_tree is not None and len(self.bike_geometries) > 0:
                idx = self.bike_tree.nearest(pt)
                dist_m = pt.distance(self.bike_geometries[idx]) * m_per_deg
                distances[i] = min(distances[i], dist_m)

            # Расстояние до транспортной дороги (с штрафом 1.5x)
            if self.transport_tree is not None and len(self.transport_geometries) > 0:
                idx = self.transport_tree.nearest(pt)
                dist_m = pt.distance(self.transport_geometries[idx]) * m_per_deg * 1.5
                distances[i] = min(distances[i], dist_m)

        self.infra_distances = distances
        print(f"  Ср. расстояние до велоинфраструктуры: {distances.mean():.0f} м")

    @staticmethod
    def haversine(coord1, coord2):
        lon1, lat1 = map(radians, coord1)
        lon2, lat2 = map(radians, coord2)
        dlon, dlat = lon2 - lon1, lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        return 6371000 * 2 * atan2(sqrt(a), sqrt(1 - a))

    @staticmethod
    def haversine_vectorized(coords1, coords2):
        """Векторизованная формула гаверсинуса.

        Args:
            coords1: numpy array (N, 2) — массив (lon, lat)
            coords2: numpy array (M, 2) — массив (lon, lat)
        Returns:
            numpy array (N, M) — матрица расстояний в метрах
        """
        lon1 = np.radians(coords1[:, 0])[:, np.newaxis]
        lat1 = np.radians(coords1[:, 1])[:, np.newaxis]
        lon2 = np.radians(coords2[:, 0])[np.newaxis, :]
        lat2 = np.radians(coords2[:, 1])[np.newaxis, :]
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return 6371000 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # ------------------------------------------------------------------
    # Автоматический расчёт площади и числа дарксторов
    # ------------------------------------------------------------------

    def _compute_city_area_km2(self):
        """Вычисляет площадь полигона города в км² через UTM-проекцию."""
        gdf = gpd.GeoDataFrame(geometry=[self.city_polygon], crs="EPSG:4326")
        utm_crs = gdf.estimate_utm_crs()
        gdf = gdf.to_crs(utm_crs)
        area_km2 = gdf.geometry.area.iloc[0] / 1e6
        print(f"Площадь города {self.city_name}: {area_km2:.1f} км²")
        return area_km2

    def _compute_n_stores(self):
        """Расчёт числа дарксторов по площади города.
        Используется корень из отношения площадей: при росте площади
        число дарксторов растёт медленнее, т.к. покрытие перекрывается."""
        area_km2 = self._compute_city_area_km2()
        n = max(3, round(BASE_N_STORES * (area_km2 / BASE_AREA_KM2) ** 0.5))
        print(f"Автоматический расчёт: {n} дарксторов (площадь {area_km2:.1f} км²)")
        return n

    # ------------------------------------------------------------------
    # Предобработка
    # ------------------------------------------------------------------

    def preprocess_data(self):
        self._prepare_city_pbf()  # полигон + вырезание города
        self.extract_city_data()  # маленький PBF — быстро
        self.build_road_network()  # маленький PBF — быстро

        if len(self.residential_buildings) == 0:
            print("ВНИМАНИЕ: Жилые здания не найдены. Проверьте PBF файл и название города.")
            return

        self.residential_pts = np.array([(b["lon"], b["lat"], b["population"]) for b in self.residential_buildings])
        self.candidate_pts = np.array(self.candidate_locations[:1000])

        # Велоинфраструктура
        if self.bike_infra:
            self._extract_bike_infrastructure()
            self._compute_infra_distances()

        total_population = self.residential_pts[:, 2].sum()
        print(f"Общее население {self.city_name}: {total_population:,.0f} чел")

        if self.n_stores is None:
            self.n_stores = self._compute_n_stores()

    # ------------------------------------------------------------------
    # Оценка и оптимизация
    # ------------------------------------------------------------------

    def evaluate_solution(self, individual: List[int]) -> Tuple[float, float, float]:
        """Оценка решения: % непокрытых, кол-во дарксторов, общие затраты"""
        if len(individual) == 0:
            if self.bike_infra:
                return 1.0, 1000, 1e9, 10000.0
            return 1.0, 1000, 1e9

        warehouses = self.candidate_pts[individual]
        houses = self.residential_pts[:, :2]
        populations = self.residential_pts[:, 2]

        dist_matrix = self.haversine_vectorized(houses, warehouses)
        min_distances = dist_matrix.min(axis=1)
        covered = min_distances <= TIME_15MIN_METERS
        covered_population = populations[covered].sum()

        coverage = 1 - (covered_population / populations.sum())
        num_stores = len(individual)
        cost = num_stores * 1500000

        if self.bike_infra:
            avg_infra_dist = np.mean([self.infra_distances[idx] for idx in individual])
            return coverage, num_stores, cost, avg_infra_dist

        return coverage, num_stores, cost

    def greedy_solution(self, n_stores=None):
        """Жадный алгоритм: на каждом шаге добавляем даркстор,
        который покрывает максимум ещё не покрытого населения."""
        if n_stores is None:
            n_stores = self.n_stores
        print(f"Запуск жадного алгоритма ({n_stores} дарксторов)...")

        houses = self.residential_pts[:, :2]
        populations = self.residential_pts[:, 2]

        # Vectorized: compute all distances at once
        dist_matrix = self.haversine_vectorized(houses, self.candidate_pts)  # (n_houses, n_candidates)
        coverage_masks = dist_matrix <= TIME_15MIN_METERS  # bool matrix

        # Precompute population covered by each candidate
        candidate_pops = coverage_masks.T @ populations  # (n_candidates,)

        selected = []
        covered_mask = np.zeros(len(houses), dtype=bool)

        for step in range(n_stores):
            best_idx = -1
            best_score = -1
            for c_idx in range(len(self.candidate_pts)):
                if c_idx in selected:
                    continue
                new_pop = populations[coverage_masks[:, c_idx] & ~covered_mask].sum()

                # Велоинфраструктура: бонус за близость к дорожкам/дорогам
                if self.bike_infra and self.infra_distances is not None:
                    infra_bonus = 1.0 / (1.0 + self.infra_distances[c_idx] / 500)  # 1.0 у дорожки, 0.5 в 500м
                    score = new_pop * (0.7 + 0.3 * infra_bonus)
                else:
                    score = new_pop

                if score > best_score:
                    best_score = score
                    best_idx = c_idx

            if best_idx == -1:
                break

            selected.append(best_idx)
            covered_mask |= coverage_masks[:, best_idx]
            covered_pop = populations[covered_mask].sum()
            total_pop = populations.sum()
            print(f"  Шаг {step + 1}: кандидат {best_idx}, покрытие {100 * covered_pop / total_pop:.1f}%")

        return selected

    def _seeded_individual(self, greedy_indices):
        """Создаёт особь на основе жадного решения с мутацией."""
        n_candidates = len(self.candidate_pts)
        n_keep = np.random.randint(1, len(greedy_indices) + 1)
        keep = np.random.choice(greedy_indices, size=n_keep, replace=False).tolist()
        n_extra = np.random.randint(0, 4)
        extras = np.random.choice(n_candidates, size=n_extra, replace=False).tolist()
        return creator.Individual(list(set(keep + extras)))

    def create_nsga2(self):
        if hasattr(creator, "FitnessMulti"):
            del creator.FitnessMulti
        if hasattr(creator, "Individual"):
            del creator.Individual

        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0) if self.bike_infra else (-1.0, -1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        n_candidates = len(self.candidate_pts)
        n_stores = self.n_stores

        def random_individual(min_stores=max(1, n_stores - 2), max_stores=n_stores + 2):
            n = np.random.randint(min_stores, max_stores + 1)
            return creator.Individual(np.random.choice(n_candidates, size=n, replace=False).tolist())

        greedy_indices = self.greedy_solution()

        toolbox = base.Toolbox()
        toolbox.register("individual", random_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def create_seeded_population(n, greedy_indices):
            """Создаёт популяцию: ~30% от жадного решения, ~70% случайных."""
            pop = []
            n_seeded = max(1, n // 3)
            for _ in range(n_seeded):
                ind = self._seeded_individual(greedy_indices)
                pop.append(ind)
            for _ in range(n - n_seeded):
                pop.append(random_individual())
            return pop

        def cx_set(ind1, ind2):
            """Кроссовер: берём случайные элементы из объединения родителей"""
            union = list(set(ind1) | set(ind2))
            if len(union) < 2:
                return ind1, ind2
            np.random.shuffle(union)
            split = np.random.randint(1, len(union))
            ind1[:] = creator.Individual(union[:split])
            ind2[:] = creator.Individual(union[split:])
            return ind1, ind2

        def mut_set(individual, indpb=0.2):
            """Мутация: заменяем случайные индексы на новые"""
            for i in range(len(individual)):
                if np.random.random() < indpb:
                    individual[i] = np.random.randint(n_candidates)
            individual[:] = creator.Individual(list(set(individual)))
            if len(individual) == 0:
                individual.append(np.random.randint(n_candidates))
            return (individual,)

        toolbox.register("evaluate", self.evaluate_solution)
        toolbox.register("mate", cx_set)
        toolbox.register("mutate", mut_set, indpb=0.2)
        toolbox.register("select", tools.selNSGA2)

        return toolbox, greedy_indices, create_seeded_population

    def optimize(self, population_size=200, generations=100):
        print("Запуск NSGA-II оптимизации...")
        toolbox, greedy_indices, create_seeded_population = self.create_nsga2()

        pop = create_seeded_population(population_size, greedy_indices)
        hof = tools.ParetoFront()

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("min", np.min, axis=0)

        algorithms.eaMuPlusLambda(
            pop,
            toolbox,
            mu=population_size,
            lambda_=population_size,
            cxpb=0.7,
            mutpb=0.3,
            ngen=generations,
            stats=stats,
            halloffame=hof,
            verbose=True,
        )

        return hof

    # ------------------------------------------------------------------
    # Визуализация и экспорт
    # ------------------------------------------------------------------

    def visualize_results(self, pareto_front):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        if self.city_polygon:
            city_boundary_gdf = gpd.GeoDataFrame(geometry=[self.city_polygon], crs="EPSG:4326")
            city_boundary_gdf.plot(ax=ax1, facecolor="none", edgecolor="gray", linewidth=1)

        city_gdf = gpd.GeoDataFrame(
            geometry=[Point(lon, lat) for lon, lat, _ in self.residential_pts],
            crs="EPSG:4326",
        )
        city_gdf.plot(ax=ax1, color="lightblue", markersize=1, alpha=0.6, label="Жилые дома")

        # Велодорожки
        if self.bike_infra and self.bike_geometries:
            bike_gdf = gpd.GeoDataFrame(geometry=self.bike_geometries, crs="EPSG:4326")
            bike_gdf.plot(ax=ax1, color="green", linewidth=1.5, alpha=0.7, label="Велодорожки")
        if self.bike_infra and self.transport_geometries:
            trans_gdf = gpd.GeoDataFrame(geometry=self.transport_geometries, crs="EPSG:4326")
            trans_gdf.plot(ax=ax1, color="orange", linewidth=0.5, alpha=0.4, label="Транспортные дороги")

        best_solution = min(pareto_front, key=lambda ind: ind.fitness.values[0])

        for i, wh in enumerate(self.candidate_pts[best_solution]):
            circle = Point(wh).buffer(TIME_15MIN_METERS / 111000)
            gpd.GeoSeries([circle]).plot(ax=ax1, color="red", alpha=0.3)
            ax1.plot(wh[0], wh[1], "ro", markersize=10, label="Дарксторы" if i == 0 else "")

        ax1.set_title(f"Оптимальное расположение дарксторов - {self.city_name}")
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys())
        ax1.set_aspect("equal")

        coverages = [ind.fitness.values[0] for ind in pareto_front]
        stores = [ind.fitness.values[1] for ind in pareto_front]
        ax2.scatter(stores, coverages, c="red")
        ax2.set_xlabel("Количество дарксторов")
        ax2.set_ylabel("% непокрытых домов")
        ax2.set_title("Pareto фронт")

        plt.tight_layout()

        mode = "density" if self.density_candidates else "grid"
        if self.bike_infra:
            mode = "bike_" + mode
        output_dir = os.path.join("output", mode, self.city_name.lower().replace(" ", "_"))
        os.makedirs(output_dir, exist_ok=True)
        output_png = os.path.join(output_dir, f"{self.city_name.lower()}_darkstores_optimized.png")
        plt.savefig(output_png, dpi=300, bbox_inches="tight")
        plt.close()

        best_coverage = min([ind.fitness.values[0] for ind in pareto_front])
        print("\nЛУЧШИЙ РЕЗУЛЬТАТ:")
        print(f"   Покрытие: {100 * (1 - best_coverage):.1f}% жилых домов")
        print(f"   Дарксторов: {len(best_solution)}")
        print(f"   Стоимость: {best_solution.fitness.values[2] / 1e6:.1f} млн руб")
        if self.bike_infra and len(best_solution.fitness.values) > 3:
            print(f"   Ср. дистанция до велоинфраструктуры: {best_solution.fitness.values[3]:.0f} м")

    def export_geojson(self, pareto_front, output_dir=None):
        """Экспорт результатов в GeoJSON: дарксторы, зоны покрытия, жилые дома, граница города."""
        if output_dir is None:
            mode = "density" if self.density_candidates else "grid"
            if self.bike_infra:
                mode = "bike_" + mode
            output_dir = os.path.join("output", mode, self.city_name.lower().replace(" ", "_"))
        os.makedirs(output_dir, exist_ok=True)

        best_solution = min(pareto_front, key=lambda ind: ind.fitness.values[0])

        # --- Граница города ---
        if self.city_polygon:
            boundary_path = os.path.join(output_dir, "city_boundary.geojson")
            boundary_gdf = gpd.GeoDataFrame(
                geometry=[self.city_polygon],
                crs="EPSG:4326",
                data={"city": [self.city_name]},
            )
            boundary_gdf.to_file(boundary_path, driver="GeoJSON")
            print(f"Граница города -> {boundary_path}")

        # --- Дарксторы ---
        darkstore_features = []
        for i, idx in enumerate(best_solution):
            wh = self.candidate_pts[idx]
            darkstore_features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [float(wh[0]), float(wh[1])]},
                    "properties": {
                        "id": i + 1,
                        "candidate_index": int(idx),
                        "lon": float(wh[0]),
                        "lat": float(wh[1]),
                        "city": self.city_name,
                        "infra_dist_m": round(float(self.infra_distances[idx]), 1) if self.infra_distances is not None else None,
                    },
                }
            )

        darkstore_gdf = gpd.GeoDataFrame.from_features(darkstore_features, crs="EPSG:4326")
        darkstore_path = os.path.join(output_dir, "darkstores.geojson")
        darkstore_gdf.to_file(darkstore_path, driver="GeoJSON")
        print(f"Дарксторы -> {darkstore_path}")

        # --- Зоны покрытия ---
        coverage_features = []
        for i, idx in enumerate(best_solution):
            wh = self.candidate_pts[idx]
            circle = Point(wh).buffer(TIME_15MIN_METERS / 111000)
            coverage_features.append(
                {
                    "type": "Feature",
                    "geometry": circle.__geo_interface__,
                    "properties": {
                        "darkstore_id": i + 1,
                        "radius_m": int(TIME_15MIN_METERS),
                        "city": self.city_name,
                    },
                }
            )

        coverage_gdf = gpd.GeoDataFrame.from_features(coverage_features, crs="EPSG:4326")
        coverage_path = os.path.join(output_dir, "coverage_zones.geojson")
        coverage_gdf.to_file(coverage_path, driver="GeoJSON")
        print(f"Зоны покрытия -> {coverage_path}")

        # --- Жилые дома ---
        warehouses = self.candidate_pts[best_solution]
        houses = self.residential_pts[:, :2]

        dist_matrix = self.haversine_vectorized(houses, warehouses)
        min_distances = dist_matrix.min(axis=1)
        is_covered = min_distances <= TIME_15MIN_METERS

        house_features = []
        for i, (house_lon, house_lat, house_pop) in enumerate(self.residential_pts):
            house_features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [float(house_lon), float(house_lat)]},
                    "properties": {
                        "population": int(house_pop),
                        "covered": bool(is_covered[i]),
                    },
                }
            )

        houses_gdf = gpd.GeoDataFrame.from_features(house_features, crs="EPSG:4326")
        houses_path = os.path.join(output_dir, "residential_houses.geojson")
        houses_gdf.to_file(houses_path, driver="GeoJSON")
        print(f"Жилые дома -> {houses_path}")

        # --- Pareto-фронт ---
        pareto_features = []
        for ind in pareto_front:
            vals = ind.fitness.values
            cov, n_stores, cost = vals[0], vals[1], vals[2]
            infra_dist = vals[3] if len(vals) > 3 else None
            stores_coords = [[float(self.candidate_pts[idx][0]), float(self.candidate_pts[idx][1])] for idx in ind]
            pareto_features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "MultiPoint", "coordinates": stores_coords},
                    "properties": {
                        "uncovered_pct": round(float(cov) * 100, 2),
                        "coverage_pct": round((1 - float(cov)) * 100, 2),
                        "num_stores": int(n_stores),
                        "cost_mln_rub": round(float(cost) / 1e6, 1),
                        "avg_infra_dist_m": round(float(infra_dist), 1) if infra_dist is not None else None,
                    },
                }
            )

        pareto_gdf = gpd.GeoDataFrame.from_features(pareto_features, crs="EPSG:4326")
        pareto_path = os.path.join(output_dir, "pareto_front.geojson")
        pareto_gdf.to_file(pareto_path, driver="GeoJSON")
        print(f"Pareto-фронт -> {pareto_path}")

        # --- Велоинфраструктура ---
        if self.bike_infra:
            if self.bike_geometries:
                bike_gdf = gpd.GeoDataFrame(geometry=self.bike_geometries, crs="EPSG:4326")
                bike_path = os.path.join(output_dir, "bike_paths.geojson")
                bike_gdf.to_file(bike_path, driver="GeoJSON")
                print(f"Велодорожки -> {bike_path}")
            if self.transport_geometries:
                trans_gdf = gpd.GeoDataFrame(geometry=self.transport_geometries, crs="EPSG:4326")
                trans_path = os.path.join(output_dir, "transport_roads.geojson")
                trans_gdf.to_file(trans_path, driver="GeoJSON")
                print(f"Транспортные дороги -> {trans_path}")

        print(f"\nЭкспорт завершён. Файлы в папке {output_dir}/")


if __name__ == "__main__":
    cities = [
        ("Тверь", None),
        ("Санкт-Петербург", None),
        ("Екатеринбург", None),
    ]

    for city_name, n_stores in cities:
        print("=" * 60)
        print(f"  ГОРОД: {city_name}")
        print("=" * 60)

        optimizer = CityDarkstoreOptimizer(
            pbf_path="data/russia.osm.pbf",
            city_name=city_name,
            n_stores=n_stores,
        )

        print("\nШаг 1: Подготовка данных")
        optimizer.preprocess_data()

        print(f"\nШаг 2: NSGA-II оптимизация ({optimizer.n_stores} дарксторов)")
        pareto_solutions = optimizer.optimize(population_size=100, generations=30)

        print("\nШаг 3: Визуализация")
        optimizer.visualize_results(pareto_solutions)

        print("\nШаг 4: Экспорт в GeoJSON")
        optimizer.export_geojson(pareto_solutions)

        print(f"\nГород {city_name} обработан!\n")
