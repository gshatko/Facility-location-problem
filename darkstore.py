import warnings
from math import atan2, cos, radians, sin, sqrt
from typing import Dict, List, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmium
from deap import algorithms, base, creator, tools
from shapely.geometry import Point

warnings.filterwarnings("ignore")

BIKE_SPEED_KMH = 20
TIME_15MIN_METERS = (BIKE_SPEED_KMH / 3.6) * 15 * 60  # ~5000 метров


class CityDarkstoreOptimizer:
    def __init__(self, pbf_path: str, city_bbox: Tuple[float, float, float, float], city_name: str = "City"):
        self.pbf_path = pbf_path
        self.city_bbox = city_bbox  # (min_lon, max_lon, min_lat, max_lat)
        self.city_name = city_name
        self.residential_buildings = []
        self.candidate_locations = []
        self.road_network = None

    def extract_city_data(self):
        print(f"Извлечение данных {self.city_name} из OSM...")
        optimizer = self  # ссылка для доступа к estimate_population

        class DataHandler(osmium.SimpleHandler):

            def __init__(self, buildings, candidates, opt, bbox):
                osmium.SimpleHandler.__init__(self)
                self.buildings = buildings
                self.candidates = candidates
                self.opt = opt
                self.bbox = bbox

            def way(self, w):
                tags = w.tags
                if tags is None:
                    return

                bbox = self.bbox

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
                    if bbox[0] <= cx <= bbox[1] and bbox[2] <= cy <= bbox[3]:
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
                    if bbox[0] <= cx <= bbox[1] and bbox[2] <= cy <= bbox[3]:
                        self.candidates.append((cx, cy))

        data_handler = DataHandler(self.residential_buildings, self.candidate_locations, optimizer, self.city_bbox)
        data_handler.apply_file(self.pbf_path, locations=True)

        print(f"Найдено {len(self.residential_buildings)} жилых домов")
        print(f"{len(self.candidate_locations)} кандидатных локаций")

        self._add_grid_candidates()

    def estimate_population(self, tags) -> int:
        building_type = tags.get("building", "")
        try:
            levels = int(tags.get("building:levels", 1))
        except (ValueError, TypeError):
            levels = 1

        if building_type == "yes":
            return min(20 * levels, 200)
        return min(50 * levels, 500)

    def _add_grid_candidates(self):
        lons = np.arange(self.city_bbox[0], self.city_bbox[1], 0.005)
        lats = np.arange(self.city_bbox[2], self.city_bbox[3], 0.005)
        for lon in lons[::5]:
            for lat in lats[::5]:
                self.candidate_locations.append((lon, lat))

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

    @staticmethod
    def haversine(coord1, coord2):
        lon1, lat1 = map(radians, coord1)
        lon2, lat2 = map(radians, coord2)
        dlon, dlat = lon2 - lon1, lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        return 6371000 * 2 * atan2(sqrt(a), sqrt(1 - a))

    def preprocess_data(self):
        self.extract_city_data()
        self.build_road_network()  # NOTE: граф построен, но пока не используется в evaluate_solution

        if len(self.residential_buildings) == 0:
            print("Жилые здания не найдены")
            return

        self.residential_pts = np.array([(b["lon"], b["lat"], b["population"]) for b in self.residential_buildings])
        self.candidate_pts = np.array(self.candidate_locations[:1000])

        total_population = self.residential_pts[:, 2].sum()
        print(f"Общее население {self.city_name}: {total_population:,.0f} чел")

    def evaluate_solution(self, individual: List[int]) -> Tuple[float, float, float]:
        if len(individual) == 0:
            return 1.0, 1000, 1e9

        warehouses = self.candidate_pts[individual]

        covered_population = 0

        for house_lon, house_lat, house_pop in self.residential_pts:
            distances = [self.haversine((house_lon, house_lat), (wh[0], wh[1])) for wh in warehouses]
            min_dist = min(distances)

            if min_dist <= TIME_15MIN_METERS:
                covered_population += house_pop

        coverage = 1 - (covered_population / self.residential_pts[:, 2].sum())
        num_stores = len(individual)
        cost = num_stores * 1500000
        return coverage, num_stores, cost

    def greedy_solution(self, n_stores=5):
        print(f"Запуск жадного алгоритма ({n_stores} дарксторов)...")

        n_candidates = len(self.candidate_pts)
        n_houses = len(self.residential_pts)

        # Предрасчёт: для каждого кандидата — какие дома он покрывает и сколько населения
        candidate_coverage = []  # list of (covered_mask, covered_pop)
        for c_idx in range(n_candidates):
            wh = self.candidate_pts[c_idx]
            mask = np.array([
                self.haversine((h[0], h[1]), (wh[0], wh[1])) <= TIME_15MIN_METERS
                for h in self.residential_pts
            ])
            pop = self.residential_pts[mask, 2].sum()
            candidate_coverage.append((mask, pop))

        selected = []
        covered_mask = np.zeros(n_houses, dtype=bool)

        for step in range(n_stores):
            best_idx = -1
            best_new_pop = -1

            for c_idx in range(n_candidates):
                if c_idx in selected:
                    continue
                mask, _ = candidate_coverage[c_idx]
                new_pop = self.residential_pts[mask & ~covered_mask, 2].sum()
                if new_pop > best_new_pop:
                    best_new_pop = new_pop
                    best_idx = c_idx

            if best_idx == -1:
                break

            selected.append(best_idx)
            covered_mask |= candidate_coverage[best_idx][0]
            covered_pop = self.residential_pts[covered_mask, 2].sum()
            total_pop = self.residential_pts[:, 2].sum()
            print(f"  Шаг {step + 1}: кандидат {best_idx}, "
                  f"покрытие {100 * covered_pop / total_pop:.1f}%")

        return selected

    def _seeded_individual(self, greedy_indices):
        n_candidates = len(self.candidate_pts)
        # Копируем часть жадных индексов
        n_keep = np.random.randint(1, len(greedy_indices) + 1)
        keep = np.random.choice(greedy_indices, size=n_keep, replace=False).tolist()
        # Добавляем случайные
        n_extra = np.random.randint(0, 4)
        extras = np.random.choice(n_candidates, size=n_extra, replace=False).tolist()
        return creator.Individual(list(set(keep + extras)))

    def create_nsga2(self):
        # Удаляем существующие классы DEAP при повторном вызове
        if hasattr(creator, "FitnessMulti"):
            del creator.FitnessMulti
        if hasattr(creator, "Individual"):
            del creator.Individual

        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        n_candidates = len(self.candidate_pts)

        def random_individual(min_stores=3, max_stores=6):
            n = np.random.randint(min_stores, max_stores + 1)
            return creator.Individual(np.random.choice(n_candidates, size=n, replace=False).tolist())

        # Жадное решение для seeding
        greedy_indices = self.greedy_solution(n_stores=6)

        toolbox = base.Toolbox()
        toolbox.register("individual", random_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def create_seeded_population(n, greedy_indices):
            pop = []
            n_seeded = max(1, n // 3)
            for _ in range(n_seeded):
                ind = self._seeded_individual(greedy_indices)
                pop.append(ind)
            for _ in range(n - n_seeded):
                pop.append(random_individual())
            return pop

        def cx_set(ind1, ind2):
            union = list(set(ind1) | set(ind2))
            if len(union) < 2:
                return ind1, ind2
            np.random.shuffle(union)
            split = np.random.randint(1, len(union))
            ind1[:] = creator.Individual(union[:split])
            ind2[:] = creator.Individual(union[split:])
            return ind1, ind2

        def mut_set(individual, indpb=0.2):
            for i in range(len(individual)):
                if np.random.random() < indpb:
                    individual[i] = np.random.randint(n_candidates)
            # Удаляем дубликаты
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

    def visualize_results(self, pareto_front):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        city_gdf = gpd.GeoDataFrame(
            geometry=[Point(lon, lat) for lon, lat, _ in self.residential_pts],
            crs="EPSG:4326",
        )
        city_gdf.plot(ax=ax1, color="lightblue", markersize=1, alpha=0.6, label="Жилые дома")

        best_solution = min(pareto_front, key=lambda ind: ind.fitness.values[0])
        best_wh = self.candidate_pts[best_solution]

        for i, wh in enumerate(best_wh):
            circle = Point(wh).buffer(TIME_15MIN_METERS / 111000)  # ~в градусах
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
        output_png = f"{self.city_name.lower()}_darkstores_optimized.png"
        plt.savefig(output_png, dpi=300, bbox_inches="tight")
        plt.show()

        # Лучшее решение
        best_coverage = min([ind.fitness.values[0] for ind in pareto_front])
        print(f"\nЛУЧШИЙ РЕЗУЛЬТАТ:")
        print(f"   Покрытие: {100 * (1 - best_coverage):.1f}% жилых домов")
        print(f"   Дарксторов: {len(best_solution)}")
        print(f"   Стоимость: {best_solution.fitness.values[2] / 1e6:.1f} млн руб")

    def export_geojson(self, pareto_front, output_dir="output"):
        import os
        os.makedirs(output_dir, exist_ok=True)

        best_solution = min(pareto_front, key=lambda ind: ind.fitness.values[0])
        best_wh = self.candidate_pts[best_solution]

        # --- Дарксторы ---
        darkstore_features = []
        for i, idx in enumerate(best_solution):
            wh = self.candidate_pts[idx]
            darkstore_features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(wh[0]), float(wh[1])]},
                "properties": {
                    "id": i + 1,
                    "candidate_index": int(idx),
                    "lon": float(wh[0]),
                    "lat": float(wh[1]),
                    "city": self.city_name,
                },
            })

        darkstore_gdf = gpd.GeoDataFrame.from_features(darkstore_features, crs="EPSG:4326")
        darkstore_path = os.path.join(output_dir, "darkstores.geojson")
        darkstore_gdf.to_file(darkstore_path, driver="GeoJSON")
        print(f"Дарксторы -> {darkstore_path}")

        # --- Зоны покрытия (15 мин на велосипеде) ---
        coverage_features = []
        for i, idx in enumerate(best_solution):
            wh = self.candidate_pts[idx]
            circle = Point(wh).buffer(TIME_15MIN_METERS / 111000)
            coverage_features.append({
                "type": "Feature",
                "geometry": circle.__geo_interface__,
                "properties": {
                    "darkstore_id": i + 1,
                    "radius_m": int(TIME_15MIN_METERS),
                    "city": self.city_name,
                },
            })

        coverage_gdf = gpd.GeoDataFrame.from_features(coverage_features, crs="EPSG:4326")
        coverage_path = os.path.join(output_dir, "coverage_zones.geojson")
        coverage_gdf.to_file(coverage_path, driver="GeoJSON")
        print(f"Зоны покрытия -> {coverage_path}")

        # --- Жилые дома с признаком покрытия ---
        warehouses = self.candidate_pts[best_solution]
        house_features = []
        for house_lon, house_lat, house_pop in self.residential_pts:
            distances = [self.haversine((house_lon, house_lat), (wh[0], wh[1])) for wh in warehouses]
            is_covered = min(distances) <= TIME_15MIN_METERS
            house_features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(house_lon), float(house_lat)]},
                "properties": {
                    "population": int(house_pop),
                    "covered": bool(is_covered),
                },
            })

        houses_gdf = gpd.GeoDataFrame.from_features(house_features, crs="EPSG:4326")
        houses_path = os.path.join(output_dir, "residential_houses.geojson")
        houses_gdf.to_file(houses_path, driver="GeoJSON")
        print(f"Жилые дома -> {houses_path}")

        # --- Pareto-фронт ---
        pareto_features = []
        for ind in pareto_front:
            cov, n_stores, cost = ind.fitness.values
            stores_coords = [[float(self.candidate_pts[idx][0]), float(self.candidate_pts[idx][1])] for idx in ind]
            pareto_features.append({
                "type": "Feature",
                "geometry": {
                    "type": "MultiPoint",
                    "coordinates": stores_coords,
                },
                "properties": {
                    "uncovered_pct": round(float(cov) * 100, 2),
                    "coverage_pct": round((1 - float(cov)) * 100, 2),
                    "num_stores": int(n_stores),
                    "cost_mln_rub": round(float(cost) / 1e6, 1),
                },
            })

        pareto_gdf = gpd.GeoDataFrame.from_features(pareto_features, crs="EPSG:4326")
        pareto_path = os.path.join(output_dir, "pareto_front.geojson")
        pareto_gdf.to_file(pareto_path, driver="GeoJSON")
        print(f"Pareto-фронт -> {pareto_path}")

        print(f"\nЭкспорт завершён. Файлы в папке {output_dir}/")


if __name__ == "__main__":
    # Пример: Тверь
    TVER_BBOX = (35.75, 36.05, 56.80, 56.92)

    optimizer = CityDarkstoreOptimizer(
        pbf_path="data/russia.osm.pbf",
        city_bbox=TVER_BBOX,
        city_name="Тверь",
    )

    print("Шаг 1: Подготовка данных")
    optimizer.preprocess_data()

    print("Шаг 2: NSGA-II оптимизация (3-6 дарксторов)")
    pareto_solutions = optimizer.optimize(population_size=200, generations=100)

    print("\nШаг 3: Визуализация")
    optimizer.visualize_results(pareto_solutions)

    print("\nШаг 4: Экспорт в GeoJSON")
    optimizer.export_geojson(pareto_solutions)

    print("\nРезультаты сохранены в <city>_darkstores_optimized.png и output/")
