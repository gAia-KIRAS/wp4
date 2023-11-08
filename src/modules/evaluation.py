import geopandas as gpd
import pandas as pd
import numpy as np
import warnings
import random

from modules.abstract_module import Module


class Evaluation(Module):
    def __init__(self, config, io):
        super().__init__(config, io)
        self._cd_id = self._config.eval_conf['cd_id']
        self._results = self._io.get_results_cd()
        self._CRS = 4326

        self._aoi = self.load_area_of_interest()

        self._tile_filters = self._config.filters['tile']
        self._compute_baseline_eval = self._config.eval_conf['baseline_eval']

        # Some general time limits. Will be filtered in detail when merging with the results
        self._date_start = '2018-01-01'
        self._date_end = '2022-12-31'
        self._max_dist = 0.0002

    def load_area_of_interest(self):
        """
        Loads the area of interest (AOI) as a GeoDataFrame.

        Returns:
            aoi: GeoDataFrame with the AOI
        """
        filepath = self._io.config.aoi_path['gpkg']
        aoi = gpd.read_file(filepath)
        aoi.to_crs(self._CRS, inplace=True)

        return aoi

    def run(self, on_the_server=False):
        # Prepare results
        results, n_before_filter = self._prepare_results()

        # Get min and max lat = y / lon = x
        dim = {
            'min_lat': results.geometry.y.min(),
            'max_lat': results.geometry.y.max(),
            'min_lon': results.geometry.x.min(),
            'max_lon': results.geometry.x.max()
        }

        if self._compute_baseline_eval:
            # Compute baseline evaluation
            self._compute_baseline_evaluation(dim, n_before_filter)

        landslide_ids, detected_landslide_ids = [], []

        if self._config.eval_conf['type'] != 'polygons':
            detected_landslide_ids, landslide_ids, results_gt = self.match_with_inventory(
                detected_landslide_ids, dim, landslide_ids, results)

        if self._config.eval_conf['type'] != 'points':
            detected_landslide_ids, landslide_ids, results_gt_poly = self.match_with_polygon_inventory(
                detected_landslide_ids, dim, landslide_ids, results)

        eval_df = self.combine_results(results, results_gt, results_gt_poly)
        fp, tp, precision, recall = self.calculate_results(detected_landslide_ids, eval_df, landslide_ids, results)
        self.print_general_results(detected_landslide_ids, fp, landslide_ids, precision, recall, tp)
        self.disaggregate_results(eval_df, results)

        return 0

    def disaggregate_results(self, eval_df, results):
        # Calculate the number of true positives and false positives by year and tile
        years = results['year'].unique()
        tiles = results['tile'].unique()
        tp, fp, prec = {}, {}, {}
        for year in years:
            for tile in tiles:
                tp[(year, tile)] = len(eval_df[(eval_df['y'] == 1) & (eval_df['year'] == year) &
                                               (eval_df['tile'] == tile)])
                fp[(year, tile)] = len(results[(results['year'] == year) & (results['tile'] == tile)]) - tp[
                    (year, tile)]
                prec[(year, tile)] = tp[(year, tile)] / (tp[(year, tile)] + fp[(year, tile)])
        # Build the dataframe year, tile, tp, fp, precision
        disaggregated_res = pd.DataFrame({'year': [y for y in years for _ in tiles],
                                          'tile': [t for _ in years for t in tiles],
                                          'tp': [tp[(y, t)] for y in years for t in tiles],
                                          'fp': [fp[(y, t)] for y in years for t in tiles],
                                          'precision': [prec[(y, t)] for y in years for t in tiles]}).sort_values(
            ['year', 'tile'])
        print(disaggregated_res)

    def match_with_inventory(self, detected_landslide_ids, dim, landslide_ids, results):
        # Load inventory (ground truth)
        inventory = self._load_inventory(dim)
        landslide_ids += inventory['landslide_id'].unique().tolist()
        print(f'Loaded {len(inventory)} inventory entries for cd_id {self._cd_id}')
        # For every prediction, add 0 / 1 depending on whether there is a close match with the inventory
        results_gt = gpd.sjoin_nearest(results, inventory, how='left', distance_col='distance',
                                       max_distance=self._max_dist).sort_values(by='distance')
        results_gt['date_distance'] = (results_gt['date'] - results_gt['detected_breakpoint']).dt.days
        results_gt = results_gt.merge(inventory[['landslide_id', 'geometry']].rename(
            columns={'geometry': 'geometry_gt'}
        ), on='landslide_id', how='left')
        detected_landslide_ids += results_gt['landslide_id'].unique().tolist()
        detected_landslide_ids = [x for x in detected_landslide_ids if not np.isnan(x)]
        results_gt.drop_duplicates(subset=['lon', 'lat', 'detected_breakpoint'], inplace=True)
        # If there is no match, set y = 0
        results_gt['y'] = 0
        results_gt.loc[(results_gt['distance'] < self._max_dist) & (results_gt['date_distance'] <= 30) &
                       (results_gt['date_distance'] >= -30), 'y'] = 1
        return detected_landslide_ids, landslide_ids, results_gt

    def match_with_polygon_inventory(self, detected_landslide_ids, dim, landslide_ids, results):
        # Load polygon inventory
        inventory = self._load_polygon_inventory(dim)
        landslide_ids += inventory['landslide_id'].unique().tolist()
        print(f'Loaded {len(inventory)} inventory entries for cd_id {self._cd_id}')
        # For every prediction, add 0 / 1 depending on whether there is a close match with the inventory
        results_gt_poly = gpd.sjoin(results, inventory, how='left', predicate='intersects')
        results_gt_poly['y'] = 0
        results_gt_poly.loc[results_gt_poly['landslide_id'].notnull(), 'y'] = 1
        detected_landslide_ids += results_gt_poly['landslide_id'].unique().tolist()
        detected_landslide_ids = [x for x in detected_landslide_ids if not np.isnan(x)]
        results_gt_poly = (results_gt_poly.groupby(['detected_breakpoint', 'lat', 'lon'], as_index=False).
                           agg({'y': 'max'}))
        return detected_landslide_ids, landslide_ids, results_gt_poly

    def combine_results(self, results, results_gt, results_gt_poly):
        # Merge results
        eval_df = results[['detected_breakpoint', 'lat', 'lon', 'year', 'tile']].drop_duplicates()
        if self._config.eval_conf['type'] != 'polygons':
            eval_df = eval_df.merge(results_gt[['detected_breakpoint', 'lat', 'lon', 'y']].rename(
                columns={'y': 'y_points'}), on=['detected_breakpoint', 'lat', 'lon'], how='left')
        if self._config.eval_conf['type'] != 'points':
            eval_df = eval_df.merge(results_gt_poly[['detected_breakpoint', 'lat', 'lon', 'y']].rename(
                columns={'y': 'y_poly'}), on=['detected_breakpoint', 'lat', 'lon'], how='left')
        # Create y column (y_points or y_poly)
        if self._config.eval_conf['type'] == 'points':
            eval_df['y'] = eval_df['y_points']
        elif self._config.eval_conf['type'] == 'polygons':
            eval_df['y'] = eval_df['y_poly']
        else:
            eval_df['y'] = eval_df['y_points'] + eval_df['y_poly']
            eval_df['y'] = eval_df['y'].fillna(0)
        return eval_df

    def calculate_results(self, detected_landslide_ids, eval_df, landslide_ids, results):
        tp = len(eval_df[eval_df['y'] == 1])
        fp = len(results) - tp
        # Calculate the precision
        precision = tp / (tp + fp)
        # Calculate the recall: how many of the landslides were detected
        recall = len(detected_landslide_ids) / len(landslide_ids)
        return fp, tp, precision, recall

    def print_general_results(self, detected_landslide_ids, fp, landslide_ids, precision, recall, tp):
        print(f"""
        --------------------
        Evaluation Results
        --------------------
        True positives: {tp}
        False positives: {fp}

        Precision: {precision}
        Recall: {round(recall, 3)} = {len(detected_landslide_ids)} / {len(landslide_ids)}
        """)

    def _compute_baseline_evaluation(self, dim, n):
        print(f'-------------------\nComputing baseline evaluation for cd_id {self._cd_id}\n-------------------\n')
        random_results = self.generate_random_results(dim, n)

        # Filter according to AOI
        random_results = gpd.sjoin(random_results, self._aoi, how='inner', predicate='intersects')
        random_results.drop(columns=['index_right', 'name', 'area'], inplace=True)

        # Validate
        landslide_ids, detected_landslide_ids = [], []
        if self._config.eval_conf['type'] != 'polygons':
            detected_landslide_ids, landslide_ids, results_gt = self.match_with_inventory(
                detected_landslide_ids, dim, landslide_ids, random_results)

        if self._config.eval_conf['type'] != 'points':
            detected_landslide_ids, landslide_ids, results_gt_poly = self.match_with_polygon_inventory(
                detected_landslide_ids, dim, landslide_ids, random_results)

        eval_df = self.combine_results(random_results, results_gt, results_gt_poly)
        fp, tp, precision, recall = self.calculate_results(detected_landslide_ids, eval_df, landslide_ids,
                                                           random_results)
        self.print_general_results(detected_landslide_ids, fp, landslide_ids, precision, recall, tp)
        self.disaggregate_results(eval_df, random_results)
        print(f'-------------------\n'
              f'End of baseline evaluation for cd_id {self._cd_id}'
              f'\n-------------------\n')

    def generate_random_results(self, dim, n):
        # Generate n random predictions. Must be within dim. Date is random between 2018 and 2022
        random_results = pd.DataFrame({
            'detected_breakpoint': pd.date_range(start='2018-01-01', end='2022-12-31', periods=n),
            'r': np.random.rand(n),
        })
        random_results['lat'] = round(dim['min_lat'] + (dim['max_lat'] - dim['min_lat']) * random_results['r'], 10)
        random_results['lon'] = round(dim['min_lon'] + (dim['max_lon'] - dim['min_lon']) * random_results['r'], 10)
        random_results['d_prob'] = 1
        random_results['year'] = random_results['detected_breakpoint'].dt.year
        random_results['tile'] = '33TUM'
        random_results.drop(columns=['r'], inplace=True)

        # Convert to GeoDataFrame using the coordinates
        random_results = gpd.GeoDataFrame(
            random_results, geometry=gpd.points_from_xy(random_results['lon'], random_results['lat']), crs=self._CRS)

        return random_results

    def _prepare_results(self):
        results = self._results[self._results['cd_id'] == self._cd_id]

        results = results[['detected_breakpoint', 'lat', 'lon', 'd_prob', 'year', 'tile']].sort_values(
            by='d_prob', ascending=False)
        if self._tile_filters:
            results = results[results['tile'].isin(self._tile_filters)]
        results['detected_breakpoint'] = pd.to_datetime(results['detected_breakpoint'], format='%Y%m%d')
        assert len(results) > 0, f'No results found for cd_id {self._cd_id}'

        # Convert to GeoDataFrame using the coordinates
        results = gpd.GeoDataFrame(results, geometry=gpd.points_from_xy(results['lon'], results['lat']), crs=self._CRS)
        results.to_crs(self._CRS, inplace=True)

        # Remove duplicates in lat, lon, detected_breakpoint. Keep the highest probability
        results.drop_duplicates(subset=['lon', 'lat', 'detected_breakpoint'], inplace=True)

        # Filter according to area of interest
        n_before_filter = len(results)
        results = gpd.sjoin(results, self._aoi, how='inner', predicate='intersects')
        results.drop(columns=['index_right', 'name', 'area'], inplace=True)

        return results, n_before_filter

    def _load_inventory(self, dim: dict):
        inventory = gpd.read_file(self._io.config.inventory_path['shp'])
        inventory.to_crs(self._CRS, inplace=True)

        inventory = inventory[
            ~inventory.TYP_CODE.isin(['Bergsturz', 'Blocksturz', 'Erdfall', 'Felssturz', 'Steinschlag'])]
        inventory = inventory[['OBJEKTID', 'EREIG_ZEIT', 'geometry']].rename(
            columns={'OBJEKTID': 'landslide_id', 'EREIG_ZEIT': 'date'})

        # Filter by coordinates
        inventory = inventory[(inventory.geometry.x >= dim['min_lon']) & (inventory.geometry.x <= dim['max_lon']) &
                              (inventory.geometry.y >= dim['min_lat']) & (inventory.geometry.y <= dim['max_lat'])]

        # Fix dates
        date1 = pd.to_datetime(inventory['date'], errors='coerce', format='%Y')
        date2 = pd.to_datetime(inventory['date'], errors='coerce', format='%m.%Y')
        date3 = pd.to_datetime(inventory['date'], errors='coerce', format='%d.%m.%Y')
        date3 = date3.fillna(date2)
        date3 = date3.fillna(date1)
        inventory['date'] = date3

        return inventory[(inventory.date >= self._date_start) & (inventory.date < self._date_end)]

    def _load_polygon_inventory(self, dim: dict):
        inventory = gpd.read_file(self._io.config.inventory_poly_path['shp'])
        inventory.to_crs(self._CRS, inplace=True)
        inventory = inventory[['OBJECTID', 'DATUM_VON', 'geometry']].rename(
            columns={'OBJECTID': 'landslide_id', 'DATUM_VON': 'date'})

        # Filter by coordinates (geometry object is a POLYGON)
        inventory = inventory.cx[dim['min_lon']:dim['max_lon'], dim['min_lat']:dim['max_lat']]

        # Fix dates
        inventory['date'] = pd.to_datetime(inventory['date'], format='%Y-%m-%d')

        return inventory[(inventory.date >= self._date_start) & (inventory.date < self._date_end)]


if __name__ == '__main__':
    from config.config import Config
    from io_manager.io_manager import IO
    from config.io_config import IOConfig

    config = Config()
    io_config = IOConfig()
    io = IO(io_config)
    module = Evaluation(config, io)
    module.run()
