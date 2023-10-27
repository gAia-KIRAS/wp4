import geopandas as gpd
import pandas as pd

from modules.abstract_module import Module


class Evaluation(Module):
    def __init__(self, config, io):
        super().__init__(config, io)
        self._cd_id = self._config.eval_conf['cd_id']
        self._results = self._io.get_results_cd()

        self._CRS = 4326

        # Some general time limits. Will be filtered in detail when merging with the results
        self._date_start = '2018-01-01'
        self._date_end = '2022-12-31'
        self._max_dist = 0.005

    def run(self, on_the_server=False):
        # Prepare results
        results = self._prepare_results()

        # Get min and max lat = y / lon = x
        dim = {
            'min_lat': results.geometry.y.min(),
            'max_lat': results.geometry.y.max(),
            'min_lon': results.geometry.x.min(),
            'max_lon': results.geometry.x.max()
        }

        if self._config.eval_conf['type'] != 'polygons':
            # Load inventory (ground truth)
            inventory = self._load_inventory(dim)
            print(f'Loaded {len(inventory)} inventory entries for cd_id {self._cd_id}')

            # For every prediction, add 0 / 1 depending on whether there is a close match with the inventory
            results_gt = gpd.sjoin_nearest(results, inventory, how='left', distance_col='distance',
                                           max_distance=self._max_dist).sort_values(by='distance')
            results_gt['date_distance'] = (results_gt['date'] - results_gt['detected_breakpoint']).dt.days
            results_gt = results_gt.merge(inventory[['landslide_id', 'geometry']].rename(
                columns={'geometry': 'geometry_gt'}
            ), on='landslide_id', how='left')

            results_gt.drop_duplicates(subset=['lon', 'lat', 'detected_breakpoint'], inplace=True)

            # If there is no match, set y = 0
            results_gt['y'] = 0
            results_gt.loc[(results_gt['distance'] < self._max_dist) & (results_gt['date_distance'] <= 30) &
                           (results_gt['date_distance'] >= -30), 'y'] = 1

        if self._config.eval_conf['type'] != 'points':
            # Load polygon inventory
            inventory = self._load_polygon_inventory(dim)
            print(f'Loaded {len(inventory)} inventory entries for cd_id {self._cd_id}')

            # For every prediction, add 0 / 1 depending on whether there is a close match with the inventory
            results_gt_poly = gpd.sjoin(results, inventory, how='left', predicate='intersects')
            results_gt_poly['y'] = 0
            results_gt_poly.loc[results_gt_poly['landslide_id'].notnull(), 'y'] = 1
            results_gt_poly = (results_gt_poly.groupby(['detected_breakpoint', 'lat', 'lon'], as_index=False).
                               agg({'y': 'max'}))

        # Merge results
        # Create empty with columns detected_breakpoint, lat, lon
        eval_df = results[['detected_breakpoint', 'lat', 'lon']].drop_duplicates()
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

        # Calculate the number of true positives, false positives, true negatives, false negatives
        tp = len(eval_df[eval_df['y'] == 1])
        fp = len(results) - tp

        # Calculate the precision
        precision = tp / (tp + fp)

        print(f"""
        --------------------
        Evaluation Results
        --------------------
        True positives: {tp}
        False positives: {fp}
        
        Precision: {precision}
        """)

        return 0

    def _prepare_results(self):
        results = self._results[self._results['cd_id'] == self._cd_id]
        results = results[['detected_breakpoint', 'lat', 'lon', 'd_prob']].sort_values(by='d_prob', ascending=False)
        results['detected_breakpoint'] = pd.to_datetime(results['detected_breakpoint'], format='%Y%m%d')
        assert len(results) > 0, f'No results found for cd_id {self._cd_id}'

        # Convert to GeoDataFrame using the coordinates
        results = gpd.GeoDataFrame(results, geometry=gpd.points_from_xy(results['lon'], results['lat']), crs=self._CRS)
        results.to_crs(self._CRS, inplace=True)

        # Remove duplicates in lat, lon, detected_breakpoint. Keep highest probability
        results.drop_duplicates(subset=['lon', 'lat', 'detected_breakpoint'], inplace=True)
        return results

    def _load_inventory(self, dim: dict):
        inventory = gpd.read_file(self._io.config.inventory_path['shp'])
        inventory.to_crs(self._CRS, inplace=True)
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
