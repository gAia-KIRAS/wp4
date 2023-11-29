import geopandas as gpd
import pandas as pd
import numpy as np
import warnings
import random

from modules.abstract_module import Module


class Evaluation(Module):
    """
    This module evaluates the results of the change detection module. It compares the predictions with the ground truth
    (inventory) and calculates the precision and recall.

    Attributes:
        _cd_id (str): id of the change detection
        _results (pd.DataFrame): table with the results
        _CRS (int): coordinate reference system
        _aoi (GeoDataFrame): area of interest
        _tile_filters (list): list of tiles to filter the results
        _compute_baseline_eval (bool): whether to compute the baseline evaluation or not
        _build_train_dataset (bool): whether to build the train dataset for the LogRegression model or not
        _date_start (str): start date for the evaluation. It is a static parameter for now (data goes from 2018 to 2022)
        _date_end (str): end date for the evaluation. It is a static parameter for now (data goes from 2018 to 2022)
        _max_dist (float): maximum distance between a prediction and a landslide in order to consider it a match.
        It is given in CRS units, and it corresponds to 6 meters.


    """
    def __init__(self, config, io):
        super().__init__(config, io)
        self._cd_id = self._config.eval_conf['cd_id']
        self._results = self._io.get_results_cd()
        self._CRS = 4326

        self._aoi = self.load_area_of_interest()

        self._tile_filters = self._config.filters['tile']
        self._compute_baseline_eval = self._config.eval_conf['baseline_eval']
        self._build_train_dataset = self._config.eval_conf['build_train_dataset']
        self._take_positives = self._config.eval_conf['take_positives']

        try:
            df = pd.read_csv(f'{io.config.base_local_dir}/operation_records/train_features.csv')
            self._train_feat = df[['row', 'column', 'year', 'tile', 'date']].rename(
                columns={'date': 'date_pred'}
            )
            self._train_feat['date_pred'] = pd.to_datetime(self._train_feat['date_pred'], format='%Y-%m-%d')
        except FileNotFoundError:
            self._train_feat = pd.DataFrame(columns=['row', 'column', 'year', 'tile', 'date_pred'])

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

    def run(self, on_the_server: bool = False) -> None:
        """
        Runs the evaluation for the cd_id specified in the config.

        Args:
            on_the_server: whether execution is on the server or not
        """
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

        if self._build_train_dataset:
            self._build(eval_df)

        if self._take_positives:
            self._take_positives_and_save(eval_df)

    def _take_positives_and_save(self, eval_df) -> None:
        eval_df[eval_df['y'] == 1].to_csv(f'{self._io.config.base_local_dir}'
                                          f'/operation_records/positives.csv', index=False)

    def _build(self, eval_df) -> None:
        """
        Builds the train dataset for the LogRegression model. It takes 100 points with y = 1 and 100 points with y = 0.
        Saves the dataset in operation_records/train_inv.csv. This dataset still has no features, it just contains the
        points that will be used for training.

        Args:
            eval_df: table with the evaluation results

        """
        df = pd.DataFrame(columns=['row', 'column', 'date', 'lat', 'lon', 'year', 'tile', 'y'])
        # Take 100 points with y = 1 and 100 points with y = 0
        for y in [0, 1]:
            df = pd.concat([df, eval_df[eval_df['y'] == y].sample(n=100, random_state=42)])
        df.to_csv(f'{self._io.config.base_local_dir}/operation_records/train_inv.csv', index=False)

    @staticmethod
    def disaggregate_results(eval_df, results) -> None:
        """
        Computes and displays precision and recall for every year and tile.

        Args:
            eval_df: (pd.DataFrame) table with the evaluation output
            results: (pd.DataFrame) original table with predictions (not matched with ground truth)

        """
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

    def match_with_inventory(self, detected_landslide_ids, dim, landslide_ids, results) -> tuple:
        """
        Matches the predictions with the point-based inventory. Adds a column y to the results table.

        Args:
            detected_landslide_ids: ids of the landslides that were detected until now
            dim: min and max longitude and latitude
            landslide_ids: ids of the landslides in the inventory
            results: table with the predictions

        Returns:
            detected_landslide_ids: updated ids of the landslides that were detected
            landslide_ids: ids of the landslides in the inventory
            results_gt: table with the predictions and the y column
        """
        # Load inventory (ground truth)
        inventory = self.load_inventory(dim)
        landslide_ids += inventory['landslide_id'].unique().tolist()
        print(f'Loaded {len(inventory)} inventory entries for cd_id {self._cd_id}')

        # For every prediction, add 0 / 1 depending on whether there is a close match with the inventory
        results_gt = gpd.sjoin_nearest(results, inventory, how='left', distance_col='distance',
                                       max_distance=self._max_dist).sort_values(by='distance')
        results_gt['date_distance'] = (results_gt['date'] - results_gt['date_pred']).dt.days
        results_gt = results_gt.merge(inventory[['landslide_id', 'geometry']].rename(
            columns={'geometry': 'geometry_gt'}
        ), on='landslide_id', how='left')
        detected_landslide_ids += results_gt['landslide_id'].unique().tolist()
        detected_landslide_ids = [x for x in detected_landslide_ids if not np.isnan(x)]
        results_gt.drop_duplicates(subset=['lon', 'lat', 'date_pred'], inplace=True)

        # If there is no match, set y = 0
        results_gt['y'] = 0
        results_gt.loc[(results_gt['distance'] < self._max_dist) & (results_gt['date_distance'] <= 30) &
                       (results_gt['date_distance'] >= -30), 'y'] = 1
        return detected_landslide_ids, landslide_ids, results_gt

    def match_with_polygon_inventory(self, detected_landslide_ids, dim, landslide_ids, results) -> tuple:
        """
        Matches the predictions with the polygon-based inventory. Adds a column y to the results table.

        Args:
            detected_landslide_ids: ids of the landslides that were detected until now
            dim: min and max longitude and latitude
            landslide_ids: ids of the landslides in the inventory
            results: table with the predictions

        Returns:
            detected_landslide_ids: updated ids of the landslides that were detected
            landslide_ids: ids of the landslides in the inventory
            results_gt_poly: table with the predictions and the y column
        """
        # Load polygon inventory
        inventory = self.load_polygon_inventory(dim)
        landslide_ids += inventory['landslide_id'].unique().tolist()
        print(f'Loaded {len(inventory)} inventory entries for cd_id {self._cd_id}')

        # For every prediction, add 0 / 1 depending on whether there is a close match with the inventory
        results_gt_poly = gpd.sjoin(results, inventory, how='left', predicate='intersects')
        results_gt_poly['y'] = 0
        results_gt_poly.loc[results_gt_poly['landslide_id'].notnull(), 'y'] = 1
        detected_landslide_ids += results_gt_poly['landslide_id'].unique().tolist()
        detected_landslide_ids = [x for x in detected_landslide_ids if not np.isnan(x)]
        results_gt_poly = (results_gt_poly.groupby(['row', 'column', 'date_pred', 'lat', 'lon'], as_index=False).
                           agg({'y': 'max'}))
        return detected_landslide_ids, landslide_ids, results_gt_poly

    def combine_results(self, results, results_gt, results_gt_poly) -> pd.DataFrame:
        """
        Combines the matches with point and polygon inventories: y = y_points + y_poly

        Args:
            results: original predictions
            results_gt: predictions matches with point inventory
            results_gt_poly: predictions matches with polygon inventory

        Returns:
            eval_df: table with the evaluation results
        """
        # Merge results
        eval_df = results[['row', 'column', 'date_pred', 'lat', 'lon', 'year', 'tile']].drop_duplicates()
        if self._config.eval_conf['type'] != 'polygons':
            eval_df = eval_df.merge(results_gt[['row', 'column', 'date_pred', 'lat', 'lon', 'y']].rename(
                columns={'y': 'y_points'}), on=['row', 'column', 'date_pred', 'lat', 'lon'], how='left')
        if self._config.eval_conf['type'] != 'points':
            eval_df = eval_df.merge(results_gt_poly[['row', 'column', 'date_pred', 'lat', 'lon', 'y']].rename(
                columns={'y': 'y_poly'}), on=['row', 'column', 'date_pred', 'lat', 'lon'], how='left')
        # Create y column (y_points or y_poly)
        if self._config.eval_conf['type'] == 'points':
            eval_df['y'] = eval_df['y_points']
        elif self._config.eval_conf['type'] == 'polygons':
            eval_df['y'] = eval_df['y_poly']
        else:
            eval_df['y'] = eval_df['y_points'] + eval_df['y_poly']
            eval_df['y'] = eval_df['y'].fillna(0)
        return eval_df

    @staticmethod
    def calculate_results(detected_landslide_ids, eval_df, landslide_ids, results) -> tuple:
        """
        Calculates the evaluation results: tp, fp, precision, recall

        Args:
            detected_landslide_ids: list of ids of the landslides that were detected
            eval_df: table with the evaluation results
            landslide_ids: list of ids of the landslides in the inventory
            results: original predictions

        Returns:
            fp, tp, precision, recall
        """

        tp = len(eval_df[eval_df['y'] == 1])
        fp = len(results) - tp
        # Calculate the precision
        precision = tp / (tp + fp)
        # Calculate the recall: how many of the landslides were detected
        recall = len(detected_landslide_ids) / len(landslide_ids)
        return fp, tp, precision, recall

    @staticmethod
    def print_general_results(detected_landslide_ids, fp, landslide_ids, precision, recall, tp) -> None:
        """
        Prints the general evaluation results.

        Args:
            detected_landslide_ids: list of ids of the landslides that were detected
            tp:  number of true positives
            fp: number of false positives
            landslide_ids: list of ids of the landslides in the inventory
            precision: precision of the predictions
            recall: recall of the predictions
        """
        print(f"""
        --------------------
        Evaluation Results
        --------------------
        True positives: {tp}
        False positives: {fp}

        Precision: {precision}
        Recall: {round(recall, 3)} = {len(detected_landslide_ids)} / {len(landslide_ids)}
        """)

    def _compute_baseline_evaluation(self, dim, n) -> None:
        """
        Compute baseline evaluation, which is the evaluation of n random predictions.
        Print results to console.

        Args:
            dim: min and max longitude and latitude
            n: number of random predictions to generate
        """
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

    def generate_random_results(self, dim, n) -> pd.DataFrame:
        """
        Generates n random predictions.

        Args:
            dim: min and max longitude and latitude
            n: number of random predictions to generate

        Returns:
            random_results: table with the random predictions. Same columns and format as the results table.
        """
        # Generate n random predictions. Must be within dim. Date is random between 2018 and 2022
        random_results = pd.DataFrame({
            'date_pred': pd.date_range(start='2018-01-01', end='2022-12-31', periods=n),
            'r': np.random.rand(n),
        })
        random_results['lat'] = round(dim['min_lat'] + (dim['max_lat'] - dim['min_lat']) * random_results['r'], 10)
        random_results['lon'] = round(dim['min_lon'] + (dim['max_lon'] - dim['min_lon']) * random_results['r'], 10)
        random_results['d_prob'] = 1
        random_results['year'] = random_results['date_pred'].dt.year
        random_results['tile'] = '33TUM'
        random_results.drop(columns=['r'], inplace=True)

        # Convert to GeoDataFrame using the coordinates
        random_results = gpd.GeoDataFrame(
            random_results, geometry=gpd.points_from_xy(random_results['lon'], random_results['lat']), crs=self._CRS)

        return random_results

    def _prepare_results(self) -> tuple:
        """
        Loads the results for the cd_id specified in the config.
        Filters according to the tile filters specified in the config.
        Filters according to the area of interest but keep number of results before filtering.

        Returns:
            results: table with the results
            n_before_filter: number of results before filtering by area of interest
        """

        results = self._results[self._results['version'] == self._cd_id]

        results = results[['row', 'column', 'date', 'lat', 'lon', 'probability', 'year', 'tile']].sort_values(
            by='probability', ascending=False)
        if self._tile_filters:
            results = results[results['tile'].isin(self._tile_filters)]
        results['date'] = pd.to_datetime(results['date'], format='%Y-%m-%d')
        results.rename(columns={'date': 'date_pred'}, inplace=True)
        assert len(results) > 0, f'No results found for cd_id {self._cd_id}'

        # Remove points that were in self._train_feat
        self._train_feat['aux'] = 1
        results = results.merge(self._train_feat, on=['row', 'column', 'year', 'tile', 'date_pred'], how='left')
        results = results[results['aux'].isnull()].drop(columns=['aux'])

        # Convert to GeoDataFrame using the coordinates
        results = gpd.GeoDataFrame(results, geometry=gpd.points_from_xy(results['lon'], results['lat']), crs=self._CRS)
        results.to_crs(self._CRS, inplace=True)

        # Remove duplicates in lat, lon, date_pred. Keep the highest probability
        results.drop_duplicates(subset=['lon', 'lat', 'date_pred'], inplace=True)

        # Filter according to area of interest
        n_before_filter = len(results)
        results = gpd.sjoin(results, self._aoi, how='inner', predicate='intersects')
        results.drop(columns=['index_right', 'name', 'area'], inplace=True)

        return results, n_before_filter

    def load_inventory(self, dim: dict = None):
        """
        Loads the inventory for the cd_id specified in the config.
        Filter by type of landslide and by min and max latitude and longitude.
        Filter by date (only keep entries between _date_start and _date_end).

        Args:
            dim: min and max longitude and latitude. No filter if None.

        Returns:
            inventory: table with the inventory
        """
        inventory = gpd.read_file(self._io.config.inventory_path['shp'])
        inventory.to_crs(self._CRS, inplace=True)

        inventory = inventory[
            ~inventory.TYP_CODE.isin(['Bergsturz', 'Blocksturz', 'Erdfall', 'Felssturz', 'Steinschlag'])]
        inventory = inventory[['OBJEKTID', 'EREIG_ZEIT', 'geometry']].rename(
            columns={'OBJEKTID': 'landslide_id', 'EREIG_ZEIT': 'date'})

        # Filter by coordinates
        if dim:
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

    def load_polygon_inventory(self, dim: dict = None):
        """
        Loads the polygon inventory for the cd_id specified in the config.
        Filter by min and max latitude and longitude.
        Filter by date (only keep entries between _date_start and _date_end).

        Args:
            dim: min and max longitude and latitude. No filter if None.

        Returns:

        """
        inventory = gpd.read_file(self._io.config.inventory_poly_path['shp'])
        inventory.to_crs(self._CRS, inplace=True)
        inventory = inventory[['OBJECTID', 'DATUM_VON', 'geometry']].rename(
            columns={'OBJECTID': 'landslide_id', 'DATUM_VON': 'date'})

        # Filter by coordinates (geometry object is a POLYGON)
        if dim:
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
