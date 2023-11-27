import pickle

import pandas as pd
import geopandas as gpd

from config.config import Config
from config.io_config import IOConfig
from io_manager.io_manager import IO
from modules.abstract_module import Module


class ModelFusion(Module):

    def __init__(self, config: Config, io: IO):
        super().__init__(config, io)

        self._path_weather = (f"{self._io.config.base_local_dir}"
                              f"/weather/datasets/gdf_weather_2010_2022.parquet")

    def _read_positives(self):
        df = pd.read_csv(f'{self._io.config.base_local_dir}/operation_records/positives.csv')
        df = df[['detected_breakpoint', 'lat', 'lon']]

        df['detected_breakpoint'] = pd.to_datetime(df['detected_breakpoint'].astype(str), format='%Y-%m-%d')
        df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lon'], df['lat'], crs="EPSG:4326"), crs="EPSG:4326")

        return df

    def run(self, on_the_server: bool = False) -> None:
        # 1. Load the model
        model_pkl_file = f'{self._io.config.base_local_dir}/weather/models/rf_step30_from2010_filtered_25fs.pkl'
        with open(model_pkl_file, 'rb') as file:
            model = pickle.load(file)

        # 2. Print all object info
        feature_names_in = model.feature_names_in_

        # 3. Predict the model on the true positives of the CD model
        positives = self._read_positives()

        # 4. Create the features dataframe
        X = self._build_predict_dataset(positives, feature_names_in)

        # 5. Predict on X
        pred = model.predict(X)
        proba = model.predict_proba(X)

        # 6. Evaluate the model considering that all samples were positives.
        print(f'Number of samples: {len(X)}')
        print(f'Predicted positives: {sum(pred)}')
        print(f'Accuracy: {sum(pred)/len(X)}')

    def _read_weather(self):
        return gpd.read_parquet(self._path_weather)

    def _create_features(self, df, features: list = None, n_days: int = 14):
        if features is None:
            features = ['prec', 'wind', 'relhum', 'temp']

        uniquep_won = df.geometry.unique()
        df.set_index('geometry', inplace=True)
        new_entries = []
        for i in range(0, len(uniquep_won)):
            point = uniquep_won[i]
            X_data = df.loc[point]
            X_data = X_data.reset_index(drop=False)

            land_dates = X_data[X_data.landslide == 1]
            try:
                for idx in land_dates.index:
                    time_window = X_data.iloc[idx - n_days: idx]
                    month_time = X_data.iloc[idx - 30: idx]

                    d = self._calculate_features(time_window, month_time, features)
                    d['landslide'] = X_data.iloc[idx].landslide
                    new_entries.append(pd.DataFrame(d))

            except Exception as e:
                print(e)
        point_data = pd.concat(new_entries, ignore_index=True)
        return point_data

    def _calculate_features(self, time_window, month_time, features):
        cal_weather = {}
        if 'prec' in features:
            cal_weather['prec_max'] = [time_window.prec.max()]
            cal_weather['prec_mean'] = [time_window.prec.mean()]
            cal_weather['prec_min'] = [time_window.prec.min()]
            cal_weather['prec_kurt'] = [time_window.prec.kurtosis()]
            cal_weather['prec_kurt_month'] = [month_time.prec.kurtosis()]
            cal_weather['sum_prec'] = [time_window.prec.sum()]
            cal_weather['sum_prec_month'] = [month_time.prec.sum()]
            cal_weather['prec_skew'] = [time_window.prec.skew()]
            cal_weather['prec_skew_month'] = [month_time.prec.skew()]

        if 'wind' in features:
            cal_weather['wind_max'] = [time_window.wind.max()]
            cal_weather['wind_max_month'] = [month_time.wind.max()]
            cal_weather['wind_mean'] = [time_window.wind.mean()]
            cal_weather['wind_min'] = [time_window.wind.min()]
            cal_weather['wind_skew'] = [time_window.wind.skew()]
            cal_weather['wind_skew_month'] = [month_time.wind.skew()]
            cal_weather['wind_kurt'] = [time_window.wind.kurtosis()]
            cal_weather['wind_kurt_month'] = [month_time.wind.kurtosis()]
            cal_weather['wind_light'] = [len(time_window[time_window.wind <= 5])]
            cal_weather['wind_mod'] = [len(time_window[(time_window.wind > 5) & (time_window.wind <= 11)])]
            cal_weather['wind_strong'] = [len(time_window[(time_window.wind > 11) & (time_window.wind <= 17)])]
            cal_weather['wind_sev'] = [len(time_window[(time_window.wind > 17) & (time_window.wind <= 23)])]
            cal_weather['wind_extr'] = [len(time_window[time_window.wind > 23])]

        if 'mslp' in features:
            cal_weather['mslp_min'] = [time_window.mslp.min()]
            cal_weather['mslp_mean'] = [time_window.mslp.mean()]
            cal_weather['mslp_max'] = [time_window.mslp.max()]
            cal_weather['mslp_high'] = [len(time_window[time_window.mslp >= 102000])]
            cal_weather['mslp_norm'] = [len(time_window[(time_window.mslp > 101300) & (time_window.mslp < 102000)])]
            cal_weather['mslp_shall'] = [len(time_window[(time_window.mslp > 100000) & (time_window.mslp <= 101300)])]
            cal_weather['mslp_low'] = [len(time_window[(time_window.mslp > 98000) & (time_window.mslp <= 100000)])]
            cal_weather['mslp_verylow'] = [len(time_window[time_window.mslp <= 98000])]
            cal_weather['mslp_skew'] = [time_window.mslp.skew()]
            cal_weather['mslp_skew_month'] = [month_time.mslp.skew()]
            cal_weather['mslp_kurt'] = [time_window.mslp.kurtosis()]
            cal_weather['mslp_kurt_month'] = [month_time.mslp.kurtosis()]

        if 'relhum' in features:
            cal_weather['relhum_min'] = [time_window.relative_humidity.min()]
            cal_weather['relhum_mean'] = [time_window.relative_humidity.mean()]
            cal_weather['relhum_max'] = [time_window.relative_humidity.max()]
            cal_weather['relhum_veryhigh'] = [len(time_window[time_window.relative_humidity > 80])]
            cal_weather['relhum_high'] = [
                len(time_window[(time_window.relative_humidity >= 60) & (time_window.relative_humidity <= 80)])]
            cal_weather['relhum_norm'] = [
                len(time_window[(time_window.relative_humidity >= 30) & (time_window.relative_humidity < 60)])]
            cal_weather['relhum_low'] = [len(time_window[time_window.relative_humidity < 30])]
            cal_weather['relhum_skew'] = [time_window.relative_humidity.skew()]
            cal_weather['relhum_skew_month'] = [month_time.relative_humidity.skew()]
            cal_weather['relhum_kurt'] = [time_window.relative_humidity.kurtosis()]
            cal_weather['relhum_kurt_month'] = [month_time.relative_humidity.kurtosis()]

        if 'temp' in features:
            cal_weather['temp_skew'] = [time_window.temp.skew()]
            cal_weather['temp_skew_month'] = [month_time.temp.skew()]
            cal_weather['temp_max'] = [time_window.temp.max()]
            cal_weather['temp_min'] = [time_window.temp.min()]
            cal_weather['temp_mean'] = [time_window.temp.mean()]
            cal_weather['temp_mean_month'] = [month_time.temp.mean()]
            cal_weather['temp_kurt'] = [time_window.temp.kurtosis()]
            cal_weather['temp_kurt_month'] = [month_time.temp.kurtosis()]
            cal_weather['temp_std'] = [time_window.temp.std()]
        return cal_weather


    def _merge_calc_features(self, positives, df_weather):
        common_points = gpd.sjoin_nearest(positives, df_weather[df_weather.time == "2018-04-01"],
                                          how='inner', lsuffix='inv', rsuffix='weather')
        land_filtered = common_points[['detected_breakpoint', 'geometry', 'weath_geom']].reset_index(
            drop=True)

        df_merged = pd.merge(df_weather, land_filtered, left_on='geometry', right_on='weath_geom').drop(
            ['weath_geom_x', 'weath_geom_y'], axis=1)
        df_merged = df_merged.rename({'geometry_x': 'geometry', 'geometry_y': 'land_geom'}, axis=1)
        df_merged.loc[df_merged.time == df_merged.detected_breakpoint, 'landslide'] = 1

        df_merged = self._create_features(df_merged, features=['prec', 'wind', 'relhum', 'temp'])

        return df_merged

    def _build_predict_dataset(self, positives, feature_names_in):
        df_weather = self._read_weather()
        df_weather['weath_geom'] = df_weather.geometry  # save geo point of weather data
        df_weather["landslide"] = 0

        df_features = self._merge_calc_features(positives, df_weather)
        X = df_features[feature_names_in]

        return X


if __name__ == '__main__':
    from pyinstrument import Profiler
    with Profiler() as prof:
        config = Config()
        io_config = IOConfig()
        io = IO(io_config)

        model_fusion = ModelFusion(config, io)
        model_fusion.run()
