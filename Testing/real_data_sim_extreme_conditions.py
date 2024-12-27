import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18

class RealData:
    def __init__(self, file_path, start_date, end_date):
        # Read the data from CSV
        self.data = pd.read_csv(file_path)
        
        # Convert time column to datetime with timezone
        self.data['time'] = pd.to_datetime(self.data['time'], utc=True)
        
        # Set time column as the index
        self.data.set_index('time', inplace=True)
        
        # Define column groups for easier reference
        self.renewable_columns = [
            'generation biomass', 'generation geothermal', 
            'generation hydro run-of-river and poundage', 'generation hydro water reservoir',
            'generation marine', 'generation other renewable', 'generation solar', 
            'generation waste', 'generation wind offshore', 'generation wind onshore'
        ]
        self.non_renewable_columns = [
            'generation fossil brown coal/lignite', 'generation fossil coal-derived gas', 
            'generation fossil gas', 'generation fossil hard coal', 'generation fossil oil', 
            'generation fossil oil shale', 'generation fossil peat', 'generation nuclear', 
            'generation other'
        ]
        self.price_column = 'price actual'
        self.storage_column = 'generation hydro pumped storage consumption'
        self.forecast_columns = ['price day ahead']
        self.load_demand_column = 'total load actual'
        
        # Filter data for the given date range
        self.data_month = self.data.loc[start_date:end_date]
        
        # Remove zero values
        self.data_month = self.data_month[
            (self.data_month[self.renewable_columns].sum(axis=1) != 0) &
            (self.data_month[self.non_renewable_columns].sum(axis=1) != 0) &
            (self.data_month[self.price_column] != 0) &
            (self.data_month[self.storage_column] != 0) &
            (self.data_month[self.forecast_columns].sum(axis=1) != 0) &
            (self.data_month[self.load_demand_column] != 0)
        ]
        
        # Calculate total values for each category
        self.total_renewable = self.data_month[self.renewable_columns].sum(axis=1)
        self.total_non_renewable = self.data_month[self.non_renewable_columns].sum(axis=1)
        self.total_forecast = self.data_month[self.forecast_columns].sum(axis=1)
        self.storage_s = self.data_month[self.storage_column]
        self.load_demand = self.data_month[self.load_demand_column]
        
        # Adjust non-renewable to match load demand
        self.storage_s = abs(self.storage_s)
        self.total_renewable = self.total_renewable - self.storage_s
        self.load_demand = self.load_demand + self.storage_s
        
        # Normalize values to be between 0 and 1
        self.total_renewable = self.total_renewable / self.total_renewable.max()
        self.total_non_renewable = (self.total_non_renewable / self.total_non_renewable.max())-0.3
        self.storage_s = (self.storage_s / self.storage_s.max()) * 0.2  # Limit storage between 0 and 0.2
        self.load_demand = self.load_demand / self.load_demand.max()
        self.total_forecast = self.total_forecast / self.total_forecast.max()
        
        # Splitting total renewable energy into mg1, mg2, mg3
        self.mg1 = self.total_renewable * np.random.uniform(0, 0.4)
        self.mg2 = self.total_renewable * np.random.uniform(0, 0.3)
        self.mg3 = self.total_renewable - self.mg1 - self.mg2
        
        self.pv1 = self.mg1 * np.random.uniform(0, 0.4)
        self.wt1 = self.mg1 - self.pv1
        self.pv2 = self.mg2 * np.random.uniform(0, 0.4)
        self.wt2 = self.mg2 - self.pv2
        self.pv3 = self.mg3 * np.random.uniform(0, 0.4)
        self.wt3 = self.mg3 - self.pv3
        
        # Splitting total battery storage into bc1, bc2, bc3
        self.total_battery = self.storage_s
        self.bc1 = self.total_battery * np.random.uniform(0.3, 0.4)
        self.bc2 = self.total_battery * np.random.uniform(0.2, 0.3)
        self.bc3 = self.total_battery - self.bc1 - self.bc2
        
        # Creating FLoad (Load Demand with added noise)
        self.FLoad = self.load_demand + np.random.normal(0, 0.5 * self.load_demand.std(), self.load_demand.shape)
        self.total_non_renewable = self.load_demand - (self.total_renewable + self.storage_s)
        self.total_non_renewable = abs(self.total_non_renewable)

    def plot_microgrid_generation(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.data_month.index, self.mg1, label='MG1')
        plt.plot(self.data_month.index, self.mg2, label='MG2')
        plt.plot(self.data_month.index, self.mg3, label='MG3')
        plt.xlabel('Time')
        plt.ylabel('Energy [Normalized]')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_energy_storage_load(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.data_month.index, self.total_renewable, label='RE')
        plt.plot(self.data_month.index, self.total_non_renewable, label='NR')
        plt.plot(self.data_month.index, self.load_demand, label='Load')
        plt.plot(self.data_month.index, self.FLoad, label='FLoad', linestyle='--')
        plt.ylabel('Energy [Normalized]')
        plt.xlabel('Time')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_price_forecast(self):
        # Add noise to the actual price to create the forecast price
        noise = np.random.normal(0, 0.55 * self.data_month[self.price_column].std(), self.data_month[self.price_column].shape)
        self.total_forecast = self.data_month[self.price_column] + noise
        
        # Normalize the actual price and forecast price
        actual_price_normalized = self.data_month[self.price_column] / self.data_month[self.price_column].max()
        forecast_price_normalized = self.total_forecast / self.total_forecast.max()
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.data_month.index, actual_price_normalized, label='Actual Price', color='red')
        plt.plot(self.data_month.index, forecast_price_normalized, label='FPrice', color='blue')
        plt.ylabel('Price')
        plt.xlabel('Time')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

real_data = RealData('energy_dataset.csv', '2017-08-20', '2017-08-27')
real_data.plot_microgrid_generation()
real_data.plot_energy_storage_load()
real_data.plot_price_forecast()