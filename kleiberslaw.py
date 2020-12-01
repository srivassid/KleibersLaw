import pandas as pd
import glob
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.display.width = 0

class Klieber():

    def __init__(self):
        pass

    def read_files(self):
        self.files = glob.glob('cities/electricity/*.csv')
        self.df = pd.concat(pd.read_csv(f) for f in self.files)
        self.df['City Name'] = self.df['City Name'].astype(str)
        self.df['City Name'] = self.df['City Name'].apply(lambda x:x.split(' ')[0])
        self.df['City Name'] = self.df['City Name'].apply(lambda x: x.split('-')[0])

        self.df = self.df.groupby(['City Name']).agg({'Consumption of Electricity (in lakh units)-Total Consumption':'sum',
                                                      'Population':'first'}).reset_index()
        self.df = self.df.rename(columns={'Consumption of Electricity (in lakh units)-Total Consumption':'Electricity'})
        self.df['ElectricityPerCapita'] = (self.df['Electricity'] / self.df['Population'])
        self.df["ElectricityLog"] = np.log10(self.df['Electricity'])
        self.df['ElectricityPerCapitaLog'] = np.log2(self.df['Electricity'] / self.df['Population'])
        self.df['PopulationLog'] = np.log2(self.df[['Population']])

        self.df = self.df.sort_values(by='ElectricityPerCapitaLog', ascending=True)
        print(self.df)
        return self.df

    def plot_data(self, df):
        f, ax = plt.subplots(figsize=(10, 10))
        # ax.set(xscale="log", yscale="log")
        # # slope, intercept, r_value, p_value, std_err = stats.linregress(df[['Population']], df[['Electricity']])

        g = sns.regplot(df[['PopulationLog']], df[['ElectricityPerCapitaLog']] , ax=ax,  fit_reg=True)
        g.set(xlim=(1.5,None))
        def label_point(x, y, val, ax):
            a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
            for i, point in a.iterrows():
                ax.text(point['x'] + .02, point['y'], str(point['val'].iloc[0]))

        label_point(df[['PopulationLog']], df[['ElectricityPerCapitaLog']], df[['City Name']], plt.gca())
        plt.show()

class StreetLights():

    def __init__(self):
        pass

    def street_lights(self):
        self.files = glob.glob('cities/streetLights/*.csv')
        self.df = pd.concat(pd.read_csv(f, engine='python', usecols=['City Name','Number of Poles','Population']) for f in self.files)
        self.df = self.df.groupby(['City Name']).agg({'Number of Poles':'sum', 'Population':'first'}).reset_index()
        self.df['PolesPerCapita'] = self.df['Number of Poles'] / (self.df['Population'] * 100000)
        self.df['PolesPerCapitaLog'] = np.log2(self.df['PolesPerCapita'])
        self.df['PopulationLog'] = np.log10(self.df['Population'])
        self.df['NumberofPolesLog'] = np.log10(self.df['Number of Poles'])
        self.df = self.df.sort_values(by='PopulationLog',ascending=True)
        print(self.df.head(10))
        print(self.df.shape)
        return self.df

    def plot_data(self, df):
        f, ax = plt.subplots(figsize=(10, 10))
        # ax.set(xscale="log", yscale="log")
        g = sns.regplot(df[['PopulationLog']], df[['NumberofPolesLog']], ax=ax, fit_reg=True)
        def label_point(x, y, val, ax):
            a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
            for i, point in a.iterrows():
                ax.text(point['x'] + .02, point['y'], str(point['val'].iloc[0]))

        label_point(df[['PopulationLog']], df[['NumberofPolesLog']], df[['City Name']], plt.gca())
        plt.show()

class Roads():

    def __init__(self):
        pass

    def read_files(self):
        self.files = glob.glob('cities/roads/*.csv')
        self.df = pd.concat(
            pd.read_csv(f, engine='python', usecols=['City Name', 'Length of Roads (in km)', 'Population']) for f in self.files)
        self.df = self.df.groupby(['City Name']).agg({'Length of Roads (in km)': 'sum',
                                                      'Population':'first'}).reset_index()
        self.df['RoadLengthPerCapita'] = self.df['Length of Roads (in km)'] / self.df['Population']
        self.df['RoadLengthPerCapitaLog'] = np.log2(self.df['RoadLengthPerCapita'])
        self.df['PopulationLog'] = np.log10(self.df['Population'])
        self.df['LengthOfRoadsLog'] = np.log10(self.df['Length of Roads (in km)'])
        self.df = self.df.sort_values(by='RoadLengthPerCapita',ascending=False)
        print(self.df)
        return self.df

    def plot_data(self, df):
        f, ax = plt.subplots(figsize=(10, 10))
        # ax.set(xscale="log", yscale="log")
        g = sns.regplot(df[['PopulationLog']], df[['LengthOfRoadsLog']], ax=ax, fit_reg=True)

        def label_point(x, y, val, ax):
            a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
            for i, point in a.iterrows():
                ax.text(point['x'] + .02, point['y'], str(point['val'].iloc[0]))

        label_point(df[['PopulationLog']], df[['LengthOfRoadsLog']], df[['City Name']], plt.gca())
        plt.show()

if __name__ == '__main__':
    kobj = Klieber()
    df = kobj.read_files()
    kobj.plot_data(df)

    lightsObj = StreetLights()
    df = lightsObj.street_lights()
    lightsObj.plot_data(df)

    road = Roads()
    df = road.read_files()
    road.plot_data(df)
