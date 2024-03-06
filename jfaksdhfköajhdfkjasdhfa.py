import os
import urllib
import zipfile
from math import radians, sin, cos, asin, sqrt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import folium

class AirportData():
    filenames = ['routes.csv', 'airports.csv', 'airplanes.csv', 'airlines.csv']
    download_dir = 'downloads/'
    data_url = 'https://gitlab.com/adpro1/adpro2024/-/raw/main/Files/flight_data.zip?inline=false'
    zip_file_name = 'flight_data.zip'
    download_necessary = False

    def __init__(self):
        # Checking whether files are aleady downloaded
        for filename in self.filenames:
            if not os.path.exists(self.download_dir + filename):
                print(f"Datafile '{self.download_dir + filename}' not found.")
                self.download_necessary = True
            else:
                print(f"File {self.download_dir + filename}' found.")

        # Download if necessary
        if self.download_necessary:
            print('Downloading all files...')
            # Download the file
            urllib.request.urlretrieve(self.data_url, self.download_dir + self.zip_file_name)

            # Check if the file has been successfully downloaded
            if os.path.exists(self.download_dir + self.zip_file_name):
                # Extract the downloaded zip file
                with zipfile.ZipFile(self.download_dir + self.zip_file_name, 'r') as zip_ref:
                    zip_ref.extractall(self.download_dir)
                print("Download and extraction complete.")
            else:
                print("Failed to download data files.")

        # Reading data into dataframes
        print('Importing Datafiles...')
        self.routes_df = pd.read_csv(self.download_dir + self.filenames[0], index_col='index')
        self.airports_df = pd.read_csv(self.download_dir + self.filenames[1], index_col='index')
        self.airplanes_df = pd.read_csv(self.download_dir + self.filenames[2], index_col='index')
        self.airlines_df = pd.read_csv(self.download_dir + self.filenames[3], index_col='index')
        print('Import done.')

        # TODO: Removing superfluous columns

        # Data cleaning
        # routes_df: replace '\N' with NaN
        self.routes_df["Source airport ID"] = self.routes_df["Source airport ID"].replace('\\N', np.nan)
        self.routes_df["Airline ID"] = self.routes_df["Airline ID"].replace('\\N', np.nan)
        self.routes_df["Destination airport ID"] = self.routes_df["Destination airport ID"].replace('\\N', np.nan)
        self.routes_df = self.routes_df.dropna(subset=['Source airport ID', 'Destination airport ID'])

        # Datatype casting
        # routes_df
        self.routes_df['Airline ID'] = self.routes_df["Airline ID"].astype(float)
        self.routes_df['Source airport ID'] = self.routes_df["Source airport ID"].astype(float)
        self.routes_df['Destination airport ID'] = self.routes_df["Destination airport ID"].astype(float)

        # airports_df
        self.airports_df['Airport ID'] = self.airports_df["Airport ID"].astype(float)


    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        r = 6371  # Radius of Earth in kilometers
        return c * r

    def get_coordinates_and_calculate_distance(self, row, airports_df):
        try:
            source_id = row['Source airport ID']
            destination_id = row['Destination airport ID']

            source_airport = airports_df.loc[airports_df['Airport ID'] == source_id]
            destination_airport = airports_df.loc[airports_df['Airport ID'] == destination_id]

            source_latitude = source_airport['Latitude'].values[0]
            source_longitude = source_airport['Longitude'].values[0]

            destination_latitude = destination_airport['Latitude'].values[0]
            destination_longitude = destination_airport['Longitude'].values[0]

        except Exception as e:
            # should only happen if airports cannot be matched over their id, cannot be found
            #print(':(')
            #print(e)
            pass

        else:
            return self.calculate_distance(source_latitude, source_longitude, destination_latitude, destination_longitude)

    def distance_analysis(self):
        """
        Analyze and plot the distribution of flight distances.

        This method iterates over the DataFrame containing flight route data,
        calculates the great circle distance between origin and destination airports,
        and plots the distribution of these distances.
        """

        self.routes_df['Distance'] = self.routes_df.apply(lambda row: self.get_coordinates_and_calculate_distance(row, self.airports_df), axis=1)

        # Plotting the distribution of distances
        plt.figure(figsize=(10, 6))
        plt.hist(self.routes_df['Distance'], bins=100, color='blue', alpha=0.7)
        plt.title('Distribution of Flight Distances')
        plt.xlabel('Distance (km)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

    def plot_airports_in_country(self, country):
        """
        Plots the airports in the specified country on a map.

        Args:
            country (str): The name of the country.

        Returns:
            None

        Raises:
            None
        """

        if country in self.airports_df['Country'].values:
            airports_to_plot = self.airports_df.loc[self.airports_df['Country'] == country]

            # create a map
            this_map = folium.Map(prefer_canvas=True)

            def plotDot(dataframe):
                '''input: series that contains a numeric named latitude and a numeric named longitude
                this function creates a CircleMarker and adds it to your this_map'''
                folium.CircleMarker(location=[dataframe.Latitude, dataframe.Longitude],
                                    radius=2,
                                    weight=5).add_child(
                    folium.Tooltip(f"ID {int(dataframe['Airport ID'])} - {dataframe.Name}")).add_to(this_map)

            # use df.apply(,axis=1) to "iterate" through every row in your dataframe
            airports_to_plot.apply(plotDot, axis=1)

            # Set the zoom to the maximum possible
            this_map.fit_bounds(this_map.get_bounds())
            display(this_map)
        else:
            print('Country ' + country + ' cannot be found in the airports dataframe')

    def plot_flights_from_airport(self, airport_code, internal=False):

        """
            Plots flights from a given airport on a map.

            Parameters:
            - airport_code (str): The code of the airport to plot flights from.
            - internal (bool, optional): If True, only plots flights to airports in the same country as the source airport.
                Defaults to False.

            This method plots flights from a given airport on a map using the Folium library. It retrieves the coordinates
            of the source and destination airports from the airports dataframe and adds markers and polylines to the map
            to represent the flights. If the internal parameter is set to True, it only plots flights to airports in the
            same country as the source airport.

            If the airport code is not found in the routes dataframe, it prints a message indicating that the airport
            cannot be found.

            Note: This method requires the Folium library to be installed.

            Example usage:
            >>> plot_flights_from_airport('JFK', internal=True)

        """

        if airport_code in self.routes_df['Source airport'].values:
            routes_to_plot = self.routes_df.loc[self.routes_df['Source airport'] == airport_code]
            if internal:
                airport_country = \
                    self.airports_df.loc[self.airports_df['Airport ID'] == routes_to_plot.iloc[0]['Source airport ID']][
                        'Country'].values[0]

                dest_airports_to_plot = self.airports_df.loc[self.airports_df['Country'] == airport_country]['IATA']
                routes_to_plot = routes_to_plot[routes_to_plot['Destination airport'].isin(dest_airports_to_plot)]

            def get_coordinates(airportID):
                airport_lat = self.airports_df.loc[self.airports_df['Airport ID'] == airportID]['Latitude'].values[0]
                airport_long = self.airports_df.loc[self.airports_df['Airport ID'] == airportID]['Longitude'].values[0]
                return airport_lat, airport_long

            routes_to_plot['Source lat'] = routes_to_plot['Source airport ID'].apply(lambda x: get_coordinates(x)[0])
            routes_to_plot['Source long'] = routes_to_plot['Source airport ID'].apply(lambda x: get_coordinates(x)[1])

            routes_to_plot['Destination lat'] = routes_to_plot['Destination airport ID'].apply(
                lambda x: get_coordinates(x)[0])
            routes_to_plot['Destination long'] = routes_to_plot['Destination airport ID'].apply(
                lambda x: get_coordinates(x)[1])

            # create a map
            this_map = folium.Map(prefer_canvas=True)

            def plotRoute(dataframe):
                folium.CircleMarker(location=[dataframe['Source lat'], dataframe['Source long']],
                                    radius=2,
                                    weight=5).add_child(
                    folium.Tooltip(f"ID: {int(dataframe['Source airport ID'])} - {dataframe['Source airport']}")).add_to(
                    this_map)
                folium.CircleMarker(location=[dataframe['Destination lat'], dataframe['Destination long']],
                                    radius=2,
                                    weight=5).add_child(folium.Tooltip(
                    f"ID: {int(dataframe['Destination airport ID'])} - {dataframe['Destination airport']}")).add_to(this_map)
                folium.PolyLine([[dataframe['Source lat'], dataframe['Source long']],
                                 [dataframe['Destination lat'], dataframe['Destination long']]]).add_to(this_map)

            # use df.apply(,axis=1) to "iterate" through every row in your dataframe
            routes_to_plot.apply(plotRoute, axis=1)

            # Set the zoom to the maximum possible
            this_map.fit_bounds(this_map.get_bounds())
            display(this_map)

        else:
            print('Airport ' + airport_code + ' cannot be found in the routes dataframe')


    def plot_flights_from_country(self, country, internal=False):
        """
        Plots flights from a specific country on a map.

        Parameters:
            country (str): The name of the country.
            internal (bool, optional): If True, only plots internal flights within the country.
                                      If False, plots all flights from the country. Default is False.
        """

        if country in self.airports_df['Country'].values:
            src_airports_to_plot = self.airports_df.loc[self.airports_df['Country'] == country]['Airport ID']
            routes_to_plot = self.routes_df[self.routes_df['Source airport ID'].isin(src_airports_to_plot)]

            if internal:
                dest_airports_to_plot = self.airports_df.loc[self.airports_df['Country'] == country]['IATA']
                routes_to_plot = routes_to_plot[routes_to_plot['Destination airport'].isin(dest_airports_to_plot)]

            def get_coordinates(airportID):
                airport_lat = self.airports_df.loc[self.airports_df['Airport ID'] == airportID]['Latitude'].values[0]
                airport_long = self.airports_df.loc[self.airports_df['Airport ID'] == airportID]['Longitude'].values[0]
                return airport_lat, airport_long

            routes_to_plot['Source lat'] = routes_to_plot['Source airport ID'].apply(lambda x: get_coordinates(x)[0])
            routes_to_plot['Source long'] = routes_to_plot['Source airport ID'].apply(lambda x: get_coordinates(x)[1])

            routes_to_plot['Destination lat'] = routes_to_plot['Destination airport ID'].apply(
                lambda x: get_coordinates(x)[0])
            routes_to_plot['Destination long'] = routes_to_plot['Destination airport ID'].apply(
                lambda x: get_coordinates(x)[1])

            # create a map
            this_map = folium.Map(prefer_canvas=True)

            def plotRoute(dataframe):
                folium.CircleMarker(location=[dataframe['Source lat'], dataframe['Source long']],
                                    radius=2,
                                    weight=5).add_child(
                    folium.Tooltip(f"ID: {int(dataframe['Source airport ID'])} - {dataframe['Source airport']}")).add_to(
                    this_map)
                folium.CircleMarker(location=[dataframe['Destination lat'], dataframe['Destination long']],
                                    radius=2,
                                    weight=5).add_child(folium.Tooltip(
                    f"ID: {int(dataframe['Destination airport ID'])} - {dataframe['Destination airport']}")).add_to(this_map)
                folium.PolyLine([[dataframe['Source lat'], dataframe['Source long']],
                                 [dataframe['Destination lat'], dataframe['Destination long']]]).add_to(this_map)

            # use df.apply(,axis=1) to "iterate" through every row in your dataframe
            routes_to_plot.apply(plotRoute, axis=1)

            # Set the zoom to the maximum possible
            this_map.fit_bounds(this_map.get_bounds())
            display(this_map)

        else:
            print('Country ' + country + ' cannot be found in the airports dataframe')


    def models_per_rout_country(self, country=False, N=10):
        """
        Calculate and plot the N most common airplane models per route for a given country.

        Parameters:
        - country (str): The country to filter the routes by. If not provided, all routes will be considered.
        - N (int): The number of top airplane models to display. Default is 10.
        Calculate and plot the N most common airplane models per route for a given country.

        Returns: Plot of the N most common airplane models per route for a given country.
        None
        """

        # Merge dataframes
        routes_airports = self.routes_df.merge(self.airports_df, left_on='Source airport ID', right_on='Airport ID')
        routes_airports_planes = routes_airports.merge(self.airplanes_df, how="left", left_on='Equipment', right_on='IATA code')

        # filter routes by selected country
        routes_country = routes_airports_planes[routes_airports_planes['Country'] == country] if country else routes_airports_planes

        # group by airplane model and count number of routes
        airplane_model_counts = routes_country.groupby('Name_y').size().reset_index(name='Count')

        # Sort by count in descending order
        airplane_model_counts = airplane_model_counts.sort_values('Count', ascending=False)

        # Select the top N airplane models
        top_airplane_models = airplane_model_counts.head(N)

        # Plot the N most used airplane models
        top_airplane_models.plot(x='Name_y', y='Count', kind='bar', xlabel='Airplane Model', ylabel='Number of Routes', title="Most common models per route for " + country)

