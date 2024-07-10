import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, lag, avg
from pyspark.sql.window import Window
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
#import pandas as pd

# spark session
spark = SparkSession.builder \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .appName("WeatherForecasting") \
    .getOrCreate()

class WeatherApp:
    def __init__(self, root):
        # main app window
        self.root = root
        self.root.title("Weather Prediction App")
        # create frame for content
        self.frame = tk.Frame(root)
        self.frame.pack(fill=tk.BOTH, expand=1)
        # create scrolled text widget for data table
        self.text = scrolledtext.ScrolledText(self.frame, width=120, height=20)
        self.text.pack(padx=10, pady=10)
        # create a load-data button
        self.load_button = ttk.Button(self.frame, text="Load Data and Plot", command=self.load_data_and_plot)
        self.load_button.pack(padx=10, pady=10)
        
    def load_data_and_plot(self):
        # select csv file
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        
        if file_path:
            # load and preprocess data
            weather_data = spark.read.csv(file_path, header=True, inferSchema=True)
            weather_data = weather_data.select("Date", "Temperature AVG", "Relative Humidity AVG", "Wind Speed Daily AVG", "Wind Direction AVG", "Solare Radiation AVG")
            weather_data = weather_data.na.drop()
            weather_data = weather_data.withColumn("Date", to_date(col("Date"), "MM/dd/yyyy"))
            # generate lagged values and rolling average
            window = Window.orderBy("Date")
            weather_data = weather_data.withColumn("lag_1_temp", lag("Temperature AVG", 1).over(window))
            weather_data = weather_data.withColumn("lag_2_temp", lag("Temperature AVG", 2).over(window))
            weather_data = weather_data.withColumn("RollingAvg_Temperature", avg("Temperature AVG").over(window.rowsBetween(-7, 0)))
            weather_data = weather_data.na.drop()
            # split data into training and testing sets
            train_data, test_data = weather_data.randomSplit([0.8, 0.2], seed=1234)
            # features and model pipelines
            feature_cols = ["lag_1_temp", "lag_2_temp", "RollingAvg_Temperature", 'Relative Humidity AVG', 'Wind Speed Daily AVG', "Wind Direction AVG", "Solare Radiation AVG"]
            assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
            rf = RandomForestRegressor(labelCol="Temperature AVG", featuresCol="features")
            pipeline = Pipeline(stages=[assembler, rf])
            # train the model
            model = pipeline.fit(train_data)
            predictions = model.transform(test_data)
            # display table and graph
            self.display_data(predictions)
            self.plot_graph(predictions)
    
    def display_data(self, data):
        '''
        Display data in table format onto the GUI
        Input: data received from training the model
        '''
        self.text.delete(1.0, tk.END)
        data_pd = data.select("Date", "Temperature AVG", "Prediction").toPandas()
        self.text.insert(tk.END, data_pd.to_string(index=False))
    
    def plot_graph(self, data):
        '''
        Display data in line graph format onto the GUI
        Input: data received from training the model
        '''
        data_pd = data.select("Date", "Temperature AVG", "prediction").toPandas()
        
        fig = Figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        ax.plot(data_pd["Date"], data_pd["Temperature AVG"], label="Actual Temperature")
        ax.plot(data_pd["Date"], data_pd["prediction"], label="Predicted Temperature", linestyle="--")
        ax.set_xlabel("Date")
        ax.set_ylabel("Temperature")
        ax.set_title("Actual vs Predicted Temperature")
        ax.legend()
        
        canvas = FigureCanvasTkAgg(fig, master=self.frame)
        canvas.draw()
        canvas.get_tk_widget().pack(padx=10, pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = WeatherApp(root)
    root.mainloop()
