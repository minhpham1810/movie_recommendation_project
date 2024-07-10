import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, lag, avg
from pyspark.sql.window import Window
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
import pandas as pd

# Initialize Spark session
spark = SparkSession.builder \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .appName("WeatherForecasting") \
    .getOrCreate()

class WeatherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Weather Prediction App")
        
        self.frame = tk.Frame(root)
        self.frame.pack(fill=tk.BOTH, expand=1)
        
        self.text = scrolledtext.ScrolledText(self.frame, width=120, height=20)
        self.text.pack(padx=10, pady=10)
        
        self.plot_button = ttk.Button(self.frame, text="Load Data and Plot", command=self.load_data_and_plot)
        self.plot_button.pack(padx=10, pady=10)
        
    def load_data_and_plot(self):
        # Load and preprocess data
        weather_data = spark.read.csv("C:/Users/MSI PC/Desktop/Rainier_Weather.csv", header=True, inferSchema=True)
        weather_data = weather_data.select("Date", "Temperature AVG", "Relative Humidity AVG", "Wind Speed Daily AVG", "Wind Direction AVG", "Solare Radiation AVG")
        weather_data = weather_data.na.drop()
        weather_data = weather_data.withColumn("Date", to_date(col("Date"), "MM/dd/yyyy"))
        
        window = Window.orderBy("Date")
        weather_data = weather_data.withColumn("lag_1_temp", lag("Temperature AVG", 1).over(window))
        weather_data = weather_data.withColumn("lag_2_temp", lag("Temperature AVG", 2).over(window))
        weather_data = weather_data.withColumn("RollingAvg_Temperature", avg("Temperature AVG").over(window.rowsBetween(-7, 0)))
        weather_data = weather_data.na.drop()
        
        train_data, test_data = weather_data.randomSplit([0.8, 0.2], seed=1234)
        
        feature_cols = ["lag_1_temp", "lag_2_temp", "RollingAvg_Temperature", 'Relative Humidity AVG', 'Wind Speed Daily AVG', "Wind Direction AVG", "Solare Radiation AVG"]
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        
        rf = RandomForestRegressor(labelCol="Temperature AVG", featuresCol="features")
        pipeline = Pipeline(stages=[assembler, rf])
        
        model = pipeline.fit(train_data)
        predictions = model.transform(test_data)
        
        self.display_data(predictions)
        self.plot_graph(predictions)
    
    def display_data(self, data):
        self.text.delete(1.0, tk.END)
        data_pd = data.select("Date", "Temperature AVG", "Prediction").toPandas()
        self.text.insert(tk.END, data_pd.to_string(index=False))
    
    def plot_graph(self, data):
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
