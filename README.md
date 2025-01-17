# House Price Prediction

This project demonstrates a simple machine learning application using linear regression to predict house prices based on area (in square meters). It is a beginner-friendly project designed to introduce the core concepts of data analysis, model training, and prediction.

## Table of Contents
- [Introduction](#introduction)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Workflow](#workflow)
- [How to Run the Project](#how-to-run-the-project)
- [Results](#results)
- [Future Enhancements](#future-enhancements)

---

## Introduction

This project uses a simple dataset containing house areas (in sq.m) and their corresponding prices. The aim is to create a model that learns the relationship between area and price to predict prices for new house areas.

Linear regression is chosen for this task as it is easy to understand and widely used for predictive modeling in machine learning.

The goal of this project is to train a linear regression model to predict house prices. The workflow involves:

Training a model using a dataset (homeprices.csv).
Predicting prices for new areas provided in area.csv.
Exporting the predictions to a new file (prediction.csv).
This project is beginner-friendly and demonstrates the key concepts of data loading, model training, and result visualization in machine learning.

---

## Technologies Used

- **Python**: Programming language for building the project.
- **Pandas**: For data manipulation and analysis.
- **Scikit-learn**: For implementing linear regression.
- **Matplotlib**: For visualizing data and predictions.

---

## Dataset

The dataset consists of two columns:
1. `area`: The area of the house in square meters.
2. `price`: The price of the house in currency units.

Example:
| Area (sq.m) | Price (currency units) |
|-------------|-------------------------|
| 100         | 500                    |
| 200         | 1000                   |
| 300         | 1500                   |

You can add or modify the dataset as required.

---

## Workflow

1. **Import Libraries**:
   ```python
   import pandas as pd
   from sklearn.linear_model import LinearRegression
   import matplotlib.pyplot as plt
   ```

2. **Prepare the Dataset**:
   Load and structure the data for training and testing:
   ```python
   data = {'area': [100, 200, 300, 400, 500],
           'price': [500, 1000, 1500, 2000, 2500]}
   df = pd.DataFrame(data)
   ```

3. **Train the Model**:
   ```python
   reg = LinearRegression()
   reg.fit(df[['area']], df['price'])
   ```

4. **Predict New Values**:
   ```python
   prediction = reg.predict([[330]])
   print(f"Predicted price for 330 sq.m: {prediction[0]}")
   ```

5. **Visualize Results**:
   ```python
   plt.scatter(df['area'], df['price'], color='blue')
   plt.plot(df['area'], reg.predict(df[['area']]), color='red')
   plt.xlabel('Area (sq.m)')
   plt.ylabel('Price')
   plt.title('Linear Regression: House Price Prediction')
   plt.show()
   ```

---

## How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/house-price-prediction.git
   ```
2. Navigate to the project folder:
   ```bash
   cd house-price-prediction
   ```
3. Install dependencies (optional, if not already installed):
   ```bash
   pip install pandas scikit-learn matplotlib
   ```
4. Run the Python script:
   ```bash
   python house_price_prediction.py
   ```

---

## Results

The trained model predicts house prices based on the given area. For example, if the area is 330 sq.m, the model might predict a price of 1650 currency units. Results are also visualized using a scatter plot with the regression line.

---

## Future Enhancements

- Expand the dataset with more features like location, number of rooms, etc.
- Experiment with advanced regression techniques.
- Build a user-friendly web interface for predictions.

---

## Author
Anithasri

Feel free to contribute to this project or reach out for collaboration! For suggestions or feedback, open an issue or contact me on GitHub.


