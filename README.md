# climate-data-analysis
This repository contains a climate data analysis project, focusing on visualizing temperature trends and building machine learning models to predict future temperature changes. It uses Python libraries like pandas, matplotlib, seaborn, and scikit-learn to perform exploratory data analysis (EDA) and develop predictive models.

## Linear Regression Model Evaluation with Streamlit

This project demonstrates the implementation and evaluation of a Linear Regression model using synthetic data. The goal is to predict target values based on input features while providing meaningful metrics and visualizations to assess model performance. The project utilizes Streamlit for an interactive and user-friendly web-based interface.

## Features

Synthetic Data Generation: Generates regression data with customizable features and noise.  

Model Training: Implements a Linear Regression model using scikit-learn.  

## Evaluation Metrics:

Mean Squared Error (MSE)  

Root Mean Squared Error (RMSE)  

Mean Absolute Error (MAE)  

R² Score  

## Interactive Visualizations:

Scatter plot of True vs Predicted values.  

Residuals histogram to analyze prediction errors.  

Model Coefficients: Displays feature importance in a tabular format.  

## Prerequisites

To run this project, ensure you have the following installed:  

Python 3.7+  

## Required libraries:

pip install numpy pandas matplotlib seaborn scikit-learn streamlit  

## How to Run

Clone the repository:  

git clone <repository-url>  
cd <repository-folder>  

## Install the dependencies:

pip install -r requirements.txt  

## Run the Streamlit app:

streamlit run app.py  

Open the provided local URL in your web browser to interact with the app.   

Project Structure

├── app.py                # Main Streamlit application file
├── eda             #eda function performed on data 
├── ml_model               # my machine learning model implementation
├── README.md             # Project documentation



## Key Libraries Used

NumPy: For numerical computations.  

Pandas: For handling datasets.  

Matplotlib & Seaborn: For creating visualizations.  

Scikit-learn: For model training and evaluation.  

Streamlit: For building the interactive web application.  

# Future Enhancements

Add support for user-uploaded datasets.   

Include additional regression models for comparison.  

Provide advanced feature engineering options.  

## License

This project is licensed under the MIT License. See the LICENSE file for details.  

## Acknowledgments

Scikit-learn Documentation  

Streamlit Documentation  

Feel free to contribute by submitting issues or pull requests!
