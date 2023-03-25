## LSTM time series analysis
This project aims to predict future stock prices using a Long-Short-Term Memory (LSTM) neural network. The project is written in Python and uses the TensorFlow framework.

### Getting Started
To get started with this project, follow the instructions below:

- Clone this repository to your local machine.
- Install the required packages by running the following command in the terminal: `pip install -r requirements.txt`.
- Navigate to the project directory in the terminal.
- Run the project by executing the command `python main.py`.

### Project Structure
The project consists of three Python files:

- `main.py` - The main file of the project, which contains the code to preprocess the data, train the LSTM model, and generate predictions.
- `lstm.py` - A module that defines the LSTM model used in the project.
- `plotter.py` - A module that defines the Plotter class, which is used to plot the data and the model performance.

### Usage
The `main.py` file contains the main code for the project. To run the project, execute the following command in the terminal:
```
python main.py
```
This will preprocess the data, train the LSTM model, and generate predictions.

### Detailed description about files

- `main.py` - The `main.py` file is the main driver of the project. It imports the necessary libraries, including pandas, sklearn, numpy, and the two classes from the other two files. The file initializes the TIME_STEPS and loads a CSV file using the Pandas library. The file then preprocesses the data by removing rows with missing data, scaling the data using the MinMaxScaler from the Sklearn library, and creating a 3D array of the data. The file then creates an instance of the LSTM model, compiles it, and fits it to the data. Finally, the file creates an instance of the Plotter class and uses it to plot the different aspects of the project and saves all the information.

- `lstm.py` - The `lstm.py` file contains a class LongShortTermMemory which defines an LSTM model. The class initializes with the project_folder and defines two methods, get_defined_metrics() and get_callback(), which return a list of metrics and an EarlyStopping callback respectively. The create_model() method builds the LSTM model by creating a Sequential model with four LSTM layers, each followed by a Dropout layer to reduce overfitting, and a Dense output layer. The model is compiled with the Mean Squared Error loss function.

- `plotter.py` - The `plotter.py` file contains a class Plotter that defines methods to plot different aspects of the project. The class initializes with the blocking, project_folder, short_name, currency, and stock_ticker. The plot_histogram_data_split() method plots the training and validation data as well as histograms of the training data. The plot_loss() method plots the loss and validation loss from the history of the LSTM model. The plot_mse() method plots the Mean Squared Error and validation MSE from the history of the LSTM model. The project_plot_predictions() method plots the predicted and actual prices of the stock.

