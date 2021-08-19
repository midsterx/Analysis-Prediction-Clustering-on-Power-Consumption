# Analysis, Prediction & Clustering on an individual household's Electrical Power Consumption

Presented at International Conference on Data Engineering and Communication Systems (ICDECS), 2019. Conference proceedings published at International Journal of Innovative Technology and Exploring Engineering (IJITEE): https://www.ijitee.org/wp-content/uploads/papers/v9i2S/B10291292S19.pdf

## About the Dataset:
It contains 2075259 measurements gathered between December 2006 and November 2010 (47 months). 

## Attributes: 
1. date: Date in format dd/mm/yyyy
2. time: time in format hh:mm:ss
3. global_active_power: household global minute-averaged active power (in kilowatt)
4. global_reactive_power: household global minute-averaged reactive power (in kilowatt)
5. voltage: minute-averaged voltage (in volt)
6. global_intensity: household global minute-averaged current intensity (in ampere)
7. sub_metering_1: energy sub-metering No. 1 (in watt-hour of active energy). It corresponds to the kitchen, containing mainly a dishwasher, an oven and a microwave (hot plates are not electric but gas powered).
8. sub_metering_2: energy sub-metering No. 2 (in watt-hour of active energy). It corresponds to the laundry room, containing a washing-machine, a tumble-drier, a refrigerator and a light.
9. sub_metering_3: energy sub-metering No. 3 (in watt-hour of active energy). It corresponds to an electric water-heater and an air-conditioner.

## Notes:
1. (global_active_power*1000/60 - sub_metering_1 - sub_metering_2 - sub_metering_3) represents the active energy consumed every minute (in watt hour) in the household by electrical equipment not measured in sub-meterings 1, 2 and 3.
2. The dataset contains some missing values in the measurements (nearly 1,25% of the rows). All calendar timestamps are present in the dataset but for some timestamps, the measurement values are missing: a missing value is represented by the absence of value between two consecutive semi-colon attribute separators. For instance, the dataset shows missing values on April 28, 2007.

## Steps:
Sequence of Files to be run:
1. Create an empty folder named 'dataset' write outside the git-cloned folder, and place the initial dataset inside the folder
2. Files in Cleaning and Preprocessing
3. Files in Prediction - SARIMA, LSTM and Hybrid in that order
4. Files in Clustering

