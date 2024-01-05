import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
# Step 1 ------ Read the csv to dataframe
weather = pd.read_csv("weather.csv", index_col="DATE")
# print(weather.apply(pd.isnull).sum()/weather.shape[0])

# Step 2 ------ Preparing the data for machine learning by selecting the column that we need
core_weather = weather[["PRCP", "SNOW", "SNWD", "TMAX", "TMIN"]].copy()
core_weather.columns = ["precip", "snow", "snow_depth", "temp_max", "temp_min"]
# print(core_weather.apply(pd.isnull).sum()/core_weather.shape[0])
# print(core_weather["snow"].value_counts()) to check if a
del core_weather["snow"]  # deleting the snow column because it has  a major value of 0
del core_weather["snow_depth"] # deleting the snow column because it has  a major value of 0
core_weather["precip"] = core_weather["precip"].fillna(0)
core_weather = core_weather.fillna(method="ffill")
# print(core_weather.apply(pd.isnull).sum()/core_weather.shape[0]) to confirm if there are till null values


# Step 3 ------ checking the datatypes in the data frame

# print(core_weather.dtypes) to check if our data types are numeric
# print(core_weather.index) to check the indexes of the dataframe
core_weather.index = pd.to_datetime(core_weather.index)  # converting the indexes to datetime
# print(core_weather.apply(lambda x: (x==9999).sum()))


# Step 4 ------ analyzing our weather data
# core_weather[["temp_max", "temp_min"]].plot() to see if there are missing data in temp_max, temp_min
# print(core_weather.index.year.value_counts().sort_index()) to print years that has missing data
# print(core_weather.groupby(core_weather.index.month).sum()["precip"])

# Step 5 ------ Training our first machine learning model
core_weather["target"] = core_weather.shift(-1)["temp_max"]  # creating a target tomorrow temperature using the
# previous day max temp
core_weather = core_weather.iloc[:-1, :].copy()

reg = Ridge(alpha=.1)
predictors = ["precip", "temp_max", "temp_min"]
# train = core_weather.loc[:"2020-12-31"]
# test = core_weather.loc["2021-01-01":]
# reg.fit(train[predictors], train["target"])  # fitting our model into the training dataset using predictors and then try
# to predict our target
# predictions = reg.predict(test[predictors])  # trying to generate predictions on our test dataset using
# the predictor columns
# print(mean_absolute_error(test["target"], predictions)) see how well we did using mean_absolute_error which subtracts
# the actual from the predictions takes the absolute value then find the average from all the predictions


# Step 6 ------ evaluating our model
# combined = pd.concat([test["target"], pd.Series(predictions, index=test.index)], axis=1)
# combined.columns = ["actual", "predictions"]
# print(combined)
# combined.plot()

# Step 7 ------ creating a function to make predictions


def create_predictions(predictors, core_weather, reg):

    train = core_weather.loc[:"2020-12-31"]
    test = core_weather.loc["2021-01-01":]
    reg.fit(train[predictors], train["target"])
    predictions = reg.predict(test[predictors])
    error = mean_absolute_error(test["target"], predictions)
    combined = pd.concat([test["target"], pd.Series(predictions, index=test.index)], axis=1)
    combined.columns = ["actual", "predictions"]
    return error, combined


core_weather["month_max"] = core_weather["temp_max"].rolling(30).mean()  # finding the monthly avg temp
# print(core_weather)
core_weather["month_day_max"] = core_weather["month_max"] / core_weather["temp_max"]
core_weather["max_min"] = core_weather["temp_max"] / core_weather["temp_min"]
predictors = ["precip", "temp_max", "temp_min", "month_max", "month_day_max", "max_min"]
core_weather = core_weather.iloc[30:, :].copy()
# error, combined = create_predictions(predictors, core_weather, reg)
# print(error)
# combined.plot()

# Adding monthly and yearly avg
core_weather["monthly_avg"] = core_weather["temp_max"].groupby(core_weather.index.month).apply(lambda x: x.expanding(1)
                                                                                               .mean())
core_weather["weekly_avg"] = core_weather["temp_max"].groupby(core_weather.index.week).apply(lambda x: x.expanding(1)
                                                                                             .mean())
core_weather["day_of_year_avg"] = core_weather["temp_max"].groupby(core_weather.index.day_of_year).apply(lambda x:
                                                                                                         x.expanding(1)
                                                                                                         .mean())
predictors = ["precip", "temp_max", "temp_min", "month_max", "month_day_max", "max_min", "day_of_year_avg",
              "weekly_avg", "monthly_avg"]
error, combined = create_predictions(predictors, core_weather, reg)
print(error)


# Running model diagnostics
print(reg.coef_)
print(core_weather.corr()["target"])  # correlations between our predictions and the target
combined["diff"] = (combined["actual"] - combined["predictions"]).abs()  # difference between the actual value and the
# predicted values
print(combined.sort_values("diff", ascending=False).head())
print(core_weather)
# core_weather.to_csv("core_weather.csv")
combined.plot()
plt.show(block=True)