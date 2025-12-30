
#LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_info = pd.read_csv("DATASET/delhi_information.csv")
df_prices = pd.read_csv("DATASET/delhi_prices.csv")

df_info.info ()
df_info.head()

#analisi a subset 
#df_I_VISITARA=[df_info["Airline"] == "Vistara"] 
#df_I_MUMBAI=[df_info["Destination"] == "Mumbai"]
#df_I_VISITARA.info

#Q 1 a: How many unique flights to Bangalore are there in the flight information dataset provided? 
#Q 1 b: Is the dataset sufficiently large to meet the client's needs?

df_I_BANGALORE = df_info[df_info["Destination"] == "Bangalore"]
df_I_BANGALORE.info
df_I_BANGALORE.head()
df_unique = df_I_BANGALORE.drop_duplicates(subset="Flight Number")
df_unique.info()
n_unique = len(df_unique)
print(n_unique)
#CHECK RESULTS WITH OTHER METHOD
df_I_BANGALORE["Flight Number"].nunique()

#A 1a: The number of unique flights to Bangalore is 187
#A 1b: C , We don't know because we still haven't explored the dataset in sufficient detail (we must consider it is highly populated.)
#-----------------------------------------------------

#Q 2a: Please determine which months of the year this dataset covers. What is the sum of the unique numerical values of the months?
#Hint: consider June = 6, january=1 , September =9

#Q 2b: Based on the data coverage of the dataset, is this dataset relevant to the client in solving his problem?

#I'm adapting the date format to the european model
df_info["Journey Date"] = pd.to_datetime(df_info["Journey Date"], dayfirst=True 
)

#create a Month column from Journey Date
df_info["Month"] = df_info["Journey Date"].dt.month #already using the number-month 
unique_months = df_info["Month"].unique() #finding only unique months
sum_months = sum(unique_months)
print(unique_months)
print(sum_months)


#A 2a: The dataset covers the months of January, February and March. The sum of the unique numerical values of these months is 6.
#A 2b: Answer B. It depends, it may still be useful to understand pricing strategies and its limitations
#-----------------------------------------------------
#Q 3a: How many unique destinations has the airline with the least flights? 

#Q 3b: Our clients is considering entering new routes already served by competitors. What is the most commercially significant implication?

#ANSWER TO Q 3a
flights_per_airline = (df_info.groupby("Airline")["Flight Number"].nunique())
least_flights_airline = flights_per_airline.idxmin()
least_flights_airline

#idxmin() tells us which is the company with least flights

#find the company that serves the least destinations
#FAST WAY
unique_destinations_per_airline = (df_info.groupby("Airline")["Destination"].nunique())

print(unique_destinations_per_airline)

#SLOW WAY (FIrst option used)

airlines = df_info["Airline"].unique()

for airline in airlines:
    df_airline = df_info[df_info["Airline"] == airline]
    df_unique = df_airline.drop_duplicates(subset="Destination")
    print(f"\n{airline}: {len(df_unique)} unique destinations")






#A 3a: The Airline with the least flights is AllianceAir, with 1 destination
#A 3b:The most commercially significant implication is that in the market, right now besides Alliance Air and AkasaAir , everyone already serves for sure at least 1 destination that also another company has. Therefore, it would be wise to adjust its strategy according to this fact.
#Entering routes already served by competitors implies direct competition, which is likely to lead to price pressure and lower profit margins. Therefore, the client would need a clear competitive advantage, such as lower operating costs, better schedules, or service differentiation, in order to compete effectively on these routes.
#Given that most destinations are already served by multiple airlines, entering these routes would intensify competition rather than create a monopoly or niche advantage.
#RIGHT ANSWER
#(letter A) Entering routes already served by competitors increases competitive overlap, which limits pricing flexibility and reduces pricing power. As a result, the airline may face slower revenue growth over time, since fares cannot be adjusted freely without risking demand loss to competitors.

#-----------------------------------------------------
#Q 4a: What is the difference between the minimum and maximum number of days prior to the flight that fares have been recorded ? 
#Q 4b: WIthout further analysis, which hypothesis do you think best reflects the relationship between days prior to the flight and fare to the flight? 

min_days = df_prices["Days Before Journey Date"].min()
min_days
max_days = df_prices["Days Before Journey Date"].max()
max_days 
difference = max_days - min_days
difference

#we didn't use groupby because the answer does not require by company or to cross-over data

#A 4a: The difference is 49, 50-1=49
#A 4b: Closer to the date of a flight, fares are higher because airlines capitalise on last-minute demand. The demand will be less ealstic , and having fixed seats remained will raise the average price

#------------------------------------------------------------------------------------------------------------------
#Q 5a: Which day of the week has the most business flights?
#Q 5b: The previous analysis shows there is a concentration of business class flights on certain days, and the client has a fixed fleet and is aiming
#to improve revenues capitalizing these days. What is the most commercially appropriate response to this insight?

df_business = df_info[df_info["Class"] == "Business"]
business_per_day = df_business.groupby("Journey Day").size()
business_per_day.idxmax()
business_per_day

# A 5a: Monday (3793), followed by Saturday and Thursday (3284)
# A 5b: ANSWER C : Given that the client operates with a fixed fleet size and aims to improve profitability, the most commercially appropriate response is to reconfigure certain aircraft to offer more business class seats on peak routes and days while maintaining total capacity. 
# This strategy allows the airline to better align capacity with observed demand patterns, capture higher-yield passengers when demand is strongest, and enhance revenue without requiring fleet expansion or significant operational changes.

#-----------------------------------------------------------------------------------------------------------------------------------------
#Q 6a: Which field did you use as the unique identifier in the process of combining the datasets together?
#Q 6b: Without performing further analysis, which of the following numerical variables are most likely to have a non-linear relationship with Fare(Rupes)?


#A 6a: I used as unique identifier the Flight ID
#A 6b: ANSWER A Airline fares typically follow dynamic pricing strategies, 
# where prices change slowly when the departure date is far away and increase more sharply as the flight date approaches, resulting in a non-linear relationship between fare and days before departure.

#-----------------------------------------------------------------------------------------------------------------------------------------------------

#Q 7a: based on the combined datasets, which airline has the  second lowest average in Indian Rupees? 
#Hint: all subsequent questions require you to combine datasets

# merge datasets
df_merged = pd.merge(df_info, df_prices, on="Flight ID", how="inner")

# average fare per airline
avg_fare_per_airline = (df_merged.groupby("Airline")["Fare (Rupees)"].mean().sort_values())
avg_fare_per_airline

# second lowest average fare
second_lowest_airline = avg_fare_per_airline.index[1]
second_lowest_airline

#A 7a: AkasaAir , considering the average

#------------------------------------------------------------------------------------------------------------------------------------
#Q 8a: What is the difference in fare (Rupees) between the most expensive and the cheapest airline for flights to Hyderabad? 
#Q 8b: One airline in your dataset shows a significantly higher average fare than all others. Your client assumes this means the 
#airline is consistently more expensive. Which of the following is the most appropriate interpretation?


df_hyd = df_merged[df_merged["Destination"] == "Hyderabad"].copy()
min_row = df_hyd.loc[df_hyd["Fare (Rupees)"].idxmin()]
max_row = df_hyd.loc[df_hyd["Fare (Rupees)"].idxmax()]
min_row
max_row
df_difference= max_row["Fare (Rupees)"] - min_row["Fare (Rupees)"]
df_difference

#A 8a: The difference is 86557 , between a Business flight of Vistara airline and an Economy with Spicejet
#A 8b: The company with the highest average is Vistara.The high average fare may reflect departure time mix or a higher share of business-class seats, and may not indicate a consistently higher price for comparable products.
# option A  makes tempting conclusions but with strong assumptions we do not have proof of 

#--------------------------------------------------------------------------------------------------------------------------------------
# Q 9a: What is the absolute value of difference in average Fare (Rupees) 
# between 25 days before the journey and 5 days before the journey, for non-stop Economy flights to Hyderabad 
# on Indigo Airlines departing between 6 Am and 6 PM on weekdays?

df_Q9 = df_merged[ 
    (df_merged["Destination"] == "Hyderabad") &
    (df_merged["Class"] == "Economy") &
    (df_merged["Airline"] == "Indigo") &
    (df_merged["Number Of Stops"] == 0) &
    (df_merged["Departure"] == "6 AM - 12 PM") &
    (~df_merged["Journey Day"].isin(["Saturday", "Sunday"]))
].copy()

df_Q9

fare_25 = df_Q9[df_Q9["Days Before Journey Date"] == 25]["Fare (Rupees)"].mean()
fare_5  = df_Q9[df_Q9["Days Before Journey Date"] == 5]["Fare (Rupees)"].mean()

difference = abs(fare_25 - fare_5)
difference


#PLOT A BAR CHART COMPARING PRICES DIFFERENT AIRLINES OFFER FOR A TRIP TO HYDERABAD
avg_price_hyd = ( df_hyd.groupby("Airline")["Fare (Rupees)"].mean())

plt.bar(avg_price_hyd.index, avg_price_hyd.values)
plt.xticks(rotation=45)
plt.ylabel("Average Fare (Rupees)")
plt.title("Average Fare to Hyderabad by Airline")
plt.show()

# PLOT How many flights each airline operates to Hyderabad
df_hyd_flights = df_hyd.groupby("Airline")["Flight ID"].count()

plt.bar(df_hyd_flights.index, df_hyd_flights.values)
plt.xticks (rotation=45)
plt.ylabel ("Number of flights to Hyderaband")
plt.title("Flights to hyderaband per Airline")
plt.show()

# A 9a: the difference is 1125.83
# A 9b: ANSWER D The chart only compares average fares across airlines and does not show how prices change over time or who purchases last-minute tickets. 
# Therefore, it does not provide evidence to support either of the two claims.

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Q 10a: For flights to the most common flight destination, how expensive in Rupees are flights between 6Am and 12PM?
#Q 10b: Imagine the client is considering expanding into a new city and can only bid for a limited number of take-off slots. Slots are either Morning, Afternoon or Evening. 
# The client is new to this market and wants to prioritise a strategy that balances revenue potential, sustainable expansion and learning about customer behaviour
#Which of the following options would you likely reccommend? No additional analysis is needed 


#most common flight destination
flights_per_destination = df_merged.groupby("Destination")["Flight ID"].count()
flights_per_destination.sort_values(ascending=False)


# PLOT most popular flight destinations 
plt.bar(flights_per_destination.index, flights_per_destination.values)
plt.xticks(rotation=35)
plt.ylabel("Number of flights")
plt.title("Flights per Destination")
plt.show()

#A 10a: the plot
#A 10b: A 1 Morning & 2 afternoon, This option balances revenue potential and learning: a morning slot captures higher-yield, time-sensitive demand (often business travellers), while afternoon slots provide broader exposure to leisure and mixed demand

#-------------------------------------------------------------------------------------------------------------------------------------

#DETERMINE HOW PRICES VARY BY DEPARTURE TIME FOR FLIGHTS TO THE MOST COMMON FLIGHT DESTINATIONS 

#Q10 Mumbai e Bangalore = most popular destinations 

df_destinations = df_merged[df_merged["Destination"].isin(["Mumbai", "Bangalore"])].copy()

price_by_dep = (df_destinations
                .groupby(["Destination", "Departure"])["Fare (Rupees)"]
                .mean()
                .reset_index()
                .sort_values(["Destination", "Departure"]))

print(price_by_dep)

price_by_dep_rupees = price_by_dep.sort_values(["Fare (Rupees)"])
print(price_by_dep_rupees)

# se vuoi una label unica per ogni barra (Destination + Departure)
price_by_dep["label"] = price_by_dep["Destination"] + " | " + price_by_dep["Departure"]

plt.figure(figsize=(10,5))
plt.bar(price_by_dep["label"], price_by_dep["Fare (Rupees)"])
plt.xticks(rotation=35, ha="right") #ha ="right" alignment to the right of tickers
plt.ylabel("Average Fare (Rupees)")
plt.title("Average Fare by Destination and Departure Window")
plt.tight_layout() # to avoid cutting unwillingly parts 
plt.show()

#Uncomfortable departure times tend to be cheaper while more convenient time slots are more expensive; 
# however, the impact of departure time on prices is not uniform and varies by destination and demand structure.

# STANDARDIZE THE NUMERICAL VARIABLES (EXCEPT FARE) USING MEAN NORMALIZATION 
#HINT: THE FORMULA FOR MEAN NORMALIZATION IS X MINUS THE MEAN DIVIDED BY THE MAXIMUM VALUE MINUS MINIMUM VALUE FOR A COLUMN,  
# WHERE X IS A PARTICULAR VALUE WITHIN THE COLUMN




#Q 11a:What do you expect to happen to the standard deviation of the numerical columns after their mean normalisation?
#Hint: No need for calculation or further analyses
#Formula x - xmean /(max val - min val)


#Q 11b:


#A 11a: decrease ANSWER B, Mean normalisation rescales the data into a smaller range, which mechanically reduces the standard deviation.
#A 11b:Mean normalisation rescales the data by dividing deviations from the mean by the range, which compresses the spread of values and therefore reduces the standard deviation.

#---------------------------------------------------------------------------------------------------------------------------------------

#Q 12A: How many variables pairs of input variables have a high degree of multi-collinearity?
#Hint: Assume that pairs of variables with a correlation coefficient of 0.8 or higher or -0.8 or lower are likely to suffer from multicollinearity

#Q 12b: What is the primary concern with multicollinearity in a pricing model? 

#Q 12c:What are the variables you should exclude from the model to avoid multicolinnearity?

df_merged_num = df_merged[[
    "Days Before Journey Date",
    "Duration (Hours)",
    "Number Of Stops",
    "Fare (GBP)",
    "Fare (Rupees)",
    "Month"
]]
df_merged_num.columns

correlation_matrix = df_merged_num.corr(method ="pearson") 
sns.heatmap (correlation_matrix, annot= True)
plt.title ("CORRELATION MATRIX")
plt.xlabel("VAR X")
plt.ylabel ("VAR Y")
plt.show ()

#A 12a: There is  perfect multicollinearity among fares, because they are the same price expressed in two different currencies. There is also multicollinearity among Journey date and the Month of reference. The high correlation (0.89) arises because the dataset covers only a limited set of months, so the “Days Before Journey Date” values are mechanically tied to the calendar coverage rather than a causal relationship. It reflects sampling/coverage bias (when observations were recorded) more than a meaningful pricing or operational dependency.

#A 12b:The main issue with multicollinearity is unstable and unreliable coefficient estimates, which makes it difficult to interpret the individual effect of each variable, even if overall model fit looks acceptable.

#A 12c: Correct answer: O — Fare (GBP). Fare (GBP) is just a currency conversion of Fare (Rupees) (≈ perfect correlation), so keeping both adds redundant information and can destabilise coefficient estimates without improving the model.


#-----------------------------------------------------------------------------------------------------------------------------------------
#Q 13a: how many new columns have been created in the dataset from hot-coding? 

# WHAT IS HOT-CODING? 
# Using a dummy variable to represent a more complex  situation (such as days of the week)

#1° JOURNEY DAY
journey_day_dummies = pd.get_dummies(
    df_merged["Journey Day"],
    prefix="Day",
    drop_first=True
)

df_encoded = pd.concat([df_merged, journey_day_dummies], axis=1)

#to check it 
df_encoded.filter(like="Day_").head()

#2° AIRLINE
airline_dummies = pd.get_dummies(
    df_merged["Airline"],
    prefix="Airline",
    drop_first=True
)

df_encoded = pd.concat([df_encoded, airline_dummies], axis=1)


#to check it 
df_encoded.filter(like="Airline_").head()

#3° CLASS
class_dummies = pd.get_dummies(
    df_merged["Class"],
    prefix="Class",
    drop_first=True
)

df_encoded = pd.concat([df_encoded, class_dummies], axis=1)


#to check it 
df_encoded.filter(like="Class_").head(18000)

#4° DEPARTURE
departure_dummies = pd.get_dummies(
    df_merged["Departure"],
    prefix="Departure",
    drop_first=True
)

df_encoded = pd.concat([df_encoded, departure_dummies], axis=1)


#to check it 
df_encoded.filter(like="Departure_").head(18000)

#4° DESTINATION
destination_dummies = pd.get_dummies(
    df_merged["Destination"],
    prefix="Destination",
    drop_first=True
)

df_encoded = pd.concat([df_encoded, destination_dummies], axis=1)


#to check it 
df_encoded.filter(like="Destination_").head(18000)


#A 13a: 18 new columns: airline 7, class 3, departure 3, destination 5
#A 13b: ANSWER A - Including many one-hot encoded variables with very few observations (e.g. rare numbers of stops) leads to unstable and noisy coefficient estimates. This makes it easy to misinterpret those coefficients as true price premiums or discounts, creating a real commercial risk if pricing decisions are based on statistically unreliable effects.

#------------------------------------------------------------------------------------------------------------------------------------------

# CREATE A MULTIVARIATE REGRESSION MODEL TO PREDICT THE FARE OF INDIAN RUPEES
#Hint: use all the rows in the model, but be selective with the columns you decide to include, run a first pass only with the numerical and one hot-encoded columns as independent variables

#1) Clean dataset 

df_regression = df_merged.drop(columns=['Fare (GBP)','Flight ID','Journey Date_x', 'Journey Day', 'Airline','Flight Number', 'Class', 'Origin', 'Departure','Arrival', 'Destination','Month', 'Journey Date_y'])

df_regression

#SCATTERPLOTS


g_stops = sns.lmplot(
    x='Number Of Stops',
    y='Fare (Rupees)',
    data=df_regression,
    scatter_kws={'alpha': 0.1, 's': 10}
)
g_stops.fig.suptitle('Relationship between Number of Stops and Fare (Rupees)', fontsize=12)
g_stops.fig.subplots_adjust(top=0.9)
plt.show()

g_duration = sns.lmplot(
    x='Duration (Hours)',
    y='Fare (Rupees)',
    data=df_regression,
    scatter_kws={'alpha': 0.1, 's': 10}
)
g_duration.fig.suptitle('Relationship between Duration (Hours) and Fare (Rupees)', fontsize=12)
g_duration.fig.subplots_adjust(top=0.9)
plt.show()

g_days = sns.lmplot(
    x='Days Before Journey Date',
    y='Fare (Rupees)',
    data=df_regression,
    scatter_kws={'alpha': 0.1, 's': 10}
)
g_days.fig.suptitle('Relationship between Days Before Journey Date and Fare (Rupees)', fontsize=12)
g_days.fig.subplots_adjust(top=0.9)
plt.show()


df_encoded2 = df_encoded.drop(columns=[
    'Flight ID',
    'Journey Date_x',
    'Journey Day',
    'Airline',
    'Flight Number',
    'Class',
    'Origin',
    'Departure',
    'Number Of Stops',
    'Arrival',
    'Destination',
    'Duration (Hours)',
    'Month',
    'Journey Date_y',
    'Days Before Journey Date',
    'Fare (Rupees)',
    'Fare (GBP)'
])



df_regression2 = pd.concat(
    [df_regression, df_encoded2],
    axis=1
)

from sklearn.model_selection import train_test_split

# Target (y) as a Series 
y = df_regression2['Fare (Rupees)']

# Features (X): everything besides the Rupees
X = df_regression2.drop(columns=['Fare (Rupees)'])

# Split to test the validity of the model 
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

from sklearn.linear_model import LinearRegression

print("X_train:", X_train.shape, "X_test:", X_test.shape)
print("y_train:", y_train.shape, "y_test:", y_test.shape)

# model
lr = LinearRegression()
lr.fit(X_train, y_train)

# R^2
r2_test = lr.score(X_test, y_test)
r2_train = lr.score(X_train, y_train)

print("R2 test :", r2_test)
print("R2 train:", r2_train)

from sklearn.metrics import mean_absolute_error, mean_squared_error , r2_score

y_pred = lr.predict(X_test)
mean_absolute_error(y_test, y_pred)
mean_squared_error(y_test, y_pred)
r2_score(y_test, y_pred)

y_test.describe()

#array of coefficients
lr.coef_
lr.intercept_

# We do have an high MAE Mean regression error, meaning that results in our models are quite different 
# in terms of magnitude and variance from others. More specifically, on tickets that on average are
# of 20494.984846 = €193.73 , there is a mistake of approximately €42-43. Now this imprecision is caused
# by the fact that the dataset is hugely heterogenic and therefore includes prices of tickets from 
# different classes (Business, Economy ecc.). Moreover, Fares do not behave linearly, therefore 
# avoiding standard normalization with z score, that would destroy the natural heterogeneity of data
# in such cases, the most reasonable option is to adopt a log transformation of the Fares, and reinterpret
#results afterwards 



#LOG TRANSFORMATION 

y_log = np.log(df_regression2['Fare (Rupees)'])
# Features (X): everything besides the Rupees
X = df_regression2.drop(columns=['Fare (Rupees)'])

# Split to test the validity of the model 
X_train, X_test, y_log_train, y_log_test = train_test_split(
    X, y_log,
    test_size=0.2,
    random_state=42
)

print(X_train.shape, X_test.shape, y_log_train.shape, y_log_test.shape)

from sklearn.linear_model import LinearRegression

print("X_train:", X_train.shape, "X_test:", X_test.shape)
print("y_train:", y_log_train.shape, "y_test:", y_log_test.shape)

# model
lr2 = LinearRegression()
lr2.fit(X_train, y_log_train)

# R^2
r2_test_log = lr2.score(X_test, y_log_test)
r2_train_log = lr2.score(X_train, y_log_train)

print("R2 test :", r2_test_log)
print("R2 train:", r2_train_log)

from sklearn.metrics import mean_absolute_error, mean_squared_error , r2_score

y_log_pred = lr2.predict(X_test)
mean_absolute_error(y_log_test, y_log_pred)
mean_squared_error(y_log_test, y_log_pred)
r2_score(y_log_test, y_log_pred)

#array of coefficients
lr2.coef_
lr2.intercept_

#to reinterpret coefficients:

coef_df = pd.DataFrame({
    'variable': X.columns,
    'beta_log': lr2.coef_
})

coef_df['perc_effect'] = 100 * (np.exp(coef_df['beta_log']) - 1)

coef_df

coef_df['variable'].value_counts()


# By modeling airfares in logarithmic form, the analysis accounts for the intrinsic heterogeneity of ticket prices and allows prediction errors to be interpreted in relative terms. 
# The log-linear specification improves model fit and yields an average prediction error of approximately 19% on the test set (R^2 increased to almost 90% due to less heteroskedasticity thanks to log form) 

















# SAME REGRESSION BUT NOT TO PREDICT BUT TO INFERE ON OUR DATA
import sklearn
import statsmodels.api as sm

# =========================
# 1) Definisci y (log target)
# =========================
y2 = np.log(df_regression2['Fare (Rupees)'])

# =========================
# 2) Definisci X (features)
# =========================
X2 = df_regression2.drop(columns=['Fare (Rupees)'])

# =========================
# 3) Aggiungi intercetta
# =========================
X_sm = sm.add_constant(X2)

# =========================
# 4) Stima OLS
# =========================
model = sm.OLS(y2, X_sm)
results = model.fit()

# =========================
# 5) Output completo
# =========================
print(results.summary())
