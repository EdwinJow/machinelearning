import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()

data = pd.read_csv('california_housing.csv')

print(data.head())

# data.fillna(value=0, inplace=True)

y = data.median_income
X = data[['housing_median_age', 'total_rooms', 'population', 'median_house_value']]

train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)

model.fit(train_X, train_y)

prediction = model.predict(val_X)
print(mean_absolute_error(val_y, prediction))

# def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
#     model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
#     model.fit(predictors_train, targ_train)
#     preds_val = model.predict(predictors_val)
#     mae = mean_absolute_error(targ_val, preds_val)
#     return(mae)

# # compare MAE with differing values of max_leaf_nodes
# for max_leaf_nodes in [5, 50, 500, 5000]:
#     my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
#     print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
