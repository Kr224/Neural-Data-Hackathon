from scipy.stats import ttest_ind

import segregation as s
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



# Load the data
file_path = '../data/participants_Spreng.xlsx'
df = pd.read_excel(file_path, engine='openpyxl')
selected_columns=list()

thresholds = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]  # Add more thresholds as needed
for threshold in thresholds:
    # Assuming s is an object with calculateSS method
    ss_values = s.calculateSS(thresh=threshold)

    # Create a new column in the dataframe for each threshold
    df[f'SS_{threshold}'] = ss_values
    selected_columns.append(f'SS_{threshold}')

    # Perform t-test for SS values between young and old groups

    young_col = f'SS_{threshold}'  # Replace 'SS' with the actual column name
    old_col = f'SS_{threshold}'  # Replace 'SS' with the actual column name
    t_stat, p_value = ttest_ind(df[df['agegroup'] == 'Y'][young_col], df[df['agegroup'] == 'O'][old_col])
    print(f'Threshold: {threshold}, T-statistic: {t_stat}, P-value: {p_value}')


    # Set the style of seaborn
    sns.set(style="whitegrid")


    # Plot histogram
    sns.histplot(data=df, x='agegroup', y=f'SS_{threshold}', hue='agegroup', multiple="stack", bins=20,
                 palette='pastel')

    # Add labels and title
    plt.xlabel('Age Group')
    plt.ylabel('SS Values')
    plt.title(f'Histogram of SS Values for Young and Old Age Groups (Threshold={threshold})')
    plt.legend(title='Age Group')

    # Show the plot
    plt.show()


df_selected = df[selected_columns]
# Handle missing values (replace NaNs with mean, median, or other strategies)
df.fillna(df_selected.mean(), inplace=True)


# Split the data into features (X) and target variable (y)
# X = df_selected.drop(['id', 'agegroup'], axis=1)
# y = df_selected['agegroup']

selected = ['SS_0.5']
# selected.append('bfas_neuroticism')
# selected.append('bfas_openness')


X = df[selected] # Include all relevant SS columns
# .values.reshape(-1, 1)
y = df['agegroup']  # Assuming 'agegroup' is the target variable


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a regression model (Random Forest is used here as an example)
model = LogisticRegression()
model.fit(X_train, y_train)


# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{confusion_mat}')
print(f'Classification Report:\n{classification_rep}')

